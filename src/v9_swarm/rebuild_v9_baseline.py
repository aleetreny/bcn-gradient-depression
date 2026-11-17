"""Rebuild the v9 baseline cache locally without a public-window evaluation.

Requires existing v8.5 baseline, enriched candidates, streams and tracking
features. Generated match-derived predictions must remain local.
"""
from __future__ import annotations
from pathlib import Path
import pickle
import sys
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.data.common import DEAD
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import load_all as load_sportec
from src.models.transition import change_features
from src.v9_swarm.rebuild_candidates import _feature_cache_paths
from src.v9_swarm.duel_and_mbr_v9 import SymDomainTransferCalibratorV9, SymDuelTurnoverArbitratorV9, SymSpellMBRDecoderV9
from src.v9_swarm.smart_tagging_arbitrator import SmartTaggingArbitrator, extract_simulated_smart_tags_sample, extract_simulated_smart_tags_sportec, load_bar_ath_smart_tags
from src.v9_swarm.whistle_and_stoppage import Fox40WhistleDetector, StoppageStartRefiner

def _clone_bundle(b: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "preds_v3": {k: v.copy() for k, v in b["preds_v3"].items()},
        "preds_v6_std": {k: v.copy() for k, v in b["preds_v6_std"].items()},
        "preds_v6_gen": {k: v.copy() for k, v in b["preds_v6_gen"].items()},
        "preds_sample": {k: v.copy() for k, v in b["preds_sample"].items()},
        "pred_atd_rep": b["pred_atd_rep"].copy(),
        "pred_atd_raw": b["pred_atd_raw"].copy(),
    }

def build_v9_grand_baseline(
    suite: Any,
    streams: Dict[str, Any],
    X_v3: Dict[str, Any],
    X_v6: Dict[str, Any],
    X_sample: Dict[int, Any],
    X_atd_rep: Any,
    root: Path,
    output_path: Path,
) -> Dict[str, Any]:
    """Reconstruct the historical v9 baseline from existing local inputs."""
    from games import GAMES as GAME_INFO, anchor_video_frame
    mids = sorted(suite.gt_v3.keys())
    gids = sorted(suite.gt_sample.keys())
    atd = suite.atd
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    with open((root / "experiments/outputs/v9_swarm") / "_diag_enr_cands.pkl", "rb") as f:
        enr = pickle.load(f)
    duel_cands = enr["duel"]
    mbr_cands = enr["mbr"]

    whistle_path = root / "local_data" / "v9_swarm" / "bar_ath_whistle.npz"
    w_dual = Fox40WhistleDetector().extract_and_cache_bar_ath_whistle(
        root, cache_path=whistle_path, cleanup_wav=True
    )["w_dual"]
    sp_tags_bar, phase_score_bar = load_bar_ath_smart_tags(root, n_frames=atd.n_frames)
    sp_tags_sportec = {mid: extract_simulated_smart_tags_sportec(root, mid, suite.sportec[mid]) for mid in mids}
    sp_tags_sample = {gid: extract_simulated_smart_tags_sample(root, gid) for gid in gids}

    v85_base = suite.baselines["v85_forensic_strict_cv"]
    duel_full = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][m] for m in mids])
    mbr_dec = SymSpellMBRDecoderV9()
    dom_cal_full = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][g] for g in gids])
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    b = _clone_bundle(v85_base)
    for m in mids:
        train_m = [tm for tm in mids if tm != m]
        arb_lomo = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][tm] for tm in train_m])
        b["preds_v3"][m], _ = arb_lomo.apply_from_candidates(b["preds_v3"][m], duel_cands["v3"][m])
        b["preds_v6_std"][m], _ = arb_lomo.apply_from_candidates(b["preds_v6_std"][m], duel_cands["v6_std"][m])
        b["preds_v6_gen"][m], _ = arb_lomo.apply_from_candidates(b["preds_v6_gen"][m], duel_cands["v6_gen"][m])
    for g in gids:
        b["preds_sample"][g], _ = duel_full.apply_from_candidates(b["preds_sample"][g], duel_cands["j2"][g])
    b["pred_atd_rep"], _ = duel_full.apply_from_candidates(b["pred_atd_rep"], duel_cands["atd_rep"])

    for m in mids:
        b["preds_v3"][m], _ = mbr_dec.apply_from_candidates(b["preds_v3"][m], mbr_cands["v3"][m])
        b["preds_v6_std"][m], _ = mbr_dec.apply_from_candidates(b["preds_v6_std"][m], mbr_cands["v6_std"][m])
        b["preds_v6_gen"][m], _ = mbr_dec.apply_from_candidates(b["preds_v6_gen"][m], mbr_cands["v6_gen"][m])
    for g in gids:
        b["preds_sample"][g], _ = mbr_dec.apply_from_candidates(b["preds_sample"][g], mbr_cands["j2"][g])
    b["pred_atd_rep"], _ = mbr_dec.apply_from_candidates(b["pred_atd_rep"], mbr_cands["atd_rep"])

    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, p4d, _ = streams[sp_key][m]
            b[b_key][m] = refiner.apply(
                b[b_key][m], p, p4d, X_dict[m], suite.half_v3[m], enable_sliver_and_nocontact=(sp_key != "v6_gen")
            )
    for g in gids:
        p, _, p4d, _ = streams["sample"][g]
        b["preds_sample"][g] = refiner.apply(
            b["preds_sample"][g], p, p4d, X_sample[g], suite.half_sample[g], enable_sliver_and_nocontact=True
        )
    p_atd, c_atd, p4d_atd, fams_atd = streams["atd_rep"]
    b["pred_atd_rep"] = refiner.apply(
        b["pred_atd_rep"], p_atd, p4d_atd, X_atd_rep, half_atd_dec, whistle_z=w_dual, enable_sliver_and_nocontact=True
    )
    b["pred_atd_rep"][second_start:kick2] = DEAD
    b["pred_atd_rep"][~atd.graded] = DEAD

    for sp_key, b_key, X_dict in [
        ("v3", "preds_v3", X_v3),
        ("v6_std", "preds_v6_std", X_v6),
        ("v6_gen", "preds_v6_gen", X_v6),
    ]:
        for m in mids:
            p, _, _, _ = streams[sp_key][m]
            b[b_key][m] = arbitrator.apply(b[b_key][m], p, X_dict[m], suite.half_v3[m], sp_tags_sportec[m])
    for g in gids:
        p, _, _, _ = streams["sample"][g]
        b["preds_sample"][g] = arbitrator.apply(b["preds_sample"][g], p, X_sample[g], suite.half_sample[g], sp_tags_sample[g])
    b["pred_atd_rep"] = arbitrator.apply(
        b["pred_atd_rep"], p_atd, X_atd_rep, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar, whistle_z=w_dual
    )
    b["pred_atd_rep"][second_start:kick2] = DEAD
    b["pred_atd_rep"][~atd.graded] = DEAD

    b_strict = _clone_bundle(b)
    for g in gids:
        train_g = [tg for tg in gids if tg != g]
        cal_j2 = SymDomainTransferCalibratorV9().fit_domain_cv([mbr_cands["j2"][tg] for tg in train_g])
        b["preds_sample"][g], _ = cal_j2.apply_from_candidates(b["preds_sample"][g], mbr_cands["j2"][g])
    atd_grand_pred, _ = dom_cal_full.apply_from_candidates(b["pred_atd_rep"], mbr_cands["atd_rep"])
    atd_grand_pred[second_start:kick2] = DEAD
    atd_grand_pred[~atd.graded] = DEAD
    b["pred_atd_rep"] = atd_grand_pred
    with output_path.open("xb") as f:
        pickle.dump({"strict": b_strict, "grand": b}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return b


def rebuild_v9_baseline(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this reconstruction module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)
    for directory in (root / "scripts", root / "evaluation", root / "py_pkgs"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    swarm = root / "experiments/outputs/v9_swarm"
    with (swarm / "baseline_preds_cache.pkl").open("rb") as handle:
        baselines = pickle.load(handle)
    with (swarm / "_diag_streams_s3.pkl").open("rb") as handle:
        streams = pickle.load(handle)["streams"]
    sportec = load_sportec()
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    suite = SimpleNamespace(
        sportec=sportec, atd=atd, baselines=baselines,
        gt_v3={mid: match.labels for mid, match in sportec.items()},
        half_v3={mid: match.half for mid, match in sportec.items()},
        gt_sample={gid: match.labels for gid, match in samples.items()},
        half_sample={gid: match.half for gid, match in samples.items()},
    )
    def features(key, half):
        paths = _feature_cache_paths(root / "local_data/cache", key)
        return pd.concat([change_features(pd.read_parquet(paths[0]), half),
                          *(pd.read_parquet(path) for path in paths[1:])], axis=1)
    x3 = {mid: features(f"sportec_{mid}_degrade_v3_s0", match.half) for mid, match in sportec.items()}
    x6 = {mid: features(f"sportec_{mid}_degrade_v6_s0", match.half) for mid, match in sportec.items()}
    xs = {gid: features(f"sample_{gid}_raw", match.half) for gid, match in samples.items()}
    xa = features("atd_rep_True_True_None", atd.half)
    build_v9_grand_baseline(suite, streams, x3, x6, xs, xa, root, output_path)
    return output_path
