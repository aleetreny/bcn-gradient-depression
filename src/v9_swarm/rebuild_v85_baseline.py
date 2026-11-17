"""Rebuild the v8.5 strict baseline from local verified intermediate streams.

The result includes only the strict baseline needed by the v9 reconstruction.
No public-window scoring is performed and the output remains local.
"""
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from src.data.common import DEAD
from src.data.metrica_atd import load as load_atd
from src.models.transition import change_features
from src.v9_swarm.rebuild_candidates import _feature_cache_paths
from src.decode.forensic_shields_v85 import ShieldedBoundaryOffsetAligner, ShieldedTouchTypeArbitrator
from src.decode.spell_reranker import apply_spell_reranker
from src.decode.viterbi_hsmm import decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.v10_first_principles.d3_reteam_and_repro import apply_vis_gated_dead_wedge_v85
from src.v10_first_principles.rebuild_trajectory_inference import _verified_file


def rebuild_v85_baseline(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
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
    config = json.loads((root / "_local/v85_baseline_inputs.json").read_text())
    if set(config) != {"v85_streams", "control_streams", "diagnostic_streams"}:
        raise ValueError("configuration must name v85_streams, control_streams and diagnostic_streams")
    with _verified_file(root, config["v85_streams"]).open("rb") as f:
        streams = pickle.load(f)
    with _verified_file(root, config["control_streams"]).open("rb") as f:
        control = pickle.load(f)
    with _verified_file(root, config["diagnostic_streams"]).open("rb") as f:
        diagnostic = pickle.load(f)["streams"]
    outputs = root / "experiments/outputs"
    with (outputs / "forensic_v7/v85_cands_cache.pkl").open("rb") as f:
        candidates = pickle.load(f)
    with (outputs / "exp_w6_oof/w6_results_summary.pkl").open("rb") as f:
        rerank = pickle.load(f)["full_rerank_cfg"]
    with (outputs / "exp_b_sym_oof/dec_v5_sym.pkl").open("rb") as f:
        _, _, _, decoder = pickle.load(f)
    for directory in (root / "scripts", root / "evaluation"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from games import GAMES as GAME_INFO, anchor_video_frame

    atd = load_atd()
    half = np.where(atd.graded, atd.half, 0)
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kickoff = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    matches = sorted(streams["v3"])
    _, mode1, mode2, boundary, _ = candidates["hyb"]["v3"]
    arbitrator = ShieldedTouchTypeArbitrator().fit_from_candidates(mode1, mode2, matches)
    aligner = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(boundary)
    cfg = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)
    baseline = {f"preds_{key}": {mid: entry["pred"].copy() for mid, entry in streams[key].items()}
                for key in ("v3", "v6_std", "v6_gen", "sample")}
    # The historical v8.5 stream cache can include a post-impact step on v3.
    # Recompute the strict pre-impact v3 path rather than assuming it is inert.
    for mid in matches:
        entry = streams["v3"][mid]
        raw_p, raw_c, raw_p4d, raw_families = diagnostic["v3"][mid]
        if not (np.array_equal(entry["proba"], raw_p.astype(np.float32)) and
                np.array_equal(entry["pchange"], raw_c.astype(np.float32)) and
                np.array_equal(entry["p4d"], raw_p4d.astype(np.float32))):
            raise ValueError(f"v8.5 and diagnostic streams disagree for {mid}")
        match_half = entry["half"]
        paths = _feature_cache_paths(root / "local_data/cache", f"sportec_{mid}_degrade_v3_s0")
        features = pd.concat([change_features(pd.read_parquet(paths[0]), match_half),
                              *(pd.read_parquet(path) for path in paths[1:])], axis=1)
        preds, m1, m2, boundaries, _ = candidates["hyb"]["v3"]
        base = preds[mid].copy()
        train = [other for other in matches if other != mid]
        p, c, p4d = raw_p.astype(np.float64), raw_c.astype(np.float64), raw_p4d.astype(np.float64)
        families = [values.astype(np.float64) for values in raw_families]
        arb = ShieldedTouchTypeArbitrator().fit_from_candidates(m1, m2, train)
        pred = arb.apply(base.copy(), p, match_half, features, c, p4d, fams=families,
                         m1_cands=m1[mid], m2_cands=m2[mid])
        pred = apply_vis_gated_dead_wedge_v85(pred, p, match_half, features, c, p4d)
        aln = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit(
            {other: boundaries[other] for other in train})
        baseline["preds_v3"][mid] = aln.apply(
            pred, p, match_half, features, p_change=c, p_trans_4d=p4d,
            family_probas=families, base_pred=base, precomputed_cands=boundaries[mid])
    for name in ("atd_rep", "atd_raw"):
        raw_p, raw_c, raw_p4d, raw_families = control["hyb"][name]
        features = control[f"X_{name}"]
        p, c, p4d = raw_p.astype(np.float64), raw_c.astype(np.float64), raw_p4d.astype(np.float64)
        families = [values.astype(np.float64) for values in raw_families]
        decoded = decode_v4(raw_p, half, decoder, p_change=raw_c)
        decoded = apply_xdx_arbitrator(decoded, raw_p, half, features, p_change=raw_c, p_trans_4d=raw_p4d, cfg=cfg)
        base = apply_spell_reranker(decoded, raw_p, half, features, raw_c, raw_p4d, raw_families, cfg=rerank)
        pred = arbitrator.apply(base.copy(), p, half, features, c, p4d, fams=families)
        pred = apply_vis_gated_dead_wedge_v85(pred, p, half, features, c, p4d)
        pred = aligner.apply(pred, p, half, features, p_change=c, p_trans_4d=p4d,
                             family_probas=families, base_pred=base)
        pred[second_start:kickoff] = DEAD
        pred[~atd.graded] = DEAD
        baseline[f"pred_{name}"] = pred
    with output_path.open("xb") as f:
        pickle.dump({"v85_forensic_strict_cv": baseline}, f, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
