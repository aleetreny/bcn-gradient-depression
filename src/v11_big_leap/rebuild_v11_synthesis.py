"""Synthesize v11 prediction bundles from explicit local inputs without J3 scoring.

This reproduces the historical algorithm using the supplied inputs. It does
not validate their provenance, establish rule eligibility, or submit a file.
All generated match predictions remain local.
"""
from pathlib import Path
import json
import pickle
import sys
from types import SimpleNamespace

import numpy as np

from src.data.common import DEAD
from src.data.metrica_atd import load as load_atd
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import load_all as load_sportec
from src.v9_swarm.duel_and_mbr_v9 import SymDuelTurnoverArbitratorV9, SymSpellMBRDecoderV9
from src.v9_swarm.rebuild_v9_baseline import _clone_bundle
from src.v9_swarm.smart_tagging_arbitrator import SmartTaggingArbitrator, load_bar_ath_smart_tags
from src.v9_swarm.whistle_and_stoppage import Fox40WhistleDetector, StoppageStartRefiner
from src.v10_first_principles.d3_reteam_and_repro import apply_d3_post_kick_sliver_bridge, decode_v85_bar_ath, load_v85_fitted_shield_models
from src.v10_first_principles.rebuild_b1_b2 import _features
from src.v10_first_principles.rebuild_trajectory_inference import _verified_file
from src.v11_big_leap.unified_stream_and_lattice_v11 import SymBigLeapLatticeV11, SymTouchBoundaryAlignerV11, synthesize_unified_stream_v11


def rebuild_v11_synthesis(root: Path, output_path: Path) -> Path:
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
    for directory in (root / "scripts", root / "evaluation", root / "py_pkgs"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from games import GAMES as GAME_INFO, anchor_video_frame

    config = json.loads((root / "_local/v11_synthesis_inputs.json").read_text())
    if set(config) != {"v9", "candidates", "d3_streams"}:
        raise ValueError("configuration must name v9, candidates and d3_streams")
    def read(path):
        with path.open("rb") as handle:
            return pickle.load(handle)
    v9_cache = read(_verified_file(root, config["v9"]))
    c11 = read(_verified_file(root, config["candidates"]))
    d3_verified = read(_verified_file(root, config["d3_streams"]))
    v9_strict, v9_grand = v9_cache["strict"], v9_cache["grand"]
    outputs = root / "experiments/outputs"
    swarm, v10, v11 = outputs / "v9_swarm", outputs / "v10_first_principles", outputs / "v11_big_leap"
    streams = read(swarm / "_diag_streams_s3.pkl")["streams"]
    duel_cands = read(swarm / "_diag_enr_cands.pkl")["duel"]
    ps_j23 = read(swarm / "pset_v2_oof/pset_v2_j23_preds.pkl")
    l1_j23 = read(swarm / "l1_trees_oof/l1_trees_j23_preds.pkl")
    t10_j23 = read(v10 / "traj_v10_oof/traj_v10_j23_preds.pkl")
    pc11_j23 = read(v11 / "phys_conformer_v11_oof/pc11_j23_preds.pkl")
    sportec = load_sportec()
    samples = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    mids, gids = sorted(sportec), sorted(samples)
    suite = SimpleNamespace(half_sample={gid: m.half for gid, m in samples.items()})
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)
    X_sample = {gid: _features(root, f"sample_{gid}_raw", samples[gid].half) for gid in gids}
    ps_sample = {g: {key[5:]: value for key, value in ps_j23["sample"][g].items()} for g in gids}
    l1_sample, t10_sample, pc11_sample = l1_j23["sample"], t10_j23["sample"], pc11_j23["sample"]
    ps_atd_rt = {key[5:]: val for key, val in ps_j23["atd_reteam_rep"].items()}
    l1_atd_rt, t10_atd_rt, pc11_atd_rt = l1_j23["atd_reteam_rep"], t10_j23["atd_reteam_rep"], pc11_j23["atd_reteam_rep"]
    v9_air_rt = 0.5 * (np.asarray(ps_j23["atd_reteam_rep"]["pset_p_aerial"], dtype=np.float64) + np.asarray(l1_atd_rt["air"], dtype=np.float64))
    v9_duel_rt = np.asarray(l1_atd_rt["duel"], dtype=np.float64)
    whistle = root / "local_data/v9_swarm/bar_ath_whistle.npz"
    w_dual = Fox40WhistleDetector().extract_and_cache_bar_ath_whistle(root, cache_path=whistle, cleanup_wav=True)["w_dual"]
    sp_tags_bar, phase_score_bar = load_bar_ath_smart_tags(root, n_frames=atd.n_frames)
    # 2. Build v11 Strict-CV and v11 Grand bundles in Strict 7-Fold Nested LOMO
    v11_strict = _clone_bundle(v9_strict)
    v11_grand = _clone_bundle(v9_grand)

    for m in mids:
        tr_m = [tm for tm in mids if tm != m]
        lat = SymBigLeapLatticeV11().fit_lomo(
            [c11["mbr"]["v3"][tm] for tm in tr_m],
            [c11["b1"]["XYX"]["v3"][tm] for tm in tr_m],
            [c11["b1"]["XYD"]["v3"][tm] for tm in tr_m],
            [c11["b1"]["DYX"]["v3"][tm] + c11["b1"]["DYX"]["v6_std"][tm] for tm in tr_m],
            [c11["vit"]["v3"][tm] for tm in tr_m],
        )
        al = SymTouchBoundaryAlignerV11().fit_lomo([c11["b2"]["v3"][tm] for tm in tr_m])
        for sp_key, b_key in [("v3", "preds_v3"), ("v6_std", "preds_v6_std"), ("v6_gen", "preds_v6_gen")]:
            p_cur, _ = lat.apply_from_candidates(
                v11_grand[b_key][m],
                c11["mbr"][sp_key][m],
                c11["b1"]["XYX"][sp_key][m],
                c11["b1"]["XYD"][sp_key][m],
                c11["b1"]["DYX"][sp_key][m],
                vit_cands=c11["vit"][sp_key][m],
            )
            p_cur, _ = al.apply_from_candidates(p_cur, c11["b2"][sp_key][m])
            v11_strict[b_key][m] = p_cur.copy()
            v11_grand[b_key][m] = p_cur.copy()

    lat_full = SymBigLeapLatticeV11().fit_lomo(
        [c11["mbr"]["v3"][tm] for tm in mids],
        [c11["b1"]["XYX"]["v3"][tm] for tm in mids],
        [c11["b1"]["XYD"]["v3"][tm] for tm in mids],
        [c11["b1"]["DYX"]["v3"][tm] + c11["b1"]["DYX"]["v6_std"][tm] for tm in mids],
        [c11["vit"]["v3"][tm] for tm in mids],
    )
    al_full = SymTouchBoundaryAlignerV11().fit_lomo([c11["b2"]["v3"][tm] for tm in mids])

    # Apply to J2 (both on v9_strict for strict_cv and on v9_grand for grand)
    for g in gids:
        p_s, c_s, p4d_s, fams_s = streams["sample"][g]
        # Strict-CV J2 (without 4C)
        pr_st, _ = lat_full.apply(
            v11_strict["preds_sample"][g],
            p_s, c_s, p4d_s, fams_s,
            suite.half_sample[g], X_sample[g], ps_sample[g], l1_sample[g],
            t10=t10_sample[g], pc11=pc11_sample[g],
        )
        # Apply VIT discrepancy candidates on top if valid
        valid_vit_st = [
            c for c in c11["vit"]["sample"][g]
            if np.all(pr_st[int(c["ca"]) : int(c["cb"])] == int(c["x_lbl"]))
        ]
        if valid_vit_st and lat_full.cfg.active_vit:
            from src.v10_first_principles.touch_segment_lattice import apply_scored_candidates
            from src.v11_big_leap.unified_stream_and_lattice_v11 import score_vit_diff_v11
            pr_st, _ = apply_scored_candidates(
                pr_st, valid_vit_st, score_vit_diff_v11, lat_full.cfg.active_vit, lat_full.cfg.th_vit, use_x_lbl=False
            )
        pr_st, _ = al_full.apply(
            pr_st, suite.half_sample[g], p_s, c_s, p4d_s, fams_s, ps_sample[g], l1_sample[g], X_sample[g],
            t10=t10_sample[g], pc11=pc11_sample[g],
        )
        v11_strict["preds_sample"][g] = pr_st

        # Grand J2 (with 4C)
        pr_gr, _ = lat_full.apply_from_candidates(
            v11_grand["preds_sample"][g],
            c11["mbr"]["j2"][g],
            c11["b1"]["XYX"]["sample"][g],
            c11["b1"]["XYD"]["sample"][g],
            c11["b1"]["DYX"]["sample"][g],
            vit_cands=c11["vit"]["sample"][g],
        )
        pr_gr, _ = al_full.apply_from_candidates(pr_gr, c11["b2"]["sample"][g])
        v11_grand["preds_sample"][g] = pr_gr

    # Apply to BAR-ATH Control (for strict_cv and grand)
    p_atd, c_atd, p4d_atd, fams_atd = streams["atd_rep"]
    pr_atd_st, f_atd_st = lat_full.apply_from_candidates(
        v11_strict["pred_atd_rep"],
        c11["atd_ctrl"]["mbr"],
        c11["atd_ctrl"]["b1"]["XYX"],
        c11["atd_ctrl"]["b1"]["XYD"],
        c11["atd_ctrl"]["b1"]["DYX"],
        vit_cands=c11["atd_ctrl"]["vit"],
    )
    pr_atd_st, f_b2_st = al_full.apply_from_candidates(pr_atd_st, c11["atd_ctrl"]["b2"])
    pr_atd_st[second_start:kick2] = DEAD
    pr_atd_st[~atd.graded] = DEAD
    v11_strict["pred_atd_rep"] = pr_atd_st

    pr_atd_gr, f_atd_gr = lat_full.apply_from_candidates(
        v11_grand["pred_atd_rep"],
        c11["atd_ctrl"]["mbr"],
        c11["atd_ctrl"]["b1"]["XYX"],
        c11["atd_ctrl"]["b1"]["XYD"],
        c11["atd_ctrl"]["b1"]["DYX"],
        vit_cands=c11["atd_ctrl"]["vit"],
    )
    pr_atd_gr, f_b2_gr = al_full.apply_from_candidates(pr_atd_gr, c11["atd_ctrl"]["b2"])
    pr_atd_gr[second_start:kick2] = DEAD
    pr_atd_gr[~atd.graded] = DEAD
    v11_grand["pred_atd_rep"] = pr_atd_gr

    # 3. Build v11 D3 True Video Re-Teamed BAR-ATH Prediction
    X_atd_rt = d3_verified["X_atd_rep"]
    st_rt = d3_verified["hyb"]["atd_rep"]
    p_rt, c_rt, p4d_rt, fams_rt = st_rt
    v85_shield_bundle = load_v85_fitted_shield_models(root)
    pr_v85_rt, _ = decode_v85_bar_ath(st_rt, X_atd_rt, atd, v85_shield_bundle, s_tag="hyb", use_pit_domino=False)

    # Also decode the 13-family unified stream on BAR-ATH D3 reteam for Dual-Viterbi discrepancy (`VIT-DIFF`)
    from src.decode.viterbi_hsmm import decode as decode_v4
    p_u_rt, c_u_rt, p4d_u_rt, fams_u_rt = synthesize_unified_stream_v11(
        p_rt, c_rt, p4d_rt, fams_rt, half_atd_dec, ps_atd_rt, l1_atd_rt, t10=t10_atd_rt, pc11=pc11_atd_rt
    )
    pr_vit11_rt = decode_v4(
        p_u_rt, half_atd_dec, v85_shield_bundle["dec_v5_hyb"], p_change=c_u_rt
    )

    duel_full = SymDuelTurnoverArbitratorV9().fit_lomo([duel_cands["v3"][m] for m in mids])
    mbr_dec = SymSpellMBRDecoderV9()
    refiner = StoppageStartRefiner()
    arbitrator = SmartTaggingArbitrator()

    pr_rt_4a, _ = duel_full.apply(
        pr_v85_rt.copy(), p_rt.astype(np.float64), c_rt.astype(np.float64), p4d_rt.astype(np.float64),
        [fp.astype(np.float64) for fp in fams_rt], half_atd_dec, X_atd_rt
    )
    pr_rt_4ab, _ = mbr_dec.apply(
        pr_rt_4a.copy(), p_rt.astype(np.float64), c_rt.astype(np.float64), p4d_rt.astype(np.float64),
        [fp.astype(np.float64) for fp in fams_rt], half_atd_dec, X_atd_rt,
        v9_air=v9_air_rt, v9_duel=v9_duel_rt
    )
    pr_rt_l4 = refiner.apply(
        pr_rt_4ab.copy(), p_rt, p4d_rt, X_atd_rt, half_atd_dec, whistle_z=w_dual, enable_sliver_and_nocontact=True
    )
    pr_rt_m5 = arbitrator.apply(
        pr_rt_l4.copy(), p_rt, X_atd_rt, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar, whistle_z=w_dual
    )
    pr_rt_bridge = apply_d3_post_kick_sliver_bridge(
        pr_rt_m5.copy(), p_rt, X_atd_rt, half_atd_dec, sp_tags_bar, phase_score=phase_score_bar
    )
    pr_rt_v11, f_rt_b1 = lat_full.apply(
        pr_rt_bridge.copy(), p_rt.astype(np.float64), c_rt.astype(np.float64), p4d_rt.astype(np.float64),
        [fp.astype(np.float64) for fp in fams_rt], half_atd_dec, X_atd_rt, ps_atd_rt, l1_atd_rt,
        t10=t10_atd_rt, pc11=pc11_atd_rt, pred_vit_v11=pr_vit11_rt
    )
    pr_rt_v11, f_rt_b2 = al_full.apply(
        pr_rt_v11, half_atd_dec, p_rt.astype(np.float64), c_rt.astype(np.float64), p4d_rt.astype(np.float64),
        [fp.astype(np.float64) for fp in fams_rt], ps_atd_rt, l1_atd_rt, X_atd_rt,
        t10=t10_atd_rt, pc11=pc11_atd_rt,
    )
    pr_rt_v11[second_start:kick2] = DEAD
    pr_rt_v11[~atd.graded] = DEAD

    v11_d3_reteam = _clone_bundle(v11_grand)
    v11_d3_reteam["pred_atd_rep"] = pr_rt_v11

    with output_path.open("xb") as handle:
        pickle.dump({"strict": v11_strict, "grand": v11_grand, "d3": v11_d3_reteam}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_path
