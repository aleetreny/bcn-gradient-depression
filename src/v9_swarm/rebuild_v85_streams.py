"""Assemble a local v8.5 stream cache from historical predictions and candidates.

Uses the historical fixed decoder settings, including DeadWedge. This is a
replay of existing inputs and does not establish their selection provenance.
"""
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

from src.models.transition import change_features
from src.v9_swarm.rebuild_candidates import _feature_cache_paths


def rebuild_v85_streams(root: Path, output_path: Path) -> Path:
    """Rebuild the historical stream cache without scoring the target match."""
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this reconstruction module")
    output_path = root / output_path
    local_path = output_path.resolve()
    try:
        relative = local_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if local_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.is_symlink() or local_path.exists():
        raise FileExistsError(output_path)

    cache = _assemble_streams(root)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with local_path.open("xb") as handle:
        try:
            pickle.dump(cache, handle)
            handle.flush()
        except BaseException:
            local_path.unlink()
            raise
    return local_path


def _assemble_streams(root: Path) -> dict:
    for directory in (root / "scripts", root / "evaluation", root / "py_pkgs"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))

    from games import GAMES as GAME_INFO, anchor_video_frame
    from src.data.common import DEAD
    from src.data.metrica_atd import load as load_atd
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import MATCH_IDS, load_all as load_sportec
    from src.decode.forensic_shields_v85 import (
        PostImpactTurnoverClassifier, PostImpactTurnoverConfig,
        ShieldedBoundaryOffsetAligner, ShieldedTouchTypeArbitrator
    )
    from src.decode.hybrid_gate_v8 import blend_6fam_heads_v8, compute_hybrid_gate_v8, compute_w_clean_v8
    from src.decode.spell_reranker import apply_spell_reranker
    from src.decode.transductive_calib import DecoupledSynthConfig, calibrate_match_transductive, synthesize_head_decoupled_proba
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
    from exp_v6_super_champion import compute_physical_gates
    from scripts.forensic_v7.run_unified_v8_forensic_champion import apply_vis_gated_dead_wedge

    OUT_DIR = root / "experiments" / "outputs"
    FOR_DIR = OUT_DIR / "forensic_v7"


    def load_physv2_table(key, half):
        v1_path, *feature_paths = _feature_cache_paths(root / "local_data" / "cache", key)
        base = change_features(pd.read_parquet(v1_path), half)
        return pd.concat([base, *(pd.read_parquet(path) for path in feature_paths)], axis=1)

    sportec = load_sportec()
    matches = sorted(sportec.keys())
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3, X_v6, gates_v3, gates_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
        X_v6[mid] = load_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in GAMES:
        X_sample[gid] = load_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half)
        gates_sample[gid] = compute_physical_gates(X_sample[gid], sample_matches[gid].half)

    X_atd_rep = load_physv2_table("atd_rep_True_True_None", atd.half)
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    sym_dir = OUT_DIR / "exp_b_sym_oof"
    plus_dir = OUT_DIR / "exp_c_plus_oof"
    g_dir = OUT_DIR / "exp_g_oof"
    h_dir = OUT_DIR / "exp_h_oof"
    w3_dir = OUT_DIR / "exp_w3_oof"
    w4_dir = OUT_DIR / "exp_w4_oof"
    w5_dir = OUT_DIR / "exp_w5_tree_oof"
    w5b_dir = OUT_DIR / "exp_w5b_oof"
    w6_dir = OUT_DIR / "exp_w6_oof"
    w7a_dir = OUT_DIR / "exp_w7a_oof"
    w7b_dir = OUT_DIR / "exp_w7b_oof"

    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        _, _, _, dec_v5_hyb = pickle.load(f)

    oof = {}
    for T in MATCH_IDS:
        zt = np.load(sym_dir / f"tcn_sym_{T}.npz")
        zp = np.load(plus_dir / f"plus_{T}.npz")
        zg = np.load(g_dir / f"ghost_sym_{T}.npz")
        zh = np.load(h_dir / f"bigru_sym_{T}.npz")
        zw3 = np.load(w3_dir / f"b4d_{T}.npz")
        zw4 = np.load(w4_dir / f"physv2_{T}.npz")
        zw5 = np.load(w5_dir / f"cb_xgb_{T}.npz")
        zw5b_tcn = np.load(w5b_dir / f"tcn_physv2_{T}.npz")
        zw5b_gru = np.load(w5b_dir / f"bigru_physv2_{T}.npz")
        zw7a = np.load(w7a_dir / f"pset_{T}.npz")
        zw7b_v6 = np.load(w7b_dir / f"v6_genuine_{T}.npz")
        oof[T] = {
            "g_pl": zp["symgbm_plus_tta"], "tr_pl": zp["symtrans_plus_tta"],
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"], "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v3": zg["g_v6_on_v3"], "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v6_on_v6": zg["g_v6_on_v6"], "tr_v6_on_v6": zg["tr_v6_on_v6"],
            "tcn": zt["symtcn_tta_proba"], "tcnc": zt["symtcn_tta_change"],
            "gru": zh["symbigru_tta_proba"], "gruc": zh["symbigru_tta_change"],
            "p4d_v3": zw3["p4d_v3"], "p4d_v6": zw3["p4d_v6"],
            "g_p2_v3": zw4["symgbm_physv2_tta"], "tr_p2_v3": zw4["symtrans_physv2_tta"],
            "g_p2_v6": zw4["symgbm_physv2_tta_on_v6"], "tr_p2_v6": zw4["symtrans_physv2_tta_on_v6"],
            "cb_p2_v3": zw5["symcatboost_physv2_tta"], "xgb_p2_v3": zw5["symxgb_physv2_tta"],
            "cb_p2_v6": zw5["symcatboost_physv2_tta_on_v6"], "xgb_p2_v6": zw5["symxgb_physv2_tta_on_v6"],
            "tcn_p2_v3": zw5b_tcn["v3_proba"], "tcnc_p2_v3": zw5b_tcn["v3_change"],
            "tcn_p2_v6": zw5b_tcn["v6_proba"], "tcnc_p2_v6": zw5b_tcn["v6_change"],
            "gru_p2_v3": zw5b_gru["v3_proba"], "gruc_p2_v3": zw5b_gru["v3_change"],
            "gru_p2_v6": zw5b_gru["v6_proba"], "gruc_p2_v6": zw5b_gru["v6_change"],
            "pset_v3": zw7a["v3_proba"], "psetc_v3": zw7a["v3_change"],
            "pset_v6": zw7a["v6_proba"], "psetc_v6": zw7a["v6_change"],
        }

    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f: pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f: sample_v6, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f: sample_h, atd_rep_h = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f: b4d_j23 = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f: pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f: pe_w5 = pickle.load(f)
    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f: pe_w5b = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f: pset_j23 = pickle.load(f)
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f: w6_sum = pickle.load(f)
    with open(FOR_DIR / "v85_cands_cache.pkl", "rb") as f: v8_cands_bundle = pickle.load(f)

    best_calib_cfg = w6_sum["best_calib_cfg"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)
    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20), w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20), w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    cache = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_rep": {}}
    for m_te in matches:
        tr_ms = [m for m in matches if m != m_te]
        hl = sportec[m_te].half
        d = oof[m_te]
        for tr, X_df, g_t in [("v3", X_v3[m_te], gates_v3[m_te]), ("v6_std", X_v6[m_te], gates_v6[m_te]), ("v6_gen", X_v6[m_te], gates_v6[m_te])]:
            preds_rr, m1_d, m2_d, bdy_d, pit_d = v8_cands_bundle["hyb"][tr]
            if tr == "v3":
                a = compute_hybrid_gate_v8(X_df, hl, g_t[0])
                wc = compute_w_clean_v8(X_df, hl, g_t[0])
                p, c = synthesize_head_decoupled_proba(d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"], hl, d["g_v6_on_v3"], d["tr_v6_on_v3"], w_ghost=g_t[0], w_gap=g_t[1], cfg=scfg_main, tcn_physv2_proba=d["tcn_p2_v3"], bigru_physv2_proba=d["gru_p2_v3"], tcn_physv2_change=d["tcnc_p2_v3"], bigru_physv2_change=d["gruc_p2_v3"], alpha_p2=a)
                p, c = blend_6fam_heads_v8(p, c, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"], d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"], a, wc, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True)
                p = calibrate_match_transductive(p, hl, X_df, best_calib_cfg).astype(np.float64)
                c = c.astype(np.float64)
                p4d = d["p4d_v3"].astype(np.float64)
                f64 = [x.astype(np.float64) for x in [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]]
            elif tr == "v6_std":
                a = compute_hybrid_gate_v8(X_df, hl, g_t[0])
                wc = compute_w_clean_v8(X_df, hl, g_t[0])
                p, c = synthesize_head_decoupled_proba(d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"], hl, d["g_v6_on_v6"], d["tr_v6_on_v6"], w_ghost=g_t[0], w_gap=g_t[1], cfg=scfg_main, tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"], tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"], alpha_p2=a)
                p, c = blend_6fam_heads_v8(p, c, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"], d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"], a, wc, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True)
                p = calibrate_match_transductive(p, hl, X_df, best_calib_cfg).astype(np.float64)
                c = c.astype(np.float64)
                p4d = d["p4d_v6"].astype(np.float64)
                f64 = [x.astype(np.float64) for x in [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]]
            else:
                wc = compute_w_clean_v8(X_df, hl, g_t[0])
                p, c = synthesize_head_decoupled_proba(d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"], hl, d["g_v6_on_v6"], d["tr_v6_on_v6"], w_ghost=g_t[0], w_gap=g_t[1], cfg=scfg_main, tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"], tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"], alpha_p2=np.ones(len(hl), dtype=np.float32))
                p, c = blend_6fam_heads_v8(p, c, d["tcn_p2_v6"], d["gru_p2_v6"], d["gru_p2_v6"], d["pset_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"], d["gruc_p2_v6"], d["psetc_v6"], np.ones(len(hl), dtype=np.float32), wc, 0.05, 0.30, 0.18, 0.08, adaptive_regime=False)
                p = calibrate_match_transductive(p, hl, X_df, best_calib_cfg).astype(np.float64)
                c = c.astype(np.float64)
                p4d = d["p4d_v6"].astype(np.float64)
                f64 = [x.astype(np.float64) for x in [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]]

            pr_base = preds_rr[m_te].copy()
            pr = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms).apply(pr_base.copy(), p, hl, X_df, c, p4d, fams=f64, m1_cands=m1_d[m_te], m2_cands=m2_d[m_te])
            if tr != "v6_gen":
                pr = apply_vis_gated_dead_wedge(pr, p, hl, X_df, c, p4d, vis_thresh=0.85)
            pr = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit({m: bdy_d[m] for m in tr_ms}).apply(pr, p, hl, X_df, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base, precomputed_cands=bdy_d[m_te])
            if tr == "v3":
                pr = PostImpactTurnoverClassifier(PostImpactTurnoverConfig(enable_missing_player_domino=False)).fit_from_candidates(pit_d, tr_ms).apply(pr, p, hl, X_df, c, p4d, fams=f64, cands=pit_d[m_te])
            cache[tr][m_te] = {"pred": pr, "proba": p.astype(np.float32), "pchange": c.astype(np.float32), "p4d": p4d.astype(np.float32)}

    _, m1_v3, m2_v3, bdy_v3, pit_v3 = v8_cands_bundle["hyb"]["v3"]
    arb_full = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_v3, m2_v3, matches)
    aln_full = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_v3)
    pit_full = PostImpactTurnoverClassifier().fit_from_candidates(pit_v3, matches)

    for gid in GAMES:
        ms = sample_matches[gid]
        dh, dpl, dv6, dp2, dw5, dw5b, dpset = sample_h[gid], pe_plus["sample"][gid], sample_v6[gid], pe_p2["sample"][gid], pe_w5["sample"][gid], pe_w5b["sample"][gid], pset_j23["sample"][gid]
        a_s = compute_hybrid_gate_v8(X_sample[gid], ms.half, gates_sample[gid][0])
        wc_s = compute_w_clean_v8(X_sample[gid], ms.half, gates_sample[gid][0])
        ps, cs = synthesize_head_decoupled_proba(dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"], ms.half, dv6["g_v6"], dv6["tr_v6"], w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main, tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"], tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"], alpha_p2=a_s)
        ps, cs = blend_6fam_heads_v8(ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"], dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"], a_s, wc_s, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True)
        ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
        fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]
        p4d_s = b4d_j23["sample"][gid]
        pr_xdx = apply_xdx_arbitrator(decode_v4(ps, ms.half, dec_v5_hyb, p_change=cs), ps, ms.half, X_sample[gid], p_change=cs, p_trans_4d=p4d_s, cfg=xdx_cfg_st1)
        pr_base = apply_spell_reranker(pr_xdx, ps, ms.half, X_sample[gid], cs, p4d_s, fs, cfg=full_rerank_cfg)
        p, c, p4d, f64 = ps.astype(np.float64), cs.astype(np.float64), p4d_s.astype(np.float64), [x.astype(np.float64) for x in fs]
        pr = arb_full.apply(pr_base.copy(), p, ms.half, X_sample[gid], c, p4d, fams=f64)
        pr = apply_vis_gated_dead_wedge(pr, p, ms.half, X_sample[gid], c, p4d, vis_thresh=0.85)
        pr = aln_full.apply(pr, p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base)
        cache["sample"][gid] = {"pred": pr, "proba": ps.astype(np.float32), "pchange": cs.astype(np.float32), "p4d": p4d_s.astype(np.float32)}

    dpl_a, dp2_a, dw5_a, dw5b_a, dpset_a = pe_plus["atd_rep"], pe_p2["atd_rep"], pe_w5["atd_rep"], pe_w5b["atd_rep"], pset_j23["atd_rep"]
    t2_p, g2_p, t2_c, g2_c, gv6_p, gv6_c = atd_rep_h["t_sym"], atd_rep_h["b_sym"], atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"]
    a_a = compute_hybrid_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0])
    wc_a = compute_w_clean_v8(X_atd_rep, atd.half, gate_atd_rep[0])
    pa, ca = synthesize_head_decoupled_proba(dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], t2_c, g2_c, atd.half, gv6_p, gv6_c, w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg_main, tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"], tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"], alpha_p2=a_a)
    pa, ca = blend_6fam_heads_v8(pa, ca, t2_p, g2_p, dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"], t2_c, g2_c, dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"], a_a, wc_a, 0.05, 0.30, 0.18, 0.08, adaptive_regime=True)
    pa = calibrate_match_transductive(pa, half_atd_dec, X_atd_rep, best_calib_cfg)
    fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
    p4d_a = b4d_j23["atd_rep"]
    pr_xdx_a = apply_xdx_arbitrator(decode_v4(pa, half_atd_dec, dec_v5_hyb, p_change=ca), pa, half_atd_dec, X_atd_rep, p_change=ca, p_trans_4d=p4d_a, cfg=xdx_cfg_st1)
    pr_base_a = apply_spell_reranker(pr_xdx_a, pa, half_atd_dec, X_atd_rep, ca, p4d_a, fa, cfg=full_rerank_cfg)
    p, c, p4d, f64 = pa.astype(np.float64), ca.astype(np.float64), p4d_a.astype(np.float64), [x.astype(np.float64) for x in fa]
    pr_a = arb_full.apply(pr_base_a.copy(), p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
    pr_a = apply_vis_gated_dead_wedge(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, vis_thresh=0.85)
    pr_a = aln_full.apply(pr_a, p, half_atd_dec, X_atd_rep, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base_a)
    pr_a = pit_full.apply(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
    pr_a[second_start:kick2] = DEAD
    pr_a[~atd.graded] = DEAD
    cache["atd_rep"] = {"pred": pr_a, "proba": pa.astype(np.float32), "pchange": ca.astype(np.float32), "p4d": p4d_a.astype(np.float32)}

    for regime in ("v3", "v6_std", "v6_gen"):
        for match_id in cache[regime]:
            cache[regime][match_id]["labels"] = sportec[match_id].labels
            cache[regime][match_id]["half"] = sportec[match_id].half
    for game_id in cache["sample"]:
        cache["sample"][game_id]["labels"] = sample_matches[game_id].labels
        cache["sample"][game_id]["half"] = sample_matches[game_id].half
    return cache
