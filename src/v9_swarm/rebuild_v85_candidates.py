"""Rebuild historical v8.5 Sportec candidate caches without public scoring.

Uses existing model outputs, feature caches and public Sportec labels. Local
match-derived outputs must remain on this machine.
"""
from pathlib import Path
import pickle
import sys

import numpy as np


def rebuild_v85_candidates(root: Path, output_path: Path) -> Path:
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
    from src.data.metrica_atd import load as load_atd
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.sportec import MATCH_IDS, load_all as load_sportec
    from src.decode.boundary_offset_aligner import extract_boundary_candidates
    from src.decode.forensic_shields_v85 import extract_post_impact_turnover_candidates
    from src.decode.hybrid_gate_v8 import blend_6fam_heads_v8, compute_hybrid_gate_v8, compute_inpaint_gate_v8, compute_w_clean_v8
    from src.decode.spell_reranker import apply_spell_reranker
    from src.decode.touch_type_arbitrator import extract_mode1_candidates, extract_mode2_candidates, preclean_tracking_arrays
    from src.decode.transductive_calib import DecoupledSynthConfig, TransductiveCalibConfig, calibrate_match_transductive, compute_seq481_gate, synthesize_head_decoupled_proba
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
    from exp_v6_super_champion import compute_physical_gates
    from exp_w7b_super_champion_v7 import compute_local_newton_gate
    from src.v10_first_principles.rebuild_b1_b2 import _features
    def load_cached_physv2_table(key, half):
        return _features(root, key, half)
    OUT_DIR = root / "experiments/outputs"
    W8A_DIR = OUT_DIR / "exp_w8a_oof"
    sportec = load_sportec()
    matches = sorted(sportec.keys())
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3, X_v6, gates_v3, gates_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
        X_v6[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in GAMES:
        X_sample[gid] = load_cached_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half)
        gates_sample[gid] = compute_physical_gates(X_sample[gid], sample_matches[gid].half)

    X_atd_raw = load_cached_physv2_table("atd", atd.half)
    X_atd_rep = load_cached_physv2_table("atd_rep_True_True_None", atd.half)
    gate_atd_raw = compute_physical_gates(X_atd_raw, atd.half)
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
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)

    oof, inp_oof = {}, {}
    for T in MATCH_IDS:
        zb = np.load(sym_dir / f"sym_{T}.npz")
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
        zw8a = np.load(W8A_DIR / f"w8a_inpaint_{T}.npz")
        oof[T] = {
            "g_v1": zb["symgbm_tta"], "tr_v1": zb["symtrans_tta"],
            "g_pl": zp["symgbm_plus_tta"], "tr_pl": zp["symtrans_plus_tta"],
            "g_pl_on_v6": zw7b_v6["symgbm_plus_tta_on_v6"], "tr_pl_on_v6": zw7b_v6["symtrans_plus_tta_on_v6"],
            "g_v6_on_v3": zg["g_v6_on_v3"], "tr_v6_on_v3": zg["tr_v6_on_v3"],
            "g_v3_on_v6": zg["g_v3_on_v6"], "tr_v3_on_v6": zg["tr_v3_on_v6"],
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
        inp_oof[T] = {
            "p_inp_v3": zw8a["p_inp_v3"],
            "c_inp_v3": zw8a["c_inp_v3"],
            "w_inp_v3": compute_inpaint_gate_v8(X_v3[T], sportec[T].half, gates_v3[T][0], gates_v3[T][1]),
            "p_inp_v6": zw8a["p_inp_v6"],
            "c_inp_v6": zw8a["c_inp_v6"],
            "w_inp_v6": compute_inpaint_gate_v8(X_v6[T], sportec[T].half, gates_v6[T][0], gates_v6[T][1]),
        }

    with open(plus_dir / "eval_preds_plus.pkl", "rb") as f:
        pe_plus = pickle.load(f)
    with open(g_dir / "v6_j23_preds.pkl", "rb") as f:
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(h_dir / "bigru_j23_preds.pkl", "rb") as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(w3_dir / "b4d_j23_preds.pkl", "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(w4_dir / "eval_preds_physv2.pkl", "rb") as f:
        pe_p2 = pickle.load(f)
    with open(w5_dir / "eval_preds_cb_xgb.pkl", "rb") as f:
        pe_w5 = pickle.load(f)
    with open(w5b_dir / "seq_physv2_j23_preds.pkl", "rb") as f:
        pe_w5b = pickle.load(f)
    with open(w6_dir / "atd_raw_v6_bigru.pkl", "rb") as f:
        atd_raw_extra = pickle.load(f)
    with open(w7a_dir / "pset_j23_preds.pkl", "rb") as f:
        pset_j23 = pickle.load(f)
    with open(w6_dir / "w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    with open(W8A_DIR / "w8a_j23_preds.pkl", "rb") as f:
        j23_inp = pickle.load(f)

    for gid in GAMES:
        j23_inp["sample"][gid]["w_inp"] = compute_inpaint_gate_v8(
            X_sample[gid], sample_matches[gid].half, gates_sample[gid][0], gates_sample[gid][1]
        )
    j23_inp["atd_raw"]["w_inp"] = compute_inpaint_gate_v8(X_atd_raw, atd.half, gate_atd_raw[0], gate_atd_raw[1])
    j23_inp["atd_rep"]["w_inp"] = compute_inpaint_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0], gate_atd_rep[1])

    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)

    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    def build_streams(
        s3_d: float = 0.05,
        s3_ha: float = 0.30,
        s3_c: float = 0.18,
        ps_ha: float = 0.08,
        gate_mode: str = "hybrid_v8",
        adaptive_regime: bool = True,
        inp_scale: float = 0.0,
    ) -> dict:
        streams = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_raw": None, "atd_rep": None}

        def _get_alpha(X_df, h_arr, wg_arr):
            if gate_mode == "hybrid_v8":
                return compute_hybrid_gate_v8(X_df, h_arr, wg_arr)
            if gate_mode == "local_newton":
                return compute_local_newton_gate(X_df, h_arr, wg_arr)
            return compute_seq481_gate(X_df, h_arr, wg_arr)

        for T in MATCH_IDS:
            d = oof[T]
            di = inp_oof[T]
            h = sportec[T].half

            a_v3 = _get_alpha(X_v3[T], h, gates_v3[T][0])
            wc_v3 = compute_w_clean_v8(X_v3[T], h, gates_v3[T][0])
            p3, c3 = synthesize_head_decoupled_proba(
                d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v3"], bigru_physv2_proba=d["gru_p2_v3"],
                tcn_physv2_change=d["tcnc_p2_v3"], bigru_physv2_change=d["gruc_p2_v3"],
                alpha_p2=a_v3,
            )
            p3, c3 = blend_6fam_heads_v8(
                p3, c3, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"],
                d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"],
                a_v3, wc_v3, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=di["p_inp_v3"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v3"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v3"]) if inp_scale > 0 else None,
            )
            p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            a_v6 = _get_alpha(X_v6[T], h, gates_v6[T][0])
            wc_v6 = compute_w_clean_v8(X_v6[T], h, gates_v6[T][0])
            p6s, c6s = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn"], d["gru"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=a_v6,
            )
            p6s, c6s = blend_6fam_heads_v8(
                p6s, c6s, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"],
                a_v6, wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=di["p_inp_v6"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v6"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v6"]) if inp_scale > 0 else None,
            )
            p6s = calibrate_match_transductive(p6s, h, X_v6[T], best_calib_cfg)
            f6s = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]
            streams["v6_std"][T] = (p6s, c6s, d["p4d_v6"], f6s)

            p6g, c6g = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn_p2_v6"], d["gru_p2_v6"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=np.ones(len(h), dtype=np.float32),
            )
            p6g, c6g = blend_6fam_heads_v8(
                p6g, c6g, d["tcn_p2_v6"], d["gru_p2_v6"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc_p2_v6"], d["gruc_p2_v6"], d["gruc_p2_v6"], d["psetc_v6"],
                np.ones(len(h), dtype=np.float32), wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=False,
                p_inp=di["p_inp_v6"] if inp_scale > 0 else None,
                c_inp=di["c_inp_v6"] if inp_scale > 0 else None,
                w_inp=(inp_scale * di["w_inp_v6"]) if inp_scale > 0 else None,
            )
            p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)
            f6g = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]
            streams["v6_gen"][T] = (p6g, c6g, d["p4d_v6"], f6g)

        for gid in GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            dp2 = pe_p2["sample"][gid]
            dw5 = pe_w5["sample"][gid]
            dw5b = pe_w5b["sample"][gid]
            dpset = pset_j23["sample"][gid]
            dinp = j23_inp["sample"][gid]
            a_s = _get_alpha(X_sample[gid], ms.half, gates_sample[gid][0])
            wc_s = compute_w_clean_v8(X_sample[gid], ms.half, gates_sample[gid][0])

            ps, cs = synthesize_head_decoupled_proba(
                dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
                dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
                ms.half, dv6["g_v6"], dv6["tr_v6"],
                w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
                tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"],
                tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"],
                alpha_p2=a_s,
            )
            ps, cs = blend_6fam_heads_v8(
                ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
                dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"],
                a_s, wc_s, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=dinp["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp["w_inp"]) if inp_scale > 0 else None,
            )
            ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
            fs = [dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"], dh["t_sym"], dh["b_sym"], dw5b["tcn_physv2_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"]]
            streams["sample"][gid] = (ps, cs, b4d_j23["sample"][gid], fs)

        for vkey, X_df, gate_t, t2_p, g2_p, t2_c, g2_c, gv6_p, gv6_c in [
            ("atd_raw", X_atd_raw, gate_atd_raw, pe_plus["atd_raw"]["t_sym"], atd_raw_extra["b_sym"], pe_plus["atd_raw"]["tc_sym"], atd_raw_extra["bc_sym"], atd_raw_extra["g_v6"], atd_raw_extra["tr_v6"]),
            ("atd_rep", X_atd_rep, gate_atd_rep, atd_rep_h["t_sym"], atd_rep_h["b_sym"], atd_rep_h["tc_sym"], atd_rep_h["bc_sym"], atd_rep_v6["g_v6"], atd_rep_v6["tr_v6"]),
        ]:
            dpl_a = pe_plus[vkey]
            dp2_a = pe_p2[vkey]
            dw5_a = pe_w5[vkey]
            dw5b_a = pe_w5b[vkey]
            dpset_a = pset_j23[vkey]
            dinp_a = j23_inp[vkey]
            a_a = _get_alpha(X_df, atd.half, gate_t[0])
            wc_a = compute_w_clean_v8(X_df, atd.half, gate_t[0])
            pa, ca = synthesize_head_decoupled_proba(
                dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
                t2_p, g2_p, dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], t2_c, g2_c,
                atd.half, gv6_p, gv6_c,
                w_ghost=gate_t[0], w_gap=gate_t[1], cfg=scfg_main,
                tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
                tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
                alpha_p2=a_a,
            )
            pa, ca = blend_6fam_heads_v8(
                pa, ca, t2_p, g2_p, dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"],
                t2_c, g2_c, dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"],
                a_a, wc_a, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=adaptive_regime,
                p_inp=dinp_a["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp_a["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp_a["w_inp"]) if inp_scale > 0 else None,
            )
            pa = calibrate_match_transductive(pa, half_atd_dec, X_df, best_calib_cfg)
            fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
            streams[vkey] = (pa, ca, b4d_j23[vkey], fa)
        return streams

    st_v8_hyb = build_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.0)
    st_v8_inp = build_streams(0.05, 0.30, 0.18, 0.08, gate_mode="hybrid_v8", adaptive_regime=True, inp_scale=0.35)


    v8_cands_bundle = {"hyb": {}, "inp": {}}
    for s_tag, st_cur in [("hyb", st_v8_hyb), ("inp", st_v8_inp)]:
        for tr, X_dict in [("v3", X_v3), ("v6_std", X_v6), ("v6_gen", X_v6)]:
            preds_rr, m1_d, m2_d, bdy_d, pit_d = {}, {}, {}, {}, {}
            for m in matches:
                p, c, p4d, fams = st_cur[tr][m]
                hl = sportec[m].half
                gt = sportec[m].labels
                X_df = X_dict[m]
                p_xdx = apply_xdx_arbitrator(
                    decode_v4(p, hl, loo_dec_hyb[m], p_change=c),
                    p, hl, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
                )
                p_rr = apply_spell_reranker(p_xdx, p, hl, X_df, c, p4d, fams, cfg=lomo_rerank_cfgs[m])
                preds_rr[m] = p_rr
                cxv = preclean_tracking_arrays(X_df, len(p_rr))
                f64 = [f.astype(np.float64) for f in fams]
                m1_d[m] = extract_mode1_candidates(
                    p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
                )
                m2_d[m] = extract_mode2_candidates(
                    p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
                )
                bdy_d[m] = extract_boundary_candidates(
                    p_rr, p.astype(np.float64), hl, X_df,
                    p_change=c.astype(np.float64), p_trans_4d=p4d.astype(np.float64),
                    family_probas=f64, labels=gt,
                )
                pit_d[m] = extract_post_impact_turnover_candidates(
                    p_rr, p.astype(np.float64), hl, cxv, c.astype(np.float64), p4d.astype(np.float64), f64, gt=gt, match_id=m
                )
            v8_cands_bundle[s_tag][tr] = (preds_rr, m1_d, m2_d, bdy_d, pit_d)
    with output_path.open("xb") as handle:
        pickle.dump(v8_cands_bundle, handle)
    return output_path
