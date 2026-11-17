"""Rebuild historical v8.5 diagnostic streams from local intermediate inputs.

The historical path includes DeadWedge; reproducing it does not establish selection-rule eligibility.
The cache is match-derived and must remain local.
"""
from pathlib import Path
import pickle
import sys

import numpy as np


def rebuild_diagnostic_streams(root: Path, output_path: Path) -> Path:
    ROOT = Path(root).resolve()
    if ROOT != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this reconstruction module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)
    for directory in (ROOT / "scripts", ROOT / "scripts" / "forensic_v7", ROOT / "evaluation", ROOT / "py_pkgs"):
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
    from src.data.sportec import load_all as load_sportec
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.metrica_atd import load as load_atd
    import pandas as pd
    from src.models.transition import change_features
    from src.v9_swarm.rebuild_candidates import _feature_cache_paths

    def load_cached_physv2_table(key, half):
        paths = _feature_cache_paths(ROOT / "local_data" / "cache", key)
        return pd.concat([change_features(pd.read_parquet(paths[0]), half),
                          *(pd.read_parquet(path) for path in paths[1:])], axis=1)
    from src.decode.forensic_shields_v85 import (
        ShieldedTouchTypeArbitrator, ShieldedBoundaryOffsetAligner
    )
    from prototype_v7_forensic_upgrades import apply_dead_wedge_arbitrator
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
    from src.decode.spell_reranker import apply_spell_reranker
    from src.decode.hybrid_gate_v8 import blend_6fam_heads_v8, compute_hybrid_gate_v8, compute_w_clean_v8
    from src.decode.transductive_calib import DecoupledSynthConfig, calibrate_match_transductive, synthesize_head_decoupled_proba
    from exp_v6_super_champion import compute_physical_gates

    sportec = load_sportec()
    matches = sorted(sportec.keys())
    sample_matches = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3 = {m: load_cached_physv2_table(f'sportec_{m}_degrade_v3_s0', sportec[m].half) for m in matches}
    X_v6 = {m: load_cached_physv2_table(f'sportec_{m}_degrade_v6_s0', sportec[m].half) for m in matches}
    X_sample = {gid: load_cached_physv2_table(f'sample_{gid}_raw', sample_matches[gid].half) for gid in GAMES}
    X_atd_rep = load_cached_physv2_table('atd_rep_True_True_None', atd.half)

    gates_v3 = {m: compute_physical_gates(X_v3[m], sportec[m].half) for m in matches}
    gates_v6 = {m: compute_physical_gates(X_v6[m], sportec[m].half) for m in matches}
    gates_sample = {gid: compute_physical_gates(X_sample[gid], sample_matches[gid].half) for gid in GAMES}
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    OUT_DIR = ROOT / 'experiments' / 'outputs'
    with open(OUT_DIR / 'forensic_v7' / 'v85_cands_cache.pkl', 'rb') as f:
        v8_cands_bundle = pickle.load(f)
    with open(OUT_DIR / 'exp_w6_oof' / 'w6_results_summary.pkl', 'rb') as f:
        w6_sum = pickle.load(f)
    with open(OUT_DIR / 'exp_b_sym_oof' / 'dec_v5_sym.pkl', 'rb') as f:
        _, _, _, dec_v5_hyb = pickle.load(f)
    with open(OUT_DIR / 'exp_c_plus_oof' / 'eval_preds_plus.pkl', 'rb') as f:
        pe_plus = pickle.load(f)
    with open(OUT_DIR / 'exp_g_oof' / 'v6_j23_preds.pkl', 'rb') as f:
        sample_v6, atd_rep_v6 = pickle.load(f)
    with open(OUT_DIR / 'exp_h_oof' / 'bigru_j23_preds.pkl', 'rb') as f:
        sample_h, atd_rep_h = pickle.load(f)
    with open(OUT_DIR / 'exp_w3_oof' / 'b4d_j23_preds.pkl', 'rb') as f:
        b4d_j23 = pickle.load(f)
    with open(OUT_DIR / 'exp_w4_oof' / 'eval_preds_physv2.pkl', 'rb') as f:
        pe_p2 = pickle.load(f)
    with open(OUT_DIR / 'exp_w5_tree_oof' / 'eval_preds_cb_xgb.pkl', 'rb') as f:
        pe_w5 = pickle.load(f)
    with open(OUT_DIR / 'exp_w5b_oof' / 'seq_physv2_j23_preds.pkl', 'rb') as f:
        pe_w5b = pickle.load(f)
    with open(OUT_DIR / 'exp_w7a_oof' / 'pset_j23_preds.pkl', 'rb') as f:
        pset_j23 = pickle.load(f)

    best_calib_cfg = w6_sum['best_calib_cfg']
    full_rerank_cfg = w6_sum['full_rerank_cfg']
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)
    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )
    s3_d, s3_ha, s3_c, ps_ha = 0.05, 0.30, 0.18, 0.08

    streams = {'v3': {}, 'v6_std': {}, 'v6_gen': {}, 'sample': {}, 'atd_rep': None}
    for T in matches:
        zt = np.load(OUT_DIR / 'exp_b_sym_oof' / f'tcn_sym_{T}.npz')
        zp = np.load(OUT_DIR / 'exp_c_plus_oof' / f'plus_{T}.npz')
        zg = np.load(OUT_DIR / 'exp_g_oof' / f'ghost_sym_{T}.npz')
        zh = np.load(OUT_DIR / 'exp_h_oof' / f'bigru_sym_{T}.npz')
        zw3 = np.load(OUT_DIR / 'exp_w3_oof' / f'b4d_{T}.npz')
        zw4 = np.load(OUT_DIR / 'exp_w4_oof' / f'physv2_{T}.npz')
        zw5 = np.load(OUT_DIR / 'exp_w5_tree_oof' / f'cb_xgb_{T}.npz')
        zw5b_tcn = np.load(OUT_DIR / 'exp_w5b_oof' / f'tcn_physv2_{T}.npz')
        zw5b_gru = np.load(OUT_DIR / 'exp_w5b_oof' / f'bigru_physv2_{T}.npz')
        zw7a = np.load(OUT_DIR / 'exp_w7a_oof' / f'pset_{T}.npz')
        zw7b_v6 = np.load(OUT_DIR / 'exp_w7b_oof' / f'v6_genuine_{T}.npz')
        h = sportec[T].half

        a_v3 = compute_hybrid_gate_v8(X_v3[T], h, gates_v3[T][0])
        wc_v3 = compute_w_clean_v8(X_v3[T], h, gates_v3[T][0])
        p3, c3 = synthesize_head_decoupled_proba(
            zp['symgbm_plus_tta'], zw4['symgbm_physv2_tta'], zw5['symcatboost_physv2_tta'], zw5['symxgb_physv2_tta'],
            zt['symtcn_tta_proba'], zh['symbigru_tta_proba'], zp['symtrans_plus_tta'], zw4['symtrans_physv2_tta'], zt['symtcn_tta_change'], zh['symbigru_tta_change'],
            h, zg['g_v6_on_v3'], zg['tr_v6_on_v3'], w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg_main,
            tcn_physv2_proba=zw5b_tcn['v3_proba'], bigru_physv2_proba=zw5b_gru['v3_proba'],
            tcn_physv2_change=zw5b_tcn['v3_change'], bigru_physv2_change=zw5b_gru['v3_change'], alpha_p2=a_v3
        )
        p3, c3 = blend_6fam_heads_v8(
            p3, c3, zt['symtcn_tta_proba'], zh['symbigru_tta_proba'], zw5b_gru['v3_proba'], zw7a['v3_proba'],
            zt['symtcn_tta_change'], zh['symbigru_tta_change'], zw5b_gru['v3_change'], zw7a['v3_change'],
            a_v3, wc_v3, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True
        )
        p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
        f3 = [zp['symgbm_plus_tta'], zw4['symgbm_physv2_tta'], zw5['symcatboost_physv2_tta'], zw5['symxgb_physv2_tta'],
              zt['symtcn_tta_proba'], zh['symbigru_tta_proba'], zw5b_tcn['v3_proba'], zw5b_gru['v3_proba'], zw7a['v3_proba']]
        streams['v3'][T] = (p3, c3, zw3['p4d_v3'], f3)

        a_v6 = compute_hybrid_gate_v8(X_v6[T], h, gates_v6[T][0])
        wc_v6 = compute_w_clean_v8(X_v6[T], h, gates_v6[T][0])
        p6s, c6s = synthesize_head_decoupled_proba(
            zw7b_v6['symgbm_plus_tta_on_v6'], zw4['symgbm_physv2_tta_on_v6'], zw5['symcatboost_physv2_tta_on_v6'], zw5['symxgb_physv2_tta_on_v6'],
            zt['symtcn_tta_proba'], zh['symbigru_tta_proba'], zw7b_v6['symtrans_plus_tta_on_v6'], zw4['symtrans_physv2_tta_on_v6'], zt['symtcn_tta_change'], zh['symbigru_tta_change'],
            h, zg['g_v6_on_v6'], zg['tr_v6_on_v6'], w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
            tcn_physv2_proba=zw5b_tcn['v6_proba'], bigru_physv2_proba=zw5b_gru['v6_proba'],
            tcn_physv2_change=zw5b_tcn['v6_change'], bigru_physv2_change=zw5b_gru['v6_change'], alpha_p2=a_v6
        )
        p6s, c6s = blend_6fam_heads_v8(
            p6s, c6s, zt['symtcn_tta_proba'], zh['symbigru_tta_proba'], zw5b_gru['v6_proba'], zw7a['v6_proba'],
            zt['symtcn_tta_change'], zh['symbigru_tta_change'], zw5b_gru['v6_change'], zw7a['v6_change'],
            a_v6, wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True
        )
        p6s = calibrate_match_transductive(p6s, h, X_v6[T], best_calib_cfg)
        f6s = [zw7b_v6['symgbm_plus_tta_on_v6'], zw4['symgbm_physv2_tta_on_v6'], zw5['symcatboost_physv2_tta_on_v6'], zw5['symxgb_physv2_tta_on_v6'],
               zt['symtcn_tta_proba'], zh['symbigru_tta_proba'], zw5b_tcn['v6_proba'], zw5b_gru['v6_proba'], zw7a['v6_proba']]
        streams['v6_std'][T] = (p6s, c6s, zw3['p4d_v6'], f6s)

        p6g, c6g = synthesize_head_decoupled_proba(
            zw7b_v6['symgbm_plus_tta_on_v6'], zw4['symgbm_physv2_tta_on_v6'], zw5['symcatboost_physv2_tta_on_v6'], zw5['symxgb_physv2_tta_on_v6'],
            zw5b_tcn['v6_proba'], zw5b_gru['v6_proba'], zw7b_v6['symtrans_plus_tta_on_v6'], zw4['symtrans_physv2_tta_on_v6'], zw5b_tcn['v6_change'], zw5b_gru['v6_change'],
            h, zg['g_v6_on_v6'], zg['tr_v6_on_v6'], w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
            tcn_physv2_proba=zw5b_tcn['v6_proba'], bigru_physv2_proba=zw5b_gru['v6_proba'],
            tcn_physv2_change=zw5b_tcn['v6_change'], bigru_physv2_change=zw5b_gru['v6_change'], alpha_p2=np.ones(len(h), dtype=np.float32)
        )
        p6g, c6g = blend_6fam_heads_v8(
            p6g, c6g, zw5b_tcn['v6_proba'], zw5b_gru['v6_proba'], zw5b_gru['v6_proba'], zw7a['v6_proba'],
            zw5b_tcn['v6_change'], zw5b_gru['v6_change'], zw5b_gru['v6_change'], zw7a['v6_change'],
            np.ones(len(h), dtype=np.float32), wc_v6, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=False
        )
        p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)
        f6g = [zw7b_v6['symgbm_plus_tta_on_v6'], zw4['symgbm_physv2_tta_on_v6'], zw5['symcatboost_physv2_tta_on_v6'], zw5['symxgb_physv2_tta_on_v6'],
               zw5b_tcn['v6_proba'], zw5b_gru['v6_proba'], zg['g_v6_on_v6'], zw7a['v6_proba']]
        streams['v6_gen'][T] = (p6g, c6g, zw3['p4d_v6'], f6g)

    for gid in GAMES:
        ms = sample_matches[gid]
        dh = sample_h[gid]
        dpl = pe_plus['sample'][gid]
        dv6 = sample_v6[gid]
        dp2 = pe_p2['sample'][gid]
        dw5 = pe_w5['sample'][gid]
        dw5b = pe_w5b['sample'][gid]
        dpset = pset_j23['sample'][gid]
        a_s = compute_hybrid_gate_v8(X_sample[gid], ms.half, gates_sample[gid][0])
        wc_s = compute_w_clean_v8(X_sample[gid], ms.half, gates_sample[gid][0])
        ps, cs = synthesize_head_decoupled_proba(
            dpl['g_plus_sym'], dp2['g_physv2_sym'], dw5['cb_physv2_sym'], dw5['xgb_physv2_sym'],
            dh['t_sym'], dh['b_sym'], dpl['tr_plus_sym'], dp2['tr_physv2_sym'], dh['tc_sym'], dh['bc_sym'],
            ms.half, dv6['g_v6'], dv6['tr_v6'], w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
            tcn_physv2_proba=dw5b['tcn_physv2_sym'], bigru_physv2_proba=dw5b['bigru_physv2_sym'],
            tcn_physv2_change=dw5b['tcnc_physv2_sym'], bigru_physv2_change=dw5b['bigruc_physv2_sym'], alpha_p2=a_s
        )
        ps, cs = blend_6fam_heads_v8(
            ps, cs, dh['t_sym'], dh['b_sym'], dw5b['bigru_physv2_sym'], dpset['pset_proba'],
            dh['tc_sym'], dh['bc_sym'], dw5b['bigruc_physv2_sym'], dpset['pset_change'],
            a_s, wc_s, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True
        )
        ps = calibrate_match_transductive(ps, ms.half, X_sample[gid], best_calib_cfg)
        fs = [dpl['g_plus_sym'], dp2['g_physv2_sym'], dw5['cb_physv2_sym'], dw5['xgb_physv2_sym'],
              dh['t_sym'], dh['b_sym'], dw5b['tcn_physv2_sym'], dw5b['bigru_physv2_sym'], dpset['pset_proba']]
        streams['sample'][gid] = (ps, cs, b4d_j23['sample'][gid], fs)

    dpl_a = pe_plus['atd_rep']
    dp2_a = pe_p2['atd_rep']
    dw5_a = pe_w5['atd_rep']
    dw5b_a = pe_w5b['atd_rep']
    dpset_a = pset_j23['atd_rep']
    a_a = compute_hybrid_gate_v8(X_atd_rep, atd.half, gate_atd_rep[0])
    wc_a = compute_w_clean_v8(X_atd_rep, atd.half, gate_atd_rep[0])
    pa, ca = synthesize_head_decoupled_proba(
        dpl_a['g_plus_sym'], dp2_a['g_physv2_sym'], dw5_a['cb_physv2_sym'], dw5_a['xgb_physv2_sym'],
        atd_rep_h['t_sym'], atd_rep_h['b_sym'], dpl_a['tr_plus_sym'], dp2_a['tr_physv2_sym'], atd_rep_h['tc_sym'], atd_rep_h['bc_sym'],
        atd.half, atd_rep_v6['g_v6'], atd_rep_v6['tr_v6'], w_ghost=gate_atd_rep[0], w_gap=gate_atd_rep[1], cfg=scfg_main,
        tcn_physv2_proba=dw5b_a['tcn_physv2_sym'], bigru_physv2_proba=dw5b_a['bigru_physv2_sym'],
        tcn_physv2_change=dw5b_a['tcnc_physv2_sym'], bigru_physv2_change=dw5b_a['bigruc_physv2_sym'], alpha_p2=a_a
    )
    pa, ca = blend_6fam_heads_v8(
        pa, ca, atd_rep_h['t_sym'], atd_rep_h['b_sym'], dw5b_a['bigru_physv2_sym'], dpset_a['pset_proba'],
        atd_rep_h['tc_sym'], atd_rep_h['bc_sym'], dw5b_a['bigruc_physv2_sym'], dpset_a['pset_change'],
        a_a, wc_a, s3_d, s3_ha, s3_c, ps_ha, adaptive_regime=True
    )
    pa = calibrate_match_transductive(pa, half_atd_dec, X_atd_rep, best_calib_cfg)
    fa = [dpl_a['g_plus_sym'], dp2_a['g_physv2_sym'], dw5_a['cb_physv2_sym'], dw5_a['xgb_physv2_sym'],
          atd_rep_h['t_sym'], atd_rep_h['b_sym'], dw5b_a['tcn_physv2_sym'], dw5b_a['bigru_physv2_sym'], dpset_a['pset_proba']]
    streams['atd_rep'] = (pa, ca, b4d_j23['atd_rep'], fa)

    # Also compute pr_s3 (v8.5 output right before PIT) for all tracks!
    _, m1_d_full, m2_d_full, bdy_d_full, pit_d_full = v8_cands_bundle['hyb']['v3']
    arb_shielded_full = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d_full, m2_d_full, matches)
    aln_shielded_full = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_d_full)

    pr_s3 = {'v3': {}, 'v6_std': {}, 'v6_gen': {}, 'sample': {}, 'atd_rep': None}
    pr_base_all = {'v3': {}, 'v6_std': {}, 'v6_gen': {}, 'sample': {}, 'atd_rep': None}
    for m_te in matches:
        tr_ms = [m for m in matches if m != m_te]
        hl = sportec[m_te].half
        for tr, X_df in [('v3', X_v3[m_te]), ('v6_std', X_v6[m_te]), ('v6_gen', X_v6[m_te])]:
            preds_rr, m1_d, m2_d, bdy_d, _ = v8_cands_bundle['hyb'][tr]
            p_raw, c_raw, p4d_raw, fams_raw = streams[tr][m_te]
            p, c, p4d = p_raw.astype(np.float64), c_raw.astype(np.float64), p4d_raw.astype(np.float64)
            f64 = [f.astype(np.float64) for f in fams_raw]
            pr_base = preds_rr[m_te].copy()
            pr_base_all[tr][m_te] = pr_base
            arb = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms)
            pr = arb.apply(pr_base.copy(), p, hl, X_df, c, p4d, fams=f64, m1_cands=m1_d[m_te], m2_cands=m2_d[m_te])
            if tr != 'v6_gen':
                pr = apply_dead_wedge_arbitrator(pr, p, hl, X_df, c, p4d)
            aln = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit({m: bdy_d[m] for m in tr_ms})
            pr = aln.apply(pr, p, hl, X_df, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base, precomputed_cands=bdy_d[m_te])
            pr_s3[tr][m_te] = pr

    for gid in GAMES:
        ms = sample_matches[gid]
        p_raw, c_raw, p4d_raw, fams_raw = streams['sample'][gid]
        p, c, p4d = p_raw.astype(np.float64), c_raw.astype(np.float64), p4d_raw.astype(np.float64)
        f64 = [f.astype(np.float64) for f in fams_raw]
        pr_xdx = apply_xdx_arbitrator(decode_v4(p_raw, ms.half, dec_v5_hyb, p_change=c_raw), p_raw, ms.half, X_sample[gid], p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1)
        pr_base = apply_spell_reranker(pr_xdx, p_raw, ms.half, X_sample[gid], c_raw, p4d_raw, fams_raw, cfg=full_rerank_cfg)
        pr_base_all['sample'][gid] = pr_base
        pr = arb_shielded_full.apply(pr_base.copy(), p, ms.half, X_sample[gid], c, p4d, fams=f64)
        pr = apply_dead_wedge_arbitrator(pr, p, ms.half, X_sample[gid], c, p4d)
        pr = aln_shielded_full.apply(pr, p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base)
        pr_s3['sample'][gid] = pr

    p_raw, c_raw, p4d_raw, fams_raw = streams['atd_rep']
    p, c, p4d = p_raw.astype(np.float64), c_raw.astype(np.float64), p4d_raw.astype(np.float64)
    f64 = [f.astype(np.float64) for f in fams_raw]
    pr_xdx_a = apply_xdx_arbitrator(decode_v4(p_raw, half_atd_dec, dec_v5_hyb, p_change=c_raw), p_raw, half_atd_dec, X_atd_rep, p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1)
    pr_base_a = apply_spell_reranker(pr_xdx_a, p_raw, half_atd_dec, X_atd_rep, c_raw, p4d_raw, fams_raw, cfg=full_rerank_cfg)
    pr_base_all['atd_rep'] = pr_base_a
    pr_a = arb_shielded_full.apply(pr_base_a.copy(), p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
    pr_a = apply_dead_wedge_arbitrator(pr_a, p, half_atd_dec, X_atd_rep, c, p4d)
    pr_a = aln_shielded_full.apply(pr_a, p, half_atd_dec, X_atd_rep, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base_a)
    pr_s3['atd_rep'] = pr_a

    with output_path.open("xb") as f:
        pickle.dump({"streams": streams, "pr_s3": pr_s3, "pr_base": pr_base_all}, f)
    return output_path
