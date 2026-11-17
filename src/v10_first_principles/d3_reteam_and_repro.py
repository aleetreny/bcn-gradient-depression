"""Subagent D (`v10_first_principles`): True `D3` Video Re-Teamed Stream Extraction,
Byte-for-Byte `v8.5` Verification, `v9`/`v10` True Re-Teamed Stack & `Regla 3` Reproducibility Helpers.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from games import GAMES as GAME_INFO, anchor_video_frame
from scripts.exp_v6_super_champion import compute_physical_gates
from scripts.exp_w7a_player_set_net import change_features
from scripts.forensic_v7.prototype_v7_forensic_upgrades import apply_dead_wedge_arbitrator
from src.data.common import AWAY, DEAD, FPS, HOME, possession_spells, runs
from src.data.sportec import MATCH_IDS
from src.decode.forensic_shields_v85 import (
    PostImpactTurnoverClassifier,
    ShieldedBoundaryOffsetAligner,
    ShieldedTouchTypeArbitrator,
)
from src.decode.hybrid_gate_v8 import (
    blend_6fam_heads_v8,
    compute_hybrid_gate_v8,
    compute_inpaint_gate_v8,
    compute_w_clean_v8,
)
from src.decode.spell_reranker import apply_spell_reranker
from src.decode.transductive_calib import (
    DecoupledSynthConfig,
    TransductiveCalibConfig,
    calibrate_match_transductive,
    synthesize_head_decoupled_proba,
)
from src.decode.viterbi_hsmm import decode as decode_v4
from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator
from src.v9_swarm.whistle_and_stoppage import swap_ha_features_df


def sha256_file(path: Path) -> str:
    """Compute exact SHA-256 hex digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_reteam_cache_dir(worktree_root: Path) -> Path:
    """Resolve the feature cache directory for a D3 re-teamed worktree, falling back to
    `local_data/d3_reteam_cache/{count,all}` or `local_data/cache` when `_local/` is absent.
    """
    cdir = worktree_root / "local_data" / "cache"
    if cdir.is_dir():
        return cdir
    if (worktree_root / "features_v1").is_dir():
        return worktree_root
    main_root = Path(__file__).resolve().parents[2]
    wt_name = worktree_root.name.replace("v85sc-", "")
    fallback_dir = main_root / "local_data" / "d3_reteam_cache" / wt_name
    if (fallback_dir / "features_v1").is_dir():
        return fallback_dir
    return main_root / "local_data" / "cache"


def load_worktree_physv2_table(worktree_root: Path, prefix: str, half: np.ndarray) -> pd.DataFrame:
    """Load the 481-column `v1ctl_physics_v2` DataFrame from `worktree_root / local_data / cache`
    (with automatic fallback to `local_data/d3_reteam_cache/` when `_local/` is absent).
    """
    cdir = _resolve_reteam_cache_dir(worktree_root)
    p_f1 = sorted((cdir / "features_v1").glob(f"{prefix}*.parquet"))[0]
    p_c1 = sorted((cdir / "control_v1").glob(f"{prefix}*.parquet"))[0]
    p_pl = sorted((cdir / "plus_v1").glob(f"{prefix}*.parquet"))[0]
    p_p2 = sorted((cdir / "physv2").glob(f"{prefix}*.parquet"))[0]
    df1 = change_features(pd.read_parquet(p_f1), half)
    return pd.concat(
        [df1, pd.read_parquet(p_c1), pd.read_parquet(p_pl), pd.read_parquet(p_p2)],
        axis=1,
    )


def apply_vis_gated_dead_wedge_v85(
    pred: np.ndarray,
    proba: np.ndarray,
    half: np.ndarray,
    X_df: pd.DataFrame,
    p_change: np.ndarray,
    p4d: np.ndarray,
) -> np.ndarray:
    """Exact visibility-gated DeadWedgeArbitrator from `run_unified_v8_forensic_champion.py`."""
    if "ball_visible" in X_df.columns:
        vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=1.0)
    elif "d1_h_vis" in X_df.columns:
        vis = (np.nan_to_num(X_df["d1_h_vis"].to_numpy(np.float64), nan=99.0) < 90.0).astype(np.float64)
    else:
        vis = np.ones(len(pred), dtype=np.float64)

    live_m = ((half == 1) | (half == 2)) & ((pred == HOME) | (pred == AWAY))
    if np.any(live_m) and float(np.mean(vis[live_m])) < 0.80:
        return pred.copy()

    return apply_dead_wedge_arbitrator(pred, proba, half, X_df, p_change, p4d)


def extract_bar_ath_streams_from_worktree(
    worktree_root: Path,
    main_root: Path,
    atd: Any,
    s3_d: float = 0.05,
    s3_ha: float = 0.30,
    s3_c: float = 0.18,
    ps_ha: float = 0.08,
) -> dict[str, Any]:
    """Reconstruct exact `atd_rep` and `atd_raw` 9-family hybrid streams `(p, c, p4d, fams)`
    and `X_atd_rep` / `X_atd_raw` from a worktree or repo-local D3 cache.
    """
    half_atd_dec = np.where(atd.graded, atd.half, 0)
    X_atd_raw = load_worktree_physv2_table(worktree_root, "atd_0f0834", atd.half)
    X_atd_rep = load_worktree_physv2_table(worktree_root, "atd_rep_True_True_None", atd.half)
    gate_atd_raw = compute_physical_gates(X_atd_raw, atd.half)
    gate_atd_rep = compute_physical_gates(X_atd_rep, atd.half)

    out_dir = worktree_root / "experiments" / "outputs"
    main_out_dir = main_root / "experiments" / "outputs"
    wt_tag = worktree_root.name.replace("v85sc-", "")
    cached_d3_pkl = main_out_dir / "v10_first_principles" / "d3_reteamed_bar_ath_streams.pkl"
    if not out_dir.is_dir() and cached_d3_pkl.is_file() and wt_tag in ("control", "count", "all"):
        with open(cached_d3_pkl, "rb") as f:
            d3_b = pickle.load(f)[wt_tag]
        return {
            "hyb": {"atd_rep": d3_b["atd_rep"], "atd_raw": d3_b["atd_raw"]},
            "inp": {"atd_rep": d3_b["atd_rep_inp"], "atd_raw": d3_b["atd_raw"]},
            "X_atd_rep": X_atd_rep,
            "X_atd_raw": X_atd_raw,
            "gate_atd_rep": gate_atd_rep,
            "gate_atd_raw": gate_atd_raw,
        }

    def _resolve_pkl(rel_path: str) -> Path:
        p_wt = out_dir / rel_path
        return p_wt if p_wt.is_file() else (main_out_dir / rel_path)

    with open(_resolve_pkl("exp_c_plus_oof/eval_preds_plus.pkl"), "rb") as f:
        pe_plus = pickle.load(f)
    with open(_resolve_pkl("exp_g_oof/v6_j23_preds.pkl"), "rb") as f:
        _, atd_rep_v6 = pickle.load(f)
    with open(_resolve_pkl("exp_h_oof/bigru_j23_preds.pkl"), "rb") as f:
        _, atd_rep_h = pickle.load(f)
    with open(_resolve_pkl("exp_w3_oof/b4d_j23_preds.pkl"), "rb") as f:
        b4d_j23 = pickle.load(f)
    with open(_resolve_pkl("exp_w4_oof/eval_preds_physv2.pkl"), "rb") as f:
        pe_p2 = pickle.load(f)
    with open(_resolve_pkl("exp_w5_tree_oof/eval_preds_cb_xgb.pkl"), "rb") as f:
        pe_w5 = pickle.load(f)
    with open(_resolve_pkl("exp_w5b_oof/seq_physv2_j23_preds.pkl"), "rb") as f:
        pe_w5b = pickle.load(f)
    with open(_resolve_pkl("exp_w6_oof/atd_raw_v6_bigru.pkl"), "rb") as f:
        atd_raw_extra = pickle.load(f)
    with open(_resolve_pkl("exp_w7a_oof/pset_j23_preds.pkl"), "rb") as f:
        pset_j23 = pickle.load(f)
    with open(_resolve_pkl("exp_w6_oof/w6_results_summary.pkl"), "rb") as f:
        w6_sum = pickle.load(f)
    with open(_resolve_pkl("exp_w8a_oof/w8a_j23_preds.pkl"), "rb") as f:
        j23_inp = pickle.load(f)

    j23_inp["atd_raw"]["w_inp"] = compute_inpaint_gate_v8(
        X_atd_raw, atd.half, gate_atd_raw[0], gate_atd_raw[1]
    )
    j23_inp["atd_rep"]["w_inp"] = compute_inpaint_gate_v8(
        X_atd_rep, atd.half, gate_atd_rep[0], gate_atd_rep[1]
    )

    best_calib_cfg: TransductiveCalibConfig = w6_sum["best_calib_cfg"]
    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58,
        w_tree_ha=0.50,
        alpha_tv=0.08,
    )

    def _build_mode(inp_scale: float = 0.0) -> dict[str, tuple]:
        res = {}
        for vkey, X_df, gate_t, t2_p, g2_p, t2_c, g2_c, gv6_p, gv6_c in [
            (
                "atd_raw",
                X_atd_raw,
                gate_atd_raw,
                pe_plus["atd_raw"]["t_sym"],
                atd_raw_extra["b_sym"],
                pe_plus["atd_raw"]["tc_sym"],
                atd_raw_extra["bc_sym"],
                atd_raw_extra["g_v6"],
                atd_raw_extra["tr_v6"],
            ),
            (
                "atd_rep",
                X_atd_rep,
                gate_atd_rep,
                atd_rep_h["t_sym"],
                atd_rep_h["b_sym"],
                atd_rep_h["tc_sym"],
                atd_rep_h["bc_sym"],
                atd_rep_v6["g_v6"],
                atd_rep_v6["tr_v6"],
            ),
        ]:
            dpl_a = pe_plus[vkey]
            dp2_a = pe_p2[vkey]
            dw5_a = pe_w5[vkey]
            dw5b_a = pe_w5b[vkey]
            dpset_a = pset_j23[vkey]
            dinp_a = j23_inp[vkey]
            a_a = compute_hybrid_gate_v8(X_df, atd.half, gate_t[0])
            wc_a = compute_w_clean_v8(X_df, atd.half, gate_t[0])
            pa, ca = synthesize_head_decoupled_proba(
                dpl_a["g_plus_sym"],
                dp2_a["g_physv2_sym"],
                dw5_a["cb_physv2_sym"],
                dw5_a["xgb_physv2_sym"],
                t2_p,
                g2_p,
                dpl_a["tr_plus_sym"],
                dp2_a["tr_physv2_sym"],
                t2_c,
                g2_c,
                atd.half,
                gv6_p,
                gv6_c,
                w_ghost=gate_t[0],
                w_gap=gate_t[1],
                cfg=scfg_main,
                tcn_physv2_proba=dw5b_a["tcn_physv2_sym"],
                bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
                tcn_physv2_change=dw5b_a["tcnc_physv2_sym"],
                bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
                alpha_p2=a_a,
            )
            pa, ca = blend_6fam_heads_v8(
                pa,
                ca,
                t2_p,
                g2_p,
                dw5b_a["bigru_physv2_sym"],
                dpset_a["pset_proba"],
                t2_c,
                g2_c,
                dw5b_a["bigruc_physv2_sym"],
                dpset_a["pset_change"],
                a_a,
                wc_a,
                s3_d,
                s3_ha,
                s3_c,
                ps_ha,
                adaptive_regime=True,
                p_inp=dinp_a["p_inp"] if inp_scale > 0 else None,
                c_inp=dinp_a["c_inp"] if inp_scale > 0 else None,
                w_inp=(inp_scale * dinp_a["w_inp"]) if inp_scale > 0 else None,
            )
            pa = calibrate_match_transductive(pa, half_atd_dec, X_df, best_calib_cfg)
            fa = [
                dpl_a["g_plus_sym"],
                dp2_a["g_physv2_sym"],
                dw5_a["cb_physv2_sym"],
                dw5_a["xgb_physv2_sym"],
                t2_p,
                g2_p,
                dw5b_a["tcn_physv2_sym"],
                dw5b_a["bigru_physv2_sym"],
                dpset_a["pset_proba"],
            ]
            res[vkey] = (pa, ca, b4d_j23[vkey], fa)
        return res

    hyb = _build_mode(inp_scale=0.0)
    inp = _build_mode(inp_scale=0.35)
    return {
        "hyb": hyb,
        "inp": inp,
        "X_atd_rep": X_atd_rep,
        "X_atd_raw": X_atd_raw,
        "gate_atd_rep": gate_atd_rep,
        "gate_atd_raw": gate_atd_raw,
    }


def load_v85_fitted_shield_models(main_root: Path) -> dict[str, Any]:
    """Load the fitted 7-match v8.5 shield models (`ShieldedTouchTypeArbitrator`,
    `ShieldedBoundaryOffsetAligner`, `PostImpactTurnoverClassifier`) and decoder parameters.
    """
    out_dir = main_root / "experiments" / "outputs"
    with open(out_dir / "exp_b_sym_oof/dec_v5_sym.pkl", "rb") as f:
        _, loo_dec_hyb, _, dec_v5_hyb = pickle.load(f)
    with open(out_dir / "exp_w6_oof/w6_results_summary.pkl", "rb") as f:
        w6_sum = pickle.load(f)
    with open(out_dir / "forensic_v7/v85_cands_cache.pkl", "rb") as f:
        v8_cands_bundle = pickle.load(f)

    matches = sorted(MATCH_IDS)
    models: dict[str, dict[str, Any]] = {}
    for s_tag in ("hyb", "inp"):
        _, m1_d, m2_d, bdy_d, pit_d = v8_cands_bundle[s_tag]["v3"]
        models[s_tag] = {
            "arb": ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, matches),
            "aln": ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_d),
            "pit": PostImpactTurnoverClassifier().fit_from_candidates(pit_d, matches),
        }
    return {
        "dec_v5_hyb": dec_v5_hyb,
        "loo_dec_hyb": loo_dec_hyb,
        "full_rerank_cfg": w6_sum["full_rerank_cfg"],
        "lomo_rerank_cfgs": w6_sum["lomo_rerank_cfgs"],
        "xdx_cfg_st1": XDXConfig(
            enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True
        ),
        "models": models,
    }


def decode_v85_bar_ath(
    stream_tuple: tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]],
    X_atd_rep: pd.DataFrame,
    atd: Any,
    v85_bundle: dict[str, Any],
    s_tag: str = "hyb",
    use_pit_domino: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the exact `v8.5` decoding pipeline on a `BAR-ATH` `(p, c, p4d, fams)` stream tuple."""
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    p_raw, c_raw, p4d_raw, fams_raw = stream_tuple
    p = p_raw.astype(np.float64)
    c = c_raw.astype(np.float64)
    p4d = p4d_raw.astype(np.float64)
    f64 = [f.astype(np.float64) for f in fams_raw]

    pr_xdx_a = apply_xdx_arbitrator(
        decode_v4(p_raw, half_atd_dec, v85_bundle["dec_v5_hyb"], p_change=c_raw),
        p_raw,
        half_atd_dec,
        X_atd_rep,
        p_change=c_raw,
        p_trans_4d=p4d_raw,
        cfg=v85_bundle["xdx_cfg_st1"],
    )
    pr_base_a = apply_spell_reranker(
        pr_xdx_a,
        p_raw,
        half_atd_dec,
        X_atd_rep,
        c_raw,
        p4d_raw,
        fams_raw,
        cfg=v85_bundle["full_rerank_cfg"],
    )
    pr_a = pr_base_a.copy()

    mods = v85_bundle["models"][s_tag]
    pr_a = mods["arb"].apply(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)
    pr_a = apply_vis_gated_dead_wedge_v85(pr_a, p, half_atd_dec, X_atd_rep, c, p4d)
    pr_a = mods["aln"].apply(
        pr_a, p, half_atd_dec, X_atd_rep, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base_a
    )
    if use_pit_domino:
        pr_a = mods["pit"].apply(pr_a, p, half_atd_dec, X_atd_rep, c, p4d, fams=f64)

    pr_a[second_start:kick2] = DEAD
    pr_a[~atd.graded] = DEAD
    return pr_a, pr_base_a


def apply_d3_post_kick_sliver_bridge(
    pred: np.ndarray,
    proba: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    sp_tags: list[dict],
    phase_score: np.ndarray | None = None,
    max_sliver_len: int = 25,
    max_wedge_len: int = 90,
    min_after_len: int = 45,
) -> np.ndarray:
    """100% H<->A equivariant (`0.000e+00`) extension to `SmartTaggingArbitrator` Rule 1:
    When a `DEAD -> Live` restart matches a `SET PIECES` tag (`team = t_sp`), and the decoder outputs
    `DEAD (>= 25f) -> t_sp (micro-sliver <= 25f) -> X (wrong-team wedge <= 90f) -> t_sp (>= 45f)`
    where directional `phase_score` strongly corroborates `t_sp` (`phase_score[a:b, t_sp] >= 25.0`
    and `>= 2.0 * phase_score[a:b, X]`), bridge the wrong-team wedge `X` into `t_sp`.
    """
    out = np.asarray(pred, dtype=np.int32).copy()
    half = np.asarray(half, dtype=np.int32)
    n = len(out)
    chg = np.flatnonzero((out[1:] != out[:-1]) | (half[1:] != half[:-1])) + 1
    s_arr = np.concatenate([[0], chg])
    e_arr = np.concatenate([chg, [n]])
    v_arr = out[s_arr]

    for i in range(1, len(v_arr) - 2):
        if not (half[s_arr[i - 1]] == half[s_arr[i]] == half[s_arr[i + 1]] == half[s_arr[i + 2]]):
            continue
        if int(v_arr[i - 1]) != int(DEAD) or (e_arr[i - 1] - s_arr[i - 1]) < 25:
            continue
        sliver_tm = int(v_arr[i])
        wedge_tm = int(v_arr[i + 1])
        after_tm = int(v_arr[i + 2])
        if sliver_tm not in (int(HOME), int(AWAY)) or wedge_tm != (1 - sliver_tm) or after_tm != sliver_tm:
            continue
        sliver_len = int(e_arr[i] - s_arr[i])
        wedge_a, wedge_b = int(s_arr[i + 1]), int(e_arr[i + 1])
        wedge_len = wedge_b - wedge_a
        after_len = int(e_arr[i + 2] - s_arr[i + 2])
        if sliver_len > max_sliver_len or wedge_len > max_wedge_len or after_len < min_after_len:
            continue

        restart_f = int(s_arr[i])
        cands = [
            sp
            for sp in sp_tags
            if -80 <= ((sp["start"] + 5 * FPS) - restart_f) <= 40
            and sp.get("team", -1) in (int(HOME), int(AWAY))
        ]
        if not cands:
            continue
        best_sp = min(cands, key=lambda sp: abs((sp["start"] + 5 * FPS) - restart_f))
        t_sp = int(best_sp["team"])
        if t_sp != sliver_tm:
            continue

        if phase_score is not None:
            sc_sp = float(phase_score[wedge_a:wedge_b, t_sp].sum())
            sc_wx = float(phase_score[wedge_a:wedge_b, wedge_tm].sum())
            if sc_sp >= 25.0 and sc_sp >= 2.0 * max(sc_wx, 1e-6):
                out[wedge_a:wedge_b] = t_sp

    return out


def verify_d3_sliver_bridge_equivariance(
    pred: np.ndarray,
    proba: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    sp_tags: list[dict],
    phase_score: np.ndarray,
) -> float:
    """Verify exact 0.000e+00 H<->A equivariance of `apply_d3_post_kick_sliver_bridge`."""
    out_orig = apply_d3_post_kick_sliver_bridge(pred, proba, X_df, half, sp_tags, phase_score)

    pred_swap = np.where(pred == HOME, AWAY, np.where(pred == AWAY, HOME, pred))
    proba_swap = proba[:, [1, 0, 2]].copy()
    X_swap = swap_ha_features_df(X_df)
    sp_swap = []
    for sp in sp_tags:
        sc = dict(sp)
        if sc.get("team") == HOME:
            sc["team"] = int(AWAY)
        elif sc.get("team") == AWAY:
            sc["team"] = int(HOME)
        sp_swap.append(sc)
    phase_swap = phase_score[:, [1, 0]].copy()

    out_swap = apply_d3_post_kick_sliver_bridge(pred_swap, proba_swap, X_swap, half, sp_swap, phase_swap)
    out_swap_back = np.where(out_swap == HOME, AWAY, np.where(out_swap == AWAY, HOME, out_swap))
    return float(np.mean(out_orig != out_swap_back))


def compute_d3_reteam_anatomy(atd: Any, main_root: Path) -> dict[str, Any]:
    """Quantify how many `12v10+` swap frames and stretches exist on `BAR-ATH` before and after
    each `D3` video jersey re-teaming variant (`--count`, `--all`, `--pairs`).
    """
    vis = ~np.isnan(atd.P[:, :, 0])
    nh_raw = (vis & (atd.slot_team == HOME)).sum(axis=1)
    na_raw = (vis & (atd.slot_team == AWAY)).sum(axis=1)
    over_raw = (nh_raw > 11) | (na_raw > 11)
    rs, re, rv = runs(over_raw.astype(int))
    raw_stretches = [(int(s), int(e), int(e - s)) for s, e, v in zip(rs, re, rv) if v == 1]

    out: dict[str, Any] = {
        "raw": {
            "over_11_frames": int(over_raw.sum()),
            "over_11_stretches": int(len(raw_stretches)),
            "over_11_stretches_ge_1s": int(sum(1 for _, _, d in raw_stretches if d >= 25)),
            "max_nh": int(nh_raw.max()),
            "max_na": int(na_raw.max()),
        }
    }

    vdir = main_root / "local_data" / "video"
    for key, fname in [
        ("count", "players_reteam.npz"),
        ("all", "players_reteam_all.npz"),
        ("pairs", "players_reteam_pairs.npz"),
    ]:
        p_npz = vdir / fname
        if not p_npz.is_file():
            continue
        segs = np.load(p_npz)["segments"]
        nh = nh_raw.copy()
        na = na_raw.copy()
        player_frames_swapped = 0
        frame_union = np.zeros(atd.n_frames, dtype=bool)
        for col, a, b in segs:
            col, a, b = int(col), int(a), int(b)
            v_seg = vis[a:b, col]
            n_v = int(v_seg.sum())
            player_frames_swapped += n_v
            frame_union[a:b] |= v_seg
            if atd.slot_team[col] == HOME:
                nh[a:b] -= v_seg
                na[a:b] += v_seg
            else:
                na[a:b] -= v_seg
                nh[a:b] += v_seg
        over = (nh > 11) | (na > 11)
        rs_k, re_k, rv_k = runs(over.astype(int))
        st_k = [(int(s), int(e), int(e - s)) for s, e, v in zip(rs_k, re_k, rv_k) if v == 1]
        # Count how many raw >11 stretches were completely or partially fixed
        completely_fixed = sum(1 for s, e, _ in raw_stretches if not np.any(over[s:e]))
        partially_or_fully_fixed = sum(
            1 for s, e, _ in raw_stretches if int(over[s:e].sum()) < int(over_raw[s:e].sum())
        )
        out[key] = {
            "n_segments": int(len(segs)),
            "player_frames_swapped": int(player_frames_swapped),
            "frames_touched": int(frame_union.sum()),
            "pct_frames_touched": round(100.0 * float(frame_union.mean()), 2),
            "over_11_frames_before": int(over_raw.sum()),
            "over_11_frames_after": int(over.sum()),
            "over_11_frames_fixed": int(over_raw.sum() - over.sum()),
            "pct_over_11_frames_fixed": round(
                100.0 * float(over_raw.sum() - over.sum()) / max(int(over_raw.sum()), 1), 2
            ),
            "raw_stretches_total": int(len(raw_stretches)),
            "raw_stretches_completely_fixed": int(completely_fixed),
            "raw_stretches_improved": int(partially_or_fully_fixed),
            "remaining_over_11_stretches": int(len(st_k)),
            "max_nh_after": int(nh.max()),
            "max_na_after": int(na.max()),
        }
    return out


def compute_episode_physical_coherence(
    pred: np.ndarray,
    X_df: pd.DataFrame,
    atd: Any,
    reteam_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute episode-level physical coherence metrics on BAR-ATH graded frames:
    - Number of sub-34-frame (`< 1.36s`) illegal micro-spells
    - Clear-control physical proximity agreement (`d1_min < 1.5m` and `|d1_h - d1_a| > 3.0m`)
    - Possession rate & short-spell fraction inside vs outside `D3` re-teamed windows.
    """
    pr_g = np.where(atd.graded, pred, -1).astype(np.int8)
    s_p, e_p, t_p = possession_spells(pr_g, atd.half)
    d_frames = e_p - s_p
    sub_34_spells = [
        {"start": int(s_p[i]), "end": int(e_p[i]), "frames": int(d_frames[i]), "team": int(t_p[i])}
        for i in range(len(d_frames))
        if d_frames[i] < 34
    ]

    d1_h = np.nan_to_num(X_df["d1_h_impulse_gated"].to_numpy(np.float64), nan=99.0)
    d1_a = np.nan_to_num(X_df["d1_a_impulse_gated"].to_numpy(np.float64), nan=99.0)
    vis = np.nan_to_num(X_df["ball_visible"].to_numpy(np.float64), nan=0.0) > 0.5
    live = atd.graded & ((pred == HOME) | (pred == AWAY)) & vis

    clear_h = live & (d1_h < 1.5) & ((d1_a - d1_h) > 3.0)
    clear_a = live & (d1_a < 1.5) & ((d1_h - d1_a) > 3.0)
    clear_mask = clear_h | clear_a
    phys_expected = np.where(clear_h, HOME, AWAY)
    clear_agree = (
        100.0 * float(np.mean(pred[clear_mask] == phys_expected[clear_mask]))
        if np.any(clear_mask)
        else float("nan")
    )

    reteam_clear_agree = float("nan")
    if reteam_mask is not None:
        rt_clear = clear_mask & reteam_mask
        if np.any(rt_clear):
            reteam_clear_agree = 100.0 * float(np.mean(pred[rt_clear] == phys_expected[rt_clear]))

    return {
        "n_spells_total": int(len(d_frames)),
        "n_sub_34f_micro_spells": int(len(sub_34_spells)),
        "sub_34f_micro_spells": sub_34_spells,
        "clear_proximity_frames": int(clear_mask.sum()),
        "clear_proximity_team_agree_pct": round(clear_agree, 2),
        "reteam_window_clear_proximity_agree_pct": round(reteam_clear_agree, 2),
    }
