"""Rebuild v9 duel or MBR candidate caches from local intermediate inputs.

These are match-derived local caches, not submission files or experiment results.
"""
from pathlib import Path
import pickle
import re

import numpy as np
import pandas as pd
import evaluation.possession_metrics as pm
from src.data.common import AWAY, DEAD, HOME
from src.decode.forensic_shields_v85 import to_official, _f1_from_spells, rle_runs_half
from src.models.transition import change_features

def extract_duel_cands_v9(pr, p, c, p4d, f64, hl, X_df, gt=None):
    T = len(pr)
    d1h = np.nan_to_num(X_df['dist_ball_H'].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df['dist_ball_A'].to_numpy(np.float64), nan=50.0)
    vis = np.nan_to_num(X_df['ball_visible'].to_numpy(np.float64), nan=1.0)
    chord = np.nan_to_num(X_df['chord_in_flight'].to_numpy(np.float64), nan=0.0)
    chord_to = np.nan_to_num(X_df['chord_turnover_prob'].to_numpy(np.float64), nan=0.0)
    imp4 = np.nan_to_num(X_df['impulse_mag_d4'].to_numpy(np.float64), nan=0.0)
    cos2 = np.nan_to_num(X_df['impulse_cos_d2'].to_numpy(np.float64), nan=1.0)
    cos4 = np.nan_to_num(X_df['impulse_cos_d4'].to_numpy(np.float64), nan=1.0)
    bang = np.nan_to_num(X_df['ball_angle_change'].to_numpy(np.float64), nan=0.0)

    tree_P = (f64[0] + f64[1] + f64[2] + f64[3]) / 4.0
    seq244_P = (f64[4] + f64[5]) / 2.0
    seq481_P = (f64[6] + f64[7]) / 2.0 if len(f64) >= 9 else seq244_P
    seq_P = (seq244_P + seq481_P) / 2.0
    pset_P = f64[-1]

    if gt is not None:
        gt_off = to_official(gt, hl, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pr, hl, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    r = rle_runs_half(pr, hl)
    cands = []
    for sa, sb, x_lbl, h_cur in r:
        if x_lbl not in (HOME, AWAY) or h_cur <= 0: continue
        if sb - sa < 80: continue
        y_lbl = 1 - x_lbl
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0

        sig_y = np.maximum(p[sa:sb, y_lbl], np.maximum(seq_P[sa:sb, y_lbl], pset_P[sa:sb, y_lbl]))
        bdy_in = np.maximum(0.85 * c[sa:sb], p4d[sa:sb, col_to_Y])
        bdy_out = np.maximum(0.85 * c[sa:sb], p4d[sa:sb, col_to_X])

        cand_windows = []
        for rel_a in range(18, (sb - sa) - 18 - 12):
            if bdy_in[rel_a] < 0.14: continue
            if rel_a > 0 and bdy_in[rel_a] < bdy_in[rel_a - 1]: continue
            if rel_a + 1 < len(bdy_in) and bdy_in[rel_a] < bdy_in[rel_a + 1]: continue
            b_start = rel_a + 12
            b_stop = min((sb - sa) - 18, rel_a + 80 + 1)
            if b_stop <= b_start: continue
            rel_b = b_start + int(np.argmax(bdy_out[b_start:b_stop]))
            if bdy_out[rel_b] < 0.14: continue
            if float(np.max(sig_y[rel_a:rel_b])) < 0.22: continue
            ca, cb = int(sa + rel_a), int(sa + rel_b)
            if cand_windows and ca < cand_windows[-1][1] + 10: continue
            cand_windows.append((ca, cb))

        for ca, cb in cand_windows:
            L = cb - ca
            sl = slice(ca, cb)
            ca_int = ca + max(1, min(3, L // 5))
            cb_int = cb - max(1, min(3, L // 5))
            sl_int = slice(ca_int, max(ca_int + 1, cb_int))
            wa0, wa1 = max(0, ca - 4), min(T, ca + 5)
            wb0, wb1 = max(0, cb - 4), min(T, cb + 5)

            d1_Y = d1h if y_lbl == HOME else d1a
            d1_X = d1a if y_lbl == HOME else d1h

            b4d_in_Y = float(np.max(p4d[wa0:wa1, col_to_Y]))
            b4d_out_X = float(np.max(p4d[wb0:wb1, col_to_X]))
            b4d_in_inv = float(np.max(p4d[wa0:wa1, col_to_X]))
            b4d_out_inv = float(np.max(p4d[wb0:wb1, col_to_Y]))
            b4d_geom = float(np.sqrt(max(0.0, b4d_in_Y * b4d_out_X)))
            b4d_dir_asym = float((b4d_in_Y - b4d_in_inv) + (b4d_out_X - b4d_out_inv))
            pc_in = float(np.max(c[wa0:wa1]))
            pc_out = float(np.max(c[wb0:wb1]))
            pc_geom = float(np.sqrt(max(0.0, pc_in * pc_out)))

            mean_pY = float(np.mean(p[sl, y_lbl]))
            max_pY = float(np.max(p[sl, y_lbl]))
            int_pY = float(np.mean(p[sl_int, y_lbl]))
            mean_pX = float(np.mean(p[sl, x_lbl]))
            mean_pD = float(np.mean(p[sl, DEAD]))
            max_pD = float(np.max(p[sl, DEAD]))

            tree_pY = float(np.mean(tree_P[sl, y_lbl]))
            tree_max_pY = float(np.max(tree_P[sl, y_lbl]))
            seq244_pY = float(np.mean(seq244_P[sl, y_lbl]))
            seq481_pY = float(np.mean(seq481_P[sl, y_lbl]))
            seq_pY = float(np.mean(seq_P[sl, y_lbl]))
            seq_max_pY = float(np.max(seq_P[sl, y_lbl]))
            pset_pY = float(np.mean(pset_P[sl, y_lbl]))
            pset_max_pY = float(np.max(pset_P[sl, y_lbl]))
            seq_minus_tree = float(max(seq_pY, pset_pY) - tree_pY)
            fam_vote_Y = float(np.mean([1.0 if np.mean(fp[sl, y_lbl] - fp[sl, x_lbl]) > 0 else 0.0 for fp in f64]))
            fam_peak_vote_Y = float(np.mean([1.0 if np.max(fp[sl, y_lbl]) >= 0.45 else 0.0 for fp in f64]))

            min_d1_Y_int = float(np.min(d1_Y[sl_int]))
            min_d1_X_int = float(np.min(d1_X[sl_int]))
            d1_adv_int = float(np.mean(np.clip(d1_X[sl_int] - d1_Y[sl_int], -15.0, 15.0)))
            y_win_frames = float(np.sum((d1_Y[sl] < 1.65) & (d1_Y[sl] <= d1_X[sl] + 0.25)))
            duel_2m_frames = float(np.sum((d1_Y[sl] < 2.0) & (d1_X[sl] < 2.2)))

            # 3D Ballistic Flight / Aerial Arc & Deflection Physics
            chord_pre = float(np.max(chord[max(0, ca - 15):min(T, ca + 3)]))
            chord_int = float(np.mean(chord[sl_int]))
            chord_turn = float(np.max(chord_to[wa0:wb1]))
            max_imp4_win = float(np.max(imp4[wa0:wb1]))
            min_cos4_win = float(np.min(cos4[wa0:wb1]))
            min_cos2_win = float(np.min(cos2[wa0:wb1]))
            max_ang_win = float(np.max(bang[wa0:wb1])) / 180.0
            frac_vis = float(np.mean(vis[sl]))
            undeflected_aerial = float(chord_int * (1.0 if min_d1_Y_int > 1.35 and min_cos4_win > 0.35 else 0.0))

            dy = 0.0
            if gt is not None:
                shift = 1 if h_cur == 2 else 0
                sa_s, sb_s = sa + shift, sb + shift
                ca_s, cb_s = ca + shift, cb + shift
                mod_spells = []
                for spa, spb, splbl in base_spells:
                    if spa == sa_s and spb == sb_s and splbl == x_lbl:
                        mod_spells.append((sa_s, ca_s, x_lbl))
                        mod_spells.append((ca_s, cb_s, y_lbl))
                        mod_spells.append((cb_s, sb_s, x_lbl))
                    else:
                        mod_spells.append((spa, spb, splbl))
                dy = float(_f1_from_spells(gt_spells, mod_spells) - base_f1)

            cands.append({
                'ca': ca, 'cb': cb, 'L': L, 'y_lbl': y_lbl, 'x_lbl': x_lbl, 'dy': dy,
                'b4d_in_Y': b4d_in_Y, 'b4d_out_X': b4d_out_X, 'b4d_geom': b4d_geom, 'b4d_dir_asym': b4d_dir_asym,
                'pc_in': pc_in, 'pc_out': pc_out, 'pc_geom': pc_geom,
                'mean_pY': mean_pY, 'max_pY': max_pY, 'int_pY': int_pY, 'mean_pX': mean_pX, 'mean_pD': mean_pD, 'max_pD': max_pD,
                'tree_pY': tree_pY, 'tree_max_pY': tree_max_pY, 'seq244_pY': seq244_pY, 'seq481_pY': seq481_pY,
                'seq_pY': seq_pY, 'seq_max_pY': seq_max_pY, 'pset_pY': pset_pY, 'pset_max_pY': pset_max_pY,
                'seq_minus_tree': seq_minus_tree, 'fam_vote_Y': fam_vote_Y, 'fam_peak_vote_Y': fam_peak_vote_Y,
                'min_d1_Y_int': min_d1_Y_int, 'min_d1_X_int': min_d1_X_int, 'd1_adv_int': d1_adv_int,
                'y_win_frames': y_win_frames, 'duel_2m_frames': duel_2m_frames,
                'chord_pre': chord_pre, 'chord_int': chord_int, 'chord_turn': chord_turn,
                'max_imp4_win': max_imp4_win, 'min_cos4_win': min_cos4_win, 'min_cos2_win': min_cos2_win,
                'max_ang_win': max_ang_win, 'frac_vis': frac_vis, 'undeflected_aerial': undeflected_aerial,
            })
    return cands

def extract_mbr_xyx_cands(pr, p, c, p4d, f64, hl, X_df, gt=None):
    T = len(pr)
    d1h = np.nan_to_num(X_df['dist_ball_H'].to_numpy(np.float64), nan=50.0)
    d1a = np.nan_to_num(X_df['dist_ball_A'].to_numpy(np.float64), nan=50.0)
    vis = np.nan_to_num(X_df['ball_visible'].to_numpy(np.float64), nan=1.0)
    chord = np.nan_to_num(X_df['chord_in_flight'].to_numpy(np.float64), nan=0.0)
    imp4 = np.nan_to_num(X_df['impulse_mag_d4'].to_numpy(np.float64), nan=0.0)
    cos4 = np.nan_to_num(X_df['impulse_cos_d4'].to_numpy(np.float64), nan=1.0)
    ghost = np.nan_to_num(X_df['ghost_incoherence'].to_numpy(np.float64), nan=0.0) if 'ghost_incoherence' in X_df.columns else np.zeros(T)

    tree_P = (f64[0] + f64[1] + f64[2] + f64[3]) / 4.0
    seq244_P = (f64[4] + f64[5]) / 2.0
    seq481_P = (f64[6] + f64[7]) / 2.0 if len(f64) >= 9 else seq244_P
    seq_P = (seq244_P + seq481_P) / 2.0
    pset_P = f64[-1]

    if gt is not None:
        gt_off = to_official(gt, hl, True)
        valid_r = pm.valid_runs(gt_off)
        gt_spells = pm.possession_spells(gt_off, valid_r)
        p_off = to_official(pr, hl, True)
        p_off[gt_off == pm.X] = pm.X
        base_spells = pm.possession_spells(p_off, valid_r)
        base_f1 = _f1_from_spells(gt_spells, base_spells)
    else:
        gt_spells, base_spells, base_f1 = [], [], 0.0

    # K=7 scales for Expected-spellF1 / Expected-IoU {0.3, 0.4, 0.5, 0.6, 0.7} scoring
    scales = np.array([0.65, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50], dtype=np.float64)
    iou_ths = np.array([0.3, 0.4, 0.5, 0.6, 0.7], dtype=np.float64)

    runs = rle_runs_half(pr, hl)
    cands = []
    for i in range(1, len(runs) - 1):
        sa, sb, y_lbl, h_cur = runs[i]
        pa, pb, l_lbl, h_l = runs[i-1]
        na, nb, r_lbl, h_r = runs[i+1]
        if h_cur <= 0 or h_l != h_cur or h_r != h_cur: continue
        L = sb - sa
        if L > 95: continue
        if not (y_lbl in (HOME, AWAY) and l_lbl == r_lbl and l_lbl in (HOME, AWAY) and l_lbl != y_lbl):
            continue
        x_lbl = l_lbl
        col_to_Y = 1 if y_lbl == HOME else 0
        col_to_X = 1 if x_lbl == HOME else 0

        sl = slice(sa, sb)
        wa0, wa1 = max(0, sa - 5), min(T, sa + 6)
        wb0, wb1 = max(0, sb - 5), min(T, sb + 6)

        b4d_in_Y = float(np.max(p4d[wa0:wa1, col_to_Y]))
        b4d_out_X = float(np.max(p4d[wb0:wb1, col_to_X]))
        b4d_geom = float(np.sqrt(max(0.0, b4d_in_Y * b4d_out_X)))
        pc_in = float(np.max(c[wa0:wa1]))
        pc_out = float(np.max(c[wb0:wb1]))
        pc_geom = float(np.sqrt(max(0.0, pc_in * pc_out)))

        mean_pY = float(np.mean(p[sl, y_lbl]))
        max_pY = float(np.max(p[sl, y_lbl]))
        mean_pX = float(np.mean(p[sl, x_lbl]))
        mean_pD = float(np.mean(p[sl, DEAD]))

        tree_pY = float(np.mean(tree_P[sl, y_lbl]))
        seq_pY = float(np.mean(seq_P[sl, y_lbl]))
        pset_pY = float(np.mean(pset_P[sl, y_lbl]))
        fam_vote_Y = float(np.mean([1.0 if np.mean(fp[sl, y_lbl] - fp[sl, x_lbl]) > 0 else 0.0 for fp in f64]))

        # Compute K=7 x 9-family Expected Local spellF1 gain of merging [pa, nb) vs keeping [pa, sa), [sa, sb), [sb, nb)
        # Under each family m and transition cost scale gamma_k:
        # Does family m at scale gamma_k prefer a single spell X on [pa, nb) or splitting into 3 spells at [sa, sb)?
        L_left = sa - pa
        L_right = nb - sb
        L_merged = nb - pa
        # IoU of unsplit [pa, sa) and [sb, nb) against a true merged spell [pa, nb):
        iou_left_vs_merged = L_left / float(L_merged)
        iou_right_vs_merged = L_right / float(L_merged)
        # In official spellF1 Hungarian matching, if GT is a single spell [pa, nb) of X,
        # predicting 3 spells ([pa, sa) X, [sa, sb) Y, [sb, nb) X) creates:
        # - 1 FP for Y,
        # - 1 FP for the shorter flank of X, and
        # - the longer flank of X only matches if max(L_left, L_right) / L_merged >= theta!
        max_flank_ratio = max(iou_left_vs_merged, iou_right_vs_merged)
        f1_if_gt_merged_and_we_keep = float(np.mean([ (2.0 * (1.0 if max_flank_ratio >= th else 0.0)) / 4.0 for th in iou_ths ]))
        # Whereas if we merge, and GT is merged, local F1 is 1.0!
        # Conversely, if GT is 3 spells (X, Y, X) and we merge into 1 spell X, local F1 is:
        f1_if_gt_split_and_we_merge = float(np.mean([ (2.0 * (1.0 if max_flank_ratio >= th else 0.0)) / 4.0 for th in iou_ths ]))

        # Posterior probability P(split is real | fams, K=7 scales, b4d_geom, pc_geom)
        # Log-odds of Y vs X inside [sa, sb) across 9 families vs transition cost 2 * gamma_k * (-log bdy)
        bdy_comb = max(1e-3, 0.55 * b4d_geom + 0.45 * pc_geom)
        trans_cost_base = -np.log(np.clip(bdy_comb, 0.02, 0.98)) * 18.0
        split_votes = []
        for fp in f64:
            ll_yx = float(np.sum(np.log(np.clip(fp[sl, y_lbl], 1e-4, 1.0)) - np.log(np.clip(fp[sl, x_lbl], 1e-4, 1.0))))
            for sc in scales:
                prob_split = 1.0 / (1.0 + np.exp(-np.clip((ll_yx - sc * trans_cost_base) / 6.0, -20.0, 20.0)))
                split_votes.append(prob_split)
        p_split_lattice = float(np.mean(split_votes))
        # Expected local spellF1 gain of MERGING vs KEEPING:
        exp_f1_merge = (1.0 - p_split_lattice) * 1.0 + p_split_lattice * f1_if_gt_split_and_we_merge
        exp_f1_keep = (1.0 - p_split_lattice) * f1_if_gt_merged_and_we_keep + p_split_lattice * 1.0
        exp_mbr_merge_gain = float(exp_f1_merge - exp_f1_keep)

        d1_Y = d1h[sl] if y_lbl == HOME else d1a[sl]
        d1_X = d1a[sl] if y_lbl == HOME else d1h[sl]
        min_d1_Y = float(np.min(d1_Y))
        min_d1_X = float(np.min(d1_X))
        eff_contact_Y = float(np.sum((d1_Y < 1.65) & (d1_Y <= d1_X + 0.25)))
        max_imp4 = float(np.max(imp4[sl]))
        min_cos4 = float(np.min(cos4[sl]))
        frac_vis = float(np.mean(vis[sl]))
        chord_mean = float(np.mean(chord[sl]))
        ghost_mean = float(np.mean(ghost[sl]))

        dy = 0.0
        if gt is not None:
            pr_m = pr.copy()
            pr_m[sa:sb] = x_lbl
            m_off = to_official(pr_m, hl, True)
            m_off[gt_off == pm.X] = pm.X
            dy = float(_f1_from_spells(gt_spells, pm.possession_spells(m_off, valid_r)) - base_f1)

        cands.append({
            'sa': sa, 'sb': sb, 'L': L, 'L_left': L_left, 'L_right': L_right, 'y_lbl': y_lbl, 'x_lbl': x_lbl, 'dy': dy,
            'b4d_in_Y': b4d_in_Y, 'b4d_out_X': b4d_out_X, 'b4d_geom': b4d_geom, 'pc_geom': pc_geom,
            'mean_pY': mean_pY, 'max_pY': max_pY, 'mean_pX': mean_pX, 'mean_pD': mean_pD,
            'tree_pY': tree_pY, 'seq_pY': seq_pY, 'pset_pY': pset_pY, 'fam_vote_Y': fam_vote_Y,
            'p_split_lattice': p_split_lattice, 'exp_mbr_merge_gain': exp_mbr_merge_gain,
            'min_d1_Y': min_d1_Y, 'min_d1_X': min_d1_X, 'eff_contact_Y': eff_contact_Y,
            'max_imp4': max_imp4, 'min_cos4': min_cos4, 'frac_vis': frac_vis,
            'chord_mean': chord_mean, 'ghost_mean': ghost_mean
        })
    return cands

def _feature_cache_paths(cache: Path, key: str) -> tuple[Path, ...]:
    """Find one match fingerprint shared by all four feature producers."""
    parts = ("features_v1", "control_v1", "plus_v1", "physv2")
    filename = re.compile(rf"{re.escape(key)}_([0-9a-f]{{24}})\.parquet")
    paths_by_part = {}
    for part in parts:
        matches = {}
        for path in (cache / part).glob(f"{key}_*.parquet"):
            match = filename.fullmatch(path.name)
            if match and path.is_file():
                matches[match[1]] = path
        paths_by_part[part] = matches

    common = set.intersection(*(set(matches) for matches in paths_by_part.values()))
    if not common:
        fingerprints = ", ".join(
            f"{part}={sorted(matches)}" for part, matches in paths_by_part.items()
        )
        raise FileNotFoundError(
            f"No common feature cache fingerprint for {key} under {cache}: {fingerprints}"
        )
    if len(common) > 1:
        raise ValueError(
            f"Ambiguous feature caches for {key} under {cache}: {sorted(common)}"
        )
    fingerprint = common.pop()
    return tuple(paths_by_part[part][fingerprint] for part in parts)


def rebuild_candidates(root: Path, output_path: Path, kind: str) -> Path:
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
    extract = {"duel": extract_duel_cands_v9, "mbr": extract_mbr_xyx_cands}[kind]
    from src.data.sportec import load_all as load_sportec
    from src.data.metrica_sample import GAMES, load as load_sample
    from src.data.metrica_atd import load as load_atd

    def table(prefix, half):
        cache = root / "local_data" / "cache"
        paths = _feature_cache_paths(cache, prefix)
        return pd.concat([change_features(pd.read_parquet(paths[0]), half),
                          *(pd.read_parquet(p) for p in paths[1:])], axis=1)

    with (root / "experiments/outputs/v9_swarm/_diag_streams_s3.pkl").open("rb") as f:
        diag = pickle.load(f)
    streams, pred = diag["streams"], diag["pr_s3"]
    sportec = load_sportec()
    sample = {gid: load_sample(gid) for gid in GAMES}
    atd = load_atd()
    result = {key: {} for key in ("v3", "v6_std", "v6_gen", "j2")}
    for mid in sorted(sportec):
        match = sportec[mid]
        for split, feature_version in (("v3", "v3"), ("v6_std", "v6"), ("v6_gen", "v6")):
            features = table(f"sportec_{mid}_degrade_{feature_version}_s0", match.half)
            result[split][mid] = extract(pred[split][mid], *streams[split][mid], match.half, features, match.labels)
    for gid in GAMES:
        match = sample[gid]
        features = table(f"sample_{gid}_raw", match.half)
        result["j2"][gid] = extract(pred["sample"][gid], *streams["sample"][gid], match.half, features, match.labels)
    features = table("atd_rep_True_True_None", atd.half)
    result["atd_rep"] = extract(pred["atd_rep"], *streams["atd_rep"],
                                np.where(atd.graded, atd.half, 0), features, None)
    with output_path.open("xb") as f:
        pickle.dump(result, f)
    return output_path
