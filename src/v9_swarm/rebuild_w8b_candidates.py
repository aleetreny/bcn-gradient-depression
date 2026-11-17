"""Regenerate historical W8B training candidates on public Sportec folds."""

from pathlib import Path
import pickle

import numpy as np

import evaluation.possession_metrics as pm
from src.data.common import HOME, AWAY, DEAD
from src.data.sportec import MATCH_IDS
from src.eval.metrics import to_official


TRACKS = ("v3", "v6_std", "v6_gen")
DELTAS = np.arange(-12, 13, dtype=int)


def rle_runs_half(y, half):
    T = len(y)
    ch = np.flatnonzero((y[1:] != y[:-1]) | (half[1:] != half[:-1])) + 1
    starts = np.r_[0, ch]
    ends = np.r_[ch, T]
    return [(int(a), int(b), int(y[a]), int(half[a])) for a, b in zip(starts, ends)]

def preclean_xv(xv):
    c = {}
    vis = np.nan_to_num(xv['ball_visible'], nan=0.0)
    c['vis'] = vis
    d1_ig_h = np.clip(np.nan_to_num(xv['d1_h_impulse_gated'], nan=15.0), 0.0, 30.0)
    d1_ig_a = np.clip(np.nan_to_num(xv['d1_a_impulse_gated'], nan=15.0), 0.0, 30.0)
    c['d1_ig_h'] = d1_ig_h
    c['d1_ig_a'] = d1_ig_a
    # For visible ball distance, fallback to d1_impulse_gated when ball_visible==0 or NaN
    d1_h_raw = xv['dist_ball_H']
    d1_a_raw = xv['dist_ball_A']
    c['d1_h'] = np.clip(np.where(np.isfinite(d1_h_raw), d1_h_raw, d1_ig_h), 0.0, 30.0)
    c['d1_a'] = np.clip(np.where(np.isfinite(d1_a_raw), d1_a_raw, d1_ig_a), 0.0, 30.0)
    c['d1_h_vis'] = np.where(np.isfinite(d1_h_raw) & (vis > 0.5), d1_h_raw, 30.0)
    c['d1_a_vis'] = np.where(np.isfinite(d1_a_raw) & (vis > 0.5), d1_a_raw, 30.0)

    c['bspd'] = np.clip(np.nan_to_num(xv['ball_speed'], nan=5.0), 0.0, 35.0)
    c['bacc'] = np.clip(np.nan_to_num(xv['ball_accel'], nan=0.0), -50.0, 50.0)
    c['bang'] = np.clip(np.nan_to_num(np.abs(xv['ball_angle_change']), nan=0.0), 0.0, 3.1416)
    c['imp_d2'] = np.clip(np.nan_to_num(xv['impulse_mag_d2'], nan=0.0), 0.0, 35.0)
    c['imp_d4'] = np.clip(np.nan_to_num(xv['impulse_mag_d4'], nan=0.0), 0.0, 35.0)
    c['imp_norm_d4'] = np.clip(np.nan_to_num(xv['impulse_norm_d4'], nan=0.0), 0.0, 35.0)
    c['cos_d2'] = np.clip(np.nan_to_num(xv['impulse_cos_d2'], nan=1.0), -1.0, 1.0)
    c['cos_d4'] = np.clip(np.nan_to_num(xv['impulse_cos_d4'], nan=1.0), -1.0, 1.0)
    c['chord_flight'] = np.clip(np.nan_to_num(xv['chord_in_flight'], nan=0.0), 0.0, 1.0)
    c['nk_h'] = np.clip(np.nan_to_num(xv['next_knot_d1_h'], nan=10.0), 0.0, 30.0)
    c['nk_a'] = np.clip(np.nan_to_num(xv['next_knot_d1_a'], nan=10.0), 0.0, 30.0)
    c['pk_h'] = np.clip(np.nan_to_num(xv['prev_knot_d1_h'], nan=10.0), 0.0, 30.0)
    c['pk_a'] = np.clip(np.nan_to_num(xv['prev_knot_d1_a'], nan=10.0), 0.0, 30.0)
    c['cs_h'] = np.clip(np.nan_to_num(xv['chord_same_team_prob_h'], nan=0.5), 0.0, 1.0)
    c['cs_a'] = np.clip(np.nan_to_num(xv['chord_same_team_prob_a'], nan=0.5), 0.0, 1.0)
    c['cturn'] = np.clip(np.nan_to_num(xv['chord_turnover_prob'], nan=0.5), 0.0, 1.0)
    c['out_pitch'] = np.clip(np.nan_to_num(xv['ball_outside_pitch'], nan=0.0), 0.0, 1.0)
    c['gdisc'] = np.clip(np.nan_to_num(xv['ball_focus_discrepancy'], nan=0.0), 0.0, 35.0)
    c['fstill'] = np.clip(np.nan_to_num(xv['frac_still_all'], nan=0.0), 0.0, 1.0)
    return c


def build_episode_features(a, b, y_lbl, x_lbl, a_prev, b_next, P, p_ch, p4d, cxv, fams):
    T = len(P)
    L = float(b - a)
    sl = slice(a, b)

    # 1. Probabilities & family consensus inside [a, b)
    pY = P[sl, y_lbl]
    pX = P[sl, x_lbl]
    pD = P[sl, DEAD]
    mean_pY = float(np.mean(pY))
    max_pY = float(np.max(pY))
    q75_pY = float(np.percentile(pY, 75))
    mean_pX = float(np.mean(pX))
    mean_pD = float(np.mean(pD))
    margin_YX = float(np.mean(pY - pX))
    max_margin_YX = float(np.max(pY - pX))
    logodds_YX = float(np.mean(np.log((pY + 1e-4) / (pX + 1e-4))))

    # 9-model consensus: tree (0,1,2), seq (3,4,5,6,7), pset (8)
    tree_pY = float(np.mean([np.mean(fams[k][sl, y_lbl]) for k in (0, 1, 2)]))
    tree_max_pY = float(np.mean([np.max(fams[k][sl, y_lbl]) for k in (0, 1, 2)]))
    tree_margin_YX = float(np.mean([np.mean(fams[k][sl, y_lbl] - fams[k][sl, x_lbl]) for k in (0, 1, 2)]))
    seq_pY = float(np.mean([np.mean(fams[k][sl, y_lbl]) for k in (3, 4, 5, 6, 7)]))
    seq_margin_YX = float(np.mean([np.mean(fams[k][sl, y_lbl] - fams[k][sl, x_lbl]) for k in (3, 4, 5, 6, 7)]))
    pset_pY = float(np.mean(fams[8][sl, y_lbl])) if len(fams) > 8 else mean_pY
    pset_margin_YX = float(np.mean(fams[8][sl, y_lbl] - fams[8][sl, x_lbl])) if len(fams) > 8 else margin_YX
    fam_margins = [float(np.mean(fp[sl, y_lbl] - fp[sl, x_lbl])) for fp in fams]
    fam_vote_Y = float(np.mean([1.0 if m > 0 else 0.0 for m in fam_margins]))
    fam_min_margin = float(np.min(fam_margins))
    tree_minus_seq = tree_margin_YX - seq_margin_YX

    # Flanking context
    sl_pre = slice(max(a_prev, a - 20), a)
    sl_post = slice(b, min(b_next, b + 20))
    pre_pX = float(np.mean(P[sl_pre, x_lbl])) if sl_pre.stop > sl_pre.start else mean_pX
    post_pX = float(np.mean(P[sl_post, x_lbl])) if sl_post.stop > sl_post.start else mean_pX
    post_pD = float(np.mean(P[sl_post, DEAD])) if sl_post.stop > sl_post.start else mean_pD
    l_flank = min(float(a - a_prev) / 100.0, 2.0)
    r_flank = min(float(b_next - b) / 100.0, 2.0)

    # 2. Directional boundaries (SymBoundary4Dir: col 0 is ->H, col 1 is ->A, col 2 is ->D, col 3 is D->Live)
    col_to_Y = 0 if y_lbl == HOME else 1
    col_to_X = 0 if x_lbl == HOME else 1
    wa0, wa1 = max(0, a - 4), min(T, a + 5)
    wb0, wb1 = max(0, b - 4), min(T, b + 5)
    b4d_in_Y = float(np.max(p4d[wa0:wa1, col_to_Y])) if wa1 > wa0 else 0.0
    b4d_in_fromD = float(np.max(p4d[wa0:wa1, 3])) if wa1 > wa0 else 0.0
    b4d_out_X = float(np.max(p4d[wb0:wb1, col_to_X])) if wb1 > wb0 else 0.0
    b4d_out_D = float(np.max(p4d[wb0:wb1, 2])) if wb1 > wb0 else 0.0
    pc_in = float(np.max(p_ch[wa0:wa1])) if wa1 > wa0 else 0.0
    pc_out = float(np.max(p_ch[wb0:wb1])) if wb1 > wb0 else 0.0

    # 3. Contact Kinematics Features (t_in, t_out)
    w_imp = slice(max(0, a - 3), min(T, b + 4))
    imp_d2_max = float(np.max(cxv['imp_d2'][w_imp])) / 30.0
    imp_d4_max = float(np.max(cxv['imp_d4'][w_imp])) / 30.0
    imp_norm_d4_max = float(np.max(cxv['imp_norm_d4'][w_imp])) / 30.0
    imp_in = float(np.max(cxv['imp_d2'][wa0:wa1])) / 30.0 if wa1 > wa0 else 0.0
    imp_out = float(np.max(cxv['imp_d2'][wb0:wb1])) / 30.0 if wb1 > wb0 else 0.0

    # Ball velocity deflection & deceleration/control ratio
    ang_max = float(np.max(cxv['bang'][w_imp])) / 3.1416
    cos_d2_min = float(np.min(cxv['cos_d2'][w_imp]))
    cos_d4_min = float(np.min(cxv['cos_d4'][w_imp]))
    v_seg = cxv['bspd'][sl]
    v_pre_idx = max(0, a - 2)
    v_in_ref = max(float(cxv['bspd'][v_pre_idx]), 1.0)
    decel_ratio = min(float(np.min(v_seg)) / v_in_ref, 3.0)
    mean_v_seg = float(np.mean(v_seg)) / 30.0
    min_v_seg = float(np.min(v_seg)) / 30.0

    # Effective contact frame count of Y (d1_Y < 1.3m with ball_visible == 1) vs min distance of X
    vis = cxv['vis'][sl]
    frac_vis = float(np.mean(vis))
    d1_Y_vis = (cxv['d1_h_vis'] if y_lbl == HOME else cxv['d1_a_vis'])[sl]
    d1_X_vis = (cxv['d1_h_vis'] if x_lbl == HOME else cxv['d1_a_vis'])[sl]
    d1_Y = (cxv['d1_h'] if y_lbl == HOME else cxv['d1_a'])[sl]
    d1_X = (cxv['d1_h'] if x_lbl == HOME else cxv['d1_a'])[sl]
    d1_ig_Y = (cxv['d1_ig_h'] if y_lbl == HOME else cxv['d1_ig_a'])[sl]
    d1_ig_X = (cxv['d1_ig_h'] if x_lbl == HOME else cxv['d1_ig_a'])[sl]

    eff_contact_cnt_Y = float(np.sum(d1_Y_vis < 1.3)) / 25.0
    eff_contact_frac_Y = float(np.mean(d1_Y_vis < 1.3))
    eff_contact_cnt_X = float(np.sum(d1_X_vis < 1.3)) / 25.0
    min_d1_Y = float(np.min(d1_Y)) / 30.0
    min_d1_X = float(np.min(d1_X)) / 30.0
    mean_d1_diff_XY = float(np.mean(np.clip(d1_X - d1_Y, -20.0, 20.0))) / 20.0
    min_d1_ig_Y = float(np.min(d1_ig_Y)) / 30.0
    min_d1_ig_X = float(np.min(d1_ig_X)) / 30.0
    chord_flight = float(np.mean(cxv['chord_flight'][sl]))

    # 4. Post-Impulse Fate Features over [t_out, t_out + 15]
    sl_fate = slice(b, min(T, b + 15))
    if sl_fate.stop > sl_fate.start:
        d1_Y_post = (cxv['d1_h'] if y_lbl == HOME else cxv['d1_a'])[sl_fate]
        d1_X_post = (cxv['d1_h'] if x_lbl == HOME else cxv['d1_a'])[sl_fate]
        post_d1_diff_XY = float(np.mean(np.clip(d1_X_post - d1_Y_post, -20.0, 20.0))) / 20.0
        post_min_d1_Y = float(np.min(d1_Y_post)) / 30.0
        post_min_d1_X = float(np.min(d1_X_post)) / 30.0
        post_out_pitch = float(np.mean(cxv['out_pitch'][sl_fate]))
    else:
        post_d1_diff_XY, post_min_d1_Y, post_min_d1_X, post_out_pitch = 0.0, min_d1_Y, min_d1_X, 0.0

    nk_Y = (cxv['nk_h'] if y_lbl == HOME else cxv['nk_a'])[sl]
    nk_X = (cxv['nk_h'] if x_lbl == HOME else cxv['nk_a'])[sl]
    pk_Y = (cxv['pk_h'] if y_lbl == HOME else cxv['pk_a'])[sl]
    pk_X = (cxv['pk_h'] if x_lbl == HOME else cxv['pk_a'])[sl]
    cs_Y = (cxv['cs_h'] if y_lbl == HOME else cxv['cs_a'])[sl]
    cs_X = (cxv['cs_h'] if x_lbl == HOME else cxv['cs_a'])[sl]

    next_knot_diff_XY = float(np.mean(np.clip(nk_X - nk_Y, -20.0, 20.0))) / 20.0
    prev_knot_diff_XY = float(np.mean(np.clip(pk_X - pk_Y, -20.0, 20.0))) / 20.0
    chord_same_Y = float(np.mean(cs_Y))
    chord_same_X = float(np.mean(cs_X))
    chord_turn = float(np.mean(cxv['cturn'][sl]))
    gdisc = float(np.mean(cxv['gdisc'][sl])) / 30.0

    return np.round(np.array([
        L / 80.0, l_flank, r_flank,
        mean_pY, max_pY, q75_pY, mean_pX, mean_pD, margin_YX, max_margin_YX, logodds_YX,
        tree_pY, tree_max_pY, tree_margin_YX, seq_pY, seq_margin_YX, pset_pY, pset_margin_YX,
        fam_vote_Y, fam_min_margin, tree_minus_seq,
        pre_pX, post_pX, post_pD,
        b4d_in_Y, b4d_in_fromD, b4d_out_X, b4d_out_D, pc_in, pc_out,
        imp_d2_max, imp_d4_max, imp_norm_d4_max, imp_in, imp_out,
        ang_max, cos_d2_min, cos_d4_min, decel_ratio, mean_v_seg, min_v_seg,
        frac_vis, eff_contact_cnt_Y, eff_contact_frac_Y, eff_contact_cnt_X,
        min_d1_Y, min_d1_X, mean_d1_diff_XY, min_d1_ig_Y, min_d1_ig_X, chord_flight,
        post_d1_diff_XY, post_min_d1_Y, post_min_d1_X, post_out_pitch,
        next_knot_diff_XY, prev_knot_diff_XY, chord_same_Y, chord_same_X, chord_turn, gdisc,
    ], dtype=np.float64), 10)


def _track_arrays(match, track, clean_features):
    version = "v3" if track == "v3" else "v6"
    fold = match["w7b"]
    return (
        fold[f"pred_{track}_rerank"],
        fold[f"proba_{track}"],
        fold[f"pchange_{track}"],
        match[f"p4d_{version}"],
        clean_features[version],
        match[f"fams_{track}"],
    )


def _official_spells(pred, half, gt):
    official_gt = to_official(gt, half, True)
    valid = pm.valid_runs(official_gt)
    official_pred = to_official(pred, half, True)
    official_pred[official_gt == pm.X] = pm.X
    return pm.possession_spells(official_gt, valid), pm.possession_spells(official_pred, valid)


def _f1_from_spells(gt_spells, pred_spells):
    overlaps = pm.spell_overlaps(gt_spells, pred_spells)
    return f1_from_ov(overlaps, len(gt_spells), len(pred_spells))


def _mode1(m, arrays, half, gt, max_len=85):
    pred, P, p_ch, p4d, cxv, fams = arrays
    gt_spells, base_spells = _official_spells(pred, half, gt)
    base_f1 = _f1_from_spells(gt_spells, base_spells)

    r = rle_runs_half(pred, half)
    out = []
    for k in range(1, len(r) - 1):
        a_prev, b_prev, l_lbl, h_prev = r[k - 1]
        a, b, y_lbl, h_cur = r[k]
        a_next, b_next, r_lbl, h_next = r[k + 1]
        if y_lbl not in (HOME, AWAY):
            continue
        L = b - a
        if L < 5 or L > max_len:
            continue
        if h_cur <= 0 or h_prev != h_cur or h_next != h_cur:
            continue
        x_lbl = 1 - y_lbl
        if l_lbl == x_lbl and r_lbl == x_lbl:
            pat = 'XYX'
        elif l_lbl == x_lbl and r_lbl == DEAD:
            pat = 'XYD'
        elif l_lbl == DEAD and r_lbl == x_lbl:
            pat = 'DYX'
        else:
            continue
        feat = build_episode_features(a, b, y_lbl, x_lbl, a_prev, b_next, P, p_ch, p4d, cxv, fams)
        shift = 1 if h_cur == 2 else 0
        a_s, b_s = a + shift, b + shift
        mod_spells = []
        i_sp = 0
        while i_sp < len(base_spells):
            sa, sb, slbl = base_spells[i_sp]
            if sa == a_s and sb == b_s and slbl == y_lbl:
                if pat == 'XYX' and mod_spells and mod_spells[-1][1] == a_s and mod_spells[-1][2] == x_lbl:
                    if i_sp + 1 < len(base_spells) and base_spells[i_sp + 1][0] == b_s and base_spells[i_sp + 1][2] == x_lbl:
                        mod_spells[-1] = (mod_spells[-1][0], base_spells[i_sp + 1][1], x_lbl)
                        i_sp += 2
                        continue
                elif pat == 'XYD' and mod_spells and mod_spells[-1][1] == a_s and mod_spells[-1][2] == x_lbl:
                    mod_spells[-1] = (mod_spells[-1][0], b_s, x_lbl)
                    i_sp += 1
                    continue
                elif pat == 'DYX' and i_sp + 1 < len(base_spells) and base_spells[i_sp + 1][0] == b_s and base_spells[i_sp + 1][2] == x_lbl:
                    mod_spells.append((a_s, base_spells[i_sp + 1][1], x_lbl))
                    i_sp += 2
                    continue
            mod_spells.append((sa, sb, slbl))
            i_sp += 1
        mod_f1 = _f1_from_spells(gt_spells, mod_spells)
        delta_absorb = mod_f1 - base_f1
        frac_x = float(np.mean(gt[a:b] == x_lbl))
        frac_y = float(np.mean(gt[a:b] == y_lbl))
        frac_d = float(np.mean(gt[a:b] == DEAD))
        out.append({
            'm': m, 'a': a, 'b': b, 'L': L, 'y_lbl': y_lbl, 'x_lbl': x_lbl,
            'a_prev': a_prev, 'b_next': b_next, 'pat': pat,
            'feat': feat, 'delta_absorb': delta_absorb,
            'frac_x': frac_x, 'frac_y': frac_y, 'frac_d': frac_d,
        })
    return out


def _mode2(match_id, arrays, half, gt, min_len=8, max_len=65):
    # Keep the historical episode features and original cache dtypes.
    pred, P, p_change, p_trans_4d, cxv, fams = arrays
    gt_spells, base_spells = _official_spells(pred, half, gt)
    base_f1 = _f1_from_spells(gt_spells, base_spells)

    if fams is not None and len(fams) >= 3:
        tree_P = (fams[0] + fams[1] + fams[2]) / 3.0
    else:
        tree_P = P

    r = rle_runs_half(pred, half)
    out = []
    for sa, sb, x_lbl, h_cur in r:
        if x_lbl not in (HOME, AWAY) or h_cur <= 0:
            continue
        if sb - sa < 50:
            continue
        y_lbl = 1 - x_lbl
        col_to_Y = 0 if y_lbl == HOME else 1
        d1_ig_y = cxv["d1_ig_h"] if y_lbl == HOME else cxv["d1_ig_a"]
        sig = np.maximum(P[sa:sb, y_lbl], tree_P[sa:sb, y_lbl])
        bdy_sig = np.maximum(p_change[sa:sb], p_trans_4d[sa:sb, col_to_Y])
        active = (sig > 0.24) | ((sig > 0.18) & (d1_ig_y[sa:sb] < 1.6) & (bdy_sig > 0.12))
        if not np.any(active):
            continue

        idx = np.flatnonzero(active)
        splits = np.where(np.diff(idx) > 4)[0] + 1
        groups = np.split(idx, splits)
        for grp in groups:
            ca = int(sa + grp[0])
            cb = int(sa + grp[-1] + 1)
            if cb - ca < min_len:
                mid = (ca + cb) // 2
                ca = max(sa + 12, mid - 5)
                cb = min(sb - 12, mid + 5)
            if ca - sa < 12 or sb - cb < 12:
                continue
            L = cb - ca
            if L < min_len or L > max_len:
                continue
            if float(np.max(sig[ca - sa : cb - sa])) < 0.28:
                continue

            feat = build_episode_features(ca, cb, y_lbl, x_lbl, sa, sb, P, p_change, p_trans_4d, cxv, fams)
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
            delta_insert = _f1_from_spells(gt_spells, mod_spells) - base_f1
            frac_y = float(np.mean(gt[ca:cb] == y_lbl))
            frac_x = float(np.mean(gt[ca:cb] == x_lbl))

            out.append(
                {
                    "m": match_id,
                    "a": ca,
                    "b": cb,
                    "L": L,
                    "y_lbl": y_lbl,
                    "x_lbl": x_lbl,
                    "a_prev": sa,
                    "b_next": sb,
                    "pat": "INS_Y",
                    "feat": feat,
                    "delta_insert": delta_insert,
                    "frac_y": frac_y,
                    "frac_x": frac_x,
                }
            )
    return out


def build_offset_features(tb, btype, s_old, s_new, L_left, L_right, P, p_ch, p4d, cxv, fams):
    """Build (25, 26) 100% H<->A-invariant feature matrix for offsets d in [-12, +12]."""
    T = len(P)
    feats = np.zeros((len(DELTAS), 26), dtype=np.float64)

    tree_p_old = (fams[0][:, s_old] + fams[1][:, s_old] + fams[2][:, s_old]) / 3.0
    tree_p_new = (fams[0][:, s_new] + fams[1][:, s_new] + fams[2][:, s_new]) / 3.0
    seq_p_old = (fams[3][:, s_old] + fams[4][:, s_old] + fams[5][:, s_old] + fams[6][:, s_old] + fams[7][:, s_old]) / 5.0
    seq_p_new = (fams[3][:, s_new] + fams[4][:, s_new] + fams[5][:, s_new] + fams[6][:, s_new] + fams[7][:, s_new]) / 5.0
    pset_p_old = fams[8][:, s_old] if len(fams) > 8 else P[:, s_old]
    pset_p_new = fams[8][:, s_new] if len(fams) > 8 else P[:, s_new]

    if btype == 'HA':
        col_dir = 0 if s_new == HOME else 1
        col_rev = 0 if s_old == HOME else 1
    elif btype == 'L2D':
        col_dir = 2
        col_rev = 3
    else:  # D2L
        col_dir = 3
        col_rev = 2

    if btype == 'HA':
        d1_new = cxv['d1_h'] if s_new == HOME else cxv['d1_a']
        d1_ig_old = cxv['d1_ig_h'] if s_old == HOME else cxv['d1_ig_a']
        d1_ig_new = cxv['d1_ig_h'] if s_new == HOME else cxv['d1_ig_a']
    elif btype == 'L2D':
        d1_new = cxv['d1_a'] if s_old == HOME else cxv['d1_h']
        d1_ig_old = cxv['d1_ig_h'] if s_old == HOME else cxv['d1_ig_a']
        d1_ig_new = cxv['d1_ig_a'] if s_old == HOME else cxv['d1_ig_h']
    else:  # D2L
        d1_new = cxv['d1_h'] if s_new == HOME else cxv['d1_a']
        d1_ig_old = cxv['d1_ig_a'] if s_new == HOME else cxv['d1_ig_h']
        d1_ig_new = cxv['d1_ig_h'] if s_new == HOME else cxv['d1_ig_a']

    for idx_d, d in enumerate(DELTAS):
        tc = int(np.clip(tb + d, 4, T - 5))

        p_new_post4 = float(np.mean(P[tc:tc+4, s_new]))
        p_new_pre4 = float(np.mean(P[tc-4:tc, s_new]))
        p_old_post4 = float(np.mean(P[tc:tc+4, s_old]))
        p_old_pre4 = float(np.mean(P[tc-4:tc, s_old]))
        dp_step4 = (p_new_post4 - p_new_pre4) - (p_old_post4 - p_old_pre4)

        w8_pre = slice(max(0, tc - 8), tc)
        w8_post = slice(tc, min(T, tc + 8))
        dp_step8 = (float(np.mean(P[w8_post, s_new])) - float(np.mean(P[w8_pre, s_new]))) - (
            float(np.mean(P[w8_post, s_old])) - float(np.mean(P[w8_pre, s_old]))
        )

        if d < 0:
            sl_seg = slice(max(0, tb + d), tb)
            seg_prob_gain = float(np.sum(P[sl_seg, s_new] - P[sl_seg, s_old])) / 12.0
            seg_tree_gain = float(np.sum(tree_p_new[sl_seg] - tree_p_old[sl_seg])) / 12.0
            seg_seq_gain = float(np.sum(seq_p_new[sl_seg] - seq_p_old[sl_seg])) / 12.0
            seg_pset_gain = float(np.sum(pset_p_new[sl_seg] - pset_p_old[sl_seg])) / 12.0
        elif d > 0:
            sl_seg = slice(tb, min(T, tb + d))
            seg_prob_gain = float(np.sum(P[sl_seg, s_old] - P[sl_seg, s_new])) / 12.0
            seg_tree_gain = float(np.sum(tree_p_old[sl_seg] - tree_p_new[sl_seg])) / 12.0
            seg_seq_gain = float(np.sum(seq_p_old[sl_seg] - seq_p_new[sl_seg])) / 12.0
            seg_pset_gain = float(np.sum(pset_p_old[sl_seg] - pset_p_new[sl_seg])) / 12.0
        else:
            seg_prob_gain, seg_tree_gain, seg_seq_gain, seg_pset_gain = 0.0, 0.0, 0.0, 0.0

        p_diff_at_tc = float(P[tc, s_new] - P[tc, s_old])
        tree_diff_at_tc = float(tree_p_new[tc] - tree_p_old[tc])

        b4d_dir_tc = float(p4d[tc, col_dir])
        b4d_dir_max1 = float(np.max(p4d[tc-1:tc+2, col_dir]))
        b4d_rev_tc = float(p4d[tc, col_rev])
        pch_tc = float(p_ch[tc])
        pch_max1 = float(np.max(p_ch[tc-1:tc+2]))

        imp_d2_tc = float(cxv['imp_d2'][tc]) / 30.0
        imp_d4_tc = float(cxv['imp_d4'][tc]) / 30.0
        imp_norm_d4_tc = float(cxv['imp_norm_d4'][tc]) / 30.0
        acc_tc = float(np.abs(cxv['bacc'][tc])) / 40.0
        ang_tc = float(cxv['bang'][tc]) / 3.1416
        vis_tc = float(cxv['vis'][tc])

        d1_new_tc = float(d1_ig_new[tc]) / 30.0
        d1_old_tc = float(d1_ig_old[tc]) / 30.0
        d1_cross_step = (float(np.mean(d1_ig_old[tc:tc+4] - d1_ig_new[tc:tc+4])) - float(np.mean(d1_ig_old[tc-4:tc] - d1_ig_new[tc-4:tc]))) / 20.0
        contact_new_tc = 1.0 if (vis_tc > 0.5 and float(d1_new[tc]) < 1.35) else 0.0

        min_line_tc = min(float(cxv['touchline'][tc]), float(cxv['goalline'][tc])) / 30.0
        out_pitch_step = float(np.mean(cxv['out_pitch'][tc:tc+4]) - np.mean(cxv['out_pitch'][tc-4:tc]))
        offset_pen = abs(float(d)) / 12.0

        feats[idx_d] = np.round([
            dp_step4, dp_step8, seg_prob_gain, seg_tree_gain, seg_seq_gain, seg_pset_gain,
            p_diff_at_tc, tree_diff_at_tc,
            b4d_dir_tc, b4d_dir_max1, b4d_rev_tc, pch_tc, pch_max1,
            imp_d2_tc, imp_d4_tc, imp_norm_d4_tc, acc_tc, ang_tc, vis_tc,
            d1_new_tc, d1_old_tc, d1_cross_step, contact_new_tc,
            min_line_tc, out_pitch_step, offset_pen,
        ], 10)
    return feats

def f1_from_ov(ov, n_gt, n_pr):
    f1s = []
    for th in pm.SPELL_IOUS:
        tp, fp, fn = pm.match_spells(ov, n_gt, n_pr, th)
        f1s.append(2.0 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0)
    return float(np.mean(f1s))

def single_spell_overlaps(gt_s, sp, pred_idx):
    a_p, b_p, c_p = sp
    out = []
    for i, (a_g, b_g, c_g) in enumerate(gt_s):
        if c_p != c_g or b_p <= a_g or b_g <= a_p:
            continue
        inter = min(b_g, b_p) - max(a_g, a_p)
        uni = max(b_g, b_p) - min(a_g, a_p)
        out.append((inter / float(uni), i, pred_idx))
    return out


def _boundaries(m, arrays, half, gt, min_spell_len=20):
    pred, P, p_ch, p4d, cxv, fams = arrays
    gt_s, base_spells = _official_spells(pred, half, gt)
    base_ov = pm.spell_overlaps(gt_s, base_spells)
    n_gt, n_pr = len(gt_s), len(base_spells)
    base_f1 = f1_from_ov(base_ov, n_gt, n_pr)

    start_map = {sp[0]: idx for idx, sp in enumerate(base_spells)}
    end_map = {sp[1]: idx for idx, sp in enumerate(base_spells)}

    r = rle_runs_half(pred, half)
    out = []
    for k in range(len(r) - 1):
        a1, b1, s_old, h1 = r[k]
        a2, b2, s_new, h2 = r[k + 1]
        if h1 <= 0 or h1 != h2 or b1 != a2:
            continue
        tb = b1
        L_left = b1 - a1
        L_right = b2 - a2
        if L_left < min_spell_len or L_right < min_spell_len:
            continue
        if s_old in (HOME, AWAY) and s_new in (HOME, AWAY):
            btype = 'HA'
        elif s_old in (HOME, AWAY) and s_new == DEAD:
            btype = 'L2D'
        elif s_old == DEAD and s_new in (HOME, AWAY):
            btype = 'D2L'
        else:
            continue

        shift = 1 if h1 == 2 else 0
        tb_s = tb + shift
        idx_left = end_map.get(tb_s, None)
        idx_right = start_map.get(tb_s, None)

        rem_set = {x for x in (idx_left, idx_right) if x is not None}
        ov_without = [item for item in base_ov if item[2] not in rem_set]
        ov_orig_lr = [item for item in base_ov if item[2] in rem_set]
        orig_iou_sum = sum(item[0] for item in ov_orig_lr)

        feats = build_offset_features(tb, btype, s_old, s_new, L_left, L_right, P, p_ch, p4d, cxv, fams)
        valid_mask = np.ones(len(DELTAS), dtype=bool)
        f1_deltas = np.zeros(len(DELTAS), dtype=np.float64)
        iou_deltas = np.zeros(len(DELTAS), dtype=np.float64)
        frame_deltas = np.zeros(len(DELTAS), dtype=np.float64)

        vis_tb = float(cxv['vis'][tb]) > 0.5
        for idx_d, d in enumerate(DELTAS):
            tc = tb + int(d)
            if tc < 4 or tc >= len(pred) - 5:
                valid_mask[idx_d] = False
                continue
            if L_left + d < min_spell_len or L_right - d < min_spell_len:
                valid_mask[idx_d] = False
                continue
            vis_tc = float(cxv['vis'][tc]) > 0.5
            if (not vis_tb) or (not vis_tc):
                if d != 0:
                    valid_mask[idx_d] = False
                    continue
            if d == 0:
                continue

            # Frame accuracy delta
            if d < 0:
                sl = slice(tb + int(d), tb)
                frame_deltas[idx_d] = float(np.sum(gt[sl] == s_new) - np.sum(gt[sl] == s_old))
            else:
                sl = slice(tb, tb + int(d))
                frame_deltas[idx_d] = float(np.sum(gt[sl] == s_old) - np.sum(gt[sl] == s_new))

            new_lr = []
            if btype == 'HA' and idx_left is not None and idx_right is not None:
                sp_l = (base_spells[idx_left][0], tb_s + int(d), base_spells[idx_left][2])
                sp_r = (tb_s + int(d), base_spells[idx_right][1], base_spells[idx_right][2])
                new_lr = single_spell_overlaps(gt_s, sp_l, idx_left) + single_spell_overlaps(gt_s, sp_r, idx_right)
            elif btype == 'L2D' and idx_left is not None:
                sp_l = (base_spells[idx_left][0], tb_s + int(d), base_spells[idx_left][2])
                new_lr = single_spell_overlaps(gt_s, sp_l, idx_left)
            elif btype == 'D2L' and idx_right is not None:
                sp_r = (tb_s + int(d), base_spells[idx_right][1], base_spells[idx_right][2])
                new_lr = single_spell_overlaps(gt_s, sp_r, idx_right)

            iou_deltas[idx_d] = sum(item[0] for item in new_lr) - orig_iou_sum
            if abs(iou_deltas[idx_d]) > 1e-9:
                ov_mod = ov_without + new_lr
                ov_mod.sort(key=lambda item: (-item[0], item[1], item[2]))
                f1_deltas[idx_d] = f1_from_ov(ov_mod, n_gt, n_pr) - base_f1

        out.append({
            'm': m, 'tb': tb, 'btype': btype, 's_old': s_old, 's_new': s_new,
            'L_left': L_left, 'L_right': L_right, 'vis_tb': vis_tb,
            'feats': feats, 'valid_mask': valid_mask,
            'f1_deltas': f1_deltas, 'iou_deltas': iou_deltas, 'frame_deltas': frame_deltas,
        })
    return out


def _clean_features(table, boundaries):
    features = preclean_xv(table)
    if boundaries:
        features["touchline"] = np.clip(np.nan_to_num(table["ball_last_to_touchline"], nan=20.0), -5.0, 40.0)
        features["goalline"] = np.clip(np.nan_to_num(table["ball_last_to_goalline"], nan=30.0), -5.0, 60.0)
        features["spd_ch"] = np.clip(np.nan_to_num(table["ball_speed_change"], nan=0.0), -30.0, 30.0)
        features["imp_d8"] = np.clip(np.nan_to_num(table["impulse_mag_d8"], nan=0.0), 0.0, 35.0)
    return features


def extract_sportec_candidates(cache, kind):
    """Produce one historical W8B candidate cache from the public folds."""
    if set(cache) != set(MATCH_IDS):
        raise ValueError("W8B training inputs must contain exactly the public Sportec matches")
    extract = {"mode1": _mode1, "mode2": _mode2, "boundaries": _boundaries}[kind]
    result = {track: {} for track in TRACKS}
    for mid in sorted(cache):
        match = cache[mid]
        cleaned = {
            version: _clean_features(match[f"x{version}"], kind == "boundaries")
            for version in ("v3", "v6")
        }
        for track in TRACKS:
            arrays = _track_arrays(match, track, cleaned)
            result[track][mid] = extract(mid, arrays, match["hl"], match["gt"])
    return result


def rebuild_w8b_candidates(root: Path, output_path: Path, kind: str) -> Path:
    """Read the canonical Sportec cache and write one new local candidate pickle."""
    root = Path(root).resolve()
    output_path = Path(output_path).resolve()
    relative = output_path.relative_to(root)
    if relative.parts[:1] != ("_local",) and relative.parts[:2] != ("experiments", "outputs"):
        raise ValueError("output must be in _local/ or experiments/outputs/")
    if output_path.suffix != ".pkl":
        raise ValueError("output must be a .pkl cache")
    if output_path.exists():
        raise FileExistsError(output_path)

    with (root / "experiments/outputs/exp_w8b_oof/sportec_fast_cache.pkl").open("rb") as source:
        cache = pickle.load(source)
    result = extract_sportec_candidates(cache, kind)
    with output_path.open("xb") as target:
        pickle.dump(result, target)
    return output_path
