"""Error decomposition: where a prediction loses spellF1, by error type.

    from src.eval.errors import decompose
    d = decompose(gt, pred, half)          # common codes; dict of tables and counts

Types (all computed inside a half, never across):
- team_change timing: every ground-truth change of team in play (H->A, A->H, including through a
  D of < 1 s) is matched to the nearest predicted change to the same team within +/- MAX_LAG
  frames. offset = predicted frame - true frame (> 0 late, < 0 early). Unmatched ones are missed.
- short_spell_missed: ground-truth possessions shorter than SHORT (2 s) with no same-team
  predicted spell at IoU >= 0.3 (the lowest spellF1 threshold).
- spurious_spell: predicted possessions with no same-team ground-truth spell at IoU >= 0.3, split
  into flicker (< 2 s) and long.
- dead_missed / dead_invented: ground-truth D runs (>= 1 s) predicted mostly alive, and predicted
  D runs (>= 1 s) lying mostly on ground-truth alive frames.
- restart timing: offset of the predicted D -> alive edge against the true one, within MAX_LAG.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, FPS, HOME, possession_spells, runs

SHORT = 2 * FPS
MAX_LAG = 3 * FPS
MIN_D = FPS
IOU = 0.3


def _iou_best(spells_a, spells_b):
    """For each spell in a, the best IoU with a same-team spell in b.

    Spells from possession_spells are sorted and disjoint, so both starts and stops of b are
    sorted and the overlapping candidates of a spell are one contiguous slice.
    """
    sa, ea, ta = spells_a
    sb, eb, tb = spells_b
    best = np.zeros(len(sa))
    lo = np.searchsorted(eb, sa, side="right")
    hi = np.searchsorted(sb, ea, side="left")
    for i in range(len(sa)):
        js = np.arange(lo[i], hi[i])
        if len(js) == 0:
            continue
        inter = np.minimum(eb[js], ea[i]) - np.maximum(sb[js], sa[i])
        union = np.maximum(eb[js], ea[i]) - np.minimum(sb[js], sa[i])
        best[i] = np.max(np.where((tb[js] == ta[i]) & (inter > 0), inter / union, 0.0))
    return best


def team_changes(labels, half, bridge=MIN_D):
    """(frames, new_team) of changes of team in play; a D shorter than `bridge` is looked through."""
    s, e, t = possession_spells(labels, half)
    keep = (t[1:] != t[:-1]) & (half[s[1:]] == half[s[:-1]]) & ((s[1:] - e[:-1]) < bridge)
    return s[1:][keep], t[1:][keep]


def _match_edges(gt_frames, gt_kind, pred_frames, pred_kind, max_lag):
    """Greedy nearest matching of edges of the same kind within max_lag. Returns offsets and misses."""
    offsets, missed = [], 0
    used = np.zeros(len(pred_frames), bool)
    for f, k in zip(gt_frames, gt_kind):
        cand = np.flatnonzero((pred_kind == k) & ~used & (np.abs(pred_frames - f) <= max_lag))
        if len(cand) == 0:
            missed += 1
            continue
        j = cand[np.argmin(np.abs(pred_frames[cand] - f))]
        used[j] = True
        offsets.append(int(pred_frames[j] - f))
    return np.array(offsets, int), missed, int((~used).sum())


def _restarts(labels, half):
    key = np.where(labels >= 0, labels, -1).astype(int) * 10 + half
    s, e, v = runs(key)
    lab = v // 10
    out_f, out_k = [], []
    for i in range(1, len(s)):
        if lab[i - 1] == DEAD and lab[i] in (HOME, AWAY) and half[s[i]] == half[s[i - 1]] and (e[i - 1] - s[i - 1]) >= MIN_D:
            out_f.append(s[i])
            out_k.append(lab[i])
    return np.array(out_f, int), np.array(out_k, int)


def _dead_runs(labels, half):
    key = np.where(labels >= 0, labels, -1).astype(int) * 10 + half
    s, e, v = runs(key)
    keep = (v // 10 == DEAD) & ((e - s) >= MIN_D) & (half[s] > 0)
    return s[keep], e[keep]


def decompose(gt: np.ndarray, pred: np.ndarray, half: np.ndarray) -> dict:
    graded = (gt >= 0) & (half > 0)
    gt = np.where(graded, gt, -1)
    pred = np.where(graded, pred, -1)

    g_sp = possession_spells(gt, half)
    p_sp = possession_spells(pred, half)
    g_len = g_sp[1] - g_sp[0]
    p_len = p_sp[1] - p_sp[0]
    g_best = _iou_best(g_sp, p_sp)
    p_best = _iou_best(p_sp, g_sp)

    gf, gk = team_changes(gt, half)
    pf, pk = team_changes(pred, half)
    tc_off, tc_missed, tc_extra = _match_edges(gf, gk, pf, pk, MAX_LAG)

    rf, rk = _restarts(gt, half)
    prf, prk = _restarts(pred, half)
    rs_off, rs_missed, _ = _match_edges(rf, rk, prf, prk, MAX_LAG)

    gds, gde = _dead_runs(gt, half)
    pds, pde = _dead_runs(pred, half)
    dead_missed = sum(np.mean(pred[a:b] != DEAD) > 0.5 for a, b in zip(gds, gde))
    dead_invented = sum(np.mean(np.isin(gt[a:b], (HOME, AWAY))) > 0.5 for a, b in zip(pds, pde))

    def pct(x):
        return np.percentile(x, [10, 50, 90]).round(1).tolist() if len(x) else [np.nan] * 3

    table = pd.DataFrame([
        ("GT possessions", len(g_len), "", ""),
        ("  matched (IoU >= 0.3)", int((g_best >= IOU).sum()), "", ""),
        ("  short (< 2 s), missed", int(((g_len < SHORT) & (g_best < IOU)).sum()),
         "of {} short".format(int((g_len < SHORT).sum())), ""),
        ("  long (>= 2 s), missed", int(((g_len >= SHORT) & (g_best < IOU)).sum()), "", ""),
        ("predicted possessions", len(p_len), "", ""),
        ("  spurious flicker (< 2 s)", int(((p_len < SHORT) & (p_best < IOU)).sum()), "", ""),
        ("  spurious long (>= 2 s)", int(((p_len >= SHORT) & (p_best < IOU)).sum()), "", ""),
        ("team changes (GT)", len(gf), "", ""),
        ("  matched within 3 s", len(tc_off), "offset p10/p50/p90 (frames)", str(pct(tc_off))),
        ("  late > 12 frames", int((tc_off > 12).sum()), "", ""),
        ("  early > 12 frames", int((tc_off < -12).sum()), "", ""),
        ("  missed", tc_missed, "", ""),
        ("  extra predicted changes", tc_extra, "", ""),
        ("restarts D->play (GT, D >= 1 s)", len(rf), "offset p10/p50/p90 (frames)", str(pct(rs_off))),
        ("  missed", rs_missed, "", ""),
        ("D runs >= 1 s (GT)", len(gds), "", ""),
        ("  predicted mostly alive (D missed)", int(dead_missed), "", ""),
        ("D runs >= 1 s predicted", len(pds), "", ""),
        ("  mostly on GT alive (D invented)", int(dead_invented), "", ""),
    ], columns=["item", "count", "detail", "value"])
    return {"table": table, "team_change_offsets": tc_off, "restart_offsets": rs_off,
            "gt_spell_best_iou": g_best, "pred_spell_best_iou": p_best,
            "gt_spell_len": g_len, "pred_spell_len": p_len}
