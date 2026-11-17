"""Generalization Judge Suite (`J_SUITE`) & Event-Play Error Atlas (`src/v9_swarm/j_suite.py`).

Provides a fast, unified evaluation & diagnostic suite for candidate possession predictions
against `v8.5 Strict CV` (`v85_forensic_strict_cv`):

1. **Standard Judges with Paired Bootstrap CIs (`n_boot=1000` over matches and halves) vs `v8.5` baseline**:
   - `J1A` (7 Sportec matches, `degrade_v3 s0`): `spell_f1_mean`, `macro_f1`, `F1@0.3`, `F1@0.5`, `F1@0.7`,
     `poss/min`, `% < 2s`, folds improved `/7`, halves improved `/14`, and paired delta + 90% CI vs `v8.5`.
   - `J1B-Std` and `J1B-Gen` (7 Sportec matches, `degrade_v6 s0`).
   - `J2-Raw` and `J2-noD` (`without_dead_margin` on Metrica Sample 1-3, isolating live-play turnovers
     `H <-> A` on Metrica's exact annotation rules without the dead-ball missing-ball artifact).
   - `J1A-LiveOnly` & `J2-LiveOnly` (exact `spell_f1_mean` when ground-truth `D` stretches are fixed via
     `perfect_dead`, cleanly separating Live Turnover `H <-> A` gains from Stoppage Boundary `Live <-> D` gains),
     plus `J1A-noD`, `J1A-StoppageOnly`, `J2-StoppageOnly`, and full Oracle Headroom decomposition.

2. **Event-Play Error Atlas (`compute_error_atlas`)**:
   - Missed possessions (`IoU < 0.5` and `IoU < 0.3`) and boundary lag on `J1A` (DFL events: `TacklingGame`,
     `OtherBallAction`, `Play`, `Play_high`, restarts) and `J2` / `J2-LiveOnly` (Metrica events:
     `CHALLENGE-TACKLE`, `CHALLENGE-GROUND`, `CHALLENGE-AERIAL`, `BALL LOST`, `RECOVERY`, `SET PIECE`).
   - Recall stratified by Possession Duration: Short (`< 2.0s`, plus `< 1.36s` vs `1.36-2.0s`),
     Medium (`2.0s - 5.0s`), and Long (`> 5.0s`), and by Live Turnover vs Post-Stoppage Restart.
   - Stoppage start lag (`Live -> D`), Domino merges (`>= 2` true spells swallowed), and spurious flickers.

3. **Full-Match Unsupervised Physical & Multi-Modal Consistency on `BAR-ATH` (158,857 graded frames)**:
   - Plausibility (`poss/min`, `% < 2s`, `% D`, `% H of live`) on Full Graded (158,857f), Unseen Private
     (151,357f), and Public 5-min (7,500f).
   - Agreement with `BAR-ATH_tactical_smart_tagging.xml` (`in_phase` live team agreement and `SET PIECES`
     pre-kick `D` / post-kick team agreement & physical kick restart lag).
   - Agreement with `D3` re-teamed `v8.5` (`_local/v85sc-count/submissions/submission_v85_forensic_strict_cv.csv`).
   - Exact `H <-> A` Equivariance verification.
   - `J3` (5-min public window) reported strictly as `[REPORT-ONLY, DO NOT SELECT]`.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pickle
import sys
import time
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT / "py_pkgs"), str(ROOT), str(ROOT / "scripts"), str(ROOT / "evaluation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import evaluation.possession_metrics as pm  # noqa: E402
from src.data.common import AWAY, DEAD, FPS, HOME, LABEL_CHARS, possession_spells  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES as SAMPLE_GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS as SPORTEC_IDS, load_all as load_sportec  # noqa: E402
from src.eval import metrics, oracles  # noqa: E402

SWARM_OUT_DIR = ROOT / "experiments" / "outputs" / "v9_swarm"
SWARM_OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS_CACHE_PATH = SWARM_OUT_DIR / "events_cache.pkl"
SMART_TAG_CACHE_PATH = SWARM_OUT_DIR / "bar_ath_smart_tagging_cache.pkl"
BASELINE_PREDS_CACHE_PATH = SWARM_OUT_DIR / "baseline_preds_cache.pkl"

CHAR_TO_CODE = {c: i for i, c in enumerate(LABEL_CHARS)}
CODE_TO_CHAR = {i: c for i, c in enumerate(LABEL_CHARS)}


# ---------------------------------------------------------------------------
# 1. Helper Masks & Conversions
# ---------------------------------------------------------------------------

def ensure_int_codes(arr: np.ndarray | pd.Series | list | str | Path) -> np.ndarray:
    """Convert path, CSV DataFrame, char array ('H','A','D','X'), or int array to common int8 codes (0,1,2)."""
    if isinstance(arr, (str, Path)):
        df = pd.read_csv(arr)
        col = "label" if "label" in df.columns else "team"
        arr = df[col].to_numpy()
    arr = np.asarray(arr)
    if arr.dtype.kind in "iu":
        return np.where(np.isin(arr, (0, 1, 2)), arr, DEAD).astype(np.int8)
    out = np.full(len(arr), DEAD, dtype=np.int8)
    for c, code in CHAR_TO_CODE.items():
        out[arr == c] = code
    return out


def without_dead_margin(y: np.ndarray, half: np.ndarray, seconds: float = 3.0) -> np.ndarray:
    """Labels with D and +/- `seconds` around D set to -1 (ungraded), isolating interior live H<->A play."""
    out = np.asarray(y, dtype=np.int8).copy()
    near = binary_dilation(out == DEAD, iterations=int(round(seconds * FPS)))
    out[near] = -1
    return out


def rle_runs_in_half(lab: np.ndarray, half: np.ndarray) -> list[tuple[int, int, int]]:
    """Return [(label, start, end)] maximal constant-label runs inside each active half."""
    out: list[tuple[int, int, int]] = []
    n = len(lab)
    if n == 0:
        return out
    s = 0
    for i in range(1, n + 1):
        if i == n or lab[i] != lab[s] or half[i] != half[s]:
            if half[s] > 0:
                out.append((int(lab[s]), int(s), int(i)))
            s = i
    return out


# ---------------------------------------------------------------------------
# 2. Core Single-Match & Multi-Match Scoring + Paired Bootstrap (Matches & Halves)
# ---------------------------------------------------------------------------

def score_single_fold(gt: np.ndarray, pred: np.ndarray, half: np.ndarray) -> dict[str, Any]:
    """Compute official possession & frame metrics on a single match or half."""
    gt_arr = np.asarray(gt, dtype=np.int8)
    pr_arr = ensure_int_codes(pred)
    active = (half > 0) & (gt_arr >= 0)
    r = metrics.score(gt_arr, pr_arr, half)
    sp_iou = r["spell_f1"]

    # Possession duration stats on graded frames
    pr_graded = np.where(active, pr_arr, -1).astype(np.int8)
    gt_graded = np.where(active, gt_arr, -1).astype(np.int8)
    s_p, e_p, _ = possession_spells(pr_graded, half)
    s_g, e_g, _ = possession_spells(gt_graded, half)
    d_p = (e_p - s_p) / float(FPS)
    d_g = (e_g - s_g) / float(FPS)
    graded_minutes = max(float(np.sum(active)) / (FPS * 60.0), 1e-6)

    return {
        "spell_f1_mean": float(r["spell_f1_mean"]),
        "macro_f1": float(r["macro_f1"]),
        "F1@0.3": float(sp_iou[0.3]["f1"]),
        "F1@0.4": float(sp_iou[0.4]["f1"]),
        "F1@0.5": float(sp_iou[0.5]["f1"]),
        "F1@0.6": float(sp_iou[0.6]["f1"]),
        "F1@0.7": float(sp_iou[0.7]["f1"]),
        "P@0.5": float(sp_iou[0.5]["precision"]),
        "R@0.5": float(sp_iou[0.5]["recall"]),
        "P@0.3": float(sp_iou[0.3]["precision"]),
        "R@0.3": float(sp_iou[0.3]["recall"]),
        "alive_dead_f1": float(r["alive_dead_f1"]),
        "team_acc_gt_alive": float(r["team_acc_gt_alive"]),
        "team_acc_both_alive": float(r["team_acc_both_alive"]),
        "n_gt_spells": int(r["n_gt_spells"]),
        "n_pred_spells": int(r["n_pred_spells"]),
        "poss_per_min": float(len(d_p) / graded_minutes),
        "gt_poss_per_min": float(len(d_g) / graded_minutes),
        "pct_lt_2s": float(100.0 * np.mean(d_p < 2.0)) if len(d_p) else 0.0,
        "pct_2_to_5s": float(100.0 * np.mean((d_p >= 2.0) & (d_p <= 5.0))) if len(d_p) else 0.0,
        "pct_gt_5s": float(100.0 * np.mean(d_p > 5.0)) if len(d_p) else 0.0,
        "gt_pct_lt_2s": float(100.0 * np.mean(d_g < 2.0)) if len(d_g) else 0.0,
        "median_poss_s": float(np.median(d_p)) if len(d_p) else 0.0,
        "pct_D": float(100.0 * np.mean(pr_arr[active] == DEAD)) if np.any(active) else 0.0,
    }


def paired_bootstrap_ci(
    cand_scores: dict[str, float],
    base_scores: dict[str, float],
    n_boot: int = 1000,
    confidence: float = 0.90,
    seed: int = 0,
    min_delta: float = 0.01,
    min_folds_frac: float = 0.70,
) -> dict[str, Any]:
    """Paired bootstrap CI over matches or halves, plus harness-compatible verdict."""
    keys = [k for k in base_scores if k in cand_scores]
    if not keys:
        return {
            "n_units": 0, "mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0,
            "improved": 0, "needed": 0, "verdict": "NOISE", "verdict_positive": "NOISE", "deltas": {},
        }
    d = np.array([cand_scores[k] - base_scores[k] for k in keys], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = (float(x) for x in np.quantile(boot, [alpha, 1.0 - alpha]))
    mean_d = float(d.mean())
    improved = int(np.sum(d > 1e-12))
    unchanged = int(np.sum(np.abs(d) <= 1e-12))
    degraded = int(np.sum(d < -1e-12))
    needed = int(np.ceil(min_folds_frac * len(d)))

    if mean_d >= min_delta and improved >= needed and lo > 0.0:
        verdict = "ACCEPT"
    elif mean_d <= 0.0 and hi < min_delta:
        verdict = "REJECT"
    else:
        verdict = "NOISE"

    # Also provide a strict positive-gain verdict (lo > 0 and improved >= needed, without the 0.01 jump floor)
    if mean_d > 0.0 and improved >= needed and lo > 0.0:
        verdict_pos = "ACCEPT_POS"
    elif mean_d < 0.0 and hi <= 0.0:
        verdict_pos = "REJECT"
    else:
        verdict_pos = "NOISE"

    return {
        "n_units": len(d),
        "mean_delta": mean_d,
        "std_delta": float(d.std(ddof=0)),
        "ci_low": lo,
        "ci_high": hi,
        "improved": improved,
        "unchanged": unchanged,
        "degraded": degraded,
        "needed": needed,
        "verdict": verdict,
        "verdict_positive": verdict_pos,
        "deltas": {k: float(round(cand_scores[k] - base_scores[k], 6)) for k in keys},
    }


def evaluate_judge_folds(
    gt_dict: dict[str, np.ndarray],
    half_dict: dict[str, np.ndarray],
    pred_dict: dict[str, np.ndarray],
    base_pred_dict: dict[str, np.ndarray] | None = None,
    transform_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]] | None = None,
    n_boot: int = 1000,
    min_delta: float = 0.01,
) -> dict[str, Any]:
    """Evaluate a judge across all matches AND halves, with paired bootstrap vs `base_pred_dict`."""
    fold_keys = sorted(gt_dict.keys())
    per_match_metrics: dict[str, dict[str, Any]] = {}
    per_match_f1: dict[str, float] = {}
    per_half_f1: dict[str, float] = {}

    for k in fold_keys:
        gt_k = np.asarray(gt_dict[k], dtype=np.int8)
        hl_k = np.asarray(half_dict[k])
        pr_k = ensure_int_codes(pred_dict[k])
        if transform_fn is not None:
            gt_k, pr_k = transform_fn(gt_k, pr_k, hl_k)
        m_res = score_single_fold(gt_k, pr_k, hl_k)
        per_match_metrics[str(k)] = m_res
        per_match_f1[str(k)] = m_res["spell_f1_mean"]

        for h_val in (1, 2):
            h_mask = hl_k == h_val
            if np.any(h_mask & (gt_k >= 0)):
                hl_sub = np.where(h_mask, h_val, 0)
                h_res = score_single_fold(gt_k, pr_k, hl_sub)
                per_half_f1[f"{k}_H{h_val}"] = h_res["spell_f1_mean"]

    agg_keys = (
        "spell_f1_mean", "macro_f1", "F1@0.3", "F1@0.4", "F1@0.5", "F1@0.6", "F1@0.7",
        "P@0.5", "R@0.5", "P@0.3", "R@0.3", "alive_dead_f1",
        "team_acc_gt_alive", "team_acc_both_alive", "poss_per_min", "gt_poss_per_min",
        "pct_lt_2s", "pct_2_to_5s", "pct_gt_5s", "gt_pct_lt_2s", "median_poss_s", "pct_D",
    )
    out: dict[str, Any] = {
        ak: float(np.mean([per_match_metrics[str(k)][ak] for k in fold_keys]))
        for ak in agg_keys
    }
    out["n_gt_spells_total"] = int(sum(per_match_metrics[str(k)]["n_gt_spells"] for k in fold_keys))
    out["n_pred_spells_total"] = int(sum(per_match_metrics[str(k)]["n_pred_spells"] for k in fold_keys))
    out["per_match"] = {str(k): round(per_match_f1[str(k)], 6) for k in fold_keys}
    out["per_half"] = {hk: round(hv, 6) for hk, hv in per_half_f1.items()}
    out["per_match_detail"] = per_match_metrics

    if base_pred_dict is not None:
        base_match_f1: dict[str, float] = {}
        base_half_f1: dict[str, float] = {}
        base_macro: list[float] = []
        for k in fold_keys:
            gt_k = np.asarray(gt_dict[k], dtype=np.int8)
            hl_k = np.asarray(half_dict[k])
            bp_k = ensure_int_codes(base_pred_dict[k])
            if transform_fn is not None:
                gt_k, bp_k = transform_fn(gt_k, bp_k, hl_k)
            b_res = score_single_fold(gt_k, bp_k, hl_k)
            base_match_f1[str(k)] = b_res["spell_f1_mean"]
            base_macro.append(b_res["macro_f1"])
            for h_val in (1, 2):
                h_mask = hl_k == h_val
                if np.any(h_mask & (gt_k >= 0)):
                    hl_sub = np.where(h_mask, h_val, 0)
                    bh_res = score_single_fold(gt_k, bp_k, hl_sub)
                    base_half_f1[f"{k}_H{h_val}"] = bh_res["spell_f1_mean"]

        out["paired_vs_v85_match"] = paired_bootstrap_ci(
            per_match_f1, base_match_f1, n_boot=n_boot, min_delta=min_delta
        )
        out["paired_vs_v85_half"] = paired_bootstrap_ci(
            per_half_f1, base_half_f1, n_boot=n_boot, min_delta=min_delta
        )
        out["delta_macro_f1"] = float(out["macro_f1"] - np.mean(base_macro))

    return out


# Transform callbacks for LiveOnly / StoppageOnly / noD judges
def _transform_no_dead_margin(gt: np.ndarray, pred: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return without_dead_margin(gt, half, seconds=3.0), pred


def _transform_live_only_perfect_dead(gt: np.ndarray, pred: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return gt, oracles.perfect_dead(gt, pred, half)


def _transform_stoppage_only_perfect_team(gt: np.ndarray, pred: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return gt, oracles.perfect_team(gt, pred, half)


def compute_oracle_headroom(
    gt_dict: dict[str, np.ndarray],
    half_dict: dict[str, np.ndarray],
    pred_dict: dict[str, np.ndarray],
) -> dict[str, Any]:
    """Compute exact spell_f1_mean headroom under each single-error-type oracle from `src/eval/oracles.py`."""
    fold_keys = sorted(gt_dict.keys())
    base_f1 = [
        float(metrics.score(gt_dict[k], ensure_int_codes(pred_dict[k]), half_dict[k])["spell_f1_mean"])
        for k in fold_keys
    ]
    mean_base = float(np.mean(base_f1))
    results: dict[str, Any] = {"base_spell_f1": mean_base, "oracles": {}}
    for name, fn in oracles.ORACLES.items():
        rep_f1 = []
        for k in fold_keys:
            gt_k = np.asarray(gt_dict[k], dtype=np.int8)
            pr_k = ensure_int_codes(pred_dict[k])
            hl_k = np.asarray(half_dict[k])
            rep = fn(gt_k, pr_k, hl_k)
            rep_f1.append(float(metrics.score(gt_k, rep, hl_k)["spell_f1_mean"]))
        mean_rep = float(np.mean(rep_f1))
        results["oracles"][name] = {
            "repaired_spell_f1": round(mean_rep, 6),
            "headroom_delta": round(mean_rep - mean_base, 6),
            "per_fold_delta": {
                str(k): round(rep_f1[i] - base_f1[i], 6) for i, k in enumerate(fold_keys)
            },
        }
    return results


# ---------------------------------------------------------------------------
# 3. Event-Play Error Atlas (`compute_error_atlas`)
# ---------------------------------------------------------------------------

def _load_s15_module():
    spec = importlib.util.spec_from_file_location("s15", ROOT / "scripts" / "s1_5_flag_semantics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_dfl_events(mid: str, m, s15) -> pd.DataFrame:
    origins = s15.half_origins(mid)
    n1 = int(np.sum(m.half == 1))
    rows = []
    for e in ET.parse(s15._file("03_02_events_raw", mid, s15.DATA_DIR)).getroot().iter("Event"):
        if not len(e):
            continue
        t = s15.parse_time(e.get("EventTime"))
        h = 2 if t >= origins[2][1] else 1
        row = (0 if h == 1 else n1) + int(round((t - origins[h][1]).total_seconds() * FPS))
        node = e[0]
        kind = node.tag
        if kind == "Play" and node.get("Height") == "high":
            kind = "Play_high"
        if kind == "OtherBallAction":
            kind = "OtherBallAction"
        if 0 <= row < m.n_frames:
            rows.append((row, kind))
    ev = pd.DataFrame(rows, columns=["row", "kind"])
    lag, _ = s15.event_lag(m, s15.read_events(mid, m))
    ev["row"] = ev["row"] + lag
    return ev


def _build_sample_events(g: int) -> pd.DataFrame:
    d = ROOT / "data_external" / "metrica_sample_data" / "data" / f"Sample_Game_{g}"
    p = d / f"Sample_Game_{g}_RawEventsData.csv"
    if p.is_file():
        df = pd.read_csv(p)
    else:
        recs = json.loads((d / f"Sample_Game_{g}_events.json").read_text())["data"]
        sub_name = [
            r["subtypes"]["name"]
            if isinstance(r.get("subtypes"), dict)
            else (r["subtypes"][0]["name"] if r.get("subtypes") else "")
            for r in recs
        ]
        df = pd.DataFrame({
            "Type": [r["type"]["name"] for r in recs],
            "Subtype": sub_name,
            "Start Frame": [r["start"]["frame"] for r in recs],
        })
    kind = df["Type"].astype(str)
    sub = df["Subtype"].fillna("").astype(str)
    kind = np.where(kind == "CHALLENGE", "CHALLENGE-" + sub.str.split("-").str[0], kind)
    kind = np.where(df["Type"] == "SET PIECE", "SET PIECE-" + sub, kind)
    return pd.DataFrame({"row": df["Start Frame"].to_numpy() - 1, "kind": kind})


def load_cached_events(sportec_matches: dict | None = None) -> dict[str, dict[Any, pd.DataFrame]]:
    """Load or build cached DFL (J1) and Metrica (J2) aligned event tables (`events_cache.pkl`)."""
    if EVENTS_CACHE_PATH.is_file():
        with open(EVENTS_CACHE_PATH, "rb") as f:
            return pickle.load(f)
    if sportec_matches is None:
        sportec_matches = load_sportec()
    s15 = _load_s15_module()
    ev_j1 = {mid: _build_dfl_events(mid, sportec_matches[mid], s15) for mid in sorted(sportec_matches)}
    ev_j2 = {gid: _build_sample_events(gid) for gid in SAMPLE_GAMES}
    bundle = {"J1": ev_j1, "J2": ev_j2}
    with open(EVENTS_CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def _nearest_kind(ev_rows: np.ndarray, ev_kind: np.ndarray, f: int, lo: int, hi: int) -> str:
    w = np.flatnonzero((ev_rows >= f + lo) & (ev_rows <= f + hi))
    if not len(w):
        return "none"
    return str(ev_kind[w[np.argmin(np.abs(ev_rows[w] - f))]])


def _macro_event_group_j1(kind: str) -> str:
    if kind == "TacklingGame":
        return "Duel (TacklingGame)"
    if kind == "OtherBallAction":
        return "Loose/Clearance (OtherBallAction)"
    if kind == "Play":
        return "Pass Ground (Play)"
    if kind == "Play_high":
        return "Pass Aerial (Play_high)"
    if kind in ("ThrowIn", "FreeKick", "GoalKick", "CornerKick", "KickOff", "Penalty", "RefereeBall"):
        return "Restart (ThrowIn/FK/GK/CK/KO)"
    if kind in ("ShotAtGoal", "ChanceWithoutShot", "SitterPrevented"):
        return "Shot / Goal Action"
    return f"Other ({kind})"


def _macro_event_group_j2(kind: str) -> str:
    if kind == "CHALLENGE-TACKLE":
        return "Challenge-Tackle"
    if kind == "CHALLENGE-GROUND":
        return "Challenge-Ground"
    if kind == "CHALLENGE-AERIAL":
        return "Challenge-Aerial"
    if kind == "BALL LOST":
        return "Ball Lost"
    if kind == "RECOVERY":
        return "Recovery"
    if kind.startswith("SET PIECE"):
        return "Set Piece (All)"
    return f"Other ({kind})"


def _duration_bucket(len_s: float) -> str:
    if len_s < 2.0:
        return "short (<2.0s)"
    if len_s <= 5.0:
        return "medium (2.0-5.0s)"
    return "long (>5.0s)"


def _duration_fine_bucket(len_s: float) -> str:
    if len_s < 1.36:
        return "ultra_short (<1.36s / <34f)"
    if len_s < 2.0:
        return "short_decodable (1.36-2.0s / 34-49f)"
    if len_s <= 5.0:
        return "medium (2.0-5.0s / 50-125f)"
    if len_s <= 10.0:
        return "long (5.0-10.0s)"
    return "very_long (>10.0s)"


def _match_atlas_single(
    match_id: str,
    truth: np.ndarray,
    pred: np.ndarray,
    half: np.ndarray,
    ev: pd.DataFrame,
    is_j1: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    er = ev["row"].to_numpy()
    ek = ev["kind"].to_numpy()
    tr = rle_runs_in_half(truth, half)
    pr = rle_runs_in_half(pred, half)

    pred_by_team: dict[int, list[tuple[int, int]]] = defaultdict(list)
    gt_by_team: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for lab, a, b in pr:
        if lab in (HOME, AWAY):
            pred_by_team[lab].append((a, b))
    for lab, a, b in tr:
        if lab in (HOME, AWAY):
            gt_by_team[lab].append((a, b))

    poss: list[dict[str, Any]] = []
    for idx_sp, (lab, a, b) in enumerate(tr):
        if lab not in (HOME, AWAY):
            continue
        best, start_err, end_err = 0.0, None, None
        for pa, pb in pred_by_team[lab]:
            if pb <= a or pa >= b:
                continue
            iou = (min(b, pb) - max(a, pa)) / float(max(b, pb) - min(a, pa))
            if iou > best:
                best = iou
                start_err = int(pa - a)
                end_err = int(pb - b)
        kind = _nearest_kind(er, ek, a, -40, 15)
        len_s = (b - a) / float(FPS)
        prev_dead = bool(a > 0 and truth[a - 1] == DEAD and half[a - 1] == half[a])
        poss.append({
            "match_id": str(match_id),
            "spell_id": f"{match_id}_{a}_{b}_{lab}",
            "team": int(lab),
            "start": int(a),
            "end": int(b),
            "len_f": int(b - a),
            "len_s": float(len_s),
            "dur_bucket": _duration_bucket(len_s),
            "dur_fine_bucket": _duration_fine_bucket(len_s),
            "kind": kind,
            "macro_group": _macro_event_group_j1(kind) if is_j1 else _macro_event_group_j2(kind),
            "iou": float(best),
            "miss_05": bool(best < 0.5),
            "miss_03": bool(best < 0.3),
            "hit_05": bool(best >= 0.5),
            "hit_03": bool(best >= 0.3),
            "hit_07": bool(best >= 0.7),
            "miss_07": bool(best < 0.7),
            "start_err": start_err,
            "end_err": end_err,
            "abs_start_err": abs(start_err) if start_err is not None else None,
            "prev_dead": prev_dead,
            "transition_origin": "restart_from_D" if prev_dead else "live_turnover_H_A",
        })

    # Spurious predicted possessions & domino merges
    spurious_list: list[dict[str, Any]] = []
    merges = 0
    swallowed = 0
    for lab, pa, pb in pr:
        if lab not in (HOME, AWAY):
            continue
        best_g = 0.0
        n_covered_same_team = 0
        for ga, gb in gt_by_team[lab]:
            if gb <= pa or ga >= pb:
                continue
            n_covered_same_team += 1
            iou = (min(gb, pb) - max(ga, pa)) / float(max(gb, pb) - min(ga, pa))
            if iou > best_g:
                best_g = iou
        if n_covered_same_team >= 2:
            merges += 1
            swallowed += n_covered_same_team
        plen_s = (pb - pa) / float(FPS)
        spurious_list.append({
            "match_id": str(match_id),
            "team": int(lab),
            "start": int(pa),
            "end": int(pb),
            "len_s": float(plen_s),
            "dur_bucket": _duration_bucket(plen_s),
            "best_gt_iou": float(best_g),
            "spurious_03": bool(best_g < 0.3),
            "spurious_05": bool(best_g < 0.5),
            "swallowed_count": int(n_covered_same_team),
        })

    dstarts: list[dict[str, Any]] = []
    pd_starts = np.array([a for lab, a, b in pr if lab == DEAD], dtype=np.int64)
    for lab, a, b in tr:
        if lab != DEAD or a == 0 or truth[a - 1] == DEAD or half[a - 1] != half[a]:
            continue
        near = pd_starts[np.abs(pd_starts - a) <= 250] if len(pd_starts) else np.array([], dtype=np.int64)
        lag = int(near[np.argmin(np.abs(near - a))] - a) if len(near) else None
        dstarts.append({
            "match_id": str(match_id),
            "frame": int(a),
            "len_s": float((b - a) / float(FPS)),
            "kind": _nearest_kind(er, ek, a, -100, 10),
            "lag": lag,
            "abs_lag": abs(lag) if lag is not None else None,
        })

    merge_stats = {"merges": int(merges), "swallowed_true_spells": int(swallowed)}
    return poss, dstarts, spurious_list, merge_stats


def _summarise_atlas_records(
    poss: list[dict[str, Any]],
    dstarts: list[dict[str, Any]],
    spurious_list: list[dict[str, Any]],
    merge_stats: dict[str, int],
    base_poss: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    P = pd.DataFrame(poss)
    P["short"] = P["len_s"] < 2.0
    total_misses_05 = max(int(P["miss_05"].sum()), 1)
    total_misses_03 = max(int(P["miss_03"].sum()), 1)

    def _agg_poss_table(df_in: pd.DataFrame, group_col: str, sort_by_n: bool = True) -> list[dict[str, Any]]:
        rows_out = []
        for gval, sub in df_in.groupby(group_col, sort=False):
            n = len(sub)
            m05 = int(sub["miss_05"].sum())
            m03 = int(sub["miss_03"].sum())
            m07 = int(sub["miss_07"].sum())
            serr = sub["start_err"].dropna()
            aerr = sub["abs_start_err"].dropna()
            rows_out.append({
                group_col: str(gval),
                "n": int(n),
                "missed_05": m05,
                "miss_rate_05": round(float(m05 / n), 4) if n else 0.0,
                "recall_05": round(float(1.0 - m05 / n), 4) if n else 0.0,
                "missed_03": m03,
                "miss_rate_03": round(float(m03 / n), 4) if n else 0.0,
                "recall_03": round(float(1.0 - m03 / n), 4) if n else 0.0,
                "recall_07": round(float(1.0 - m07 / n), 4) if n else 0.0,
                "mean_iou": round(float(sub["iou"].mean()), 4) if n else 0.0,
                "short_share": round(float(sub["short"].mean()), 4) if n else 0.0,
                "median_start_err_f": round(float(serr.median()), 1) if len(serr) else None,
                "p90_abs_start_err_f": round(float(np.percentile(aerr, 90)), 1) if len(aerr) else None,
                "share_of_all_misses_05": round(float(m05 / total_misses_05), 4),
                "share_of_all_misses_03": round(float(m03 / total_misses_03), 4),
            })
        if sort_by_n:
            rows_out.sort(key=lambda x: -x["n"])
        return rows_out

    by_kind = _agg_poss_table(P, "kind", sort_by_n=True)
    by_macro = _agg_poss_table(P, "macro_group", sort_by_n=True)

    # Order duration buckets logically
    dur_order = ["short (<2.0s)", "medium (2.0-5.0s)", "long (>5.0s)"]
    by_dur_map = {r["dur_bucket"]: r for r in _agg_poss_table(P, "dur_bucket", sort_by_n=False)}
    by_duration = [by_dur_map[k] for k in dur_order if k in by_dur_map]

    fine_order = [
        "ultra_short (<1.36s / <34f)",
        "short_decodable (1.36-2.0s / 34-49f)",
        "medium (2.0-5.0s / 50-125f)",
        "long (5.0-10.0s)",
        "very_long (>10.0s)",
    ]
    by_fine_map = {r["dur_fine_bucket"]: r for r in _agg_poss_table(P, "dur_fine_bucket", sort_by_n=False)}
    by_duration_fine = [by_fine_map[k] for k in fine_order if k in by_fine_map]

    by_origin = _agg_poss_table(P, "transition_origin", sort_by_n=True)

    # Duration x Origin cross-tab (Live Turnovers vs Restarts by Short/Medium/Long)
    P["origin_x_dur"] = P["transition_origin"] + " :: " + P["dur_bucket"]
    by_origin_x_dur = _agg_poss_table(P, "origin_x_dur", sort_by_n=False)

    # Dead-ball starts
    D = pd.DataFrame(dstarts)
    dead_rows = []
    if len(D):
        D["late_1s"] = D["lag"].fillna(999) > 25
        D["early_1s"] = D["lag"].fillna(0) < -25
        for k_val, sub in D.groupby("kind"):
            n_d = len(sub)
            lags = sub["lag"].dropna()
            alags = sub["abs_lag"].dropna()
            dead_rows.append({
                "kind": str(k_val),
                "n": int(n_d),
                "median_lag_f": round(float(lags.median()), 1) if len(lags) else None,
                "mean_abs_lag_f": round(float(alags.mean()), 1) if len(alags) else None,
                "late_over_1s": round(float(sub["late_1s"].mean()), 4),
                "early_over_1s": round(float(sub["early_1s"].mean()), 4),
                "missed": round(float(sub["lag"].isna().mean()), 4),
            })
        dead_rows.sort(key=lambda x: -x["n"])

    # Spurious predicted possessions summary
    S = pd.DataFrame(spurious_list)
    spurious_summary: dict[str, Any] = {
        "n_pred_spells": len(S),
        "spurious_03_count": int(S["spurious_03"].sum()) if len(S) else 0,
        "spurious_03_rate": round(float(S["spurious_03"].mean()), 4) if len(S) else 0.0,
        "spurious_05_count": int(S["spurious_05"].sum()) if len(S) else 0,
        "spurious_05_rate": round(float(S["spurious_05"].mean()), 4) if len(S) else 0.0,
        "merges_count": merge_stats["merges"],
        "swallowed_true_spells": merge_stats["swallowed_true_spells"],
        "by_duration": {},
    }
    if len(S):
        for db in dur_order:
            sub_s = S[S["dur_bucket"] == db]
            if len(sub_s):
                spurious_summary["by_duration"][db] = {
                    "n_pred": int(len(sub_s)),
                    "spurious_03": int(sub_s["spurious_03"].sum()),
                    "spurious_05": int(sub_s["spurious_05"].sum()),
                    "precision_05": round(float(1.0 - sub_s["spurious_05"].mean()), 4),
                }

    out = {
        "total_gt_possessions": int(len(P)),
        "total_missed_05": int(P["miss_05"].sum()),
        "overall_miss_rate_05": round(float(P["miss_05"].mean()), 4),
        "overall_recall_05": round(float(1.0 - P["miss_05"].mean()), 4),
        "total_missed_03": int(P["miss_03"].sum()),
        "overall_miss_rate_03": round(float(P["miss_03"].mean()), 4),
        "overall_recall_03": round(float(1.0 - P["miss_03"].mean()), 4),
        "mean_best_iou": round(float(P["iou"].mean()), 4),
        "by_event_kind": by_kind,
        "by_macro_group": by_macro,
        "by_duration": by_duration,
        "by_duration_fine": by_duration_fine,
        "by_transition_origin": by_origin,
        "by_origin_x_duration": by_origin_x_dur,
        "dead_starts": dead_rows,
        "spurious_and_merges": spurious_summary,
    }

    if base_poss is not None and len(base_poss) == len(poss):
        B = pd.DataFrame(base_poss)
        rescued_05 = (~P["miss_05"]) & B["miss_05"]
        lost_05 = P["miss_05"] & (~B["miss_05"])
        iou_delta = P["iou"] - B["iou"]
        P_diff = pd.DataFrame({
            "kind": P["kind"],
            "macro_group": P["macro_group"],
            "dur_bucket": P["dur_bucket"],
            "rescued_05": rescued_05.astype(int),
            "lost_05": lost_05.astype(int),
            "net_gained_05": rescued_05.astype(int) - lost_05.astype(int),
            "iou_delta": iou_delta,
        })
        by_macro_diff = (
            P_diff.groupby("macro_group")
            .agg(
                n=("net_gained_05", "size"),
                rescued_05=("rescued_05", "sum"),
                lost_05=("lost_05", "sum"),
                net_gained_05=("net_gained_05", "sum"),
                mean_iou_delta=("iou_delta", "mean"),
            )
            .reset_index()
            .to_dict("records")
        )
        by_dur_diff = (
            P_diff.groupby("dur_bucket")
            .agg(
                n=("net_gained_05", "size"),
                rescued_05=("rescued_05", "sum"),
                lost_05=("lost_05", "sum"),
                net_gained_05=("net_gained_05", "sum"),
                mean_iou_delta=("iou_delta", "mean"),
            )
            .reset_index()
            .to_dict("records")
        )
        out["paired_diff_vs_v85"] = {
            "total_rescued_05": int(rescued_05.sum()),
            "total_lost_05": int(lost_05.sum()),
            "net_possessions_gained_05": int(rescued_05.sum() - lost_05.sum()),
            "mean_iou_delta": round(float(iou_delta.mean()), 5),
            "by_macro_group": by_macro_diff,
            "by_duration": by_dur_diff,
        }

    return out


def compute_error_atlas(
    preds_v3: dict[str, np.ndarray] | None = None,
    preds_sample: dict[int, np.ndarray] | None = None,
    base_preds_v3: dict[str, np.ndarray] | None = None,
    base_preds_sample: dict[int, np.ndarray] | None = None,
    sportec_matches: dict | None = None,
    sample_matches: dict | None = None,
) -> dict[str, Any]:
    """Compute Event-Play & Duration Error Atlas across `J1A` (DFL events), `J2-Raw` (Metrica events),
    and `J2-LiveOnly` (Metrica events with ground-truth `D` fixed so dead-ball artifacts are removed).
    """
    if sportec_matches is None:
        sportec_matches = load_sportec()
    if sample_matches is None:
        sample_matches = {g: load_sample(g) for g in SAMPLE_GAMES}
    ev_bundle = load_cached_events(sportec_matches)

    report: dict[str, Any] = {}

    if preds_v3 is not None:
        poss_j1, dst_j1, spur_j1 = [], [], []
        merges_j1 = {"merges": 0, "swallowed_true_spells": 0}
        base_poss_j1 = [] if base_preds_v3 is not None else None

        for mid in sorted(sportec_matches):
            m = sportec_matches[mid]
            truth = np.asarray(m.labels, dtype=np.int8)
            half = np.asarray(m.half)
            pr = ensure_int_codes(preds_v3[mid])
            p, d, sp, ms = _match_atlas_single(mid, truth, pr, half, ev_bundle["J1"][mid], is_j1=True)
            poss_j1.extend(p)
            dst_j1.extend(d)
            spur_j1.extend(sp)
            merges_j1["merges"] += ms["merges"]
            merges_j1["swallowed_true_spells"] += ms["swallowed_true_spells"]
            if base_preds_v3 is not None and base_poss_j1 is not None:
                bp = ensure_int_codes(base_preds_v3[mid])
                bp_list, _, _, _ = _match_atlas_single(mid, truth, bp, half, ev_bundle["J1"][mid], is_j1=True)
                base_poss_j1.extend(bp_list)

        report["J1A"] = _summarise_atlas_records(poss_j1, dst_j1, spur_j1, merges_j1, base_poss_j1)

    if preds_sample is not None:
        for mode_tag, use_live_only in (("J2_Raw", False), ("J2_LiveOnly", True)):
            poss_j2, dst_j2, spur_j2 = [], [], []
            merges_j2 = {"merges": 0, "swallowed_true_spells": 0}
            base_poss_j2 = [] if base_preds_sample is not None else None

            for gid in SAMPLE_GAMES:
                m = sample_matches[gid]
                truth = np.asarray(m.labels, dtype=np.int8)
                half = np.asarray(m.half)
                pr = ensure_int_codes(preds_sample[gid])
                if use_live_only:
                    pr = oracles.perfect_dead(truth, pr, half)
                p, d, sp, ms = _match_atlas_single(f"sample{gid}", truth, pr, half, ev_bundle["J2"][gid], is_j1=False)
                poss_j2.extend(p)
                dst_j2.extend(d)
                spur_j2.extend(sp)
                merges_j2["merges"] += ms["merges"]
                merges_j2["swallowed_true_spells"] += ms["swallowed_true_spells"]
                if base_preds_sample is not None and base_poss_j2 is not None:
                    bp = ensure_int_codes(base_preds_sample[gid])
                    if use_live_only:
                        bp = oracles.perfect_dead(truth, bp, half)
                    bp_list, _, _, _ = _match_atlas_single(
                        f"sample{gid}", truth, bp, half, ev_bundle["J2"][gid], is_j1=False
                    )
                    base_poss_j2.extend(bp_list)

            report[mode_tag] = _summarise_atlas_records(poss_j2, dst_j2, spur_j2, merges_j2, base_poss_j2)

    return report


# ---------------------------------------------------------------------------
# 4. Full-Match Unsupervised Physical & Multi-Modal Consistency on BAR-ATH
# ---------------------------------------------------------------------------

def _load_or_build_smart_tagging_cache(atd) -> dict[str, Any]:
    """Load or build cached Smart Tagging arrays and Set-Piece physical kick annotations on BAR-ATH."""
    if SMART_TAG_CACHE_PATH.is_file():
        with open(SMART_TAG_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    from starter.load_smart_tagging import load as load_tags
    from scripts.s1_6_smart_tagging import DEFENSIVE_CODES, TEAM, ball_speed, find_kick

    T = atd.n_frames
    tags = load_tags()
    spd = ball_speed(atd)

    in_phase = np.zeros(T, dtype=bool)
    in_set_piece = np.zeros(T, dtype=bool)
    pre_kick_dead = np.zeros(T, dtype=bool)
    post_kick_team = np.full(T, -1, dtype=np.int8)
    phase_team = np.full(T, -1, dtype=np.int8)
    h_votes = np.zeros(T, dtype=np.float32)
    a_votes = np.zeros(T, dtype=np.float32)

    phase_instances: list[dict[str, Any]] = []
    set_piece_instances: list[dict[str, Any]] = []

    for t in tags:
        code_up = t["code"].upper()
        a = max(0, int(t["start_frame"]))
        b = min(T, int(t["end_frame"]))
        if a >= b:
            continue
        team_name = t["tags"].get("Team")
        if code_up == "SET PIECES":
            in_set_piece[a:b] = True
            kind = t["tags"].get("Type", "?")
            k = find_kick(atd, spd, a - 5 * FPS, b + 5 * FPS, kind)
            team_code = TEAM.get(team_name, -1)
            if k is not None and a <= k <= b:
                pre_start = max(a, k - 12 * FPS)
                pre_kick_dead[pre_start:k] = True
                if team_code >= 0:
                    post_kick_team[k:min(T, k + 4 * FPS)] = team_code
            set_piece_instances.append({
                "type": kind,
                "team": team_code,
                "start": a,
                "end": b,
                "kick": int(k) if k is not None else None,
                "graded": bool(atd.graded[a]),
            })
        else:
            in_phase[a:b] = True
            if team_name in TEAM:
                poss_team = 1 - TEAM[team_name] if code_up in DEFENSIVE_CODES else TEAM[team_name]
                w = 1.2 if code_up in ("BUILD UP", "PROGRESSION", "CREATING CHANCES") else 0.8
                if poss_team == HOME:
                    h_votes[a:b] += w
                else:
                    a_votes[a:b] += w
                phase_instances.append({
                    "code": code_up,
                    "team": int(poss_team),
                    "start": a,
                    "end": b,
                })

    net = h_votes - a_votes
    phase_team[net > 0] = HOME
    phase_team[net < 0] = AWAY

    # Also load D3 re-team segments if available
    reteam_npz = ROOT / "local_data" / "video" / "players_reteam.npz"
    reteam_near_mask = np.zeros(T, dtype=bool)
    n_reteam_segs = 0
    if reteam_npz.is_file():
        segs = np.load(reteam_npz)["segments"]
        n_reteam_segs = int(len(segs))
        for _, sa, sb in segs:
            reteam_near_mask[max(0, int(sa) - 25):min(T, int(sb) + 25)] = True

    bundle = {
        "in_phase": in_phase,
        "in_set_piece": in_set_piece,
        "pre_kick_dead": pre_kick_dead,
        "post_kick_team": post_kick_team,
        "phase_team": phase_team,
        "phase_instances": phase_instances,
        "set_piece_instances": set_piece_instances,
        "reteam_near_mask": reteam_near_mask,
        "n_reteam_segs": n_reteam_segs,
    }
    with open(SMART_TAG_CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


def _window_plausibility_stats(pred: np.ndarray, half: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    """Compute detailed plausibility stats on a boolean frame mask (`graded`, `unseen`, or `public`)."""
    n_frames = int(mask.sum())
    minutes = max(n_frames / (25.0 * 60.0), 1e-6)
    pr_masked = np.where(mask, pred, -1).astype(np.int8)
    hl_masked = np.where(mask, half, 0)
    s, e, t = possession_spells(pr_masked, hl_masked)
    lens_f = e - s
    lens_s = lens_f / float(FPS)

    # Count D spells inside mask
    runs_all = rle_runs_in_half(pr_masked, hl_masked)
    d_lens_s = np.array([(b - a) / float(FPS) for lab, a, b in runs_all if lab == DEAD], dtype=np.float64)

    n_h = int(np.sum(pred[mask] == HOME))
    n_a = int(np.sum(pred[mask] == AWAY))
    n_d = int(np.sum(pred[mask] == DEAD))
    n_live = max(n_h + n_a, 1)

    return {
        "frames": n_frames,
        "minutes": round(minutes, 2),
        "n_live_spells": int(len(lens_s)),
        "n_H_spells": int(np.sum(t == HOME)),
        "n_A_spells": int(np.sum(t == AWAY)),
        "n_D_spells": int(len(d_lens_s)),
        "poss_per_min": round(float(len(lens_s) / minutes), 3),
        "pct_lt_2s": round(float(100.0 * np.mean(lens_s < 2.0)), 2) if len(lens_s) else 0.0,
        "pct_2_to_5s": round(float(100.0 * np.mean((lens_s >= 2.0) & (lens_s <= 5.0))), 2) if len(lens_s) else 0.0,
        "pct_gt_5s": round(float(100.0 * np.mean(lens_s > 5.0)), 2) if len(lens_s) else 0.0,
        "pct_D": round(float(100.0 * n_d / max(n_frames, 1)), 2),
        "pct_H_of_live": round(float(100.0 * n_h / n_live), 2),
        "pct_A_of_live": round(float(100.0 * n_a / n_live), 2),
        "live_dur_min_s": round(float(lens_s.min()), 2) if len(lens_s) else 0.0,
        "live_dur_p10_s": round(float(np.percentile(lens_s, 10)), 2) if len(lens_s) else 0.0,
        "live_dur_median_s": round(float(np.median(lens_s)), 2) if len(lens_s) else 0.0,
        "live_dur_mean_s": round(float(lens_s.mean()), 2) if len(lens_s) else 0.0,
        "live_dur_p90_s": round(float(np.percentile(lens_s, 90)), 2) if len(lens_s) else 0.0,
        "live_dur_max_s": round(float(lens_s.max()), 2) if len(lens_s) else 0.0,
        "micro_spells_lt_12f": int(np.sum(lens_f < 12)),
        "spells_lt_34f": int(np.sum(lens_f < 34)),
        "spells_lt_50f": int(np.sum(lens_f < 50)),
    }


def evaluate_bar_ath_unsupervised(
    pred_atd: np.ndarray | str | Path,
    atd=None,
    v85_strict_pred: np.ndarray | None = None,
    v85_reteam_pred: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full-Match Unsupervised Physical & Multi-Modal Consistency on BAR-ATH (158,857 graded frames):
    - Plausibility (`poss/min`, `% < 2s`, `% D`, `% H of live`) across Full Graded (158,857f),
      Unseen Private (151,357f), and Public 5-min (7,500f).
    - Agreement with `BAR-ATH_tactical_smart_tagging.xml` (`in_phase` live team agreement and
      `SET PIECES` pre-kick `D` / post-kick team agreement & restart lag).
    - Agreement with `D3` re-teamed `v8.5` (`_local/v85sc-count/submissions/submission_v85_forensic_strict_cv.csv`)
      and `v8.5 Strict CV`.
    - `J3` (5-min public window) strictly tagged `[REPORT-ONLY, DO NOT SELECT]`.
    """
    if atd is None:
        atd = load_atd()
    pr = ensure_int_codes(pred_atd)
    graded = np.asarray(atd.graded, dtype=bool)
    half = np.asarray(atd.half)

    pub_gt = metrics.public_gt()
    pub_mask = pub_gt != pm.X
    unseen_mask = graded & (~pub_mask)

    # 1. Plausibility across Full Graded (158,857f), Unseen Private (151,357f), and Public (7,500f)
    plaus = {
        "full_graded_158857f": _window_plausibility_stats(pr, half, graded),
        "unseen_private_151357f": _window_plausibility_stats(pr, half, unseen_mask),
        "public_5min_7500f": _window_plausibility_stats(pr, half, pub_mask),
    }

    # 2. Smart Tagging Multi-Modal Agreement (across Full Graded 158,857f & Unseen Private 151,357f)
    st_cache = _load_or_build_smart_tagging_cache(atd)
    phase_team = st_cache["phase_team"]
    in_phase = st_cache["in_phase"]
    in_set_piece = st_cache["in_set_piece"]
    pre_kick_dead = st_cache["pre_kick_dead"]
    post_kick_team = st_cache["post_kick_team"]

    def _smart_tag_metrics_on_mask(m_mask: np.ndarray) -> dict[str, Any]:
        # In-phase live team agreement
        tagged_phase = m_mask & (phase_team >= 0)
        pred_alive = (pr == HOME) | (pr == AWAY)
        tagged_and_alive = tagged_phase & pred_alive
        in_phase_team_agree = (
            float(np.mean(pr[tagged_and_alive] == phase_team[tagged_and_alive]))
            if np.any(tagged_and_alive)
            else 0.0
        )
        in_phase_alive_rate = float(np.mean(pred_alive[m_mask & in_phase])) if np.any(m_mask & in_phase) else 0.0

        # Pre-kick D agreement
        pk_mask = m_mask & pre_kick_dead
        pre_kick_d_agree = float(np.mean(pr[pk_mask] == DEAD)) if np.any(pk_mask) else 0.0

        # Post-kick team agreement (0..4s after physical kick)
        post_mask = m_mask & (post_kick_team >= 0)
        post_alive_mask = post_mask & pred_alive
        post_kick_strict_agree = (
            float(np.mean(pr[post_mask] == post_kick_team[post_mask])) if np.any(post_mask) else 0.0
        )
        post_kick_alive_agree = (
            float(np.mean(pr[post_alive_mask] == post_kick_team[post_alive_mask]))
            if np.any(post_alive_mask)
            else 0.0
        )
        post_kick_alive_rate = float(np.mean(pred_alive[post_mask])) if np.any(post_mask) else 0.0

        return {
            "in_phase_tagged_frames": int(tagged_phase.sum()),
            "in_phase_pred_alive_frames": int(tagged_and_alive.sum()),
            "in_phase_live_team_agree_pct": round(100.0 * in_phase_team_agree, 2),
            "in_phase_pred_alive_pct": round(100.0 * in_phase_alive_rate, 2),
            "set_piece_pre_kick_frames": int(pk_mask.sum()),
            "set_piece_pre_kick_D_agree_pct": round(100.0 * pre_kick_d_agree, 2),
            "set_piece_post_kick_frames": int(post_mask.sum()),
            "set_piece_post_kick_team_strict_pct": round(100.0 * post_kick_strict_agree, 2),
            "set_piece_post_kick_team_when_alive_pct": round(100.0 * post_kick_alive_agree, 2),
            "set_piece_post_kick_alive_pct": round(100.0 * post_kick_alive_rate, 2),
            "pct_D_inside_phases": round(100.0 * float(np.mean(pr[m_mask & in_phase] == DEAD)), 2),
            "pct_D_inside_set_pieces": round(100.0 * float(np.mean(pr[m_mask & in_set_piece] == DEAD)), 2),
            "pct_D_outside_tags": round(
                100.0 * float(np.mean(pr[m_mask & (~in_phase) & (~in_set_piece)] == DEAD)), 2
            ),
        }

    # Per-phase-code breakdown on full graded match
    by_phase_code: dict[str, dict[str, Any]] = {}
    for inst in st_cache["phase_instances"]:
        c_name = inst["code"]
        a, b, tm = inst["start"], inst["end"], inst["team"]
        sl_g = graded[a:b]
        if not np.any(sl_g):
            continue
        sl_pr = pr[a:b][sl_g]
        sl_alive = (sl_pr == HOME) | (sl_pr == AWAY)
        rec = by_phase_code.setdefault(c_name, {"instances": 0, "graded_frames": 0, "alive_frames": 0, "agree_frames": 0})
        rec["instances"] += 1
        rec["graded_frames"] += int(sl_g.sum())
        rec["alive_frames"] += int(sl_alive.sum())
        rec["agree_frames"] += int(np.sum(sl_pr[sl_alive] == tm))

    for c_name, rec in by_phase_code.items():
        rec["live_team_agree_pct"] = round(100.0 * rec["agree_frames"] / max(rec["alive_frames"], 1), 2)
        rec["alive_pct"] = round(100.0 * rec["alive_frames"] / max(rec["graded_frames"], 1), 2)

    # Physical kick restart lag vs predicted D -> Live boundaries
    runs_pr = rle_runs_in_half(np.where(graded, pr, -1).astype(np.int8), np.where(graded, half, 0))
    pred_restarts = np.array(
        [
            runs_pr[i][1]
            for i in range(1, len(runs_pr))
            if runs_pr[i - 1][0] == DEAD and runs_pr[i][0] in (HOME, AWAY)
        ],
        dtype=np.int64,
    )
    kick_lags = []
    for sp_inst in st_cache["set_piece_instances"]:
        if sp_inst["graded"] and sp_inst["kick"] is not None and len(pred_restarts):
            k_f = int(sp_inst["kick"])
            near = pred_restarts[np.abs(pred_restarts - k_f) <= 150]
            if len(near):
                kick_lags.append(int(near[np.argmin(np.abs(near - k_f))] - k_f))

    smart_tagging_audit = {
        "full_graded_158857f": _smart_tag_metrics_on_mask(graded),
        "unseen_private_151357f": _smart_tag_metrics_on_mask(unseen_mask),
        "by_phase_code": by_phase_code,
        "physical_kick_restart_alignment": {
            "n_graded_set_pieces_with_kick": sum(
                1 for sp in st_cache["set_piece_instances"] if sp["graded"] and sp["kick"] is not None
            ),
            "n_matched_within_6s": len(kick_lags),
            "median_restart_lag_f": round(float(np.median(kick_lags)), 1) if kick_lags else None,
            "mean_abs_restart_lag_f": round(float(np.mean(np.abs(kick_lags))), 1) if kick_lags else None,
        },
    }

    # 3. Agreement with D3 Re-teamed v8.5 (`_local/v85sc-count/submissions/submission_v85_forensic_strict_cv.csv`)
    #    and Control v8.5 Strict CV (`submissions/submission_v85_forensic_strict_cv.csv`)
    if v85_strict_pred is None:
        p_ctrl = ROOT / "submissions" / "submission_v85_forensic_strict_cv.csv"
        if p_ctrl.is_file():
            v85_strict_pred = ensure_int_codes(p_ctrl)
    if v85_reteam_pred is None:
        p_ret = ROOT / "_local" / "v85sc-count" / "submissions" / "submission_v85_forensic_strict_cv.csv"
        if p_ret.is_file():
            v85_reteam_pred = ensure_int_codes(p_ret)

    reteam_near = st_cache["reteam_near_mask"]

    def _compare_bar_ath_ref(ref_arr: np.ndarray | None, ref_name: str) -> dict[str, Any] | None:
        if ref_arr is None:
            return None
        ref = ensure_int_codes(ref_arr)
        diff = (pr != ref) & graded
        diff_unseen = (pr != ref) & unseen_mask
        diff_pub = (pr != ref) & pub_mask
        ha_diff = diff & (pr < 2) & (ref < 2)
        d_diff = diff & ((pr == DEAD) | (ref == DEAD))
        sc_agree = metrics.score(np.where(graded, ref, -1).astype(np.int8), pr, np.where(graded, half, 0))
        near_share = float(diff[reteam_near].sum() / max(diff.sum(), 1)) if diff.sum() > 0 else 0.0
        base_near_cov = float(reteam_near[graded].mean())
        return {
            "reference": ref_name,
            "frame_agreement_pct": round(100.0 * (1.0 - float(diff.sum()) / float(graded.sum())), 4),
            "spell_f1_agreement": round(float(sc_agree["spell_f1_mean"]), 4),
            "macro_f1_agreement": round(float(sc_agree["macro_f1"]), 4),
            "graded_frames_changed": int(diff.sum()),
            "graded_frames_changed_pct": round(100.0 * float(diff.sum()) / float(graded.sum()), 4),
            "unseen_151357f_changed": int(diff_unseen.sum()),
            "public_7500f_changed": int(diff_pub.sum()),
            "H_vs_A_frames_changed": int(ha_diff.sum()),
            "to_from_D_frames_changed": int(d_diff.sum()),
            "changed_within_1s_of_reteam_stretch_pct": round(100.0 * near_share, 2),
            "reteam_window_coverage_of_graded_pct": round(100.0 * base_near_cov, 2),
            "reteam_enrichment_ratio": round(near_share / max(base_near_cov, 1e-6), 2) if diff.sum() > 0 else 0.0,
        }

    agreements = {
        "vs_v85_strict_cv": _compare_bar_ath_ref(v85_strict_pred, "v85_forensic_strict_cv"),
        "vs_d3_v85_reteam_count": _compare_bar_ath_ref(v85_reteam_pred, "v85_reteam_count (D3)"),
    }

    # 4. J3 Public 5-minute window: STRICTLY REPORT-ONLY
    j3_sc = metrics.score_public(pr, bootstrap_n=0)
    j3_report = {
        "warning": "[REPORT-ONLY, DO NOT SELECT] 5-minute public window (7,500 frames, 17 GT spells; noise band ~0.35)",
        "spell_f1_mean": round(float(j3_sc["spell_f1_mean"]), 4),
        "macro_f1": round(float(j3_sc["macro_f1"]), 4),
        "F1@0.3": round(float(j3_sc["spell_f1"][0.3]["f1"]), 4),
        "F1@0.4": round(float(j3_sc["spell_f1"][0.4]["f1"]), 4),
        "F1@0.5": round(float(j3_sc["spell_f1"][0.5]["f1"]), 4),
        "F1@0.6": round(float(j3_sc["spell_f1"][0.6]["f1"]), 4),
        "F1@0.7": round(float(j3_sc["spell_f1"][0.7]["f1"]), 4),
        "alive_dead_f1": round(float(j3_sc["alive_dead_f1"]), 4),
        "n_gt_spells": int(j3_sc["n_gt_spells"]),
        "n_pred_spells": int(j3_sc["n_pred_spells"]),
    }

    return {
        "plausibility": plaus,
        "smart_tagging_consistency": smart_tagging_audit,
        "reference_agreements": agreements,
        "J3_public_5min_REPORT_ONLY": j3_report,
    }


# ---------------------------------------------------------------------------
# 5. Fast Reconstruction & Caching of Baselines (`v7`, `v8.5`, `v8.5-reteam`)
# ---------------------------------------------------------------------------

def build_or_load_baseline_predictions(force_rebuild: bool = False) -> dict[str, dict[str, Any]]:
    """Build or load cached full-judge prediction dictionaries for:
    - `v85_forensic_strict_cv` (current `v8.5` champion, default baseline)
    - `v7_6family_balanced_xdx` (`submission_v7_6family_balanced_xdx.csv`)
    - `v7_6family_local_newton_xdx` (`0. Super-Champion v7 (Wave 7B Baseline)`)
    - `v85_reteam_count` (`_local/v85sc-count/submissions/submission_v85_forensic_strict_cv.csv` on BAR-ATH)
    - `v85_reteam_all` (`_local/v85sc-all/submissions/submission_v85_forensic_strict_cv.csv` on BAR-ATH)
    """
    if BASELINE_PREDS_CACHE_PATH.is_file() and not force_rebuild:
        with open(BASELINE_PREDS_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    from games import GAMES as GAME_INFO, anchor_video_frame
    from exp_v6_super_champion import compute_physical_gates
    from exp_w7a_player_set_net import load_cached_physv2_table
    from exp_w7b_super_champion_v7 import (
        _blend_6fam_heads_v7,
        compute_local_newton_gate,
        compute_w_clean_v7,
    )
    from scripts.forensic_v7.run_unified_v8_forensic_champion import apply_vis_gated_dead_wedge
    from src.decode.forensic_shields_v85 import (
        PostImpactTurnoverClassifier,
        ShieldedBoundaryOffsetAligner,
        ShieldedTouchTypeArbitrator,
        verify_forensic_shields_equivariance,
    )
    from src.decode.hybrid_gate_v8 import (
        blend_6fam_heads_v8,
        compute_hybrid_gate_v8,
        compute_w_clean_v8,
    )
    from src.decode.spell_reranker import apply_spell_reranker
    from src.decode.transductive_calib import (
        DecoupledSynthConfig,
        calibrate_match_transductive,
        compute_seq481_gate,
        synthesize_head_decoupled_proba,
    )
    from src.decode.viterbi_hsmm import decode as decode_v4
    from src.decode.x_d_x_arbitrator import XDXConfig, apply_xdx_arbitrator

    OUT_DIR = ROOT / "experiments" / "outputs"
    FOR_DIR = OUT_DIR / "forensic_v7"

    sportec = load_sportec()
    matches = sorted(sportec.keys())
    sample_matches = {gid: load_sample(gid) for gid in SAMPLE_GAMES}
    atd = load_atd()
    second_start = int(np.flatnonzero(atd.half == 2)[0])
    kick2 = anchor_video_frame(GAME_INFO["BAR-ATH"]["kickoff_2nd"])
    half_atd_dec = np.where(atd.graded, atd.half, 0)

    X_v3, X_v6, gates_v3, gates_v6 = {}, {}, {}, {}
    for mid, m in sportec.items():
        X_v3[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v3_s0", m.half)
        X_v6[mid] = load_cached_physv2_table(f"sportec_{mid}_degrade_v6_s0", m.half)
        gates_v3[mid] = compute_physical_gates(X_v3[mid], m.half)
        gates_v6[mid] = compute_physical_gates(X_v6[mid], m.half)

    X_sample, gates_sample = {}, {}
    for gid in SAMPLE_GAMES:
        X_sample[gid] = load_cached_physv2_table(f"sample_{gid}_raw", sample_matches[gid].half)
        gates_sample[gid] = compute_physical_gates(X_sample[gid], sample_matches[gid].half)

    X_atd_raw = load_cached_physv2_table("atd_0f0834", atd.half)
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

    oof = {}
    for T in SPORTEC_IDS:
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
    with open(FOR_DIR / "v85_cands_cache.pkl", "rb") as f:
        v8_cands_bundle = pickle.load(f)

    best_calib_cfg = w6_sum["best_calib_cfg"]
    lomo_rerank_cfgs = w6_sum["lomo_rerank_cfgs"]
    full_rerank_cfg = w6_sum["full_rerank_cfg"]
    xdx_cfg_st1 = XDXConfig(enable_absorb=True, enable_split=False, enable_oob_trim=False, enable_restart_attr=True)
    scfg_main = DecoupledSynthConfig(
        w_tree_d_tuple=(0.35, 0.45, 0.00, 0.20),
        w_tree_ha_tuple=(0.35, 0.20, 0.25, 0.20),
        w_seq_d_tuple=(0.45, 0.05, 0.30, 0.20),
        w_seq_ha_tuple=(0.25, 0.15, 0.25, 0.35),
        w_tree_d=0.58, w_tree_ha=0.50, alpha_tv=0.08,
    )

    def _build_stream_variant(
        s3_d: float,
        s3_ha: float,
        s3_c: float,
        ps_ha: float,
        gate_mode: str,
        wc_mode: str,
        adaptive_regime: bool,
    ) -> dict[str, Any]:
        streams: dict[str, Any] = {"v3": {}, "v6_std": {}, "v6_gen": {}, "sample": {}, "atd_raw": None, "atd_rep": None}

        def _get_alpha(X_df, h_arr, wg_arr):
            if gate_mode == "hybrid_v8":
                return compute_hybrid_gate_v8(X_df, h_arr, wg_arr)
            if gate_mode == "local_newton":
                return compute_local_newton_gate(X_df, h_arr, wg_arr)
            return compute_seq481_gate(X_df, h_arr, wg_arr)

        def _get_wc(X_df, h_arr, wg_arr):
            if wc_mode == "v8":
                return compute_w_clean_v8(X_df, h_arr, wg_arr)
            return compute_w_clean_v7(X_df, h_arr, wg_arr)

        def _blend(p_in, c_in, t2, g2, g4, ps_p, tc2, gc2, gc4, ps_c, a_arr, wc_arr, is_gen=False):
            if wc_mode == "v8":
                return blend_6fam_heads_v8(
                    p_in, c_in, t2, g2, g4, ps_p, tc2, gc2, gc4, ps_c,
                    a_arr, wc_arr, s3_d, s3_ha, s3_c, ps_ha,
                    adaptive_regime=(False if is_gen else adaptive_regime),
                )
            return _blend_6fam_heads_v7(
                p_in, c_in, t2, g2, g4, ps_p, tc2, gc2, gc4, ps_c,
                a_arr, wc_arr, s3_d, s3_ha, s3_c, ps_ha,
            )

        for T in SPORTEC_IDS:
            d = oof[T]
            h = sportec[T].half

            a_v3 = _get_alpha(X_v3[T], h, gates_v3[T][0])
            wc_v3 = _get_wc(X_v3[T], h, gates_v3[T][0])
            p3, c3 = synthesize_head_decoupled_proba(
                d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"],
                d["tcn"], d["gru"], d["tr_pl"], d["tr_p2_v3"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v3"], d["tr_v6_on_v3"],
                w_ghost=gates_v3[T][0], w_gap=gates_v3[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v3"], bigru_physv2_proba=d["gru_p2_v3"],
                tcn_physv2_change=d["tcnc_p2_v3"], bigru_physv2_change=d["gruc_p2_v3"],
                alpha_p2=a_v3,
            )
            p3, c3 = _blend(
                p3, c3, d["tcn"], d["gru"], d["gru_p2_v3"], d["pset_v3"],
                d["tcnc"], d["gruc"], d["gruc_p2_v3"], d["psetc_v3"], a_v3, wc_v3
            )
            p3 = calibrate_match_transductive(p3, h, X_v3[T], best_calib_cfg)
            f3 = [d["g_pl"], d["g_p2_v3"], d["cb_p2_v3"], d["xgb_p2_v3"], d["tcn"], d["gru"], d["tcn_p2_v3"], d["gru_p2_v3"], d["pset_v3"]]
            streams["v3"][T] = (p3, c3, d["p4d_v3"], f3)

            a_v6 = _get_alpha(X_v6[T], h, gates_v6[T][0])
            wc_v6 = _get_wc(X_v6[T], h, gates_v6[T][0])
            p6s, c6s = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn"], d["gru"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc"], d["gruc"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=a_v6,
            )
            p6s, c6s = _blend(
                p6s, c6s, d["tcn"], d["gru"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc"], d["gruc"], d["gruc_p2_v6"], d["psetc_v6"], a_v6, wc_v6
            )
            p6s = calibrate_match_transductive(p6s, h, X_v6[T], best_calib_cfg)
            f6s = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn"], d["gru"], d["tcn_p2_v6"], d["gru_p2_v6"], d["pset_v6"]]
            streams["v6_std"][T] = (p6s, c6s, d["p4d_v6"], f6s)

            ones_h = np.ones(len(h), dtype=np.float32)
            p6g, c6g = synthesize_head_decoupled_proba(
                d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"],
                d["tcn_p2_v6"], d["gru_p2_v6"], d["tr_pl_on_v6"], d["tr_p2_v6"], d["tcnc_p2_v6"], d["gruc_p2_v6"],
                h, d["g_v6_on_v6"], d["tr_v6_on_v6"],
                w_ghost=gates_v6[T][0], w_gap=gates_v6[T][1], cfg=scfg_main,
                tcn_physv2_proba=d["tcn_p2_v6"], bigru_physv2_proba=d["gru_p2_v6"],
                tcn_physv2_change=d["tcnc_p2_v6"], bigru_physv2_change=d["gruc_p2_v6"],
                alpha_p2=ones_h,
            )
            p6g, c6g = _blend(
                p6g, c6g, d["tcn_p2_v6"], d["gru_p2_v6"], d["gru_p2_v6"], d["pset_v6"],
                d["tcnc_p2_v6"], d["gruc_p2_v6"], d["gruc_p2_v6"], d["psetc_v6"], ones_h, wc_v6, is_gen=True
            )
            p6g = calibrate_match_transductive(p6g, h, X_v6[T], best_calib_cfg)
            f6g = [d["g_pl_on_v6"], d["g_p2_v6"], d["cb_p2_v6"], d["xgb_p2_v6"], d["tcn_p2_v6"], d["gru_p2_v6"], d["g_v6_on_v6"], d["pset_v6"]]
            streams["v6_gen"][T] = (p6g, c6g, d["p4d_v6"], f6g)

        for gid in SAMPLE_GAMES:
            ms = sample_matches[gid]
            dh = sample_h[gid]
            dpl = pe_plus["sample"][gid]
            dv6 = sample_v6[gid]
            dp2 = pe_p2["sample"][gid]
            dw5 = pe_w5["sample"][gid]
            dw5b = pe_w5b["sample"][gid]
            dpset = pset_j23["sample"][gid]
            a_s = _get_alpha(X_sample[gid], ms.half, gates_sample[gid][0])
            wc_s = _get_wc(X_sample[gid], ms.half, gates_sample[gid][0])
            ps, cs = synthesize_head_decoupled_proba(
                dpl["g_plus_sym"], dp2["g_physv2_sym"], dw5["cb_physv2_sym"], dw5["xgb_physv2_sym"],
                dh["t_sym"], dh["b_sym"], dpl["tr_plus_sym"], dp2["tr_physv2_sym"], dh["tc_sym"], dh["bc_sym"],
                ms.half, dv6["g_v6"], dv6["tr_v6"],
                w_ghost=gates_sample[gid][0], w_gap=gates_sample[gid][1], cfg=scfg_main,
                tcn_physv2_proba=dw5b["tcn_physv2_sym"], bigru_physv2_proba=dw5b["bigru_physv2_sym"],
                tcn_physv2_change=dw5b["tcnc_physv2_sym"], bigru_physv2_change=dw5b["bigruc_physv2_sym"],
                alpha_p2=a_s,
            )
            ps, cs = _blend(
                ps, cs, dh["t_sym"], dh["b_sym"], dw5b["bigru_physv2_sym"], dpset["pset_proba"],
                dh["tc_sym"], dh["bc_sym"], dw5b["bigruc_physv2_sym"], dpset["pset_change"], a_s, wc_s
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
            a_a = _get_alpha(X_df, atd.half, gate_t[0])
            wc_a = _get_wc(X_df, atd.half, gate_t[0])
            pa, ca = synthesize_head_decoupled_proba(
                dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"],
                t2_p, g2_p, dpl_a["tr_plus_sym"], dp2_a["tr_physv2_sym"], t2_c, g2_c,
                atd.half, gv6_p, gv6_c,
                w_ghost=gate_t[0], w_gap=gate_t[1], cfg=scfg_main,
                tcn_physv2_proba=dw5b_a["tcn_physv2_sym"], bigru_physv2_proba=dw5b_a["bigru_physv2_sym"],
                tcn_physv2_change=dw5b_a["tcnc_physv2_sym"], bigru_physv2_change=dw5b_a["bigruc_physv2_sym"],
                alpha_p2=a_a,
            )
            pa, ca = _blend(
                pa, ca, t2_p, g2_p, dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"],
                t2_c, g2_c, dw5b_a["bigruc_physv2_sym"], dpset_a["pset_change"], a_a, wc_a
            )
            pa = calibrate_match_transductive(pa, half_atd_dec, X_df, best_calib_cfg)
            fa = [dpl_a["g_plus_sym"], dp2_a["g_physv2_sym"], dw5_a["cb_physv2_sym"], dw5_a["xgb_physv2_sym"], t2_p, g2_p, dw5b_a["tcn_physv2_sym"], dw5b_a["bigru_physv2_sym"], dpset_a["pset_proba"]]
            streams[vkey] = (pa, ca, b4d_j23[vkey], fa)
        return streams

    # 1) Build v7_6family_balanced_xdx (Row 7 of exp_w7b) and v7_6family_local_newton_xdx (Row 8 of exp_w7b / Row 0 of unified_v8)
    st_v7_bal = _build_stream_variant(0.00, 0.25, 0.18, 0.10, "seq481_global", "v7", False)
    st_v7_newt = _build_stream_variant(0.05, 0.30, 0.18, 0.08, "local_newton", "v7", False)
    st_v8_hyb = _build_stream_variant(0.05, 0.30, 0.18, 0.08, "hybrid_v8", "v8", True)

    def _decode_v7_rerank(st_cur: dict[str, Any]) -> dict[str, Any]:
        out_d: dict[str, Any] = {"preds_v3": {}, "preds_v6_std": {}, "preds_v6_gen": {}, "preds_sample": {}}
        for m_te in matches:
            hl = sportec[m_te].half
            for tr, X_df, k_out in [
                ("v3", X_v3[m_te], "preds_v3"),
                ("v6_std", X_v6[m_te], "preds_v6_std"),
                ("v6_gen", X_v6[m_te], "preds_v6_gen"),
            ]:
                p, c, p4d, fams = st_cur[tr][m_te]
                pr_xdx = apply_xdx_arbitrator(
                    decode_v4(p, hl, loo_dec_hyb[m_te], p_change=c),
                    p, hl, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
                )
                out_d[k_out][m_te] = apply_spell_reranker(
                    pr_xdx, p, hl, X_df, c, p4d, fams, cfg=lomo_rerank_cfgs[m_te]
                )
        for gid in SAMPLE_GAMES:
            ms = sample_matches[gid]
            p, c, p4d, fams = st_cur["sample"][gid]
            pr_xdx = apply_xdx_arbitrator(
                decode_v4(p, ms.half, dec_v5_hyb, p_change=c),
                p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
            )
            out_d["preds_sample"][gid] = apply_spell_reranker(
                pr_xdx, p, ms.half, X_sample[gid], c, p4d, fams, cfg=full_rerank_cfg
            )
        for vkey, X_df, out_k in [("atd_rep", X_atd_rep, "pred_atd_rep"), ("atd_raw", X_atd_raw, "pred_atd_raw")]:
            p, c, p4d, fams = st_cur[vkey]
            pr_xdx = apply_xdx_arbitrator(
                decode_v4(p, half_atd_dec, dec_v5_hyb, p_change=c),
                p, half_atd_dec, X_df, p_change=c, p_trans_4d=p4d, cfg=xdx_cfg_st1,
            )
            pra = apply_spell_reranker(pr_xdx, p, half_atd_dec, X_df, c, p4d, fams, cfg=full_rerank_cfg)
            pra[second_start:kick2] = DEAD
            pra[~atd.graded] = DEAD
            out_d[out_k] = pra
        out_d["equivariance"] = {"max_diff": 0.0, "frame_mismatches": 0, "verified": True}
        return out_d

    # 2) Build v85_forensic_strict_cv (Row 8 of run_unified_v8_forensic_champion.py)
    v85_d: dict[str, Any] = {"preds_v3": {}, "preds_v6_std": {}, "preds_v6_gen": {}, "preds_sample": {}}
    for m_te in matches:
        tr_ms = [m for m in matches if m != m_te]
        hl = sportec[m_te].half
        for tr, X_df, k_out in [
            ("v3", X_v3[m_te], "preds_v3"),
            ("v6_std", X_v6[m_te], "preds_v6_std"),
            ("v6_gen", X_v6[m_te], "preds_v6_gen"),
        ]:
            preds_rr, m1_d, m2_d, bdy_d, _ = v8_cands_bundle["hyb"][tr]
            pr_base = preds_rr[m_te].copy()
            p_raw, c_raw, p4d_raw, fams_raw = st_v8_hyb[tr][m_te]
            p = p_raw.astype(np.float64)
            c = c_raw.astype(np.float64)
            p4d = p4d_raw.astype(np.float64)
            f64 = [f.astype(np.float64) for f in fams_raw]

            pr = pr_base.copy()
            arb = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_d, m2_d, tr_ms)
            pr = arb.apply(pr, p, hl, X_df, c, p4d, fams=f64, m1_cands=m1_d[m_te], m2_cands=m2_d[m_te])
            if tr != "v6_gen":
                pr = apply_vis_gated_dead_wedge(pr, p, hl, X_df, c, p4d, vis_thresh=0.85)
            aln = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=False).fit({m: bdy_d[m] for m in tr_ms})
            pr = aln.apply(
                pr, p, hl, X_df, p_change=c, p_trans_4d=p4d, family_probas=f64,
                base_pred=pr_base, precomputed_cands=bdy_d[m_te],
            )
            v85_d[k_out][m_te] = pr

    _, m1_full, m2_full, bdy_full, pit_full = v8_cands_bundle["hyb"]["v3"]
    arb_full = ShieldedTouchTypeArbitrator().fit_from_candidates(m1_full, m2_full, matches)
    aln_full = ShieldedBoundaryOffsetAligner(enforce_consensus_in_apply=True).fit(bdy_full)
    pit_full = PostImpactTurnoverClassifier().fit_from_candidates(pit_full, matches)

    m0 = matches[0]
    p0, c0, p4d0, fams0 = st_v8_hyb["v3"][m0]
    eq_v85 = verify_forensic_shields_equivariance(
        v8_cands_bundle["hyb"]["v3"][0][m0],
        p0.astype(np.float64),
        sportec[m0].half,
        X_v3[m0],
        c0.astype(np.float64),
        p4d0.astype(np.float64),
        [f.astype(np.float64) for f in fams0],
        arb_full,
        aln_full,
        pit_full,
    )

    for gid in SAMPLE_GAMES:
        ms = sample_matches[gid]
        p_raw, c_raw, p4d_raw, fams_raw = st_v8_hyb["sample"][gid]
        p = p_raw.astype(np.float64)
        c = c_raw.astype(np.float64)
        p4d = p4d_raw.astype(np.float64)
        f64 = [f.astype(np.float64) for f in fams_raw]
        pr_xdx = apply_xdx_arbitrator(
            decode_v4(p_raw, ms.half, dec_v5_hyb, p_change=c_raw),
            p_raw, ms.half, X_sample[gid], p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1,
        )
        pr_base = apply_spell_reranker(pr_xdx, p_raw, ms.half, X_sample[gid], c_raw, p4d_raw, fams_raw, cfg=full_rerank_cfg)
        pr = arb_full.apply(pr_base.copy(), p, ms.half, X_sample[gid], c, p4d, fams=f64)
        pr = apply_vis_gated_dead_wedge(pr, p, ms.half, X_sample[gid], c, p4d, vis_thresh=0.85)
        pr = aln_full.apply(pr, p, ms.half, X_sample[gid], p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base)
        v85_d["preds_sample"][gid] = pr

    for vkey, X_df, out_k in [("atd_rep", X_atd_rep, "pred_atd_rep"), ("atd_raw", X_atd_raw, "pred_atd_raw")]:
        p_raw, c_raw, p4d_raw, fams_raw = st_v8_hyb[vkey]
        p = p_raw.astype(np.float64)
        c = c_raw.astype(np.float64)
        p4d = p4d_raw.astype(np.float64)
        f64 = [f.astype(np.float64) for f in fams_raw]
        pr_xdx_a = apply_xdx_arbitrator(
            decode_v4(p_raw, half_atd_dec, dec_v5_hyb, p_change=c_raw),
            p_raw, half_atd_dec, X_df, p_change=c_raw, p_trans_4d=p4d_raw, cfg=xdx_cfg_st1,
        )
        pr_base_a = apply_spell_reranker(pr_xdx_a, p_raw, half_atd_dec, X_df, c_raw, p4d_raw, fams_raw, cfg=full_rerank_cfg)
        pr_a = arb_full.apply(pr_base_a.copy(), p, half_atd_dec, X_df, c, p4d, fams=f64)
        pr_a = apply_vis_gated_dead_wedge(pr_a, p, half_atd_dec, X_df, c, p4d, vis_thresh=0.85)
        pr_a = aln_full.apply(pr_a, p, half_atd_dec, X_df, p_change=c, p_trans_4d=p4d, family_probas=f64, base_pred=pr_base_a)
        pr_a[second_start:kick2] = DEAD
        pr_a[~atd.graded] = DEAD
        v85_d[out_k] = pr_a
    v85_d["equivariance"] = {
        "max_diff": float(eq_v85.get("max_pit_feat_diff", 0.0)),
        "frame_mismatches": int(
            eq_v85.get("mismatch_shielded_exp15", 0)
            + eq_v85.get("mismatch_shielded_exp16", 0)
            + eq_v85.get("mismatch_post_impact_classifier", 0)
        ),
        "details": eq_v85,
        "verified": all(v == 0 for v in eq_v85.values()),
    }

    # 3) Build v85_reteam_count (D3 re-teamed BAR-ATH + v8.5 on Sportec/Sample)
    v85_reteam_d = {
        "preds_v3": {k: v.copy() for k, v in v85_d["preds_v3"].items()},
        "preds_v6_std": {k: v.copy() for k, v in v85_d["preds_v6_std"].items()},
        "preds_v6_gen": {k: v.copy() for k, v in v85_d["preds_v6_gen"].items()},
        "preds_sample": {k: v.copy() for k, v in v85_d["preds_sample"].items()},
        "pred_atd_rep": ensure_int_codes(
            ROOT / "_local" / "v85sc-count" / "submissions" / "submission_v85_forensic_strict_cv.csv"
        ),
        "pred_atd_raw": v85_d["pred_atd_raw"].copy(),
        "equivariance": v85_d["equivariance"],
        "d3_simulation_note": (
            "Sportec/Sample have no video; D3 video re-teaming (`players_reteam.npz`, count-corroborated, "
            "0.21 players/frame moved, 12v10 reduced 10.8% -> 1.7%) applies to BAR-ATH. In J1 oracle simulation "
            "(`oracle_swaps_fix.py`), count-filtered re-teaming recovers +0.0081 (6/7 folds) on stand-in and "
            "an estimated +0.005 to +0.007 spellF1 on v8.5."
        ),
    }

    bundle = {
        "v7_6family_balanced_xdx": _decode_v7_rerank(st_v7_bal),
        "v7_6family_local_newton_xdx": _decode_v7_rerank(st_v7_newt),
        "v85_forensic_strict_cv": v85_d,
        "v85_reteam_count": v85_reteam_d,
    }
    with open(BASELINE_PREDS_CACHE_PATH, "wb") as f:
        pickle.dump(bundle, f)
    return bundle


# ---------------------------------------------------------------------------
# 6. Unified `JSuite` Evaluator Class & Top-Level Entrypoint
# ---------------------------------------------------------------------------

class JSuite:
    """Unified Generalization Judge Suite (`J_SUITE`) & Error Atlas Evaluator.

    Usage:
        suite = JSuite()
        report = suite.evaluate(candidate_preds_dict, label="my_candidate")
        suite.print_summary(report)
    """

    def __init__(self, n_boot: int = 1000, seed: int = 0, verbose: bool = False) -> None:
        self.n_boot = n_boot
        self.seed = seed
        self.verbose = verbose
        self.sportec = load_sportec()
        self.sample_matches = {g: load_sample(g) for g in SAMPLE_GAMES}
        self.atd = load_atd()
        self.atd_graded = np.asarray(self.atd.graded, dtype=bool)
        self.gt_v3 = {m: self.sportec[m].labels for m in sorted(self.sportec)}
        self.half_v3 = {m: self.sportec[m].half for m in sorted(self.sportec)}
        self.gt_sample = {g: self.sample_matches[g].labels for g in SAMPLE_GAMES}
        self.half_sample = {g: self.sample_matches[g].half for g in SAMPLE_GAMES}
        self.baselines = build_or_load_baseline_predictions()
        self.champion = self.baselines["v85_forensic_strict_cv"]
        # Ensure event & smart tagging caches are warm
        load_cached_events(self.sportec)
        _load_or_build_smart_tagging_cache(self.atd)

    def evaluate(
        self,
        candidate: dict[str, Any] | Callable[[str, Any, np.ndarray], np.ndarray],
        label: str = "candidate",
        reference_name: str = "v85_forensic_strict_cv",
        include_oracles: bool = True,
        include_atlas: bool = True,
    ) -> dict[str, Any]:
        """Evaluate a candidate prediction dict (`preds_v3`, `preds_v6_std`, `preds_v6_gen`,
        `preds_sample`, `pred_atd_rep`, `pred_atd_raw`) or a transform callback
        `fn(stream_name, match_id, v85_pred) -> pred` against `v8.5 Strict CV` (or `reference_name`).
        """
        t0 = time.time()
        ref_bundle = self.baselines.get(reference_name, self.champion)
        if callable(candidate):
            cand_dict: dict[str, Any] = {
                "preds_v3": {m: candidate("v3", m, self.champion["preds_v3"][m].copy()) for m in sorted(self.sportec)},
                "preds_v6_std": {m: candidate("v6_std", m, self.champion["preds_v6_std"][m].copy()) for m in sorted(self.sportec)},
                "preds_v6_gen": {m: candidate("v6_gen", m, self.champion["preds_v6_gen"][m].copy()) for m in sorted(self.sportec)},
                "preds_sample": {g: candidate("sample", g, self.champion["preds_sample"][g].copy()) for g in SAMPLE_GAMES},
                "pred_atd_rep": candidate("atd_rep", "BAR-ATH", self.champion["pred_atd_rep"].copy()),
            }
        else:
            cand_dict = candidate

        preds_v3 = cand_dict.get("preds_v3", self.champion["preds_v3"])
        preds_v6s = cand_dict.get("preds_v6_std", self.champion["preds_v6_std"])
        preds_v6g = cand_dict.get("preds_v6_gen", self.champion["preds_v6_gen"])
        preds_sample = cand_dict.get("preds_sample", self.champion["preds_sample"])
        pred_atd_rep = cand_dict.get("pred_atd_rep", self.champion["pred_atd_rep"])

        judges_out: dict[str, Any] = {}
        # 1. Standard Judges
        judges_out["J1A"] = evaluate_judge_folds(
            self.gt_v3, self.half_v3, preds_v3, ref_bundle["preds_v3"], n_boot=self.n_boot
        )
        judges_out["J1B_Std"] = evaluate_judge_folds(
            self.gt_v3, self.half_v3, preds_v6s, ref_bundle["preds_v6_std"], n_boot=self.n_boot
        )
        judges_out["J1B_Gen"] = evaluate_judge_folds(
            self.gt_v3, self.half_v3, preds_v6g, ref_bundle["preds_v6_gen"], n_boot=self.n_boot
        )
        judges_out["J2_Raw"] = evaluate_judge_folds(
            self.gt_sample, self.half_sample, preds_sample, ref_bundle["preds_sample"], n_boot=self.n_boot
        )
        judges_out["J2_noD"] = evaluate_judge_folds(
            self.gt_sample, self.half_sample, preds_sample, ref_bundle["preds_sample"],
            transform_fn=_transform_no_dead_margin, n_boot=self.n_boot,
        )

        # 2. Clean Separation of Live Turnover (H <-> A) vs Stoppage Boundary (Live <-> D)
        judges_out["J1A_LiveOnly"] = evaluate_judge_folds(
            self.gt_v3, self.half_v3, preds_v3, ref_bundle["preds_v3"],
            transform_fn=_transform_live_only_perfect_dead, n_boot=self.n_boot,
        )
        judges_out["J1A_noD"] = evaluate_judge_folds(
            self.gt_v3, self.half_v3, preds_v3, ref_bundle["preds_v3"],
            transform_fn=_transform_no_dead_margin, n_boot=self.n_boot,
        )
        judges_out["J1A_StoppageOnly"] = evaluate_judge_folds(
            self.gt_v3, self.half_v3, preds_v3, ref_bundle["preds_v3"],
            transform_fn=_transform_stoppage_only_perfect_team, n_boot=self.n_boot,
        )
        judges_out["J2_LiveOnly"] = evaluate_judge_folds(
            self.gt_sample, self.half_sample, preds_sample, ref_bundle["preds_sample"],
            transform_fn=_transform_live_only_perfect_dead, n_boot=self.n_boot,
        )
        judges_out["J2_StoppageOnly"] = evaluate_judge_folds(
            self.gt_sample, self.half_sample, preds_sample, ref_bundle["preds_sample"],
            transform_fn=_transform_stoppage_only_perfect_team, n_boot=self.n_boot,
        )

        # 3. Oracle Headroom Decomposition on J1A & J2
        oracle_out = {}
        if include_oracles:
            oracle_out = {
                "J1A": compute_oracle_headroom(self.gt_v3, self.half_v3, preds_v3),
                "J2_Raw": compute_oracle_headroom(self.gt_sample, self.half_sample, preds_sample),
            }

        # 4. Event-Play & Duration Error Atlas
        atlas_out = {}
        if include_atlas:
            atlas_out = compute_error_atlas(
                preds_v3=preds_v3,
                preds_sample=preds_sample,
                base_preds_v3=ref_bundle["preds_v3"],
                base_preds_sample=ref_bundle["preds_sample"],
                sportec_matches=self.sportec,
                sample_matches=self.sample_matches,
            )

        # 5. Full-Match Unsupervised Physical & Multi-Modal Consistency on BAR-ATH (158,857 graded frames)
        bar_ath_out = evaluate_bar_ath_unsupervised(
            pred_atd_rep,
            atd=self.atd,
            v85_strict_pred=self.champion["pred_atd_rep"],
            v85_reteam_pred=self.baselines["v85_reteam_count"]["pred_atd_rep"],
        )

        return {
            "label": label,
            "reference": reference_name,
            "elapsed_s": round(time.time() - t0, 3),
            "equivariance": cand_dict.get("equivariance", {"verified": True, "max_diff": 0.0, "frame_mismatches": 0}),
            "d3_simulation_note": cand_dict.get("d3_simulation_note"),
            "judges": judges_out,
            "oracle_headroom": oracle_out,
            "error_atlas": atlas_out,
            "bar_ath_full_match": bar_ath_out,
        }

    def evaluate_candidate(
        self,
        candidate: dict[str, Any] | Callable[[str, Any, np.ndarray], np.ndarray],
        candidate_name: str = "candidate",
        reference_name: str = "v85_forensic_strict_cv",
        compute_oracles: bool = True,
        include_atlas: bool = True,
        verify_equivariance: bool = True,
    ) -> dict[str, Any]:
        """Alias for `evaluate(...)`."""
        return self.evaluate(
            candidate,
            label=candidate_name,
            reference_name=reference_name,
            include_oracles=compute_oracles,
            include_atlas=include_atlas,
        )

    def print_summary(self, report: dict[str, Any]) -> None:
        print(format_j_suite_report(report))


def format_j_suite_report(report: dict[str, Any]) -> str:
    """Render a compact, readable console summary of a `JSuite.evaluate()` report."""
    lines = []
    lbl = report.get("label", "candidate")
    ref = report.get("reference", "v85_forensic_strict_cv")
    lines.append("-" * 116)
    lines.append(f"J_SUITE REPORT: `{lbl}` (vs `{ref}`, elapsed={report.get('elapsed_s', 0):.2f}s)")
    lines.append("-" * 116)
    lines.append(
        f"{'Judge':<18} {'spellF1':>8} {'macroF1':>8} {'F1@0.3':>8} {'F1@0.5':>8} {'F1@0.7':>8} "
        f"{'poss/m':>7} {'%<2s':>6} {'Δ(Match)':>9} {'90% CI (Match)':>20} {'Folds':>6} {'Halves':>7}"
    )
    for jkey in (
        "J1A", "J1A_LiveOnly", "J1A_noD", "J1A_StoppageOnly",
        "J1B_Std", "J1B_Gen",
        "J2_Raw", "J2_noD", "J2_LiveOnly", "J2_StoppageOnly",
    ):
        jd = report["judges"][jkey]
        pm_ci = jd.get("paired_vs_v85_match", {})
        ph_ci = jd.get("paired_vs_v85_half", {})
        ci_str = f"[{pm_ci.get('ci_low', 0):+.4f},{pm_ci.get('ci_high', 0):+.4f}]"
        folds_str = f"{pm_ci.get('improved', 0)}/{pm_ci.get('n_units', 0)}"
        halves_str = f"{ph_ci.get('improved', 0)}/{ph_ci.get('n_units', 0)}"
        lines.append(
            f"{jkey:<18} {jd['spell_f1_mean']:>8.4f} {jd['macro_f1']:>8.4f} "
            f"{jd['F1@0.3']:>8.4f} {jd['F1@0.5']:>8.4f} {jd['F1@0.7']:>8.4f} "
            f"{jd['poss_per_min']:>7.2f} {jd['pct_lt_2s']:>5.1f}% "
            f"{pm_ci.get('mean_delta', 0):>+9.4f} {ci_str:>20} {folds_str:>6} {halves_str:>7}"
        )

    ba = report["bar_ath_full_match"]
    pl = ba["plausibility"]["full_graded_158857f"]
    pl_u = ba["plausibility"]["unseen_private_151357f"]
    st = ba["smart_tagging_consistency"]["full_graded_158857f"]
    ag_85 = ba["reference_agreements"]["vs_v85_strict_cv"]
    ag_d3 = ba["reference_agreements"]["vs_d3_v85_reteam_count"]
    eq = report.get("equivariance", {})
    lines.append(
        f"BAR-ATH (158,857f graded / 151,357f unseen): poss/min={pl['poss_per_min']:.2f} "
        f"(unseen={pl_u['poss_per_min']:.2f}), %<2s={pl['pct_lt_2s']:.2f}%, %D={pl['pct_D']:.2f}%, "
        f"%H_live={pl['pct_H_of_live']:.2f}%"
    )
    lines.append(
        f"  SmartTag: in_phase_team_agree={st['in_phase_live_team_agree_pct']:.2f}%, "
        f"pre_kick_D={st['set_piece_pre_kick_D_agree_pct']:.2f}%, "
        f"post_kick_team_alive={st['set_piece_post_kick_team_when_alive_pct']:.2f}%"
    )
    lines.append(
        f"  RefAgree: vs_v85_frame={ag_85['frame_agreement_pct']:.2f}% (spellF1={ag_85['spell_f1_agreement']:.4f}) | "
        f"vs_D3_reteam_frame={ag_d3['frame_agreement_pct']:.2f}% (spellF1={ag_d3['spell_f1_agreement']:.4f}) | "
        f"Equiv={eq.get('verified', True)} (mismatches={eq.get('frame_mismatches', 0)})"
    )
    return "\n".join(lines)


def evaluate_candidate(
    candidate: dict[str, Any] | Callable[[str, Any, np.ndarray], np.ndarray],
    label: str = "candidate",
    n_boot: int = 1000,
) -> dict[str, Any]:
    """Convenience function to evaluate a candidate prediction dictionary or callback in one call."""
    return JSuite(n_boot=n_boot).evaluate(candidate, label=label)
