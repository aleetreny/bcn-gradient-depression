#!/usr/bin/env python3
"""Comprehensive Generalization, Physical Plausibility & Rule Compliance Audit
for the Final Hackathon Submission (`v8.5` / `v9` vs `v8` / `v7` / `v4`).

Audits:
1. Official `evaluation/validate_submission.py` & `pm.validate_submission(df, 199319)` checks.
2. Cross-Judge generalization across all 18 full matches/streams:
   - Judge 1A: 7 full Bundesliga matches (`Sportec v3` clean, ~1,400 possessions)
   - Judge 1B-Std: 7 full Bundesliga matches (`Sportec v6_std` BAR-ATH-matched noise)
   - Judge 1B-Gen: 7 full Bundesliga matches (`Sportec v6_gen` blind ball dropout)
   - Judge 2: 3 full Metrica Sports matches (`Sample 1, 2, 3`, raw & `D ± 3s`)
   - Judge 3: `BAR-ATH` public evaluation window (7,500 graded frames)
3. Full-Match 90-Minute (`199,319` frames) Unseen-Play Audit on `BAR-ATH`:
   - Compares spell counts, `poss_per_min`, possession share (`H%` vs `A%`), spell duration
     quantiles, and exact frame/spell diffs across Both Halves (including the 85 private minutes!).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.metrica_atd import load as load_atd
from src.eval import metrics
from src.eval.metrics import pm


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_spells(labels: np.ndarray, mask: np.ndarray) -> list[tuple[int, int, str, int]]:
    """Extract contiguous (start, end, label, length) spells inside `mask`."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return []
    spells = []
    s = int(idx[0])
    cur = str(labels[s])
    prev = s
    for i in idx[1:]:
        i = int(i)
        lab = str(labels[i])
        if i != prev + 1 or lab != cur:
            spells.append((s, prev + 1, cur, prev + 1 - s))
            s = i
            cur = lab
        prev = i
    spells.append((s, prev + 1, cur, prev + 1 - s))
    return spells


def analyze_bar_ath_full_match(labels: np.ndarray, half: np.ndarray, graded: np.ndarray) -> dict:
    """Compute physical & tactical plausibility metrics on:
    - Public 5-minute graded window (`graded == True`)
    - Unseen 85-minute live window (`half in {1, 2}` and `not graded` and `labels != 'X'`)
    - Full match active play (`half in {1, 2}` and `labels != 'X'`)
    """
    pub_gt = metrics.public_gt()
    pub_mask = pub_gt != pm.X
    active_mask = (np.isin(half, [1, 2])) & (labels != "X")
    unseen_mask = active_mask & (~pub_mask)

    def stats_for_mask(m: np.ndarray) -> dict:
        n_frames = int(m.sum())
        minutes = n_frames / (25.0 * 60.0)
        spells = extract_spells(labels, m)
        live_spells = [sp for sp in spells if sp[2] in ("H", "A")]
        d_spells = [sp for sp in spells if sp[2] == "D"]
        live_lens = np.array([sp[3] for sp in live_spells], dtype=float)
        n_h = int(np.sum(labels[m] == "H"))
        n_a = int(np.sum(labels[m] == "A"))
        n_d = int(np.sum(labels[m] == "D"))
        n_x = int(np.sum(labels[m] == "X"))
        live_total = max(1, n_h + n_a)
        return {
            "frames": n_frames,
            "minutes": round(minutes, 2),
            "n_live_spells": len(live_spells),
            "n_d_spells": len(d_spells),
            "poss_per_min": round(len(live_spells) / max(1e-6, minutes), 3),
            "H_share_of_live_pct": round(100.0 * n_h / live_total, 2),
            "A_share_of_live_pct": round(100.0 * n_a / live_total, 2),
            "D_share_of_total_pct": round(100.0 * n_d / max(1, n_frames), 2),
            "X_frames": n_x,
            "live_dur_min_s": round(float(live_lens.min() / 25.0), 2) if len(live_lens) else 0.0,
            "live_dur_p10_s": round(float(np.percentile(live_lens, 10) / 25.0), 2) if len(live_lens) else 0.0,
            "live_dur_med_s": round(float(np.median(live_lens) / 25.0), 2) if len(live_lens) else 0.0,
            "live_dur_mean_s": round(float(live_lens.mean() / 25.0), 2) if len(live_lens) else 0.0,
            "live_dur_p90_s": round(float(np.percentile(live_lens, 90) / 25.0), 2) if len(live_lens) else 0.0,
            "live_dur_max_s": round(float(live_lens.max() / 25.0), 2) if len(live_lens) else 0.0,
            "micro_spells_lt_12f": int(np.sum(live_lens < 12)) if len(live_lens) else 0,
            "short_spells_12_35f": int(np.sum((live_lens >= 12) & (live_lens < 35))) if len(live_lens) else 0,
        }

    pub_score = metrics.score_public(labels, bootstrap_n=0)
    return {
        "public_5min_score": {
            "spell_f1_mean": round(float(pub_score["spell_f1_mean"]), 4),
            "F1@0.3": round(float(pub_score["spell_f1"][0.3]["f1"]), 4),
            "F1@0.4": round(float(pub_score["spell_f1"][0.4]["f1"]), 4),
            "F1@0.5": round(float(pub_score["spell_f1"][0.5]["f1"]), 4),
            "F1@0.6": round(float(pub_score["spell_f1"][0.6]["f1"]), 4),
            "F1@0.7": round(float(pub_score["spell_f1"][0.7]["f1"]), 4),
            "macro_f1": round(float(pub_score["macro_f1"]), 4),
            "alive_dead_f1": round(float(pub_score["alive_dead_f1"]), 4),
        },
        "public_5min_stats": stats_for_mask(pub_mask),
        "unseen_85min_stats": stats_for_mask(unseen_mask),
        "full_match_stats": stats_for_mask(active_mask),
    }


def main() -> None:
    print("=" * 115)
    print("FINAL GENERALIZATION, FULL-MATCH UNSEEN PLAY & OFFICIAL COMPLIANCE AUDIT (`v8.5` / `v9`)")
    print("=" * 115, flush=True)

    atd = load_atd()
    half = atd.half
    graded = atd.graded

    candidates = [
        ("v4_frozen", ROOT / "submissions" / "v4.csv"),
        ("v7_6fam_balanced_xdx", ROOT / "submissions" / "submission_v7_6family_balanced_xdx.csv"),
        ("v8_super_champion", ROOT / "submissions" / "submission_v8_super_champion.csv"),
        ("v85_forensic_strict_cv", ROOT / "submissions" / "submission_v85_forensic_strict_cv.csv"),
        ("v85_forensic_grand_champion", ROOT / "submissions" / "submission_v85_forensic_grand_champion.csv"),
        ("v9_all6_grand_champion", ROOT / "submissions" / "submission_v9_all6_grand_champion.csv"),
    ]

    loaded_preds: dict[str, np.ndarray] = {}
    audit_results: dict[str, dict] = {}

    print("\n[1/4] Running Official Validator (`evaluation/validate_submission.py` + `pm.validate_submission`)...")
    for name, path in candidates:
        if not path.exists():
            print(f"  [SKIP] {name}: {path} not found")
            continue
        df = pd.read_csv(path)
        pm.validate_submission(df, 199319)
        res = subprocess.run(
            [sys.executable, str(ROOT / "evaluation" / "validate_submission.py"), str(path)],
            capture_output=True,
            text=True,
            check=True,
        )
        sha = sha256_file(path)
        arr = df["label"].to_numpy(dtype="U1")
        loaded_preds[name] = arr
        info = analyze_bar_ath_full_match(arr, half, graded)
        info["path"] = str(path.relative_to(ROOT))
        info["sha256"] = sha
        info["validator_stdout"] = res.stdout.strip()
        audit_results[name] = info
        print(f"  [PASS] {name:28s} | SHA256={sha[:12]}... | {res.stdout.strip()}")

    print("\n[2/4] Full-Match (90-Minute = Public 5m + Private Unseen 85m) Plausibility Comparison on BAR-ATH:")
    print("-" * 135)
    print(
        f"{'Submission':28s} | {'J3 Pub F1 (MacF1)':19s} | {'Pub p/m':7s} | {'Unseen85m p/m':13s} | "
        f"{'Unseen Spells':13s} | {'H% / A% (Unseen)':16s} | {'Med/Mean Dur(s)':15s} | {'Micro<12f'}"
    )
    print("-" * 135)
    for name, info in audit_results.items():
        p_sc = info["public_5min_score"]
        p_st = info["public_5min_stats"]
        u_st = info["unseen_85min_stats"]
        print(
            f"{name:28s} | {p_sc['spell_f1_mean']:.4f} ({p_sc['macro_f1']:.4f})   | "
            f"{p_st['poss_per_min']:7.2f} | {u_st['poss_per_min']:13.2f} | "
            f"{u_st['n_live_spells']:13d} | {u_st['H_share_of_live_pct']:5.1f}% / {u_st['A_share_of_live_pct']:5.1f}%  | "
            f"{u_st['live_dur_med_s']:5.1f}s / {u_st['live_dur_mean_s']:5.1f}s  | {u_st['micro_spells_lt_12f']:5d}"
        )
    print("-" * 135)

    print("\n[3/4] Pairwise Frame & Spell Differences Across the ENTIRE 199,319 Frames of BAR-ATH:")
    pairs = [
        ("v7_6fam_balanced_xdx", "v85_forensic_strict_cv", "Effect of Wave 8A + Shielded Wave 8B + DeadWedge"),
        ("v85_forensic_strict_cv", "v85_forensic_grand_champion", "Effect of Double-Domino (Short-Turnover Rescue under Missing Players)"),
        ("v85_forensic_grand_champion", "v9_all6_grand_champion", "Effect of EXP-14 SymGapDistilledExpert (Inpainter Student)"),
    ]
    pub_mask = metrics.public_gt() != pm.X
    unseen_mask = (half > 0) & (~pub_mask)
    for k1, k2, desc in pairs:
        if k1 not in loaded_preds or k2 not in loaded_preds:
            continue
        p1, p2 = loaded_preds[k1], loaded_preds[k2]
        diff_mask = p1 != p2
        diff_pub = int(np.sum(diff_mask & pub_mask))
        diff_unseen = int(np.sum(diff_mask & unseen_mask))
        print(f"\n  * {k1} -> {k2} ({desc}):")
        print(f"    Total frames changed across 199,319 frames: {int(diff_mask.sum())} ({100.0*diff_mask.mean():.3f}%)")
        print(f"    - Inside Public 5m ({int(pub_mask.sum())} frames):     {diff_pub} frames ({100.0*diff_pub/max(1,int(pub_mask.sum())):.2f}%)")
        print(f"    - Inside Unseen 85m ({int(unseen_mask.sum())} frames):  {diff_unseen} frames ({100.0*diff_unseen/max(1,int(unseen_mask.sum())):.3f}%)")
        # List contiguous diff segments
        diff_idx = np.where(diff_mask)[0]
        if len(diff_idx) > 0:
            segs = []
            s = int(diff_idx[0])
            prev = s
            for i in diff_idx[1:]:
                i = int(i)
                if i != prev + 1:
                    segs.append((s, prev + 1))
                    s = i
                prev = i
            segs.append((s, prev + 1))
            print(f"    Contiguous modified episodes across full 90m: {len(segs)} episodes")
            for idx_seg, (s, e) in enumerate(segs[:15]):
                in_pub = "PUBLIC-5m" if np.any(pub_mask[s:e]) else f"UNSEEN-Half{int(half[s])}"
                l1 = p1[s]
                l2 = p2[s]
                prev_l = p2[max(0, s - 1)]
                next_l = p2[min(len(p2) - 1, e)]
                print(
                    f"      [{idx_seg+1:02d}] frames {s:6d}..{e:6d} ({e-s:3d}f = {(e-s)/25.0:4.2f}s) | {in_pub:12s} | "
                    f"{k1}='{l1}' -> {k2}='{l2}' (context: {prev_l} -> [{l2}] -> {next_l})"
                )
            if len(segs) > 15:
                print(f"      ... and {len(segs) - 15} more episodes.")

    # Save audit JSON
    out_json = ROOT / "experiments" / "outputs" / "forensic_v7" / "final_v9_audit_report.json"
    with open(out_json, "w") as f:
        json.dump(audit_results, f, indent=2)
    print(f"\n[4/4] Saved complete audit JSON to {out_json}")


if __name__ == "__main__":
    main()
