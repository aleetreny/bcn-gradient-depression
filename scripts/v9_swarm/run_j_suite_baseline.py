#!/usr/bin/env python3
"""Run the unified Generalization Judge Suite (`J_SUITE`) & Error Atlas baseline audit.

Benchmarks and compares:
  1. `v7_6family_balanced_xdx` (prior 6-family balanced champion)
  2. `v7_6family_local_newton_xdx` (prior 6-family local-Newton champion)
  3. `v85_forensic_strict_cv` (current `v8.5` strict-CV 9-family champion)
  4. `v85_reteam_count` (`_local/v85sc-count/submissions/submission_v85_forensic_strict_cv.csv` on BAR-ATH)

Outputs:
  - `experiments/outputs/v9_swarm/j_suite_baseline.json`
  - `experiments/outputs/v9_swarm/j_suite_baseline_summary.md`

Usage:
  PYTHONPATH=.:py_pkgs .venv311/bin/python scripts/v9_swarm/run_j_suite_baseline.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.v9_swarm.j_suite import (
    SWARM_OUT_DIR,
    JSuite,
    ensure_int_codes,
    format_j_suite_report,
)


def _verify_submission_equivalence(suite: JSuite) -> dict[str, Any]:
    """Verify that reconstructed BAR-ATH predictions match official submission CSVs."""
    checks: dict[str, Any] = {}
    sub_pairs = [
        ("v85_forensic_strict_cv", ROOT / "submissions" / "submission_v85_forensic_strict_cv.csv"),
        ("v7_6family_balanced_xdx", ROOT / "submissions" / "submission_v7_6family_balanced_xdx.csv"),
        ("v7_6family_local_newton_xdx", ROOT / "submissions" / "submission_v7_6family_local_newton_xdx.csv"),
        (
            "v85_reteam_count",
            ROOT / "_local" / "v85sc-count" / "submissions" / "submission_v85_forensic_strict_cv.csv",
        ),
    ]
    for name, csv_path in sub_pairs:
        if not csv_path.is_file() or name not in suite.baselines:
            checks[name] = {"verified": False, "reason": "missing_file_or_baseline"}
            continue
        sub_codes = ensure_int_codes(csv_path)
        rec_codes = ensure_int_codes(suite.baselines[name]["pred_atd_rep"])
        graded = suite.atd_graded
        exact_all = float(np.mean(sub_codes == rec_codes))
        exact_graded = float(np.mean(sub_codes[graded] == rec_codes[graded]))
        n_diff_graded = int(np.sum(sub_codes[graded] != rec_codes[graded]))
        checks[name] = {
            "csv_path": str(csv_path.relative_to(ROOT)),
            "n_total_frames": int(len(sub_codes)),
            "n_graded_frames": int(graded.sum()),
            "exact_match_rate_all": round(exact_all, 6),
            "exact_match_rate_graded": round(exact_graded, 6),
            "n_diff_graded_frames": n_diff_graded,
            "exact_100pct_graded": bool(n_diff_graded == 0),
        }
    return checks


def _build_markdown_summary(
    evals: dict[str, dict[str, Any]],
    v85_vs_v7_eval: dict[str, Any],
    sub_checks: dict[str, Any],
    elapsed_s: float,
) -> str:
    """Build a detailed Markdown summary of the baseline audit and where v8.5 loses points."""
    v85 = evals["v85_forensic_strict_cv"]
    v7b = evals["v7_6family_balanced_xdx"]
    v7n = evals["v7_6family_local_newton_xdx"]
    v85_rt = evals["v85_reteam_count"]

    lines: list[str] = []
    lines.append("# Generalization Judge Suite (`J_SUITE`) & Error Atlas Baseline Audit")
    lines.append("")
    lines.append(f"- **Runtime**: `{elapsed_s:.2f}s` (`n_boot=1000` paired bootstrap over matches & halves)")
    lines.append("- **Output JSON**: `experiments/outputs/v9_swarm/j_suite_baseline.json`")
    lines.append("")

    # Section 0: Verification against official submissions
    lines.append("## 0. Official Submission Equivalence Check (`BAR-ATH`, `158,857` graded frames)")
    lines.append("")
    lines.append("| Candidate | Official CSV | Graded Frames | Graded Exact Match | Diff Frames |")
    lines.append("|---|---|---:|---:|---:|")
    for k, chk in sub_checks.items():
        if chk.get("exact_match_rate_graded") is not None:
            lines.append(
                f"| `{k}` | `{chk['csv_path']}` | {chk['n_graded_frames']:,} | "
                f"`{chk['exact_match_rate_graded']*100:.4f}%` | `{chk['n_diff_graded_frames']}` |"
            )
    lines.append("")

    # Section 1: Standard Judges + LiveOnly / StoppageOnly Decomposition
    lines.append("## 1. Standard Judges & Live Turnover vs Stoppage Boundary Decomposition")
    lines.append("")
    lines.append(
        "| Judge | Description | `v7_balanced` | `v7_local_newton` | `v85_strict_cv` | "
        "`Δ(v8.5 - v7_bal)` | `90% CI (Match)` | `90% CI (Half)` | Folds `v8.5 > v7` | Halves `v8.5 > v7` |"
    )
    lines.append("|---|---|---:|---:|---:|---:|---|---|---:|---:|")

    judge_descriptions = {
        "J1A": "Sportec 7m `degrade_v3 s0` (All)",
        "J1A_LiveOnly": "Sportec 7m `v3 s0` (**GT Dead Fixed** -> pure `H<->A`)",
        "J1A_noD": "Sportec 7m `v3 s0` (`without_dead_margin`)",
        "J1A_StoppageOnly": "Sportec 7m `v3 s0` (**GT Team Fixed** -> pure `Live<->D`)",
        "J1B_Std": "Sportec 7m `degrade_v6 s0` (Standard decoder)",
        "J1B_Gen": "Sportec 7m `degrade_v6 s0` (Generalist decoder)",
        "J2_Raw": "Metrica Sample 1-3 Raw (has `NaN=D` artifact)",
        "J2_noD": "Metrica Sample 1-3 `without_dead_margin`",
        "J2_LiveOnly": "Metrica Sample 1-3 (**GT Dead Fixed** -> pure `H<->A`)",
        "J2_StoppageOnly": "Metrica Sample 1-3 (**GT Team Fixed** -> pure `Live<->D`)",
    }

    for jkey, desc in judge_descriptions.items():
        s_v7b = v7b["judges"][jkey]["spell_f1_mean"]
        s_v7n = v7n["judges"][jkey]["spell_f1_mean"]
        s_v85 = v85["judges"][jkey]["spell_f1_mean"]
        jd_comp = v85_vs_v7_eval["judges"][jkey]
        dm = jd_comp["paired_vs_v85_match"]
        dh = jd_comp["paired_vs_v85_half"]
        lines.append(
            f"| **{jkey}** | {desc} | `{s_v7b:.4f}` | `{s_v7n:.4f}` | **`{s_v85:.4f}`** | "
            f"`{dm['mean_delta']:+.4f}` | `[{dm['ci_low']:+.4f}, {dm['ci_high']:+.4f}]` | "
            f"`[{dh['ci_low']:+.4f}, {dh['ci_high']:+.4f}]` | `{dm['improved']}/{dm['n_units']}` | `{dh['improved']}/{dh['n_units']}` |"
        )
    lines.append("")

    # Detailed metrics table for v8.5 across all judges
    lines.append("### `v85_forensic_strict_cv` Detailed Metrics Across All Judges")
    lines.append("")
    lines.append(
        "| Judge | `spell_f1_mean` | `macro_f1` | `F1@0.3` | `F1@0.5` | `F1@0.7` | "
        "`P@0.5` | `R@0.5` | `poss/min` (GT) | `% < 2.0s` (GT) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for jkey in judge_descriptions:
        sm = v85["judges"][jkey]
        lines.append(
            f"| **{jkey}** | `{sm['spell_f1_mean']:.4f}` | `{sm['macro_f1']:.4f}` | "
            f"`{sm['F1@0.3']:.4f}` | `{sm['F1@0.5']:.4f}` | `{sm['F1@0.7']:.4f}` | "
            f"`{sm['P@0.5']:.4f}` | `{sm['R@0.5']:.4f}` | "
            f"`{sm['poss_per_min']:.2f}` (`{sm['gt_poss_per_min']:.2f}`) | "
            f"`{sm['pct_lt_2s']:.1f}%` (`{sm['gt_pct_lt_2s']:.1f}%`) |"
        )
    lines.append("")

    # Section 2: Where v8.5 Loses Points - Oracle Headroom
    lines.append("## 2. Oracle Headroom Decomposition on `v85_forensic_strict_cv` (Where `v8.5` Loses Points)")
    lines.append("")
    lines.append(
        "| Oracle Intervention | `J1A` (`v3 s0`) `spell_f1` | `J1A` Headroom Gain | `J2-Raw` `spell_f1` | `J2-Raw` Headroom Gain |"
    )
    lines.append("|---|---:|---:|---:|---:|")
    oh_j1 = v85["oracle_headroom"]["J1A"]
    oh_j2 = v85["oracle_headroom"]["J2_Raw"]
    lines.append(
        f"| **Baseline (`v8.5`)** | `{oh_j1['base_spell_f1']:.4f}` | `+0.0000` | `{oh_j2['base_spell_f1']:.4f}` | `+0.0000` |"
    )
    for okey, orec1 in oh_j1["oracles"].items():
        orec2 = oh_j2["oracles"][okey]
        lines.append(
            f"| `{okey}` | `{orec1['repaired_spell_f1']:.4f}` | **`{orec1['headroom_delta']:+.4f}`** | "
            f"`{orec2['repaired_spell_f1']:.4f}` | **`{orec2['headroom_delta']:+.4f}`** |"
        )
    lines.append("")

    # Section 3: Duration Recall Breakdown
    lines.append("## 3. Possession Duration Recall Breakdown (`IoU >= 0.5` & `IoU >= 0.3`)")
    lines.append("")
    lines.append("### 3A. `J1A` (7 Sportec Matches, `2,803` GT Live Possessions)")
    lines.append("")
    lines.append(
        "| Duration Bucket | GT Spells | `v7_bal` Recall@0.5 | `v8.5` Recall@0.5 | `v8.5` Misses@0.5 | "
        "Share of All `v8.5` Misses | `v8.5` Recall@0.3 | `v8.5` Recall@0.7 | Mean IoU |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    v7_dur_j1 = {r["dur_bucket"]: r for r in v7b["error_atlas"]["J1A"]["by_duration"]}
    for r in v85["error_atlas"]["J1A"]["by_duration"]:
        db = r["dur_bucket"]
        r7 = v7_dur_j1[db]
        lines.append(
            f"| **{db}** | {r['n']} | `{r7['recall_05']*100:.1f}%` | **`{r['recall_05']*100:.1f}%`** | "
            f"{r['missed_05']} | `{r['share_of_all_misses_05']*100:.1f}%` | "
            f"`{r['recall_03']*100:.1f}%` | `{r['recall_07']*100:.1f}%` | `{r['mean_iou']:.3f}` |"
        )
    lines.append("")
    lines.append("#### Fine-Grained Duration Buckets on `J1A` (Separating Post-Processor `<34f` Floor vs Decodable `34-49f`)")
    lines.append("")
    lines.append("| Fine Duration Bucket | GT Spells | `v8.5` Recall@0.5 | `v8.5` Misses@0.5 | Share of All Misses | Mean IoU |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for r in v85["error_atlas"]["J1A"]["by_duration_fine"]:
        lines.append(
            f"| `{r['dur_fine_bucket']}` | {r['n']} | **`{r['recall_05']*100:.1f}%`** | "
            f"{r['missed_05']} | `{r['share_of_all_misses_05']*100:.1f}%` | `{r['mean_iou']:.3f}` |"
        )
    lines.append("")

    lines.append("### 3B. `J2-LiveOnly` & `J2-Raw` (3 Metrica Matches, `701` GT Live Possessions)")
    lines.append("")
    lines.append(
        "| Duration Bucket | GT Spells | `J2-Raw` (`v7_bal`) Rec@0.5 | `J2-Raw` (`v8.5`) Rec@0.5 | "
        "`J2-LiveOnly` (`v8.5`) Rec@0.5 | `J2-LiveOnly` Misses@0.5 | Share of `J2-LiveOnly` Misses |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    v7_dur_j2 = {r["dur_bucket"]: r for r in v7b["error_atlas"]["J2_Raw"]["by_duration"]}
    v85_dur_j2_raw = {r["dur_bucket"]: r for r in v85["error_atlas"]["J2_Raw"]["by_duration"]}
    for r in v85["error_atlas"]["J2_LiveOnly"]["by_duration"]:
        db = r["dur_bucket"]
        r7 = v7_dur_j2[db]
        r85_raw = v85_dur_j2_raw[db]
        lines.append(
            f"| **{db}** | {r['n']} | `{r7['recall_05']*100:.1f}%` | `{r85_raw['recall_05']*100:.1f}%` | "
            f"**`{r['recall_05']*100:.1f}%`** | {r['missed_05']} | `{r['share_of_all_misses_05']*100:.1f}%` |"
        )
    lines.append("")

    # Section 4: Event-Play Error Atlas
    lines.append("## 4. Event-Play Error Atlas (`compute_error_atlas`)")
    lines.append("")
    lines.append("### 4A. `J1A` DFL Event Categories (7 Sportec Matches, `v8.5` vs `v7_balanced`)")
    lines.append("")
    lines.append(
        "| DFL Event Kind | GT Spells | `% < 2s` | `v7_bal` Miss@0.5 | `v8.5` Miss@0.5 | "
        "`v8.5` Missed Spells | Share of All `v8.5` Misses | Median Start Err (f) | P90 |Start Err| (f) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    v7_ev_j1 = {r["kind"]: r for r in v7b["error_atlas"]["J1A"]["by_event_kind"]}
    for r in v85["error_atlas"]["J1A"]["by_event_kind"]:
        k = r["kind"]
        r7 = v7_ev_j1.get(k, {})
        m7 = f"{r7['miss_rate_05']*100:.1f}%" if "miss_rate_05" in r7 else "N/A"
        lines.append(
            f"| `{k}` | {r['n']} | `{r['short_share']*100:.1f}%` | `{m7}` | **`{r['miss_rate_05']*100:.1f}%`** | "
            f"{r['missed_05']} | `{r['share_of_all_misses_05']*100:.1f}%` | "
            f"`{r['median_start_err_f']}` | `{r['p90_abs_start_err_f']}` |"
        )
    lines.append("")

    lines.append("### 4B. `J2-Raw` & `J2-LiveOnly` Metrica Event Categories (3 Metrica Matches)")
    lines.append("")
    lines.append(
        "| Metrica Event Macro Group | GT Spells | `% < 2s` | `J2-Raw` (`v7_bal`) Miss@0.5 | "
        "`J2-Raw` (`v8.5`) Miss@0.5 | `J2-LiveOnly` (`v8.5`) Miss@0.5 | `J2-LiveOnly` Missed | Share of `J2-LiveOnly` Misses | Median Start Err (f) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    v7_mg_j2 = {r["macro_group"]: r for r in v7b["error_atlas"]["J2_Raw"]["by_macro_group"]}
    v85_mg_j2_raw = {r["macro_group"]: r for r in v85["error_atlas"]["J2_Raw"]["by_macro_group"]}
    for r in v85["error_atlas"]["J2_LiveOnly"]["by_macro_group"]:
        mg = r["macro_group"]
        r7 = v7_mg_j2.get(mg, {})
        r85_raw = v85_mg_j2_raw.get(mg, {})
        lines.append(
            f"| **{mg}** | {r['n']} | `{r['short_share']*100:.1f}%` | `{r7.get('miss_rate_05', 0)*100:.1f}%` | "
            f"`{r85_raw.get('miss_rate_05', 0)*100:.1f}%` | **`{r['miss_rate_05']*100:.1f}%`** | "
            f"{r['missed_05']} | `{r['share_of_all_misses_05']*100:.1f}%` | `{r['median_start_err_f']}` |"
        )
    lines.append("")

    lines.append("### 4C. Stoppage Start Lag (`Live -> D` transitions on `J1A`, `25 fps`)")
    lines.append("")
    lines.append("| Stoppage Cause | GT Stoppages | Median Lag (f) | Mean |Lag| (f) | Late `> 1.0s` (`> +25f`) | Early `< -1.0s` | Missed Stoppage |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in v85["error_atlas"]["J1A"]["dead_starts"]:
        med_str = f"{r['median_lag_f']:+.1f}f" if r["median_lag_f"] is not None else "N/A"
        abs_str = f"{r['mean_abs_lag_f']:.1f}f" if r["mean_abs_lag_f"] is not None else "N/A"
        lines.append(
            f"| `{r['kind']}` | {r['n']} | `{med_str}` | `{abs_str}` | "
            f"**`{r['late_over_1s']*100:.1f}%`** | `{r['early_over_1s']*100:.1f}%` | `{r['missed']*100:.1f}%` |"
        )
    lines.append("")

    lines.append("### 4D. Spurious Predicted Possessions & Domino Merges (`v8.5` vs `v7_balanced`)")
    lines.append("")
    sp_j1_85 = v85["error_atlas"]["J1A"]["spurious_and_merges"]
    sp_j1_7b = v7b["error_atlas"]["J1A"]["spurious_and_merges"]
    sp_j2_85 = v85["error_atlas"]["J2_LiveOnly"]["spurious_and_merges"]
    sp_j2_7b = v7b["error_atlas"]["J2_LiveOnly"]["spurious_and_merges"]
    lines.append("| Judge / Model | Total Pred Spells | Spurious (`IoU < 0.5`) | Spurious (`IoU < 0.3`) | Domino Merges (`1 pred -> >=2 GT`) | Swallowed GT Spells |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(
        f"| `J1A` (`v7_balanced`) | {sp_j1_7b['n_pred_spells']} | {sp_j1_7b['spurious_05_count']} (`{sp_j1_7b['spurious_05_rate']*100:.1f}%`) | "
        f"{sp_j1_7b['spurious_03_count']} (`{sp_j1_7b['spurious_03_rate']*100:.1f}%`) | {sp_j1_7b['merges_count']} | {sp_j1_7b['swallowed_true_spells']} |"
    )
    lines.append(
        f"| `J1A` (`v85_strict_cv`) | {sp_j1_85['n_pred_spells']} | {sp_j1_85['spurious_05_count']} (`{sp_j1_85['spurious_05_rate']*100:.1f}%`) | "
        f"{sp_j1_85['spurious_03_count']} (`{sp_j1_85['spurious_03_rate']*100:.1f}%`) | {sp_j1_85['merges_count']} | {sp_j1_85['swallowed_true_spells']} |"
    )
    lines.append(
        f"| `J2-LiveOnly` (`v7_balanced`) | {sp_j2_7b['n_pred_spells']} | {sp_j2_7b['spurious_05_count']} (`{sp_j2_7b['spurious_05_rate']*100:.1f}%`) | "
        f"{sp_j2_7b['spurious_03_count']} (`{sp_j2_7b['spurious_03_rate']*100:.1f}%`) | {sp_j2_7b['merges_count']} | {sp_j2_7b['swallowed_true_spells']} |"
    )
    lines.append(
        f"| `J2-LiveOnly` (`v85_strict_cv`) | {sp_j2_85['n_pred_spells']} | {sp_j2_85['spurious_05_count']} (`{sp_j2_85['spurious_05_rate']*100:.1f}%`) | "
        f"{sp_j2_85['spurious_03_count']} (`{sp_j2_85['spurious_03_rate']*100:.1f}%`) | {sp_j2_85['merges_count']} | {sp_j2_85['swallowed_true_spells']} |"
    )
    lines.append("")

    # Section 5: BAR-ATH Full-Match Unsupervised Physical & Multi-Modal Consistency
    lines.append("## 5. Full-Match Unsupervised Physical & Multi-Modal Consistency on `BAR-ATH` (`158,857` Graded Frames)")
    lines.append("")
    lines.append(
        "| Metric / Audit Check | `v7_6family_balanced` | `v7_6family_newton` | `v85_forensic_strict_cv` | `v85_reteam_count` (`D3`) |"
    )
    lines.append("|---|---:|---:|---:|---:|")

    rows_ba = [
        ("Graded `poss/min` (`158,857f`)", lambda e: f"`{e['bar_ath_full_match']['plausibility']['full_graded_158857f']['poss_per_min']:.2f}`"),
        ("Private Unseen `poss/min` (`151,357f`)", lambda e: f"`{e['bar_ath_full_match']['plausibility']['unseen_private_151357f']['poss_per_min']:.2f}`"),
        ("Graded Live Spells Count", lambda e: f"`{e['bar_ath_full_match']['plausibility']['full_graded_158857f']['n_live_spells']}`"),
        ("Graded `% < 2.0s`", lambda e: f"`{e['bar_ath_full_match']['plausibility']['full_graded_158857f']['pct_lt_2s']:.2f}%`"),
        ("Graded `% Dead (D)`", lambda e: f"`{e['bar_ath_full_match']['plausibility']['full_graded_158857f']['pct_D']:.2f}%`"),
        ("Graded `% Home of Live` (`H / (H+A)`)", lambda e: f"`{e['bar_ath_full_match']['plausibility']['full_graded_158857f']['pct_H_of_live']:.2f}%`"),
        ("Smart Tagging `in_phase` Live Team Agree (`75,771f`)", lambda e: f"`{e['bar_ath_full_match']['smart_tagging_consistency']['full_graded_158857f']['in_phase_live_team_agree_pct']:.2f}%`"),
        ("Smart Tagging `in_phase` Pred Dead Rate (`84,782f`)", lambda e: f"`{e['bar_ath_full_match']['smart_tagging_consistency']['full_graded_158857f']['pct_D_inside_phases']:.2f}%`"),
        ("Smart Tagging `SET PIECES` Pre-Kick `D` Agree", lambda e: f"`{e['bar_ath_full_match']['smart_tagging_consistency']['full_graded_158857f']['set_piece_pre_kick_D_agree_pct']:.2f}%`"),
        ("Smart Tagging `SET PIECES` Post-Kick Team Agree (When Alive)", lambda e: f"`{e['bar_ath_full_match']['smart_tagging_consistency']['full_graded_158857f']['set_piece_post_kick_team_when_alive_pct']:.2f}%`"),
        ("Physical Kick Restart Median Lag (f)", lambda e: f"`{e['bar_ath_full_match']['smart_tagging_consistency']['physical_kick_restart_alignment']['median_restart_lag_f']}f`"),
        ("Agreement vs `v85_strict_cv` (All `158,857f`)", lambda e: f"`{e['bar_ath_full_match']['reference_agreements']['vs_v85_strict_cv']['frame_agreement_pct']:.2f}%`"),
        ("Agreement vs `v85_strict_cv` (`spell_f1_mean`)", lambda e: f"`{e['bar_ath_full_match']['reference_agreements']['vs_v85_strict_cv']['spell_f1_agreement']:.4f}`"),
        ("Agreement vs `D3` Re-Teamed `v8.5` (All `158,857f`)", lambda e: f"`{e['bar_ath_full_match']['reference_agreements']['vs_d3_v85_reteam_count']['frame_agreement_pct']:.2f}%`"),
        ("Agreement vs `D3` Re-Teamed `v8.5` (`spell_f1_mean`)", lambda e: f"`{e['bar_ath_full_match']['reference_agreements']['vs_d3_v85_reteam_count']['spell_f1_agreement']:.4f}`"),
        ("Changed Within `±1s` of D3 Re-Team Stretch", lambda e: f"`{e['bar_ath_full_match']['reference_agreements']['vs_v85_strict_cv']['changed_within_1s_of_reteam_stretch_pct']:.1f}%`"),
        ("Exact `H <-> A` Equivariance (`frame_mismatches`)", lambda e: f"`{e['equivariance'].get('frame_mismatches', 0)}` (`{e['equivariance'].get('verified', True)}`)"),
        ("`J3` (5-min Public Window `spell_f1_mean`) **[REPORT-ONLY]**", lambda e: f"`{e['bar_ath_full_match']['J3_public_5min_REPORT_ONLY']['spell_f1_mean']:.4f}`"),
    ]
    for label, fn in rows_ba:
        lines.append(f"| {label} | {fn(v7b)} | {fn(v7n)} | **{fn(v85)}** | {fn(v85_rt)} |")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    print("=" * 116)
    print("INITIALIZING GENERALIZATION JUDGE SUITE (J_SUITE) & ERROR ATLAS")
    print("=" * 116)
    suite = JSuite(n_boot=1000, seed=42, verbose=True)

    # Verify submission equivalence first
    sub_checks = _verify_submission_equivalence(suite)
    print("\n[Submission Equivalence Check on BAR-ATH (158,857 graded frames)]")
    for name, info in sub_checks.items():
        print(f"  - {name:<28}: {info}")

    # Evaluate baselines against v85_forensic_strict_cv
    candidates = [
        "v7_6family_balanced_xdx",
        "v7_6family_local_newton_xdx",
        "v85_forensic_strict_cv",
        "v85_reteam_count",
    ]

    evals: dict[str, dict[str, Any]] = {}
    for cand_name in candidates:
        print(f"\n>>> Evaluating `{cand_name}` vs `v85_forensic_strict_cv` ...")
        res = suite.evaluate_candidate(
            suite.baselines[cand_name],
            candidate_name=cand_name,
            reference_name="v85_forensic_strict_cv",
            compute_oracles=(cand_name in ("v85_forensic_strict_cv", "v7_6family_balanced_xdx")),
            verify_equivariance=True,
        )
        evals[cand_name] = res
        print(format_j_suite_report(res))

    # Also evaluate v85_forensic_strict_cv directly against v7_6family_balanced_xdx
    # so we have exact paired bootstrap CIs for (v8.5 - v7_balanced)
    print("\n>>> Computing direct paired bootstrap comparison: `v85_forensic_strict_cv` vs `v7_6family_balanced_xdx` ...")
    v85_vs_v7_eval = suite.evaluate_candidate(
        suite.baselines["v85_forensic_strict_cv"],
        candidate_name="v85_forensic_strict_cv_vs_v7_balanced",
        reference_name="v7_6family_balanced_xdx",
        compute_oracles=False,
        verify_equivariance=False,
    )
    print(format_j_suite_report(v85_vs_v7_eval))

    elapsed_s = time.time() - t0

    # Save complete baseline JSON audit
    SWARM_OUT_DIR.mkdir(parents=True, exist_ok=True)
    json_out_path = SWARM_OUT_DIR / "j_suite_baseline.json"
    md_out_path = SWARM_OUT_DIR / "j_suite_baseline_summary.md"

    payload = {
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_sec": round(elapsed_s, 2),
            "n_boot": 1000,
            "seed": 42,
            "candidates_evaluated": candidates,
        },
        "submission_equivalence_checks": sub_checks,
        "evaluations_vs_v85": evals,
        "v85_vs_v7_balanced_paired_comparison": {
            "judges": v85_vs_v7_eval["judges"],
            "error_atlas_deltas": {
                k: v.get("paired_diff_vs_v85")
                for k, v in v85_vs_v7_eval["error_atlas"].items()
                if isinstance(v, dict) and "paired_diff_vs_v85" in v
            },
        },
    }

    json_out_path.write_text(json.dumps(payload, indent=2))
    md_summary = _build_markdown_summary(evals, v85_vs_v7_eval, sub_checks, elapsed_s)
    md_out_path.write_text(md_summary)

    print("\n" + "=" * 116)
    print(f"SAVED JSON AUDIT TO: {json_out_path}")
    print(f"SAVED MARKDOWN SUMMARY TO: {md_out_path}")
    print(f"TOTAL RUNTIME: {elapsed_s:.2f}s")
    print("=" * 116)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
