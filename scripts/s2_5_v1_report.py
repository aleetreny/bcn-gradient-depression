"""S2.5: check submission v1 on BAR-ATH: validity, public score, plausibility panel, set pieces, visual review.

    .venv311/bin/python scripts/s2_5_v1_report.py [--sub submissions/v1.csv] [--out experiments/outputs/020]

- validity with the official evaluation/validate_submission.py; public score with bootstrap (1000);
- plausibility panel by 15-minute stretch of each half: possessions/min, % D, median possession,
  % < 2 s, for BAR-ATH v1 and the baseline, next to the same system on Sportec (out-of-fold S2.3
  probabilities decoded with configs/decoder_v1.json) and the Sportec truth;
- set pieces: share of Smart Tagging SET PIECES instances with predicted D inside [start - 10 s, end]
  (coverage only: an always-D prediction would score 100 %; it is not a detection precision);
- visual review: three random 2-minute windows (seed 5) of graded frames outside the public window.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import DEAD, FPS, possession_spells  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics, plots  # noqa: E402
from src.models.baseline import predict as baseline  # noqa: E402
from starter.load_smart_tagging import load as load_tags  # noqa: E402

SEG = 15 * 60 * FPS
PUBLIC = (11185, 18685)


def panel(pred, half, graded):
    rows = []
    for h in (1, 2):
        idx = np.flatnonzero((half == h) & graded)
        for k, a in enumerate(range(0, len(idx), SEG)):
            sl = idx[a:a + SEG]
            if len(sl) < 5 * 60 * FPS:
                continue
            p = np.full(len(pred), -1, np.int8)
            p[sl] = pred[sl]
            s, e, _ = possession_spells(p, half)
            d = (e - s) / FPS
            rows.append({"half": h, "segment": "{}-{}'".format(k * 15, k * 15 + 15), "poss_per_min": len(s) / (len(sl) / FPS / 60),
                         "pct_D": 100 * float(np.mean(pred[sl] == DEAD)),
                         "median_poss_s": float(np.median(d)) if len(d) else np.nan,
                         "pct_poss_lt_2s": 100 * float(np.mean(d < 2)) if len(d) else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sub", type=Path, default=ROOT / "submissions/v1.csv")
    parser.add_argument("--oof", type=Path, default=ROOT / "local_data/oof/s2_3_d3_degrade_v2")
    parser.add_argument("--decoder-config", type=Path, default=ROOT / "configs/decoder_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/020")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    valid = subprocess.run([sys.executable, "evaluation/validate_submission.py", str(args.sub)], cwd=ROOT,
                           capture_output=True, text=True)
    pub = metrics.score_public(args.sub, bootstrap_n=1000)
    atd = load_atd()
    v1 = metrics.pm.label_codes(metrics.pm.read_label_csv(str(args.sub)))
    v1 = np.where(v1 == 3, DEAD, v1).astype(np.int8)
    base = baseline(atd, 50)

    dec = DecoderParams(**{k: v for k, v in json.loads(args.decoder_config.read_text()).items() if not k.startswith("_")})
    sp_pred, sp_true = [], []
    for mid, m in load_all().items():
        sp_pred.append(panel(decode(np.load(args.oof / "{}.npy".format(mid)), m.half, dec), m.half, m.half > 0).assign(match=mid))
        sp_true.append(panel(m.labels, m.half, m.half > 0).assign(match=mid))
    cols = ["poss_per_min", "pct_D", "median_poss_s", "pct_poss_lt_2s"]
    p_v1 = panel(v1, atd.half, atd.graded).set_index(["half", "segment"])[cols]
    p_base = panel(base, atd.half, atd.graded).set_index(["half", "segment"])[cols]
    p_sp = pd.concat(sp_pred).groupby(["half", "segment"])[cols].mean()
    p_spt = pd.concat(sp_true).groupby(["half", "segment"])[cols].mean()
    overall = pd.DataFrame({
        "BAR-ATH v1": panel(v1, atd.half, atd.graded)[cols].mean(),
        "BAR-ATH baseline": panel(base, atd.half, atd.graded)[cols].mean(),
        "Sportec, same system (OOF)": pd.concat(sp_pred)[cols].mean(),
        "Sportec truth": pd.concat(sp_true)[cols].mean()}).T

    tags = [t for t in load_tags() if t["code"].upper() == "SET PIECES" and atd.graded[t["start_frame"]]]
    hit = [bool(np.any(v1[max(0, t["start_frame"] - 10 * FPS):t["end_frame"]] == DEAD)) for t in tags]
    hit_base = [bool(np.any(base[max(0, t["start_frame"] - 10 * FPS):t["end_frame"]] == DEAD)) for t in tags]
    by_type = pd.DataFrame({"type": [t["tags"].get("Type") for t in tags], "v1": hit, "baseline": hit_base}).groupby("type").mean()

    rng = np.random.default_rng(5)
    candidates = np.flatnonzero(atd.graded)
    candidates = candidates[(candidates < PUBLIC[0] - 120 * FPS) | (candidates > PUBLIC[1])]
    windows = []
    while len(windows) < 3:
        a = int(rng.choice(candidates))
        if atd.graded[a:a + 120 * FPS].all() and all(abs(a - w) > 120 * FPS for w in windows):
            windows.append(a)
    sp_mask = np.zeros(atd.n_frames, bool)
    for t in tags:
        sp_mask[t["start_frame"]:t["end_frame"]] = True
    for i, a in enumerate(sorted(windows)):
        plots.timeline(args.out / "window_{}.png".format(i + 1), [("v1", v1), ("baseline", base)], a, a + 120 * FPS,
                       title="BAR-ATH random window {} (video frames {}-{}): no truth here, visual check".format(i + 1, a, a + 120 * FPS),
                       extra=[("ball visible", atd.ball_xy_visible), ("SET PIECES tag", sp_mask)])

    pd.concat({"BAR-ATH v1": p_v1, "BAR-ATH baseline": p_base, "Sportec same system": p_sp, "Sportec truth": p_spt}, axis=1).to_csv(
        args.out / "panel_by_15min.csv", float_format="%.2f")
    overall.to_csv(args.out / "panel_overall.csv", float_format="%.2f")
    by_type.to_csv(args.out / "set_pieces_with_D.csv", float_format="%.3f")
    summary = {"valid": valid.stdout.strip() + valid.stderr.strip(), "public": metrics.headline(pub),
               "set_pieces_with_predicted_D": {"v1": float(np.mean(hit)), "baseline": float(np.mean(hit_base)), "n": len(tags)},
               "windows": sorted(windows)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
        print(json.dumps(summary, indent=1, default=float))
        print(overall)
        print(pd.concat({"v1": p_v1[["poss_per_min", "pct_D"]], "Sportec system": p_sp[["poss_per_min", "pct_D"]],
                         "Sportec truth": p_spt[["poss_per_min", "pct_D"]]}, axis=1))
        print(by_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
