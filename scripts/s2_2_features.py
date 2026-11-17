"""S2.2: features v1 on every source; % NaN by group and source, and domain shift per feature.

    .venv311/bin/python scripts/s2_2_features.py [--out experiments/outputs/016]

Builds (and caches, local_data/cache/features_v1/) the v1 features for BAR-ATH, the 7 Sportec matches
clean, and degraded with configs/degrade_v1.json (seed 0). Reports % NaN per feature group and
source (graded frames), and the Kolmogorov-Smirnov distance between BAR-ATH and degraded Sportec per
feature (every 10th graded frame), flagging the features whose distribution transfers worst.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from src.data.degrade import VERSION as DEG_VERSION, degrade  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.features.v1 import cached_features, feature_group  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/016")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v2.json")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    tag = "d{}_{}".format(DEG_VERSION, args.config.stem)

    atd = load_atd()
    sources = {"BAR-ATH": [cached_features(atd, "atd").iloc[np.flatnonzero(atd.graded)]]}
    sources["Sportec clean"], sources["Sportec degraded v1"] = [], []
    for mid, m in load_all().items():
        sources["Sportec clean"].append(cached_features(m, "sportec_{}_clean".format(mid)))
        sources["Sportec degraded v1"].append(cached_features(degrade(m, cfg, 0), "sportec_{}_{}_s0".format(mid, tag)))
    data = {k: pd.concat(v, ignore_index=True) for k, v in sources.items()}
    cols = list(data["BAR-ATH"].columns)
    groups = pd.Series({c: feature_group(c) for c in cols})

    nan = pd.DataFrame({k: d.isna().mean() * 100 for k, d in data.items()})
    nan["group"] = groups
    by_group = nan.groupby("group").agg(["mean"]).round(1)
    by_group.columns = [c[0] for c in by_group.columns]
    by_group.insert(0, "features", groups.value_counts())

    a, d = data["BAR-ATH"].iloc[::10], data["Sportec degraded v1"].iloc[::10]
    c = data["Sportec clean"].iloc[::10]
    ks = pd.DataFrame({"group": groups,
                       "ks_atd_vs_degraded": [ks_2samp(a[k].dropna(), d[k].dropna()).statistic if a[k].notna().any() and d[k].notna().any() else np.nan for k in cols],
                       "ks_atd_vs_clean": [ks_2samp(a[k].dropna(), c[k].dropna()).statistic if a[k].notna().any() and c[k].notna().any() else np.nan for k in cols]},
                      index=cols)
    ks["improvement"] = ks["ks_atd_vs_clean"] - ks["ks_atd_vs_degraded"]
    ks_group = ks.groupby("group")[["ks_atd_vs_clean", "ks_atd_vs_degraded"]].median().round(3)

    nan.to_csv(args.out / "nan_by_feature.csv", float_format="%.2f")
    by_group.to_csv(args.out / "nan_by_group.csv")
    ks.sort_values("ks_atd_vs_degraded", ascending=False).to_csv(args.out / "domain_shift_ks.csv", float_format="%.3f")
    (args.out / "summary.json").write_text(json.dumps({"n_features": len(cols), "rows": {k: len(v) for k, v in data.items()}}, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 10):
        print("features:", len(cols))
        print(by_group)
        print("\nmedian KS by group (0 = same distribution, 1 = disjoint)")
        print(ks_group)
        print("\nworst-transferring features (ATD vs degraded)")
        print(ks.sort_values("ks_atd_vs_degraded", ascending=False).head(15).round(3))
        print("\nbest-transferring features")
        print(ks.sort_values("ks_atd_vs_degraded").head(10).round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
