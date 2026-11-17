"""S2.6: where does v1 still lose spellF1? Error decomposition and oracle repairs of v1 (as S1.7 for the baseline).

    .venv311/bin/python scripts/s2_6_v1_errors.py [--out experiments/outputs/021]

Sportec: v1 out of fold (S2.3 OOF probabilities decoded with configs/decoder_v1.json, tuned on the
same 7 matches -- optimistic), or with --nested the fully nested per-fold predictions of S2.4b.
Public: submissions/v1.csv. Same oracles as S1.7 (src/eval/oracles.py), one error type at a time,
and the error table of src/eval/errors.py pooled over the 7 matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.common import DEAD  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.sportec import load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import errors, metrics, oracles  # noqa: E402

COMBOS = {"perfect D + timing": ("perfect D", "perfect change timing (±3 s)")}


def apply(name, gt, pred, half):
    if name in COMBOS:
        for n in COMBOS[name]:
            pred = oracles.ORACLES[n](gt, pred, half)
        return pred
    return oracles.ORACLES[name](gt, pred, half)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oof", type=Path, default=ROOT / "local_data/oof/s2_3_d3_degrade_v2")
    parser.add_argument("--decoder-config", type=Path, default=ROOT / "configs/decoder_v1.json")
    parser.add_argument("--sub", type=Path, default=ROOT / "submissions/v1.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/021")
    parser.add_argument("--nested", type=Path, default=None,
                        help="local_data/oof/nested_<tag>: use each fold's own decoded prediction (S2.4b) "
                             "instead of decoding the OOF probabilities with the all-7-match decoder")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dec = DecoderParams(**{k: v for k, v in json.loads(args.decoder_config.read_text()).items() if not k.startswith("_")})
    names = list(oracles.ORACLES) + list(COMBOS)

    per, tables, offsets = {n: [] for n in ["v1"] + names}, [], []
    for mid, m in load_all().items():
        if args.nested:
            pred = np.load(args.nested / mid / "pred_{}.npy".format(mid))
        else:  # decoder tuned on the same 7 matches (review 003, point 4): optimistic diagnostic
            pred = decode(np.load(args.oof / "{}.npy".format(mid)), m.half, dec)
        per["v1"].append(metrics.score(m.labels, pred, m.half)["spell_f1_mean"])
        for n in names:
            per[n].append(metrics.score(m.labels, apply(n, m.labels, pred, m.half), m.half)["spell_f1_mean"])
        d = errors.decompose(m.labels, pred, m.half)
        tables.append(d["table"].set_index("item")["count"].rename(mid))
        offsets.append(d["team_change_offsets"])
    base = np.array(per["v1"])

    atd = load_atd()
    gt = metrics.public_gt()
    gt = np.where(gt == 3, -1, gt).astype(np.int8)
    v1 = metrics.pm.label_codes(metrics.pm.read_label_csv(str(args.sub)))
    v1 = np.where(v1 == 3, DEAD, v1).astype(np.int8)
    pub = {"v1": metrics.score_public(v1, bootstrap_n=0)["spell_f1_mean"]}
    for n in names:
        pub[n] = metrics.score_public(apply(n, gt, v1, atd.half), bootstrap_n=0)["spell_f1_mean"]
    dec_pub = errors.decompose(gt, v1, atd.half)

    rows = []
    for n in ["v1"] + names:
        v = np.array(per[n])
        rows.append({"repair": n, "sportec_spellF1": v.mean(), "sportec_delta": v.mean() - base.mean(),
                     "folds_improved": int((v > base).sum()), "public_spellF1": pub[n], "public_delta": pub[n] - pub["v1"]})
    table = pd.DataFrame(rows).set_index("repair")
    pooled = pd.concat(tables, axis=1)
    pooled["total"] = pooled.sum(axis=1)
    offs = np.concatenate(offsets)

    table.to_csv(args.out / "oracles_v1.csv", float_format="%.4f")
    pooled.to_csv(args.out / "errors_v1_sportec.csv")
    dec_pub["table"].to_csv(args.out / "errors_v1_public.csv", index=False)
    summary = {"team_change_offset_frames_p10_p25_p50_p75_p90": np.percentile(offs, [10, 25, 50, 75, 90]).tolist(),
               "n_matched_changes": int(len(offs))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.float_format", "{:.3f}".format):
        print(table)
        print(pooled["total"].to_string())
        print(json.dumps(summary))
        print(dec_pub["table"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
