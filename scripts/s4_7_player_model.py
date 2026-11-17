"""F4.7 screen: player-level owner model (src/features/players.py), alone and fused with the GBM.

    .venv311/bin/python scripts/s4_7_player_model.py [--config configs/degrade_v3.json] [--out experiments/outputs/045]

Outer LOMO on Sportec (6 matches -> T; training rows from the s1 degradation every 5th frame, the test
match degraded with seed 0 on every frame). Reported per fold:
  - team accuracy on alive frames of the player model alone, overall and by ball state (real / straight
    line / missing), against the stored v1ctl GBM outer prediction;
  - spellF1 of the GBM and of the fused model, decoded with the fold's v1ctl GBM decoder parameters.
    Fusion: P(home | alive) = sigmoid(logit GBM + w * logit player model), alive from the GBM; w is chosen
    on the six training matches' cross-fitted player predictions (no test labels).
And on Metrica Sample with the full models (decoder_v1ctl_gbm.json).
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from s2_1_degradation import load_config  # noqa: E402
from s3_7_nested_full import build_features  # noqa: E402
from s3_11_tcn_local import without_dead_margin  # noqa: E402
from src.data.ball_interp import linear_mask  # noqa: E402
from src.data.common import HOME  # noqa: E402
from src.data.degrade import degrade  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.data.sportec import MATCH_IDS, load_all  # noqa: E402
from src.decode.viterbi_hsmm import DecoderParams, decode  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.features.players import PlayerOwnerModel, owner_labels, slot_features  # noqa: E402

W_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5)


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def fuse(p_gbm, ph_player, w):
    pa = 1 - p_gbm[:, 2]
    ph = p_gbm[:, 0] / np.maximum(pa, 1e-9)
    z = logit(ph) + w * np.where(np.isfinite(ph_player), logit(np.nan_to_num(ph_player, nan=0.5)), 0.0)
    h = 1 / (1 + np.exp(-z))
    return np.stack([pa * h, pa * (1 - h), 1 - pa], 1).astype(np.float32)


def ball_state(m):
    vis = m.ball_xy_visible
    return np.where(~vis, "missing", np.where(linear_mask(m.B, m.half), "line", "real"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/degrade_v3.json")
    parser.add_argument("--out", type=Path, default=ROOT / "experiments/outputs/045")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    sportec = load_all()
    ref = ROOT / "local_data" / "oof" / "nested_degrade_v3_v1ctl"
    train_F, train_y, test = {}, {}, {}
    for mid, m in sportec.items():
        d = degrade(m, replace(cfg), 1)
        F = slot_features(d, every=5)
        train_F[mid], train_y[mid] = F, owner_labels(m, d, F)
        d0 = degrade(m, cfg, 0)
        test[mid] = (d0, slot_features(d0, every=1))
        print("features", mid, flush=True)

    rows = []
    for k, T in enumerate(MATCH_IDS):
        others = [x for x in MATCH_IDS if x != T]
        model = PlayerOwnerModel(seed=k).fit([train_F[x] for x in others], [train_y[x] for x in others])
        m = sportec[T]
        d0, F0 = test[T]
        ph, _ = model.team_proba(F0, m.n_frames)
        g = np.load(ref / T / "gbm_outer_{}.npy".format(T))
        params = DecoderParams(**json.loads((ref / T / "params_gbm.json").read_text()))
        # fusion weight from the training matches only: one inner model per left-out training match (s1 rows)
        best_w, best_s = 0.0, -1
        inner = {}
        for J in others[:3]:                                   # three inner folds keep the screen affordable
            mi = PlayerOwnerModel(seed=100 + k).fit([train_F[x] for x in others if x != J], [train_y[x] for x in others if x != J])
            dJ, FJ = test[J]
            inner[J] = mi.team_proba(FJ, sportec[J].n_frames)[0]
        for w in W_GRID:
            s = np.mean([metrics.score(sportec[J].labels, decode(fuse(np.load(ref / T / "gbm_inner_{}.npy".format(J)), inner[J], w),
                                                                     sportec[J].half, params), sportec[J].half)["spell_f1_mean"] for J in inner])
            if s > best_s:
                best_w, best_s = w, s
        y = m.labels
        alive = np.isin(y, (0, 1))
        st = ball_state(d0)
        pa = g[:, 0] / np.maximum(g[:, 0] + g[:, 1], 1e-9)
        row = {"fold": T, "w": best_w,
               "gbm": metrics.score(y, decode(g, m.half, params), m.half)["spell_f1_mean"],
               "fused": metrics.score(y, decode(fuse(g, ph, best_w), m.half, params), m.half)["spell_f1_mean"]}
        for name, sel in (("all", alive), ("real", alive & (st == "real")), ("line", alive & (st == "line")), ("missing", alive & (st == "missing"))):
            okp = sel & np.isfinite(ph)
            row["acc_player_" + name] = float(np.mean((ph[okp] > 0.5) == (y[okp] == HOME)))
            row["acc_gbm_" + name] = float(np.mean((pa[sel] > 0.5) == (y[sel] == HOME)))
        rows.append(row)
        print({kk: (round(v, 4) if isinstance(v, float) else v) for kk, v in row.items()}, flush=True)
    folds = pd.DataFrame(rows).set_index("fold")
    folds.to_csv(args.out / "folds.csv", float_format="%.4f")

    # Metrica Sample: full player model + full v1ctl GBM, median fusion weight of the folds
    full = PlayerOwnerModel(seed=99).fit(list(train_F.values()), list(train_y.values()))
    gbm = pickle.loads((ROOT / "local_data/models/full_degrade_v3_v1ctl/models.pkl").read_bytes())["gbm"]
    dec = DecoderParams(**{kk: v for kk, v in json.loads((ROOT / "configs/decoder_v1ctl_gbm.json").read_text()).items() if not kk.startswith("_")})
    w = float(folds["w"].median())
    srows = []
    for gm in GAMES:
        ms = load_sample(gm)
        p = gbm.predict_proba(build_features(ms, "sample_{}".format(gm), "v1ctl"))
        ph, _ = full.team_proba(slot_features(ms, every=1), ms.n_frames)
        r = {"game": gm}
        for name, q in (("gbm", p), ("fused", fuse(p, ph, w))):
            pred = decode(q, ms.half, dec)
            r[name] = metrics.score(ms.labels, pred, ms.half)["spell_f1_mean"]
            r[name + "_noD"] = metrics.score(without_dead_margin(ms.labels, ms.half), pred, ms.half)["spell_f1_mean"]
        alive = np.isin(ms.labels, (0, 1)) & np.isfinite(ph)
        r["acc_player"] = float(np.mean((ph[alive] > 0.5) == (ms.labels[alive] == HOME)))
        srows.append(r)
        print(r, flush=True)
    samp = pd.DataFrame(srows).set_index("game")
    samp.to_csv(args.out / "sample.csv", float_format="%.4f")
    summary = {"lomo": folds[["gbm", "fused"]].mean().round(4).to_dict(), "folds_fused_better": int((folds.fused > folds.gbm).sum()),
               "team_acc": folds.filter(like="acc_").mean().round(4).to_dict(), "sample": samp.mean().round(4).to_dict(), "w_median": w}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
