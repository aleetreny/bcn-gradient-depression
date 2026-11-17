"""Diagnostic anatomy of remaining errors in v5_sym_linear and v5_sym_hybrid on Judge 1 and Judge 2."""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from exp_b_symmetry_eval import blend_proba
from src.data.sportec import MATCH_IDS, load_all
from src.decode.viterbi_hsmm import decode as decode_v4
from src.eval import errors, metrics


def main():
    sportec = load_all()
    sym_dir = ROOT / "experiments/outputs/exp_b_sym_oof"
    with open(sym_dir / "dec_v5_sym.pkl", "rb") as f:
        loo_dec_lin, loo_dec_hyb, dec_v5_lin, dec_v5_hyb = pickle.load(f)
    with open(sym_dir / "fold_decs.pkl", "rb") as f:
        fold_dec_std, _, _, _ = pickle.load(f)

    # Check if shifting p_change or log-odds difference forward/backward or max-pooling p_change helps!
    for shift_c in (-8, -5, -3, 0, 3, 5):
        scores = []
        for T in MATCH_IDS:
            sb = np.load(sym_dir / f"sym_{T}.npz")
            st = np.load(sym_dir / f"tcn_sym_{T}.npz")
            p = blend_proba(sb["symgbm_tta"], st["symtcn_tta_proba"], 0.55, "hybrid")
            c = 0.50 * sb["symtrans_tta"] + 0.50 * st["symtcn_tta_change"]
            c_sh = np.roll(c, shift_c)
            pred = decode_v4(p, sportec[T].half, fold_dec_std[T], p_change=c_sh)
            scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
        print(f"shift_c={shift_c:+2d}: LOMO={np.mean(scores):.4f}")

    # Check shifting H-vs-A log-odds slightly or adding a forward derivative (lead compensation) to H/A log-odds!
    for lead_frames in (0, 3, 5, 8, 10, 12):
        for alpha_lead in (0.10, 0.20, 0.30, 0.40):
            if lead_frames == 0 and alpha_lead > 0.10:
                continue
            scores = []
            for T in MATCH_IDS:
                sb = np.load(sym_dir / f"sym_{T}.npz")
                st = np.load(sym_dir / f"tcn_sym_{T}.npz")
                p = blend_proba(sb["symgbm_tta"], st["symtcn_tta_proba"], 0.55, "hybrid").copy()
                if lead_frames > 0:
                    # Lead-compensated H/A probability: mix p[t] with p[t + lead_frames] inside each half (only for H vs A ratio, keeping D fixed!)
                    for h in (1, 2):
                        idx = np.flatnonzero(sportec[T].half == h)
                        p_sub = p[idx]
                        p_fut = np.vstack([p_sub[lead_frames:], np.repeat(p_sub[-1:], lead_frames, axis=0)])
                        p_live = p_sub[:, :2].sum(axis=1, keepdims=True)
                        ha_mix = (1.0 - alpha_lead) * p_sub[:, :2] + alpha_lead * p_fut[:, :2]
                        ha_sum = np.clip(ha_mix.sum(axis=1, keepdims=True), 1e-6, None)
                        p[idx, :2] = p_live * (ha_mix / ha_sum)
                c = 0.50 * sb["symtrans_tta"] + 0.50 * st["symtcn_tta_change"]
                pred = decode_v4(p, sportec[T].half, fold_dec_std[T], p_change=c)
                scores.append(metrics.score(sportec[T].labels, pred, sportec[T].half)["spell_f1_mean"])
            print(f"lead={lead_frames:2d} alpha={alpha_lead:.2f}: LOMO={np.mean(scores):.4f}")


if __name__ == "__main__":
    main()
