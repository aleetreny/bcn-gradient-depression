"""Night line D3: rebuild the two BAR-ATH prediction caches of the v8.5 chain that no script in the repo writes.

    (run through scripts/noche/v85_shortcut.py, inside the isolated copy, after exp_c_plus_eval.py)

experiments/outputs/exp_g_oof/v6_j23_preds.pkl and exp_w3_oof/b4d_j23_preds.pkl were written by code that lived in
another workspace and is gone. Both are predictions of saved 7-match models (SymGBM_v6 + SymTrans_v6 from
exp_g_ghost_sym.py; SymBoundary4DirGBM from exp_w3_decoder_4dir_xdx.py) on Metrica Sample and BAR-ATH, so they
can be rebuilt from those models with the same feature tables and calls as the scripts that read them. In the
control copy they must equal the original files value for value (checked by the caller).
"""
from __future__ import annotations

import glob
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from exp_b_symmetry_eval import predict_gbm_tta, predict_trans_tta  # noqa: E402
from exp_e2_broadcast_repair import enhanced_broadcast_alignment, repair_atd_match  # noqa: E402
from s3_7_nested_full import build_features  # noqa: E402
from s4_6_broadcast_align import load_broadcast  # noqa: E402
from src.data.metrica_atd import load as load_atd  # noqa: E402
from src.data.metrica_sample import GAMES, load as load_sample  # noqa: E402
from src.features.v1 import match_fingerprint  # noqa: E402
from src.models.transition import change_features  # noqa: E402

OUT = ROOT / "experiments" / "outputs"


def load_cached_plus(prefix, half):
    """features_v1 (with change features) + control_v1 + plus_v1, as exp_w3_decoder_4dir_xdx.load_cached_plus."""
    f = [sorted(glob.glob(str(ROOT / f"local_data/cache/{d}/{prefix}*.parquet")))[0]
         for d in ("features_v1", "control_v1", "plus_v1")]
    return pd.concat([change_features(pd.read_parquet(f[0]), half), pd.read_parquet(f[1]), pd.read_parquet(f[2])],
                     axis=1)


def main():
    atd = load_atd()
    Pb, Bb, team_b = load_broadcast()
    hom = pd.read_csv(ROOT / "data/broadcast/BAR-ATH_broadcast_homography.csv")
    mapping, _ = enhanced_broadcast_alignment(atd, Pb, hom["valid"].to_numpy().astype(bool))
    atd_rep = repair_atd_match(atd, Pb, Bb, team_b, mapping, True, True, None)
    keys = {"atd_raw": ("atd", "atd_" + match_fingerprint(atd)),
            "atd_rep": ("atd_rep_True_True_None", "atd_rep_True_True_None_" + match_fingerprint(atd_rep))}
    samples = {gid: load_sample(gid) for gid in GAMES}

    gbm = pickle.loads((OUT / "exp_g_oof/symgbm_v6_full.pkl").read_bytes())
    trans = pickle.loads((OUT / "exp_g_oof/symtrans_v6_full.pkl").read_bytes())

    def v6(m, key):
        X = build_features(m, key, "v1ctl")
        return {"g_v6": predict_gbm_tta(gbm, X, m.half, m.pitch), "tr_v6": predict_trans_tta(trans, X, m.half, m.pitch)}

    sample_v6 = {gid: v6(ms, f"sample_{gid}_raw") for gid, ms in samples.items()}
    v6_atd = {"atd_rep": v6(atd_rep, keys["atd_rep"][0])}

    # the samples with the model's default pitch, BAR-ATH with its own (105.3 x 68): both match the originals exactly
    b4d = pickle.loads((OUT / "exp_w3_oof/symb4d_full.pkl").read_bytes())
    b4d_j23 = {"sample": {gid: b4d.predict(load_cached_plus(f"sample_{gid}_raw", ms.half), ms.half)
                          for gid, ms in samples.items()},
               "atd_raw": b4d.predict(load_cached_plus(keys["atd_raw"][1], atd.half), atd.half, atd.pitch),
               "atd_rep": b4d.predict(load_cached_plus(keys["atd_rep"][1], atd.half), atd.half, atd.pitch)}

    with open(OUT / "exp_g_oof/v6_j23_preds.pkl", "wb") as f:
        pickle.dump((sample_v6, v6_atd["atd_rep"]), f)            # the original stores the repaired stream
    with open(OUT / "exp_w3_oof/b4d_j23_preds.pkl", "wb") as f:
        pickle.dump(b4d_j23, f)
    print("wrote v6_j23_preds.pkl and b4d_j23_preds.pkl", flush=True)


if __name__ == "__main__":
    main()
