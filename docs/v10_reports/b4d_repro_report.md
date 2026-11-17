# Rule 3 Reproducibility Report: `SymBoundary4DirGBM` (`EXP-02`) & Complete `v8.5` Pipeline

## 1. Exact Specification Extracted from `symb4d_full.pkl` & `b4d_{T}.npz`

- **Model Class**: `src.models.boundary_4dir.SymBoundary4DirGBM`
- **Hyperparameters (`params`)**: `DEFAULT_4DIR_PARAMS`
  - `n_estimators=400`, `learning_rate=0.05`, `num_leaves=63`, `min_child_samples=100`, `subsample=0.8`, `subsample_freq=1`, `colsample_bytree=0.65`, `reg_lambda=1.5`, `verbose=-1`, `n_jobs=4`
- **Subsampling**:
  - `subsample_every=1` inside `SymBoundary4DirGBM.fit(...)` because each training variant was pre-subsampled at step `[::5]` (`keep = np.flatnonzero(yd["y_h2a"] >= 0)[::5]`) **after** calling `augment_boundary_features(X, half, pitch)`.
  - Total rows in full 7-match symmetric training matrix `X_sym`: `1,203,192` (`7 matches × 3 variants × 2 orientations × ~28,647 frames`).
- **Random Seeds**:
  - **7 Outer LOMO Folds (`b4d_{T}.npz`)**: `seed = k` (`0..6`) for `k, T in enumerate(MATCH_IDS)`.
  - **Full 7-Match Model (`symb4d_full.pkl`)**: `seed = 42` (`random_state=42` for `model_h2a_`, `142` for `model_l2d_`, `242` for `model_d2l_`).
- **Training Variants**:
  - Trained on the 3 `TRAIN_VARIANTS` of `degrade_v3` from `scripts/s3_7_nested_full.py`: `("s1", None, 1)`, `("s2", None, 2)`, `("static05_s1", {"static_hide_prob": 0.5}, 1)`.
  - Confirmed via exact match of `base_rates_` (`h2a=0.015236969660702532`, `live2d=0.009120738834699698`, `d2live=0.009125725569983843`) and LightGBM bin boundaries (`feature_infos`, e.g. `ball_last_to_touchline min=-5.7877516746521` unique to `degrade_v3_static05_s1`).
- **601-Feature Construction**:
  - Starts from the 352-column `v1ctl_plus` table: `change_features(features_v1, half)` (`160 + 60 = 220` cols) + `control_v1` (`24` cols) + `plus_v1` (`108` cols).
  - `augment_boundary_features(X, half, pitch)` centers 10 pitch coordinate columns around `(L/2, W/2)` rounded to `1e-4`, drops 7 asymmetric `centroid_x_H` rolling columns (`352 - 7 = 345` cols), and appends `256` multi-scale features (`3` delta steps `{3, 6, 12}` × `2` `[cdiff, fdiff]` × `36` `KEY_BOUNDARY_COLS` = `216`, plus `2` `[lmax6, lmin6]` × `20` `EXTREMA_COLS` = `40`), yielding `345 + 256 = 601` features.

## 2. Verification Results (`experiments/outputs/v9_swarm/b4d_repro/` vs `experiments/outputs/exp_w3_oof/`)

| Fold / Artifact | `max |p4d_v3 - ref|` | `mean |p4d_v3 - ref|` | `max |p4d_v6 - ref|` | `mean |p4d_v6 - ref|` |
|---|---:|---:|---:|---:|
| `J03WMX` (`seed=0`) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `J03WN1` (`seed=1`) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `J03WOH` (`seed=2`) | `1.711e-04` | `1.565e-09` | `2.179e-04` | `1.801e-09` |
| `J03WOY` (`seed=3`) | `7.375e-08` | `2.298e-13` | `8.162e-08` | `2.660e-13` |
| `J03WPY` (`seed=4`) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `J03WQQ` (`seed=5`) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |
| `J03WR9` (`seed=6`) | `0.000e+00` | `0.000e+00` | `0.000e+00` | `0.000e+00` |

- **PR-AUC Match**:
  - `degrade_v3` 4-channel PR-AUC: `H->A = 0.6012`, `A->H = 0.5813`, `Live->D = 0.8823`, `D->Live = 0.9212` (`1D Turnover PR-AUC = 0.6167`, matching reference to 8 decimal places).
  - `degrade_v6` 4-channel PR-AUC: `H->A = 0.4362`, `A->H = 0.4393`, `Live->D = 0.7973`, `D->Live = 0.8848` (matching reference to 8 decimal places).
- **Downstream `v8.5` Unified Synthesis Verification (Candidate Extraction Recomputed from Scratch)**:
  - `J1A (v3) = 0.758767`, `J1B-Std = 0.745135`, `J1B-Gen = 0.704279`, `J2-Raw = 0.838925`, `J2-noD = 0.875581` (100% identical to official `v8.5`).
  - `submission_v85_forensic_strict_cv.csv`: SHA-256 `2c834b92408ead731ff18be2713b94b4710df082bb9712796ca7668c2effc329` (**100% byte-for-byte identical**).
  - `submission_v85_forensic_grand_champion.csv`: SHA-256 `f923326ac28f3ac4590aa1de98e8a1211b3db281d6bf79ec5251b9deb6595309` (**100% byte-for-byte identical**).
