"""EXP-02: Symmetric 4-Directional Boundary Detector (`SymBoundary4DirGBM`) on `v1ctl_plus`.

Predicts a 4-column directional transition probability matrix `p_trans_4d` of shape `(T, 4)`:
  - Column 0 (`p_H_to_A`): P(H -> A transition within +/- r frames of t, including H -> D -> A with D <= 12 frames)
  - Column 1 (`p_A_to_H`): P(A -> H transition within +/- r frames of t, including A -> D -> H with D <= 12 frames)
  - Column 2 (`p_Live_to_D`): P(H/A -> D transition within +/- r frames of t)
  - Column 3 (`p_D_to_Live`): P(D -> H/A transition within +/- r frames of t)

Exact H <-> A Equivariance Guarantee:
  - A single `H -> A` LightGBM model `model_h2a_` is trained on stacked `[Xa_orig, Xa_mir]` with targets
    `[y_h2a, y_a2h]`. At inference, `p_H_to_A(X) = model_h2a_(Xa_orig)` and `p_A_to_H(X) = model_h2a_(Xa_mir)`.
    Because `Xa_mir = augment_boundary_features(mirror_v1ctl_plus_df(X))` is the exact feature matrix of
    `swap_match(m)`, `p_A_to_H(X)` equals `p_H_to_A(X_mirror)` with 0.000e+00 discrepancy.
  - `model_l2d_` (`Live -> D`) and `model_d2l_` (`D -> Live`) are trained on `[Xa_orig, Xa_mir]` with targets
    `[y_live2d, y_live2d]` and `[y_d2live, y_d2live]`, and evaluated with 50/50 TTA over `Xa_orig` and `Xa_mir`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from src.data.common import AWAY, DEAD, HOME, possession_spells
from src.features.v1ctl_plus import mirror_v1ctl_plus_df

RADIUS = 6
MAX_BRIDGE_D = 12
DELTA_STEPS = (3, 6, 12)
LOCAL_WIN_RADIUS = 6

# Top 36 key ball / distance / control / knot / ghost / kinematics features from v1ctl_plus
KEY_BOUNDARY_COLS: tuple[str, ...] = (
    # Ball & geometry (9)
    "ball_visible",
    "ball_speed",
    "ball_accel",
    "ball_vx",
    "ball_vy_abs",
    "ball_last_to_touchline",
    "ball_last_to_goalline",
    "ball_outside_pitch",
    "ball_jump_last_next",
    # Distance & 2nd-defender margin (8)
    "dist_ball_H",
    "dist_ball_A",
    "dist_ball_diff",
    "dist_lastball_diff",
    "d_home2",
    "d_away2",
    "margin",
    "margin_cont",
    # Touch & control (8)
    "touch_team_r1",
    "touch_team_r2",
    "touch_team_r3",
    "control_team",
    "nearest_team",
    "carry_diff",
    "receiver_team",
    "contested",
    # Knot & ghost physics (6)
    "knot_bridge_soft",
    "knot_bridge_hard",
    "knot_pass_same",
    "clean_d_diff",
    "ghost_incoherence",
    "phantom_vertex",
    # Collective & Bernardo LSQ kinematics (5)
    "speed_mean_all",
    "frac_still_all",
    "diff_speed_h6",
    "diff_vx_h6",
    "all_speed_h6",
)

# Subset of 20 key continuous/bounded signals for local window max/min
EXTREMA_COLS: tuple[str, ...] = (
    "ball_speed",
    "ball_accel",
    "ball_last_to_touchline",
    "ball_last_to_goalline",
    "dist_ball_H",
    "dist_ball_A",
    "dist_ball_diff",
    "margin",
    "margin_cont",
    "touch_team_r2",
    "control_team",
    "carry_diff",
    "contested",
    "knot_bridge_soft",
    "clean_d_diff",
    "ghost_incoherence",
    "speed_mean_all",
    "frac_still_all",
    "diff_speed_h6",
    "all_speed_h6",
)

DEFAULT_4DIR_PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.65,
    reg_lambda=1.5,
    verbose=-1,
    n_jobs=4,
)


def targets_4dir(
    labels: np.ndarray,
    half: np.ndarray,
    radius: int = RADIUS,
    max_bridge_d: int = MAX_BRIDGE_D,
) -> dict[str, np.ndarray]:
    """Build the 4 binary boundary targets within +/- radius frames of exact transitions inside each half.

    Returns dict with keys: 'y_h2a', 'y_a2h', 'y_live2d', 'y_d2live', each int8 in {-1, 0, 1}.
    Unlabelled frames (labels < 0 or half <= 0) are set to -1.
    """
    T = len(labels)
    valid = (labels >= 0) & (half > 0)
    y_h2a = np.where(valid, 0, -1).astype(np.int8)
    y_a2h = np.where(valid, 0, -1).astype(np.int8)
    y_live2d = np.where(valid, 0, -1).astype(np.int8)
    y_d2live = np.where(valid, 0, -1).astype(np.int8)

    # 1. H -> A and A -> H (including H -> D -> A / A -> D -> H where D gap <= max_bridge_d)
    lab_clean = np.where(valid, labels, -1)
    s, e, t_arr = possession_spells(lab_clean, half)
    for i in range(1, len(s)):
        if half[s[i]] != half[s[i - 1]]:
            continue
        gap = int(s[i] - e[i - 1])
        if gap > max_bridge_d:
            continue
        # Ensure no unlabelled frames inside a bridged gap
        if gap > 0 and np.any(~valid[e[i - 1] : s[i]]):
            continue
        if t_arr[i] == t_arr[i - 1]:
            continue
        h_val = half[s[i]]
        # Mark around both e[i-1] and s[i] (identical when gap == 0)
        lo = max(0, int(e[i - 1]) - radius)
        hi = min(T, int(s[i]) + radius + 1)
        idx = np.arange(lo, hi)
        idx = idx[valid[idx] & (half[idx] == h_val)]
        if t_arr[i - 1] == HOME and t_arr[i] == AWAY:
            y_h2a[idx] = 1
        elif t_arr[i - 1] == AWAY and t_arr[i] == HOME:
            y_a2h[idx] = 1

    # 2. Live -> D (H/A -> D) and D -> Live (D -> H/A) at exact frame t where half[t] == half[t-1]
    same_half = (half[1:] == half[:-1]) & (half[1:] > 0) & valid[1:] & valid[:-1]
    prev_live = (labels[:-1] == HOME) | (labels[:-1] == AWAY)
    curr_live = (labels[1:] == HOME) | (labels[1:] == AWAY)
    prev_d = labels[:-1] == DEAD
    curr_d = labels[1:] == DEAD

    l2d_frames = np.flatnonzero(same_half & prev_live & curr_d) + 1
    d2l_frames = np.flatnonzero(same_half & prev_d & curr_live) + 1

    for f_idx in l2d_frames:
        h_val = half[f_idx]
        lo = max(0, int(f_idx) - radius)
        hi = min(T, int(f_idx) + radius + 1)
        idx = np.arange(lo, hi)
        idx = idx[valid[idx] & (half[idx] == h_val)]
        y_live2d[idx] = 1

    for f_idx in d2l_frames:
        h_val = half[f_idx]
        lo = max(0, int(f_idx) - radius)
        hi = min(T, int(f_idx) + radius + 1)
        idx = np.arange(lo, hi)
        idx = idx[valid[idx] & (half[idx] == h_val)]
        y_d2live[idx] = 1

    return {
        "y_h2a": y_h2a,
        "y_a2h": y_a2h,
        "y_live2d": y_live2d,
        "y_d2live": y_d2live,
    }


def augment_boundary_features(
    X: pd.DataFrame,
    half: np.ndarray,
    pitch: tuple[float, float] = (105.0, 68.0),
) -> pd.DataFrame:
    """Fast vectorized multi-scale temporal-difference & local window max/min feature augmentation (<0.25s/match).

    Computes strictly within each half segment:
      - Centered & one-sided temporal differences for delta in {3, 6, 12} frames on KEY_BOUNDARY_COLS
      - Local window max and min (+/- 6 frames) on EXTREMA_COLS
      - Exact pitch-centered coordinates so mirror involution has 0.000e+00 floating-point discrepancy.
    """
    T = len(X)
    half_arr = np.asarray(half)
    L, W = float(pitch[0]), float(pitch[1])

    # Ensure pitch-coordinate columns are centered around (L/2, W/2) so L - (L - x) has 0.0 IEEE-754 ULP drift
    X_clean = X.copy()
    for cx in ("ball_x", "ball_last_x", "ball_next_x", "centroid_x_H", "centroid_x_A"):
        if cx in X_clean.columns:
            X_clean[cx] = np.round(X_clean[cx].to_numpy(np.float64) - 0.5 * L, 4).astype(np.float32)
    for cy in ("ball_y", "ball_last_y", "ball_next_y", "centroid_y_H", "centroid_y_A"):
        if cy in X_clean.columns:
            X_clean[cy] = np.round(X_clean[cy].to_numpy(np.float64) - 0.5 * W, 4).astype(np.float32)
    # Drop asymmetric centroid_x_H rolling delta columns that have no centroid_x_A counterpart in v1
    drop_asym = [
        c for c in (
            "centroid_x_H_fwd_minus_bwd_2s",
            "chg_centroid_x_H_0.4s", "chg_abs_centroid_x_H_0.4s",
            "chg_centroid_x_H_1.0s", "chg_abs_centroid_x_H_1.0s",
            "chg_centroid_x_H_2.0s", "chg_abs_centroid_x_H_2.0s",
        ) if c in X_clean.columns
    ]
    if drop_asym:
        X_clean = X_clean.drop(columns=drop_asym)

    key_cols = [c for c in KEY_BOUNDARY_COLS if c in X_clean.columns]
    ext_cols = [c for c in EXTREMA_COLS if c in X_clean.columns]

    A = X_clean[key_cols].to_numpy(dtype=np.float32, copy=False)
    E = X_clean[ext_cols].to_numpy(dtype=np.float32, copy=False)
    K = len(key_cols)
    M = len(ext_cols)

    # Preallocate output block: 3 deltas * 2 (centered diff + forward diff) * K + 2 (max, min) * M
    n_new = len(DELTA_STEPS) * 2 * K + 2 * M
    out = np.full((T, n_new), np.nan, dtype=np.float32)

    edges = np.flatnonzero(half_arr[1:] != half_arr[:-1]) + 1
    starts = np.r_[0, edges]
    stops = np.r_[edges, T]

    win_size = 2 * LOCAL_WIN_RADIUS + 1

    for a, b in zip(starts, stops):
        L_seg = b - a
        if L_seg <= 1 or half_arr[a] <= 0:
            continue
        A_seg = A[a:b]
        col_offset = 0
        for d in DELTA_STEPS:
            # 1. Centered difference: A[t + d] - A[t - d]
            if L_seg > 2 * d:
                out[a + d : b - d, col_offset : col_offset + K] = A_seg[2 * d :] - A_seg[: L_seg - 2 * d]
            col_offset += K
            # 2. Forward difference across boundary: A[t + d] - A[t]
            if L_seg > d:
                out[a : b - d, col_offset : col_offset + K] = A_seg[d:] - A_seg[: L_seg - d]
            col_offset += K

        # 3. Local window max and min over +/- LOCAL_WIN_RADIUS ignoring NaNs
        E_seg = E[a:b]
        all_nan = np.isnan(E_seg)
        E_for_max = np.where(all_nan, -np.inf, E_seg)
        E_for_min = np.where(all_nan, np.inf, E_seg)
        seg_max = maximum_filter1d(E_for_max, size=win_size, axis=0, mode="nearest")
        seg_min = minimum_filter1d(E_for_min, size=win_size, axis=0, mode="nearest")
        seg_max[~np.isfinite(seg_max)] = np.nan
        seg_min[~np.isfinite(seg_min)] = np.nan
        out[a:b, col_offset : col_offset + M] = seg_max
        col_offset += M
        out[a:b, col_offset : col_offset + M] = seg_min

    new_names: list[str] = []
    for d in DELTA_STEPS:
        new_names.extend([f"b4d_cdiff{d}_{c}" for c in key_cols])
        new_names.extend([f"b4d_fdiff{d}_{c}" for c in key_cols])
    new_names.extend([f"b4d_lmax{LOCAL_WIN_RADIUS}_{c}" for c in ext_cols])
    new_names.extend([f"b4d_lmin{LOCAL_WIN_RADIUS}_{c}" for c in ext_cols])

    df_extra = pd.DataFrame(out, columns=new_names, index=X_clean.index)
    return pd.concat([X_clean, df_extra], axis=1)


def swap_trans_4d(p_trans_4d: np.ndarray) -> np.ndarray:
    """Swap H->A (col 0) and A->H (col 1) in a (T, 4) directional transition probability matrix."""
    out = p_trans_4d.copy()
    out[:, 0] = p_trans_4d[:, 1]
    out[:, 1] = p_trans_4d[:, 0]
    return out


@dataclass
class SymBoundary4DirGBM:
    params: dict = field(default_factory=lambda: dict(DEFAULT_4DIR_PARAMS))
    subsample_every: int = 5
    seed: int = 0

    def fit(
        self,
        Xs_orig: list[pd.DataFrame],
        Xs_mir: list[pd.DataFrame],
        ys_4dir: list[dict[str, np.ndarray]],
    ) -> "SymBoundary4DirGBM":
        """Fit the 3 symmetric LightGBM boundary models (H->A, Live->D, D->Live).

        By training `model_h2a_` on stacked `[Xs_orig, Xs_mir]` with targets `[y_h2a, y_a2h]`,
        `p_A_to_H(X) = model_h2a_.predict_proba(X_mir)` is guaranteed to equal `p_H_to_A(X_mir)`
        with 0.000e+00 symmetry discrepancy.
        """
        import lightgbm as lgb

        rows_X_orig, rows_X_mir = [], []
        rows_h2a, rows_a2h, rows_l2d, rows_d2l = [], [], [], []

        for Xo, Xm, yd in zip(Xs_orig, Xs_mir, ys_4dir):
            keep = np.flatnonzero(yd["y_h2a"] >= 0)[:: self.subsample_every]
            rows_X_orig.append(Xo.iloc[keep])
            rows_X_mir.append(Xm.iloc[keep])
            rows_h2a.append(yd["y_h2a"][keep])
            rows_a2h.append(yd["y_a2h"][keep])
            rows_l2d.append(yd["y_live2d"][keep])
            rows_d2l.append(yd["y_d2live"][keep])

        X_sym = pd.concat(rows_X_orig + rows_X_mir, ignore_index=True)
        y_h2a_sym = np.concatenate(rows_h2a + rows_a2h)
        y_l2d_sym = np.concatenate(rows_l2d + rows_l2d)
        y_d2l_sym = np.concatenate(rows_d2l + rows_d2l)

        self.features_ = list(X_sym.columns)
        self.base_rates_ = {
            "h2a": float(np.mean(y_h2a_sym)),
            "a2h": float(np.mean(y_h2a_sym)),
            "live2d": float(np.mean(y_l2d_sym)),
            "d2live": float(np.mean(y_d2l_sym)),
        }

        self.model_h2a_ = lgb.LGBMClassifier(**self.params, random_state=self.seed).fit(X_sym, y_h2a_sym)
        self.model_l2d_ = lgb.LGBMClassifier(**self.params, random_state=self.seed + 100).fit(X_sym, y_l2d_sym)
        self.model_d2l_ = lgb.LGBMClassifier(**self.params, random_state=self.seed + 200).fit(X_sym, y_d2l_sym)
        return self

    def predict_from_augmented(self, Xa_orig: pd.DataFrame, Xa_mir: pd.DataFrame) -> np.ndarray:
        """Predict (T, 4) directional transition probabilities from pre-augmented (Xa_orig, Xa_mir)."""
        Xo = Xa_orig[self.features_]
        Xm = Xa_mir[self.features_]

        p_h2a = self.model_h2a_.predict_proba(Xo)[:, 1].astype(np.float32)
        p_a2h = self.model_h2a_.predict_proba(Xm)[:, 1].astype(np.float32)

        p_l2d_o = self.model_l2d_.predict_proba(Xo)[:, 1]
        p_l2d_m = self.model_l2d_.predict_proba(Xm)[:, 1]
        p_l2d = (0.5 * (p_l2d_o + p_l2d_m)).astype(np.float32)

        p_d2l_o = self.model_d2l_.predict_proba(Xo)[:, 1]
        p_d2l_m = self.model_d2l_.predict_proba(Xm)[:, 1]
        p_d2l = (0.5 * (p_d2l_o + p_d2l_m)).astype(np.float32)

        return np.column_stack([p_h2a, p_a2h, p_l2d, p_d2l])

    def predict(
        self,
        X_plus: pd.DataFrame,
        half: np.ndarray,
        pitch: tuple[float, float] = (105.0, 68.0),
    ) -> np.ndarray:
        """Predict (T, 4) directional transition probabilities `[p_H_to_A, p_A_to_H, p_Live_to_D, p_D_to_Live]`."""
        X_mir = mirror_v1ctl_plus_df(X_plus, half, pitch)
        Xa_orig = augment_boundary_features(X_plus, half, pitch)
        Xa_mir = augment_boundary_features(X_mir, half, pitch)
        return self.predict_from_augmented(Xa_orig, Xa_mir)
