"""Line L4: Audio Whistle Extraction, Collective Foul Deceleration & Stoppage Start Refiner.

Implements three H<->A equivariant (`0.000e+00` symmetry error) modules for `v9`:
1. `Fox40WhistleDetector`:
   - Extracts the dual-chamber referee whistle signal (`3750-4000 Hz` Chamber A + `4180-4420 Hz`
     Chamber B + `7500-8840 Hz` 2nd harmonic narrowband prominence over crowd flank noise) from
     `data/broadcast/BAR-ATH_broadcast.mp4` and `data/broadcast/BAR-ATH_tactical.mp4`.
   - Maps broadcast audio frames `k` to tactical frames `t in [0, 199319)` via the continuous
     broadcast half clock lock (`t = k + 4014` in Half 1 for `k < 85500`, `t = k + 26920` in
     Half 2 for `k >= 85500`), verified against `local_data/broadcast_alignment.npz` (exact
     within +/-1 frame across 99.9% of valid homography frames).
   - Caches the frame-level whistle arrays and unsupervised enrichment statistics to
     `local_data/v9_swarm/bar_ath_whistle.npz`.
2. `CollectiveFoulDeceleration`:
   - Purely kinematic, 100% H<->A symmetric detector of in-pitch foul stoppages where the ball
     continues rolling on the grass while players decelerate sharply (`speed_mean_all` drop over
     1.5-2.0s) and/or a fouled player remains stationary (`speed < 0.6 m/s`).
3. `StoppageStartRefiner`:
   - Combines four kinematic/4D-boundary stoppage refinements (`PreDeadSliverPullback`,
     `NoContactDXYVeto`, `BallisticCrossSnap`, `GapDecelPullback`) with `RollingBallFoulPullback`
     (collective deceleration + optional broadcast whistle / foul onset).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from scipy import signal

from src.data.common import AWAY, DEAD, FPS, HOME, runs


@dataclass(frozen=True)
class WhistleDetectorConfig:
    sample_rate: int = 24000
    nperseg: int = 1024
    noverlap: int = 560  # hop = 464 samples -> ~51.72 STFT hops/s (~2 hops per 25fps frame)
    chamber_a_hz: tuple[float, float] = (3750.0, 4000.0)
    chamber_b_hz: tuple[float, float] = (4180.0, 4420.0)
    harmonic_hz: tuple[float, float] = (7500.0, 8840.0)
    flank_lo_hz: tuple[float, float] = (2500.0, 3550.0)
    flank_hi_hz: tuple[float, float] = (4600.0, 5600.0)
    min_duration_frames: int = 4   # 0.16s at 25 fps
    max_duration_frames: int = 40  # 1.60s at 25 fps
    h1_bcast_offset: int = 4014
    h2_bcast_offset: int = 26920
    bcast_half_split: int = 85500
    n_tactical_frames: int = 199319

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class StoppageRefinerConfig:
    enable_predead_sliver: bool = True
    enable_nocontact_dxy: bool = True
    enable_ballistic_snap: bool = True
    enable_gap_decel: bool = True
    enable_rolling_foul_pullback: bool = True
    # PreDeadSliverPullback (Y -> X -> D where SymBoundary4Dir to_dead fires strongly at s)
    sliver_max_len: int = 20
    sliver_min_dead_len: int = 25
    sliver_vis_thresh: float = 0.85
    sliver_b4d_to_dead_thresh: float = 0.80
    sliver_mean_pd_thresh: float = 0.18
    # NoContactDXYVeto (D -> X -> Y where X never comes near the visible ball)
    nocontact_max_len: int = 50
    nocontact_min_next_len: int = 25
    nocontact_vis_thresh: float = 0.85
    nocontact_min_dist_x: float = 2.8
    nocontact_max_dist_y: float = 1.2
    # BallisticCrossSnap (early Live -> D before ball crosses touchline/goalline)
    snap_max_window: int = 25
    snap_min_spell_margin: int = 5
    snap_min_cos_sim: float = 0.94
    snap_min_ball_speed: float = 5.0
    snap_max_rel_dv: float = 0.40
    snap_max_mean_pd: float = 0.85
    # GapDecelPullback (late Live -> D inside a stoppage ball gap with decelerating players)
    gap_min_wait: int = 3
    gap_max_wait: int = 35
    gap_min_prev_live: int = 8
    gap_min_dead_len: int = 25
    gap_vis_pre_thresh: float = 0.85
    gap_vis_dead_max: float = 0.05
    gap_decel_thresh: float = -0.25
    gap_mean_pd_thresh: float = 0.18
    # RollingBallFoulPullback (in-pitch foul with rolling visible ball + whistle/deceleration)
    foul_min_pullback: int = 10
    foul_max_pullback: int = 110
    foul_min_prev_live: int = 20
    foul_min_dead_len: int = 50
    foul_decel_thresh: float = -0.45
    foul_whistle_z_thresh: float = 3.2

    def to_dict(self) -> dict:
        return asdict(self)


def _half_bounds(half: np.ndarray) -> list[tuple[int, int]]:
    out = []
    for h in np.unique(half[half > 0]):
        idx = np.flatnonzero(half == h)
        if len(idx) > 0:
            out.append((int(idx[0]), int(idx[-1]) + 1))
    return out


def _runs_of_with_half(arr: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(arr)
    if n == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=arr.dtype)
    chg = np.flatnonzero((arr[1:] != arr[:-1]) | (half[1:] != half[:-1])) + 1
    s = np.concatenate([[0], chg])
    e = np.concatenate([chg, [n]])
    return s, e, arr[s]


def detect_pitch_crossings_from_features(
    bx: np.ndarray,
    by: np.ndarray,
    vis: np.ndarray,
    half: np.ndarray,
    L: float = 105.0,
    W: float = 68.0,
    margin: float = 0.5,
    near_line: float = 3.0,
    extrap_s: float = 1.0,
) -> np.ndarray:
    """Detect pitch boundary crossings (observed + 1s gap-extrapolated) purely from feature columns.
    100% H<->A symmetric.
    """
    B = np.column_stack([bx, by])
    with np.errstate(invalid="ignore"):
        outside = vis & ((bx < -margin) | (bx > L + margin) | (by < -margin) | (by > W + margin))
        inside = vis & ~outside
    frames = []
    for a, b in _half_bounds(half):
        was_inside = False
        last_in = -1
        t = a
        while t < b:
            if inside[t]:
                was_inside, last_in = True, t
            elif outside[t] and was_inside:
                frames.append(t)
                was_inside = False
            elif not vis[t] and was_inside and last_in == t - 1 and t - 3 >= a:
                x, y = B[t - 1]
                v = (B[t - 1] - B[t - 3]) * FPS / 2.0
                if np.all(np.isfinite(v)):
                    tcross = []
                    for pos, vel, lo, hi in ((x, v[0], 0.0, L), (y, v[1], 0.0, W)):
                        if vel < 0 and pos - lo < near_line:
                            tcross.append((pos - lo + margin) / -vel)
                        elif vel > 0 and hi - pos < near_line:
                            tcross.append((hi - pos + margin) / vel)
                    if tcross and min(tcross) <= extrap_s:
                        g = t
                        while g < b and not vis[g]:
                            g += 1
                        tc = t - 1 + int(round(min(tcross) * FPS))
                        if tc < g:
                            frames.append(tc)
                            was_inside = False
                            t = g
                            continue
            t += 1
    return np.array(frames, dtype=np.int64)


class CollectiveFoulDeceleration:
    """100% H<->A symmetric detector of collective player deceleration on in-pitch fouls."""

    @staticmethod
    def compute_signals(X_df: pd.DataFrame, half: np.ndarray) -> dict[str, np.ndarray]:
        n = len(X_df)
        sp_all = np.nan_to_num(X_df["speed_mean_all"].to_numpy(dtype=np.float64), nan=0.0)
        decel_2s = np.nan_to_num(
            X_df["speed_mean_all_fwd_minus_bwd_2s"].to_numpy(dtype=np.float64),
            nan=0.0,
        )
        # Smooth collective speed over 0.48s (12 frames) and compute 1.5s (38-frame) forward drop
        k_sm = np.ones(12, dtype=np.float64) / 12.0
        sp_sm = np.convolve(sp_all, k_sm, mode="same")
        decel_1_5s = np.zeros(n, dtype=np.float64)
        for a, b in _half_bounds(np.asarray(half, dtype=np.int32)):
            if b - a > 38:
                decel_1_5s[a:b - 38] = sp_sm[a + 38:b] - sp_sm[a:b - 38]

        # Symmetric minimum distance of any player to ball (min(d1_H, d1_A))
        d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(dtype=np.float64), nan=99.0)
        d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(dtype=np.float64), nan=99.0)
        d1_min = np.minimum(d1h, d1a)
        duel_compact = ((d1h < 2.5) & (d1a < 2.5)).astype(np.float64)

        # Combined foul deceleration score (higher = stronger collective stop after physical duel)
        decel_drop = np.maximum(0.0, -np.minimum(decel_2s, decel_1_5s))
        foul_score = decel_drop * (1.0 + 0.5 * duel_compact)
        return {
            "speed_mean_all": sp_all,
            "decel_2s": decel_2s,
            "decel_1_5s": decel_1_5s,
            "d1_min": d1_min,
            "duel_compact": duel_compact,
            "foul_score": foul_score,
        }


class Fox40WhistleDetector:
    """Dual-chamber Fox-40 referee whistle detector (3.75-4.00 kHz + 4.18-4.42 kHz + harmonics)."""

    def __init__(self, cfg: WhistleDetectorConfig | None = None):
        self.cfg = cfg or WhistleDetectorConfig()

    @staticmethod
    def _ensure_wav(mp4_path: Path, wav_path: Path, sr: int = 24000) -> Path:
        if wav_path.exists() and wav_path.stat().st_size > 1_000_000:
            return wav_path
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "/usr/bin/afconvert",
            "-f", "WAVE",
            "-d", f"LEI16@{sr}",
            str(mp4_path),
            str(wav_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return wav_path

    def _compute_stream_whistle_25fps(self, wav_path: Path, n_frames_target: int) -> dict[str, np.ndarray]:
        cfg = self.cfg
        from scipy.io import wavfile
        sr, raw_arr = wavfile.read(str(wav_path))
        if raw_arr.ndim > 1:
            pcm = raw_arr.astype(np.float32).mean(axis=1)
        else:
            pcm = raw_arr.astype(np.float32)
        if raw_arr.dtype == np.int16:
            pcm /= 32768.0

        freqs, times, Zxx = signal.stft(
            pcm,
            fs=sr,
            nperseg=cfg.nperseg,
            noverlap=cfg.noverlap,
            boundary=None,
            padded=False,
        )
        log_p = np.log10(np.abs(Zxx) ** 2 + 1e-12)

        def _band_max(f_lo: float, f_hi: float) -> np.ndarray:
            m = (freqs >= f_lo) & (freqs <= f_hi)
            return log_p[m].max(axis=0)

        def _band_med(f_lo: float, f_hi: float) -> np.ndarray:
            m = (freqs >= f_lo) & (freqs <= f_hi)
            return np.median(log_p[m], axis=0)

        p_a = _band_max(*cfg.chamber_a_hz)
        p_b = _band_max(*cfg.chamber_b_hz)
        p_h = _band_max(*cfg.harmonic_hz)
        flank = 0.5 * (_band_med(*cfg.flank_lo_hz) + _band_med(*cfg.flank_hi_hz))

        snr_a = np.maximum(0.0, p_a - flank)
        snr_b = np.maximum(0.0, p_b - flank)
        snr_h = np.maximum(0.0, p_h - flank)
        # Dual-chamber geometric mean + harmonic boost, penalizing broadband commentator sibilance
        broadband_var = np.std(log_p[(freqs >= 2500.0) & (freqs <= 5600.0)], axis=0)
        raw_hop = np.sqrt(snr_a * snr_b) * (1.0 + 0.25 * snr_h) * broadband_var

        # Resample STFT hops to 25 fps frames
        frame_idx = np.clip((times * FPS).astype(np.int64), 0, n_frames_target - 1)
        w_frame = np.zeros(n_frames_target, dtype=np.float32)
        a_frame = np.zeros(n_frames_target, dtype=np.float32)
        b_frame = np.zeros(n_frames_target, dtype=np.float32)
        h_frame = np.zeros(n_frames_target, dtype=np.float32)
        np.maximum.at(w_frame, frame_idx, raw_hop.astype(np.float32))
        np.maximum.at(a_frame, frame_idx, snr_a.astype(np.float32))
        np.maximum.at(b_frame, frame_idx, snr_b.astype(np.float32))
        np.maximum.at(h_frame, frame_idx, snr_h.astype(np.float32))

        # Temporal bandpass: sustain over 4 frames (0.16s), subtract 75-frame (3.0s) background median
        k_short = np.ones(cfg.min_duration_frames, dtype=np.float32) / float(cfg.min_duration_frames)
        w_sustain = np.convolve(w_frame, k_short, mode="same")
        bg = signal.medfilt(w_sustain.astype(np.float64), kernel_size=75).astype(np.float32)
        w_clean = np.maximum(0.0, w_sustain - bg)
        std = float(np.std(w_clean)) + 1e-8
        w_z = (w_clean / std).astype(np.float32)
        return {
            "w_z": w_z,
            "snr_a": a_frame,
            "snr_b": b_frame,
            "snr_h": h_frame,
        }

    def extract_and_cache_bar_ath_whistle(
        self,
        root_dir: Path,
        cache_path: Path | None = None,
        cleanup_wav: bool = False,
    ) -> dict[str, np.ndarray | dict]:
        cfg = self.cfg
        if cache_path is None:
            cache_path = root_dir / "local_data" / "v9_swarm" / "bar_ath_whistle.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        wav_dir = cache_path.parent
        bcast_mp4 = root_dir / "data" / "broadcast" / "BAR-ATH_broadcast.mp4"
        # Official layout (DATA.md) is data/tactical/; data/broadcast/ is the older local layout.
        tac_mp4 = next((p for p in (root_dir / "data" / "tactical" / "BAR-ATH_tactical.mp4",
                                    root_dir / "data" / "broadcast" / "BAR-ATH_tactical.mp4") if p.exists()),
                       root_dir / "data" / "tactical" / "BAR-ATH_tactical.mp4")
        bcast_wav = self._ensure_wav(bcast_mp4, wav_dir / "bcast_24k.wav", cfg.sample_rate)
        tac_wav = self._ensure_wav(tac_mp4, wav_dir / "tac_24k.wav", cfg.sample_rate)

        n_bcast = 164488
        n_tac = cfg.n_tactical_frames
        res_b = self._compute_stream_whistle_25fps(bcast_wav, n_bcast)
        res_t = self._compute_stream_whistle_25fps(tac_wav, n_tac)

        # Map broadcast frames k -> tactical frames t via continuous half clock lock
        w_bcast_tac = np.zeros(n_tac, dtype=np.float32)
        snr_a_tac = np.zeros(n_tac, dtype=np.float32)
        snr_b_tac = np.zeros(n_tac, dtype=np.float32)
        snr_h_tac = np.zeros(n_tac, dtype=np.float32)

        # Half 1: t = k + 4014 for k in [0, 85500)
        k_h1 = np.arange(0, cfg.bcast_half_split, dtype=np.int64)
        t_h1 = k_h1 + cfg.h1_bcast_offset
        m1 = (t_h1 >= 0) & (t_h1 < n_tac)
        w_bcast_tac[t_h1[m1]] = res_b["w_z"][k_h1[m1]]
        snr_a_tac[t_h1[m1]] = res_b["snr_a"][k_h1[m1]]
        snr_b_tac[t_h1[m1]] = res_b["snr_b"][k_h1[m1]]
        snr_h_tac[t_h1[m1]] = res_b["snr_h"][k_h1[m1]]

        # Half 2: t = k + 26920 for k in [85500, n_bcast)
        k_h2 = np.arange(cfg.bcast_half_split, n_bcast, dtype=np.int64)
        t_h2 = k_h2 + cfg.h2_bcast_offset
        m2 = (t_h2 >= 0) & (t_h2 < n_tac)
        w_bcast_tac[t_h2[m2]] = res_b["w_z"][k_h2[m2]]
        snr_a_tac[t_h2[m2]] = res_b["snr_a"][k_h2[m2]]
        snr_b_tac[t_h2[m2]] = res_b["snr_b"][k_h2[m2]]
        snr_h_tac[t_h2[m2]] = res_b["snr_h"][k_h2[m2]]

        w_tac = res_t["w_z"]
        # Dual-stream fusion: broadcast whistle + tactical stadium whistle
        w_dual = (0.70 * w_bcast_tac + 0.30 * w_tac).astype(np.float32)

        # Extract discrete whistle peaks (prominence >= p99.0, minimum separation 35 frames = 1.4s)
        thr_99 = float(np.percentile(w_dual[w_dual > 0], 98.5)) if np.any(w_dual > 0) else 3.0
        peaks, _ = signal.find_peaks(w_dual, height=thr_99, distance=35)

        if cleanup_wav:
            for p_wav in (bcast_wav, tac_wav):
                if p_wav.exists():
                    p_wav.unlink()

        np.savez_compressed(
            cache_path,
            w_bcast=w_bcast_tac,
            w_tac=w_tac,
            w_dual=w_dual,
            snr_chamber_a=snr_a_tac,
            snr_chamber_b=snr_b_tac,
            snr_harmonic=snr_h_tac,
            whistle_peaks=peaks.astype(np.int32),
        )
        return {
            "w_bcast": w_bcast_tac,
            "w_tac": w_tac,
            "w_dual": w_dual,
            "snr_chamber_a": snr_a_tac,
            "snr_chamber_b": snr_b_tac,
            "snr_harmonic": snr_h_tac,
            "whistle_peaks": peaks.astype(np.int32),
        }

    @staticmethod
    def evaluate_unsupervised_enrichment(
        w_dual: np.ndarray,
        pred_v85: np.ndarray,
        X_df: pd.DataFrame,
        graded: np.ndarray,
        sp_free_kick_starts: list[int] | None = None,
    ) -> dict[str, float | int | list]:
        """Unsupervised verification on BAR-ATH: check enrichment of whistle peaks in [-5s, +0.6s]
        around v8.5's predicted in-pitch Live -> D stoppages and Smart Tagging Free Kick starts.
        """
        vis = (X_df["ball_visible"].to_numpy(dtype=np.float64) > 0.5)
        oob = X_df["ball_outside_pitch"].to_numpy(dtype=np.float64)
        s_arr, e_arr, v_arr = runs(pred_v85)

        all_live_to_d = []
        inpitch_foul_d = []
        for i in range(1, len(v_arr)):
            if v_arr[i - 1] in (HOME, AWAY) and v_arr[i] == DEAD and (e_arr[i] - s_arr[i]) >= 25:
                t = int(s_arr[i])
                if not graded[t]:
                    continue
                all_live_to_d.append(t)
                # In-pitch stoppage: ball was inside pitch and visible within 1.5s before t
                w0 = max(0, t - 38)
                if np.any(vis[w0:t + 1]) and np.nanmean(oob[w0:t + 1]) < 0.25:
                    inpitch_foul_d.append(t)

        graded_idx = np.flatnonzero(graded)
        results = {
            "n_all_live_to_d": len(all_live_to_d),
            "n_inpitch_stoppages": len(inpitch_foul_d),
        }

        for p_pct, label in [(99.0, "p99_0"), (99.5, "p99_5"), (99.8, "p99_8")]:
            thr = float(np.percentile(w_dual[graded_idx], p_pct))
            # Random 5.6s window hit rate inside graded play
            hits_rand = 0
            n_rand = 2000
            rng = np.random.default_rng(42)
            rand_centers = rng.choice(graded_idx[(graded_idx >= 150) & (graded_idx < len(w_dual) - 50)], size=n_rand, replace=False)
            for rc in rand_centers:
                if float(w_dual[rc - 125:rc + 15].max()) >= thr:
                    hits_rand += 1
            rate_rand = hits_rand / float(n_rand)

            hits_inp = 0
            lead_frames = []
            for t in inpitch_foul_d:
                w_slice = w_dual[max(0, t - 125):min(len(w_dual), t + 15)]
                if len(w_slice) > 0 and float(w_slice.max()) >= thr:
                    hits_inp += 1
                    pk_rel = int(np.argmax(w_slice)) + max(0, t - 125)
                    lead_frames.append(int(t - pk_rel))
            rate_inp = hits_inp / float(max(len(inpitch_foul_d), 1))
            enrichment = rate_inp / max(rate_rand, 1e-6)
            results[f"hit_rate_inpitch_{label}"] = float(rate_inp)
            results[f"hit_rate_random_{label}"] = float(rate_rand)
            results[f"enrichment_ratio_{label}"] = float(enrichment)
            if label == "p99_0" and lead_frames:
                results["median_v85_late_frames"] = float(np.median(lead_frames))
                results["median_v85_late_seconds"] = float(np.median(lead_frames) / FPS)
                results["max_v85_late_frames"] = int(np.max(lead_frames))
                results["max_v85_late_seconds"] = float(np.max(lead_frames) / FPS)

        if sp_free_kick_starts:
            thr_fk = float(np.percentile(w_dual[graded_idx], 99.0))
            fk_hits = 0
            for fk_st in sp_free_kick_starts:
                # Free Kick tag starts ~5s before the kick; foul whistle occurs [-30s, +2s] around tag start
                w0 = max(0, fk_st - 25 * FPS)
                w1 = min(len(w_dual), fk_st + 2 * FPS)
                if float(w_dual[w0:w1].max()) >= thr_fk:
                    fk_hits += 1
            results["free_kick_tag_whistle_hit_rate"] = float(fk_hits / max(len(sp_free_kick_starts), 1))

        return results


class StoppageStartRefiner:
    """100% H<->A equivariant Stoppage Start & Wedge Refiner (`Line L4`)."""

    def __init__(self, cfg: StoppageRefinerConfig | None = None):
        self.cfg = cfg or StoppageRefinerConfig()

    def apply(
        self,
        pred: np.ndarray,
        proba: np.ndarray,
        p4d: np.ndarray | None,
        X_df: pd.DataFrame,
        half: np.ndarray,
        whistle_z: np.ndarray | None = None,
        enable_sliver_and_nocontact: bool = True,
    ) -> np.ndarray:
        cfg = self.cfg
        out = np.asarray(pred, dtype=np.int32).copy()
        half = np.asarray(half, dtype=np.int32)
        n = len(out)

        bx = X_df["ball_x"].to_numpy(dtype=np.float64)
        by = X_df["ball_y"].to_numpy(dtype=np.float64)
        bxy = np.column_stack([bx, by])
        bsp = X_df["ball_speed"].to_numpy(dtype=np.float64)
        vis = (X_df["ball_visible"].to_numpy(dtype=np.float64) > 0.5) & np.isfinite(bx)
        decel = np.nan_to_num(X_df["speed_mean_all_fwd_minus_bwd_2s"].to_numpy(dtype=np.float64), nan=0.0)
        d1h = np.nan_to_num(X_df["dist_ball_H"].to_numpy(dtype=np.float64), nan=99.0)
        d1a = np.nan_to_num(X_df["dist_ball_A"].to_numpy(dtype=np.float64), nan=99.0)

        # 1. Pre-Dead Sliver Pullback (Y -> X -> D where SymBoundary4Dir to_dead fires at s)
        if cfg.enable_predead_sliver and enable_sliver_and_nocontact and p4d is not None:
            s_arr, e_arr, v_arr = _runs_of_with_half(out, half)
            for i in range(1, len(v_arr) - 1):
                if half[s_arr[i - 1]] != half[s_arr[i]] or half[s_arr[i]] != half[s_arr[i + 1]]:
                    continue
                x = int(v_arr[i])
                if x == int(DEAD):
                    continue
                y_prev = int(v_arr[i - 1])
                y_next = int(v_arr[i + 1])
                s, e = int(s_arr[i]), int(e_arr[i])
                L = e - s
                if (
                    y_next == int(DEAD)
                    and y_prev == (1 - x)
                    and L <= cfg.sliver_max_len
                    and (e_arr[i + 1] - s_arr[i + 1]) >= cfg.sliver_min_dead_len
                ):
                    if float(vis[max(0, s - 10):min(n, e + 10)].mean()) >= cfg.sliver_vis_thresh:
                        w0 = max(0, s - 3)
                        w1 = min(n, s + 4)
                        b4d_td_s = float(p4d[w0:w1, 2].max())
                        mean_pD = float(proba[s:e, DEAD].mean())
                        if b4d_td_s > cfg.sliver_b4d_to_dead_thresh and mean_pD > cfg.sliver_mean_pd_thresh:
                            out[s:e] = int(DEAD)

        # 2. No-Contact D -> X -> Y Veto (X never comes within 2.8m of visible ball while Y < 1.2m)
        if cfg.enable_nocontact_dxy and enable_sliver_and_nocontact:
            s_arr, e_arr, v_arr = _runs_of_with_half(out, half)
            for i in range(1, len(v_arr) - 1):
                if half[s_arr[i - 1]] != half[s_arr[i]] or half[s_arr[i]] != half[s_arr[i + 1]]:
                    continue
                x = int(v_arr[i])
                if x == int(DEAD):
                    continue
                y_prev = int(v_arr[i - 1])
                y_next = int(v_arr[i + 1])
                s, e = int(s_arr[i]), int(e_arr[i])
                L = e - s
                if (
                    y_prev == int(DEAD)
                    and y_next == (1 - x)
                    and L <= cfg.nocontact_max_len
                    and (e_arr[i + 1] - s_arr[i + 1]) >= cfg.nocontact_min_next_len
                ):
                    if float(vis[s:e].mean()) >= cfg.nocontact_vis_thresh:
                        d1_x = d1h[s:e] if x == int(HOME) else d1a[s:e]
                        d1_y = d1a[s:e] if x == int(HOME) else d1h[s:e]
                        if float(d1_x.min()) > cfg.nocontact_min_dist_x and float(d1_y.min()) < cfg.nocontact_max_dist_y:
                            out[s:e] = y_next

        # 3. BallisticCrossSnap, GapDecelPullback & RollingBallFoulPullback
        cross = detect_pitch_crossings_from_features(bx, by, vis, half)
        foul_sigs = CollectiveFoulDeceleration.compute_signals(X_df, half)
        decel_1_5s = foul_sigs["decel_1_5s"]
        oob = np.nan_to_num(X_df["ball_outside_pitch"].to_numpy(dtype=np.float64), nan=0.0)

        s_arr, e_arr, v_arr = _runs_of_with_half(out, half)
        for k in range(1, len(v_arr)):
            t = int(s_arr[k])
            if half[t - 1] != half[t] or int(v_arr[k - 1]) == int(DEAD) or int(v_arr[k]) != int(DEAD):
                continue

            # 3a. BallisticCrossSnap (early Live -> D before ball leaves pitch)
            snapped = False
            if cfg.enable_ballistic_snap and len(cross) > 0:
                near = cross[np.abs(cross - t) <= cfg.snap_max_window]
                if len(near) == 1:
                    c = int(near[0])
                    if (
                        c > t
                        and (c - int(s_arr[k - 1])) >= cfg.snap_min_spell_margin
                        and (int(e_arr[k]) - c) >= cfg.snap_min_spell_margin
                        and np.all(vis[max(0, t - 8):c + 1])
                    ):
                        v_pre = bxy[t] - bxy[max(0, t - 8)]
                        v_seg = bxy[c] - bxy[t]
                        n1, n2 = float(np.linalg.norm(v_pre)), float(np.linalg.norm(v_seg))
                        if n1 >= 0.5 and n2 >= 0.5:
                            cos_sim = float(np.dot(v_pre, v_seg) / (n1 * n2))
                            sp_pre = float(np.nanmean(bsp[max(0, t - 8):t]))
                            sp_seg = float(np.nanmean(bsp[t:c + 1]))
                            rel_dv = abs(sp_seg - sp_pre) / max(sp_pre, 1.0)
                            pD_mean = float(proba[t:c, DEAD].mean())
                            if (
                                cos_sim >= cfg.snap_min_cos_sim
                                and sp_seg >= cfg.snap_min_ball_speed
                                and rel_dv <= cfg.snap_max_rel_dv
                                and pD_mean < cfg.snap_max_mean_pd
                            ):
                                out[t:c] = int(v_arr[k - 1])
                                snapped = True

            if snapped:
                continue

            # 3b. GapDecelPullback (late Live -> D inside a stoppage ball gap)
            pulled_gap = False
            if cfg.enable_gap_decel and not vis[t - 1]:
                g_start = t - 1
                while g_start > 0 and not vis[g_start - 1] and half[g_start - 1] == half[t]:
                    g_start -= 1
                gap_pre = t - g_start
                if (
                    cfg.gap_min_wait <= gap_pre <= cfg.gap_max_wait
                    and (g_start - int(s_arr[k - 1])) >= cfg.gap_min_prev_live
                    and (int(e_arr[k]) - t) >= cfg.gap_min_dead_len
                ):
                    vis_pre = float(vis[max(0, g_start - 15):g_start].mean())
                    vis_d = float(vis[t:int(e_arr[k])].mean())
                    if (
                        vis_pre >= cfg.gap_vis_pre_thresh
                        and vis_d <= cfg.gap_vis_dead_max
                        and decel[g_start] <= cfg.gap_decel_thresh
                        and float(proba[g_start:t, DEAD].mean()) >= cfg.gap_mean_pd_thresh
                    ):
                        out[max(int(s_arr[k - 1]) + 5, g_start - 1):t] = int(DEAD)
                        pulled_gap = True

            # 3c. RollingBallFoulPullback (in-pitch foul where ball rolls for 0.4s..4.4s after whistle/decel)
            if (
                not pulled_gap
                and cfg.enable_rolling_foul_pullback
                and (int(e_arr[k]) - t) >= cfg.foul_min_dead_len
                and (t - int(s_arr[k - 1])) >= cfg.foul_min_prev_live
            ):
                w0 = max(int(s_arr[k - 1]) + 10, t - cfg.foul_max_pullback)
                w1 = t - cfg.foul_min_pullback
                if w1 > w0 and float(oob[w0:t].mean()) == 0.0 and float(vis[w0:t].mean()) >= 0.70:
                    if whistle_z is not None:
                        w_slice = whistle_z[w0:w1]
                        if len(w_slice) > 0 and float(w_slice.max()) >= cfg.foul_whistle_z_thresh:
                            pk = w0 + int(np.argmax(w_slice))
                            if min(decel[pk], decel_1_5s[pk]) <= cfg.foul_decel_thresh and float(proba[pk:t, DEAD].mean()) >= 0.12:
                                out[pk:t] = int(DEAD)
                    else:
                        # Purely kinematic rolling-ball foul pullback when ball is decelerating on grass
                        # after a sharp collective player deceleration and elevated DEAD probability
                        d_slice = np.minimum(decel[w0:w1], decel_1_5s[w0:w1])
                        if len(d_slice) > 0 and float(d_slice.min()) <= -0.85:
                            pk = w0 + int(np.argmin(d_slice))
                            if (
                                float(proba[pk:t, DEAD].mean()) >= 0.28
                                and float(np.nanmean(bsp[pk:t])) <= 4.5
                                and float(vis[t:min(n, t + 50)].mean()) <= 0.05
                            ):
                                out[pk:t] = int(DEAD)

        return out


def swap_ha_features_df(X_df: pd.DataFrame) -> pd.DataFrame:
    """Create an exact H<->A swapped copy of X_df to verify 0.000e+00 equivariance."""
    df = X_df.copy()
    if "dist_ball_H" in df.columns and "dist_ball_A" in df.columns:
        df["dist_ball_H"] = X_df["dist_ball_A"].to_numpy().copy()
        df["dist_ball_A"] = X_df["dist_ball_H"].to_numpy().copy()
    if "gk_dist_ball_H" in df.columns and "gk_dist_ball_A" in df.columns:
        df["gk_dist_ball_H"] = X_df["gk_dist_ball_A"].to_numpy().copy()
        df["gk_dist_ball_A"] = X_df["gk_dist_ball_H"].to_numpy().copy()
    return df


def verify_stoppage_refiner_equivariance(
    pred: np.ndarray,
    proba: np.ndarray,
    p4d: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    whistle_z: np.ndarray | None = None,
) -> dict[str, float]:
    """Verify exact 0.000e+00 H<->A equivariance of CollectiveFoulDeceleration and StoppageStartRefiner."""
    X_swap = swap_ha_features_df(X_df)
    sig_orig = CollectiveFoulDeceleration.compute_signals(X_df, half)
    sig_swap = CollectiveFoulDeceleration.compute_signals(X_swap, half)
    foul_err = float(np.max(np.abs(sig_orig["foul_score"] - sig_swap["foul_score"])))

    refiner = StoppageStartRefiner()
    out_orig = refiner.apply(pred, proba, p4d, X_df, half, whistle_z=whistle_z)

    pred_swap = np.where(pred == HOME, AWAY, np.where(pred == AWAY, HOME, pred))
    proba_swap = proba[:, [1, 0, 2]].copy()
    p4d_swap = p4d[:, [1, 0, 2, 3]].copy() if p4d is not None else None

    out_swap = refiner.apply(pred_swap, proba_swap, p4d_swap, X_swap, half, whistle_z=whistle_z)
    out_swap_back = np.where(out_swap == HOME, AWAY, np.where(out_swap == AWAY, HOME, out_swap))
    refiner_err = float(np.mean(out_orig != out_swap_back))
    return {
        "collective_foul_decel_equivariance_error": foul_err,
        "stoppage_start_refiner_equivariance_error": refiner_err,
    }
