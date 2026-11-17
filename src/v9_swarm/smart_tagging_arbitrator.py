"""M5-v9: Smart Tagging Set-Piece, Foul Stoppage & Blackout Arbitrator.

Implements a 100% H<->A equivariant (`0.000e+00` symmetry error) multi-modal fusion layer
on top of `v8.5` (`run_unified_v8_forensic_champion.py`) and `StoppageStartRefiner` (`Line L4`):
1. Parses `data/tactical/BAR-ATH_tactical_smart_tagging.xml`:
   - Strictly excludes noisy spatial zone tags (`PLAYERS IN THE BOX`, `BALL IN FINAL THIRD`,
     `BALL IN THE BOX`) which have only 55-61% possession accuracy.
   - Uses `SET PIECES` (`96.2%` public accuracy; `98.1%` on Sportec `03_02_events_raw`) and
     high-precision directional phase tags (`BUILD UP`, `CREATING CHANCES`,
     `DEFENDING IN ATTACKING/DEFENSIVE THIRD`: `100%`; `LONG BALL`: `90.2%`;
     `PROGRESSION` / `DEFENDING IN MIDDLE THIRD`: `81.7%`; `GOALS`: `100%`).
   - Automatically infers `Team` for `SET PIECES` with `Team="N/A"` (e.g., `Throw-in`) when an
     unambiguous directional phase tag overlaps the first 6 seconds of the set piece.
2. Provides `extract_simulated_smart_tags_sportec()` and `extract_simulated_smart_tags_sample()`
   to validate the exact same arbitration rules on `J1A` (`Sportec v3`), `J1B-Std`, `J1B-Gen`,
   `J2-Raw`, and `J2-noD` using realistic `BAR-ATH` window offsets (`start = kick - 5s`,
   `end = start + 35s`).
3.Guarantees `0.000e+00` H<->A equivariance under simultaneous swapping of tracking teams and
   Smart Tagging team labels.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from src.data.common import AWAY, DEAD, FPS, HOME, runs
from src.v9_swarm.whistle_and_stoppage import (
    CollectiveFoulDeceleration,
    StoppageStartRefiner,
    swap_ha_features_df,
)

TEAM_MAP_BAR_ATH = {"Barcelona": int(HOME), "Atletic Club": int(AWAY)}
DEFENSIVE_PHASE_CODES = {
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN ATTACKING THIRD",
    "DEFENSIVE TRANSITION",
}
HIGH_PREC_PHASE_WEIGHTS = {
    "BUILD UP": 1.00,
    "CREATING CHANCES": 1.00,
    "DEFENDING IN ATTACKING THIRD": 1.00,
    "DEFENDING IN DEFENSIVE THIRD": 1.00,
    "GOALS": 1.25,
    "LONG BALL": 0.90,
    "PROGRESSION": 0.82,
    "DEFENDING IN MIDDLE THIRD": 0.82,
}


@dataclass(frozen=True)
class SmartTaggingArbitratorConfig:
    enable_post_dead_wedge_fix: bool = True
    enable_post_dead_dxd_fix: bool = True
    enable_swallowed_setpiece_recovery: bool = True
    enable_free_kick_foul_pullback: bool = True
    enable_blackout_phase_anchor: bool = True
    # Matching window between SET PIECES expected kick (start + 5s) and D -> Live restart a
    sp_match_min_offset_frames: int = -80  # -3.2s
    sp_match_max_offset_frames: int = 12   # +0.48s
    # Rule 1: D -> X -> t_sp post-dead wrong-team wedge
    wedge_min_next_len: int = 45
    wedge_short_len: int = 45
    wedge_short_max_px: float = 0.78
    wedge_med_len: int = 60
    wedge_med_max_px: float = 0.65
    wedge_blind_len: int = 115
    wedge_blind_max_vis: float = 0.30
    wedge_blind_max_margin: float = 0.35
    # Rule 2: D -> X -> D uncertain short set piece
    dxd_max_len: int = 75
    dxd_max_margin: float = 0.25
    # Rule 3: Swallowed SET PIECES inside long DEAD spell
    swallowed_min_vis_frames: int = 45
    swallowed_min_max_bs: float = 8.0
    swallowed_max_mean_pd: float = 0.50
    # Rule 4: Free Kick stoppage pullback
    fk_pullback_max_frames: int = 110
    fk_pullback_min_frames: int = 10
    # Rule 5: Blackout phase anchor
    blackout_min_frames: int = 100
    blackout_max_prob_margin: float = 0.18

    def to_dict(self) -> dict:
        return asdict(self)


def load_bar_ath_smart_tags(
    root_dir: Path,
    n_frames: int = 199319,
) -> tuple[list[dict], np.ndarray]:
    """Load and parse official `BAR-ATH_tactical_smart_tagging.xml` into:
      - `sp_tags`: list of SET PIECES dicts with `start`, `end`, `team` (explicit or phase-inferred), `type`
      - `phase_score`: `(n_frames, 2)` array of high-precision directional phase weights for `[HOME, AWAY]`.
    """
    from starter.load_smart_tagging import load as load_tags

    raw_tags = load_tags("BAR-ATH")
    phase_score = np.zeros((n_frames, 2), dtype=np.float64)
    sp_tags: list[dict] = []

    for t in raw_tags:
        c_up = str(t["code"]).upper()
        tm_str = str(t["tags"].get("Team", ""))
        a = max(0, int(t["start_frame"]))
        b = min(n_frames, int(t["end_frame"]))
        if b <= a:
            continue
        if c_up == "SET PIECES":
            tm_id = TEAM_MAP_BAR_ATH.get(tm_str, -1)
            sp_tags.append({
                "start": a,
                "end": b,
                "explicit_team": tm_id,
                "team": tm_id,
                "type": str(t["tags"].get("Type", "?")),
            })
            if tm_id in (HOME, AWAY):
                phase_score[a:min(b, a + 8 * FPS), tm_id] += 1.25
        elif c_up in HIGH_PREC_PHASE_WEIGHTS and tm_str in TEAM_MAP_BAR_ATH:
            w = HIGH_PREC_PHASE_WEIGHTS[c_up]
            tm_raw = TEAM_MAP_BAR_ATH[tm_str]
            poss_tm = (1 - tm_raw) if c_up in DEFENSIVE_PHASE_CODES else tm_raw
            phase_score[a:b, poss_tm] += w

    # Infer team for SET PIECES where Team == "N/A" (e.g., Throw-in) if high-precision phase tags agree
    for sp in sp_tags:
        if sp["team"] == -1:
            w0 = sp["start"]
            w1 = min(n_frames, sp["start"] + 6 * FPS)
            sh = float(phase_score[w0:w1, HOME].sum())
            sa = float(phase_score[w0:w1, AWAY].sum())
            if max(sh, sa) >= 25.0 and abs(sh - sa) / max(sh + sa, 1e-6) >= 0.50:
                sp["team"] = int(HOME) if sh > sa else int(AWAY)

    return sp_tags, phase_score


def extract_simulated_smart_tags_sportec(root_dir: Path, mid: str, m) -> list[dict]:
    """Extract Smart Tagging-equivalent `SET PIECES` intervals (`start = kick - 5s`, `end = start + 35s`)
    from Sportec `DFL_03_02_events_raw` for LOMO validation.
    """
    spec = importlib.util.spec_from_file_location("s15", root_dir / "scripts" / "s1_5_flag_semantics.py")
    s15 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(s15)

    meta_root = ET.parse(s15._file("02_01_matchinformation", mid, s15.DATA_DIR)).getroot()
    home_id = next(t.get("TeamId") for t in meta_root.iter("Team") if "home" in t.get("Role", "").lower())
    away_id = next(
        t.get("TeamId")
        for t in meta_root.iter("Team")
        if "guest" in t.get("Role", "").lower() or "away" in t.get("Role", "").lower()
    )
    origins = s15.half_origins(mid)
    n1 = int(np.sum(m.half == 1))
    lag, _ = s15.event_lag(m, s15.read_events(mid, m))

    sp_tags = []
    for e in ET.parse(s15._file("03_02_events_raw", mid, s15.DATA_DIR)).getroot().iter("Event"):
        if not len(e):
            continue
        node = e[0]
        if node.tag not in ("ThrowIn", "FreeKick", "CornerKick", "GoalKick", "Penalty", "KickOff"):
            continue
        t = s15.parse_time(e.get("EventTime"))
        h = 2 if t >= origins[2][1] else 1
        kick_f = (0 if h == 1 else n1) + int(round((t - origins[h][1]).total_seconds() * FPS)) + lag
        tid = node.get("Team") or node.get("TeamPossession")
        tm = int(HOME) if tid == home_id else (int(AWAY) if tid == away_id else -1)
        if 0 <= kick_f < m.n_frames and tm in (HOME, AWAY):
            st_f = max(0, kick_f - 5 * FPS)
            sp_tags.append({
                "start": st_f,
                "end": min(m.n_frames, st_f + 35 * FPS),
                "team": tm,
                "type": node.tag,
            })
    return sp_tags


def extract_simulated_smart_tags_sample(root_dir: Path, gid: int) -> list[dict]:
    """Extract Smart Tagging-equivalent `SET PIECES` intervals (`start = kick - 5s`, `end = start + 35s`)
    from Metrica Sample 1-3 raw event files via `src.data.metrica_sample.read_events` for OOD validation.
    """
    from src.data.metrica_sample import read_events

    ev = read_events(gid)
    sp_tags = []
    is_sp = (ev["type"] == "SET PIECE") | ((ev["type"] == "PASS") & ev["subtype"].str.contains("GOAL KICK"))
    for _, r in ev[is_sp].iterrows():
        tm = int(HOME) if r["team"] == "H" else (int(AWAY) if r["team"] == "A" else -1)
        kf = int(r["start_frame"]) - 1
        if tm in (HOME, AWAY) and kf >= 0:
            st_f = max(0, kf - 5 * FPS)
            sp_tags.append({
                "start": st_f,
                "end": st_f + 35 * FPS,
                "team": tm,
                "type": str(r["subtype"] or r["type"]),
            })
    return sp_tags


class SmartTaggingArbitrator:
    """100% H<->A equivariant Set-Piece, Stoppage & Blackout Arbitrator (`M5-v9`)."""

    def __init__(self, cfg: SmartTaggingArbitratorConfig | None = None):
        self.cfg = cfg or SmartTaggingArbitratorConfig()

    def apply(
        self,
        pred: np.ndarray,
        proba: np.ndarray,
        X_df: pd.DataFrame,
        half: np.ndarray,
        sp_tags: list[dict],
        phase_score: np.ndarray | None = None,
        whistle_z: np.ndarray | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        out = np.asarray(pred, dtype=np.int32).copy()
        half = np.asarray(half, dtype=np.int32)
        n = len(out)
        vis = (X_df["ball_visible"].to_numpy(dtype=np.float64) > 0.5)
        bs = np.nan_to_num(X_df["ball_speed"].to_numpy(dtype=np.float64), nan=0.0)
        decel = np.nan_to_num(X_df["speed_mean_all_fwd_minus_bwd_2s"].to_numpy(dtype=np.float64), nan=0.0)

        chg = np.flatnonzero((out[1:] != out[:-1]) | (half[1:] != half[:-1])) + 1
        s_arr = np.concatenate([[0], chg])
        e_arr = np.concatenate([chg, [n]])
        v_arr = out[s_arr]

        # 1 & 2. Arbitrate D -> Live restarts against matched SET PIECES tags
        for i in range(1, len(v_arr)):
            if half[s_arr[i - 1]] != half[s_arr[i]]:
                continue
            if int(v_arr[i - 1]) != int(DEAD) or (e_arr[i - 1] - s_arr[i - 1]) < 25:
                continue
            x = int(v_arr[i])
            if x not in (int(HOME), int(AWAY)):
                continue
            a, b = int(s_arr[i]), int(e_arr[i])
            L = b - a

            cands = [
                sp
                for sp in sp_tags
                if cfg.sp_match_min_offset_frames <= ((sp["start"] + 5 * FPS) - a) <= cfg.sp_match_max_offset_frames
                and sp.get("team", -1) in (HOME, AWAY)
            ]
            if not cands:
                continue
            best_sp = min(cands, key=lambda sp: abs((sp["start"] + 5 * FPS) - a))
            t_sp = int(best_sp["team"])
            if t_sp == x:
                continue

            # Rule 1: D -> X -> t_sp where X != t_sp is a post-dead wrong-team wedge
            if (
                cfg.enable_post_dead_wedge_fix
                and i + 1 < len(v_arr)
                and half[s_arr[i + 1]] == half[a]
                and int(v_arr[i + 1]) == t_sp
            ):
                next_L = int(e_arr[i + 1] - s_arr[i + 1])
                vis_seg = float(vis[a:b].mean())
                pX = float(proba[a:b, x].mean())
                pSP = float(proba[a:b, t_sp].mean())
                if next_L >= cfg.wedge_min_next_len and (
                    (L <= cfg.wedge_short_len and pX < cfg.wedge_short_max_px)
                    or (L <= cfg.wedge_med_len and pX < cfg.wedge_med_max_px)
                    or (
                        L <= cfg.wedge_blind_len
                        and vis_seg <= cfg.wedge_blind_max_vis
                        and (pX - pSP) < cfg.wedge_blind_max_margin
                    )
                ):
                    out[a:b] = t_sp
                    continue

            # Rule 2: D -> X -> D where X != t_sp and model is uncertain
            if (
                cfg.enable_post_dead_dxd_fix
                and i + 1 < len(v_arr)
                and int(v_arr[i + 1]) == int(DEAD)
                and L <= cfg.dxd_max_len
            ):
                pX = float(proba[a:b, x].mean())
                pSP = float(proba[a:b, t_sp].mean())
                if (pX - pSP) < cfg.dxd_max_margin:
                    out[a:b] = t_sp
                    continue

        # 3. Recover swallowed SET PIECES inside long DEAD spells (>= 40s, e.g. Free Kick Goal 73951..73996)
        if cfg.enable_swallowed_setpiece_recovery:
            for sp in sp_tags:
                t_sp = int(sp.get("team", -1))
                if t_sp not in (int(HOME), int(AWAY)):
                    continue
                w0 = max(0, int(sp["start"]))
                w1 = min(n, int(sp["end"]), w0 + 16 * FPS)
                if w1 - w0 < cfg.swallowed_min_vis_frames or float((out[w0:w1] != DEAD).mean()) >= 0.05:
                    continue
                v_mask = vis[w0:w1] & (out[w0:w1] == DEAD)
                s_v, e_v, val_v = runs(v_mask.astype(int))
                for j in range(len(val_v)):
                    if val_v[j] == 1 and (e_v[j] - s_v[j]) >= cfg.swallowed_min_vis_frames:
                        f0, f1 = w0 + int(s_v[j]), w0 + int(e_v[j])
                        if (
                            half[f0] > 0
                            and float(bs[f0:f1].max()) >= cfg.swallowed_min_max_bs
                            and float(proba[f0:f1, DEAD].mean()) < cfg.swallowed_max_mean_pd
                        ):
                            out[f0:f1] = t_sp

        # 4. Free Kick stoppage pullback when a Free Kick tag follows an in-pitch stoppage with whistle/decel
        if cfg.enable_free_kick_foul_pullback:
            foul_sigs = CollectiveFoulDeceleration.compute_signals(X_df, half)
            decel_1_5s = foul_sigs["decel_1_5s"]
            chg = np.flatnonzero((out[1:] != out[:-1]) | (half[1:] != half[:-1])) + 1
            s_arr = np.concatenate([[0], chg])
            e_arr = np.concatenate([chg, [n]])
            v_arr = out[s_arr]
            fk_tags = [sp for sp in sp_tags if "free" in str(sp.get("type", "")).lower()]
            for i in range(1, len(v_arr)):
                if int(v_arr[i]) != int(DEAD) or int(v_arr[i - 1]) not in (int(HOME), int(AWAY)):
                    continue
                t = int(s_arr[i])
                e_d = int(e_arr[i])
                if (e_d - t) < 50 or (t - int(s_arr[i - 1])) < 25:
                    continue
                # Check if this DEAD spell ends with a Free Kick tag
                has_fk = any(abs((fk["start"] + 5 * FPS) - e_d) <= 10 * FPS for fk in fk_tags)
                if not has_fk:
                    continue
                w0 = max(int(s_arr[i - 1]) + 15, t - cfg.fk_pullback_max_frames)
                w1 = t - cfg.fk_pullback_min_frames
                if w1 <= w0:
                    continue
                if whistle_z is not None and float(whistle_z[w0:w1].max()) >= 2.8:
                    pk = w0 + int(np.argmax(whistle_z[w0:w1]))
                    if min(decel[pk], decel_1_5s[pk]) <= -0.30 and float(proba[pk:t, DEAD].mean()) >= 0.10:
                        out[pk:t] = int(DEAD)

        # 5. Tactical blackout phase anchor during long tracking gaps (>= 4s) where model is uncertain
        if cfg.enable_blackout_phase_anchor and phase_score is not None:
            n_vis_all = (
                np.nan_to_num(X_df["n_visible_all"].to_numpy(dtype=np.float64), nan=22.0)
                if "n_visible_all" in X_df.columns
                else np.full(n, 22.0)
            )
            blackout = (~vis) & (n_vis_all < 2.0) & (out != DEAD) & (half > 0)
            s_b, e_b, val_b = runs(blackout.astype(int))
            for j in range(len(val_b)):
                if val_b[j] == 1 and (e_b[j] - s_b[j]) >= cfg.blackout_min_frames:
                    f0, f1 = int(s_b[j]), int(e_b[j])
                    pH = float(proba[f0:f1, HOME].mean())
                    pA = float(proba[f0:f1, AWAY].mean())
                    if abs(pH - pA) < cfg.blackout_max_prob_margin:
                        sh = float(phase_score[f0:f1, HOME].sum())
                        sa = float(phase_score[f0:f1, AWAY].sum())
                        if max(sh, sa) >= 50.0 and abs(sh - sa) / max(sh + sa, 1e-6) >= 0.60:
                            out[f0:f1] = int(HOME) if sh > sa else int(AWAY)

        return out


def verify_smart_tagging_equivariance(
    pred: np.ndarray,
    proba: np.ndarray,
    p4d: np.ndarray,
    X_df: pd.DataFrame,
    half: np.ndarray,
    sp_tags: list[dict],
    phase_score: np.ndarray | None = None,
    whistle_z: np.ndarray | None = None,
) -> dict[str, float]:
    """Verify exact 0.000e+00 H<->A equivariance of the combined StoppageStartRefiner + SmartTaggingArbitrator."""
    refiner = StoppageStartRefiner()
    arb = SmartTaggingArbitrator()

    p1 = refiner.apply(pred, proba, p4d, X_df, half, whistle_z=whistle_z)
    out_orig = arb.apply(p1, proba, X_df, half, sp_tags, phase_score=phase_score, whistle_z=whistle_z)

    # Swap H <-> A across all inputs
    pred_swap = np.where(pred == HOME, AWAY, np.where(pred == AWAY, HOME, pred))
    proba_swap = proba[:, [1, 0, 2]].copy()
    p4d_swap = p4d[:, [1, 0, 2, 3]].copy() if p4d is not None else None
    X_swap = swap_ha_features_df(X_df)
    sp_tags_swap = []
    for sp in sp_tags:
        sp_c = dict(sp)
        if sp_c.get("team") == HOME:
            sp_c["team"] = int(AWAY)
        elif sp_c.get("team") == AWAY:
            sp_c["team"] = int(HOME)
        sp_tags_swap.append(sp_c)
    phase_swap = phase_score[:, [1, 0]].copy() if phase_score is not None else None

    p1_swap = refiner.apply(pred_swap, proba_swap, p4d_swap, X_swap, half, whistle_z=whistle_z)
    out_swap = arb.apply(p1_swap, proba_swap, X_swap, half, sp_tags_swap, phase_score=phase_swap, whistle_z=whistle_z)
    out_swap_back = np.where(out_swap == HOME, AWAY, np.where(out_swap == AWAY, HOME, out_swap))

    err = float(np.mean(out_orig != out_swap_back))
    return {"combined_l4_m5_equivariance_error": err}
