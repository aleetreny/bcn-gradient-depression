"""Prepare a compact, self-contained public-only data bundle (~275 MB) in `local_data/v9_swarm/modal_bundle/`
for uploading to our private Modal Volume (`bcn-possession-v9`).

Includes ONLY 100% public datasets:
- 7 Sportec matches (`P, B, half, graded, attack_sign, slot_team, slot_is_gk, pitch, labels, ball_z` in float16/float32)
- 3 Metrica Sample matches (`sample_1, sample_2, sample_3`)
- `degradation_params.json`, `ball_interp_params.json`, `degrade_v3.json`, `degrade_v6.json`

Competition-gated `BAR-ATH` data is NEVER included in the Modal bundle; all `BAR-ATH` inference runs 100% locally.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.data.common import Match
from src.data.metrica_sample import GAMES, load as load_sample
from src.data.sportec import MATCH_IDS, load_all as load_sportec

OUT_DIR = ROOT / "local_data" / "v9_swarm" / "modal_bundle"
VOLUME_NAME = "bcn-possession-v9"


def save_match_compact(path: Path, m: Match, ball_z: np.ndarray | None = None) -> None:
    kw = dict(
        P=m.P.astype(np.float32),
        B=m.B.astype(np.float32),
        half=np.asarray(m.half, dtype=np.int8),
        graded=np.asarray(m.graded, dtype=bool),
        attack_sign=np.asarray(m.attack_sign, dtype=np.int8),
        slot_team=np.asarray(m.slot_team, dtype=np.int8),
        slot_is_gk=np.asarray(m.slot_is_gk, dtype=bool),
        pitch=np.asarray(m.pitch, dtype=np.float32),
        direction_normalized=np.array(bool(m.direction_normalized)),
        source=np.array(str(m.source or "match")),
    )
    if m.labels is not None:
        kw["labels"] = np.asarray(m.labels, dtype=np.int8)
    if ball_z is not None:
        kw["ball_z"] = np.asarray(ball_z, dtype=np.float32)
    np.savez_compressed(path, **kw)


def load_match_compact(path: Path) -> tuple[Match, np.ndarray | None]:
    with np.load(path) as z:
        labels = z["labels"].astype(np.int8) if "labels" in z.files else None
        ball_z = z["ball_z"].astype(np.float32) if "ball_z" in z.files else None
        meta = {"ball_z": ball_z} if ball_z is not None else {}
        m = Match(
            P=z["P"].astype(np.float32),
            B=z["B"].astype(np.float32),
            half=z["half"].astype(np.int8),
            graded=z["graded"].astype(bool),
            attack_sign=z["attack_sign"].astype(np.int8),
            slot_team=z["slot_team"].astype(np.int8),
            slot_is_gk=z["slot_is_gk"].astype(bool),
            pitch=(float(z["pitch"][0]), float(z["pitch"][1])),
            home_name="H",
            away_name="A",
            labels=labels,
            source=str(z["source"]),
            direction_normalized=bool(z["direction_normalized"]),
            meta=meta,
        )
        return m, ball_z


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare public-only Sportec + Metrica Modal bundle.")
    ap.add_argument("--upload", action="store_true", help="Upload public bundle to Modal volume bcn-possession-v9")
    args = ap.parse_args()

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure zero competition-gated BAR-ATH files ever exist in the Modal bundle directory
    for legacy_atd in OUT_DIR.glob("atd_*.npz"):
        legacy_atd.unlink()
        print(f"Removed legacy competition file from modal_bundle: {legacy_atd.name}", flush=True)

    for fn in ("degradation_params.json", "ball_interp_params.json"):
        src = ROOT / "local_data" / fn
        if src.is_file():
            shutil.copy2(src, OUT_DIR / fn)
    for fn in ("degrade_v3.json", "degrade_v6.json"):
        src = ROOT / "configs" / fn
        if src.is_file():
            shutil.copy2(src, OUT_DIR / fn)

    sportec = load_sportec()
    for mid in MATCH_IDS:
        m = sportec[mid]
        p = OUT_DIR / f"sportec_{mid}.npz"
        if not p.is_file():
            save_match_compact(p, m, ball_z=m.meta.get("ball_z"))
            print(f"Saved {p.name} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)

    for gid in GAMES:
        ms = load_sample(gid)
        p = OUT_DIR / f"sample_{gid}.npz"
        if not p.is_file():
            save_match_compact(p, ms)
            print(f"Saved {p.name} ({p.stat().st_size / 1e6:.1f} MB)", flush=True)

    tot_mb = sum(f.stat().st_size for f in OUT_DIR.glob("*") if f.is_file()) / 1e6
    print(f"Public-only Modal bundle ready in {time.time() - t0:.1f}s | Total size: {tot_mb:.1f} MB", flush=True)

    if args.upload:
        t_up = time.time()
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "create", VOLUME_NAME],
            check=False,
        )
        subprocess.run(
            [sys.executable, "-m", "modal", "volume", "put", VOLUME_NAME, str(OUT_DIR), "/modal_bundle", "--force"],
            check=True,
        )
        print(f"Uploaded public-only bundle to Modal Volume `{VOLUME_NAME}:/modal_bundle` in {time.time() - t_up:.1f}s", flush=True)


if __name__ == "__main__":
    main()
