"""Inspect LOCAL competition data paths, without uploading, parsing or exposing files.

Run on your own machine:
    python scripts/check_local_data.py
    python scripts/check_local_data.py --level features
    python scripts/check_local_data.py --level video

The competition's restricted data must not be uploaded to public git or hosted LLMs.
This utility only checks local existence and nonempty file sizes; it never reads content.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from games import DATA_DIR, asset_paths  # noqa: E402

LEVELS = {
    "tracking": ("atd_meta", "atd_raw"),
    "features": ("atd_meta", "atd_raw", "smart_tagging", "homography"),
    "video": ("atd_meta", "atd_raw", "smart_tagging", "homography", "video"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Check required local data files without loading the data")
    parser.add_argument("--level", choices=LEVELS, default="features",
                        help="Minimum asset set: tracking, features (default), video")
    parser.add_argument("--data-dir", default=None,
                        help="Dataset root containing tactical/ (default MTAH_DATA_DIR or ./data)")
    args = parser.parse_args()
    data_dir = Path(args.data_dir or DATA_DIR).expanduser().resolve()
    paths = asset_paths(data_dir=str(data_dir))
    print("Local data root: {}".format(data_dir))
    print("Scope: tactical; broadcast is deliberately not needed for the initial pipeline.")
    missing = []
    for kind in LEVELS[args.level]:
        path = Path(paths[kind])
        if path.is_file() and path.stat().st_size > 0:
            print("OK       {:<16} {}".format(kind, path.name))
        else:
            print("MISSING  {:<16} {}".format(kind, path.name))
            missing.append(path)
    if missing:
        print("\nPlace the original files under {}/tactical/".format(data_dir))
        print("or pass --data-dir /absolute/path/to/dataset or set MTAH_DATA_DIR.")
        print("Do not upload the restricted files to GitHub or any hosted/cloud LLM.")
        return 1
    print("\nAll required files exist and are nonempty. This does not prove they are complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
