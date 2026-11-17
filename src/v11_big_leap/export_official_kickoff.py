"""Export local v11 predictions with the official second-half kickoff boundary.

The boundary comes from published match timing, not possession ground truth.
This writes a new algorithmic output and does not alter an existing submission.
"""
import json
import os
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from evaluation.possession_metrics import read_label_csv, validate_submission
from games import GAMES, anchor_video_frame
from src.data.common import DEAD
from src.data.metrica_atd import load as load_atd
from src.v10_first_principles.rebuild_trajectory_inference import _verified_file


def apply_second_half_kickoff(pred, half, first_play_frame):
    """Return predictions with dead ball from half start up to first play."""
    pred = np.asarray(pred)
    half = np.asarray(half)
    if pred.ndim != 1 or half.shape != pred.shape or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("prediction and half arrays must be aligned one-dimensional arrays")
    starts = np.flatnonzero(half == 2)
    if not len(starts):
        raise ValueError("second half is missing")
    start = int(starts[0])
    if not start <= first_play_frame < len(pred) or np.any(half[start:first_play_frame] != 2):
        raise ValueError("kickoff must follow the second half start within that half")
    result = pred.copy()
    result[start:first_play_frame] = DEAD
    return result


def export_official_kickoff(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the checkout") from exc
    if relative.parts[:1] not in (("_local",), ("submissions",)) or output_path.suffix != ".csv":
        raise ValueError("output must be a local CSV")
    if output_path.exists():
        raise FileExistsError(output_path)
    config = json.loads((root / "_local/v11_export_inputs.json").read_text())
    if set(config) != {"synthesis", "variant"} or config["variant"] not in ("strict", "grand", "d3"):
        raise ValueError("configuration requires synthesis and an explicit variant")
    with _verified_file(root, config["synthesis"]).open("rb") as handle:
        bundle = pickle.load(handle)
    atd = load_atd()
    pred = np.asarray(bundle[config["variant"]]["pred_atd_rep"])
    # The organiser's FAQ places first play one frame after the numerical
    # games.py anchor. The latter describes the anchor as the strike frame,
    # so these published descriptions differ; this export follows the FAQ.
    first_play = anchor_video_frame(GAMES["BAR-ATH"]["kickoff_2nd"]) + 1
    pred = apply_second_half_kickoff(pred, atd.half, first_play)
    if len(pred) != len(atd.half) or not np.isin(pred, (0, 1, 2)).all():
        raise ValueError("invalid prediction length or class")
    if not np.all(pred[~atd.graded] == DEAD):
        raise ValueError("ungraded frames must be dead")
    frame = pd.DataFrame({"frame": np.arange(len(pred), dtype=np.int32),
                          "label": np.asarray(["H", "A", "D"])[pred]})
    reason = validate_submission(frame, len(atd.half))
    if reason:
        raise ValueError(f"submission validation failed: {reason}")
    with NamedTemporaryFile(mode="w", suffix=".csv", dir=output_path.parent) as temp:
        frame.to_csv(temp, index=False)
        temp.flush()
        reason = validate_submission(read_label_csv(temp.name), len(atd.half))
        if reason:
            raise ValueError(f"written CSV validation failed: {reason}")
        os.link(temp.name, output_path)
    return output_path
