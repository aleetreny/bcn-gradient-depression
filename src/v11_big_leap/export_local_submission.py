"""Export an algorithm-generated v11 prediction as a validated local CSV.

The input remains a local prediction bundle. This does not establish model
provenance, rule eligibility, clean-clone reproduction or a final submission.
"""
import json
import os
from pathlib import Path
import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd

from evaluation.possession_metrics import read_label_csv, validate_submission
from src.data.metrica_atd import load as load_atd
from src.v10_first_principles.rebuild_trajectory_inference import _verified_file


def export_local_submission(root: Path, output_path: Path) -> Path:
    root = Path(root).resolve()
    if root != Path(__file__).resolve().parents[2]:
        raise ValueError("root must be the checkout containing this module")
    output_path = Path(output_path).resolve()
    try:
        relative = output_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("output must be inside the project root") from exc
    if relative.parts[:1] not in (("_local",), ("submissions",)):
        raise ValueError("output must be under _local/ or submissions/")
    if output_path.suffix != ".csv":
        raise ValueError("output must be a CSV file")
    if output_path.exists():
        raise FileExistsError(output_path)
    config = json.loads((root / "_local/v11_export_inputs.json").read_text())
    if set(config) != {"synthesis", "variant"} or config["variant"] not in ("strict", "grand", "d3"):
        raise ValueError("configuration requires synthesis and an explicit variant")
    with _verified_file(root, config["synthesis"]).open("rb") as handle:
        bundle = pickle.load(handle)
    pred = np.asarray(bundle[config["variant"]]["pred_atd_rep"])
    atd = load_atd()
    if pred.ndim != 1 or len(pred) != len(atd.half) or not np.issubdtype(pred.dtype, np.integer):
        raise ValueError("prediction shape or dtype is invalid")
    if not np.isin(pred, (0, 1, 2)).all():
        raise ValueError("prediction contains an invalid class")
    if not np.all(pred[~atd.graded] == 2):
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
