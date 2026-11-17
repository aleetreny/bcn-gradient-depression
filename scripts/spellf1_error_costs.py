"""How much does each kind of error cost on spellF1?

Starts from a perfect prediction on the PUBLIC 5-minute ground truth and injects one error type at
a time (global lag, late dead-ball starts, merged short spells, spurious flips, D used as a
separator). Uses only the official scorer and the public GT shipped in this repo -- no gated data.

The 5-minute window holds 17 possessions, so absolute numbers are noisy; read the directions.

    python scripts/spellf1_error_costs.py
"""

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25


def main():
    gt = pm.label_codes(pd.read_csv(GT_PATH).label.values)
    runs = pm.valid_runs(gt)
    H, A, D = (pm.label_codes(np.array([c]))[0] for c in "HAD")
    segs = pm.segments(gt, runs)
    longs = [(a, b, k) for a, b, k in segs if k in (H, A) and b - a > 300]
    shorts = [(a, b, k) for a, b, k in segs if k in (H, A) and b - a < 2 * FPS]

    def score(pred):
        return round(pm.spell_scores(gt, pm.fill_abstentions(pred, runs), runs)["mean_f1"], 4)

    def merged(drop):
        """Overwrite each spell in `drop` with the previous segment's label."""
        pred, prev = gt.copy(), None
        for seg in segs:
            if seg in drop and prev is not None:
                pred[seg[0]:seg[1]] = prev
            else:
                prev = seg[2]
        return pred

    print("perfect", score(gt.copy()))

    for frames in (5, 12, 25, 50):
        pred = gt.copy()
        start, stop = runs[0]
        pred[start + frames:stop] = gt[start:stop - frames]
        print(f"everything {frames} frames late", score(pred))

    for frames in (25, 50):
        pred = gt.copy()
        for a, b, k in segs:
            if k == D and a > runs[0][0]:
                pred[a:a + frames] = pred[a - 1]
        print(f"dead ball starts {frames} frames late", score(pred))

    print("never predict D", score(merged({s for s in segs if s[2] == D})))

    for seconds in (1, 2, 3, 4):
        drop = {s for s in segs if s[2] in (H, A) and s[1] - s[0] < seconds * FPS}
        print(f"merge spells < {seconds}s", score(merged(drop)))

    pred = gt.copy()
    for a, b, k in shorts:
        pred[a:b] = D
    print("real short turnovers written as D", score(pred))
    print("real short turnovers missed (merged)", score(merged(set(shorts))))

    for kind in ("opposite team", "D"):
        for n in (1, 3):
            pred = gt.copy()
            for a, b, k in longs[:n]:
                mid = (a + b) // 2
                pred[mid:mid + FPS] = (A if k == H else H) if kind == "opposite team" else D
            print(f"{n} false 1s blip(s) written as {kind}", score(pred))


if __name__ == "__main__":
    main()
