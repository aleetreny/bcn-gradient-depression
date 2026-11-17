"""Step 0a of D3 video input consistency check: detect people and balls on the BAR-ATH tactical video.

    .venv311/bin/python scripts/noche/video_detect.py [--stride 2] [--model local_data/models/yolo/yolo11m.pt]

Runs a COCO YOLO (person, sports ball) on every `stride`-th graded frame of the tactical video (row t of
the submission = frame t of this video), at full resolution on MPS/CUDA/CPU. For every box it stores the frame,
class, confidence and pixel box, plus colour statistics of the torso (for the team colour consistency check, so the
video never has to be read again). Writes `local_data/video/detections_<model>_s<stride>.npz` in chunks
(resumable). Reads the provided video only; output stays in `local_data/` (gitignored, never committed).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Official layout (DATA.md) is data/tactical/; data/broadcast/ is the older local layout. Same file either way.
VIDEO = next((p for p in (ROOT / "data" / "tactical" / "BAR-ATH_tactical.mp4",
                          ROOT / "data" / "broadcast" / "BAR-ATH_tactical.mp4") if p.exists()),
             ROOT / "data" / "tactical" / "BAR-ATH_tactical.mp4")
HALVES = [(11185, 90651), (110808, 190197)]
OUT = ROOT / "local_data" / "video"


def torso_stats(img, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    h = y2 - y1
    y_a, y_b = y1 + int(0.15 * h), y1 + int(0.5 * h)
    crop = img[max(y_a, 0):max(y_b, y_a + 1), max(x1, 0):max(x2, x1 + 1)]
    if crop.size == 0:
        return np.full(14, np.nan, np.float32)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    bgr = crop.reshape(-1, 3).astype(np.float32)
    sat = hsv[:, 1] > 60
    hist = np.histogram(hsv[sat, 0], bins=8, range=(0, 180))[0].astype(np.float32)
    hist = hist / max(hist.sum(), 1.0)
    return np.r_[bgr.mean(0), hsv.mean(0), hist].astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--model", default=str(ROOT / "local_data/models/yolo/yolo11m.pt"))
    ap.add_argument("--video", default=str(VIDEO))
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.model)
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "{}_s{}".format(Path(a.model).stem, a.stride)
    out_file = OUT / "detections_{}.npz".format(tag)
    if out_file.is_file():
        print("Found cached detections:", out_file)
        return 0
    part_dir = OUT / ("parts_" + tag)
    part_dir.mkdir(exist_ok=True)
    frames = np.concatenate([np.arange(s, e + 1, a.stride) for s, e in HALVES])
    chunk = 2000
    cap = cv2.VideoCapture(str(a.video))
    t0 = time.time()
    for c0 in range(0, len(frames), chunk):
        part = part_dir / "{:06d}.npz".format(c0)
        if part.is_file():
            continue
        fr = frames[c0:c0 + chunk]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fr[0]))
        pos = int(fr[0])
        rows, feats = [], []
        batch_img, batch_t = [], []

        def flush():
            res = model.predict(batch_img, imgsz=1920, classes=[0, 32], conf=0.05, device="mps", verbose=False)
            for img, t, r in zip(batch_img, batch_t, res):
                b = r.boxes
                if len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy()
                cls = b.cls.cpu().numpy()
                conf = b.conf.cpu().numpy()
                for k in range(len(cls)):
                    rows.append((t, cls[k], conf[k], *xyxy[k]))
                    feats.append(torso_stats(img, xyxy[k]) if cls[k] == 0 else np.full(14, np.nan, np.float32))

        for t in fr:
            while pos < t:
                cap.grab()
                pos += 1
            ok, img = cap.read()
            pos += 1
            if not ok:
                continue
            batch_img.append(img)
            batch_t.append(int(t))
            if len(batch_img) == a.batch:
                flush()
                batch_img, batch_t = [], []
        if batch_img:
            flush()
        np.savez(part, det=np.array(rows, np.float32).reshape(-1, 7), feat=np.array(feats, np.float32).reshape(-1, 14))
        done = c0 + len(fr)
        el = time.time() - t0
        print("{}/{} frames, {:.0f} s, {} boxes in chunk".format(done, len(frames), el, len(rows)), flush=True)
    parts = sorted(part_dir.glob("*.npz"))
    det = np.concatenate([np.load(p)["det"] for p in parts])
    feat = np.concatenate([np.load(p)["feat"] for p in parts])
    np.savez(out_file, det=det, feat=feat,
             columns=np.array(["frame", "cls", "conf", "x1", "y1", "x2", "y2"]))
    print("done", det.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
