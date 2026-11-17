"""F4.9 probe: off-the-shelf ball detector (YOLO11, COCO "sports ball") on the broadcast video vs the ATD ball.

    .venv_video/bin/python scripts/s4_9_video_ball_probe.py      # separate env: ultralytics + opencv (not the build env)

900 frames sampled every 5th frame from four long valid broadcast stretches. The detection (box bottom
projected to the pitch with the inverse broadcast homography) is compared with the broadcast ATD ball;
the label-free arbiter is whether a tracked player is within 1.5 m of each ball. Frames are processed
locally and never shown to anyone.
"""
import numpy as np, pandas as pd, cv2
from ultralytics import YOLO
H = pd.read_csv("data/broadcast/BAR-ATH_broadcast_homography.csv")
raw = np.load("local_data/cache/broadcast_raw.npz")["raw"]; S = raw.shape[1] - 1
M = np.array([105.3, 68.0])
ball = raw[:, S, :2] * M; P = raw[:, :S, :2] * M; vis = np.isfinite(ball).all(1)
segs = [(9024, 11981), (15432, 19865), (20172, 22705), (24000, 26000)]
idx = np.concatenate([np.arange(a, b, 5) for a, b in segs])[:900]
cap = cv2.VideoCapture("data/broadcast/BAR-ATH_broadcast.mp4")
model = YOLO("local_data/models/yolo/yolo11m.pt"); rows = []
def nearest(p, k):
    d = np.linalg.norm(P[k] - p, axis=1); d = d[np.isfinite(d)]; return d.min() if len(d) else np.nan
for k in idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(k)); f = cap.read()[1]
    r = model.predict(f, imgsz=1280, classes=[32], conf=0.05, device="mps", verbose=False)[0]
    det = np.nan * np.ones(2); conf = 0.0
    if len(r.boxes):
        c = r.boxes.conf.cpu().numpy(); b = r.boxes.xywh.cpu().numpy()[np.argmax(c)]; conf = float(c.max())
        Hm = H.iloc[k, 1:10].to_numpy(float).reshape(3, 3)
        q = np.linalg.inv(Hm) @ np.array([b[0], b[1] + b[3] / 2, 1.0]); det = q[:2] / q[2] * M
    rows.append({"atd_vis": vis[k], "conf": conf, "near_atd": nearest(ball[k], k) if vis[k] else np.nan,
                 "near_det": nearest(det, k) if np.isfinite(det).all() else np.nan,
                 "gap": np.linalg.norm(det - ball[k]) if vis[k] and np.isfinite(det).all() else np.nan})
df = pd.DataFrame(rows)
both = df.dropna(subset=["near_atd", "near_det"])
print("frames %d, detector found %.2f, ATD visible %.2f" % (len(df), np.mean(df.conf > 0), df.atd_vis.mean()))
for c in (0.05, 0.3, 0.5):
    s = both[both.conf >= c]
    print("conf>=%.2f n=%d | nearest player to ball: ATD median %.2f m (<1.5m %.2f) | detector median %.2f m (<1.5m %.2f) | ATD-det gap median %.2f" % (
        c, len(s), s.near_atd.median(), np.mean(s.near_atd < 1.5), s.near_det.median(), np.mean(s.near_det < 1.5), s.gap.median()))
dis = both[both.gap > 3]
print("when they disagree by >3 m (n=%d): player <1.5 m from ATD ball %.2f vs from detector ball %.2f" % (len(dis), np.mean(dis.near_atd < 1.5), np.mean(dis.near_det < 1.5)))
only = df[~df.atd_vis & (df.conf > 0.3)]
print("ATD missing but detector conf>0.3: %d frames; detector ball has a player <1.5 m in %.2f" % (len(only), np.mean(only.near_det < 1.5)))
