"""Spike 7.5.0b — capture real ceiling-cam frames with a full-body person and
re-check the OpenCV Zoo person detector under real deployment conditions.

Mirrors the bridge camera (CAP_FFMPEG + ROTATE_180). Captures a burst over a
window (stand in full-body view!), saves each frame, and runs the person
detector on each, printing persons + score.

Usage (from spike_pose/, project venv):
    ../venv/bin/python3 ceiling_capture_probe.py [seconds] [interval]
"""
import os, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mp_persondet import MPPersonDet

WINDOW = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0   # capture window (s)
INTERVAL = float(sys.argv[2]) if len(sys.argv) > 2 else 1.2  # between frames (s)
OUT = os.path.join(HERE, "ceiling_test")
os.makedirs(OUT, exist_ok=True)

det = MPPersonDet(modelPath=f"{ROOT}/models/person_detection_mediapipe_2023mar.onnx",
                  nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)

cap = cv2.VideoCapture('/dev/video0', cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    sys.stderr.write("ERROR: cannot open /dev/video0 (is the bridge or another app holding it?)\n")
    sys.exit(2)
for _ in range(5):
    cap.read()  # warm-up

sys.stderr.write(f">>> STAND IN FULL-BODY VIEW. Capturing {WINDOW:.0f}s ...\n")
sys.stderr.flush()

t0 = time.time()
n = 0
best = (-1.0, None)
while time.time() - t0 < WINDOW:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.05); continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)   # camera mounted upside-down
    n += 1
    path = os.path.join(OUT, f"frame_{n:02d}.jpg")
    cv2.imwrite(path, frame)
    res = det.infer(frame)
    if res is None or len(res) == 0:
        print(f"frame_{n:02d} {frame.shape[1]}x{frame.shape[0]}: persons=0")
    else:
        scores = sorted((float(p[-1]) for p in res), reverse=True)
        bb = res[0]
        print(f"frame_{n:02d} {frame.shape[1]}x{frame.shape[0]}: persons={len(res)} "
              f"scores={[round(s,2) for s in scores]} bbox0={np.round(bb[:4],0).tolist()}")
        if scores[0] > best[0]:
            best = (scores[0], path)
    time.sleep(INTERVAL)

cap.release()
print(f"\nCaptured {n} frames in {OUT}/")
if best[1]:
    print(f"BEST detection: score={best[0]:.2f} at {best[1]}")
    print("=> person detector WORKS at ceiling angle on real frames ✓")
else:
    print("=> person detector found NOBODY in any frame ✗  "
          "(confirms ceiling-angle limitation; use YuNet-bbox ROI in 7.5.2)")
