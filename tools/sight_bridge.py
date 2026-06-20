#!/usr/bin/env python3
"""sight_bridge — streaming bbox bridge for Rust sightd.

Reads frames from camera, runs face detection (SCRFD GPU or OpenCV YuNet CPU),
outputs JSON lines to stdout for Rust sightd to consume.

Usage:
    venv/bin/python3 tools/sight_bridge.py              # GPU (SCRFD)
    venv/bin/python3 tools/sight_bridge.py --cpu        # CPU (OpenCV YuNet)
    venv/bin/python3 tools/sight_bridge.py --json-only  # no stderr, JSON only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="sightd camera bridge")
parser.add_argument("--cpu", action="store_true", help="Use OpenCV YuNet CPU detector (faster startup, no GPU)")
parser.add_argument("--json-only", action="store_true", help="Suppress stderr messages")
parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = forever)")
args = parser.parse_args()

if args.json_only:
    sys.stderr = open(os.devnull, "w")

# ── Paths ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"
sys.path.insert(0, str(ROOT))

# ── Imports (lazy, after path setup) ─────────────────────
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ── GPU preload ──────────────────────────────────────────
if not args.cpu:
    try:
        import ctypes  # noqa: E402
        py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        nvidia_base = VENV / "lib" / py_ver / "site-packages" / "nvidia"
        loaded = False
        for root_nv, _dirs, files in os.walk(str(nvidia_base)):
            for f in files:
                if f.endswith(".so") and ("cublas" in f or "cudnn" in f or "cuda_runtime" in f):
                    path = os.path.join(root_nv, f)
                    try:
                        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                        loaded = True
                    except OSError:
                        pass
        if not loaded:
            sys.stderr.write("  bridge: GPU preload failed, falling back to CPU\n")
            sys.stderr.flush()
            args.cpu = True
    except Exception:
        sys.stderr.write("  bridge: GPU unavailable, using CPU\n")
        sys.stderr.flush()
        args.cpu = True

# ── Detector ─────────────────────────────────────────────
if args.cpu:
    import cv2  # noqa: E402
    yunet_model = str(ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(yunet_model):
        sys.stderr.write(f"  bridge: YuNet model not found at {yunet_model}\n")
        sys.exit(1)
    detector = cv2.FaceDetectorYN.create(
        yunet_model, "", (640, 480), score_threshold=0.5, nms_threshold=0.3, top_k=10
    )
    sys.stderr.write("  bridge: YuNet CPU detector ready\n")
    sys.stderr.flush()
else:
    from core.face_id import FaceID  # noqa: E402
    face_id = FaceID()
    sys.stderr.write("  bridge: SCRFD GPU detector ready\n")
    sys.stderr.flush()

# ── Camera ───────────────────────────────────────────────
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    sys.stderr.write("  bridge: no camera found\n")
    sys.exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
sys.stderr.write("  bridge: camera ready, streaming...\n")
sys.stderr.flush()

# Warm-up: capture a few frames before starting
for _ in range(10):
    cam.read()

# ── Main loop ────────────────────────────────────────────
frame_id = 0
start_time = time.monotonic()

try:
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.1)
            continue

        frame_id += 1

        # Face detection
        bboxes = []
        if args.cpu:
            h, w = frame.shape[:2]
            detector.setInputSize((w, h))
            _ret, faces = detector.detect(frame)
            if faces is not None:
                for face in faces:
                    x, y, fw, fh = face[:4].astype(int)
                    bboxes.append({
                        "x1": x / w,
                        "y1": y / h,
                        "x2": (x + fw) / w,
                        "y2": (y + fh) / h,
                    })
        else:
            raw = face_id.app.get(frame)
            for face in raw:
                x1, y1, x2, y2 = face.bbox.astype(int)
                h, w = frame.shape[:2]
                bboxes.append({
                    "x1": x1 / w,
                    "y1": y1 / h,
                    "x2": x2 / w,
                    "y2": y2 / h,
                })

        elapsed = time.monotonic() - start_time
        msg = {
            "frame": frame_id,
            "elapsed": round(elapsed, 3),
            "detections": bboxes,
        }

        # Write JSON line to stdout
        sys.stdout.write(json.dumps(msg) + "\n")
        sys.stdout.flush()

        if args.max_frames and frame_id >= args.max_frames:
            break

except BrokenPipeError:
    pass  # Rust reader closed stdin
except KeyboardInterrupt:
    pass
finally:
    cam.release()
