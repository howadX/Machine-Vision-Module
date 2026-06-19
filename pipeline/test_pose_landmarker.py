#!/usr/bin/env python3
"""Этап 0: тест PoseLandmarker — извлечение 33 COCO landmarks, latency."""
import os, sys, time, glob, warnings, logging

os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
warnings.filterwarnings('ignore')

from contextlib import contextmanager

@contextmanager
def suppress_stderr():
    old = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try: yield
    finally:
        os.dup2(old, 2)
        os.close(devnull); os.close(old)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POSE_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'pose_landmarker.task')

# ── Камера ────────────────────────────────────────────────
def find_camera():
    for dev in sorted(glob.glob('/dev/video*'), key=lambda x: int(x.replace('/dev/video',''))):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean() > 10:
                    cap.release(); return dev
            cap.release()
    return None

# ── Главное ───────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(line_buffering=True)

    if not os.path.exists(POSE_MODEL):
        print(f"error: модель не найдена: {POSE_MODEL}")
        return

    cam = find_camera()
    if not cam:
        print("error: камера не найдена")
        return

    # Захват кадра
    cam_idx = int(cam.replace('/dev/video',''))
    cap = cv2.VideoCapture(cam, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    for _ in range(5): ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("error: не удалось захватить кадр")
        return

    print(f"camera: {cam}")
    print(f"frame_size: {frame.shape[1]}x{frame.shape[0]}")
    print(f"brightness: {frame.mean():.0f}")

    # PoseLandmarker
    with suppress_stderr():
        base_opts = mp_python.BaseOptions(
            model_asset_path=POSE_MODEL,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)

    # Инференс + замер latency
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        with suppress_stderr():
            result = landmarker.detect(mp_img)
        times.append(time.perf_counter() - t0)

    latency = np.mean(times) * 1000
    print(f"pose_landmarker_latency_ms: {latency:.1f}")
    print(f"pose_landmarker_latency_min_ms: {min(times)*1000:.1f}")
    print(f"pose_landmarker_latency_max_ms: {max(times)*1000:.1f}")

    if not result.pose_landmarks:
        print("pose_detected: no")
        landmarker.close()
        return

    print("pose_detected: yes")

    # Извлечение landmarks
    lm = result.pose_landmarks[0]
    n = len(lm)
    print(f"landmarks_count: {n}")

    # Координаты первых 5 + ключевых точек
    key_indices = {
        0: 'nose', 7: 'left_ear', 8: 'right_ear',
        11: 'left_shoulder', 12: 'right_shoulder',
        13: 'left_elbow', 14: 'right_elbow',
        15: 'left_wrist', 16: 'right_wrist',
        23: 'left_hip', 24: 'right_hip',
        25: 'left_knee', 26: 'right_knee',
        27: 'left_ankle', 28: 'right_ankle',
    }
    for idx, name in sorted(key_indices.items()):
        if idx < n:
            pt = lm[idx]
            print(f"  {name}: x={pt.x:.4f} y={pt.y:.4f} z={pt.z:.4f} v={pt.visibility:.2f}")

    # Быстрая эвристика позы
    hips_y = (lm[23].y + lm[24].y) / 2 if n > 24 else lm[0].y
    shoulders_y = (lm[11].y + lm[12].y) / 2 if n > 12 else lm[0].y
    nose_y = lm[0].y

    torso_ratio = abs(shoulders_y - hips_y)
    if nose_y > hips_y:
        orientation = "upright (head above hips)"
    elif nose_y < shoulders_y - 0.05:
        orientation = "lying (head below shoulders)"
    else:
        orientation = "sitting (head near hips)"

    print(f"orientation_heuristic: {orientation}")
    print(f"torso_ratio: {torso_ratio:.4f}")

    landmarker.close()
    print("\ndone: PoseLandmarker работает корректно")

if __name__ == '__main__':
    main()
