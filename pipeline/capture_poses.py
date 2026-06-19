#!/usr/bin/env python3
"""
capture_poses.py — захват датасета поз для обучения VAE.

Запуск: venv/bin/python3 pipeline/capture_poses.py

Управление:
  0-9 — сохранить текущую позу с меткой 0-9
  q   — выход
  s   — показать статистику собранных поз

Выход: pipeline/poses_dataset.csv
Формат: label, f0_x, f0_y, f0_z, f0_v, f1_x, ..., f32_v
"""

import os, sys, csv, time, glob, warnings, logging
from contextlib import contextmanager

os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
warnings.filterwarnings('ignore')

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
OUTPUT_CSV = os.path.join(ROOT, 'pipeline', 'poses_dataset.csv')

# ── Камера ────────────────────────────────────────────────
def open_camera():
    for dev in sorted(glob.glob('/dev/video*'), key=lambda x: int(x.replace('/dev/video',''))):
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ret, frame = cap.read()
            if ret and frame is not None and frame.mean() > 10:
                return cap, dev
            cap.release()
    return None, None

# ── Отрисовка landmarks ───────────────────────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),  # face
    (0, 7), (7, 8),  # ears
    (9, 10),  # mouth
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),  # arms
    (15, 17), (15, 19), (15, 21), (16, 18), (16, 20), (16, 22),  # hands
    (11, 23), (12, 24), (23, 24),  # torso
    (23, 25), (25, 27), (24, 26), (26, 28),  # legs
    (27, 29), (29, 31), (28, 30), (30, 32),  # feet
]

def draw_landmarks(frame, landmarks, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = {}
    for i, lm in enumerate(landmarks):
        px, py = int(lm.x * w), int(lm.y * h)
        pts[i] = (px, py)
        cv2.circle(frame, (px, py), 3, color, -1)

    for a, b in POSE_CONNECTIONS:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], color, 1)
    return frame

# ── Главное ───────────────────────────────────────────────
def main():
    LABELS = {
        ord(str(i)): i for i in range(10)
    }
    LABEL_NAMES = {
        0: 'standing',
        1: 'sitting',
        2: 'lying_back',
        3: 'lying_side',
        4: 'arms_crossed',
        5: 'sitting_at_desk',
        6: 'custom_6',
        7: 'custom_7',
        8: 'custom_8',
        9: 'custom_9',
    }

    print("=" * 50)
    print("  Захват датасета поз для VAE")
    print("=" * 50)
    print(f"  Выход: {OUTPUT_CSV}")
    print()
    print("  Управление:")
    for key_int, name in sorted(LABEL_NAMES.items()):
        print(f"    {key_int} — {name}")
    print("    s — статистика")
    print("    q — выход")
    print()

    if not os.path.exists(POSE_MODEL):
        print(f"[ERROR] Модель не найдена: {POSE_MODEL}")
        return

    cap, cam_path = open_camera()
    if cap is None:
        print("[ERROR] Камера не найдена")
        return
    print(f"  Камера: {cam_path}")

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

    # Загрузка существующего датасета
    samples = []
    if os.path.exists(OUTPUT_CSV):
        with open(OUTPUT_CSV, newline='') as f:
            reader = csv.reader(f)
            next(reader)  # header
            for row in reader:
                samples.append(row)
        print(f"  Загружено примеров: {len(samples)}")

    print("\n  Захват запущен. Нажимайте 0-9 для сохранения позы.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with suppress_stderr():
            result = landmarker.detect(mp_img)

        display = frame.copy()

        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            display = draw_landmarks(display, lm)

            # HUD: подсказки на экране
            cv2.putText(display, f"Samples: {len(samples)}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Список меток справа
            y = 60
            for k, name in sorted(LABEL_NAMES.items()):
                cv2.putText(display, f"{k} = {name}", (10, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 100), 1)
                y += 20
            cv2.putText(display, "q = выход  s = стат", (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        else:
            cv2.putText(display, "No pose detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow('Pose Capture', display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            # Статистика
            label_counts = {}
            for row in samples:
                lbl = int(row[0])
                name = LABEL_NAMES.get(lbl, f'class_{lbl}')
                label_counts[name] = label_counts.get(name, 0) + 1
            print(f"\n  Статистика ({len(samples)} всего):")
            for name, cnt in sorted(label_counts.items()):
                print(f"    {name}: {cnt}")
            print()
        elif key in LABELS:
            if not result.pose_landmarks:
                print("  Нет позы — не сохранено")
                continue

            label = LABELS[key]
            name = LABEL_NAMES.get(label, f'class_{label}')

            lm = result.pose_landmarks[0]
            row = [label]
            for pt in lm:
                row.extend([f"{pt.x:.6f}", f"{pt.y:.6f}", f"{pt.z:.6f}", f"{pt.visibility:.4f}"])

            samples.append(row)

            # Сохраняем CSV
            header = ['label']
            for i in range(33):
                header.extend([f'f{i}_x', f'f{i}_y', f'f{i}_z', f'f{i}_v'])

            with open(OUTPUT_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(samples)

            # Вспышка: на 3 кадра показываем зелёный фон
            flash_count = 3
            print(f"  [+] {name} — всего: {len(samples)}")

    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()

    # Финальная статистика
    label_counts = {}
    for row in samples:
        lbl = int(row[0])
        name = LABEL_NAMES.get(lbl, f'class_{lbl}')
        label_counts[name] = label_counts.get(name, 0) + 1

    print(f"\n{'='*50}")
    print(f"  ГОТОВО. Собрано: {len(samples)} примеров")
    for name, cnt in sorted(label_counts.items()):
        print(f"    {name}: {cnt}")
    print(f"  Сохранено в: {OUTPUT_CSV}")
    print(f"{'='*50}")

if __name__ == '__main__':
    main()
