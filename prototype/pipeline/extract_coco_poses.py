#!/usr/bin/env python3
"""
extract_coco_poses.py — извлечение 33 landmarks из COCO train2017 через PoseLandmarker.

Вход:  data/datasets/coco/train2017/ (изображения)
       data/datasets/coco/annotations/person_keypoints_train2017.json
Выход: pipeline/coco_poses.csv  (формат как poses_dataset.csv: label, f0_x, f0_y, ...)
"""

import os, sys, csv, json, time, glob, warnings, logging
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
COCO_DIR = os.path.join(ROOT, 'data', 'datasets', 'coco', 'val2017')
ANNOTATIONS = os.path.join(ROOT, 'data', 'datasets', 'coco', 'annotations', 'person_keypoints_val2017.json')
OUTPUT_CSV = os.path.join(ROOT, 'pipeline', 'coco_poses.csv')

MAX_IMAGES = 6000  # max poses (все доступные)

def classify_pose_heuristic(landmarks):
    """
    Эвристика: standing/sitting/lying по 33 landmarks.
    Возвращает (label, confidence).
    0=standing, 1=sitting, 2=lying
    """
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks], dtype=np.float32)

    nose_y = pts[0, 1]
    l_shoulder = pts[11]; r_shoulder = pts[12]
    l_hip = pts[23]; r_hip = pts[24]
    l_knee = pts[25]; r_knee = pts[26]
    l_ankle = pts[27]; r_ankle = pts[28]

    shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    knee_y = (l_knee[1] + r_knee[1]) / 2 if l_knee[3] > 0.3 and r_knee[3] > 0.3 else hip_y
    ankle_y = (l_ankle[1] + r_ankle[1]) / 2 if l_ankle[3] > 0.3 and r_ankle[3] > 0.3 else knee_y

    hip_vis = (l_hip[3] + r_hip[3]) / 2
    knee_vis = (l_knee[3] + r_knee[3]) / 2

    # Torso length
    shoulder_center = np.array([(l_shoulder[0]+r_shoulder[0])/2, shoulder_y])
    hip_center = np.array([(l_hip[0]+r_hip[0])/2, hip_y])
    torso_len = np.linalg.norm(shoulder_center - hip_center) or 0.01

    # Head-to-hip distance
    nose_pos = np.array([pts[0,0], nose_y])
    head_hip_dist = abs(nose_y - hip_y)

    # Если бёдра не видны — fallback
    if hip_vis < 0.05:
        return 0, 0.30  # standing (слабая уверенность)

    # Лежит: голова на уровне или ниже плеч, бёдра на уровне плеч
    if head_hip_dist < torso_len * 0.6 and hip_vis > 0.1:
        return 2, 0.80  # lying

    # Сидит: бёдра ниже плеч, колени примерно на уровне бёдер
    if knee_vis > 0.1 and abs(knee_y - hip_y) < torso_len * 1.5:
        return 1, 0.70  # sitting

    # Стоит: бёдра значительно ниже плеч
    if hip_y > shoulder_y + torso_len * 0.3:
        return 0, 0.75  # standing

    # Fallback: standing
    return 0, 0.45


def main():
    print("=" * 60)
    print("  Извлечение COCO поз через PoseLandmarker")
    print("=" * 60)

    if not os.path.exists(POSE_MODEL):
        print(f"[ERROR] Модель не найдена: {POSE_MODEL}")
        return

    # Загружаем аннотации — получаем список image_id с людьми
    print(f"  Загрузка аннотаций: {ANNOTATIONS}")
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    # image_id → file_name
    id_to_file = {img['id']: img['file_name'] for img in coco['images']}

    # Аннотации людей: image_id → список keypoints
    image_people = {}
    for ann in coco['annotations']:
        if ann['num_keypoints'] < 5:  # минимум 5 видимых точек
            continue
        img_id = ann['image_id']
        if img_id not in image_people:
            image_people[img_id] = []
        image_people[img_id].append(ann)

    print(f"  Изображений с людьми: {len(image_people)}")
    print(f"  Ограничение: {MAX_IMAGES}")

    # PoseLandmarker
    with suppress_stderr():
        base_opts = mp_python.BaseOptions(
            model_asset_path=POSE_MODEL,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        opts = vision.PoseLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,
            output_segmentation_masks=False,
        )
        landmarker = vision.PoseLandmarker.create_from_options(opts)

    # Собираем данные
    samples = []
    processed = 0
    skipped_no_pose = 0
    skipped_no_label = 0
    label_counts = {0: 0, 1: 0, 2: 0}
    max_per_class = MAX_IMAGES // 3  # балансируем классы

    t_start = time.time()

    for img_id, anns in image_people.items():
        if processed >= MAX_IMAGES:
            break

        file_name = id_to_file.get(img_id)
        if not file_name:
            continue

        img_path = os.path.join(COCO_DIR, file_name)
        if not os.path.exists(img_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        with suppress_stderr():
            result = landmarker.detect(mp_img)

        if not result.pose_landmarks:
            skipped_no_pose += 1
            processed += 1
            continue

        for lm in result.pose_landmarks:
            if processed >= MAX_IMAGES:
                break

            label, conf = classify_pose_heuristic(lm)
            # Балансировка: пропускаем если класс уже набран
            if label_counts[label] >= max_per_class:
                continue

            # Сохраняем в наш формат
            row = [label]
            for pt in lm:
                row.extend([f"{pt.x:.6f}", f"{pt.y:.6f}", f"{pt.z:.6f}", f"{pt.visibility:.4f}"])

            samples.append(row)
            label_counts[label] += 1
            processed += 1

        if processed % 500 == 0:
            elapsed = time.time() - t_start
            print(f"  Обработано: {processed} | стоя:{label_counts[0]} сидя:{label_counts[1]} лежа:{label_counts[2]} | {elapsed:.0f}s")

    landmarker.close()
    elapsed = time.time() - t_start

    # Сохраняем CSV
    header = ['label']
    for i in range(33):
        header.extend([f'f{i}_x', f'f{i}_y', f'f{i}_z', f'f{i}_v'])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(samples)

    print(f"\n{'='*60}")
    print(f"  ГОТОВО за {elapsed:.0f}с")
    print(f"  Всего: {len(samples)} примеров")
    print(f"    standing:  {label_counts[0]}")
    print(f"    sitting:   {label_counts[1]}")
    print(f"    lying:     {label_counts[2]}")
    print(f"  Пропущено: no_pose={skipped_no_pose}, no_label={skipped_no_label}")
    print(f"  Сохранено: {OUTPUT_CSV}")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
