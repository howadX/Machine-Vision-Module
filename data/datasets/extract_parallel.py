#!/usr/bin/env python3
"""Параллельное извлечение геометрических признаков из AffectNet (8 процессов)."""
import os, sys, csv, time
import cv2
import numpy as np
from multiprocessing import Pool

ROOT = '/home/dawoh/repos/Video Recognition Module'
DATA_DIR = os.path.join(ROOT, 'data/datasets/affectnet/Train')
OUTPUT_CSV = os.path.join(ROOT, 'models/descriptors/geometric_affectnet.csv')
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def process_emotion(args):
    """Обрабатывает одну эмоцию в отдельном процессе."""
    emotion_name, label = args
    
    # Каждый процесс создаёт свой FaceLandmarker
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    from mediapipe import Image, ImageFormat
    
    model_path = os.path.join(ROOT, 'models', 'mediapipe', 'face_landmarker.task')
    base = mp_python.BaseOptions(model_asset_path=model_path)
    opts = vision.FaceLandmarkerOptions(base_options=base, num_faces=1, 
                                         running_mode=vision.RunningMode.IMAGE)
    landmarker = vision.FaceLandmarker.create_from_options(opts)
    
    folder = os.path.join(DATA_DIR, emotion_name)
    if not os.path.isdir(folder):
        landmarker.close()
        return emotion_name, 0, 0, []
    
    image_files = sorted(f for f in os.listdir(folder) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png')))
    
    rows = []
    skipped = 0
    
    for fname in image_files:
        img_path = os.path.join(folder, fname)
        bgr = cv2.imread(img_path)
        if bgr is None:
            skipped += 1
            continue
        
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_img)
        
        if not result.face_landmarks:
            skipped += 1
            continue
        
        landmarks = result.face_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
        
        nose_tip = pts[4]
        left_outer = pts[33]
        right_outer = pts[263]
        interocular = np.linalg.norm(left_outer - right_outer)
        
        if interocular < 1e-8:
            skipped += 1
            continue
        
        diffs = pts - nose_tip
        distances = np.linalg.norm(diffs, axis=1)
        normalized = distances / interocular
        
        rows.append([label] + normalized.tolist())
    
    landmarker.close()
    return emotion_name, len(image_files), skipped, rows


def main():
    print(f"Параллельное извлечение: {len(EMOTIONS)} процессов")
    print(f"Источник: {DATA_DIR}")
    start = time.time()
    
    tasks = [(name, idx) for idx, name in enumerate(EMOTIONS)]
    
    with Pool(processes=len(EMOTIONS)) as pool:
        results = pool.map(process_emotion, tasks)
    
    # Сбор результатов
    all_rows = []
    total_processed = 0
    total_skipped = 0
    
    for name, total, skipped, rows in results:
        processed = len(rows)
        print(f"  {name:12s}: {processed}/{total} (пропущено: {skipped})")
        all_rows.extend(rows)
        total_processed += processed
        total_skipped += skipped
    
    # Запись CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    columns = ["emotion_label"] + [f"f{i}" for i in range(468)]
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(all_rows)
    
    elapsed = time.time() - start
    print(f"\nГотово. Обработано: {total_processed}, пропущено: {total_skipped}")
    print(f"Время: {elapsed:.0f} сек ({elapsed/60:.1f} мин)")
    print(f"Сохранено: {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
