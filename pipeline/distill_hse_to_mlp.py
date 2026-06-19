#!/usr/bin/env python3
"""distill_hse_to_mlp.py — Дистилляция HSEmotion → MLP.

Прогоняет все фото из merged/ через FaceLandmarker (468 признаков)
и HSEmotion (8 софт-меток → 7 после удаления Contempt).

Выход: models/emotion/geometric_distilled_train.csv
        models/emotion/geometric_distilled_test.csv

Формат CSV: f0,f1,...,f467,anger,disgust,fear,happy,neutral,sad,surprise
"""
import os, sys, csv, time, warnings, logging
from pathlib import Path

os.environ.setdefault('ONNXRUNTIME_LOG_LEVEL', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
warnings.filterwarnings('ignore')
for name in ('insightface', 'onnxruntime', 'absl'):
    logging.getLogger(name).setLevel(logging.ERROR)

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from emotion_hse import HSEmotionClassifier

# ── Конфигурация ──────────────────────────────────────────
FACE_LM_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'face_landmarker.task')
MERGED_DIR = os.path.join(ROOT, 'data', 'datasets', 'merged')
MERGED_TEST_DIR = os.path.join(ROOT, 'data', 'datasets', 'merged_test')
OUT_DIR = os.path.join(ROOT, 'models', 'emotion')
OUT_TRAIN = os.path.join(OUT_DIR, 'geometric_distilled_train.csv')
OUT_TEST = os.path.join(OUT_DIR, 'geometric_distilled_test.csv')

# Порядок классов HSEmotion
HSE_CLASSES = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
# Наши 7 классов (без Contempt)
OUR_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
# Индексы HSEmotion для наших классов
HSE_TO_OUR_IDX = [0, 2, 3, 4, 5, 6, 7]  # пропускаем Contempt (idx=1)


def compute_features_468(face_landmarks):
    """468 нормированных расстояний от кончика носа."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
    nose_tip = pts[4]; left_eye = pts[33]; right_eye = pts[263]
    interocular = np.linalg.norm(left_eye - right_eye)
    if interocular < 1e-6: interocular = 1.0
    return (np.linalg.norm(pts - nose_tip, axis=1) / interocular)[:468]


def process_dataset(image_dir, face_lm, hse_classifier, csv_path):
    """Обрабатывает все изображения в директории, пишет CSV."""
    images = list(Path(image_dir).rglob('*.jpg'))
    total = len(images)
    print(f"  Изображений: {total}")
    
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    processed = 0
    skipped = 0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    t_start = time.perf_counter()
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Заголовок: признаки + hard_label + 7 софт-меток
        header = [f'f{i}' for i in range(468)] + ['emotion_label'] + OUR_CLASSES
        writer.writerow(header)
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                skipped += 1
                continue
            
            # FaceLandmarker → 468 признаков
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            result = face_lm.detect(mp_img)
            
            if not result.face_landmarks:
                skipped += 1
                continue
            
            lm = result.face_landmarks[0]
            feats = compute_features_468(lm)
            
            # HSEmotion → 8 софт-меток → 7 (без Contempt)
            try:
                _, scores = hse_classifier.recognizer.predict_emotions(
                    img[:, :, ::-1], logits=False  # BGR → RGB
                )
                scores_7 = np.array([scores[i] for i in HSE_TO_OUR_IDX], dtype=np.float32)
                scores_7 /= scores_7.sum()  # ренормализация
                
                # Hard label из директории
                hard_label = img_path.parent.name  # e.g. "anger", "happy"
                
                row = list(feats) + [hard_label] + list(scores_7)
                writer.writerow(row)
                processed += 1
            except Exception:
                skipped += 1
                continue
            
            if processed % 500 == 0:
                elapsed = time.perf_counter() - t_start
                rate = processed / elapsed
                eta = (total - processed - skipped) / rate if rate > 0 else 0
                print(f"    {processed}/{total} ({processed/total*100:.0f}%) "
                      f"rate={rate:.0f} img/s ETA={eta:.0f}s")
    
    elapsed = time.perf_counter() - t_start
    print(f"  Готово: {processed} обработано, {skipped} пропущено за {elapsed:.0f}с "
          f"({processed/elapsed:.0f} img/s)")
    return processed, skipped


def main():
    sys.stdout.reconfigure(line_buffering=True)
    print("=" * 60)
    print("  HSEmotion → MLP Distillation Data")
    print("=" * 60)
    
    # ── Инициализация ──
    print("Загрузка FaceLandmarker...")
    face_opts = vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=FACE_LM_MODEL,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        ),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1, running_mode=vision.RunningMode.IMAGE,
    )
    face_lm = vision.FaceLandmarker.create_from_options(face_opts)
    
    print("Загрузка HSEmotion...")
    hse = HSEmotionClassifier(model_name='enet_b0_8_best_vgaf', use_gpu=True)
    print(f"  HSEmotion: {len(hse.class_names)} классов, {hse.device}")
    
    # Прогрев
    dummy = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    _ = hse.predict(dummy)
    dummy_rgb = cv2.cvtColor(dummy, cv2.COLOR_BGR2RGB)
    _ = face_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy_rgb))
    print()
    
    # ── TRAIN ──
    print("── TRAIN ──")
    process_dataset(MERGED_DIR, face_lm, hse, OUT_TRAIN)
    
    # ── TEST ──
    print("\n── TEST ──")
    process_dataset(MERGED_TEST_DIR, face_lm, hse, OUT_TEST)
    
    print(f"\nФайлы сохранены:")
    for p in [OUT_TRAIN, OUT_TEST]:
        size_mb = os.path.getsize(p) / 1024 / 1024 if os.path.exists(p) else 0
        print(f"  {p} ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
