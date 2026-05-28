"""
2_extract_geometric.py — извлечение 468 геометрических признаков из MediaPipe FaceLandmarker.

Вход:  data/my_emotions/{angry,disgust,fear,happy,sad,surprise,neutral}/*.jpg
Выход: models/descriptors/geometric_468.csv
Колонки: emotion_label, f0 ... f467

Для каждого изображения:
  1) MediaPipe FaceLandmarker извлекает 468 landmarks.
  2) Вычисляются евклидовы расстояния от кончика носа (landmark 4) до каждой точки.
  3) Нормировка на межзрачковое расстояние (landmarks 33 и 263 — внешние углы глаз).
"""

import csv
import os as _os
import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

# ---------------------------------------------------------------------------
# Пути относительно корня проекта
# ---------------------------------------------------------------------------
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_os.path.dirname(_SCRIPT_DIR))

DATA_DIR = _os.path.join(_ROOT, "data", "my_emotions")
MODEL_PATH = _os.path.join(_ROOT, "models", "mediapipe", "face_landmarker.task")
OUTPUT_DIR = _os.path.join(_ROOT, "models", "descriptors")
OUTPUT_CSV = _os.path.join(OUTPUT_DIR, "geometric_468.csv")

# Порядок эмоций → числовая метка
EMOTIONS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
EMOTION_LABEL = {name: idx for idx, name in enumerate(EMOTIONS)}

# Индексы ключевых landmarks
NOSE_TIP_IDX = 4       # кончик носа
LEFT_EYE_OUTER = 33    # внешний угол левого глаза (на изображении — правый)
RIGHT_EYE_OUTER = 263  # внешний угол правого глаза (на изображении — левый)
NUM_LANDMARKS = 468


def create_landmarker(model_path: str) -> vision.FaceLandmarker:
    """Создаёт и возвращает экземпляр FaceLandmarker."""
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=1,
        running_mode=vision.RunningMode.IMAGE,
    )
    return vision.FaceLandmarker.create_from_options(options)


def extract_geometric_features(
    image_path: str, landmarker: vision.FaceLandmarker
) -> np.ndarray | None:
    """
    Извлекает 468 нормализованных расстояний от кончика носа до каждой
    точки лица. Возвращает numpy-массив формы (468,) или None при ошибке.
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        print(f"  [WARN] не удалось прочитать {image_path}", file=sys.stderr)
        return None

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        print(f"  [WARN] лицо не найдено: {image_path}", file=sys.stderr)
        return None

    landmarks = result.face_landmarks[0]  # первое (и единственное) лицо

    # Координаты всех 468 точек
    pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)

    # Кончик носа
    nose_tip = pts[NOSE_TIP_IDX]

    # Внешние углы глаз (для нормировки)
    left_outer = pts[LEFT_EYE_OUTER]
    right_outer = pts[RIGHT_EYE_OUTER]
    interocular = np.linalg.norm(left_outer - right_outer)

    if interocular < 1e-8:
        print(f"  [WARN] нулевое межзрачковое расстояние: {image_path}", file=sys.stderr)
        return None

    # Евклидовы расстояния от кончика носа до каждой точки
    diffs = pts - nose_tip
    distances = np.linalg.norm(diffs, axis=1)  # (468,)

    # Нормировка на межзрачковое расстояние
    normalized = distances / interocular

    return normalized


def main() -> None:
    _os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Создание FaceLandmarker…")
    landmarker = create_landmarker(MODEL_PATH)

    rows = []
    total_processed = 0
    total_skipped = 0

    for emotion_name in EMOTIONS:
        label = EMOTION_LABEL[emotion_name]
        folder = _os.path.join(DATA_DIR, emotion_name)

        if not _os.path.isdir(folder):
            print(f"[WARN] папка не найдена: {folder}", file=sys.stderr)
            continue

        image_files = sorted(
            f for f in _os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )

        print(f"\nОбработка эмоции «{emotion_name}» (label={label}), файлов: {len(image_files)}")

        for fname in image_files:
            img_path = _os.path.join(folder, fname)
            features = extract_geometric_features(img_path, landmarker)

            if features is None:
                total_skipped += 1
                continue

            row = [label] + features.tolist()
            rows.append(row)
            total_processed += 1

    landmarker.close()

    # -----------------------------------------------------------------------
    # Запись CSV
    # -----------------------------------------------------------------------
    columns = ["emotion_label"] + [f"f{i}" for i in range(NUM_LANDMARKS)]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print(f"\nГотово. Обработано: {total_processed}, пропущено: {total_skipped}")
    print(f"Результат сохранён в {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
