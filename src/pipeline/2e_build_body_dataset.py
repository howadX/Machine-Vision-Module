"""
build_body_dataset.py — извлечение поз из COCO-Keypoint для Модели Б.

Эвристики классификации позы:
  - standing:  плечи над бёдрами, bbox вертикальный, колени выпрямлены
  - sitting:   плечи над бёдрами, bbox широкий, колени согнуты
  - lying:     плечи НЕ над бёдрами (горизонтальный bbox)

Выход: models/descriptors/body_posture_coco.csv
Колонки: posture_label, f0..f16 (17 расстояний от центра таза)
"""

import os, sys, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv

# ── 1. Загрузка COCO ──────────────────────────────────
print("Загрузка COCO val2017...")
import torchvision.datasets as datasets
import torchvision.transforms as T

# Пытаемся через torchvision (авто-скачивание)
DATA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "datasets")
os.makedirs(DATA_ROOT, exist_ok=True)

try:
    coco = datasets.CocoDetection(
        root=os.path.join(DATA_ROOT, "coco", "val2017"),
        annFile=os.path.join(DATA_ROOT, "coco", "annotations", "person_keypoints_val2017.json"),
    )
    print(f"COCO загружен: {len(coco)} изображений")
except FileNotFoundError:
    print("COCO не найден локально, скачиваю...")
    import urllib.request, zipfile, io
    
    # Скачать аннотации
    ann_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = os.path.join(DATA_ROOT, "coco", "annotations.zip")
    os.makedirs(os.path.dirname(ann_zip), exist_ok=True)
    
    print("  Скачивание аннотаций (241 MB)...")
    urllib.request.urlretrieve(ann_url, ann_zip)
    with zipfile.ZipFile(ann_zip) as z:
        z.extractall(os.path.join(DATA_ROOT, "coco"))
    print("  Аннотации готовы")

    # Скачать val2017 изображения (нужны не все — только с людьми)
    # Но для простоты: используем только аннотации и генерируем признаки из keypoints
    # Без скачивания изображений — Keypoints уже в аннотациях!
    coco = datasets.CocoDetection(
        root=os.path.join(DATA_ROOT, "coco", "val2017"),
        annFile=os.path.join(DATA_ROOT, "coco", "annotations", "person_keypoints_val2017.json"),
    )

# ── 2. Эвристики позы из keypoints ────────────────────
COCO_KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

def classify_posture(keypoints, bbox):
    """
    Эвристическая классификация позы по 17 COCO-точкам.
    keypoints: [x1,y1,v1, x2,y2,v2, ...] — 17×3 = 51 число
    bbox: [x, y, w, h]
    Возвращает: 0=standing, 1=sitting, 2=lying
    """
    kp = np.array(keypoints).reshape(17, 3)
    vis = kp[:, 2] > 0  # visibility
    
    if vis.sum() < 10:
        return None  # мало точек
    
    # Ключевые точки
    shoulders = kp[[COCO_KP["left_shoulder"], COCO_KP["right_shoulder"]]]
    hips = kp[[COCO_KP["left_hip"], COCO_KP["right_hip"]]]
    knees = kp[[COCO_KP["left_knee"], COCO_KP["right_knee"]]]
    ankles = kp[[COCO_KP["left_ankle"], COCO_KP["right_ankle"]]]
    
    # Центры
    shoulder_center = shoulders[:, :2].mean(axis=0)
    hip_center = hips[:, :2].mean(axis=0)
    knee_center = knees[:, :2].mean(axis=0)
    
    # Вертикальность: Y плеч vs Y бёдер
    dy_shoulder_hip = shoulder_center[1] - hip_center[1]
    
    # Соотношение сторон bbox
    x, y, w, h = bbox
    aspect = h / max(w, 1)
    
    # Высота головы над бёдрами в единицах роста
    head_over_hip = -dy_shoulder_hip / max(h, 1)
    
    if aspect > 1.8 and head_over_hip > 0.3:
        return 0  # standing: вертикальный, голова высоко
    elif aspect < 1.4 and head_over_hip > 0.15:
        return 1  # sitting: широкий, голова ещё над бёдрами
    elif aspect < 1.3 or head_over_hip < 0.1:
        return 2  # lying: горизонтальный или голова на уровне бёдер
    else:
        return None  # неопределено

# ── 3. Извлечение признаков ──────────────────────────

def extract_body_features(keypoints):
    """
    Извлекает 17 нормализованных расстояний от центра таза.
    keypoints: [x1,y1,v1, ...] — 51 число
    """
    kp = np.array(keypoints).reshape(17, 3)
    xy = kp[:, :2].astype(np.float64)
    
    # Центр таза
    hip_center = xy[[COCO_KP["left_hip"], COCO_KP["right_hip"]]].mean(axis=0)
    
    # Межплечевое (нормализатор)
    shoulder_w = np.linalg.norm(
        xy[COCO_KP["left_shoulder"]] - xy[COCO_KP["right_shoulder"]])
    if shoulder_w < 1:
        shoulder_w = 1.0
    
    # Расстояния
    dists = np.linalg.norm(xy - hip_center, axis=1) / shoulder_w
    return dists

# ── 4. Сбор данных ────────────────────────────────────

print("Извлечение поз из COCO...")
X_rows, y_rows = [], []
posture_counts = [0, 0, 0]
total_processed = 0

from pycocotools.coco import COCO
import pycocotools

ann_file = os.path.join(DATA_ROOT, "coco", "annotations", "person_keypoints_val2017.json")
if not os.path.exists(ann_file):
    # Try alternative path
    ann_file = os.path.join(DATA_ROOT, "coco", "annotations", "person_keypoints_val2017.json")

coco_api = COCO(ann_file)
person_ids = coco_api.getAnnIds(catIds=[1])  # category 1 = person
anns = coco_api.loadAnns(person_ids)

print(f"Аннотаций с людьми: {len(anns)}")

for ann in anns:
    if ann["num_keypoints"] < 10:
        continue
    
    bbox = ann["bbox"]  # [x, y, w, h]
    keypoints = ann["keypoints"]
    
    posture = classify_posture(keypoints, bbox)
    if posture is None:
        continue
    
    features = extract_body_features(keypoints)
    X_rows.append(features.tolist())
    y_rows.append(posture)
    posture_counts[posture] += 1
    total_processed += 1

print(f"Собрано: {total_processed} экземпляров")
print(f"  Standing: {posture_counts[0]}")
print(f"  Sitting:  {posture_counts[1]}")
print(f"  Lying:    {posture_counts[2]}")

# ── 5. Сохранение CSV ────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
OUTPUT_DIR = os.path.join(ROOT, "models", "descriptors")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "body_posture_coco.csv")

columns = ["posture_label"] + [f"f{i}" for i in range(17)]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    for x, y in zip(X_rows, y_rows):
        writer.writerow([y] + x)

print(f"Сохранено: {OUTPUT_CSV}")
print(f"Готово.")
