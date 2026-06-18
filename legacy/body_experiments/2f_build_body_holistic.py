"""
2f_build_body_holistic.py — датасет поз тела через Holistic (33 MediaPipe точки).

Скачивает COCO val2017 изображения с людьми → Holistic → 33 признака → CSV.
"""
import os, sys, csv, numpy as np, cv2, mediapipe as mp
import urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.path.join(ROOT, "data", "datasets", "coco")
os.makedirs(os.path.join(DATA_ROOT, "val2017_body"), exist_ok=True)

# ── 1. Загрузка COCO аннотаций ──────────────────────
from pycocotools.coco import COCO
ann_file = os.path.join(DATA_ROOT, "annotations", "person_keypoints_val2017.json")
coco_api = COCO(ann_file)
person_ids = coco_api.getAnnIds(catIds=[1])
anns = coco_api.loadAnns(person_ids)

# Получаем изображения с людьми
img_ids = set(ann["image_id"] for ann in anns)
imgs = coco_api.loadImgs(list(img_ids))
print(f"Изображений с людьми: {len(imgs)}")

# ── 2. Эвристики позы из COCO keypoints ────────────
COCO_KP = {
    "left_shoulder": 5, "right_shoulder": 6,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
}

def classify_posture_coco(keypoints, bbox):
    kp = np.array(keypoints).reshape(17, 3)
    vis = kp[:, 2] > 0
    if vis.sum() < 10: return None
    shoulders = kp[[5,6]]
    hips = kp[[11,12]]
    shoulder_center = shoulders[:,:2].mean(axis=0)
    hip_center = hips[:,:2].mean(axis=0)
    dy = shoulder_center[1] - hip_center[1]
    x, y, w, h = bbox
    aspect = h / max(w, 1)
    head_over_hip = -dy / max(h, 1)
    if aspect > 1.8 and head_over_hip > 0.3: return 0
    elif aspect < 1.4 and head_over_hip > 0.15: return 1
    elif aspect < 1.3 or head_over_hip < 0.1: return 2
    return None

# ── 3. Holistic Landmarker ──────────────────────────
MODEL_PATH = os.path.join(ROOT, "models", "mediapipe", "holistic_landmarker.task")
base_opts = mp_python.BaseOptions(
    model_asset_path=MODEL_PATH,
    delegate=mp_python.BaseOptions.Delegate.GPU,  # GPU ускорение
)
opts = vision.HolisticLandmarkerOptions(base_options=base_opts, running_mode=vision.RunningMode.IMAGE)

# Пул воркеров для распараллеливания
from concurrent.futures import ThreadPoolExecutor, as_completed
NUM_WORKERS = 4  # 4 параллельных Holistic-инференса

def extract_holistic_body_features(image_bgr):
    """Извлекает 33 расстояния от центра таза / межплечевое."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    # Новый landmarker на каждое изображение (разные размеры)
    lm = vision.HolisticLandmarker.create_from_options(opts)
    results = lm.detect(mp_img)
    lm.close()
    
    if not results.pose_landmarks:
        return None
    
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks], dtype=np.float64)
    vis = pts[:, 3] > 0.5
    if vis.sum() < 10: return None
    
    hip_center = (pts[23, :2] + pts[24, :2]) / 2
    shoulder_w = np.linalg.norm(pts[11, :2] - pts[12, :2])
    if shoulder_w < 1e-6: shoulder_w = 1.0
    
    dists = np.linalg.norm(pts[:, :2] - hip_center, axis=1) / shoulder_w
    return dists  # (33,)

# ── 4. Обработка (через torchvision, без скачивания всех изображений) ──
import torchvision.datasets as ds
import torchvision.transforms as T

coco_ds = ds.CocoDetection(
    root=os.path.join(DATA_ROOT, "val2017"),
    annFile=ann_file,
    transform=None,  # возвращаем PIL Image
)

MAX_IMAGES = 2000
X_rows, y_rows = [], []
posture_counts = [0, 0, 0]
total_processed = 0
total_skipped = 0

# Индекс: image_id → список аннотаций
anns_by_image = {}
for ann in anns:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

for img_id in list(anns_by_image.keys())[:MAX_IMAGES]:
    img_anns = anns_by_image[img_id]
    
    try:
        pil_img = coco_ds.coco.loadImgs(img_id)[0]
        img_path = os.path.join(DATA_ROOT, "val2017", pil_img["file_name"])
        
        if not os.path.exists(img_path):
            # Скачиваем одно изображение
            url = f"http://images.cocodataset.org/val2017/{pil_img['file_name']}"
            try:
                urllib.request.urlretrieve(url, img_path)
            except:
                total_skipped += 1
                continue
        
        img = cv2.imread(img_path)
        if img is None:
            total_skipped += 1
            continue
    except:
        total_skipped += 1
        continue
    
    # Берём аннотации этого изображения
    img_anns = [a for a in anns if a["image_id"] == img_id]
    
    for ann in img_anns:
        if ann["num_keypoints"] < 10: continue
        posture = classify_posture_coco(ann["keypoints"], ann["bbox"])
        if posture is None: continue
        
        # Обрезаем по bbox и извлекаем Holistic
        x, y, w, h = map(int, ann["bbox"])
        # Расширяем bbox для контекста
        x1 = max(0, x - w//4)
        y1 = max(0, y - h//6)
        x2 = min(img.shape[1], x + w + w//4)
        y2 = min(img.shape[0], y + h + h//6)
        crop = img[y1:y2, x1:x2]
        
        if crop.size == 0: continue
        
        features = extract_holistic_body_features(crop)
        if features is None:
            total_skipped += 1
            continue
        
        X_rows.append(features.tolist())
        y_rows.append(posture)
        posture_counts[posture] += 1
        total_processed += 1
    
    if total_processed >= 2000: break
    if total_processed % 50 == 0:
        print(f"  Обработано: {total_processed} (пропущено: {total_skipped})")

print(f"\nСобрано: {total_processed} экземпляров")
print(f"  Standing: {posture_counts[0]}, Sitting: {posture_counts[1]}, Lying: {posture_counts[2]}")

# ── 5. Сохранение ────────────────────────────────────
OUTPUT_DIR = os.path.join(ROOT, "models", "descriptors")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "body_posture_holistic_33.csv")

columns = ["posture_label"] + [f"f{i}" for i in range(33)]
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    for x, y in zip(X_rows, y_rows):
        writer.writerow([y] + x)

print(f"Сохранено: {OUTPUT_CSV}")
