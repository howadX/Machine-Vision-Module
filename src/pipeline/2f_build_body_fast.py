"""
2f_build_body_fast.py — многопоточное извлечение поз через Holistic (CPU, 20 воркеров).
"""
import os, sys, csv, numpy as np, cv2, mediapipe as mp, urllib.request
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from multiprocessing import Pool, cpu_count
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_ROOT = os.path.join(ROOT, "data", "datasets", "coco")
MODEL_PATH = os.path.join(ROOT, "models", "mediapipe", "holistic_landmarker.task")

from pycocotools.coco import COCO
ann_file = os.path.join(DATA_ROOT, "annotations", "person_keypoints_val2017.json")
coco_api = COCO(ann_file)
person_ids = coco_api.getAnnIds(catIds=[1])
anns = coco_api.loadAnns(person_ids)

# Индекс аннотаций по image_id
anns_by_image = {}
for ann in anns:
    anns_by_image.setdefault(ann["image_id"], []).append(ann)

# ── Эвристики позы ──────────────────────────────
def classify_posture_coco(keypoints, bbox):
    kp = np.array(keypoints).reshape(17, 3)
    vis = kp[:, 2] > 0
    if vis.sum() < 10: return None
    shoulders = kp[[5,6]]; hips = kp[[11,12]]
    sc = shoulders[:,:2].mean(axis=0); hc = hips[:,:2].mean(axis=0)
    dy = sc[1] - hc[1]
    x, y, w, h = bbox
    aspect = h / max(w, 1)
    head_over_hip = -dy / max(h, 1)
    if aspect > 1.8 and head_over_hip > 0.3: return 0
    elif aspect < 1.4 and head_over_hip > 0.15: return 1
    elif aspect < 1.3 or head_over_hip < 0.1: return 2
    return None

# ── Функция-воркер (один на процесс) ────────────
def process_image(img_id):
    """Извлекает признаки из одного изображения. Возвращает список (features, posture)."""
    img_anns = anns_by_image[img_id]
    results_list = []
    
    # Свой landmarker на процесс
    base_opts = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    opts = vision.HolisticLandmarkerOptions(base_options=base_opts, running_mode=vision.RunningMode.IMAGE)
    
    try:
        pil_img = coco_api.loadImgs(img_id)[0]
        img_path = os.path.join(DATA_ROOT, "val2017", pil_img["file_name"])
        if not os.path.exists(img_path):
            url = f"http://images.cocodataset.org/val2017/{pil_img['file_name']}"
            urllib.request.urlretrieve(url, img_path)
        img = cv2.imread(img_path)
        if img is None: return results_list
    except:
        return results_list
    
    for ann in img_anns:
        if ann["num_keypoints"] < 10: continue
        posture = classify_posture_coco(ann["keypoints"], ann["bbox"])
        if posture is None: continue
        
        x, y, w, h = map(int, ann["bbox"])
        x1 = max(0, x - w//4); y1 = max(0, y - h//6)
        x2 = min(img.shape[1], x + w + w//4); y2 = min(img.shape[0], y + h + h//6)
        crop = img[y1:y2, x1:x2]
        if crop.size == 0: continue
        
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        
        lm = vision.HolisticLandmarker.create_from_options(opts)
        res = lm.detect(mp_img)
        lm.close()
        
        if not res.pose_landmarks: continue
        
        pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in res.pose_landmarks], dtype=np.float64)
        if (pts[:, 3] > 0.5).sum() < 10: continue
        
        hip_center = (pts[23, :2] + pts[24, :2]) / 2
        shoulder_w = np.linalg.norm(pts[11, :2] - pts[12, :2])
        if shoulder_w < 1e-6: shoulder_w = 1.0
        dists = np.linalg.norm(pts[:, :2] - hip_center, axis=1) / shoulder_w
        
        results_list.append((dists.tolist(), posture))
    
    return results_list

# ── Главный цикл ────────────────────────────────
NUM_WORKERS = min(20, cpu_count())
MAX_IMAGES = 5000
img_ids = list(anns_by_image.keys())[:MAX_IMAGES]

print(f"Запуск на {NUM_WORKERS} ядрах, изображений: {len(img_ids)}")
t0 = time.perf_counter()

X_rows, y_rows = [], []
posture_counts = [0, 0, 0]
processed = 0

with Pool(NUM_WORKERS) as pool:
    for batch_results in pool.imap_unordered(process_image, img_ids, chunksize=10):
        for features, posture in batch_results:
            X_rows.append(features)
            y_rows.append(posture)
            posture_counts[posture] += 1
            processed += 1
        if processed % 50 == 0 and processed > 0:
            elapsed = time.perf_counter() - t0
            print(f"  {processed} экз. за {elapsed:.0f}с ({processed/elapsed:.1f} экз/с)")

elapsed = time.perf_counter() - t0
print(f"\nГотово за {elapsed:.0f}с ({processed/elapsed:.1f} экз/с)")
print(f"Собрано: {processed} экземпляров")
print(f"  Standing: {posture_counts[0]}, Sitting: {posture_counts[1]}, Lying: {posture_counts[2]}")

# Сохранение
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
