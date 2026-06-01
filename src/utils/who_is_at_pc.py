#!/usr/bin/env python3
"""who_is_at_pc.py — одно фото с камеры → Face ID → ответ текстом."""
import os, sys, cv2, numpy as np, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'utils'))
from face_id import FaceID

CAMERA_ID = 2
GALLERY_DIR = os.path.join(ROOT, 'models', 'references')
THRESHOLD = 0.35
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}

# Захват кадра
cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
time.sleep(0.5)
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("Камера не отвечает.")
    sys.exit(0)

# Инициализация
face_id = FaceID()

# Галерея
gallery = {}
for fname in sorted(os.listdir(GALLERY_DIR)):
    if not fname.endswith('.jpg'): continue
    name_key = fname[2:-4]
    img = cv2.imread(os.path.join(GALLERY_DIR, fname))
    if img is None: continue
    emb = face_id.get_embedding(img)
    if emb is not None:
        gallery[NAME_MAP.get(name_key, name_key)] = emb

if not gallery:
    print("Галерея пуста.")
    sys.exit(0)

# Распознавание
info = face_id.get_face_info(frame)
if info is None:
    print("Никого нет за ПК.")
    sys.exit(0)

name, sim = face_id.identify(info["embedding"], gallery, THRESHOLD)

if name:
    print(f"✅ За ПК: {name} (сходство: {sim:.0%})")
else:
    print(f"⚠️ За ПК: неизвестный (сходство: {sim:.0%})")
