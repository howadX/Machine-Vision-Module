#!/usr/bin/env python3
"""
whois.py — Гермес, кто перед камерой?
Захватывает кадр, находит лицо (ArcFace), идентифицирует человека и определяет эмоцию.

Запуск:
    cd ~/repos/EmoZid
    venv/bin/python3 src/whois.py

Требования: веб-камера, зависимости в venv.
"""
import cv2
import numpy as np
import os
import sys
import glob
from time import sleep

# Добавляем src/utils в PYTHONPATH (все модули лежат рядом)
SRC = os.path.dirname(os.path.abspath(__file__))    # src/utils
ROOT = os.path.dirname(os.path.dirname(SRC))         # корень проекта
sys.path.insert(0, SRC)

from face_id import FaceID
from emotion_hse import HSEmotionClassifier

# ── Пути ──────────────────────────────────────────────
MODELS_DIR = os.path.join(ROOT, "models")
REFS_DIR = os.path.join(MODELS_DIR, "references")

# ── Эталоны ───────────────────────────────────────────
REFERENCES = {
    "Даня":   "p_D.jpg",
    "Никита": "p_N.jpg",
    "Белла":  "p_I.jpg",
    "Вадим":  "p_V.jpg",
}

THRESHOLD = 0.35  # косинусное сходство (ArcFace)


# ── Поиск камеры ──────────────────────────────────────
def find_camera():
    devices = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")))
    for dev in devices:
        idx = int(dev.replace("/dev/video", ""))
        for method in ["path", "index"]:
            cam = cv2.VideoCapture(dev if method == "path" else idx, cv2.CAP_V4L2)
            if cam.isOpened():
                for _ in range(5):
                    ret, frame = cam.read()
                    if ret and frame is not None and frame.mean() > 10:
                        cam.release()
                        return dev, idx
                cam.release()
    return None, None


# ── Главное ───────────────────────────────────────────
def main():
    print("🔍 Гермес смотрит в камеру...")

    cam_path, cam_idx = find_camera()
    if cam_path is None:
        print("❌ Камера не найдена. Проверь: ls /dev/video*")
        sys.exit(1)
    print(f"   Камера: {cam_path}")

    # Захват
    frame = None
    for attempt in range(3):
        cam = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not cam.isOpened():
            cam = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for _ in range(5):
            ret, frame = cam.read()
        cam.release()
        if frame is not None and frame.mean() > 10:
            break
        sleep(0.3)

    if frame is None or frame.mean() < 10:
        print("❌ Не удалось получить кадр (чёрный экран?).")
        sys.exit(1)

    print(f"   Кадр: {frame.shape[1]}x{frame.shape[0]}, яркость={frame.mean():.0f}")

    # ── ArcFace: детекция + эмбеддинг ──────────────────
    print("   Загрузка ArcFace (buffalo_l, GPU)...")
    fid = FaceID(use_gpu=True)
    gallery = fid.build_gallery({
        name: os.path.join(REFS_DIR, fname)
        for name, fname in REFERENCES.items()
    })

    if not gallery:
        print("⚠️  Галерея пуста — не удалось загрузить эталоны.")
    else:
        print(f"   Галерея: {list(gallery.keys())}")

    face_info = fid.get_face_info(frame)
    if face_info is None:
        debug_path = "/tmp/whois_debug.jpg"
        cv2.imwrite(debug_path, frame)
        print(f"👻 Лицо не найдено (снимок: {debug_path})")
        return

    embedding = face_info["embedding"]
    bbox = face_info["bbox"]
    score = face_info["det_score"]
    print(f"   Лицо: bbox=({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f}), score={score:.2f}")

    # ── Идентификация (ArcFace) ────────────────────────
    if gallery:
        name, sim = fid.identify(embedding, gallery, threshold=THRESHOLD)
        print("\n── Сравнение с базой ──")
        for gname, gemb in gallery.items():
            sim_i = float(np.dot(embedding / np.linalg.norm(embedding),
                                 gemb / np.linalg.norm(gemb)))
            pct = min(100, sim_i * 100)
            bar = "█" * int(pct / 10) + "░" * (10 - int(pct / 10))
            print(f"  {gname:12s}  sim={sim_i:.4f}  [{bar}]  {pct:.0f}%")

        print()
        if name:
            print(f"👤 Это {name}!  (cos_sim={sim:.4f}, порог={THRESHOLD})")
        else:
            print(f"❓ Неизвестный. Ближайший: cos_sim={sim:.4f} (порог={THRESHOLD})")

    # ── Эмоция (HSEmotion) ──────────────────────────
    try:
        print("   Загрузка HSEmotion (эмоции, GPU)...")
        ec = HSEmotionClassifier(use_gpu=True)
        print(f"   {ec}")

        # Перевод англ. эмоций в русские
        RU_MAP = {
            "Anger": "Злость", "Contempt": "Презрение", "Disgust": "Отвращение",
            "Fear": "Страх", "Happiness": "Радость", "Neutral": "Нейтральная",
            "Sadness": "Грусть", "Surprise": "Удивление",
        }

        x1, y1, x2, y2 = [int(v) for v in bbox]
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            name_en, conf = ec.predict(face_crop)
            name_ru = RU_MAP.get(name_en, name_en)
            print(f"😶 Эмоция: {name_ru}  (уверенность: {conf:.0%})")
            top3 = ec.predict_top_k(face_crop, k=3)
            top3_ru = ", ".join(f"{RU_MAP.get(n, n)} ({c:.0%})" for n, c in top3)
            print(f"   Топ-3: {top3_ru}")
        else:
            print("⚠️  Не удалось вырезать лицо по bbox")
    except Exception as e:
        print(f"⚠️  Эмоция не определена: {e}")


if __name__ == "__main__":
    main()
