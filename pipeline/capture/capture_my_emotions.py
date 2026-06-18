#!/usr/bin/env python3
"""
capture_my_emotions.py — Снимает 7 эмоций через камеру для дообучения модели.

Каждая эмоция: 30 кадров с авто-детекцией лица (ArcFace).
Выход: data/my_emotions/{angry,disgust,fear,happy,sad,surprise,neutral}/*.jpg

Запуск:
    cd ~/repos/Machine\ Vision\ Module
    venv/bin/python3 pipeline/capture/capture_my_emotions.py
"""
import cv2
import os
import sys
import glob
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent.parent
sys.path.insert(0, str(ROOT / 'core'))

from face_id import FaceID

EMOTIONS = [
    ("angry",    "Злость",      "Сделай ЗЛОЕ лицо — нахмурь брови, сожми губы"),
    ("disgust",  "Отвращение",  "Покажи ОТВРАЩЕНИЕ — сморщи нос, как от запаха"),
    ("fear",     "Страх",       "Изобрази СТРАХ — широко открой глаза, приоткрой рот"),
    ("happy",    "Радость",     "УЛЫБНИСЬ широко — от души!"),
    ("sad",      "Грусть",      "Сделай ГРУСТНОЕ лицо — опусти уголки губ"),
    ("surprise", "Удивление",   "Изобрази УДИВЛЕНИЕ — подними брови, открой рот"),
    ("neutral",  "Нейтральная", "НЕЙТРАЛЬНОЕ лицо — расслабь все мышцы"),
]

OUT_DIR = ROOT / "data" / "my_emotions"
FRAMES_PER_EMOTION = 50


def find_camera():
    devices = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")))
    for dev in devices:
        idx = int(dev.replace("/dev/video", ""))
        cam = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cam.isOpened():
            cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cam.isOpened():
            ret, _ = cam.read()
            if ret:
                cam.release()
                return dev
            cam.release()
    return None


def main():
    print("=" * 55)
    print("  📷 Захват эмоций для дообучения модели")
    print(f"  Каждая эмоция: {FRAMES_PER_EMOTION} кадров")
    print(f"  Выход: {OUT_DIR}")
    print("=" * 55)

    cam_path = find_camera()
    if cam_path is None:
        print("❌ Камера не найдена.")
        sys.exit(1)
    print(f"\nКамера: {cam_path}")

    print("\nЗагрузка ArcFace для детекции лица...")
    fid = FaceID(use_gpu=True)

    for en_name, ru_name, hint in EMOTIONS:
        em_dir = OUT_DIR / en_name
        em_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'─' * 50}")
        print(f"🎭 {ru_name} ({en_name})")
        print(f"   {hint}")
        print(f"   Сделай лицо и нажми ENTER...")
        input()

        print("   3...")
        time.sleep(1)
        print("   2...")
        time.sleep(1)
        print("   1...")
        time.sleep(1)
        print(f"   📸 Снимаю {FRAMES_PER_EMOTION} кадров...")

        captured = 0
        attempts = 0
        max_attempts = FRAMES_PER_EMOTION * 3

        cam = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not cam.isOpened():
            cam = cv2.VideoCapture(int(cam_path.replace("/dev/video", "")), cv2.CAP_V4L2)

        while captured < FRAMES_PER_EMOTION and attempts < max_attempts:
            ret, frame = cam.read()
            attempts += 1
            if not ret:
                continue

            info = fid.get_face_info(frame)
            if info is None:
                continue  # лицо не найдено — пропускаем кадр

            bbox = info["bbox"]
            x1, y1, x2, y2 = [int(v) for v in bbox]

            # Расширяем bbox на 15% чтобы захватить брови/подбородок
            h, w = y2 - y1, x2 - x1
            margin_x = int(w * 0.15)
            margin_y = int(h * 0.15)
            x1 = max(0, x1 - margin_x)
            y1 = max(0, y1 - margin_y)
            x2 = min(frame.shape[1], x2 + margin_x)
            y2 = min(frame.shape[0], y2 + margin_y)

            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size == 0:
                continue

            out_path = em_dir / f"{captured:04d}.jpg"
            cv2.imwrite(str(out_path), face_crop)
            captured += 1

            pct = captured / FRAMES_PER_EMOTION
            bar = "█" * int(pct * 15) + "░" * (15 - int(pct * 15))
            print(f"\r   [{bar}] {captured}/{FRAMES_PER_EMOTION}", end="", flush=True)

        cam.release()
        print(f"\n   ✅ Сохранено: {captured} кадров")

    print(f"\n{'=' * 55}")
    print(f"  Готово! Данные в: {OUT_DIR}")
    print(f"  Запусти дообучение: venv/bin/python3 pipeline/train_emotion.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
