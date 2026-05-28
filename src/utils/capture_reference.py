#!/usr/bin/env python3
"""
capture_reference.py — переснять эталонное фото через камеру (ArcFace-совместимо).

Запуск:
    cd ~/repos/EmoZid
    venv/bin/python3 src/capture_reference.py Вадим
"""
import cv2
import os
import sys
import glob

SRC = os.path.dirname(os.path.abspath(__file__))    # src/utils
ROOT = os.path.dirname(os.path.dirname(SRC))         # корень проекта
sys.path.insert(0, SRC)
from face_id import FaceID

REFS_DIR = os.path.join(ROOT, "models", "references")
NAMES = {"Даня": "p_D.jpg", "Никита": "p_N.jpg", "Белла": "p_I.jpg", "Вадим": "p_V.jpg"}


def main():
    if len(sys.argv) < 2:
        print("Использование: venv/bin/python3 src/capture_reference.py <имя>")
        print(f"  Доступные: {', '.join(NAMES.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    if name not in NAMES:
        print(f"Неизвестное имя: {name}. Доступные: {', '.join(NAMES.keys())}")
        sys.exit(1)

    fname = NAMES[name]
    out_path = os.path.join(REFS_DIR, fname)

    # Найти камеру
    devices = sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", "")))
    cam_path = None
    for dev in devices:
        idx = int(dev.replace("/dev/video", ""))
        cam = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if not cam.isOpened():
            cam = cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cam.isOpened():
            ret, _ = cam.read()
            if ret:
                cam_path = dev
                cam.release()
                break
            cam.release()

    if cam_path is None:
        print("❌ Камера не найдена.")
        sys.exit(1)

    print(f"📷 Камера: {cam_path}")
    print(f"👤 Снимаем эталон для: {name}")
    print("   Смотри прямо в камеру. Нейтральное выражение лица.")
    print("   3...")
    import time
    time.sleep(1)
    print("   2...")
    time.sleep(1)
    print("   1...")
    time.sleep(1)

    # Захват
    cam = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
    if not cam.isOpened():
        cam = cv2.VideoCapture(int(cam_path.replace("/dev/video", "")), cv2.CAP_V4L2)
    for _ in range(5):
        ret, frame = cam.read()
    cam.release()

    if not ret or frame.mean() < 10:
        print("❌ Не удалось получить кадр.")
        sys.exit(1)

    # Проверяем, что лицо найдено
    fid = FaceID(use_gpu=False)
    info = fid.get_face_info(frame)
    if info is None:
        print("❌ Лицо не найдено. Попробуй ещё раз с лучшим освещением.")
        sys.exit(1)

    bbox = info["bbox"]
    score = info["det_score"]
    print(f"   Лицо: bbox=({bbox[0]:.0f},{bbox[1]:.0f})-({bbox[2]:.0f},{bbox[3]:.0f}), score={score:.2f}")

    # Сохраняем
    cv2.imwrite(out_path, frame)
    print(f"✅ Эталон сохранён: {out_path}")
    print(f"   Размер: {frame.shape[1]}x{frame.shape[0]}")


if __name__ == "__main__":
    main()
