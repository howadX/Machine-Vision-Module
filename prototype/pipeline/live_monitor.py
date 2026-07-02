#!/usr/bin/env python3
"""
live_monitor.py — Реалтайм-распознавание лиц и эмоций.

Запуск:
    cd ~/repos/Machine\ Vision\ Module
    venv/bin/python3 pipeline/live_monitor.py

Управление:  q — выход
"""
import cv2
import numpy as np
import os, sys, glob, time
from collections import deque
from PIL import ImageFont, ImageDraw, Image

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SRC))
sys.path.insert(0, os.path.join(ROOT, 'core'))

from face_id import FaceID
from emotion_hse import HSEmotionClassifier

# ── Конфигурация ─────────────────────────────
REFS_DIR = os.path.join(ROOT, "models", "references")
FONT_PATH = os.path.join(ROOT, "assets", "fonts", "DejaVuSans.ttf")
REFERENCES = {"Вадим": "p_V.jpg", "Даня": "p_D.jpg", "Никита": "p_N.jpg", "Белла": "p_I.jpg"}
THRESHOLD = 0.35
DETECT_EVERY_N = 2
EMOTION_EVERY_N = 5
FPS_WINDOW = 30

RU_MAP = {"Anger":"Злость","Contempt":"Презрение","Disgust":"Отвращение",
          "Fear":"Страх","Happiness":"Радость","Neutral":"Нейтральная",
          "Sadness":"Грусть","Surprise":"Удивление"}

FONT_L = ImageFont.truetype(FONT_PATH, 22) if os.path.exists(FONT_PATH) else ImageFont.load_default()
FONT_S = ImageFont.truetype(FONT_PATH, 14) if os.path.exists(FONT_PATH) else ImageFont.load_default()
FONT_XS = ImageFont.truetype(FONT_PATH, 18) if os.path.exists(FONT_PATH) else ImageFont.load_default()


def find_camera():
    for dev in sorted(glob.glob("/dev/video*"), key=lambda x: int(x.replace("/dev/video", ""))):
        idx = int(dev.replace("/dev/video", ""))
        cam = cv2.VideoCapture(dev, cv2.CAP_V4L2) or cv2.VideoCapture(idx, cv2.CAP_V4L2)
        if cam.isOpened():
            ret, _ = cam.read()
            cam.release()
            if ret:
                return dev
    return None


def draw_overlay(frame, bbox, name, sim, emotion, em_conf, top3, fps):
    x1, y1, x2, y2 = map(int, bbox)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    c = (0, 255, 0) if name else (255, 0, 0)

    d.rectangle([(x1, y1), (x2, y2)], outline=c, width=2)

    label = f"{name} ({sim:.0%})" if name else "???"
    tw = int(d.textlength(label, font=FONT_L))
    d.rectangle([(x1, y1 - 28), (x1 + tw + 10, y1)], fill=c)
    d.text((x1 + 5, y1 - 27), label, font=FONT_L, fill=(255, 255, 255))

    em = f"{RU_MAP.get(emotion, emotion)}: {em_conf:.0%}"
    d.text((x1, y2 + 5), em, font=FONT_L, fill=c)

    t3 = " | ".join(f"{RU_MAP.get(n, n)} {c_:.0%}" for n, c_ in top3[:2])
    d.text((x1, y2 + 30), t3, font=FONT_S, fill=(180, 180, 180))

    d.text((10, 10), f"{fps:.0f} FPS", font=FONT_XS, fill=(0, 255, 255))

    frame[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    print("=" * 50)
    print("  🎥 EmoZid Live")
    print("  q — выход")
    print("=" * 50)

    cam_path = find_camera()
    if not cam_path:
        print("❌ Камера не найдена"); sys.exit(1)
    print(f"Камера: {cam_path}")

    print("Загрузка ArcFace...")
    fid = FaceID(use_gpu=True)
    gallery = fid.build_gallery({n: os.path.join(REFS_DIR, f) for n, f in REFERENCES.items()})
    print(f"Галерея: {list(gallery.keys())}" if gallery else "⚠️  Галерея пуста")

    print("Загрузка HSEmotion...")
    ec = HSEmotionClassifier(use_gpu=True)
    print("Готово.\n")

    cam = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
    if not cam.isOpened():
        cam = cv2.VideoCapture(int(cam_path.replace("/dev/video", "")), cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_bbox = last_name = None
    last_sim = last_em_conf = 0.0
    last_emotion = "Neutral"
    last_top3 = []
    fc, ft = 0, deque(maxlen=FPS_WINDOW)

    while True:
        t0 = time.perf_counter()
        ret, frame = cam.read()
        if not ret: break

        if fc % DETECT_EVERY_N == 0:
            info = fid.get_face_info(frame)
            if info:
                last_bbox = info["bbox"]
                if gallery:
                    last_name, last_sim = fid.identify(info["embedding"], gallery, THRESHOLD)
                if fc % EMOTION_EVERY_N == 0:
                    x1, y1, x2, y2 = map(int, last_bbox)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        last_emotion, last_em_conf = ec.predict(crop)
                        last_top3 = ec.predict_top_k(crop, 3)
            else:
                last_bbox = None

        if last_bbox:
            draw_overlay(frame, last_bbox, last_name, last_sim,
                         last_emotion, last_em_conf, last_top3, ft[-1] if ft else 0)

        ft.append(1.0 / max(0.001, time.perf_counter() - t0))
        cv2.imshow("EmoZid Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"): break
        fc += 1

    cam.release(); cv2.destroyAllWindows()
    print("Выход.")


if __name__ == "__main__":
    main()
