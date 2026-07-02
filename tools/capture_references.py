#!/usr/bin/env python3
"""
capture_references.py — захват эталонных фото для identity-галереи.

Делает N снимков с указанным интервалом, сохраняет в models/references/.
Имена: p_V_01.jpg, p_V_02.jpg, ... (префикс настраивается).

Использование:
    venv/bin/python3 tools/capture_references.py --count 7 --interval 3 --person V
"""

import argparse
import cv2
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "models" / "references"


def list_cameras():
    """Возвращает список доступных камер."""
    available = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available


def find_camera():
    """Авто-поиск камеры."""
    cams = list_cameras()
    if not cams:
        print("Камеры не найдены", file=sys.stderr)
        sys.exit(1)

    # Приоритет: камера на ТВ (обычно /dev/video2)
    for cam_id in cams:
        cap = cv2.VideoCapture(cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            if w >= 640 or h >= 480:
                return cam_id
    return cams[0]


def main():
    parser = argparse.ArgumentParser(description="Захват эталонных фото")
    parser.add_argument("--count", type=int, default=7, help="Количество снимков (default: 7)")
    parser.add_argument("--interval", type=int, default=3, help="Интервал между снимками в секундах (default: 3)")
    parser.add_argument("--person", type=str, default="V", help="Префикс имени (default: V)")
    parser.add_argument("--camera", type=int, default=None, help="Индекс камеры (авто-поиск если не указан)")
    parser.add_argument("--width", type=int, default=1280, help="Ширина кадра")
    parser.add_argument("--height", type=int, default=720, help="Высота кадра")
    args = parser.parse_args()

    cam_id = args.camera if args.camera is not None else find_camera()
    print(f"Камера: /dev/video{cam_id}")

    cap = cv2.VideoCapture(cam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"Не удалось открыть камеру /dev/video{cam_id}", file=sys.stderr)
        sys.exit(1)

    os.makedirs(REF_DIR, exist_ok=True)

    # Найти следующий свободный индекс
    existing = sorted(REF_DIR.glob(f"p_{args.person}_*.jpg"))
    start_idx = 1
    if existing:
        nums = []
        for f in existing:
            try:
                nums.append(int(f.stem.split("_")[-1]))
            except ValueError:
                pass
        if nums:
            start_idx = max(nums) + 1

    print(f"Будет сделано {args.count} снимков с интервалом {args.interval} сек")
    print(f"Файлы: p_{args.person}_{start_idx:02d}.jpg ... p_{args.person}_{start_idx + args.count - 1:02d}.jpg")
    print(f"\nРакурсы для съёмки:")
    print(f"  1. Фронтально, нейтральное выражение (близко, ~0.7 м)")
    print(f"  2. Лёгкий поворот влево (~15–20°)")
    print(f"  3. Лёгкий поворот вправо (~15–20°)")
    print(f"  4. Лёгкий наклон головы вниз (взгляд на клавиатуру)")
    print(f"  5. Рабочее расстояние (~1.5–2 м), фронтально")
    print(f"  6. Рабочее расстояние, лёгкий поворот")
    print(f"  7. Близко (~0.5 м), фронтально (максимум деталей)")
    print(f"\nНачинаем через 3 секунды...")
    time.sleep(3)

    for i in range(args.count):
        # Прогрев: несколько кадров для автоэкспозиции
        for _ in range(10):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            print(f"  [{i+1}/{args.count}] Ошибка захвата кадра", file=sys.stderr)
            continue

        fname = f"p_{args.person}_{start_idx + i:02d}.jpg"
        fpath = REF_DIR / fname
        cv2.imwrite(str(fpath), frame)
        size_kb = fpath.stat().st_size / 1024
        print(f"  [{i+1}/{args.count}] ✓ {fname} ({size_kb:.0f} KB)")

        if i < args.count - 1:
            print(f"       Следующий ракурс через {args.interval} сек...")
            time.sleep(args.interval)

    cap.release()
    print(f"\nГотово. {args.count} снимков в {REF_DIR}/")


if __name__ == "__main__":
    main()
