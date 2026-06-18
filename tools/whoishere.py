#!/usr/bin/env python3
"""whoishere.py — визуальный контекст для LLM: один кадр с камеры → текстовый отчёт.
Вызов: python tools/whoishere.py
Вывод: plain text, key: value, без эмодзи — пригоден для парсинга LLM.

Ответственность: Face ID + базовая сцена (освещение, дистанция).
Эмоции и поза тела — зона ответственности Sight.
"""
import os, sys, glob, time, warnings, logging
from contextlib import contextmanager

# ── Подавление отладочного шума ──────────────────────────
os.environ.setdefault('ONNXRUNTIME_LOG_LEVEL', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
warnings.filterwarnings('ignore')

@contextmanager
def suppress_all():
    """Глушит stdout+stderr на уровне fd (для инициализации моделей)."""
    for name in ('insightface', 'onnxruntime', 'absl'):
        logging.getLogger(name).setLevel(logging.ERROR)
    old_out_fd = os.dup(1)
    old_err_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(old_out_fd, 1)
        os.dup2(old_err_fd, 2)
        os.close(devnull)
        os.close(old_out_fd)
        os.close(old_err_fd)


@contextmanager
def suppress_stderr():
    """Глушит только stderr на уровне fd (C-логи)."""
    old_err_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_err_fd, 2)
        os.close(devnull)
        os.close(old_err_fd)

with suppress_all():
    import cv2
    import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from face_id import FaceID

# ── Конфигурация ──────────────────────────────────────────
GALLERY_DIR = os.path.join(ROOT, 'models', 'references')
THRESHOLD = 0.35
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}


def find_camera():
    devices = sorted(glob.glob('/dev/video*'),
                     key=lambda x: int(x.replace('/dev/video', '')))
    for dev in devices:
        idx = int(dev.replace('/dev/video', ''))
        for method in ['path', 'index']:
            cap = cv2.VideoCapture(dev if method == 'path' else idx, cv2.CAP_V4L2)
            if cap.isOpened():
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.mean() > 10:
                        cap.release()
                        return dev
                cap.release()
    return None


def capture_frame(cam_path):
    cam_idx = int(cam_path.replace('/dev/video', ''))
    ret, frame = False, None
    for _ in range(3):
        cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        for __ in range(5):
            ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.mean() > 10:
            return frame
        time.sleep(0.3)
    return None


def estimate_distance(bbox, frame_shape):
    x1, y1, x2, y2 = bbox
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    ratio = face_area / frame_area
    if ratio > 0.12:   return 'near'
    elif ratio > 0.04: return 'medium'
    return 'far'


def estimate_lighting(frame):
    mean_brightness = float(frame.mean())
    if mean_brightness > 150:   level = 'bright'
    elif mean_brightness > 70:  level = 'normal'
    else:                       level = 'dim'
    return level, mean_brightness


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # 1. Камера
    cam_path = find_camera()
    if cam_path is None:
        print("camera: none")
        print("error: camera not found")
        sys.exit(1)
    print(f"camera: {cam_path}")

    # 2. Кадр
    frame = capture_frame(cam_path)
    if frame is None:
        print("error: failed to capture frame")
        sys.exit(1)

    # 3. FaceID
    with suppress_all():
        face_id = FaceID(use_gpu=True)

    # 4. Галерея эталонов
    gallery = {}
    for fname in sorted(os.listdir(GALLERY_DIR)):
        if not fname.endswith('.jpg'):
            continue
        name_key = fname[2:-4]
        img = cv2.imread(os.path.join(GALLERY_DIR, fname))
        if img is None:
            continue
        with suppress_all():
            emb = face_id.get_embedding(img)
        if emb is not None:
            gallery[NAME_MAP.get(name_key, name_key)] = emb

    # 5. Все лица
    with suppress_stderr():
        raw_faces = face_id.app.get(frame)

    # ── Вывод: сцена ───────────────────────────────────
    if not raw_faces:
        print("face_count: 0")
        lighting_level, lighting_val = estimate_lighting(frame)
        print(f"lighting: {lighting_level}")
        print(f"brightness: {lighting_val:.0f}")
        return

    print(f"face_count: {len(raw_faces)}")
    lighting_level, lighting_val = estimate_lighting(frame)
    print(f"lighting: {lighting_level}")
    print(f"brightness: {lighting_val:.0f}")

    # ── Вывод: каждое лицо ─────────────────────────────
    raw_faces.sort(key=lambda f: f.get('det_score', 0), reverse=True)

    for i, face in enumerate(raw_faces, 1):
        bbox = face['bbox'].tolist()
        det_score = float(face.get('det_score', 0))
        emb = np.asarray(face['embedding'], dtype=np.float32)

        if gallery:
            name, sim = face_id.identify(emb, gallery, THRESHOLD)
        else:
            name, sim = None, 0.0

        dist = estimate_distance(bbox, frame.shape)
        face_w = int(bbox[2] - bbox[0])
        face_h = int(bbox[3] - bbox[1])

        print(f"face_{i}:")
        print(f"  identity: {name if name else 'unknown'}")
        print(f"  confidence: {sim:.2f}")
        print(f"  det_score: {det_score:.2f}")
        print(f"  bbox: {int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}")
        print(f"  size: {face_w}x{face_h}")
        print(f"  distance: {dist}")


if __name__ == '__main__':
    main()
