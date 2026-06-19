#!/usr/bin/env python3
"""test_yunet.py — тест YuNet как замены SCRFD-детекции в MVM.

Сравнивает YuNet (CPU) + ArcFace (GPU) против SCRFD (GPU) + ArcFace (GPU):
- Скорость детекции
- Совпадение идентификаций
- Пригодность для real-time пайплайна

Интеграция: заменить `face_id.app.get(frame)` → `yunet_detect(frame)` + `face_id.get_embedding(crop)`.
"""
import os, sys, glob, time, warnings, logging
from contextlib import contextmanager

# ── Подавление шума ──────────────────────────────────────
os.environ.setdefault('ONNXRUNTIME_LOG_LEVEL', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
warnings.filterwarnings('ignore')

@contextmanager
def suppress_all():
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
    old_err_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_err_fd, 2)
        os.close(devnull)

with suppress_all():
    import cv2
    import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from face_id import FaceID

# ── Конфигурация ─────────────────────────────────────────
MODEL_DIR = os.path.join(ROOT, 'models', 'yunet')
YUNET_MODEL = os.path.join(MODEL_DIR, 'face_detection_yunet_2023mar.onnx')
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


def init_yunet(model_path, input_size=(640, 480)):
    """Инициализация YuNet-детектора."""
    detector = cv2.FaceDetectorYN_create(
        model_path, "", input_size,
        score_threshold=0.6,
        nms_threshold=0.3,
        top_k=10
    )
    return detector


def yunet_detect(detector, frame):
    """YuNet-детекция: возвращает список bbox + landmarks.

    Формат вывода FaceDetectorYN: [x, y, w, h,
        reye_x, reye_y, leye_x, leye_y, nose_x, nose_y,
        rmouth_x, rmouth_y, lmouth_x, lmouth_y,
        confidence]
    """
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(frame)
    if faces is None:
        return []
    results = []
    for face in faces:
        x, y, fw, fh = [int(v) for v in face[:4]]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w, x + fw), min(h, y + fh)
        det_score = float(face[14])
        # 5 лэндмарок: reye, leye, nose, rmouth, lmouth
        landmarks = face[4:14].reshape(5, 2)
        # Пропускаем лица с нулевым размером
        if x2 <= x1 or y2 <= y1:
            continue
        results.append({
            'bbox': [x1, y1, x2, y2],
            'landmarks': landmarks,
            'det_score': det_score,
        })
    return results


def run_yunet_pipeline(face_id, detector, frame, gallery):
    """YuNet-детекция + ArcFace-эмбеддинг."""
    t0 = time.perf_counter()

    # 1. YuNet детекция
    faces = yunet_detect(detector, frame)
    t_detect = time.perf_counter() - t0

    # 2. Эмбеддинги
    t_emb_start = time.perf_counter()
    results = []
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        with suppress_stderr():
            emb = face_id.get_embedding(crop)
        if emb is not None:
            name, sim = face_id.identify(emb, gallery, THRESHOLD)
        else:
            name, sim = None, 0.0
        results.append({
            **face,
            'embedding': emb,
            'identity': name,
            'confidence': sim,
        })
    t_embed = time.perf_counter() - t_emb_start
    t_total = time.perf_counter() - t0

    return results, t_detect, t_embed, t_total


def run_scrfd_pipeline(face_id, frame, gallery):
    """SCRFD-детекция + ArcFace-эмбеддинг (текущий метод)."""
    t0 = time.perf_counter()

    with suppress_stderr():
        raw_faces = face_id.app.get(frame)

    t_detect = time.perf_counter() - t0
    t_emb_start = time.perf_counter()

    results = []
    if raw_faces:
        raw_faces.sort(key=lambda f: f.get('det_score', 0), reverse=True)
        for face in raw_faces:
            bbox = face['bbox'].tolist()
            det_score = float(face.get('det_score', 0))
            emb = np.asarray(face['embedding'], dtype=np.float32)
            name, sim = face_id.identify(emb, gallery, THRESHOLD)
            results.append({
                'bbox': bbox,
                'det_score': det_score,
                'identity': name,
                'confidence': sim,
            })

    t_embed = time.perf_counter() - t_emb_start
    t_total = time.perf_counter() - t0

    return results, t_detect, t_embed, t_total


def main():
    sys.stdout.reconfigure(line_buffering=True)

    # 1. Камера
    cam_path = find_camera()
    if cam_path is None:
        print("camera: none\nerror: camera not found")
        sys.exit(1)
    print(f"camera: {cam_path}")

    # 2. Кадр
    frame = capture_frame(cam_path)
    if frame is None:
        print("error: failed to capture frame")
        sys.exit(1)
    h, w = frame.shape[:2]
    print(f"frame: {w}x{h}")

    # 3. Инициализация
    with suppress_all():
        face_id = FaceID(use_gpu=True)

    if not os.path.exists(YUNET_MODEL):
        print(f"error: YuNet model not found at {YUNET_MODEL}")
        print("Download: wget https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
        sys.exit(1)

    detector = init_yunet(YUNET_MODEL)

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

    # 5. Сцена
    lighting_level, lighting_val = estimate_lighting(frame)

    print()
    print("=" * 60)
    print("  YuNet vs SCRFD — сравнение детекторов")
    print("=" * 60)
    print(f"  lighting: {lighting_level} ({lighting_val:.0f})")

    # ── ЗАПУСК: YuNet ──────────────────────────────────
    yunet_results, y_t_detect, y_t_embed, y_t_total = \
        run_yunet_pipeline(face_id, detector, frame, gallery)

    # ── ЗАПУСК: SCRFD ──────────────────────────────────
    scrfd_results, s_t_detect, s_t_embed, s_t_total = \
        run_scrfd_pipeline(face_id, frame, gallery)

    # ── ВЫВОД: сравнение ───────────────────────────────
    print()
    print(f"  {'':25} {'YuNet (CPU)':>15} {'SCRFD (GPU)':>15} {'Разница':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'Детекция:':25} {y_t_detect*1000:>12.1f} ms {s_t_detect*1000:>12.1f} ms {y_t_detect/s_t_detect:>8.1%}x")
    print(f"  {'Эмбеддинг:':25} {y_t_embed*1000:>12.1f} ms {s_t_embed*1000:>12.1f} ms")
    print(f"  {'ИТОГО:':25} {y_t_total*1000:>12.1f} ms {s_t_total*1000:>12.1f} ms")
    print(f"  {'Лиц найдено:':25} {len(yunet_results):>15} {len(scrfd_results):>15}")

    # Идентификации
    yunet_ids = {r.get('identity') for r in yunet_results if r.get('identity')}
    scrfd_ids = {r.get('identity') for r in scrfd_results if r.get('identity')}
    yunet_ids_str = ', '.join(sorted(yunet_ids)) if yunet_ids else '—'
    scrfd_ids_str = ', '.join(sorted(scrfd_ids)) if scrfd_ids else '—'
    print(f"  {'Идентификации:':25} {yunet_ids_str:>15} {scrfd_ids_str:>15}")
    match = '✓ СОВПАДАЮТ' if yunet_ids == scrfd_ids else '✗ РАЗНЫЕ'
    print(f"  {'Совпадение:':25} {match:>15}")

    # Детальные результаты
    print()
    print("── YuNet результат ──")
    if not yunet_results:
        print("  face_count: 0")
    else:
        print(f"  face_count: {len(yunet_results)}")
        for i, r in enumerate(yunet_results, 1):
            x1, y1, x2, y2 = r['bbox']
            print(f"  face_{i}:")
            print(f"    identity: {r.get('identity') or 'unknown'}")
            print(f"    confidence: {r.get('confidence', 0):.2f}")
            print(f"    det_score: {r.get('det_score', 0):.2f}")
            print(f"    bbox: {x1},{y1},{x2},{y2}")
            dist = estimate_distance(r['bbox'], frame.shape)
            print(f"    distance: {dist}")

    print()
    print("── SCRFD результат ──")
    if not scrfd_results:
        print("  face_count: 0")
    else:
        print(f"  face_count: {len(scrfd_results)}")
        for i, r in enumerate(scrfd_results, 1):
            x1, y1, x2, y2 = r['bbox']
            print(f"  face_{i}:")
            print(f"    identity: {r.get('identity') or 'unknown'}")
            print(f"    confidence: {r.get('confidence', 0):.2f}")
            print(f"    det_score: {r.get('det_score', 0):.2f}")
            print(f"    bbox: {x1},{y1},{x2},{y2}")

    print()
    print("── Интеграция ──")
    print("  Паттерн замены в whoishere.py:")
    print("    raw_faces = face_id.app.get(frame)")
    print("    ↓ заменить на:")
    print("    yunet_faces = yunet_detect(detector, frame)")
    print("    for face in yunet_faces:")
    print("        emb = face_id.get_embedding(crop)")
    print("  speedup: {:.1f}x".format(s_t_total / y_t_total if y_t_total > 0 else 0))


if __name__ == '__main__':
    main()
