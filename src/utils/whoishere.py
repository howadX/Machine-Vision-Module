#!/usr/bin/env python3
"""whoishere.py — визуальный контекст для LLM: один кадр с камеры → текстовый отчёт.
Вызов: python src/utils/whoishere.py
Вывод: plain text, key: value, без эмодзи — пригоден для парсинга LLM.
"""
import os, sys, glob, time, warnings, logging, pickle
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
        os.close(old_err_fd)

with suppress_all():
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'src', 'utils'))
from face_id import FaceID

# ── Конфигурация ──────────────────────────────────────────
GALLERY_DIR = os.path.join(ROOT, 'models', 'references')
THRESHOLD = 0.35
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}

EXPERIMENTAL_EMOTION_PATH = os.path.join(ROOT, 'models', 'emotion', 'kaggle_affectnet_linear_svc.pkl')
LGB_PATH = os.path.join(ROOT, 'models', 'emotion', 'lightgbm_emotion_regions.pkl')
HOLISTIC_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'holistic_landmarker.task')
DEFAULT_EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
RU_MAP = {
    'anger': 'Злость', 'disgust': 'Отвращение', 'fear': 'Страх',
    'happy': 'Радость', 'neutral': 'Нейтральная', 'sad': 'Грусть',
    'surprise': 'Удивление',
}


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


def compute_iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def holistic_face_bbox(face_landmarks, w, h):
    xs = [lm.x for lm in face_landmarks]
    ys = [lm.y for lm in face_landmarks]
    return [min(xs) * w, min(ys) * h, max(xs) * w, max(ys) * h]


def compute_emotion_features(face_landmarks, landmark_indices):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
    nose_tip = pts[4]
    left_eye = pts[33]
    right_eye = pts[263]
    interocular = np.linalg.norm(left_eye - right_eye)
    if interocular < 1e-6:
        interocular = 1.0
    pts = pts[landmark_indices]
    dists = np.linalg.norm(pts - nose_tip, axis=1) / interocular
    return dists.reshape(1, -1)


def load_emotion_model():
    """Загружает экспериментальную модель эмоций, затем старую Regions как fallback."""
    last_error = None
    for path in (EXPERIMENTAL_EMOTION_PATH, LGB_PATH):
        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                return data['model'], data['landmark_indices'], data.get('label_order', DEFAULT_EMOTION_LABELS)
            return data, list(range(getattr(data, 'n_features_in_', 478))), DEFAULT_EMOTION_LABELS
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"emotion model not loaded: {last_error}")


def main():
    sys.stdout.reconfigure(line_buffering=True)

    cam_path = find_camera()
    if cam_path is None:
        print("camera: none")
        print("error: camera not found")
        sys.exit(1)
    print(f"camera: {cam_path}")

    frame = capture_frame(cam_path)
    if frame is None:
        print("error: failed to capture frame")
        sys.exit(1)

    h, w = frame.shape[:2]

    with suppress_all():
        face_id = FaceID(use_gpu=True)

        mp_base = mp_python.BaseOptions(
            model_asset_path=HOLISTIC_MODEL,
            delegate=mp_python.BaseOptions.Delegate.CPU,
        )
        mp_opts = vision.HolisticLandmarkerOptions(
            base_options=mp_base,
            running_mode=vision.RunningMode.IMAGE,
        )
        holistic = vision.HolisticLandmarker.create_from_options(mp_opts)

        lgb_clf, lgb_indices, emotion_labels = load_emotion_model()

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

    with suppress_stderr():
        raw_faces = face_id.app.get(frame)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    with suppress_stderr():
        h_results = holistic.detect(mp_img)

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

        emotion_name = None
        emotion_conf = 0.0
        if h_results.face_landmarks:
            h_bbox = holistic_face_bbox(h_results.face_landmarks, w, h)
            iou = compute_iou(bbox, h_bbox)
            if iou > 0.3:
                try:
                    X = compute_emotion_features(h_results.face_landmarks, lgb_indices)
                    proba = lgb_clf.predict_proba(X)[0]
                    pred_idx = int(proba.argmax())
                    emotion_conf = float(proba.max())
                    emotion_en = str(emotion_labels[pred_idx])
                    emotion_name = RU_MAP.get(emotion_en, emotion_en)
                except Exception:
                    pass

        if emotion_name:
            print(f"  emotion: {emotion_name}")
            print(f"  emotion_confidence: {emotion_conf:.2f}")
        else:
            print(f"  emotion: none")

        print(f"  det_score: {det_score:.2f}")
        print(f"  bbox: {int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}")
        print(f"  size: {face_w}x{face_h}")
        print(f"  distance: {dist}")


if __name__ == '__main__':
    main()
