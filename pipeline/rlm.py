#!/usr/bin/env python3
"""
4_identification.py — Реалтайм-распознавание лиц и эмоций.

Определение личности (Face ID) и эмоций в реальном времени.
Без игровой механики тренировки. Максимальный FPS.

Управление: ctrl + c — выход.
"""
import os as _os, sys

# ── Предзагрузка CUDA-библиотек (onnxruntime-gpu) ──
def _preload_cuda_libs():
    """Загружает CUDA-библиотеки из nvidia-* pip-пакетов ДО импорта onnxruntime."""
    import ctypes, glob as _glob
    venv_root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    nvidia_base = _os.path.join(venv_root, "venv", "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")
    if not _os.path.isdir(nvidia_base):
        return
    libs = []
    for so in _glob.iglob(_os.path.join(nvidia_base, "**", "lib", "*.so*"), recursive=True):
        libs.append(so)
    # Приоритет: сначала cublas, cudnn, cufft, curand...
    def _key(path):
        name = _os.path.basename(path).lower()
        for i, p in enumerate(["cublas", "cudnn", "cufft", "curand", "cusolver", "cusparse", "cuda_runtime"]):
            if p in name:
                return i
        return 99
    libs.sort(key=_key)
    for lib in libs:
        try:
            ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

_preload_cuda_libs()

# Подавление warnings ДО импортов
_os.environ["GRPC_VERBOSITY"] = "ERROR"
_os.environ["GLOG_minloglevel"] = "2"
_os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
_os.environ["QT_LOGGING_RULES"] = "*.debug=false"
_os.environ["QT_QPA_PLATFORM"] = "xcb"
_os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
_os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"
_os.environ["EGL_LOG_LEVEL"] = "fatal"

# Перенаправляем stderr на время шумных импортов
import io as _io
_stderr_backup = sys.stderr
sys.stderr = _io.StringIO()

import warnings
warnings.filterwarnings("ignore")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.use_absl_handler()

import glob, cv2, mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from PIL import ImageFont, ImageDraw, Image
import numpy as np
sys.stderr = _stderr_backup  # восстанавливаем stderr
import sys as _sys, select, tty, termios
import pickle
import time

# ── Пути ─────────────────────────────────────────────
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
_ROOT = _os.path.dirname(_SCRIPT_DIR)
_sys.path.insert(0, _os.path.join(_ROOT, 'core'))
from face_id import FaceID

_MODEL_PATH = _os.path.join(_ROOT, 'models', 'mediapipe', 'holistic_landmarker.task')
REFERENCES_DIR = _os.path.join(_ROOT, 'models', 'references')
FONT_PATH = _os.path.join(_ROOT, 'assets', 'fonts', 'DejaVuSans.ttf')

# ── Конфигурация ─────────────────────────────────────
DETECT_EVERY_N = 2       # SCRFD + ArcFace — каждый N-й кадр
FACEID_EVERY_N = 10       # Face ID сверка — каждый N-й кадр (тяжелее)
EMOTION_EVERY_N = 5      # эмоции — каждый кадр (LightGBM ~0.05ms)
FACEID_THRESHOLD = 0.35
DRAW_BBOX = True          # отрисовка bounding box
DRAW_LANDMARKS = True    # отрисовка контуров лица (дорого!)
DRAW_POSE = True         # отрисовка скелета тела

# ── Утилиты ──────────────────────────────────────────

def find_camera():
    """Ищет первую рабочую камеру через /dev/video* + CAP_V4L2."""
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
                        return dev
                cam.release()
    return None

# Клавиатура
_term_fd = _sys.stdin.fileno()
_term_old = termios.tcgetattr(_term_fd)

def kb_init():
    tty.setcbreak(_term_fd)

def kb_restore():
    termios.tcsetattr(_term_fd, termios.TCSADRAIN, _term_old)

def kb_hit():
    return select.select([_sys.stdin], [], [], 0)[0]

def kb_get():
    return _sys.stdin.read(1) if kb_hit() else None

def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    return cv2.resize(image, dim, interpolation=inter)

# ── Геометрические признаки ──────────────────────────

def compute_geometric_features(face_landmarks, landmark_indices=None):
    """Расстояния от кончика носа (landmark 4) до выбранных точек,
    нормированные на межзрачковое расстояние (33 ↔ 263)."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
    nose_tip = pts[4]
    left_eye = pts[33]
    right_eye = pts[263]
    interocular = np.linalg.norm(left_eye - right_eye)
    if interocular < 1e-6:
        interocular = 1.0
    if landmark_indices is not None:
        pts = pts[landmark_indices]
    dists = np.linalg.norm(pts - nose_tip, axis=1) / interocular
    return dists.reshape(1, -1)

# ── Инициализация ────────────────────────────────────

# Камера
cam_path = find_camera()
if cam_path is None:
    print("❌ Камера не найдена. Проверь: ls /dev/video*")
    exit(1)
video_capture = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
video_capture.set(3, 640)
video_capture.set(4, 480)

# Face ID (SCRFD + ArcFace)
face_id_detector = FaceID()

# MediaPipe Holistic Landmarker (face + body + hands, Tasks API)
_mp_base = mp_python.BaseOptions(model_asset_path=_MODEL_PATH)
_mp_opts = vision.HolisticLandmarkerOptions(
    base_options=_mp_base,
    running_mode=vision.RunningMode.IMAGE,
)
holistic_landmarker = vision.HolisticLandmarker.create_from_options(_mp_opts)

# Галерея эталонов (для добавления: положить фото в models/references/p_Имя.jpg)
IMG_NAMES = ['p_V.jpg']
NAMES = ['Вадим']
gallery = {}
for img_name, name in zip(IMG_NAMES, NAMES):
    img = cv2.imread(_os.path.join(REFERENCES_DIR, img_name))
    emb = face_id_detector.get_embedding(img)
    if emb is not None:
        gallery[name] = emb
        print(f"  [Галерея] {name}: эмбеддинг {emb.shape}")
    else:
        print(f"  [Галерея] {name}: лицо не найдено")

# LightGBM/экспериментальный классификатор эмоций
use_lgb = False
lgb_clf = None
lgb_landmark_indices = None
lgb_label_order = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Цепочка моделей: от новой к старой
_MODEL_CHAIN = [
    ("kaggle_affectnet_linear_svc.pkl", "Kaggle AffectNet LinearSVC", "dict"),
    ("lightgbm_emotion_regions.pkl", "LightGBM Regions", "dict"),
    ("lightgbm_affectnet_7cls.pkl", "LightGBM AffectNet 7cls", "raw"),
    ("lightgbm_geometric.pkl", "LightGBM 468", "raw"),
]

for model_file, model_name, model_type in _MODEL_CHAIN:
    try:
        _lgb_path = _os.path.join(_ROOT, 'models', 'emotion', 'fallback', model_file)
        if not _os.path.exists(_lgb_path):
            _lgb_path = _os.path.join(_ROOT, 'models', 'emotion', model_file)
        if model_type == "dict":
            _lgb_data = pickle.load(open(_lgb_path, 'rb'))
            lgb_clf = _lgb_data['model']
            lgb_landmark_indices = _lgb_data['landmark_indices']
            lgb_label_order = _lgb_data.get('label_order', lgb_label_order)
            print(f"[Эмоции] {model_name} ({_lgb_data.get('num_features', '?')} признаков) загружен")
        else:
            lgb_clf = pickle.load(open(_lgb_path, 'rb'))
            lgb_landmark_indices = list(range(getattr(lgb_clf, 'n_features_in_', 468)))
            print(f"[Эмоции] {model_name} ({getattr(lgb_clf, 'n_features_in_', '?')} признаков) загружен")
        use_lgb = True
        break
    except Exception as e:
        print(f"[Эмоции] {model_name}: {e}")

if not use_lgb:
    try:
        from emotion import EmotionClassifier
        ec = EmotionClassifier(use_gpu=True)
        print("[Эмоции] Fallback на EfficientNet")
    except Exception as e:
        print(f"[Эмоции] EfficientNet не загружен: {e}")
        print("[Эмоции] Эмоции отключены")
        use_lgb = False  # остаёмся без эмоций

# Модель Б: позы тела (33 Holistic точки → standing/sitting/lying)
body_clf = None
try:
    _body_path = _os.path.join(_ROOT, 'models', 'emotion', 'lightgbm_body_holistic_33.pkl')
    body_clf = pickle.load(open(_body_path, 'rb'))
    print("[Тело] LightGBM Holistic 44 признака (71.7% на COCO) загружен")
except Exception as e:
    print(f"[Тело] Модель не загружена: {e}")

# Модель В: VAE + PoseClassifier (новая)
import torch
import torch.nn as nn
_use_vae = False
_vae_model = None
_vae_clf = None
_vae_class_names = None
_vae_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
try:
    from pose_vae import PoseVAE
    from pose_heads import PoseClassifier
    import json as _j
    _vae_ckpt = torch.load(_os.path.join(_ROOT, 'models', 'emotion', 'vposer_coco33.pth'),
                            map_location=_vae_device, weights_only=True)
    _vae_cfg = _vae_ckpt['config']
    _vae_model = PoseVAE(
        input_dim=_vae_cfg['input_dim'], latent_dim=_vae_cfg['latent_dim'],
        hidden_dims=tuple(_vae_cfg['hidden_dims']), dropout=_vae_cfg['dropout'],
    ).to(_vae_device)
    _vae_model.load_state_dict(_vae_ckpt['model_state_dict'])
    _vae_model.eval()

    with open(_os.path.join(_ROOT, 'models', 'emotion', 'pose_classifier_config.json')) as f:
        _clf_cfg = _j.load(f)
    _vae_clf = PoseClassifier(
        latent_dim=_clf_cfg['latent_dim'], num_classes=_clf_cfg['num_classes'],
        hidden_dim=_clf_cfg['hidden_dim'], dropout=_clf_cfg['dropout'],
    ).to(_vae_device)
    _clf_ckpt = torch.load(_os.path.join(_ROOT, 'models', 'emotion', 'pose_classifier.pth'),
                            map_location=_vae_device, weights_only=True)
    _vae_clf.load_state_dict(_clf_ckpt['model_state_dict'])
    _vae_clf.eval()
    _vae_class_names = _clf_cfg['class_names_ru']
    _use_vae = True
    print(f"[Тело] VAE + PoseClassifier ({_clf_cfg['num_classes']} классов, {_clf_cfg['test_accuracy']:.1%}) загружен")
    print(f"       Классы: {', '.join(_vae_class_names)}")
except Exception as e:
    print(f"[Тело] VAE не загружен: {e}")

POSTURE_NAMES = ["standing", "sitting", "lying"]
POSTURE_RU = {"standing": "стоит", "sitting": "сидит", "lying": "лежит"}

# Маппинги
_LGB_IDX_TO_EN = ["anger", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
_ALL_LGB_RU = {
    "anger": "Злость", "angry": "Злость",
    "disgust": "Отвращение", "fear": "Страх",
    "happy": "Радость", "neutral": "Нейтральная",
    "sad": "Грусть", "surprise": "Удивление",
}
EMOTIONS_RU = ['Злость', 'Отвращение', 'Страх', 'Радость', 'Нейтральная', 'Грусть', 'Удивление']

# Шрифты
font_name = ImageFont.truetype(FONT_PATH, 22) if _os.path.exists(FONT_PATH) else ImageFont.load_default()
font_emotion = ImageFont.truetype(FONT_PATH, 22) if _os.path.exists(FONT_PATH) else ImageFont.load_default()
font_fps = ImageFont.truetype(FONT_PATH, 14) if _os.path.exists(FONT_PATH) else ImageFont.load_default()

# ── Главный цикл ─────────────────────────────────────

print("\n[Запуск] Q — выход.\n")
kb_init()

frame_count = 0
last_frame_time = time.perf_counter()
fps_val = 0.0
current_name = None
current_emotion = "—"
emotion_conf = 0.0
body_status = "тело: нет"
_vae_status = ""
last_detections = []  # кеш bbox между детекциями

while True:
    ret, frame = video_capture.read()
    if not ret:
        continue

    face_embedding = None

    # ── Детекция (каждый DETECT_EVERY_N кадр) ──────
    if frame_count % DETECT_EVERY_N == 0:
        faces = face_id_detector.app.get(frame)
        if faces:
            best = max(faces, key=lambda f: f.get("det_score", 0))
            bbox = best["bbox"]
            face_embedding = np.asarray(best.get("embedding"), dtype=np.float32)
            last_detections = [(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))]

    # ── На каждый кадр с лицом ────────────────────
    if last_detections:
        d = last_detections[0]
        if DRAW_BBOX:
            cv2.rectangle(frame, (d[0], d[1]), (d[2], d[3]), (0, 255, 0), 1)

        # MediaPipe Holistic (face + body + hands — Tasks API)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = holistic_landmarker.detect(mp_img)

        # ── Модель А: ЛИЦО → эмоции ────────────────
        if results.face_landmarks:
            if DRAW_LANDMARKS:
                vision.drawing_utils.draw_landmarks(
                    frame, results.face_landmarks,
                    vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=vision.drawing_utils.DrawingSpec(
                        color=(255, 255, 255), thickness=1))

            if use_lgb:  # LightGBM эмоции
                face_pts = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks], dtype=np.float32)
                nose_tip = face_pts[4]
                left_eye = face_pts[33]
                right_eye = face_pts[263]
                interocular = np.linalg.norm(left_eye - right_eye)
                if interocular < 1e-6: interocular = 1.0
                if lgb_landmark_indices is not None:
                    face_pts = face_pts[lgb_landmark_indices]
                X = np.linalg.norm(face_pts - nose_tip, axis=1) / interocular
                X = X.reshape(1, -1)

                proba = lgb_clf.predict_proba(X)[0]
                pred_idx = int(proba.argmax())
                emotion_conf = float(proba.max())
                try:
                    name_en = str(lgb_label_order[pred_idx])
                except (IndexError, ValueError):
                    name_en = str(pred_idx)
                current_emotion = _ALL_LGB_RU.get(name_en, name_en)

        # ── Модель Б: ТЕЛО → поза (2 этапа) ──────
        if results.pose_landmarks:
            pose_lm = results.pose_landmarks
            # Этап 1: эвристика lying (foot/head > 1.15 → стопы дальше от таза = лежит)
            pts_xy = np.array([[lm.x, lm.y] for lm in pose_lm])
            hip_center = (pts_xy[23] + pts_xy[24]) / 2
            shoulder_w = np.linalg.norm(pts_xy[11] - pts_xy[12]) or 1.0
            d_head = np.linalg.norm(pts_xy[0] - hip_center) / shoulder_w  # нос
            d_feet = (np.linalg.norm(pts_xy[27] - hip_center) + np.linalg.norm(pts_xy[28] - hip_center)) / 2 / shoulder_w
            
            foot_head_ratio = d_feet / max(d_head, 1e-6)
            
            if foot_head_ratio > 1.25:
                body_posture = "lying"
            elif body_clf is not None:
                # Этап 2: ML для standing vs sitting
                pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_lm], dtype=np.float64)
                d = np.linalg.norm(pts[:, :2] - hip_center, axis=1) / max(shoulder_w, 1e-6)
                extra = [
                    d[11]/max(d[23],1e-6), d[12]/max(d[24],1e-6),
                    (d[11]+d[12])/max(d[23]+d[24],1e-6),
                    d[25]/max(d[26],1e-6), d[27]/max(d[28],1e-6),
                    d[13]/max(d[14],1e-6), d[15]/max(d[16],1e-6),
                    d[0]/max((d[23]+d[24])/2,1e-6),
                    (d[27]+d[28])/2 / max(d[0],1e-6),
                    abs(d[27]-d[28])/max(d[23]+d[24],1e-6),
                    (d[29]+d[30])/max(d[27]+d[28],1e-6),
                ]
                X = np.concatenate([d, extra] + [[d[11]/max(d[23],1e-6)], [d[12]/max(d[24],1e-6)], [foot_head_ratio]]).reshape(1, -1)
                X = X[:, :38]  # модель на 38 признаках
                pred = int(body_clf.predict(X)[0])
                body_posture = POSTURE_NAMES[pred] if pred <= 1 else "sitting"
            else:
                body_posture = "upright"
            
            body_status = f"тело: {POSTURE_RU.get(body_posture, body_posture)}"
            # ── VAE классификация (новая модель) ──
            _vae_status = ""
            if _use_vae:
                try:
                    from pose_encoder import normalize_landmarks
                    _pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_lm], dtype=np.float32)
                    _norm, _ = normalize_landmarks(_pts)
                    _flat = _norm.reshape(1, -1).astype(np.float32)
                    _x = torch.from_numpy(_flat).to(_vae_device)
                    with torch.no_grad():
                        _z = _vae_model.encode_to_z(_x)
                        _logits = _vae_clf(_z)
                        _probs = torch.softmax(_logits, dim=1)[0].cpu().numpy()
                        _recon, _, _ = _vae_model(_x)
                        _recon_err = float(nn.functional.mse_loss(_recon, _x).item())
                        _logpz = float(_vae_model.log_prob_z(_x).item())
                    _pred_idx = int(_probs.argmax())
                    _vae_conf = float(_probs.max())
                    _vae_name = _vae_class_names[_pred_idx] if _pred_idx < len(_vae_class_names) else f'c{_pred_idx}'
                    _plaus = 'ok' if _logpz < 14.3 else 'NEW!'
                    _vae_status = f"VAE: {_vae_name} ({_vae_conf:.0%}) err={_recon_err:.3f} {_plaus}"
                except Exception:
                    _vae_status = "VAE: err"
            if DRAW_POSE:
                vision.drawing_utils.draw_landmarks(
                    frame, pose_lm,
                    vision.PoseLandmarksConnections.POSE_LANDMARKS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=vision.drawing_utils.DrawingSpec(
                        color=(0, 255, 0), thickness=2))
        else:
            body_status = "тело: нет"

        # ── Face ID (каждый FACEID_EVERY_N кадр) ──
        if frame_count % FACEID_EVERY_N == 0 and face_embedding is not None:
            emb_norm = face_embedding / np.linalg.norm(face_embedding)
            best_sim = 0.0
            best_name = None
            for name, ref_emb in gallery.items():
                ref_norm = ref_emb / np.linalg.norm(ref_emb)
                sim = float(np.dot(emb_norm, ref_norm))
                if sim > best_sim:
                    best_sim = sim
                    best_name = name
            current_name = best_name if best_sim > FACEID_THRESHOLD else None

    else:
        current_name = None
        current_emotion = "—"
        body_status = "тело: нет"
        _vae_status = ""

    # ── Оверлей (PIL, кириллица) ──────────────────
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    if current_name:
        draw.text((10, 10), current_name, font=font_name, fill=(0, 255, 0))
    if current_emotion != "—":
        emo_text = f"{current_emotion} ({emotion_conf:.0%})"
        draw.text((10, 40), emo_text, font=font_emotion, fill=(0, 200, 255))
    draw.text((10, 70), body_status, font=font_fps, fill=(200, 200, 100))
    if _vae_status:
        draw.text((10, 90), _vae_status, font=font_fps, fill=(100, 255, 150))

    # FPS (мгновенный + сглаженный)
    now = time.perf_counter()
    delta = now - last_frame_time
    last_frame_time = now
    instant_fps = 1.0 / delta if delta > 0 else 0
    fps_val = 0.9 * fps_val + 0.1 * instant_fps if fps_val > 0 else instant_fps
    draw.text((10, 120), f"FPS: {fps_val:.0f}", font=font_fps, fill=(200, 200, 200))

    frame = np.array(img_pil)

    # ── Отрисовка ─────────────────────────────────
    cv2.imshow('MVM', image_resize(frame, height=540))
    cv2.moveWindow('MVM', 100, 100)

    # ── Клавиатура ────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or (key == -1 and kb_hit() and kb_get() == 'q'):
        print("\n[Выход] Q нажата.")
        break

    frame_count += 1

# ── Очистка ──────────────────────────────────────────
video_capture.release()
kb_restore()
cv2.destroyAllWindows()
print(f"[Стоп] Кадров: {frame_count}")
