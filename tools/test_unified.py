#!/usr/bin/env python3
"""test_unified.py — прототип unified-архитектуры MVM.

Одна модель вместо двух:
  FaceLandmarker (full frame, 3ms) → bbox + 468 landmarks
    ├─ 468 landmarks → MLP → emotion         (прямые landmarks, без повторного вызова)
    ├─ bbox → crop → ArcFace → identity       (GPU, 2.3ms)
    └─ bbox → crop_body → PoseLandmarker → VAE → posture

Сравнение с текущим sight.py: SCRFD → FaceLandmarker per crop → MLP
"""
import os, sys, glob, time, warnings, logging, pickle, json
from contextlib import contextmanager

# ── Подавление шума ──────────────────────────────────────
os.environ.setdefault('ONNXRUNTIME_LOG_LEVEL', '3')
os.environ.setdefault('GRPC_VERBOSITY', 'ERROR')
os.environ.setdefault('GLOG_minloglevel', '2')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'ERROR')
os.environ.setdefault('OPENCV_VIDEOIO_DEBUG', '0')
warnings.filterwarnings('ignore')

@contextmanager
def suppress_all():
    for name in ('insightface', 'onnxruntime', 'absl'):
        logging.getLogger(name).setLevel(logging.ERROR)
    old_out_fd = os.dup(1); old_err_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1); os.dup2(devnull, 2)
    try: yield
    finally:
        sys.stdout.flush(); sys.stderr.flush()
        os.dup2(old_out_fd, 1); os.dup2(old_err_fd, 2)
        os.close(devnull); os.close(old_out_fd); os.close(old_err_fd)

@contextmanager
def suppress_stderr():
    old_err_fd = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try: yield
    finally: os.dup2(old_err_fd, 2); os.close(devnull)

with suppress_all():
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import torch
    import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'core'))
from face_id import FaceID
from pose_vae import PoseVAE
from pose_heads import PoseClassifier
from pose_encoder import normalize_landmarks

# ── Конфигурация ──────────────────────────────────────────
GALLERY_DIR = os.path.join(ROOT, 'models', 'references')
THRESHOLD = 0.35
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}

MLP_MODEL = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion.pth')
MLP_SCALER = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion_scaler.pkl')
MLP_CONFIG = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion_config.json')

VAE_PATH = os.path.join(ROOT, 'models', 'emotion', 'vposer_coco33.pth')
CLF_PATH = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier.pth')
CLF_CONFIG = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier_config.json')

FACE_LM_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'face_landmarker.task')
POSE_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'pose_landmarker.task')

with open(CLF_CONFIG) as f:
    _clf_cfg = json.load(f)
POSTURE_CLASSES_RU = _clf_cfg['class_names_ru']

CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
RU_MAP = {
    'anger': 'Злость', 'disgust': 'Отвращение', 'fear': 'Страх',
    'happy': 'Радость', 'neutral': 'Нейтральная', 'sad': 'Грусть', 'surprise': 'Удивление',
}

# ── MLP эмоций ────────────────────────────────────────────
class EmotionMLP(nn.Module):
    def __init__(self, in_dim=468, num_classes=7, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_classes),
        )
    def forward(self, x): return self.net(x)


# ── Камера ────────────────────────────────────────────────
def find_camera():
    devices = sorted(glob.glob('/dev/video*'), key=lambda x: int(x.replace('/dev/video','')))
    for dev in devices:
        idx = int(dev.replace('/dev/video',''))
        for method in ['path','index']:
            cap = cv2.VideoCapture(dev if method=='path' else idx, cv2.CAP_V4L2)
            if cap.isOpened():
                for _ in range(5):
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.mean()>10:
                        cap.release(); return dev
                cap.release()
    return None

def capture_frame(cam_path):
    cam_idx = int(cam_path.replace('/dev/video',''))
    for _ in range(3):
        cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not cap.isOpened(): cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        for __ in range(5): ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.mean()>10: return frame
        time.sleep(0.3)
    return None


# ── Кропы ─────────────────────────────────────────────────
def crop_body(frame, bbox, side_ratio=0.3, up_ratio=0.2, down_ratio=2.5):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h,w = frame.shape[:2]
    fw, fh = x2-x1, y2-y1
    px, pu, pd = int(fw*side_ratio), int(fh*up_ratio), int(fh*down_ratio)
    x1=max(0,x1-px); y1=max(0,y1-pu)
    x2=min(w,x2+px); y2=min(h,y2+pd)
    if x2<=x1 or y2<=y1: return None
    return frame[y1:y2, x1:x2]


# ── Сцена ─────────────────────────────────────────────────
def estimate_distance(bbox, frame_shape):
    x1,y1,x2,y2 = bbox
    ratio = ((x2-x1)*(y2-y1))/(frame_shape[0]*frame_shape[1])
    if ratio>0.12: return 'near'
    elif ratio>0.04: return 'medium'
    return 'far'

def estimate_lighting(frame):
    m = float(frame.mean())
    if m>150: return 'bright',m
    elif m>70: return 'normal',m
    return 'dim',m


# ── Эмоции ────────────────────────────────────────────────
def compute_features_468(face_landmarks):
    """468 нормированных расстояний от кончика носа."""
    pts = np.array([[lm.x,lm.y,lm.z] for lm in face_landmarks], dtype=np.float32)
    nose_tip=pts[4]; left_eye=pts[33]; right_eye=pts[263]
    interocular = np.linalg.norm(left_eye-right_eye)
    if interocular<1e-6: interocular=1.0
    return (np.linalg.norm(pts-nose_tip, axis=1)/interocular)[:468]

def classify_emotion_mlp(face_landmarks, model, scaler, device):
    feats = compute_features_468(face_landmarks).reshape(1, -1)
    feats_scaled = scaler.transform(feats).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(feats_scaled).to(device)
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(proba.argmax())
    return RU_MAP.get(CLASSES[pred_idx], CLASSES[pred_idx]), round(float(proba.max()), 2)


# ── Поза ──────────────────────────────────────────────────
def load_vae_pose(device):
    vae_ckpt = torch.load(VAE_PATH, map_location=device, weights_only=True)
    vae_cfg = vae_ckpt['config']
    vae = PoseVAE(
        input_dim=vae_cfg['input_dim'], latent_dim=vae_cfg['latent_dim'],
        hidden_dims=tuple(vae_cfg['hidden_dims']), dropout=vae_cfg['dropout'],
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    with open(CLF_CONFIG) as f: clf_cfg = json.load(f)
    clf = PoseClassifier(
        latent_dim=clf_cfg['latent_dim'], num_classes=clf_cfg['num_classes'],
        hidden_dim=clf_cfg['hidden_dim'], dropout=clf_cfg['dropout'],
    ).to(device)
    clf_ckpt = torch.load(CLF_PATH, map_location=device, weights_only=True)
    clf.load_state_dict(clf_ckpt['model_state_dict'])
    clf.eval()
    return vae, clf, clf_cfg['class_names_ru']

def classify_posture_vae(pose_landmarks, vae, clf, class_names, device):
    pts = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks], dtype=np.float32)
    norm, _meta = normalize_landmarks(pts)
    flat = norm.reshape(1, -1).astype(np.float32)
    x = torch.from_numpy(flat).to(device)
    with torch.no_grad():
        z = vae.encode_to_z(x)
        logits = clf(z)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        recon, _, _ = vae(x)
        recon_err = float(torch.nn.functional.mse_loss(recon, x).item())
        logpz = float(vae.log_prob_z(x).item())
    pred_idx = int(probs.argmax())
    conf = float(probs.max())
    name = class_names[pred_idx] if pred_idx < len(class_names) else f'class_{pred_idx}'
    return name, conf, recon_err, logpz


# ── PoseLandmarker на кропе тела ──────────────────────────
def run_pose_landmarker(pose_lm, bgr_crop):
    if bgr_crop is None: return None
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with suppress_stderr():
        res = pose_lm.detect(mp_img)
    if res.pose_landmarks:
        return res.pose_landmarks[0]
    return None


# ── Извлечение 5 ключевых точек ArcFace из 468 landmarks ──
# InsightFace ожидает: reye, leye, nose, rmouth, lmouth
FACELM_TO_ARCFACE_KPS = {
    'reye': 33,    # right eye center (из 468)
    'leye': 263,   # left eye center
    'nose': 4,     # nose tip
    'rmouth': 61,  # right mouth corner
    'lmouth': 291, # left mouth corner
}

class FaceStub:
    def __init__(self, kps):
        self.kps = kps


def landmarks_to_arcface_kps(face_landmarks, w, h):
    """468 NormalizedLandmark → 5 точек (reye, leye, nose, rmouth, lmouth) в пикселях."""
    pts = {}
    for name, idx in FACELM_TO_ARCFACE_KPS.items():
        lm = face_landmarks[idx]
        pts[name] = [lm.x * w, lm.y * h]
    kps = np.array([
        pts['reye'], pts['leye'], pts['nose'], pts['rmouth'], pts['lmouth']
    ], dtype=np.float32)
    return kps


def landmarks_to_bbox(face_landmarks, w, h, pad_ratio=0.2):
    """468 NormalizedLandmark → bbox [x1, y1, x2, y2] с запасом (как SCRFD).

    FaceLandmarker даёт landmarks без запаса — добавляем pad_ratio как crop_face(),
    чтобы bbox покрывал лоб/подбородок/щёки для crop_body.
    """
    xs = [lm.x * w for lm in face_landmarks]
    ys = [lm.y * h for lm in face_landmarks]
    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)
    fw, fh = x2 - x1, y2 - y1
    px, py = fw * pad_ratio, fh * pad_ratio
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(w, x2 + px)
    y2 = min(h, y2 + py)
    return [x1, y1, x2, y2]


# ═══════════════════════════════════════════════════════════
#  UNIFIED PIPELINE
# ═══════════════════════════════════════════════════════════
def run_unified_pipeline(frame, face_lm, pose_lm, face_id, gallery, mlp, scaler, vae, pose_clf, pose_names_ru, device):
    """FaceLandmarker на полном кадре → все модальности из одной детекции."""
    h, w = frame.shape[:2]
    t_total_start = time.perf_counter()

    # ── 1. FaceLandmarker на полном кадре ──
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    t0 = time.perf_counter()
    with suppress_stderr():
        flm_result = face_lm.detect(mp_img)
    t_detect = time.perf_counter() - t0

    faces_out = []
    t_emotion_total = 0.0
    t_identity_total = 0.0
    t_posture_total = 0.0

    if not flm_result.face_landmarks:
        return faces_out, t_detect, 0, 0, 0, time.perf_counter() - t_total_start

    rec = face_id.app.models['recognition']

    for i, landmarks in enumerate(flm_result.face_landmarks, 1):
        bbox = landmarks_to_bbox(landmarks, w, h)
        dist = estimate_distance(bbox, frame.shape)
        face_w = int(bbox[2] - bbox[0])
        face_h = int(bbox[3] - bbox[1])

        face_info = {
            'index': i, 'identity': 'unknown', 'confidence': 0.0,
            'bbox': [int(v) for v in bbox],
            'size': f'{face_w}x{face_h}', 'distance': dist,
            'emotion': None, 'emotion_confidence': None,
            'posture': None, 'posture_confidence': None,
            'recon_error': None, 'plausibility': None,
        }

        # ── 2. Эмоция: ПРЯМО из landmarks (без повторного FaceLandmarker!) ──
        t_emo_start = time.perf_counter()
        try:
            emo_name, emo_conf = classify_emotion_mlp(landmarks, mlp, scaler, device)
            face_info['emotion'] = emo_name
            face_info['emotion_confidence'] = emo_conf
        except Exception as e:
            pass
        t_emotion_total += time.perf_counter() - t_emo_start

        # ── 3. Identity: ArcFace через rec.get ──
        t_id_start = time.perf_counter()
        try:
            kps = landmarks_to_arcface_kps(landmarks, w, h)
            stub = FaceStub(kps)
            emb = rec.get(frame, stub)  # alignment + embedding на GPU
            if gallery:
                name, sim = face_id.identify(emb, gallery, THRESHOLD)
                face_info['identity'] = name if name else 'unknown'
                face_info['confidence'] = round(sim, 2)
        except Exception as e:
            pass
        t_identity_total += time.perf_counter() - t_id_start

        # ── 4. Поза: crop_body → PoseLandmarker → VAE ──
        t_pos_start = time.perf_counter()
        body_crop = crop_body(frame, bbox)
        pose_lm_result = run_pose_landmarker(pose_lm, body_crop)
        if pose_lm_result:
            try:
                pos_name, pos_conf, recon_err, logpz = classify_posture_vae(
                    pose_lm_result, vae, pose_clf, pose_names_ru, device
                )
                face_info['posture'] = pos_name
                face_info['posture_confidence'] = round(pos_conf, 2)
                face_info['recon_error'] = round(recon_err, 4)
                face_info['plausibility'] = 'normal' if logpz < 14.3 else 'unusual'
            except Exception:
                pass
        t_posture_total += time.perf_counter() - t_pos_start

        faces_out.append(face_info)

    t_total = time.perf_counter() - t_total_start
    return faces_out, t_detect, t_emotion_total, t_identity_total, t_posture_total, t_total


# ═══════════════════════════════════════════════════════════
#  CURRENT PIPELINE (sight.py)
# ═══════════════════════════════════════════════════════════
def run_current_pipeline(frame, face_lm_per_crop, pose_lm, face_id, gallery, mlp, scaler, vae, pose_clf, pose_names_ru, device):
    """Текущий подход: SCRFD + FaceLandmarker на каждом кропе."""
    h, w = frame.shape[:2]
    t_total_start = time.perf_counter()

    # ── 1. SCRFD ──
    t0 = time.perf_counter()
    with suppress_stderr():
        raw_faces = face_id.app.get(frame)
    t_detect = time.perf_counter() - t0

    faces_out = []
    t_emotion_total = 0.0
    t_posture_total = 0.0

    if not raw_faces:
        return faces_out, t_detect, 0, 0, time.perf_counter() - t_total_start

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

        face_info = {
            'index': i, 'identity': name if name else 'unknown',
            'confidence': round(sim, 2), 'det_score': round(det_score, 2),
            'bbox': [int(v) for v in bbox], 'size': f'{face_w}x{face_h}', 'distance': dist,
            'emotion': None, 'emotion_confidence': None,
            'posture': None, 'posture_confidence': None,
            'recon_error': None, 'plausibility': None,
        }

        # Кроп лица → FaceLandmarker (повторный вызов!) → MLP → эмоция
        t_emo_start = time.perf_counter()
        face_crop = cv2.cvtColor(
            frame[max(0,int(bbox[1])):min(h,int(bbox[3])), max(0,int(bbox[0])):min(w,int(bbox[2]))],
            cv2.COLOR_BGR2RGB
        )
        if face_crop.size > 0:
            try:
                mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_crop)
                with suppress_stderr():
                    flm_res = face_lm_per_crop.detect(mp_crop)
                if flm_res.face_landmarks:
                    emo_name, emo_conf = classify_emotion_mlp(
                        flm_res.face_landmarks[0], mlp, scaler, device
                    )
                    face_info['emotion'] = emo_name
                    face_info['emotion_confidence'] = emo_conf
            except Exception:
                pass
        t_emotion_total += time.perf_counter() - t_emo_start

        # Кроп тела → PoseLandmarker → VAE → поза
        t_pos_start = time.perf_counter()
        body_crop = crop_body(frame, bbox)
        pose_lm_result = run_pose_landmarker(pose_lm, body_crop)
        if pose_lm_result:
            try:
                pos_name, pos_conf, recon_err, logpz = classify_posture_vae(
                    pose_lm_result, vae, pose_clf, pose_names_ru, device
                )
                face_info['posture'] = pos_name
                face_info['posture_confidence'] = round(pos_conf, 2)
                face_info['recon_error'] = round(recon_err, 4)
                face_info['plausibility'] = 'normal' if logpz < 14.3 else 'unusual'
            except Exception:
                pass
        t_posture_total += time.perf_counter() - t_pos_start

        faces_out.append(face_info)

    t_total = time.perf_counter() - t_total_start
    return faces_out, t_detect, t_emotion_total, t_posture_total, t_total


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    sys.stdout.reconfigure(line_buffering=True)

    # Камера
    cam_path = find_camera()
    if cam_path is None:
        print("camera: none\nerror: camera not found"); return
    print(f"camera: {cam_path}")

    frame = capture_frame(cam_path)
    if frame is None:
        print("error: failed to capture frame"); return
    h, w = frame.shape[:2]
    lighting_level, lighting_val = estimate_lighting(frame)
    print(f"frame: {w}x{h}  lighting: {lighting_level} ({lighting_val:.0f})")

    # ── Инициализация моделей ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with suppress_all():
        face_id = FaceID(use_gpu=True)

        # FaceLandmarker для полного кадра (CPU delegate)
        full_frame_lm_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=FACE_LM_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=5,
            running_mode=vision.RunningMode.IMAGE,
        )
        face_lm_full = vision.FaceLandmarker.create_from_options(full_frame_lm_opts)

        # FaceLandmarker для кропов (в текущем пайплайне)
        per_crop_lm_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=FACE_LM_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE,
        )
        face_lm_per_crop = vision.FaceLandmarker.create_from_options(per_crop_lm_opts)

        # PoseLandmarker
        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=POSE_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=False,
        )
        pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

        # VAE + PoseClassifier
        vae, pose_clf, pose_names_ru = load_vae_pose(device)

        # MLP эмоции
        mlp = EmotionMLP()
        mlp.load_state_dict(torch.load(MLP_MODEL, map_location=device, weights_only=True))
        mlp.to(device); mlp.eval()
        with open(MLP_SCALER, 'rb') as f:
            scaler = pickle.load(f)

    # Галерея
    gallery = {}
    for fname in sorted(os.listdir(GALLERY_DIR)):
        if not fname.endswith('.jpg'): continue
        name_key = fname[2:-4]
        img = cv2.imread(os.path.join(GALLERY_DIR, fname))
        if img is None: continue
        with suppress_all(): emb = face_id.get_embedding(img)
        if emb is not None: gallery[NAME_MAP.get(name_key, name_key)] = emb

    # Прогрев всех моделей
    print("warming up...")
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    _ = face_lm_full.detect(mp_img)
    _ = face_lm_per_crop.detect(mp_img)
    _ = face_id.app.get(frame)
    _ = pose_lm.detect(mp_img)

    # ── Бенчмарк: UNIFIED ──
    print("\n" + "=" * 60)
    print("  UNIFIED (FaceLandmarker full frame)")
    print("=" * 60)
    u_times = {'detect': [], 'emotion': [], 'identity': [], 'posture': [], 'total': []}
    u_faces = None
    for _ in range(5):
        faces, t_d, t_e, t_i, t_p, t_t = run_unified_pipeline(
            frame, face_lm_full, pose_lm, face_id, gallery,
            mlp, scaler, vae, pose_clf, pose_names_ru, device
        )
        u_faces = faces
        u_times['detect'].append(t_d)
        u_times['emotion'].append(t_e)
        u_times['identity'].append(t_i)
        u_times['posture'].append(t_p)
        u_times['total'].append(t_t)

    # ── Бенчмарк: CURRENT ──
    print("=" * 60)
    print("  CURRENT (SCRFD + FaceLandmarker per crop)")
    print("=" * 60)
    c_times = {'detect': [], 'emotion': [], 'posture': [], 'total': []}
    c_faces = None
    for _ in range(5):
        faces, t_d, t_e, t_p, t_t = run_current_pipeline(
            frame, face_lm_per_crop, pose_lm, face_id, gallery,
            mlp, scaler, vae, pose_clf, pose_names_ru, device
        )
        c_faces = faces
        c_times['detect'].append(t_d)
        c_times['emotion'].append(t_e)
        c_times['posture'].append(t_p)
        c_times['total'].append(t_t)

    # ── Сравнение ──
    print("\n" + "=" * 60)
    print("  СРАВНЕНИЕ (среднее 5 прогонов)")
    print("=" * 60)
    print(f"  {'':30} {'UNIFIED':>12} {'CURRENT':>12} {'Δ':>10}")
    print(f"  {'─' * 60}")
    print(f"  {'Детекция лиц:':30} {avg_ms(u_times['detect']):>9.1f}ms {avg_ms(c_times['detect']):>9.1f}ms {delta_pct(u_times['detect'], c_times['detect']):>8}")
    print(f"  {'Эмоции:':30} {avg_ms(u_times['emotion']):>9.1f}ms {avg_ms(c_times['emotion']):>9.1f}ms {delta_pct(u_times['emotion'], c_times['emotion']):>8}")
    print(f"  {'Identity:':30} {avg_ms(u_times['identity']):>9.1f}ms {'—':>12}")
    print(f"  {'Поза:':30} {avg_ms(u_times['posture']):>9.1f}ms {avg_ms(c_times['posture']):>9.1f}ms {delta_pct(u_times['posture'], c_times['posture']):>8}")
    print(f"  {'─' * 60}")
    print(f"  {'ИТОГО:':30} {avg_ms(u_times['total']):>9.1f}ms {avg_ms(c_times['total']):>9.1f}ms {delta_pct(u_times['total'], c_times['total']):>8}")

    # Совпадение результатов
    u_emos = {f['emotion'] for f in (u_faces or [])}
    c_emos = {f['emotion'] for f in (c_faces or [])}
    u_ids = {f['identity'] for f in (u_faces or [])}
    c_ids = {f['identity'] for f in (c_faces or [])}
    u_pos = {f['posture'] for f in (u_faces or [])}
    c_pos = {f['posture'] for f in (c_faces or [])}

    print(f"\n  Совпадение эмоций:    {'✓' if u_emos == c_emos else '✗ ' + str(u_emos) + ' vs ' + str(c_emos)}")
    print(f"  Совпадение identity:  {'✓' if u_ids == c_ids else '✗ ' + str(u_ids) + ' vs ' + str(c_ids)}")
    print(f"  Совпадение позы:      {'✓' if u_pos == c_pos else '✗ ' + str(u_pos) + ' vs ' + str(c_pos)}")

    # ── Детальный вывод UNIFIED ──
    print(f"\n── UNIFIED результат ──")
    if not u_faces:
        print("  face_count: 0")
    else:
        print(f"  face_count: {len(u_faces)}")
        for f in u_faces:
            print(f"  face_{f['index']}:")
            print(f"    identity: {f['identity']}")
            print(f"    confidence: {f['confidence']:.2f}")
            if f['emotion']:
                print(f"    emotion: {f['emotion']}")
                print(f"    emotion_confidence: {f['emotion_confidence']:.2f}")
            if f['posture']:
                print(f"    posture: {f['posture']}")
                print(f"    posture_confidence: {f['posture_confidence']:.2f}")
                if f.get('recon_error') is not None:
                    print(f"    recon_error: {f['recon_error']:.4f}")
                    print(f"    plausibility: {f['plausibility']}")
            b = f['bbox']
            print(f"    bbox: {b[0]},{b[1]},{b[2]},{b[3]}")

    # Моделей в unified
    print(f"\n── Моделей задействовано ──")
    print(f"  UNIFIED:  3 (FaceLandmarker + ArcFace + PoseLandmarker) + MLP + VAE = 5")
    print(f"  CURRENT:  4 (SCRFD + FaceLandmarker×N + ArcFace + PoseLandmarker) + MLP + VAE = 6+")


def avg_ms(times):
    return sum(times) / len(times) * 1000

def delta_pct(a, b):
    """Насколько a отличается от b в процентах (отрицательное = быстрее)."""
    avg_a = sum(a) / len(a)
    avg_b = sum(b) / len(b)
    if avg_b == 0:
        return '—'
    pct = (avg_a - avg_b) / avg_b * 100
    return f'{pct:+.0f}%'


if __name__ == '__main__':
    main()
