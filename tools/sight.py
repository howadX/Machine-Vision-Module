#!/usr/bin/env python3
"""sight.py — Полный визуальный анализ для Sight.
Вызов: python sight.py [--json]
Архитектура: два кропа на человека (лицо + тело).
Эмоции: MLP (PyTorch) на 468 геометрических признаках (FaceLandmarker).
Поза: VAE (PoseVAE) + PoseClassifier на 33 COCO landmarks (PoseLandmarker).
"""
import os, sys, glob, time, warnings, logging, pickle, json
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

TZ = timezone(timedelta(hours=3))

# ── Пути ──────────────────────────────────────────────────
SIGHT_WORKSPACE = os.path.expanduser('~/.hermes/profiles/sight/workspace')
STATE_FILE = os.path.join(SIGHT_WORKSPACE, 'current_state.md')
MVM_ROOT = os.path.expanduser('~/repos/Machine Vision Module')

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
    finally:
        os.dup2(old_err_fd, 2)
        os.close(devnull); os.close(old_err_fd)

with suppress_all():
    import cv2
    import numpy as np
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    import torch
    import torch.nn as nn

sys.path.insert(0, os.path.join(MVM_ROOT, 'core'))
from face_id import FaceID

# ── Конфигурация ──────────────────────────────────────────
GALLERY_DIR = os.path.join(MVM_ROOT, 'models', 'references')
THRESHOLD = 0.35
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}

# MLP эмоции
MLP_MODEL = os.path.join(MVM_ROOT, 'models', 'emotion', 'mlp_emotion.pth')
MLP_SCALER = os.path.join(MVM_ROOT, 'models', 'emotion', 'mlp_emotion_scaler.pkl')
MLP_CONFIG = os.path.join(MVM_ROOT, 'models', 'emotion', 'mlp_emotion_config.json')

# VAE поза тела
VAE_PATH = os.path.join(MVM_ROOT, 'models', 'emotion', 'vposer_coco33.pth')
CLF_PATH = os.path.join(MVM_ROOT, 'models', 'emotion', 'pose_classifier.pth')
CLF_CONFIG = os.path.join(MVM_ROOT, 'models', 'emotion', 'pose_classifier_config.json')

# MediaPipe
POSE_MODEL = os.path.join(MVM_ROOT, 'models', 'mediapipe', 'pose_landmarker.task')
FACE_LM_MODEL = os.path.join(MVM_ROOT, 'models', 'mediapipe', 'face_landmarker.task')

import json as _json
with open(CLF_CONFIG) as f:
    _clf_cfg = _json.load(f)
POSTURE_CLASSES_RU = _clf_cfg['class_names_ru']
POSTURE_CLASSES_EN = _clf_cfg['class_names']

CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
RU_MAP = {'anger':'Злость','disgust':'Отвращение','fear':'Страх',
          'happy':'Радость','neutral':'Нейтральная','sad':'Грусть','surprise':'Удивление'}
POSTURE_CLASSES_EN = POSTURE_CLASSES_EN  # уже определено выше


# ── MLP модель ────────────────────────────────────────────
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
    ret, frame = False, None
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
def crop_face(frame, bbox, pad_ratio=0.2):
    x1,y1,x2,y2 = [int(v) for v in bbox]
    h,w = frame.shape[:2]
    fw, fh = x2-x1, y2-y1
    px, py = int(fw*pad_ratio), int(fh*pad_ratio)
    x1=max(0,x1-px); y1=max(0,y1-py)
    x2=min(w,x2+px); y2=min(h,y2+py)
    if x2<=x1 or y2<=y1: return None
    return frame[y1:y2, x1:x2]

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


# ── Эмоции (MLP) ──────────────────────────────────────────
def compute_features_468(face_landmarks):
    """468 нормированных расстояний от кончика носа."""
    pts = np.array([[lm.x,lm.y,lm.z] for lm in face_landmarks], dtype=np.float32)
    nose_tip=pts[4]; left_eye=pts[33]; right_eye=pts[263]
    interocular = np.linalg.norm(left_eye-right_eye)
    if interocular<1e-6: interocular=1.0
    return (np.linalg.norm(pts-nose_tip, axis=1)/interocular)[:468]

def classify_emotion_mlp(face_landmarks, model, scaler, device):
    """MLP: 468 признаков → StandardScaler → model → (emotion_ru, confidence)."""
    feats = compute_features_468(face_landmarks).reshape(1, -1)
    feats_scaled = scaler.transform(feats).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(feats_scaled).to(device)
        logits = model(tensor)
        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(proba.argmax())
    emotion_ru = RU_MAP.get(CLASSES[pred_idx], CLASSES[pred_idx])
    return emotion_ru, round(float(proba.max()), 2)


# ── Поза (VAE + PoseClassifier) ───────────────────────────
def load_vae_pose(device):
    """Загружает VAE + PoseClassifier."""
    from pose_vae import PoseVAE
    from pose_heads import PoseClassifier
    import json as _j

    vae_ckpt = torch.load(VAE_PATH, map_location=device, weights_only=True)
    vae_cfg = vae_ckpt['config']
    vae = PoseVAE(
        input_dim=vae_cfg['input_dim'], latent_dim=vae_cfg['latent_dim'],
        hidden_dims=tuple(vae_cfg['hidden_dims']), dropout=vae_cfg['dropout'],
    ).to(device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])
    vae.eval()

    with open(CLF_CONFIG) as f:
        clf_cfg = _j.load(f)
    clf = PoseClassifier(
        latent_dim=clf_cfg['latent_dim'], num_classes=clf_cfg['num_classes'],
        hidden_dim=clf_cfg['hidden_dim'], dropout=clf_cfg['dropout'],
    ).to(device)
    clf_ckpt = torch.load(CLF_PATH, map_location=device, weights_only=True)
    clf.load_state_dict(clf_ckpt['model_state_dict'])
    clf.eval()

    return vae, clf, clf_cfg['class_names_ru'], clf_cfg['class_names']


def classify_posture_vae(pose_landmarks, vae, clf, class_names, device):
    """VAE + PoseClassifier: landmarks → z → класс позы + confidence."""
    from pose_encoder import normalize_landmarks
    import numpy as np

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


# ── Обработка кропов ──────────────────────────────────────
def run_face_landmarker(landmarker, bgr_crop):
    if bgr_crop is None: return None
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with suppress_stderr():
        res = landmarker.detect(mp_img)
    if res.face_landmarks:
        return res.face_landmarks[0]
    return None

def run_pose_landmarker(landmarker, bgr_crop):
    if bgr_crop is None: return None
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with suppress_stderr():
        res = landmarker.detect(mp_img)
    if res.pose_landmarks:
        return res.pose_landmarks[0]
    return None


# ── Главное ───────────────────────────────────────────────
def main():
    sys.stdout.reconfigure(line_buffering=True)
    use_json = '--json' in sys.argv

    cam_path = find_camera()
    if cam_path is None:
        result = {"camera":"none","error":"camera not found","face_count":0}
        if use_json: print(json.dumps(result, ensure_ascii=False))
        return

    frame = capture_frame(cam_path)
    if frame is None:
        result = {"camera":cam_path,"error":"failed to capture frame","face_count":0}
        if use_json: print(json.dumps(result, ensure_ascii=False))
        return

    h, w = frame.shape[:2]
    lighting_level, lighting_val = estimate_lighting(frame)

    # ── Модели ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with suppress_all():
        face_id = FaceID(use_gpu=True)
        
        face_lm_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_LM_MODEL),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1, running_mode=vision.RunningMode.IMAGE,
        )
        face_lm = vision.FaceLandmarker.create_from_options(face_lm_opts)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=POSE_MODEL, delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE, num_poses=1,
            output_segmentation_masks=False,
        )
        pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

        vae, pose_clf, pose_names_ru, pose_names_en = load_vae_pose(device)

        # MLP эмоции
        mlp = EmotionMLP()
        mlp.load_state_dict(torch.load(MLP_MODEL, map_location=device, weights_only=True))
        mlp.to(device)
        mlp.eval()
        with open(MLP_SCALER, 'rb') as f:
            scaler = pickle.load(f)

    # Галерея
    gallery = {}
    for fname in sorted(os.listdir(GALLERY_DIR)):
        if not fname.endswith('.jpg'): continue
        name_key = fname[2:-4]
        img = cv2.imread(os.path.join(GALLERY_DIR,fname))
        if img is None: continue
        with suppress_all(): emb = face_id.get_embedding(img)
        if emb is not None: gallery[NAME_MAP.get(name_key,name_key)] = emb

    # SCRFD
    with suppress_stderr():
        raw_faces = face_id.app.get(frame)

    if not raw_faces:
        result = {"camera":cam_path,"face_count":0,"lighting":lighting_level,"brightness":lighting_val}
        if use_json: print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"camera: {cam_path}")
            print("face_count: 0")
            print(f"lighting: {lighting_level}")
            print(f"brightness: {lighting_val:.0f}")
        _update_state(cam_path, [], lighting_level, lighting_val)
        return

    raw_faces.sort(key=lambda f: f.get('det_score',0), reverse=True)
    faces_out = []

    for i, face in enumerate(raw_faces, 1):
        bbox = face['bbox'].tolist()
        det_score = float(face.get('det_score',0))
        emb = np.asarray(face['embedding'], dtype=np.float32)

        if gallery: name, sim = face_id.identify(emb, gallery, THRESHOLD)
        else: name, sim = None, 0.0

        dist = estimate_distance(bbox, frame.shape)
        face_w = int(bbox[2]-bbox[0]); face_h = int(bbox[3]-bbox[1])

        face_info = {
            'index': i, 'identity': name if name else 'unknown',
            'confidence': round(sim, 2), 'det_score': round(det_score, 2),
            'bbox': [int(v) for v in bbox], 'size': f"{face_w}x{face_h}", 'distance': dist,
            'emotion': None, 'emotion_confidence': None,
            'posture': None, 'posture_confidence': None,
            'recon_error': None, 'plausibility': None,
        }

        face_crop = crop_face(frame, bbox)
        body_crop = crop_body(frame, bbox)

        # Кроп лица → FaceLandmarker → MLP → эмоция
        face_lm_result = run_face_landmarker(face_lm, face_crop)
        if face_lm_result:
            try:
                emo_name, emo_conf = classify_emotion_mlp(face_lm_result, mlp, scaler, device)
                face_info['emotion'] = emo_name
                face_info['emotion_confidence'] = emo_conf
            except Exception: pass

        # Кроп тела → PoseLandmarker → VAE → поза
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
            except Exception: pass

        faces_out.append(face_info)

    # ── Вывод ─────────────────────────────────────────────
    if use_json:
        output = {
            'camera': cam_path, 'face_count': len(raw_faces),
            'lighting': lighting_level, 'brightness': round(lighting_val, 0),
            'faces': faces_out,
        }
        print(json.dumps(output, ensure_ascii=False))
    else:
        print(f"camera: {cam_path}")
        print(f"face_count: {len(raw_faces)}")
        print(f"lighting: {lighting_level}")
        print(f"brightness: {lighting_val:.0f}")
        for f in faces_out:
            print(f"face_{f['index']}:")
            print(f"  identity: {f['identity']}")
            print(f"  confidence: {f['confidence']:.2f}")
            if f['emotion']:
                print(f"  emotion: {f['emotion']}")
                print(f"  emotion_confidence: {f['emotion_confidence']:.2f}")
            else: print(f"  emotion: none")
            if f['posture']:
                print(f"  posture: {f['posture']}")
                print(f"  posture_confidence: {f['posture_confidence']:.2f}")
            else: print(f"  posture: none")
            if f.get('recon_error') is not None:
                print(f"  recon_error: {f['recon_error']:.4f}")
                print(f"  plausibility: {f['plausibility']}")
            print(f"  det_score: {f['det_score']:.2f}")
            print(f"  bbox: {','.join(map(str,f['bbox']))}")
            print(f"  size: {f['size']}")
            print(f"  distance: {f['distance']}")

    _update_state(cam_path, faces_out, lighting_level, lighting_val)


def _update_state(cam_path, faces, lighting_level, lighting_val):
    now = datetime.now(TZ).isoformat(timespec='minutes')
    known = [f['identity'] for f in faces if f['identity'] != 'unknown']
    person_str = ', '.join(known) if known else 'none'
    dominant = faces[0]['emotion'] if faces and faces[0].get('emotion') else 'none'
    posture = faces[0]['posture'] if faces and faces[0].get('posture') else 'none'
    plausibility = faces[0].get('plausibility', 'unknown') if faces else 'unknown'
    status = 'occupied' if faces else 'empty'

    content = (
        f"# Camera State\n"
        f"- Last checked: {now}\n"
        f"- Camera: {cam_path}\n"
        f"- Lighting: {lighting_level} (mean={lighting_val:.0f})\n"
        f"- Face count: {len(faces)}\n"
        f"- Persons visible: {person_str}\n"
        f"- Dominant emotion: {dominant}\n"
        f"- Posture: {posture}\n"
        f"- Plausibility: {plausibility}\n"
        f"- Status: {status}\n"
    )
    os.makedirs(SIGHT_WORKSPACE, exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        f.write(content)


if __name__ == '__main__':
    main()
