#!/usr/bin/env python3
"""sight.py — Полный визуальный анализ для Sight (v2: Pose-first + two-pass FaceLM).

Вызов: python sight.py [--json]

Архитектура:
  PoseLandmarker (full frame) → body_bbox
    └─ head_crop
         └─ FaceLM PASS 1 → face_bbox + 5 kps
              ├─ 5 kps → ArcFace (HD) → identity
              ├─ tight_crop → FaceLM PASS 2 → MLP → emotion
              └─ (far) HSEmotion CNN → emotion
  VAE posture — прямо с PoseLM landmarks
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
    finally: os.dup2(old_err_fd, 2); os.close(devnull)

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
from pose_vae import PoseVAE
from pose_heads import PoseClassifier
from pose_encoder import normalize_landmarks
from emotion_hse import HSEmotionClassifier

# ── Конфигурация ──────────────────────────────────────────
GALLERY_DIR = os.path.join(MVM_ROOT, 'models', 'references')
THRESHOLD = 0.35
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}

MLP_MODEL = os.path.join(MVM_ROOT, 'models', 'emotion', 'mlp_emotion.pth')
MLP_SCALER = os.path.join(MVM_ROOT, 'models', 'emotion', 'mlp_emotion_scaler.pkl')
VAE_PATH = os.path.join(MVM_ROOT, 'models', 'emotion', 'vposer_coco33.pth')
CLF_PATH = os.path.join(MVM_ROOT, 'models', 'emotion', 'pose_classifier.pth')
CLF_CONFIG = os.path.join(MVM_ROOT, 'models', 'emotion', 'pose_classifier_config.json')
FACE_LM_MODEL = os.path.join(MVM_ROOT, 'models', 'mediapipe', 'face_landmarker.task')
POSE_MODEL = os.path.join(MVM_ROOT, 'models', 'mediapipe', 'pose_landmarker.task')

with open(CLF_CONFIG) as f: _clf_cfg = json.load(f)
POSTURE_CLASSES_RU = _clf_cfg['class_names_ru']

CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
RU_MAP = {
    'anger': 'Злость', 'disgust': 'Отвращение', 'fear': 'Страх',
    'happy': 'Радость', 'neutral': 'Нейтральная', 'sad': 'Грусть', 'surprise': 'Удивление',
}
HSE_RU_MAP = {
    'Anger': 'Злость', 'Contempt': 'Презрение', 'Disgust': 'Отвращение',
    'Fear': 'Страх', 'Happiness': 'Радость', 'Neutral': 'Нейтральная',
    'Sadness': 'Грусть', 'Surprise': 'Удивление',
}

FACE_SIZE_THRESHOLD = 70  # px: лицо шире → 2-pass MLP, уже → HSEmotion
FACELM_TO_ARCFACE = {'reye': 33, 'leye': 263, 'nose': 4, 'rmouth': 61, 'lmouth': 291}


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
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
        if cap.isOpened():
            for _ in range(5):
                ret, frame = cap.read()
                if ret and frame is not None and frame.mean()>10:
                    cap.release(); return dev
            cap.release()
    return None

def capture_frame_hd(cam_path):
    cam_idx = int(cam_path.replace('/dev/video',''))
    for _ in range(3):
        cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
        if not cap.isOpened(): cap = cv2.VideoCapture(cam_idx, cv2.CAP_V4L2)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        for __ in range(5): ret, frame = cap.read()
        cap.release()
        if ret and frame is not None and frame.mean() > 10:
            frame_sd = cv2.resize(frame, (640, 480))
            return frame, frame_sd
        time.sleep(0.3)
    return None, None


# ── Детекция тел ──────────────────────────────────────────
def detect_bodies(pose_lm, frame_rgb, w, h):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    with suppress_stderr():
        result = pose_lm.detect(mp_img)
    bodies = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks:
            vis_torso = np.mean([lm[i].visibility for i in [11, 12, 23, 24]])
            if vis_torso < 0.35: continue
            xs = [min(max(p.x,0),1)*w for p in lm if p.visibility > 0.3]
            ys = [min(max(p.y,0),1)*h for p in lm if p.visibility > 0.3]
            if not xs: continue
            bbox = [max(0, min(xs)), max(0, min(ys)), min(w, max(xs)), min(h, max(ys))]
            pts = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            bodies.append({'bbox': bbox, 'landmarks': lm, 'pts_33': pts})
    return bodies


# ── Кроп головы ───────────────────────────────────────────
def head_crop_from_body(frame, body_bbox, head_ratio=0.5):
    x1, y1, x2, y2 = [int(v) for v in body_bbox]
    h, w = frame.shape[:2]
    body_h = y2 - y1; body_w = x2 - x1
    if body_h < 40: return None, 0, 0
    head_x1 = max(0, x1 - int(body_w * 0.3))
    head_y1 = max(0, y1 - int(body_h * 0.4))
    head_x2 = min(w, x2 + int(body_w * 0.3))
    head_y2 = min(h, y1 + int(body_h * head_ratio))
    if head_x2 <= head_x1 or head_y2 <= head_y1: return None, 0, 0
    return frame[head_y1:head_y2, head_x1:head_x2], head_x1, head_y1


# ── Сцена ─────────────────────────────────────────────────
def estimate_lighting(frame):
    m = float(frame.mean())
    if m>150: return 'bright',m
    elif m>70: return 'normal',m
    return 'dim',m


# ── Эмоции ────────────────────────────────────────────────
def compute_features_468(face_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks], dtype=np.float32)
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
    vae.load_state_dict(vae_ckpt['model_state_dict']); vae.eval()

    with open(CLF_CONFIG) as f: clf_cfg = json.load(f)
    clf = PoseClassifier(
        latent_dim=clf_cfg['latent_dim'], num_classes=clf_cfg['num_classes'],
        hidden_dim=clf_cfg['hidden_dim'], dropout=clf_cfg['dropout'],
    ).to(device)
    clf_ckpt = torch.load(CLF_PATH, map_location=device, weights_only=True)
    clf.load_state_dict(clf_ckpt['model_state_dict']); clf.eval()
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


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════
def main():
    sys.stdout.reconfigure(line_buffering=True)
    use_json = '--json' in sys.argv

    cam_path = find_camera()
    if cam_path is None:
        result = {"camera":"none","error":"camera not found","face_count":0}
        if use_json: print(json.dumps(result, ensure_ascii=False))
        else: print("camera: none\nerror: camera not found")
        _update_state("none", [], "unknown", 0)
        return

    frame_hd, frame_sd = capture_frame_hd(cam_path)
    if frame_hd is None:
        result = {"camera":cam_path,"error":"failed to capture frame","face_count":0}
        if use_json: print(json.dumps(result, ensure_ascii=False))
        else: print(f"camera: {cam_path}\nerror: failed to capture frame")
        _update_state(cam_path, [], "unknown", 0)
        return

    h_hd, w_hd = frame_hd.shape[:2]
    h_sd, w_sd = frame_sd.shape[:2]
    scale_x = w_hd / w_sd; scale_y = h_hd / h_sd
    lighting_level, lighting_val = estimate_lighting(frame_sd)

    # ── Модели ────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with suppress_all():
        face_id = FaceID(use_gpu=True)

        face_lm_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=FACE_LM_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1, running_mode=vision.RunningMode.IMAGE,
        )
        face_lm = vision.FaceLandmarker.create_from_options(face_lm_opts)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=POSE_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE, num_poses=3,
            output_segmentation_masks=False,
        )
        pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

        vae, pose_clf, pose_names_ru = load_vae_pose(device)

        mlp = EmotionMLP()
        mlp.load_state_dict(torch.load(MLP_MODEL, map_location=device, weights_only=True))
        mlp.to(device); mlp.eval()
        with open(MLP_SCALER, 'rb') as f: scaler = pickle.load(f)

    hse = HSEmotionClassifier(model_name='enet_b0_8_best_vgaf', use_gpu=True)

    # Галерея
    gallery = {}
    for fname in sorted(os.listdir(GALLERY_DIR)):
        if not fname.endswith('.jpg'): continue
        name_key = fname[2:-4]
        img = cv2.imread(os.path.join(GALLERY_DIR, fname))
        if img is None: continue
        with suppress_all(): emb = face_id.get_embedding(img)
        if emb is not None: gallery[NAME_MAP.get(name_key, name_key)] = emb

    # Прогрев
    frame_sd_rgb = cv2.cvtColor(frame_sd, cv2.COLOR_BGR2RGB)
    _ = pose_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_sd_rgb))
    rec = face_id.app.models['recognition']

    # ── PoseLandmarker → тела ──
    bodies = detect_bodies(pose_lm, frame_sd_rgb, w_sd, h_sd)

    if not bodies:
        result = {
            "camera": cam_path, "face_count": 0,
            "lighting": lighting_level, "brightness": round(lighting_val),
        }
        if use_json: print(json.dumps(result, ensure_ascii=False))
        else:
            print(f"camera: {cam_path}")
            print("face_count: 0")
            print(f"lighting: {lighting_level}")
            print(f"brightness: {lighting_val:.0f}")
        _update_state(cam_path, [], lighting_level, lighting_val)
        return

    # ── Обработка каждого тела ──
    faces_out = []

    for i, body in enumerate(bodies, 1):
        face_info = {
            'index': i, 'identity': 'unknown', 'confidence': 0.0,
            'emotion': None, 'emotion_confidence': None,
            'posture': None, 'posture_confidence': None,
            'recon_error': None, 'plausibility': None,
        }

        # ── Head crop → FaceLM PASS 1 → face_bbox + kps ──
        head_crop_sd, hx1, hy1 = head_crop_from_body(frame_sd, body['bbox'])
        if head_crop_sd is not None:
            head_rgb = cv2.cvtColor(head_crop_sd, cv2.COLOR_BGR2RGB)
            mp_head = mp.Image(image_format=mp.ImageFormat.SRGB, data=head_rgb)
            with suppress_stderr():
                face_lm_result = face_lm.detect(mp_head)

            if face_lm_result and face_lm_result.face_landmarks:
                lm_pass1 = face_lm_result.face_landmarks[0]
                hc_h, hc_w = head_crop_sd.shape[:2]
                face_xs = [p.x * hc_w for p in lm_pass1]
                face_ys = [p.y * hc_h for p in lm_pass1]
                face_w = max(face_xs) - min(face_xs)

                # ── PASS 2: tight crop → FaceLM → MLP ──
                if face_w > FACE_SIZE_THRESHOLD:
                    px = int(face_w * 0.2)
                    py = int((max(face_ys) - min(face_ys)) * 0.2)
                    fx1 = max(0, int(min(face_xs)) - px)
                    fy1 = max(0, int(min(face_ys)) - py)
                    fx2 = min(hc_w, int(max(face_xs)) + px)
                    fy2 = min(hc_h, int(max(face_ys)) + py)
                    tight_crop = head_crop_sd[fy1:fy2, fx1:fx2]

                    if tight_crop.size > 0:
                        tight_rgb = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2RGB)
                        mp_tight = mp.Image(image_format=mp.ImageFormat.SRGB, data=tight_rgb)
                        with suppress_stderr():
                            tight_result = face_lm.detect(mp_tight)
                        if tight_result and tight_result.face_landmarks:
                            emo, conf = classify_emotion_mlp(
                                tight_result.face_landmarks[0], mlp, scaler, device
                            )
                            face_info['emotion'] = emo
                            face_info['emotion_confidence'] = conf
                        elif hse is not None:
                            hse_name, hse_conf = hse.predict(tight_crop)
                            face_info['emotion'] = HSE_RU_MAP.get(hse_name, hse_name)
                            face_info['emotion_confidence'] = round(float(hse_conf), 2)
                else:
                    # Far: HSEmotion
                    if hse is not None:
                        px, py = int(face_w * 0.2), int((max(face_ys)-min(face_ys)) * 0.2)
                        fx1 = max(0, int(min(face_xs)) - px)
                        fy1 = max(0, int(min(face_ys)) - py)
                        fx2 = min(hc_w, int(max(face_xs)) + px)
                        fy2 = min(hc_h, int(max(face_ys)) + py)
                        face_crop = head_crop_sd[fy1:fy2, fx1:fx2]
                        if face_crop.size > 0:
                            hse_name, hse_conf = hse.predict(face_crop)
                            face_info['emotion'] = HSE_RU_MAP.get(hse_name, hse_name)
                            face_info['emotion_confidence'] = round(float(hse_conf), 2)

                # ── Identity: ArcFace на HD ──
                try:
                    kps_sd = np.array([
                        [lm_pass1[FACELM_TO_ARCFACE['reye']].x * hc_w + hx1,
                         lm_pass1[FACELM_TO_ARCFACE['reye']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['leye']].x * hc_w + hx1,
                         lm_pass1[FACELM_TO_ARCFACE['leye']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['nose']].x * hc_w + hx1,
                         lm_pass1[FACELM_TO_ARCFACE['nose']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['rmouth']].x * hc_w + hx1,
                         lm_pass1[FACELM_TO_ARCFACE['rmouth']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['lmouth']].x * hc_w + hx1,
                         lm_pass1[FACELM_TO_ARCFACE['lmouth']].y * hc_h + hy1],
                    ], dtype=np.float32)
                    kps_hd = kps_sd.copy()
                    kps_hd[:, 0] *= scale_x
                    kps_hd[:, 1] *= scale_y

                    emb = rec.get(frame_hd, type('Stub', (), {'kps': kps_hd})())
                    if gallery:
                        name, sim = face_id.identify(emb, gallery, THRESHOLD)
                        face_info['identity'] = name if name else 'unknown'
                        face_info['confidence'] = round(sim, 2)
                except Exception:
                    pass

            else:
                # FaceLM не нашёл лицо → HSEmotion на всём head_crop
                if hse is not None:
                    try:
                        hse_name, hse_conf = hse.predict(head_crop_sd)
                        face_info['emotion'] = HSE_RU_MAP.get(hse_name, hse_name)
                        face_info['emotion_confidence'] = round(float(hse_conf), 2)
                    except Exception:
                        pass

        # ── Поза: VAE с PoseLM ──
        try:
            pos, conf, rerr, logpz = classify_posture_vae(
                body['landmarks'], vae, pose_clf, pose_names_ru, device
            )
            face_info['posture'] = pos
            face_info['posture_confidence'] = conf
            face_info['recon_error'] = round(rerr, 4)
            face_info['plausibility'] = 'normal' if logpz < 14.3 else 'unusual'
        except Exception:
            pass

        faces_out.append(face_info)

    # ── Вывод ─────────────────────────────────────────────
    if use_json:
        output = {
            'camera': cam_path, 'face_count': len(bodies),
            'lighting': lighting_level, 'brightness': round(lighting_val),
            'faces': faces_out,
        }
        print(json.dumps(output, ensure_ascii=False))
    else:
        print(f"camera: {cam_path}")
        print(f"face_count: {len(bodies)}")
        print(f"lighting: {lighting_level}")
        print(f"brightness: {lighting_val:.0f}")
        for f in faces_out:
            print(f"face_{f['index']}:")
            print(f"  identity: {f['identity']}")
            print(f"  confidence: {f['confidence']:.2f}")
            if f['emotion']:
                print(f"  emotion: {f['emotion']}")
                print(f"  emotion_confidence: {f['emotion_confidence']:.2f}")
            else:
                print(f"  emotion: none")
            if f['posture']:
                print(f"  posture: {f['posture']}")
                print(f"  posture_confidence: {f['posture_confidence']:.2f}")
            else:
                print(f"  posture: none")
            if f.get('recon_error') is not None:
                print(f"  recon_error: {f['recon_error']:.4f}")
                print(f"  plausibility: {f['plausibility']}")

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
