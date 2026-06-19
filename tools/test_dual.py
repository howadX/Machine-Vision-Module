#!/usr/bin/env python3
"""test_dual.py — dual-entry parallel pipeline для sight.py.

Архитектура:
  Phase 1 (PARALLEL, 2 потока):
    FaceLandmarker(full, num_faces=5) ─┐
    PoseLandmarker(full, num_poses=5) ─┤ ThreadPoolExecutor
                                       │
  Phase 2 (FUSION):                    │
    face ↔ body matching ──────────────┘
    
  Phase 3 (PARALLEL per person):
    ├─ 478_lm → MLP → emotion          (CPU, <1ms)
    ├─ 5_kps → ArcFace → identity      (GPU, 2.6ms)
    └─ 33_lm → VAE → posture           (CPU, <2ms, прямо с full-frame!)
    
  Unmatched bodies (no face = спиной):
    └─ 33_lm → VAE → posture           (facing: away)
"""
import os, sys, glob, time, warnings, logging, pickle, json
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

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
from emotion_hse import HSEmotionClassifier

# ── Конфигурация ──────────────────────────────────────────
GALLERY_DIR = os.path.join(ROOT, 'models', 'references')
THRESHOLD = 0.22  # минимальный порог для ArcFace.identify (трекер фильтрует выше)
NAME_MAP = {'V': 'Вадим', 'D': 'Даня', 'N': 'Никита', 'B': 'Белла'}

MLP_MODEL = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion.pth')
MLP_SCALER = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion_scaler.pkl')
VAE_PATH = os.path.join(ROOT, 'models', 'emotion', 'vposer_coco33.pth')
CLF_PATH = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier.pth')
CLF_CONFIG = os.path.join(ROOT, 'models', 'emotion', 'pose_classifier_config.json')
FACE_LM_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'face_landmarker.task')
POSE_MODEL = os.path.join(ROOT, 'models', 'mediapipe', 'pose_landmarker.task')

with open(CLF_CONFIG) as f:
    _clf_cfg = json.load(f)

CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
RU_MAP = {
    'anger': 'Злость', 'disgust': 'Отвращение', 'fear': 'Страх',
    'happy': 'Радость', 'neutral': 'Нейтральная', 'sad': 'Грусть', 'surprise': 'Удивление',
}

# ArcFace 5-точечный маппинг из 468 landmarks
FACELM_TO_ARCFACE = {'reye': 33, 'leye': 263, 'nose': 4, 'rmouth': 61, 'lmouth': 291}

# Пороги идентификации
THRESHOLD_NEAR = 0.50   # уверенное совпадение — identity фиксируется
THRESHOLD_MEDIUM = 0.35 # вероятное совпадение
THRESHOLD_FAR = 0.22    # слабое совпадение — identity из трекера
CONFIDENCE_DECAY = 0.02 # падение confidence в секунду без переподтверждения
TRACKING_TTL = 15.0     # секунд до полного сброса трека

# Роутинг эмоций: MLP (близко) vs HSEmotion CNN (далеко)
FACE_SIZE_THRESHOLD = 70  # px: лицо шире этого → MLP, уже → HSEmotion

HSE_RU_MAP = {
    'Anger': 'Злость', 'Contempt': 'Презрение', 'Disgust': 'Отвращение',
    'Fear': 'Страх', 'Happiness': 'Радость', 'Neutral': 'Нейтральная',
    'Sadness': 'Грусть', 'Surprise': 'Удивление',
}

# ── Трекер персон ────────────────────────────────────────
class PersonTracker:
    """Хранит identity между кадрами, деградирует confidence со временем."""
    
    def __init__(self):
        self.tracks = {}  # track_id → {identity, confidence, bbox, last_seen, first_seen}
        self._next_id = 1
    
    def update(self, timestamp: float, face_bbox, arcface_identity: str, arcface_confidence: float):
        """Обновляет или создаёт трек для лица.
        
        Возвращает (identity, confidence, tracking_id, is_persistent).
        """
        # Ищем совпадение по bbox
        best_id = None
        best_iou = 0.3  # порог для матчинга треков
        for tid, track in self.tracks.items():
            iou = bbox_iou(face_bbox, track['bbox'])
            if iou > best_iou:
                best_iou = iou
                best_id = tid
        
        now = timestamp
        
        if best_id is not None:
            track = self.tracks[best_id]
            elapsed = now - track['last_seen']
            
            if arcface_confidence >= THRESHOLD_NEAR:
                # Уверенное распознавание — закрепляем identity
                track['identity'] = arcface_identity
                track['confidence'] = arcface_confidence
                track['last_seen'] = now
                track['bbox'] = face_bbox
                return arcface_identity, arcface_confidence, best_id, False
            
            elif arcface_confidence >= THRESHOLD_MEDIUM:
                # Вероятное — подтверждаем если совпадает с треком
                if arcface_identity == track['identity']:
                    track['confidence'] = arcface_confidence
                    track['last_seen'] = now
                    track['bbox'] = face_bbox
                    return arcface_identity, arcface_confidence, best_id, False
                else:
                    # Другой identity — конфликт, обновляем если новый confident
                    track['identity'] = arcface_identity
                    track['confidence'] = arcface_confidence
                    track['last_seen'] = now
                    track['bbox'] = face_bbox
                    return arcface_identity, arcface_confidence, best_id, False
            
            elif arcface_confidence >= THRESHOLD_FAR:
                # Далеко — сохраняем identity трека, деградируем confidence
                degradation = elapsed * CONFIDENCE_DECAY
                track['confidence'] = max(0.12, track['confidence'] - degradation)
                track['last_seen'] = now
                track['bbox'] = face_bbox
                persistent_identity = track['identity'] if track['identity'] != 'unknown' else None
                if persistent_identity:
                    return persistent_identity, track['confidence'], best_id, True
            
            else:
                # Слишком далеко — сбрасываем
                if elapsed > TRACKING_TTL:
                    track['identity'] = 'unknown'
                    track['confidence'] = 0
                track['last_seen'] = now
                track['bbox'] = face_bbox
                return 'unknown', arcface_confidence, best_id, False
        
        # Новый трек
        if arcface_confidence >= THRESHOLD_MEDIUM:
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                'identity': arcface_identity,
                'confidence': arcface_confidence,
                'bbox': face_bbox,
                'last_seen': now,
                'first_seen': now,
            }
            return arcface_identity, arcface_confidence, tid, False
        elif arcface_confidence >= THRESHOLD_FAR:
            # Слабое совпадение без истории — создаём как unknown
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = {
                'identity': 'unknown',
                'confidence': arcface_confidence,
                'bbox': face_bbox,
                'last_seen': now,
                'first_seen': now,
            }
            return 'unknown', arcface_confidence, tid, False
        
        return 'unknown', arcface_confidence, None, False
    
    def cleanup(self, timestamp: float):
        """Удаляет треки старше TRACKING_TTL."""
        stale = [tid for tid, t in self.tracks.items()
                 if timestamp - t['last_seen'] > TRACKING_TTL]
        for tid in stale:
            del self.tracks[tid]

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


# ═══════════════════════════════════════════════════════════
#  DETECTORS (shared, thread-safe)
# ═══════════════════════════════════════════════════════════

class FaceStub:
    def __init__(self, kps): self.kps = kps


def detect_faces(face_lm, frame_rgb, w, h):
    """FaceLandmarker на полном кадре → список лиц с landmarks."""
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    with suppress_stderr():
        result = face_lm.detect(mp_img)
    faces = []
    if result.face_landmarks:
        for lm in result.face_landmarks:
            # bbox с запасом 20%
            xs = [p.x * w for p in lm]; ys = [p.y * h for p in lm]
            x1, y1 = min(xs), min(ys)
            x2, y2 = max(xs), max(ys)
            fw, fh = x2 - x1, y2 - y1
            px, py = fw * 0.2, fh * 0.2
            bbox = [max(0, x1 - px), max(0, y1 - py),
                    min(w, x2 + px), min(h, y2 + py)]
            # 5 ключевых точек для ArcFace
            kps = np.array([
                [lm[FACELM_TO_ARCFACE['reye']].x * w, lm[FACELM_TO_ARCFACE['reye']].y * h],
                [lm[FACELM_TO_ARCFACE['leye']].x * w, lm[FACELM_TO_ARCFACE['leye']].y * h],
                [lm[FACELM_TO_ARCFACE['nose']].x * w, lm[FACELM_TO_ARCFACE['nose']].y * h],
                [lm[FACELM_TO_ARCFACE['rmouth']].x * w, lm[FACELM_TO_ARCFACE['rmouth']].y * h],
                [lm[FACELM_TO_ARCFACE['lmouth']].x * w, lm[FACELM_TO_ARCFACE['lmouth']].y * h],
            ], dtype=np.float32)
            faces.append({'bbox': bbox, 'landmarks_478': lm, 'kps': kps})
    return faces


def detect_bodies(pose_lm, frame_rgb, w, h):
    """PoseLandmarker на полном кадре → список тел с landmarks.

    Фильтрует фантомные детекции по средней visibility (>0.35).
    """
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    with suppress_stderr():
        result = pose_lm.detect(mp_img)
    bodies = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks:
            # Фильтр: средняя visibility ключевых точек торса
            vis_torso = np.mean([lm[i].visibility for i in [11, 12, 23, 24]])  # shoulders + hips
            if vis_torso < 0.35:
                continue
            # bbox по видимым точкам (с cap'ом к размерам кадра)
            xs = [min(max(p.x, 0), 1) * w for p in lm if p.visibility > 0.3]
            ys = [min(max(p.y, 0), 1) * h for p in lm if p.visibility > 0.3]
            if not xs:
                continue
            bbox = [min(xs), min(ys), max(xs), max(ys)]
            # Cap к размерам кадра
            bbox = [max(0, bbox[0]), max(0, bbox[1]), min(w, bbox[2]), min(h, bbox[3])]
            pts = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
            bodies.append({'bbox': bbox, 'landmarks': lm, 'pts_33': pts})
    return bodies


# ═══════════════════════════════════════════════════════════
#  FUSION: face ↔ body matching
# ═══════════════════════════════════════════════════════════

def bbox_iou(a, b):
    """Intersection over Union двух bbox [x1,y1,x2,y2]."""
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def bbox_center(bbox):
    """Центр bbox [x1,y1,x2,y2]."""
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def match_faces_to_bodies(faces, bodies):
    """Сопоставляет лица с телами: центр лица внутри bbox тела.

    IoU между bbox лица и bbox тела бессмыслен (разные масштабы).
    Правильный критерий: голова находится ВНУТРИ bounding box тела.
    """
    persons = []
    matched_body_ids = set()

    for face in faces:
        fc_x, fc_y = bbox_center(face['bbox'])
        best_body = None
        best_bid = -1
        best_overlap = 0

        for bid, body in enumerate(bodies):
            bx1, by1, bx2, by2 = body['bbox']
            # Проверка: центр лица внутри bbox тела (с запасом)
            margin_x = (bx2 - bx1) * 0.3
            margin_y = (by2 - by1) * 0.2
            if (bx1 - margin_x <= fc_x <= bx2 + margin_x and
                by1 - margin_y <= fc_y <= by2 + margin_y):
                # Дополнительно: overlap площади лица с телом
                overlap = bbox_iou(face['bbox'], body['bbox'])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_body = body
                    best_bid = bid

        if best_body is not None:
            persons.append({'face': face, 'body': best_body, 'facing': 'toward'})
            matched_body_ids.add(best_bid)
        else:
            persons.append({'face': face, 'body': None, 'facing': 'toward'})

    # Тела без лиц (спиной)
    for bid, body in enumerate(bodies):
        if bid not in matched_body_ids:
            persons.append({'face': None, 'body': body, 'facing': 'away'})

    return persons


# ═══════════════════════════════════════════════════════════
#  PER-PERSON PROCESSING
# ═══════════════════════════════════════════════════════════

def compute_emotion(face_lm_478, mlp, scaler, device):
    """468 landmarks → MLP → emotion_ru + confidence."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in face_lm_478], dtype=np.float32)
    nose_tip = pts[4]; left_eye = pts[33]; right_eye = pts[263]
    interocular = np.linalg.norm(left_eye - right_eye)
    if interocular < 1e-6: interocular = 1.0
    feats = (np.linalg.norm(pts - nose_tip, axis=1) / interocular)[:468]

    feats_scaled = scaler.transform(feats.reshape(1, -1)).astype(np.float32)
    with torch.no_grad():
        tensor = torch.from_numpy(feats_scaled).to(device)
        logits = mlp(tensor)
        proba = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_idx = int(proba.argmax())
    return RU_MAP.get(CLASSES[pred_idx], CLASSES[pred_idx]), round(float(proba.max()), 2)


def compute_emotion_from_crop(face_lm_crop, bgr_crop, mlp, scaler, device):
    """FaceLandmarker на КРОПЕ лица → MLP → emotion. Точнее чем full-frame landmarks."""
    if bgr_crop is None or bgr_crop.size == 0:
        return None, 0.0
    rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
    h_c, w_c = rgb.shape[:2]
    if h_c < 20 or w_c < 20:
        return None, 0.0
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    with suppress_stderr():
        res = face_lm_crop.detect(mp_img)
    if not res.face_landmarks:
        return None, 0.0
    lm = res.face_landmarks[0]
    return compute_emotion(lm, mlp, scaler, device)


def compute_identity(rec_model, frame, kps, face_id_obj, gallery):
    """5 kps → ArcFace → identity + confidence."""
    stub = FaceStub(kps)
    emb = rec_model.get(frame, stub)
    if gallery:
        name, sim = face_id_obj.identify(emb, gallery, THRESHOLD)
        return name if name else 'unknown', round(sim, 2)
    return 'unknown', 0.0


def compute_posture(pts_33, vae, clf, class_names, device):
    """33 landmarks → normalize → VAE → posture + confidence."""
    norm, _meta = normalize_landmarks(pts_33)
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


def crop_face_from_bbox(frame, bbox):
    """bbox [x1,y1,x2,y2] → BGR кроп лица."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


# ═══════════════════════════════════════════════════════════
#  FULL DUAL-ENTRY PIPELINE
# ═══════════════════════════════════════════════════════════

def head_crop_from_body(frame, body_bbox, head_ratio=0.5):
    """Из body_bbox вычисляет регион головы (верхняя часть тела).

    PoseLandmarker bbox: [x1, y1, x2, y2] от плеч до ступней.
    Голова: верхние head_ratio высоты bbox + увеличенный запас вверх.
    
    Returns: (crop, head_x1, head_y1) или (None, 0, 0)
    """
    x1, y1, x2, y2 = [int(v) for v in body_bbox]
    h, w = frame.shape[:2]
    body_h = y2 - y1
    body_w = x2 - x1
    
    if body_h < 40:  # слишком маленькое тело — невалидно
        return None, 0, 0
    
    head_x1 = max(0, x1 - int(body_w * 0.3))
    head_y1 = max(0, y1 - int(body_h * 0.4))       # больше запаса вверх (голова над плечами)
    head_x2 = min(w, x2 + int(body_w * 0.3))
    head_y2 = min(h, y1 + int(body_h * head_ratio))  # верхняя половина тела
    
    if head_x2 <= head_x1 or head_y2 <= head_y1:
        return None, 0, 0
    return frame[head_y1:head_y2, head_x1:head_x2], head_x1, head_y1


def run_dual_pipeline(frame_hd, frame_sd, face_lm_crop, pose_lm, face_id_obj, gallery,
                      mlp, scaler, vae, pose_clf, pose_names, device, 
                      mlp_distilled=None, scaler_distilled=None,
                      hse_classifier=None, tracker=None, timestamp=None):
    """Pose-first архитектура: PoseLM → head crop → FaceLM.

    Phase 1: PoseLandmarker(full SD) — детекция тел (всегда, даже спиной)
    Phase 2: Для каждого тела — head_crop → FaceLandmarker → emotion + identity
    Phase 3: VAE posture прямо с PoseLM landmarks

    Нет отдельного детектора лиц — FaceLandmarker только на кропах головы.
    """
    h_hd, w_hd = frame_hd.shape[:2]
    h_sd, w_sd = frame_sd.shape[:2]
    scale_x = w_hd / w_sd
    scale_y = h_hd / h_sd
    
    frame_sd_rgb = cv2.cvtColor(frame_sd, cv2.COLOR_BGR2RGB)
    rec = face_id_obj.app.models['recognition']
    now = timestamp if timestamp is not None else time.time()

    t_total = time.perf_counter()

    # ── Phase 1: PoseLandmarker на полном кадре ──
    t0 = time.perf_counter()
    bodies = detect_bodies(pose_lm, frame_sd_rgb, w_sd, h_sd)
    t_detect = time.perf_counter() - t0

    # ── Phase 2+3: Для каждого тела — голова + поза ──
    t_process_start = time.perf_counter()
    person_infos = []

    for body in bodies:
        info = {'facing': 'unknown',
                'identity': 'unknown', 'confidence': 0.0,
                'emotion': None, 'emotion_confidence': None,
                'posture': None, 'posture_confidence': None,
                'recon_error': None, 'plausibility': None,
                'tracked': False}

        # ── Two-pass FaceLandmarker: head_crop → face_bbox → tight_crop → MLP ──
        head_crop_sd, hx1, hy1 = head_crop_from_body(frame_sd, body['bbox'])
        if head_crop_sd is not None:
            # ── PASS 1: FaceLandmarker на head_crop → bbox лица + 5 kps ──
            try:
                head_rgb = cv2.cvtColor(head_crop_sd, cv2.COLOR_BGR2RGB)
                mp_head = mp.Image(image_format=mp.ImageFormat.SRGB, data=head_rgb)
                with suppress_stderr():
                    face_lm_result = face_lm_crop.detect(mp_head)
            except Exception:
                face_lm_result = None

            if face_lm_result and face_lm_result.face_landmarks:
                lm_pass1 = face_lm_result.face_landmarks[0]
                hc_h, hc_w = head_crop_sd.shape[:2]
                
                # bbox лица в координатах head_crop (из pass 1)
                face_xs = [p.x * hc_w for p in lm_pass1]
                face_ys = [p.y * hc_h for p in lm_pass1]
                face_w = max(face_xs) - min(face_xs)
                
                # ── PASS 2: tight face crop → FaceLM → точные landmarks → MLP ──
                if face_w > FACE_SIZE_THRESHOLD:
                    # Кроп лица с запасом 20% (как в sight.py)
                    px = int(face_w * 0.2)
                    py = int((max(face_ys) - min(face_ys)) * 0.2)
                    fx1 = max(0, int(min(face_xs)) - px)
                    fy1 = max(0, int(min(face_ys)) - py)
                    fx2 = min(hc_w, int(max(face_xs)) + px)
                    fy2 = min(hc_h, int(max(face_ys)) + py)
                    tight_crop = head_crop_sd[fy1:fy2, fx1:fx2]
                    
                    if tight_crop.size > 0:
                        try:
                            tight_rgb = cv2.cvtColor(tight_crop, cv2.COLOR_BGR2RGB)
                            mp_tight = mp.Image(image_format=mp.ImageFormat.SRGB, data=tight_rgb)
                            with suppress_stderr():
                                tight_result = face_lm_crop.detect(mp_tight)
                            if tight_result and tight_result.face_landmarks:
                                lm_tight = tight_result.face_landmarks[0]
                                # MLP на ТОЧНЫХ landmarks (как в sight.py!)
                                emo, conf = compute_emotion(lm_tight, mlp, scaler, device)
                                if emo:
                                    info['emotion'] = emo
                                    info['emotion_confidence'] = conf
                                    info['emotion_method'] = 'mlp_2pass'
                            else:
                                # Tight crop не дал лица → HSEmotion на tight crop
                                if hse_classifier is not None:
                                    hse_name, hse_conf = hse_classifier.predict(tight_crop)
                                    info['emotion'] = HSE_RU_MAP.get(hse_name, hse_name)
                                    info['emotion_confidence'] = round(float(hse_conf), 2)
                                    info['emotion_method'] = 'hse_2pass'
                        except Exception:
                            pass
                else:
                    # Лицо слишком маленькое → HSEmotion CNN
                    if hse_classifier is not None:
                        try:
                            px, py = int(face_w * 0.2), int((max(face_ys)-min(face_ys)) * 0.2)
                            fx1 = max(0, int(min(face_xs)) - px)
                            fy1 = max(0, int(min(face_ys)) - py)
                            fx2 = min(hc_w, int(max(face_xs)) + px)
                            fy2 = min(hc_h, int(max(face_ys)) + py)
                            face_crop = head_crop_sd[fy1:fy2, fx1:fx2]
                            if face_crop.size > 0:
                                hse_name, hse_conf = hse_classifier.predict(face_crop)
                                info['emotion'] = HSE_RU_MAP.get(hse_name, hse_name)
                                info['emotion_confidence'] = round(float(hse_conf), 2)
                                info['emotion_method'] = 'hse'
                        except Exception:
                            pass

                # ── Identity: ArcFace на HD (5 kps из pass 1) ──
                try:
                    kps_sd = np.array([
                        [lm_pass1[FACELM_TO_ARCFACE['reye']].x * hc_w + hx1, lm_pass1[FACELM_TO_ARCFACE['reye']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['leye']].x * hc_w + hx1, lm_pass1[FACELM_TO_ARCFACE['leye']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['nose']].x * hc_w + hx1, lm_pass1[FACELM_TO_ARCFACE['nose']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['rmouth']].x * hc_w + hx1, lm_pass1[FACELM_TO_ARCFACE['rmouth']].y * hc_h + hy1],
                        [lm_pass1[FACELM_TO_ARCFACE['lmouth']].x * hc_w + hx1, lm_pass1[FACELM_TO_ARCFACE['lmouth']].y * hc_h + hy1],
                    ], dtype=np.float32)
                    kps_hd = kps_sd.copy()
                    kps_hd[:, 0] *= scale_x
                    kps_hd[:, 1] *= scale_y
                    
                    arcface_name, arcface_conf = compute_identity(rec, frame_hd, kps_hd, face_id_obj, gallery)
                    
                    face_bbox_sd = [min(face_xs) + hx1, min(face_ys) + hy1, max(face_xs) + hx1, max(face_ys) + hy1]
                    
                    if tracker is not None:
                        tracked_name, tracked_conf, tid, is_persistent = tracker.update(
                            now, face_bbox_sd, arcface_name, arcface_conf
                        )
                        info['identity'] = tracked_name
                        info['confidence'] = round(tracked_conf, 2)
                        info['tracked'] = is_persistent
                    else:
                        info['identity'] = arcface_name
                        info['confidence'] = round(arcface_conf, 2)
                    
                    info['facing'] = 'toward'
                except Exception:
                    pass

            elif hse_classifier is not None:
                # FaceLandmarker pass 1 не нашёл лицо → HSEmotion на всём head_crop
                try:
                    hse_name, hse_conf = hse_classifier.predict(head_crop_sd)
                    info['emotion'] = HSE_RU_MAP.get(hse_name, hse_name)
                    info['emotion_confidence'] = round(float(hse_conf), 2)
                    info['emotion_method'] = 'hse_fallback'
                except Exception:
                    pass

        if info['facing'] == 'unknown':
            info['facing'] = 'away'

        # ── Поза: VAE прямо с PoseLM landmarks ──
        try:
            pos, conf, rerr, logpz = compute_posture(
                body['pts_33'], vae, pose_clf, pose_names, device
            )
            info['posture'] = pos
            info['posture_confidence'] = conf
            info['recon_error'] = round(rerr, 4)
            info['plausibility'] = 'normal' if logpz < 14.3 else 'unusual'
        except Exception:
            pass

        person_infos.append(info)

    if tracker:
        tracker.cleanup(now)

    t_process = time.perf_counter() - t_process_start
    t_total_val = time.perf_counter() - t_total

    return person_infos, bodies, t_detect, t_process, t_total_val


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

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
    """Захват кадра 1280×720 → (frame_hd, frame_sd)."""
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

def estimate_lighting(frame):
    m = float(frame.mean())
    if m>150: return 'bright',m
    elif m>70: return 'normal',m
    return 'dim',m


def main():
    sys.stdout.reconfigure(line_buffering=True)

    cam_path = find_camera()
    if cam_path is None:
        print("camera: none\nerror: camera not found"); return
    print(f"camera: {cam_path}")

    frame_hd, frame_sd = capture_frame_hd(cam_path)
    if frame_hd is None:
        print("error: failed to capture frame"); return
    h_hd, w_hd = frame_hd.shape[:2]
    lighting_level, lighting_val = estimate_lighting(frame_sd)
    print(f"frame: {w_hd}x{h_hd} (HD) + {frame_sd.shape[1]}x{frame_sd.shape[0]} (SD)")
    print(f"lighting: {lighting_level} ({lighting_val:.0f})")

    # ── Модели ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    with suppress_all():
        face_id = FaceID(use_gpu=True)

        # ТОЛЬКО FaceLandmarker для кропов (нет full-frame детекции лиц)
        face_crop_opts = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=FACE_LM_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1, running_mode=vision.RunningMode.IMAGE,
        )
        face_lm_crop = vision.FaceLandmarker.create_from_options(face_crop_opts)

        pose_opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=POSE_MODEL,
                delegate=mp_python.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE, num_poses=3,
            output_segmentation_masks=False,
        )
        pose_lm = vision.PoseLandmarker.create_from_options(pose_opts)

        # VAE
        vae_ckpt = torch.load(VAE_PATH, map_location=device, weights_only=True)
        vae_cfg = vae_ckpt['config']
        vae = PoseVAE(
            input_dim=vae_cfg['input_dim'], latent_dim=vae_cfg['latent_dim'],
            hidden_dims=tuple(vae_cfg['hidden_dims']), dropout=vae_cfg['dropout'],
        ).to(device)
        vae.load_state_dict(vae_ckpt['model_state_dict']); vae.eval()

        with open(CLF_CONFIG) as f: clf_cfg = json.load(f)
        pose_clf = PoseClassifier(
            latent_dim=clf_cfg['latent_dim'], num_classes=clf_cfg['num_classes'],
            hidden_dim=clf_cfg['hidden_dim'], dropout=clf_cfg['dropout'],
        ).to(device)
        clf_ckpt = torch.load(CLF_PATH, map_location=device, weights_only=True)
        pose_clf.load_state_dict(clf_ckpt['model_state_dict']); pose_clf.eval()
        pose_names_ru = clf_cfg['class_names_ru']

        mlp = EmotionMLP()
        mlp.load_state_dict(torch.load(MLP_MODEL, map_location=device, weights_only=True))
        mlp.to(device); mlp.eval()
        with open(MLP_SCALER, 'rb') as f:
            scaler = pickle.load(f)

    # Distilled MLP (HSEmotion → MLP, 72.2%, <1ms CPU)
    DISTILLED_MODEL = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion_distilled.pth')
    DISTILLED_SCALER = os.path.join(ROOT, 'models', 'emotion', 'mlp_emotion_distilled_scaler.pkl')
    if os.path.exists(DISTILLED_MODEL):
        mlp_distilled = EmotionMLP()
        mlp_distilled.load_state_dict(torch.load(DISTILLED_MODEL, map_location=device, weights_only=True))
        mlp_distilled.to(device); mlp_distilled.eval()
        with open(DISTILLED_SCALER, 'rb') as f:
            scaler_distilled = pickle.load(f)
        print(f"  Distilled MLP: 72.2% test accuracy, <1ms CPU")
    else:
        mlp_distilled = None
        scaler_distilled = None

    # HSEmotion CNN (загружается отдельно, с GPU)
    print("loading HSEmotion...")
    with suppress_all():
        hse_classifier = HSEmotionClassifier(model_name='enet_b0_8_best_vgaf', use_gpu=True)
    print(f"  HSEmotion: {len(hse_classifier.class_names)} классов, device={hse_classifier.device}")

    # Галерея
    gallery = {}
    for fname in sorted(os.listdir(GALLERY_DIR)):
        if not fname.endswith('.jpg'): continue
        name_key = fname[2:-4]
        img = cv2.imread(os.path.join(GALLERY_DIR, fname))
        if img is None: continue
        with suppress_all(): emb = face_id.get_embedding(img)
        if emb is not None: gallery[NAME_MAP.get(name_key, name_key)] = emb

    # Прогрев моделей
    print("warming up...")
    h_sd, w_sd = frame_sd.shape[:2]
    frame_sd_rgb = cv2.cvtColor(frame_sd, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_sd_rgb)
    _ = pose_lm.detect(mp_img)
    _ = face_id.app.get(frame_sd)

    # Прогрев FaceLandmarker на head_crop
    bodies = detect_bodies(pose_lm, frame_sd_rgb, w_sd, h_sd)
    if bodies:
        head_crop_sd, _, _ = head_crop_from_body(frame_sd, bodies[0]['bbox'])
        if head_crop_sd is not None:
            head_rgb = cv2.cvtColor(head_crop_sd, cv2.COLOR_BGR2RGB)
            mp_head = mp.Image(image_format=mp.ImageFormat.SRGB, data=head_rgb)
            _ = face_lm_crop.detect(mp_head)

    # ── Бенчмарк: DUAL (Pose-first) ──
    print("\n" + "=" * 60)
    print("  POSE-FIRST + TWO-PASS FaceLM")
    print("=" * 60)

    tracker = PersonTracker()
    times_detect, times_process, times_total = [], [], []
    all_results = None
    for run in range(5):
        person_infos, bodies, t_d, t_p, t_t = run_dual_pipeline(
            frame_hd, frame_sd, face_lm_crop, pose_lm, face_id, gallery,
            mlp, scaler, vae, pose_clf, pose_names_ru, device,
            mlp_distilled=mlp_distilled, scaler_distilled=scaler_distilled,
            hse_classifier=hse_classifier, tracker=tracker, timestamp=time.time()
        )
        times_detect.append(t_d)
        times_process.append(t_p)
        times_total.append(t_t)
        all_results = (person_infos, bodies)

    print(f"\n  Фаза 1 (PoseLM, full frame):  {avg_ms(times_detect):>6.1f}ms")
    print(f"  Фаза 2+3 (обработка, seq):    {avg_ms(times_process):>6.1f}ms")
    print(f"  ─────────────────────────────────")
    print(f"  ИТОГО:                        {avg_ms(times_total):>6.1f}ms")
    print(f"  FPS:                          {1000/avg_ms(times_total):>6.0f}")

    # ── Результаты ──
    if all_results is None:
        print("\n── Нет результатов (камера пуста) ──")
        return
    person_infos, bodies = all_results
    n_toward = sum(1 for info in person_infos if info['facing'] == 'toward')
    n_away = sum(1 for info in person_infos if info['facing'] == 'away')
    print(f"\n── Результаты ──")
    print(f"  Тел: {len(bodies)} (лицом: {n_toward}, спиной: {n_away})")

    for i, info in enumerate(person_infos, 1):
        tracked_mark = ' [tracked]' if info.get('tracked') else ''
        print(f"\n  Персона {i} [{info['facing']}{tracked_mark}]:")
        if info['identity'] != 'unknown':
            print(f"    identity: {info['identity']} ({info['confidence']:.2f})")
        else:
            reason = '(спиной)' if info['facing'] == 'away' else '(низкий confidence)'
            print(f"    identity: unknown {reason}")
        if info['emotion']:
            method = f" [{info.get('emotion_method', '?')}]" if info.get('emotion_method') else ''
            print(f"    emotion: {info['emotion']} ({info['emotion_confidence']:.2f}){method}")
        if info['posture']:
            print(f"    posture: {info['posture']} ({info['posture_confidence']:.2f})")
            if info.get('recon_error'):
                print(f"    recon_error: {info['recon_error']:.4f}  plausibility: {info['plausibility']}")

    # ── Сравнение с текущей архитектурой ──
    if times_total:
        t_dual = sum(times_total) / len(times_total) * 1000
        print(f"\n── Сравнение архитектур ──")
        print(f"  Pose-first v4:   PoseLM → head_crop → 2-pass FaceLM → MLP")
        print(f"  Pass 1: FaceLM(head_crop) → face_bbox + 5 kps")
        print(f"  Pass 2: FaceLM(tight_crop) → точные landmarks → MLP (≡ sight.py)")
        print(f"  Far: лицо ≤{FACE_SIZE_THRESHOLD}px → HSEmotion CNN")
        print(f"  Current:         SCRFD → FaceLM(crop) → PoseLM(crop) → ~40ms")
        print(f"  Детекторов:      1 (PoseLM) vs 2 (SCRFD + FaceLM)")
        print(f"  Моделей:         4 (PoseLM + FaceLM + ArcFace + MLP/VAE)")
        print(f"  Лица на дальности: FaceLM на head_crop (как старый sight.py)")


def avg_ms(times):
    return sum(times) / len(times) * 1000


if __name__ == '__main__':
    main()
