#!/usr/bin/env python3
"""sight_bridge — streaming enriched JSON bridge for Rust sightd.

Outputs JSON lines with identity, emotion, and posture per detection.

Usage:
    venv/bin/python3 tools/sight_bridge.py              # GPU full pipeline
    venv/bin/python3 tools/sight_bridge.py --cpu        # CPU detectors only
    venv/bin/python3 tools/sight_bridge.py --no-posture # skip posture (faster)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── CLI ──────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="sightd camera bridge")
parser.add_argument("--cpu", action="store_true", help="Use CPU detectors (no GPU)")
parser.add_argument("--json-only", action="store_true", help="Suppress stderr")
parser.add_argument("--no-posture", action="store_true", help="Skip posture analysis")
parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames")
args = parser.parse_args()

if args.json_only:
    sys.stderr = open(os.devnull, "w")

# ── Paths ────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
VENV = ROOT / "venv"
sys.path.insert(0, str(ROOT))

# ── Imports ──────────────────────────────────────────────
import cv2  # noqa: E402
import numpy as np  # noqa: E402

# Suppress logs
for name in ("insightface", "onnxruntime", "absl"):
    import logging
    logging.getLogger(name).setLevel(logging.ERROR)
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ── Face detector + identity ─────────────────────────────
gallery = {}
face_id = None
gpu_available = False

if not args.cpu:
    try:
        from core.face_id import FaceID
        face_id = FaceID()
        gpu_available = True
    except Exception:
        sys.stderr.write("  bridge: GPU unavailable, using CPU fallbacks\n")
        sys.stderr.flush()

if gpu_available and face_id is not None:
    face_detector = "SCRFD GPU"
    # Build identity gallery
    ref_dir = ROOT / "models" / "references"
    name_map = {"V": "Вадим", "D": "Даня", "N": "Никита"}
    ref_paths = {}
    for fname in sorted(ref_dir.glob("p_*.jpg")):
        key = fname.stem[2:]  # p_V.jpg → V
        if key in name_map:
            ref_paths[name_map[key]] = str(fname)
    if ref_paths:
        gallery = face_id.build_gallery(ref_paths)
        sys.stderr.write(f"  bridge: gallery loaded ({len(gallery)} people)\n")
    else:
        sys.stderr.write("  bridge: no reference images for identity\n")
else:
    yunet_model = str(ROOT / "models" / "yunet" / "face_detection_yunet_2023mar.onnx")
    if not os.path.exists(yunet_model):
        sys.stderr.write(f"  bridge: YuNet model not found at {yunet_model}\n")
        sys.exit(1)
    detector = cv2.FaceDetectorYN.create(
        yunet_model, "", (640, 480), score_threshold=0.5, nms_threshold=0.3, top_k=10
    )
    face_detector = "YuNet CPU"

# ── FaceLandmarker + MLP emotion ─────────────────────────
import mediapipe as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as _mp_base

face_lm_path = str(ROOT / "models" / "mediapipe" / "face_landmarker.task")
face_lm_opts = vision.FaceLandmarkerOptions(
    base_options=_mp_base.BaseOptions(
        model_asset_path=face_lm_path,
        delegate=_mp_base.BaseOptions.Delegate.CPU,
    ),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_lm_opts)

# MLP model
import torch
import pickle

mlp_path = str(ROOT / "models" / "emotion" / "mlp_emotion.pth")
scaler_path = str(ROOT / "models" / "emotion" / "mlp_emotion_scaler.pkl")
config_path = str(ROOT / "models" / "emotion" / "mlp_emotion_config.json")

with open(config_path) as f:
    mlp_config = json.load(f)
emotion_classes = mlp_config["class_names"]

with open(scaler_path, "rb") as f:
    mlp_scaler = pickle.load(f)

class MLPEmotion(torch.nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.BatchNorm1d(input_dim),
            torch.nn.Linear(input_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.net(x)

mlp_device = torch.device("cuda" if gpu_available else "cpu")
mlp_model = MLPEmotion(468, len(emotion_classes)).to(mlp_device)
mlp_model.load_state_dict(torch.load(mlp_path, map_location=mlp_device, weights_only=True))
mlp_model.eval()

LGB_INDICES = list(range(468))
RU_EMOTION = {
    "anger": "Злость", "disgust": "Отвращение", "fear": "Страх",
    "happy": "Радость", "neutral": "Нейтральная", "sad": "Грусть",
    "surprise": "Удивление",
}

# ── HSEmotion fallback (GPU) for small/distant faces ──────
hse_classifier = None
HSE_GPU_AVAILABLE = torch.cuda.is_available()
if HSE_GPU_AVAILABLE:
    try:
        # PyTorch 2.6 fix: monkey-patch torch.load
        import torch as _torch
        _orig_load = _torch.load
        def _load_with_weights_only_false(*args, **kwargs):
            kwargs.setdefault('weights_only', False)
            return _orig_load(*args, **kwargs)
        _torch.load = _load_with_weights_only_false

        # HSEmotion prints model info to stdout — capture it
        import io, contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            from core.emotion_hse import HSEmotionClassifier
            hse_classifier = HSEmotionClassifier(model_name='enet_b0_8_best_vgaf', use_gpu=True)

        sys.stderr.write("  bridge: HSEmotion GPU fallback ready\n")
    except Exception as e:
        sys.stderr.write(f"  bridge: HSEmotion load failed ({e}), skipping\n")
    sys.stderr.flush()

# Face size threshold: below this, use HSEmotion instead of MLP
FACE_SIZE_THRESHOLD = 70  # pixels

def compute_emotion_mlp(landmarks, crop_w, crop_h):
    pts = np.array([[p.x * crop_w, p.y * crop_h] for p in landmarks], dtype=np.float32)
    nose_tip = pts[4]
    left_eye = pts[33]
    right_eye = pts[263]
    interocular = np.linalg.norm(left_eye - right_eye)
    if interocular < 1.0:
        interocular = 1.0
    dists = np.linalg.norm(pts - nose_tip, axis=1) / interocular
    features = dists[LGB_INDICES].reshape(1, -1)
    features = mlp_scaler.transform(features)
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=mlp_device)
        logits = mlp_model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    name = RU_EMOTION.get(emotion_classes[idx], emotion_classes[idx])
    return name, float(probs[idx])

# ── Pose pipeline ────────────────────────────────────────
pose_enabled = not args.no_posture
pose_lm = None
pose_vae = None
pose_classifier = None
pose_class_names = None

if pose_enabled:
    pose_lm_path = str(ROOT / "models" / "mediapipe" / "pose_landmarker.task")
    if os.path.exists(pose_lm_path):
        pose_lm_opts = vision.PoseLandmarkerOptions(
            base_options=_mp_base.BaseOptions(
                model_asset_path=pose_lm_path,
                delegate=_mp_base.BaseOptions.Delegate.CPU,
            ),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,
            output_segmentation_masks=False,
        )
        pose_lm = vision.PoseLandmarker.create_from_options(pose_lm_opts)

        from core.pose_vae import PoseVAE
        from core.pose_heads import PoseClassifier
        from core.pose_encoder import prepare_vae_input

        vae_path = str(ROOT / "models" / "emotion" / "vposer_coco33.pth")
        vae_config_path = str(ROOT / "models" / "emotion" / "vposer_coco33_config.json")
        clf_path = str(ROOT / "models" / "emotion" / "pose_classifier.pth")
        clf_config_path = str(ROOT / "models" / "emotion" / "pose_classifier_config.json")

        pose_device = torch.device("cpu")

        with open(vae_config_path) as f:
            vae_cfg = json.load(f)
        pose_vae = PoseVAE(input_dim=vae_cfg["input_dim"], latent_dim=vae_cfg["latent_dim"]).to(pose_device)
        vae_sd = torch.load(vae_path, map_location=pose_device, weights_only=True)
        pose_vae.load_state_dict(vae_sd["model_state_dict"])
        pose_vae.eval()

        with open(clf_config_path) as f:
            clf_cfg = json.load(f)
        pose_class_names = clf_cfg.get("class_names_ru", clf_cfg["class_names"])
        pose_classifier = PoseClassifier(
            latent_dim=vae_cfg["latent_dim"], num_classes=clf_cfg["num_classes"]
        ).to(pose_device)
        clf_sd = torch.load(clf_path, map_location=pose_device, weights_only=True)
        pose_classifier.load_state_dict(clf_sd["model_state_dict"])
        pose_classifier.eval()

        sys.stderr.write(f"  bridge: pose pipeline ready ({len(pose_class_names)} classes)\n")
    else:
        sys.stderr.write("  bridge: pose_landmarker.task not found, posture disabled\n")
    sys.stderr.flush()

# ── Camera ───────────────────────────────────────────────
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    cam = cv2.VideoCapture(1)
if not cam.isOpened():
    sys.stderr.write("  bridge: no camera found\n")
    sys.exit(1)

cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FPS, 30)
sys.stderr.write(f"  bridge: camera ready ({face_detector})\n")
sys.stderr.flush()

for _ in range(10):
    cam.read()

# ── Helpers ──────────────────────────────────────────────
def pose_to_body_bbox(landmarks, w, h):
    xs = [p.x * w for p in landmarks]
    ys = [p.y * h for p in landmarks]
    if not xs:
        return None
    torso_idx = [11, 12, 23, 24]
    vis_torso = np.mean([landmarks[i].visibility for i in torso_idx])
    if vis_torso < 0.35:
        return None
    return [max(0, min(xs)) / w, max(0, min(ys)) / h,
            min(w, max(xs)) / w, min(h, max(ys)) / h]

def compute_posture(landmarks):
    raw = prepare_vae_input(landmarks, use_invariants=False)
    with torch.no_grad():
        x = torch.tensor(raw, dtype=torch.float32, device=pose_device).unsqueeze(0)
        z = pose_vae.encode_to_z(x)
        logits = pose_classifier(z)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return pose_class_names[idx], round(float(probs[idx]), 3)

def match_face_to_body(face_bbox, body_bboxes):
    fx = (face_bbox[0] + face_bbox[2]) / 2
    fy = (face_bbox[1] + face_bbox[3]) / 2
    for i, bb in enumerate(body_bboxes):
        m = 0.15
        if (bb[0] - m <= fx <= bb[2] + m) and (bb[1] - m <= fy <= bb[3] + m):
            return i
    return None

# ── Main loop ────────────────────────────────────────────
frame_id = 0
start_time = time.monotonic()

try:
    while True:
        ok, frame = cam.read()
        if not ok:
            time.sleep(0.1)
            continue

        frame_id += 1
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose detection
        body_bboxes = []
        posture_results = []
        if pose_lm is not None:
            mp_image = mp_python.Image(image_format=mp_python.ImageFormat.SRGB, data=frame_rgb)
            pose_result = pose_lm.detect(mp_image)
            if pose_result.pose_landmarks:
                for lm in pose_result.pose_landmarks:
                    bb = pose_to_body_bbox(lm, w, h)
                    if bb is not None:
                        body_bboxes.append(bb)
                        name, conf = compute_posture(lm)
                        posture_results.append({"name": name, "confidence": conf})

        # Face detection
        detections = []
        if gpu_available and face_id is not None:
            raw_faces = face_id.app.get(frame)
            for face in raw_faces:
                x1, y1, x2, y2 = face.bbox.astype(int)
                detections.append({"x1": x1 / w, "y1": y1 / h,
                                   "x2": x2 / w, "y2": y2 / h})
        else:
            detector.setInputSize((w, h))
            _ret, faces = detector.detect(frame)
            if faces is not None:
                for face in faces:
                    x, y, fw, fh = face[:4].astype(int)
                    detections.append({"x1": x / w, "y1": y / h,
                                       "x2": (x + fw) / w, "y2": (y + fh) / h})

        # Identity
        if gpu_available and face_id is not None and gallery:
            for det in detections:
                fx1 = int(det["x1"] * w)
                fy1 = int(det["y1"] * h)
                fx2 = int(det["x2"] * w)
                fy2 = int(det["y2"] * h)
                crop = frame[fy1:fy2, fx1:fx2]
                if crop.size > 0:
                    emb = face_id.get_embedding(crop)
                    if emb is not None:
                        name, sim = face_id.identify(emb, gallery)
                        det["identity"] = name if name else "unknown"
                        det["identity_confidence"] = round(float(sim), 3)

        # Emotion — MLP for close faces, HSEmotion for distant
        for det in detections:
            # Percentage-based padding (like sight.py: bbox + 20%)
            fw = int((det["x2"] - det["x1"]) * w)
            fh = int((det["y2"] - det["y1"]) * h)
            px = int(fw * 0.3)
            py = int(fh * 0.3)
            fx1 = max(0, int(det["x1"] * w) - px)
            fy1 = max(0, int(det["y1"] * h) - py)
            fx2 = min(w, int(det["x2"] * w) + px)
            fy2 = min(h, int(det["y2"] * h) + py)
            face_crop = frame_rgb[fy1:fy2, fx1:fx2]

            if face_crop.size == 0:
                continue

            face_w = fx2 - fx1

            # Try FaceLandmarker → MLP first
            mp_face = mp_python.Image(image_format=mp_python.ImageFormat.SRGB, data=face_crop)
            lm_result = face_landmarker.detect(mp_face)

            if lm_result.face_landmarks and face_w >= FACE_SIZE_THRESHOLD:
                # Close face: MLP (fast, CPU)
                name, conf = compute_emotion_mlp(
                    lm_result.face_landmarks[0], face_crop.shape[1], face_crop.shape[0])
                det["emotion"] = name
                det["emotion_confidence"] = round(conf, 3)
            elif hse_classifier is not None:
                # Distant/small face: HSEmotion CNN (GPU)
                try:
                    hse_name, hse_conf = hse_classifier.predict(face_crop)
                    det["emotion"] = hse_name
                    det["emotion_confidence"] = round(float(hse_conf), 3)
                except Exception:
                    pass  # silent fallback — emotion stays None

        # Match faces → bodies (posture)
        for det in detections:
            face_bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            body_idx = match_face_to_body(face_bbox, body_bboxes)
            if body_idx is not None and body_idx < len(posture_results):
                det["posture"] = posture_results[body_idx]["name"]
                det["posture_confidence"] = posture_results[body_idx]["confidence"]

        elapsed = time.monotonic() - start_time
        msg = {"frame": frame_id, "elapsed": round(elapsed, 3), "detections": detections}
        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

        if args.max_frames and frame_id >= args.max_frames:
            break

except BrokenPipeError:
    pass
except KeyboardInterrupt:
    pass
finally:
    cam.release()
