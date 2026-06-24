#!/usr/bin/env python3
"""mvm_bridge — enriched JSON bridge for mvm-cli.

Full pipeline: FaceID + HSEmotion + PoseVAE posture + 3D Workspace.
Outputs one JSON line per frame.

Usage:
    venv/bin/python3 tools/mvm_bridge.py              # Full pipeline (GPU)
    venv/bin/python3 tools/mvm_bridge.py --cpu        # CPU-only (no emotion GPU)
    venv/bin/python3 tools/mvm_bridge.py --no-posture # Face + emotion only
"""

import argparse, io, json, os, sys, time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--cpu", action="store_true")
parser.add_argument("--json-only", action="store_true")
parser.add_argument("--no-posture", action="store_true")
parser.add_argument("--max-frames", type=int, default=0)
parser.add_argument("--write-frame", type=str, default=None, help="Save current frame to this path on each tick")
args = parser.parse_args()

if args.json_only:
    sys.stderr = open(os.devnull, "w")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

# ── Log suppression ────────────────────────────────────────
for name in ("insightface", "onnxruntime", "absl"):
    import logging
    logging.getLogger(name).setLevel(logging.ERROR)
os.environ.setdefault("ONNXRUNTIME_LOG_LEVEL", "3")

# ── PyTorch compat fix ─────────────────────────────────────
import torch as _torch
_orig_load = _torch.load
def _load_fixed(*a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(*a, **kw)
_torch.load = _load_fixed

# ═══════════════════════════════════════════════════════════
# FaceID
# ═══════════════════════════════════════════════════════════

with io.StringIO() as _b:
    with __import__("contextlib").redirect_stdout(_b):
        from core.face_id import FaceID

face_id = FaceID()

gallery_dir = ROOT / "models" / "references"
NAME_MAP = {"V": "Вадим", "D": "Даня", "N": "Никита", "B": "Белла"}
gallery = {}
if gallery_dir.exists():
    # Группируем multi-angle файлы: p_V_01.jpg, p_V_02.jpg → Вадим
    ref_paths: dict[str, list[str]] = {}
    for f in sorted(gallery_dir.glob("p_*.jpg")):
        stem = f.stem[2:]  # V_01 или V
        base = stem.split("_")[0]
        if base in NAME_MAP:
            ref_paths.setdefault(NAME_MAP[base], []).append(str(f))
    if ref_paths:
        gallery = face_id.build_gallery(ref_paths)

# ═══════════════════════════════════════════════════════════
# Emotion: HSEmotion (GPU) — works on faces of any size
# ═══════════════════════════════════════════════════════════

hse = None
HSE_EMOTION_MAP = {
    "Anger": "Злость", "Disgust": "Отвращение", "Fear": "Страх",
    "Happiness": "Радость", "Neutral": "Нейтральная", "Sadness": "Грусть",
    "Surprise": "Удивление",
}

if not args.cpu and _torch.cuda.is_available():
    try:
        with io.StringIO() as _b:
            with __import__("contextlib").redirect_stdout(_b):
                from core.emotion_hse import HSEmotionClassifier
                hse = HSEmotionClassifier(model_name="enet_b0_8_best_vgaf", use_gpu=True)
        sys.stderr.write("  bridge: HSEmotion GPU ready\n")
    except Exception as e:
        sys.stderr.write(f"  bridge: HSEmotion failed ({e})\n")
sys.stderr.flush()

# ═══════════════════════════════════════════════════════════
# Posture: PoseLandmarker → PoseVAE → PoseClassifier
# ═══════════════════════════════════════════════════════════

import mediapipe as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options as _mp_base

pose_lm = None
pose_vae = None
pose_classifier = None
pose_class_names = None
ws_3d = None

if not args.no_posture:
    pose_lm_path = str(ROOT / "models" / "mediapipe" / "pose_landmarker.task")
    if os.path.exists(pose_lm_path):
        pose_lm_opts = vision.PoseLandmarkerOptions(
            base_options=_mp_base.BaseOptions(
                model_asset_path=pose_lm_path,
                delegate=_mp_base.BaseOptions.Delegate.CPU),
            running_mode=vision.RunningMode.IMAGE,
            num_poses=3,
            output_segmentation_masks=False)
        pose_lm = vision.PoseLandmarker.create_from_options(pose_lm_opts)

        from core.pose_vae import PoseVAE
        from core.pose_heads import PoseClassifier
        from core.pose_encoder import prepare_vae_input
        from core.workspace_3d import Workspace3D

        vae_path = ROOT / "models" / "emotion" / "vposer_coco33.pth"
        vae_cfg_path = ROOT / "models" / "emotion" / "vposer_coco33_config.json"
        clf_path = ROOT / "models" / "emotion" / "pose_classifier.pth"
        clf_cfg_path = ROOT / "models" / "emotion" / "pose_classifier_config.json"

        DEV = _torch.device("cpu")

        with open(vae_cfg_path) as f:
            vcfg = json.load(f)
        pose_vae = PoseVAE(input_dim=vcfg["input_dim"], latent_dim=vcfg["latent_dim"]).to(DEV)
        vae_sd = _torch.load(str(vae_path), map_location=DEV, weights_only=True)
        pose_vae.load_state_dict(vae_sd["model_state_dict"])
        pose_vae.eval()

        with open(clf_cfg_path) as f:
            ccfg = json.load(f)
        pose_class_names = ccfg.get("class_names_ru", ccfg["class_names"])
        pose_classifier = PoseClassifier(
            latent_dim=vcfg["latent_dim"], num_classes=ccfg["num_classes"]).to(DEV)
        clf_sd = _torch.load(str(clf_path), map_location=DEV, weights_only=True)
        pose_classifier.load_state_dict(clf_sd["model_state_dict"])
        pose_classifier.eval()

        ws_3d = Workspace3D(image_width=640, image_height=480)

        sys.stderr.write(f"  bridge: pose pipeline ready ({len(pose_class_names)} classes)\n")
    else:
        sys.stderr.write("  bridge: pose_landmarker.task not found\n")
    sys.stderr.flush()

# ═══════════════════════════════════════════════════════════
# Camera
# ═══════════════════════════════════════════════════════════

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    sys.stderr.write("FATAL: Cannot open camera\n")
    sys.exit(1)

for _ in range(5):
    cap.read()  # warm-up

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════

def pose_to_body_bbox(landmarks, w, h):
    """Extract body bbox from pose landmarks with torso visibility check."""
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
    """VAE + classifier posture prediction."""
    raw = prepare_vae_input(landmarks, use_invariants=False)
    with _torch.no_grad():
        x = _torch.tensor(raw, dtype=_torch.float32, device=DEV).unsqueeze(0)
        z = pose_vae.encode_to_z(x)
        recon = pose_vae.decode(z)
        mse = _torch.nn.functional.mse_loss(recon, x).item()
        logpz = pose_vae.log_prob_z(x).item()
        logits = pose_classifier(z)
        probs = _torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    return pose_class_names[idx], round(float(probs[idx]), 3), round(mse, 4), round(logpz, 2)


def match_face_to_body(face_bbox, body_bboxes):
    """Match face to body by center-point containment with margin."""
    fx = (face_bbox[0] + face_bbox[2]) / 2
    fy = (face_bbox[1] + face_bbox[3]) / 2
    m = 0.15  # margin
    for i, bb in enumerate(body_bboxes):
        if (bb[0] - m <= fx <= bb[2] + m) and (bb[1] - m <= fy <= bb[3] + m):
            return i
    return None

# ═══════════════════════════════════════════════════════════
# Main loop
# ═══════════════════════════════════════════════════════════

frame_n = 0
t0 = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        frame_n += 1
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # ── Posture + 3D ──────────────────────────────────
        body_bboxes = []
        posture_results = []
        pose_landmarks_all = []

        if pose_lm is not None:
            mp_img = mp_python.Image(image_format=mp_python.ImageFormat.SRGB, data=frame_rgb)
            pose_result = pose_lm.detect(mp_img)
            if pose_result.pose_landmarks:
                pose_landmarks_all = pose_result.pose_landmarks
                for lm in pose_landmarks_all:
                    bb = pose_to_body_bbox(lm, w, h)
                    if bb is not None:
                        body_bboxes.append(bb)
                        pname, pconf, rerr, logpz_val = compute_posture(lm)
                        posture_results.append({
                            "name": pname, "confidence": pconf,
                            "recon_error": rerr,
                            "plausibility": "normal" if logpz_val < 14.3 else "unusual",
                        })
                # 3D calibration (first valid body)
                if ws_3d is not None:
                    for lm in pose_result.pose_landmarks:
                        pts = np.array([[p.x, p.y, p.z, p.visibility] for p in lm], dtype=np.float32)
                        if ws_3d.calibrate(pts):
                            break

        # ── Face detection + identity ─────────────────────
        faces = face_id.app.get(frame)
        detections = []
        for face in faces:
            bbox = face.bbox.astype(float)
            x1 = max(0.0, bbox[0] / w)
            y1 = max(0.0, bbox[1] / h)
            x2 = min(1.0, bbox[2] / w)
            y2 = min(1.0, bbox[3] / h)

            # Identity
            emb = face.embedding
            name, sim = face_id.identify(emb, gallery)
            identity = name
            identity_confidence = round(float(sim), 3) if sim is not None else None

            # Emotion: HSEmotion on face crop
            fx1 = int(x1 * w); fy1 = int(y1 * h); fx2 = int(x2 * w); fy2 = int(y2 * h)
            emotion = None; emotion_confidence = None
            if hse is not None:
                face_crop = frame[fy1:fy2, fx1:fx2]
                if face_crop.size > 0:
                    try:
                        ename, econf = hse.predict(face_crop)
                        emotion = HSE_EMOTION_MAP.get(ename, ename)
                        emotion_confidence = round(float(econf), 3)
                    except Exception:
                        pass

            detections.append({
                "x1": round(x1, 4), "y1": round(y1, 4),
                "x2": round(x2, 4), "y2": round(y2, 4),
                "identity": identity,
                "identity_confidence": round(identity_confidence, 4) if identity_confidence else None,
                "emotion": emotion,
                "emotion_confidence": emotion_confidence,
            })

        # ── Match faces → bodies (posture + 3D) ───────────
        for det in detections:
            face_bbox = [det["x1"], det["y1"], det["x2"], det["y2"]]
            body_idx = match_face_to_body(face_bbox, body_bboxes)
            if body_idx is not None and body_idx < len(posture_results):
                pr = posture_results[body_idx]
                det["posture"] = pr["name"]
                det["posture_confidence"] = pr["confidence"]
                det["recon_error"] = pr["recon_error"]
                det["plausibility"] = pr["plausibility"]

            # 3D Workspace
            if ws_3d is not None and ws_3d.is_calibrated and body_idx is not None \
                    and body_idx < len(pose_landmarks_all):
                try:
                    body_lm = pose_landmarks_all[body_idx]
                    pts = np.array([[p.x, p.y, p.z, p.visibility] for p in body_lm], dtype=np.float32)
                    body_xyz = ws_3d.to_metric(pts)
                    hip_center = (body_xyz[23] + body_xyz[24]) / 2.0
                    det["distance"] = round(float(hip_center[2]), 2)
                    det["position_x"] = round(float(hip_center[0]), 2)
                    det["position_y"] = round(float(hip_center[1]), 2)
                    det["position_z"] = round(float(hip_center[2]), 2)
                    det["ws_confidence"] = round(ws_3d.confidence, 2)
                except Exception:
                    pass

        elapsed = time.time() - t0
        msg = {"frame": frame_n, "elapsed": round(elapsed, 4), "detections": detections}

        # Write frame to disk for Rust native enrichment (--write-frame)
        if args.write_frame:
            try:
                cv2.imwrite(args.write_frame, frame)
                msg["frame_file"] = args.write_frame
            except Exception:
                pass

        sys.stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        sys.stdout.flush()

        if args.max_frames and frame_n >= args.max_frames:
            break

except BrokenPipeError:
    pass
except KeyboardInterrupt:
    pass
finally:
    cap.release()
