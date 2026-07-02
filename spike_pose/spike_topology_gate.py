"""Spike 7.5.0 — TOPOLOGY GATE.

Proves the OpenCV Zoo BlazePose ONNX (person_det -> pose) produces 33 landmarks
in the SAME topology the PoseVAE (vposer_coco33) was trained on, by:
  Path A: OpenCV Zoo ONNX -> 33 landmarks -> Python PoseVAE -> class
  Path B: bridge MediaPipe .task -> 33 landmarks -> Python PoseVAE -> class   (known-good)
Gate passes if Path A yields a SANE class with low recon_error AND agrees with
Path B (same class, landmarks numerically close).

Run from spike_pose/ with the project venv.
"""
import os, sys, json
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)            # project root
sys.path.insert(0, HERE)                # mp_persondet / mp_pose
sys.path.insert(0, ROOT)                # core.*

from mp_persondet import MPPersonDet
from mp_pose import MPPose

import torch
from core.pose_vae import PoseVAE
from core.pose_heads import PoseClassifier
from core.pose_encoder import prepare_vae_input

DEV = torch.device("cpu")
M = os.path.join(ROOT, "models")

# ── load PoseVAE + classifier exactly like the bridge ──────────────────────
vcfg = json.load(open(f"{M}/emotion/vposer_coco33_config.json"))
ccfg = json.load(open(f"{M}/emotion/pose_classifier_config.json"))
vae = PoseVAE(input_dim=vcfg["input_dim"], latent_dim=vcfg["latent_dim"]).to(DEV)
vae.load_state_dict(torch.load(f"{M}/emotion/vposer_coco33.pth", map_location=DEV,
                               weights_only=True)["model_state_dict"]); vae.eval()
clf = PoseClassifier(latent_dim=vcfg["latent_dim"], num_classes=ccfg["num_classes"]).to(DEV)
clf.load_state_dict(torch.load(f"{M}/emotion/pose_classifier.pth", map_location=DEV,
                               weights_only=True)["model_state_dict"]); clf.eval()
CLASS = ccfg["class_names"]   # EN labels (G7)

# ── ONNX models (OpenCV Zoo) ───────────────────────────────────────────────
det = MPPersonDet(modelPath=f"{M}/person_detection_mediapipe_2023mar.onnx",
                  nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
pose = MPPose(modelPath=f"{M}/pose_estimation_mediapipe_2023mar.onnx", confThreshold=0.5)

# ── bridge MediaPipe .task (known-good reference) ──────────────────────────
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
pose_lm = mp_vision.PoseLandmarker.create_from_options(mp_vision.PoseLandmarkerOptions(
    base_options=mp_python.BaseOptions(
        model_asset_path=f"{M}/mediapipe/pose_landmarker.task",
        delegate=mp_python.BaseOptions.Delegate.CPU),
    running_mode=mp_vision.RunningMode.IMAGE, num_poses=3,
    output_segmentation_masks=False))


class LM:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(self, x, y, z, v): self.x, self.y, self.z, self.visibility = x, y, z, v


def classify(landmarks33):
    """landmarks33: list of 33 LM (MediaPipe-normalized) -> (label, conf, recon_err)."""
    raw = prepare_vae_input(landmarks33, use_invariants=False)
    with torch.no_grad():
        x = torch.tensor(raw, dtype=torch.float32, device=DEV).unsqueeze(0)
        z = vae.encode_to_z(x)
        recon = vae.decode(z)
        mse = torch.nn.functional.mse_loss(recon, x).item()
        probs = torch.softmax(clf(z), dim=1).cpu().numpy()[0]
    i = int(np.argmax(probs))
    return CLASS[i], float(probs[i]), mse, probs


def zoo_landmarks(image):
    """Run OpenCV Zoo person_det -> pose; return 33 LM in MediaPipe-normalized coords."""
    persons = det.infer(image)
    if persons is None or len(persons) == 0:
        return None, None
    h, w = image.shape[:2]
    best = max(persons, key=lambda p: float(p[-1]))
    res = pose.infer(image, best)
    if res is None:
        return None, None
    lms = res[1]                       # 39x5: x,y(pixels), z(rel hip), vis, presence
    out = []
    for k in range(33):
        x, y, z, vis = lms[k][0], lms[k][1], lms[k][2], lms[k][3]
        # MediaPipe NormalizedLandmark convention: x/w, y/h, z/w
        out.append(LM(x / w, y / h, z / w, float(vis)))
    return out, float(best[-1])


def mp_landmarks(image):
    """Run bridge MediaPipe .task; return 33 LM (already normalized)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r = pose_lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if not r.pose_landmarks:
        return None
    lm = r.pose_landmarks[0]
    return [LM(p.x, p.y, p.z, p.visibility) for p in lm[:33]]


def xy_diff(a, b):
    """mean abs diff of normalized x,y over 33 landmarks."""
    pa = np.array([[l.x, l.y] for l in a]); pb = np.array([[l.x, l.y] for l in b])
    return float(np.mean(np.abs(pa - pb)))


import glob
IMAGES = sys.argv[1:] or [f"{M}/references/p_V_01.jpg", f"{M}/references/p_V_03.jpg",
                          f"{M}/references/p_V_05.jpg", f"{M}/references/p_V.jpg"]

print(f"EN classes: {CLASS}\n")
print(f"{'image':<22}{'ZOO class':<18}{'conf':>6}{'recon':>9}   | "
      f"{'MP class':<18}{'conf':>6}{'recon':>9}  xy_diff")
print("-" * 110)
agree = total = 0
for path in IMAGES:
    im = cv2.imread(path)
    if im is None:
        print(f"{os.path.basename(path):<22}MISSING"); continue
    za, zscore = zoo_landmarks(im)
    mb = mp_landmarks(im)
    name = os.path.basename(path)
    if za is None:
        print(f"{name:<22}ZOO: no person/pose"); continue
    if mb is None:
        print(f"{name:<22}MP: no pose"); continue
    zc, zf, zr, _ = classify(za)
    mc, mf, mr, _ = classify(mb)
    d = xy_diff(za, mb)
    ok = "✓" if zc == mc else "✗"
    total += 1; agree += (zc == mc)
    print(f"{name:<22}{zc:<18}{zf:>6.2f}{zr:>9.4f}   | "
          f"{mc:<18}{mf:>6.2f}{mr:>9.4f}  {d:>6.3f} {ok}")

print("-" * 110)
print(f"GATE: class agreement {agree}/{total}; "
      f"ZOO recon_error should be ~same order as MP (sane topology).")
