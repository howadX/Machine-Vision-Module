"""Spike 7.5 — LIVE body tracking check.

Captures from the ceiling camera and runs the full native-candidate pose pipeline
per frame, for EVERY detected person (multi-persona = #3 fix preview):
  person_detection -> ROI -> pose_estimation -> 33 landmarks -> PoseVAE class.

Prints a live per-person readout: posture, conf, recon_error, torso landmark
visibility, and head/shoulder/hip image positions.

Usage (from spike_pose/, project venv):  ../venv/bin/python3 live_body_track.py [seconds]
"""
import os, sys, time, json
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from mp_persondet import MPPersonDet
from mp_pose import MPPose
import torch
from core.pose_vae import PoseVAE
from core.pose_heads import PoseClassifier
from core.pose_encoder import prepare_vae_input

WINDOW = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
M = os.path.join(ROOT, "models")
DEV = torch.device("cpu")

vcfg = json.load(open(f"{M}/emotion/vposer_coco33_config.json"))
ccfg = json.load(open(f"{M}/emotion/pose_classifier_config.json"))
vae = PoseVAE(input_dim=vcfg["input_dim"], latent_dim=vcfg["latent_dim"]).to(DEV)
vae.load_state_dict(torch.load(f"{M}/emotion/vposer_coco33.pth", map_location=DEV,
                               weights_only=True)["model_state_dict"]); vae.eval()
clf = PoseClassifier(latent_dim=vcfg["latent_dim"], num_classes=ccfg["num_classes"]).to(DEV)
clf.load_state_dict(torch.load(f"{M}/emotion/pose_classifier.pth", map_location=DEV,
                               weights_only=True)["model_state_dict"]); clf.eval()
CLASS = ccfg["class_names"]
det = MPPersonDet(modelPath=f"{M}/person_detection_mediapipe_2023mar.onnx",
                  nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
pose = MPPose(modelPath=f"{M}/pose_estimation_mediapipe_2023mar.onnx", confThreshold=0.5)


class LM:
    __slots__ = ("x", "y", "z", "visibility")
    def __init__(s, x, y, z, v): s.x, s.y, s.z, s.visibility = x, y, z, v


def classify(lms33):
    raw = prepare_vae_input(lms33, use_invariants=False)
    with torch.no_grad():
        x = torch.tensor(raw, dtype=torch.float32, device=DEV).unsqueeze(0)
        z = vae.encode_to_z(x)
        mse = torch.nn.functional.mse_loss(vae.decode(z), x).item()
        probs = torch.softmax(clf(z), dim=1).cpu().numpy()[0]
    i = int(np.argmax(probs))
    return CLASS[i], float(probs[i]), mse


cap = cv2.VideoCapture('/dev/video0', cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    sys.stderr.write("ERROR: cannot open /dev/video0\n"); sys.exit(2)
for _ in range(5): cap.read()
sys.stderr.write(f">>> LIVE body tracking — {WINDOW:.0f}s. Move around / sit / raise arms.\n"); sys.stderr.flush()

t0 = time.time(); frame_n = 0; seen_max = 0
while time.time() - t0 < WINDOW:
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.03); continue
    frame = cv2.rotate(frame, cv2.ROTATE_180)
    h, w = frame.shape[:2]
    frame_n += 1
    persons = det.infer(frame)
    np_ = 0 if persons is None else len(persons)
    seen_max = max(seen_max, np_)
    line = f"[f{frame_n:02d}] persons={np_}"
    if np_:
        for pi, p in enumerate(sorted(persons, key=lambda r: -float(r[-1]))):
            res = pose.infer(frame, p)
            if res is None:
                line += f" | P{pi}(det {p[-1]:.2f}): pose<thr"; continue
            lm = res[1]  # 39x5 pixels
            lms33 = [LM(lm[k][0]/w, lm[k][1]/h, lm[k][2]/w,
                        1/(1+np.exp(-lm[k][3]))) for k in range(33)]
            cls, cf, mse = classify(lms33)
            vis_torso = np.mean([lms33[i].visibility for i in (11, 12, 23, 24)])
            nose = lms33[0]; sh = lms33[11]; hip = lms33[23]
            line += (f" | P{pi}(det {p[-1]:.2f}): {cls}@{cf:.2f} recon={mse:.3f} "
                     f"torso_vis={vis_torso:.2f} "
                     f"head=({nose.x:.2f},{nose.y:.2f}) hip=({hip.x:.2f},{hip.y:.2f})")
    print(line, flush=True)

cap.release()
print(f"\n{frame_n} frames; max persons seen simultaneously = {seen_max}")
print("=> body tracking live ✓" if seen_max else "=> no person tracked ✗")
