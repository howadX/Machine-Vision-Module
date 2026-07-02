"""7.5.5 oracle: run the Python reference pipeline (persondet→pose→VAE→class) on the
SAME frames the Rust NativePose processed, and compare per-frame labels.
Usage: oracle_frames.py <frames_dir> <rust_labels.txt>"""
import os, sys, json, glob
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from mp_persondet import MPPersonDet
from mp_pose import MPPose
from core.pose_vae import PoseVAE
from core.pose_heads import PoseClassifier
from core.pose_encoder import prepare_vae_input

FRAMES = sys.argv[1]; RUSTLBL = sys.argv[2]
M = f"{ROOT}/models"; DEV = torch.device("cpu")
vcfg = json.load(open(f"{M}/emotion/vposer_coco33_config.json"))
ccfg = json.load(open(f"{M}/emotion/pose_classifier_config.json"))
vae = PoseVAE(input_dim=vcfg["input_dim"], latent_dim=vcfg["latent_dim"]).to(DEV)
vae.load_state_dict(torch.load(f"{M}/emotion/vposer_coco33.pth", map_location=DEV, weights_only=True)["model_state_dict"]); vae.eval()
clf = PoseClassifier(latent_dim=vcfg["latent_dim"], num_classes=ccfg["num_classes"]).to(DEV)
clf.load_state_dict(torch.load(f"{M}/emotion/pose_classifier.pth", map_location=DEV, weights_only=True)["model_state_dict"]); clf.eval()
CLASS = ccfg["class_names"]
det = MPPersonDet(modelPath=f"{M}/person_detection_mediapipe_2023mar.onnx", nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
pose = MPPose(modelPath=f"{M}/pose_estimation_mediapipe_2023mar.onnx", confThreshold=0.5)

class LM:
    def __init__(s, x, y, z, v): s.x, s.y, s.z, s.visibility = x, y, z, v

def py_label(im):
    h, w = im.shape[:2]
    persons = det.infer(im)
    if persons is None or len(persons) == 0: return "none"
    best = max(persons, key=lambda p: float(p[-1]))
    res = pose.infer(im, best)
    if res is None: return "none"
    lm = res[1]
    lms = [LM(lm[k][0]/w, lm[k][1]/h, lm[k][2]/w, 1/(1+np.exp(-lm[k][3]))) for k in range(33)]
    raw = prepare_vae_input(lms, use_invariants=False)
    with torch.no_grad():
        z = vae.encode_to_z(torch.tensor(raw).unsqueeze(0))
        probs = torch.softmax(clf(z), dim=1).numpy()[0]
    return CLASS[int(np.argmax(probs))]

# rust labels: lines "RUST f### label"
rust = {}
for line in open(RUSTLBL):
    if line.startswith("RUST"):
        _, f, lbl = line.split()
        rust[f] = lbl

py = {}
for path in sorted(glob.glob(f"{FRAMES}/f*.png")):
    f = os.path.basename(path).replace(".png", "")
    py[f] = py_label(cv2.imread(path))

keys = sorted(set(rust) & set(py))
from collections import Counter
print("PY  dist:", dict(Counter(py[k] for k in keys)))
print("RUST dist:", dict(Counter(rust[k] for k in keys)))
# agreement
both_person = [k for k in keys if py[k] != "none" and rust[k] != "none"]
agree = sum(1 for k in both_person if py[k] == rust[k])
print(f"\nframes={len(keys)}  py_person={sum(py[k]!='none' for k in keys)}  rust_person={sum(rust[k]!='none' for k in keys)}")
print(f"both detect person: {len(both_person)}  class-agreement: {agree}/{len(both_person)}" + (f" = {agree/len(both_person)*100:.0f}%" if both_person else ""))
# detection agreement
det_agree = sum(1 for k in keys if (py[k]=="none")==(rust[k]=="none"))
print(f"detection agreement (person present or not): {det_agree}/{len(keys)} = {det_agree/len(keys)*100:.0f}%")
# show a few disagreements
print("\nsample disagreements (py != rust):")
n=0
for k in keys:
    if py[k]!=rust[k] and n<12:
        print(f"  {k}: py={py[k]:<16} rust={rust[k]}"); n+=1
