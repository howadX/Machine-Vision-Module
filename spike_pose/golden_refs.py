"""7.5.5 oracle: Python reference pose labels on the committed p_V references.
Emits golden_refs.json {file: {label, person, n_det}} for the Rust regression test."""
import os, sys, json, glob
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from mp_persondet import MPPersonDet
from mp_pose import MPPose
from core.pose_vae import PoseVAE
from core.pose_heads import PoseClassifier
from core.pose_encoder import prepare_vae_input

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

out = {}
# canonical committed reference set used by the Rust regression test
REF = f"{ROOT}/mvm/models/references"
for path in sorted(glob.glob(f"{REF}/p_V*.jpg")):
    f = os.path.basename(path)
    im = cv2.imread(path); h, w = im.shape[:2]
    persons = det.infer(im)
    if persons is None or len(persons) == 0:
        out[f] = {"label": None, "n_det": 0}; continue
    best = max(persons, key=lambda p: float(p[-1]))
    res = pose.infer(im, best)
    if res is None:
        out[f] = {"label": None, "n_det": len(persons)}; continue
    lm = res[1]
    lms = [LM(lm[k][0]/w, lm[k][1]/h, lm[k][2]/w, 1/(1+np.exp(-lm[k][3]))) for k in range(33)]
    raw = prepare_vae_input(lms, use_invariants=False)
    with torch.no_grad():
        z = vae.encode_to_z(torch.tensor(raw).unsqueeze(0))
        probs = torch.softmax(clf(z), dim=1).numpy()[0]
    out[f] = {"label": CLASS[int(np.argmax(probs))], "conf": round(float(np.max(probs)), 3), "n_det": len(persons)}

json.dump(out, open(f"{HERE}/golden_refs.json", "w"), indent=1)
for f, v in out.items():
    print(f"{f}: {v}")
