"""Export PoseVAE + PoseClassifier weights to a flat f32 binary (fixed order) for the
Rust hand-rolled port, plus golden outputs on a fixed 33x4 landmark input (7.5.3 oracle)."""
import os, sys, json, struct
import numpy as np, cv2, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
from mp_persondet import MPPersonDet
from mp_pose import MPPose
from core.pose_vae import PoseVAE
from core.pose_heads import PoseClassifier
from core.pose_encoder import normalize_landmarks, prepare_vae_input

M = f"{ROOT}/models"; DEV = torch.device("cpu")
vcfg = json.load(open(f"{M}/emotion/vposer_coco33_config.json"))
ccfg = json.load(open(f"{M}/emotion/pose_classifier_config.json"))
vae = PoseVAE(input_dim=vcfg["input_dim"], latent_dim=vcfg["latent_dim"]).to(DEV)
vae.load_state_dict(torch.load(f"{M}/emotion/vposer_coco33.pth", map_location=DEV,
                               weights_only=True)["model_state_dict"]); vae.eval()
clf = PoseClassifier(latent_dim=vcfg["latent_dim"], num_classes=ccfg["num_classes"]).to(DEV)
clf.load_state_dict(torch.load(f"{M}/emotion/pose_classifier.pth", map_location=DEV,
                               weights_only=True)["model_state_dict"]); clf.eval()

sd_v = vae.state_dict(); sd_c = clf.state_dict()
# Fixed export order — MUST match the Rust loader in mvm-posture/src/classifier.rs.
ORDER = [
    ("v", "encoder.0.weight"), ("v", "encoder.0.bias"),
    ("v", "encoder.0.running_mean"), ("v", "encoder.0.running_var"),
    ("v", "encoder.1.weight"), ("v", "encoder.1.bias"),
    ("v", "encoder.4.weight"), ("v", "encoder.4.bias"),
    ("v", "fc_mu.weight"), ("v", "fc_mu.bias"),
    ("v", "fc_logvar.weight"), ("v", "fc_logvar.bias"),
    ("v", "decoder.0.weight"), ("v", "decoder.0.bias"),
    ("v", "decoder.3.weight"), ("v", "decoder.3.bias"),
    ("v", "decoder.6.weight"), ("v", "decoder.6.bias"),
    ("c", "classifier.0.weight"), ("c", "classifier.0.bias"),
    ("c", "classifier.3.weight"), ("c", "classifier.3.bias"),
    ("c", "classifier.6.weight"), ("c", "classifier.6.bias"),
]
buf = bytearray()
total = 0
for src, key in ORDER:
    t = (sd_v if src == "v" else sd_c)[key].cpu().numpy().astype(np.float32).ravel()
    buf += t.tobytes(); total += t.size
OUT = f"{ROOT}/mvm/crates/mvm-posture/src/pose_weights.bin"
open(OUT, "wb").write(buf)
print(f"wrote {OUT}: {len(buf)} bytes, {total} floats")

# ── fixed 33x4 landmark input from real p_V_03 pose (x/w,y/h,z/w,vis) ──
det = MPPersonDet(modelPath=f"{M}/person_detection_mediapipe_2023mar.onnx", nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
pose = MPPose(modelPath=f"{M}/pose_estimation_mediapipe_2023mar.onnx", confThreshold=0.5)
im = cv2.imread(f"{M}/references/p_V_03.jpg"); h, w = im.shape[:2]
best = max(det.infer(im), key=lambda p: float(p[-1]))
lm = pose.infer(im, best)[1]
lms = np.array([[lm[k][0]/w, lm[k][1]/h, lm[k][2]/w, 1/(1+np.exp(-lm[k][3]))] for k in range(33)], np.float32)

class LM:
    def __init__(s, x, y, z, v): s.x, s.y, s.z, s.visibility = x, y, z, v
landmarks = [LM(*row) for row in lms]
raw99 = prepare_vae_input(landmarks, use_invariants=False)
with torch.no_grad():
    x = torch.tensor(raw99, dtype=torch.float32).unsqueeze(0)
    z = vae.encode_to_z(x)
    recon = vae.decode(z)
    mse = torch.nn.functional.mse_loss(recon, x).item()
    logpz = vae.log_prob_z(x).item()
    logits = clf(z)[0]
    probs = torch.softmax(logits, dim=0).numpy()
idx = int(np.argmax(probs))
golden = {
    "input_33x4": lms.tolist(),
    "norm99": [round(float(v), 6) for v in raw99],
    "z32": [round(float(v), 6) for v in z[0].numpy()],
    "recon_mse": round(float(mse), 6),
    "logpz": round(float(logpz), 4),
    "logits6": [round(float(v), 6) for v in logits.numpy()],
    "probs6": [round(float(v), 6) for v in probs],
    "class": ccfg["class_names"][idx], "class_idx": idx,
}
json.dump(golden, open(f"{HERE}/golden_pose_clf.json", "w"), indent=1)
print("class", golden["class"], "probs", [round(p, 3) for p in golden["probs6"]])
print("recon_mse", golden["recon_mse"], "logpz", golden["logpz"])
print("z32[:4]", [round(v, 3) for v in golden["z32"][:4]])
