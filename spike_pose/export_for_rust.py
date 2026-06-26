"""Export anchors + golden detector/pose outputs for the Rust mvm-posture port + oracle tests."""
import os, sys, json, struct
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from mp_persondet import MPPersonDet
from mp_pose import MPPose

M = f"{ROOT}/models"
det = MPPersonDet(modelPath=f"{M}/person_detection_mediapipe_2023mar.onnx",
                  nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)
pose = MPPose(modelPath=f"{M}/pose_estimation_mediapipe_2023mar.onnx", confThreshold=0.5)

# 1) anchors -> binary f32 LE [N,2]
anchors = det.anchors.astype(np.float32)
print("anchors shape", anchors.shape)
OUT = f"{ROOT}/mvm/crates/mvm-posture/src/anchors_person.bin"
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "wb") as f:
    f.write(anchors.tobytes())
print("wrote", OUT, os.path.getsize(OUT), "bytes")

# 2) golden detector + pose on a committed fixture
img_path = f"{M}/references/p_V_03.jpg"
im = cv2.imread(img_path)
h, w = im.shape[:2]
persons = det.infer(im)
best = max(persons, key=lambda p: float(p[-1]))
best_kp = [round(float(x), 2) for x in best[4:12]]  # capture BEFORE pose.infer (it mutates best[4:8] in place!)
res = pose.infer(im, best)
lm = res[1][:33, :3]  # x,y(pixels), z
golden = {
    "image": "p_V_03.jpg", "w": w, "h": h,
    "n_persons": int(len(persons)),
    "best_box": [round(float(x), 2) for x in best[:4]],
    "best_keypoints": best_kp,  # 4 kp xy (mid-hip, full-body, shoulder-center, upper-body)
    "best_score": round(float(best[-1]), 4),
    "landmarks33_norm": [[round(float(lm[k][0]/w), 4), round(float(lm[k][1]/h), 4),
                          round(float(lm[k][2]/w), 4)] for k in range(33)],
}
OUTJ = f"{HERE}/golden_p_V_03.json"
json.dump(golden, open(OUTJ, "w"), indent=1)
print("wrote", OUTJ)
print("best_score", golden["best_score"], "box", golden["best_box"])
print("kp(mid_hip, full_body)", golden["best_keypoints"][:4])
print("landmark[0] nose", golden["landmarks33_norm"][0])
print("landmark[11] L-shoulder", golden["landmarks33_norm"][11])
print("landmark[23] L-hip", golden["landmarks33_norm"][23])
