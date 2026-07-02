"""Spike 7.5.0 — step 1: does the OpenCV Zoo person detector fire on our test images?
Uses the reference demo classes (cv.dnn) to get ground-truth pre/post-processing."""
import sys, cv2, numpy as np
sys.path.insert(0, ".")
from mp_persondet import MPPersonDet

DET = "../models/person_detection_mediapipe_2023mar.onnx"
det = MPPersonDet(modelPath=DET, nmsThreshold=0.3, scoreThreshold=0.5, topK=5000)

cands = [
    ("/tmp/mvm_frame.jpg", True),   # ceiling cam, upside-down -> rotate 180
    ("/tmp/mvm_frame.jpg", False),
    ("../models/references/p_V_01.jpg", False),
    ("../models/references/p_V.jpg", False),
    ("../models/references/p_V_03.jpg", False),
    ("../models/references/p_V_05.jpg", False),
]
for path, rot in cands:
    im = cv2.imread(path)
    if im is None:
        print(f"{path} rot={rot}: MISSING"); continue
    if rot:
        im = cv2.rotate(im, cv2.ROTATE_180)
    res = det.infer(im)
    n = 0 if res is None else len(res)
    print(f"{path} rot={rot} {im.shape}: persons={n}")
    if n:
        for p in res:
            # person row: [4 bbox + 8 keypoints + score] -> see demo
            print("   row len", len(p), "score", round(float(p[-1]),3),
                  "bbox", np.round(p[:4],1))
