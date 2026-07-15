#!/usr/bin/env python3
"""Assemble ALL available quality pose data and train initial models.

Sources (all our MPPose extractor, [x/w,y/h,z/w,vis], 33x4):
  - lab      landmarks.npz            (URFD + MultiPosture, 3-class, human/protocol labels)
  - gold #5  gold_lyingv2_fullframe   (Roboflow, 3-class, human labels, multi-source)
  - gold #6  gold_sitstand_fullframe  (Roboflow, binary 0/1 -> resolved via gold#5 model)

Leading representation = 2D-GRIP (z-free) per the z-ablation finding. We report all three.
Writes landmarks_v2.npz.
"""
import numpy as np, sys, collections
sys.path.insert(0, "spike_pose")
from r3_confound_test import features, raw_coords, rf, CLASSES
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

def f2(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features(l)

def load(path, src):
    d = np.load(path, allow_pickle=True)
    y = np.array([str(s) for s in d["y"]])
    g = np.array([f"{src}:{s}" for s in d["grp"]]) if "grp" in d else np.array([f"{src}:{i}" for i in range(len(y))])
    return d["X"], y, g

Xl, yl, gl = load("spike_pose/data_probe/landmarks.npz", "lab")
X5, y5, g5 = load("spike_pose/data_probe/gold_lyingv2_fullframe.npz", "g5")
X6, y6, g6 = load("spike_pose/data_probe/gold_sitstand_fullframe.npz", "g6")

# --- resolve gold#6 0/1 -> sitting/standing via an INDEPENDENT gold#5-trained 2D-GRIP model ---
bin_mask = np.isin(y5, ["sitting", "standing"])
m56 = rf().fit(np.array([f2(x) for x in X5[bin_mask]]), y5[bin_mask])
pred6 = m56.predict(np.array([f2(x) for x in X6]))
mapping = {}
for num in ["0", "1"]:
    maj, cnt = collections.Counter(pred6[y6 == num]).most_common(1)[0]
    frac = cnt / (y6 == num).sum()
    mapping[num] = maj
    print(f"  gold#6 class '{num}': gold#5-model says {maj} ({frac:.0%} agreement)")
if len(set(mapping.values())) != 2:
    print("  !! mapping not a clean bijection — flag for manual check");
y6m = np.array([mapping[n] for n in y6])
print(f"  -> gold#6 mapping: {mapping}")

# --- assemble ---
X = np.concatenate([Xl, X5, X6]); y = np.concatenate([yl, y5, y6m])
grp = np.concatenate([gl, g5, g6]); src = np.array(["lab"]*len(yl) + ["gold5"]*len(y5) + ["gold6"]*len(y6))
print(f"\n=== ASSEMBLED landmarks_v2 ===")
print(f"  total poses: {len(y)}   (was 1035 lab-only)")
print(f"  per source : {dict(collections.Counter(src))}")
print(f"  per class  : {dict(collections.Counter(y))}")
print(f"  groups     : {len(set(grp))}")
np.savez_compressed("spike_pose/data_probe/landmarks_v2.npz", X=X, y=y, grp=grp, src=src)
print("  wrote spike_pose/data_probe/landmarks_v2.npz")

# --- initial models: grouped-CV on the assembled set ---
print("\n=== initial models (StratifiedGroupKFold, 5-fold) ===")
F2 = np.array([f2(x) for x in X]); F3 = np.array([features(x) for x in X]); C = np.array([raw_coords(x) for x in X])
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
for name, M in [("2D-GRIP", F2), ("3D-GRIP", F3), ("raw", C)]:
    sc = cross_val_score(rf(), M, y, groups=grp, cv=sgkf, scoring="accuracy")
    print(f"  {name:8} grouped-CV = {sc.mean():.3f} +/- {sc.std():.3f}")
