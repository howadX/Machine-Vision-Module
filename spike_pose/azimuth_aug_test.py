#!/usr/bin/env python3
"""Does azimuth-AUGMENTED training fix 2D-GRIP's synthetic-rotation collapse?

Train folds are expanded with copies rotated about the vertical axis (using MediaPipe z
to synthesize the new x — noisy, but the same synthesis the test uses). If the augmented
2D-GRIP recovers on rotated test views, azimuth diversity in TRAIN is the cure and the
2D representation itself is fine; if not, 2D features fundamentally can't span views.
Also reports the price on clean (az0) accuracy.
"""
import numpy as np, sys, collections
sys.path.insert(0, "spike_pose")
from r3_confound_test import features, rf, CLASSES
from sklearn.model_selection import StratifiedGroupKFold

def azimuth_rotate(lm, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    out = lm.copy()
    x, z = lm[:, 0], lm[:, 2]
    out[:, 0] = x * c + z * s
    out[:, 2] = -x * s + z * c
    return out

def f2(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features(l)

v = np.load("spike_pose/data_probe/landmarks_v2.npz", allow_pickle=True)
X, y = v["X"], np.array([str(s) for s in v["y"]])
grp, src = v["grp"], np.array([str(s) for s in v["src"]])
core = np.isin(src, ["lab", "gold5"])
X, y, grp = X[core], y[core], grp[core]
print(f"clean core: {len(y)} poses, {len(set(grp))} groups")

TRAIN_AUG = [0, 30, 60, 90, 120, 150, 180, -30, -60, -90]
TEST_ANGLES = [0, 15, 30, 45, 60, 90]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
acc = {a: [] for a in TEST_ANGLES}
rec90 = {c: [0, 0] for c in CLASSES}
for tr, te in sgkf.split(X, y, groups=grp):
    Ftr = np.concatenate([np.array([f2(azimuth_rotate(x, a)) for x in X[tr]]) for a in TRAIN_AUG])
    ytr = np.tile(y[tr], len(TRAIN_AUG))
    m = rf().fit(Ftr, ytr)
    for a in TEST_ANGLES:
        Xr = X[te] if a == 0 else np.array([azimuth_rotate(x, a) for x in X[te]])
        pred = m.predict(np.array([f2(x) for x in Xr]))
        acc[a].append((pred == y[te]).mean())
        if a == 90:
            for c in CLASSES:
                msk = y[te] == c
                rec90[c][0] += (pred[msk] == c).sum(); rec90[c][1] += msk.sum()

print(f"\n2D-GRIP + azimuth-augmented train ({len(TRAIN_AUG)}x):")
print("  " + "".join(f"  az{a:>3}" for a in TEST_ANGLES))
print("  " + "".join(f"  {np.mean(acc[a]):.3f}" for a in TEST_ANGLES))
print("  per-class recall az90:", {c: round(rec90[c][0] / max(rec90[c][1], 1), 3) for c in CLASSES})
print("  (reference: un-augmented 2D-GRIP az0=0.942 -> az90=0.440)")
