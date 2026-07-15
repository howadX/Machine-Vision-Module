#!/usr/bin/env python3
"""Azimuth-robustness of 2D-GRIP (z=0) vs 3D-GRIP vs raw on the CLEAN CORE.

The original azimuth test (spike_feats.py) rotated 33x4 landmarks about the vertical
axis and re-computed 3D features. 2D-GRIP throws z away AFTER rotation — i.e. it sees
the orthographic re-projection of the rotated pose. Caveat: rotation uses MediaPipe's
noisy z to synthesize the new x, so this is a first-order simulation, not ground truth.

Grouped 5-fold: train on clean-view train folds, test each held-out fold clean AND
rotated at 15/30/45/60/90 deg. Reports accuracy per angle + per-class recall at 90.
"""
import numpy as np, sys, collections
sys.path.insert(0, "spike_pose")
from r3_confound_test import features, raw_coords, rf, CLASSES
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
print(f"clean core: {len(y)} poses, {len(set(grp))} groups, dist={dict(collections.Counter(y))}")

ANGLES = [0, 15, 30, 45, 60, 90]
REPS = [("2D-GRIP", f2), ("3D-GRIP", features), ("raw", raw_coords)]

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
acc = {n: {a: [] for a in ANGLES} for n, _ in REPS}          # per-fold accuracies
rec90 = {n: {c: [0, 0] for c in CLASSES} for n, _ in REPS}   # correct/total at 90deg

for tr, te in sgkf.split(X, y, groups=grp):
    for name, fn in REPS:
        m = rf().fit(np.array([fn(x) for x in X[tr]]), y[tr])
        for a in ANGLES:
            Xr = X[te] if a == 0 else np.array([azimuth_rotate(x, a) for x in X[te]])
            pred = m.predict(np.array([fn(x) for x in Xr]))
            acc[name][a].append((pred == y[te]).mean())
            if a == 90:
                for c in CLASSES:
                    msk = y[te] == c
                    rec90[name][c][0] += (pred[msk] == c).sum(); rec90[name][c][1] += msk.sum()

print(f"\n{'rep':10}" + "".join(f"  az{a:>3}" for a in ANGLES) + "   drop(0->90)")
for name, _ in REPS:
    row = [np.mean(acc[name][a]) for a in ANGLES]
    print(f"{name:10}" + "".join(f"  {v:.3f}" for v in row) + f"   {(row[0]-row[-1])*100:+.1f}pt")
print("\nper-class recall at az90:")
for name, _ in REPS:
    print(f"  {name:10}", {c: round(rec90[name][c][0] / max(rec90[name][c][1], 1), 3) for c in CLASSES})
