#!/usr/bin/env python3
"""Initial 2D-GRIP baseline on the CLEAN CORE (lab + gold#5).

2D-GRIP = the 14 gravity-referenced features computed with z forced to 0 (the z-ablation
showed MediaPipe's z is a cross-domain liability). Reports grouped-CV bootstrap CI +
per-class + confusion, then trains on all core and serializes a Rust-compatible forest.
"""
import numpy as np, sys, struct, collections
sys.path.insert(0, "spike_pose")
from r3_confound_test import features, rf, CLASSES
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

def f2(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features(l)

v = np.load("spike_pose/data_probe/landmarks_v2.npz", allow_pickle=True)
X, y = v["X"], np.array([str(s) for s in v["y"]])
grp, src = v["grp"], np.array([str(s) for s in v["src"]])
core = np.isin(src, ["lab", "gold5"])
Xc, yc, gc = X[core], y[core], grp[core]
F = np.array([f2(x) for x in Xc])
print(f"CLEAN CORE: {len(yc)} poses, {len(set(gc))} groups, dist={dict(collections.Counter(yc))}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
oof = cross_val_predict(rf(), F, yc, groups=gc, cv=sgkf)
correct = (oof == yc).astype(float)

# bootstrap CI over groups
ug = np.unique(gc); idx = {g: np.where(gc == g)[0] for g in ug}
rng = np.random.RandomState(0); boot = np.empty(1000)
for b in range(1000):
    sel = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))]); boot[b] = correct[sel].mean()
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"\n2D-GRIP grouped-CV accuracy = {correct.mean():.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
print(f"           balanced-accuracy = {balanced_accuracy_score(yc, oof):.3f}")
print("  per-class recall:", {c: round(float((oof[yc == c] == c).mean()), 3) for c in CLASSES})
cm = confusion_matrix(yc, oof, labels=CLASSES)
print("  confusion (rows=true, cols=pred)", CLASSES); print(cm)

# --- train final on all core + serialize (production forest format) ---
final = rf().fit(F, yc)

def serialize_forest(forest):
    buf = bytearray(); buf += struct.pack("<I", len(forest.estimators_))
    classes = list(forest.classes_)
    for est in forest.estimators_:
        t = est.tree_; buf += struct.pack("<I", t.node_count)
        proba = t.value.reshape(t.node_count, -1); proba = proba / proba.sum(1, keepdims=True)
        for n in range(t.node_count):
            buf += struct.pack("<ifii3f", int(t.feature[n]), float(t.threshold[n]),
                               int(t.children_left[n]), int(t.children_right[n]),
                               *[float(proba[n, c]) for c in range(3)])
    return bytes(buf)

blob = serialize_forest(final)
open("spike_pose/pose_rf_2dgrip.bin", "wb").write(blob)
print(f"\nserialized 2D-GRIP forest: {len(blob)/1024:.1f} KB, {len(final.estimators_)} trees, "
      f"classes={list(final.classes_)} -> spike_pose/pose_rf_2dgrip.bin")
print("NOTE: Rust export must compute the 14 features with z=0 (2D-GRIP variant).")
