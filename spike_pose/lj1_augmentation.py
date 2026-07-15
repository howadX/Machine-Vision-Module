#!/usr/bin/env python3
"""LJ1 experiment 2 — view augmentation: is 2D-GRIP's robustness bought from DATA?

The plan's thesis: invariance moves from the features into the data (view augmentation),
not from clever 3D features (which the z-liability just killed, exp-1). Test it.

Hypotheses (failure criteria up front):
  H-aug:  azimuth-augmenting the training set flattens 2D-GRIP's accuracy-vs-azimuth curve
          on the mocap true-3D bench WITHOUT hurting the gold R3 number.
          Fails if the curve stays as sloped as un-augmented, or gold R3 drops materially.
  H-moc:  adding our mocap multi-view poses to training improves gold R3 generalization
          (mocap is a clean, free, multi-viewpoint training asset, not only a bench).
          Fails if mocap-augmented gold R3 <= lab-only within CI.

Models (all 2D-GRIP, z=0):
  M0 lab            M1 lab + azimuth-aug (synthetic z-rotation of lab)
  M2 lab + mocap    M3 lab + mocap + azimuth-aug
Azimuth curve is only honest for M0/M1 (M2/M3 train on mocap -> would leak). gold R3 is
held-out from all four. Protocol: bench.py (SEED=0, grouped bootstrap CI).
Run: ../venv/bin/python3 spike_pose/lj1_augmentation.py
"""
import os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.ensemble import RandomForestClassifier
from mocap_sanity_gate import features_13
from mocap_to_views import project
import bench

CLASSES = bench.CLASSES
AUG_ANGLES = [-90, -60, -30, 0, 30, 60, 90]     # synthetic training viewpoints
MOCAP_AZ = [0, 30, 60, 90]                       # mocap projection viewpoints for training


def feat_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features_13(l)


def azimuth_rotate(lm, deg):
    """Rotate 33x4 landmarks about vertical using z (synthesizes a new viewpoint)."""
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    out = lm.copy()
    x, z = lm[:, 0], lm[:, 2]
    out[:, 0] = x * c + z * s; out[:, 2] = -x * s + z * c
    return out


def rf():  return RandomForestClassifier(n_estimators=300, random_state=bench.SEED)


def main():
    d = np.load(os.path.join(HERE, "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    X = d["X"]; y = np.array([str(s) for s in d["y"]]); src = np.array([str(s) for s in d["src"]])
    grp = d["grp"]
    lab = src == "lab"; gold = src == "gold5"
    Xg, yg, gg = X[gold], y[gold], grp[gold]

    mc = np.load(os.path.join(HERE, "data_probe", "mocap_views.npz"), allow_pickle=True)
    P3D, ym = mc["P3D"], np.array([str(s) for s in mc["y"]])

    # --- build training feature sets ---
    F_lab = np.array([feat_2d(x) for x in X[lab]]); y_lab = y[lab]
    F_labaug = np.vstack([np.array([feat_2d(azimuth_rotate(x, a)) for x in X[lab]]) for a in AUG_ANGLES])
    y_labaug = np.tile(y_lab, len(AUG_ANGLES))
    F_moc = np.vstack([np.array([feat_2d(project(p, a)) for p in P3D]) for a in MOCAP_AZ])
    y_moc = np.tile(ym, len(MOCAP_AZ))

    models = {
        "M0 lab":              (F_lab, y_lab),
        "M1 lab+azimuth-aug":  (F_labaug, y_labaug),
        "M2 lab+mocap":        (np.vstack([F_lab, F_moc]), np.concatenate([y_lab, y_moc])),
        "M3 lab+mocap+azim":   (np.vstack([F_labaug, F_moc]), np.concatenate([y_labaug, y_moc])),
    }
    Fg = np.array([feat_2d(x) for x in Xg])
    print(f"train sizes: lab={len(F_lab)} lab-aug={len(F_labaug)} mocap={len(F_moc)}")
    print(f"gold5 test: {len(yg)} {dict(collections.Counter(yg))}, groups={len(set(gg))}\n")

    print("== gold R3 cross-domain (held-out from all models) ==")
    fitted = {}
    for name, (Ftr, ytr) in models.items():
        clf = rf().fit(Ftr, ytr); fitted[name] = clf
        pred = clf.predict(Fg)
        pt, lo, hi = bench.bootstrap_ci(yg, pred, gg)
        rec = bench.per_class_recall(yg, pred)
        print(f"  {name:20} gold balanced-acc={pt:.3f} CI[{lo:.3f},{hi:.3f}]  "
              + " ".join(f"{c[:3]}={rec[c]:.2f}" for c in CLASSES))

    print("\n== mocap TRUE-3D azimuth curve (only M0/M1 — M2/M3 train on mocap) ==")
    for name in ["M0 lab", "M1 lab+azimuth-aug"]:
        clf = fitted[name]
        curve = bench.azimuth_curve(P3D, ym, lambda F: clf.predict(F), feat_2d)
        row = " ".join(f"az{a}={curve[a]:.3f}" for a in bench.ANGLES)
        print(f"  {name:20} {row}  spread={((max(curve.values())-min(curve.values()))*100):.1f}pt")


if __name__ == "__main__":
    main()
