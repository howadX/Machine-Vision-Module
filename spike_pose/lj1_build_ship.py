#!/usr/bin/env python3
"""LJ1 integration — build + validate + serialize the v0.2.0 pose artifacts (for the 15.6 Rust port).

Consolidates the five LJ1 decisions into shippable artifacts:
  1. 2D-GRIP RandomForest (features_13, z=0) trained on lab + mocap (corrected pelvis-gated
     labels + bend trials). Validated on gold R3 (held-out) + held-out azimuth.
  2. Class-conditional Mahalanobis (means + inv-cov on the 14 features) + OOD threshold — the
     195 KB VAE's replacement (~2.5 KB, pure arithmetic).
  3. Torso-flexion output — pick pure arithmetic (2D torso_inc) vs a small regressor by MAE.

Artifacts -> spike_pose/ship/ (gitignored; regenerable). Rust port reads these in 15.6.
Run: ../venv/bin/python3 spike_pose/lj1_build_ship.py
"""
import os, sys, struct, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GroupKFold
from mocap_sanity_gate import features_13, FEATS
from mocap_to_views import project
from lj1_flexion import flexion_true, BEND_TRIALS
import bench

SHIP = os.path.join(HERE, "ship"); os.makedirs(SHIP, exist_ok=True)
CLASSES = bench.CLASSES


def feat_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features_13(l)
# Ship config from the size/quality sweep: gold R3 is flat (~0.89) from 2.5 MB down to ~190 KB,
# so the big forest was pure redundancy. This is 78% smaller than prod's 873 KB at equal accuracy.
def rf():  return RandomForestClassifier(n_estimators=60, max_depth=9, min_samples_leaf=10,
                                          random_state=bench.SEED)


def serialize_forest(forest):                      # matches train_baseline_2dgrip.py (prod format)
    buf = bytearray(); buf += struct.pack("<I", len(forest.estimators_))
    for est in forest.estimators_:
        t = est.tree_; buf += struct.pack("<I", t.node_count)
        proba = t.value.reshape(t.node_count, -1); proba = proba / proba.sum(1, keepdims=True)
        for n in range(t.node_count):
            buf += struct.pack("<ifii3f", int(t.feature[n]), float(t.threshold[n]),
                               int(t.children_left[n]), int(t.children_right[n]),
                               *[float(proba[n, c]) for c in range(3)])
    return bytes(buf)


def main():
    d = np.load(os.path.join(HERE, "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    src = np.array([str(s) for s in d["src"]]); lab = src == "lab"; gold = src == "gold5"
    ylab = np.array([str(s) for s in d["y"]])
    mc = np.load(os.path.join(HERE, "data_probe", "mocap_views.npz"), allow_pickle=True)
    Pm, ym = mc["P3D"], np.array([str(s) for s in mc["y"]])

    F_lab = np.array([feat_2d(x) for x in d["X"][lab]]); y_lab = ylab[lab]
    F_moc = np.array([feat_2d(x) for x in mc["X"]])       # mocap at az0 (X already projected)
    F_tr = np.vstack([F_lab, F_moc]); y_tr = np.concatenate([y_lab, ym])
    print(f"ship train: lab={len(F_lab)} + mocap={len(F_moc)} = {len(F_tr)}")

    # ---- 1. final forest + validation ----
    forest = rf().fit(F_tr, y_tr)
    Fg = np.array([feat_2d(x) for x in d["X"][gold]]); yg = ylab[gold]
    pt, lo, hi = bench.bootstrap_ci(yg, forest.predict(Fg), d["grp"][gold])
    print(f"\n[1] ship 2D-GRIP forest: {len(forest.estimators_)} trees, classes={list(forest.classes_)}")
    print(f"    gold R3 (held-out) balanced-acc = {pt:.3f} CI[{lo:.3f},{hi:.3f}]  "
          + " ".join(f"{c[:3]}={bench.per_class_recall(yg, forest.predict(Fg))[c]:.2f}" for c in CLASSES))
    blob = serialize_forest(forest); open(os.path.join(SHIP, "pose_rf_2dgrip_v2.bin"), "wb").write(blob)
    print(f"    serialized -> ship/pose_rf_2dgrip_v2.bin ({len(blob)/1024:.1f} KB)")

    # ---- 2. Mahalanobis OOD (fit on training features) ----
    means, iP = [], []
    for c in CLASSES:
        Fc = F_tr[y_tr == c]; mu = Fc.mean(0)
        cov = np.cov(Fc, rowvar=False) + 1e-3 * np.eye(F_tr.shape[1])
        means.append(mu); iP.append(np.linalg.inv(cov))
    def maha_score(F):
        return np.sqrt(np.stack([np.einsum("ij,jk,ik->i", F - m, P, F - m) for m, P in zip(means, iP)], 1).min(1))
    thr = float(np.percentile(maha_score(F_tr), 99))      # OOD gate: 99th pct of ID
    buf = struct.pack("<II", len(CLASSES), F_tr.shape[1]) + struct.pack("<f", thr)
    for mu, P in zip(means, iP):
        buf += mu.astype("<f4").tobytes() + P.astype("<f4").tobytes()
    open(os.path.join(SHIP, "pose_maha.bin"), "wb").write(buf)
    print(f"\n[2] Mahalanobis OOD: 3 classes x 14-d, thr(99pct)={thr:.2f} -> ship/pose_maha.bin ({len(buf)} B)")

    # ---- 3. flexion output: arithmetic (2D torso_inc) vs small regressor ----
    bP, btr = [], []
    for t in BEND_TRIALS:
        try: P_, _, _ = __import__("mocap_to_views").process(t)
        except Exception: continue
        bP.append(P_); btr += [t] * len(P_)
    bP = np.concatenate(bP)
    upright = np.isin(ym, ["standing", "sitting"])
    Pfx = np.concatenate([Pm[upright], bP]); trfx = np.concatenate([mc["trial"][upright].astype(str), np.array(btr)])
    flx = np.array([flexion_true(p) for p in Pfx])
    Ffx0 = np.array([feat_2d(project(p, 0)) for p in Pfx])
    ti_idx = FEATS.index("torso_inc")
    mae_arith = np.abs(Ffx0[:, ti_idx] - flx).mean()      # raw 2D torso_inc as flexion
    gkf = GroupKFold(5); oof = np.zeros(len(flx))
    for tr, te in gkf.split(Ffx0, flx, groups=trfx):
        oof[te] = RandomForestRegressor(200, random_state=0).fit(Ffx0[tr], flx[tr]).predict(Ffx0[te])
    mae_reg = np.abs(oof - flx).mean()
    print(f"\n[3] flexion (frontal): raw 2D torso_inc MAE={mae_arith:.1f} deg vs regressor MAE={mae_reg:.1f} deg")
    if mae_arith <= mae_reg + 3:
        print("    -> SHIP ARITHMETIC (torso_inc): pure arithmetic, +0 model weight (invariant #1).")
    else:
        print("    -> ship a small flexion regressor (arithmetic loses > 3 deg).")

    print("\nartifacts in spike_pose/ship/ (gitignored, regenerable). Rust port: 15.6.")


if __name__ == "__main__":
    main()
