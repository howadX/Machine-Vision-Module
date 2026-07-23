#!/usr/bin/env python3
"""LJ1 experiment 1 — 2D-GRIP (z=0) vs 3D-GRIP vs raw, the z-liability decided honestly.

Hypothesis (falsifiable, failure criterion up front):
  H-z: MediaPipe's z is a CROSS-DOMAIN liability. Dropping it (2D-GRIP, z=0) generalizes
       BETTER than 3D-GRIP on the confound-free R3 gold test (train lab -> test gold5),
       because real z is estimated differently across capture rigs.
  Fails if: 3D-GRIP >= 2D-GRIP on gold with non-overlapping bootstrap CIs (z carries signal
       that survives the domain shift), OR both collapse equally (z is not the culprit).

  Control (azimuth): on the mocap true-3D bench, BOTH 2D and 3D stay flat vs azimuth —
       showing the z-liability is a DOMAIN effect (real MediaPipe z), not a viewpoint effect
       (mocap z is perfect). If 3D dropped with azimuth here, the story would be different.

Protocol: bench.py (SEED=0, grouped bootstrap CI). Representations on the 13 GRIP joints.
Run:  ../venv/bin/python3 spike_pose/lj1_2d_vs_3d.py
"""
import os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.ensemble import RandomForestClassifier
from mocap_sanity_gate import features_13, L, GRIP_IDX
import bench

CLASSES = bench.CLASSES


def feat_3d(lm):  return features_13(lm)
def feat_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features_13(l)
def raw_13(lm):                                   # normalized 13-joint coords, flattened
    p = lm[:, :3].astype(np.float32)
    hip = (p[L["Lhip"]] + p[L["Rhip"]]) / 2; sh = (p[L["Lsh"]] + p[L["Rsh"]]) / 2
    return ((p[GRIP_IDX] - hip) / (np.linalg.norm(sh - hip) + 1e-6)).reshape(-1)
def raw_13_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return raw_13(l)

REPS = [("2D-GRIP", feat_2d), ("3D-GRIP", feat_3d), ("raw-13", raw_13), ("raw-13 z=0", raw_13_2d)]


def load_v2():
    d = np.load(os.path.join(HERE, "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    return (d["X"], np.array([str(s) for s in d["y"]]),
            np.array([str(s) for s in d["src"]]), d["grp"])


def rf():  return RandomForestClassifier(n_estimators=300, random_state=bench.SEED)


def main():
    X, y, src, grp = load_v2()
    lab = src == "lab"; gold = src == "gold5"
    print(f"lab: {lab.sum()} {dict(collections.Counter(y[lab]))}")
    print(f"gold5: {gold.sum()} {dict(collections.Counter(y[gold]))}, groups={len(set(grp[gold]))}\n")

    print("== R3 cross-domain: train=lab -> test=gold5 (confound-free, human labels) ==")
    for name, fn in REPS:
        Ftr = np.array([fn(x) for x in X[lab]]); Fte = np.array([fn(x) for x in X[gold]])
        pred = rf().fit(Ftr, y[lab]).predict(Fte)
        pt, lo, hi = bench.bootstrap_ci(y[gold], pred, grp[gold])
        rec = bench.per_class_recall(y[gold], pred)
        print(f"  {name:11} gold balanced-acc={pt:.3f} CI[{lo:.3f},{hi:.3f}]  "
              f"recall " + " ".join(f"{c[:3]}={rec[c]:.2f}" for c in CLASSES))

    print("\n== in-domain reference: clean-core grouped-CV (lab+gold5) ==")
    core = lab | gold
    for name, fn in REPS:
        pf, oof = bench.grouped_cv(X[core], y[core], grp[core],
                                   lambda Xtr, ytr, Xte: rf().fit(np.array([fn(a) for a in Xtr]), ytr)
                                                              .predict(np.array([fn(a) for a in Xte])))
        pt, lo, hi = bench.bootstrap_ci(y[core], oof, grp[core])
        print(f"  {name:11} grouped-CV balanced-acc={np.mean(pf):.3f} CI[{lo:.3f},{hi:.3f}]")

    print("\n== control: mocap TRUE-3D azimuth bench (train=clean core) — is z a viewpoint issue? ==")
    mc = np.load(os.path.join(HERE, "data_probe", "mocap_views.npz"), allow_pickle=True)
    P3D, ym = mc["P3D"], np.array([str(s) for s in mc["y"]])
    for name, fn in REPS[:2]:                        # 2D vs 3D only
        clf = rf().fit(np.array([fn(x) for x in X[core]]), y[core])
        curve = bench.azimuth_curve(P3D, ym, lambda F: clf.predict(F), fn)
        row = " ".join(f"az{a}={curve[a]:.3f}" for a in bench.ANGLES)
        print(f"  {name:11} {row}  drop0->90={(curve[0]-curve[90])*100:+.1f}pt")


if __name__ == "__main__":
    main()
