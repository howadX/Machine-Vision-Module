#!/usr/bin/env python3
"""LJ1 experiment 5 — honest held-out azimuth: does mocap augmentation give REAL view-robustness?

exp-2 showed lab+mocap lifts gold R3, but its azimuth curve couldn't be measured without leakage
(it trains on mocap). Here we close that: GROUP-DISJOINT by trial. Train on lab + mocap
(train-trials, projected at 0/30/60/90); test on mocap TEST-trials (unseen people) projected at
EVERY angle incl. unseen 15/45/75. No pose leaks, no angle leaks.

Hypothesis (failure criterion up front):
  H-view: mocap augmentation flattens the accuracy-vs-azimuth curve on UNSEEN trials
          (spread < ~5 pt), while the lab-only model stays sloped. Fails if the augmented
          curve is as sloped as lab-only, or drops sharply at the unseen angles.

Averaged over GroupKFold folds (by trial). Protocol: bench.py (SEED=0).
Run: ../venv/bin/python3 spike_pose/lj1_azimuth_heldout.py
"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from mocap_sanity_gate import features_13
from mocap_to_views import project
import bench

TRAIN_AZ = [0, 30, 60, 90]
TEST_AZ = bench.ANGLES                 # 0,15,30,45,60,90 — includes unseen 15/45


def feat_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features_13(l)
def rf():  return RandomForestClassifier(n_estimators=300, random_state=bench.SEED)


def main():
    d = np.load(os.path.join(HERE, "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    src = np.array([str(s) for s in d["src"]]); lab = src == "lab"
    F_lab = np.array([feat_2d(x) for x in d["X"][lab]]); y_lab = np.array([str(s) for s in d["y"]])[lab]

    mc = np.load(os.path.join(HERE, "data_probe", "mocap_views.npz"), allow_pickle=True)
    P3D = mc["P3D"]; ym = np.array([str(s) for s in mc["y"]]); trial = np.array([str(s) for s in mc["trial"]])

    gkf = GroupKFold(n_splits=3)
    curves = {"lab-only": {a: [] for a in TEST_AZ}, "lab+mocap": {a: [] for a in TEST_AZ}}
    for trn, te in gkf.split(P3D, ym, groups=trial):
        # test features: held-out trials, projected at each angle
        Fte = {a: np.array([feat_2d(project(p, a)) for p in P3D[te]]) for a in TEST_AZ}
        yte = ym[te]
        # lab-only model
        m0 = rf().fit(F_lab, y_lab)
        # lab + mocap(train-trials) @ TRAIN_AZ
        Fmoc = np.vstack([np.array([feat_2d(project(p, a)) for p in P3D[trn]]) for a in TRAIN_AZ])
        ymoc = np.tile(ym[trn], len(TRAIN_AZ))
        m1 = rf().fit(np.vstack([F_lab, Fmoc]), np.concatenate([y_lab, ymoc]))
        for a in TEST_AZ:
            curves["lab-only"][a].append(bench.balanced_accuracy(yte, m0.predict(Fte[a])))
            curves["lab+mocap"][a].append(bench.balanced_accuracy(yte, m1.predict(Fte[a])))

    print("Held-out azimuth (group-disjoint by trial; test angles incl. unseen 15/45/75):\n")
    print(f"  {'model':10}" + "".join(f"  az{a:>2}" for a in TEST_AZ) + "   spread")
    for name in ["lab-only", "lab+mocap"]:
        row = [np.mean(curves[name][a]) for a in TEST_AZ]
        spread = (max(row) - min(row)) * 100
        print(f"  {name:10}" + "".join(f"  {v:.3f}" for v in row) + f"   {spread:.1f}pt")
    print("\nH-view PASS if lab+mocap spread < ~5pt and flatter than lab-only.")


if __name__ == "__main__":
    main()
