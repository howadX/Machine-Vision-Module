#!/usr/bin/env python3
"""LJ1 experiment 3 — the torso-flexion channel: a continuous output, and its honest limits.

Plan: bending is NOT a 4th class but a continuous torso_flexion output on top of the 3-class
posture (OWAS/REBA precedent). Ground truth = CMU true-3D torso inclination. Question: how well
can the DEPLOYED monocular 2D-GRIP features recover that angle?

Hypotheses (failure criteria up front):
  H-flex:   2D features predict true torso flexion with useful accuracy (grouped-CV MAE small
            enough for coarse REBA bins ~15 deg). Fails if MAE >~ a REBA bin (~15-20 deg).
  H-obs:    flexion observability is AZIMUTH-DEPENDENT — a forward bend seen head-on projects
            along the camera axis and is monocularly ambiguous, so error is HIGH near frontal
            and LOW in profile. Fails if MAE is flat across azimuth (no frontal blind spot).

Ground truth flexion = 3D angle of the torso (hip->shoulder) from vertical, in true P3D.
Regressor trained on standing+sitting only (lying's ~90 deg is posture, not a lean).
Protocol: bench.py grouping (by trial), SEED=0.
Run: ../venv/bin/python3 spike_pose/lj1_flexion.py
"""
import os, sys, collections, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold
from mocap_sanity_gate import features_13
from mocap_to_views import project, process
import bench

UP = np.array([0.0, 1.0, 0.0])           # true-3D world is Y-up (P3D)
# Dedicated forward-bend trials (the main set is almost all upright -> no flexion range to test).
BEND_TRIALS = ["02_06", "26_09", "26_10", "26_11", "15_06", "15_07", "13_29", "14_20"]


def flexion_true(P3D):
    """3D torso inclination from vertical (deg): 0 = upright, 90 = horizontal."""
    hip = P3D[7:9].mean(0); sh = P3D[1:3].mean(0); t = sh - hip
    n = np.linalg.norm(t)
    return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(t / n, UP), -1, 1))))


def feat_2d(lm):
    l = lm.copy(); l[:, 2] = 0.0; return features_13(l)


def main():
    mc = np.load(os.path.join(HERE, "data_probe", "mocap_views.npz"), allow_pickle=True)
    P3D = mc["P3D"]; ym = np.array([str(s) for s in mc["y"]]); trial = np.array([str(s) for s in mc["trial"]])
    flex = np.array([flexion_true(p) for p in P3D])
    print("flexion distribution by class (true 3D, deg):")
    for c in ["standing", "sitting", "lying"]:
        f = flex[ym == c]
        print(f"  {c:9} n={len(f):4d}  median={np.median(f):5.1f}  p10={np.percentile(f,10):5.1f}  p90={np.percentile(f,90):5.1f}")

    # pull dedicated forward-bend trials so the channel has a real flexion range to learn/test
    bP, btr = [], []
    for t in BEND_TRIALS:
        try:
            P_, _, _ = process(t)
        except Exception as e:
            print(f"  bend {t}: {e}"); continue
        bP.append(P_); btr += [t] * len(P_)
    bP = np.concatenate(bP); btr = np.array(btr); bfl = np.array([flexion_true(p) for p in bP])
    print(f"bend trials: {len(bfl)} poses, flexion median={np.median(bfl):.0f} p90={np.percentile(bfl,90):.0f} max={bfl.max():.0f} deg")

    # lean set = upright standing/sitting (low flexion) + bend trials (high flexion)
    upright = np.isin(ym, ["standing", "sitting"])
    P = np.concatenate([P3D[upright], bP]); tr = np.concatenate([trial[upright], btr])
    fl = np.concatenate([flex[upright], bfl])
    print(f"\nlean set (upright + bends): {len(fl)} poses, flexion range [{fl.min():.0f},{fl.max():.0f}], "
          f"std={fl.std():.1f} deg, frac>30deg={np.mean(fl>30)*100:.0f}%")

    # grouped-CV (by trial) MAE at azimuth 0 (frontal — the deployed default view)
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(fl))
    for trn, te in gkf.split(P, fl, groups=tr):
        F = np.array([feat_2d(project(p, 0)) for p in P[trn]])
        m = RandomForestRegressor(n_estimators=300, random_state=bench.SEED).fit(F, fl[trn])
        oof[te] = m.predict(np.array([feat_2d(project(p, 0)) for p in P[te]]))
    mae0 = np.abs(oof - fl).mean()
    print(f"\nH-flex: grouped-CV MAE (2D feat -> flexion, frontal az0) = {mae0:.1f} deg "
          f"(baseline predict-mean MAE = {np.abs(fl - fl.mean()).mean():.1f})")

    # H-obs: train multi-azimuth, test MAE at each azimuth (is there a frontal blind spot?)
    print("\nH-obs: flexion MAE vs camera azimuth (train multi-view, grouped by trial)")
    TRAIN_AZ = [0, 30, 60, 90]
    maes = {a: [] for a in bench.ANGLES}
    for trn, te in gkf.split(P, fl, groups=tr):
        Ftr = np.vstack([np.array([feat_2d(project(p, a)) for p in P[trn]]) for a in TRAIN_AZ])
        ytr = np.tile(fl[trn], len(TRAIN_AZ))
        m = RandomForestRegressor(n_estimators=300, random_state=bench.SEED).fit(Ftr, ytr)
        for a in bench.ANGLES:
            pred = m.predict(np.array([feat_2d(project(p, a)) for p in P[te]]))
            maes[a].append(np.abs(pred - fl[te]).mean())
    for a in bench.ANGLES:
        print(f"  azimuth {a:>3} deg   MAE = {np.mean(maes[a]):.1f} deg")


if __name__ == "__main__":
    main()
