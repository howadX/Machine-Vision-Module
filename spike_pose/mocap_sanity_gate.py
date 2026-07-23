#!/usr/bin/env python3
"""15.0 gate: do GRIP features on CMU-mocap virtual-camera projections land in the
same distribution as real MediaPipe features (lab + gold5 clean core)?

features_13: the plan's variant where vext/hext use only the 13 GRIP joints (so the
feature is reproducible on ANY skeleton, incl. a mocap rig with no face/hand points).
Compared apples-to-apples: features_13 on BOTH sides.

Reports, per class, the median +/- IQR of each feature for mocap vs clean-core, and an
'in-distribution' check: is the mocap median within the clean core's [p10,p90]?
Pass = most posture-discriminative features (torso_inc, knee_flex, hip_flex, aspect,
vext) overlap; a systematic offset would mean the virtual camera / mapping is off.
"""
import numpy as np, collections

L = dict(nose=0, Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16,
         Lhip=23, Rhip=24, Lkn=25, Rkn=26, Lank=27, Rank=28)
GRIP_IDX = list(L.values())
UP = np.array([0.0, -1.0, 0.0], np.float32)   # MediaPipe y-down -> up = -y
FEATS = ["torso_inc", "thigh_inc", "shank_inc", "knee_flex", "hip_flex",
         "elbow_flex", "vext", "hext", "aspect", "nose_above", "ank_below", "kn_below"]


def _ang(v):
    n = np.linalg.norm(v)
    return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v / n, UP), -1, 1))))


def _flex(a, b, c):
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u / nu, w / nw), -1, 1))))


def features_13(lm):
    """33x4 -> 12 interpretable GRIP features; vext/hext over the 13 GRIP joints only."""
    p = lm[:, :3].astype(np.float32)
    g = lambda k: p[L[k]]
    hip = (g("Lhip") + g("Rhip")) / 2
    sh = (g("Lsh") + g("Rsh")) / 2
    tlen = np.linalg.norm(sh - hip)
    if tlen < 1e-6:
        tlen = max(np.linalg.norm(g("Lsh") - g("Rsh")), 1e-3)
    torso_inc = _ang(sh - hip)
    thigh = 0.5 * (_ang(g("Lkn") - g("Lhip")) + _ang(g("Rkn") - g("Rhip")))
    shank = 0.5 * (_ang(g("Lank") - g("Lkn")) + _ang(g("Rank") - g("Rkn")))
    knee = 0.5 * (_flex(g("Lhip"), g("Lkn"), g("Lank")) + _flex(g("Rhip"), g("Rkn"), g("Rank")))
    hipf = 0.5 * (_flex(g("Lsh"), g("Lhip"), g("Lkn")) + _flex(g("Rsh"), g("Rhip"), g("Rkn")))
    elb = 0.5 * (_flex(g("Lsh"), g("Lel"), g("Lwr")) + _flex(g("Rsh"), g("Rel"), g("Rwr")))
    q = p[GRIP_IDX]                                    # 13 joints only
    vext = (q[:, 1].max() - q[:, 1].min()) / tlen
    rad = np.sqrt((q[:, 0] - hip[0]) ** 2 + (q[:, 2] - hip[2]) ** 2)
    hext = 2 * rad.max() / tlen
    aspect = vext / (hext + 1e-6)
    nose_above = (hip[1] - g("nose")[1]) / tlen
    ank_below = ((g("Lank")[1] + g("Rank")[1]) / 2 - hip[1]) / tlen
    kn_below = ((g("Lkn")[1] + g("Rkn")[1]) / 2 - hip[1]) / tlen
    return np.array([torso_inc, thigh, shank, knee, hipf, elb,
                     vext, hext, aspect, nose_above, ank_below, kn_below], np.float32)


def load(path, srcs=None):
    d = np.load(path, allow_pickle=True)
    X, y = d["X"], np.array([str(s) for s in d["y"]])
    if srcs is not None:
        m = np.isin(np.array([str(s) for s in d["src"]]), srcs)
        X, y = X[m], y[m]
    F = np.array([features_13(x) for x in X])
    return F, y


def main():
    Fm, ym = load("spike_pose/data_probe/mocap_views.npz")
    Fc, yc = load("spike_pose/data_probe/landmarks_v2.npz", srcs=["lab", "gold5"])
    print(f"mocap: {len(ym)} {dict(collections.Counter(ym))}")
    print(f"clean core (lab+gold5): {len(yc)} {dict(collections.Counter(yc))}\n")

    key = ["torso_inc", "knee_flex", "hip_flex", "aspect", "vext", "nose_above"]
    ki = [FEATS.index(k) for k in key]
    for cls in ["standing", "sitting", "lying"]:
        A, B = Fm[ym == cls], Fc[yc == cls]
        if len(A) == 0 or len(B) == 0:
            print(f"[{cls}] missing (mocap={len(A)} core={len(B)})"); continue
        print(f"[{cls}]  mocap n={len(A)}  core n={len(B)}")
        hits = 0
        for k, idx in zip(key, ki):
            mm = np.median(A[:, idx])
            lo, mc, hi = np.percentile(B[:, idx], [10, 50, 90])
            ok = lo <= mm <= hi
            hits += ok
            print(f"    {k:11} mocap_med={mm:7.2f}  core[p10,med,p90]=[{lo:7.2f},{mc:7.2f},{hi:7.2f}]  {'IN' if ok else 'OUT'}")
        print(f"    -> {hits}/{len(key)} key features in-distribution\n")


if __name__ == "__main__":
    main()
