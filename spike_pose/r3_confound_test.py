#!/usr/bin/env python3
"""R3 confound test: train GRIP(features) and raw(coords) on the CONFOUNDED lab set,
test on a CLEAN multi-source gold set (human labels). No weak labels, no circularity.

E1: does raw's lab-internal advantage COLLAPSE on gold (it was reading lab rig identity),
while GRIP holds (invariant-by-construction)?

Usage: python r3_confound_test.py [gold_npz ...]   (default: gold_lyingv2_poses.npz)
"""
import sys, os, collections
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

HERE = os.path.dirname(os.path.abspath(__file__))
CLASSES = ["lying", "sitting", "standing"]
UP = np.array([0.0, -1.0, 0.0], np.float32)
L = dict(nose=0, Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16,
         Lhip=23, Rhip=24, Lkn=25, Rkn=26, Lank=27, Rank=28)


def _ang(v):
    n = np.linalg.norm(v)
    return 0.0 if n < 1e-6 else float(np.degrees(np.arccos(np.clip(np.dot(v / n, UP), -1, 1))))


def _flex(a, b, c):
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u / nu, w / nw), -1, 1))))


def features(lm):
    p = lm[:, :3].astype(np.float32); g = lambda k: p[L[k]]
    hip = (g("Lhip") + g("Rhip")) / 2; sh = (g("Lsh") + g("Rsh")) / 2
    torso = sh - hip; tlen = np.linalg.norm(torso)
    if tlen < 1e-6:
        tlen = max(np.linalg.norm(g("Lsh") - g("Rsh")), 1e-3)
    thL = _ang(g("Lkn") - g("Lhip")); thR = _ang(g("Rkn") - g("Rhip"))
    shL = _ang(g("Lank") - g("Lkn")); shR = _ang(g("Rank") - g("Rkn"))
    knL = _flex(g("Lhip"), g("Lkn"), g("Lank")); knR = _flex(g("Rhip"), g("Rkn"), g("Rank"))
    hpL = _flex(g("Lsh"), g("Lhip"), g("Lkn")); hpR = _flex(g("Rsh"), g("Rhip"), g("Rkn"))
    elL = _flex(g("Lsh"), g("Lel"), g("Lwr")); elR = _flex(g("Rsh"), g("Rel"), g("Rwr"))
    ys = p[:, 1]; vext = (ys.max() - ys.min()) / tlen
    rad = np.sqrt((p[:, 0] - hip[0]) ** 2 + (p[:, 2] - hip[2]) ** 2)
    hext = 2 * rad.max() / tlen; aspect = vext / (hext + 1e-6)
    nose_ab = (hip[1] - g("nose")[1]) / tlen
    ank_bl = ((g("Lank")[1] + g("Rank")[1]) / 2 - hip[1]) / tlen
    kn_bl = ((g("Lkn")[1] + g("Rkn")[1]) / 2 - hip[1]) / tlen
    return np.array([_ang(torso), (thL + thR) / 2, abs(thL - thR), (shL + shR) / 2,
                     (knL + knR) / 2, abs(knL - knR), (hpL + hpR) / 2, (elL + elR) / 2,
                     vext, hext, aspect, nose_ab, ank_bl, kn_bl], np.float32)


def raw_coords(lm):
    p = lm[:, :3].astype(np.float32)
    hip = (p[L["Lhip"]] + p[L["Rhip"]]) / 2; sh = (p[L["Lsh"]] + p[L["Rsh"]]) / 2
    return ((p - hip) / (np.linalg.norm(sh - hip) + 1e-6)).reshape(-1)


def rf():
    return RandomForestClassifier(n_estimators=120, max_depth=16, random_state=0, n_jobs=-1)


def per_class_recall(y_true, y_pred):
    out = {}
    for c in CLASSES:
        m = y_true == c
        out[c] = round(float((y_pred[m] == c).mean()), 3) if m.sum() else None
    return out


def main():
    lab = np.load(os.path.join(HERE, "data_probe", "landmarks.npz"), allow_pickle=True)
    Xl = lab["X"]; yl = np.array([str(s) for s in lab["y"]]); gl = np.array([str(s) for s in lab["grp"]])
    Fl = np.array([features(x) for x in Xl]); Cl = np.array([raw_coords(x) for x in Xl])
    print(f"lab: {Xl.shape[0]} poses, {len(set(gl))} groups, dist={dict(collections.Counter(yl))}")

    # lab-internal grouped-CV baseline (the confounded number)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    grip_cv = cross_val_score(rf(), Fl, yl, groups=gl, cv=sgkf).mean()
    raw_cv = cross_val_score(rf(), Cl, yl, groups=gl, cv=sgkf).mean()
    print(f"\n[LAB grouped-CV, confounded]  raw={raw_cv:.3f}   GRIP={grip_cv:.3f}   (raw leads by {(raw_cv-grip_cv)*100:.1f}pt)")

    golds = sys.argv[1:] or [os.path.join(HERE, "data_probe", "gold_lyingv2_poses.npz")]
    # train ONCE on all lab
    grip_m = rf().fit(Fl, yl); raw_m = rf().fit(Cl, yl)
    for gp in golds:
        g = np.load(gp, allow_pickle=True)
        Xg = g["X"]; yg = np.array([str(s) for s in g["y"]])
        keep = np.isin(yg, CLASSES)
        Xg, yg = Xg[keep], yg[keep]
        Fg = np.array([features(x) for x in Xg]); Cg = np.array([raw_coords(x) for x in Xg])
        raw_acc = (raw_m.predict(Cg) == yg).mean(); grip_acc = (grip_m.predict(Fg) == yg).mean()
        print(f"\n=== GOLD {os.path.basename(gp)}: {len(yg)} poses, dist={dict(collections.Counter(yg))} ===")
        print(f"[train lab -> test gold]  raw={raw_acc:.3f}   GRIP={grip_acc:.3f}   "
              f"(GRIP leads by {(grip_acc-raw_acc)*100:+.1f}pt)")
        print(f"  raw  per-class recall: {per_class_recall(yg, raw_m.predict(Cg))}")
        print(f"  GRIP per-class recall: {per_class_recall(yg, grip_m.predict(Fg))}")
        drop_raw = raw_cv - raw_acc; drop_grip = grip_cv - grip_acc
        print(f"  DROP lab->gold:  raw {drop_raw*100:+.1f}pt   GRIP {drop_grip*100:+.1f}pt")
        print(f"  E1 verdict: {'CONFIRMED — raw collapses more than GRIP' if drop_raw > drop_grip + 0.03 else 'not clean-cut (inspect)'}")


if __name__ == "__main__":
    main()
