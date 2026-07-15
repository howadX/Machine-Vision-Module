#!/usr/bin/env python3
"""Confirm the z-ablation finding on gold #6 (binary sitting/standing, labels '0'/'1').

Mapping 0/1 -> sitting/standing is resolved by a TRANSPARENT physical signal (mean body
vertical extent: standing is much taller), independent of the RF models being compared.
Then: train GRIP-3D, GRIP-2D(z=0), raw on lab (sit+stand only) -> test gold #6.
"""
import numpy as np, sys, collections
sys.path.insert(0, "spike_pose")
from r3_confound_test import features, raw_coords, rf

BIN = ["sitting", "standing"]


def feats2d(lm):
    lm2 = lm.copy(); lm2[:, 2] = 0.0
    return features(lm2)


def vext(lm):  # 2D vertical body extent over confident points, transparent standing signal
    p = lm[p_conf(lm), 1] if (p_conf(lm)).any() else lm[:, 1]
    return float(p.max() - p.min())


def p_conf(lm):
    return lm[:, 3] > 0.3


lab = np.load("spike_pose/data_probe/landmarks.npz", allow_pickle=True)
Xl = lab["X"]; yl = np.array([str(s) for s in lab["y"]])
sel = np.isin(yl, BIN); Xl, yl = Xl[sel], yl[sel]

g = np.load("spike_pose/data_probe/gold_sitstand_fullframe.npz", allow_pickle=True)
Xg = g["X"]; yn = np.array([str(s) for s in g["y"]])  # '0' / '1'
print(f"gold #6: {len(yn)} poses, numeric dist={dict(collections.Counter(yn))}")

# --- resolve 0/1 -> sitting/standing via vertical extent (standing taller) ---
ve = {n: np.mean([vext(x) for x in Xg[yn == n]]) for n in ["0", "1"]}
print(f"mean vertical extent: 0={ve['0']:.3f}  1={ve['1']:.3f}  (higher = standing)")
standing_num = max(ve, key=ve.get); sitting_num = min(ve, key=ve.get)
mapping = {standing_num: "standing", sitting_num: "sitting"}
print(f"INFERRED mapping (confirm!): {mapping}")
yg = np.array([mapping[n] for n in yn])
print(f"mapped dist={dict(collections.Counter(yg))}")

Cl = np.array([raw_coords(x) for x in Xl]); Cg = np.array([raw_coords(x) for x in Xg])
print("\n[train lab(sit+stand) -> test gold #6]")
for name, fn in [("raw", None), ("3D-GRIP", features), ("2D-GRIP(z=0)", feats2d)]:
    if fn is None:
        m = rf().fit(Cl, yl); pred = m.predict(Cg)
    else:
        Fl = np.array([fn(x) for x in Xl]); Fg = np.array([fn(x) for x in Xg])
        m = rf().fit(Fl, yl); pred = m.predict(Fg)
    acc = (pred == yg).mean()
    rec = {c: round(float((pred[yg == c] == c).mean()), 2) for c in BIN}
    print(f"  {name:14} acc={acc:.3f}  recall={rec}")
