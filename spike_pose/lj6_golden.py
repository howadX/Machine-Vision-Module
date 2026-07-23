#!/usr/bin/env python3
"""LJ6 / phase-15.6 P0 — parity fixture exporter for the v0.2.0 pose port.

Emits `crates/mvm-posture/src/pose_v2_golden.in` — the bit-exact reference the Rust port must
reproduce. Replaces the stale `pose_p1_golden.in` (14-feat + VAE).

Design: the reference is computed *from the shipped artifacts themselves* — an independent reader
of `ship/pose_rf_2dgrip_v2.bin` (tree traversal) and `ship/pose_maha.bin` (min-Mahalanobis). So the
fixture pins exactly what a Rust loader of the same bytes must output. Inputs are a deterministic
spread of real MediaPipe landmarks from the training npz (2 lab samples per class + a mocap deep-bend
+ a mocap lying), exercising all 3 classes, the scarce lying class, and the flexion channel.

Tuple layout (per case), matching the P1 Rust `const GOLDEN`:
    ([[f32;4];33] input, [f32;12] features_z0, [f32;3] forest_probs, f32 maha_score, f32 flexion, &str label)
  - features_z0     = mocap_sanity_gate.features_13 with z zeroed (Python order = Rust order)
  - forest_probs    = soft-vote over the serialized forest, columns sklearn `classes_` = [lying,sitting,standing]
  - maha_score      = min_c sqrt((F-mu_c)^T P_c (F-mu_c)); OOD iff > thr (thr lives in the .bin, not here)
  - flexion         = features_z0[0] (raw torso_inc, +0 weight)
  - label           = POSE_LABELS[argmax(forest_probs)]

Run: venv/bin/python3 spike_pose/lj6_golden.py   (deterministic; empty diff on re-run)
"""
import os, sys, struct, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from mocap_sanity_gate import features_13, FEATS

SHIP = os.path.join(HERE, "ship")
OUT = os.path.join(HERE, "..", "mvm", "crates", "mvm-posture", "src", "pose_v2_golden.in")
POSE_LABELS = ["lying", "sitting", "standing"]   # sklearn classes_ order = Rust POSE_LABELS


def feat_2d(lm):
    l = np.asarray(lm, np.float32).copy(); l[:, 2] = 0.0
    return features_13(l).astype(np.float32)


# ---- independent readers of the shipped artifacts (mirror the Rust loaders) ----
def load_forest(path):
    b = open(path, "rb").read(); off = 0
    (n_trees,) = struct.unpack_from("<I", b, off); off += 4
    trees = []
    for _ in range(n_trees):
        (n_nodes,) = struct.unpack_from("<I", b, off); off += 4
        nodes = []
        for _ in range(n_nodes):
            feat, thr, left, right, p0, p1, p2 = struct.unpack_from("<ifii3f", b, off); off += 28
            nodes.append((feat, thr, left, right, (p0, p1, p2)))
        trees.append(nodes)
    assert off == len(b), f"forest trailing bytes: {off} != {len(b)}"
    return trees


def forest_proba(trees, F):
    acc = np.zeros(3, np.float64)
    for nodes in trees:
        n = 0
        while nodes[n][2] != -1:            # children_left == -1 => leaf
            feat, thr, left, right, _ = nodes[n]
            n = left if F[feat] <= thr else right
        acc += nodes[n][4]
    return (acc / len(trees)).astype(np.float32)


def load_maha(path):
    b = open(path, "rb").read()
    n_cls, n_feat = struct.unpack_from("<II", b, 0)
    (thr,) = struct.unpack_from("<f", b, 8); off = 12
    means, iP = [], []
    for _ in range(n_cls):
        mu = np.frombuffer(b, "<f4", n_feat, off).astype(np.float32); off += 4 * n_feat
        P = np.frombuffer(b, "<f4", n_feat * n_feat, off).reshape(n_feat, n_feat).astype(np.float32)
        off += 4 * n_feat * n_feat
        means.append(mu); iP.append(P)
    assert off == len(b), f"maha trailing bytes: {off} != {len(b)}"
    return thr, means, iP


def maha_score(means, iP, F):
    F = F.astype(np.float32)
    return float(np.sqrt(min((F - m) @ P @ (F - m) for m, P in zip(means, iP))))


# ---- deterministic input spread ----
def pick_inputs():
    d = np.load(os.path.join(HERE, "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    X = d["X"]; y = np.array([str(s) for s in d["y"]]); src = np.array([str(s) for s in d["src"]])
    lab = src == "lab"
    cases = []
    for c in POSE_LABELS:                    # 2 lab samples per class: first + midpoint index
        idx = np.where(lab & (y == c))[0]
        for j in (idx[0], idx[len(idx) // 2]):
            cases.append((np.asarray(X[j], np.float32), c, f"lab:{c}:{int(j)}"))
    mc = np.load(os.path.join(HERE, "data_probe", "mocap_views.npz"), allow_pickle=True)
    Xm = mc["X"]; ym = np.array([str(s) for s in mc["y"]])
    ti = np.array([feat_2d(x)[0] for x in Xm])          # torso_inc per mocap sample
    bend = int(np.argmax(ti))                            # deepest forward-lean = a bend
    cases.append((np.asarray(Xm[bend], np.float32), ym[bend], f"mocap:bend:{bend}"))
    lyi = int(np.where(ym == "lying")[0][0])
    cases.append((np.asarray(Xm[lyi], np.float32), "lying", f"mocap:lying:{lyi}"))
    return cases


# ---- Rust-literal formatting ----
def fmt(x):
    s = f"{float(np.float32(x)):.8g}"
    if not any(ch in s for ch in ".eEnN"):   # ensure a float literal (Rust rejects bare `1`/`0` in [f32;N])
        s += ".0"
    return s


def arr(v):
    return "[" + ",".join(fmt(x) for x in v) + "]"


def input33(lm):
    return "[" + ",".join(arr(row) for row in lm) + "]"


def main():
    trees = load_forest(os.path.join(SHIP, "pose_rf_2dgrip_v2.bin"))
    thr, means, iP = load_maha(os.path.join(SHIP, "pose_maha.bin"))
    print(f"forest {len(trees)} trees; maha thr={thr:.4f}, {len(means)} classes x {means[0].shape[0]}-d")

    lines = ["// GENERATED by spike_pose/lj6_golden.py — do not hand-edit. Regenerate on artifact change.",
             "// (input_33x4, features12[z=0], forest_probs3 [lying,sitting,standing], maha_score, flexion, label)",
             "&["]
    for lm, true_y, tag in pick_inputs():
        F = feat_2d(lm)
        probs = forest_proba(trees, F)
        score = maha_score(means, iP, F)
        flex = float(F[0])
        label = POSE_LABELS[int(np.argmax(probs))]
        print(f"  {tag:22s} true={true_y:8s} pred={label:8s} "
              f"probs={np.round(probs,3).tolist()} maha={score:.3f}{' OOD' if score>thr else ''} flex={flex:.1f}")
        lines.append(f'({input33(lm)},{arr(F)},{arr(probs)},{fmt(score)}f32,{fmt(flex)}f32,"{label}"),')
    lines.append("]\n")
    txt = "\n".join(lines)
    open(os.path.abspath(OUT), "w").write(txt)
    print(f"\nwrote {os.path.abspath(OUT)} ({len(txt)} B, {len(pick_inputs())} cases)")


if __name__ == "__main__":
    main()
