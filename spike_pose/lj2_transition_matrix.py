#!/usr/bin/env python3
"""E1 part 1 (LJ2): measure the PHYSICS of posture transitions from CMU.

Produces the empirical facts that parameterize the transition-prior (E1 part 2):
  1. transition matrix P(pose_t | pose_{t-1}) at production frame cadence,
  2. how often standing<->lying happens DIRECTLY (one frame, no sitting between) in
     true kinematics — the "physically impossible instant transition" the prior must veto,
  3. posture-segment dwell durations (sit/stand hold time; sit<->stand ~1-2 s per the plan).

Ground truth = the kinematic per-frame label `y` (60 fps), DOWNSAMPLED to production cadence
(~15 fps), because the stabilizer runs once per delivered human-chain frame, not at 60 fps.

Run:  ../venv/bin/python3 spike_pose/lj2_transition_matrix.py
"""
import os, numpy as np
import bench

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data_probe", "mocap_transitions.npz")
CLASSES = ["standing", "sitting", "lying"]
PROD_FPS = 15          # production human-chain cadence (~16 fps engine) — the stabilizer's clock
SRC_FPS = 60           # artifact cadence


def sequences(downsample):
    """Per-trial ground-truth label sequences at PROD_FPS (list of label lists)."""
    d = np.load(NPZ, allow_pickle=True)
    y, t, trial = d["y"], d["t"], d["trial"]
    seqs = []
    for tr in dict.fromkeys(trial.tolist()):
        m = trial == tr
        yy = y[m][np.argsort(t[m])][::downsample]
        seqs.append((tr, list(yy)))
    return seqs


def main():
    step = round(SRC_FPS / PROD_FPS)
    seqs = sequences(step)
    dt = step / SRC_FPS

    # 1. transition matrix (counts of consecutive label pairs) at prod cadence
    idx = {c: i for i, c in enumerate(CLASSES)}
    M = np.zeros((3, 3))
    direct_SL = 0     # standing<->lying in ONE prod-frame (the veto target)
    for _, s in seqs:
        for a, b in zip(s[:-1], s[1:]):
            if a in idx and b in idx:
                M[idx[a], idx[b]] += 1
                if {a, b} == {"standing", "lying"}:
                    direct_SL += 1
    P = M / M.sum(1, keepdims=True).clip(1)

    print(f"Transition matrix P(pose_t | pose_t-1) @ {PROD_FPS} fps  (rows sum to 1)")
    print(f"{'from\\to':>10} " + " ".join(f"{c:>9}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print(f"{c:>10} " + " ".join(f"{P[i, j]:9.4f}" for j in range(3)))
    print(f"\ndirect standing<->lying transitions (1 frame, no sitting between): {direct_SL} "
          f"out of {int(M.sum())} transitions  ->  {'PHYSICALLY ~NEVER (prior should veto)' if direct_SL <= 2 else 'occurs'}")

    # 2. off-diagonal switch rate (how often the true label changes frame-to-frame)
    switch = (M.sum() - np.trace(M)) / M.sum()
    print(f"per-frame switch rate (any label change): {switch:.4f}  -> P(stay) = {1 - switch:.4f}")

    # 3. posture-segment dwell durations (contiguous run lengths per class, in seconds)
    dur = {c: [] for c in CLASSES}
    for _, s in seqs:
        if not s:
            continue
        run_c, run_n = s[0], 1
        for lab in s[1:]:
            if lab == run_c:
                run_n += 1
            else:
                if run_c in dur:
                    dur[run_c].append(run_n * dt)
                run_c, run_n = lab, 1
        if run_c in dur:
            dur[run_c].append(run_n * dt)
    print("\nposture-segment dwell durations (contiguous holds, seconds):")
    for c in CLASSES:
        v = np.array(dur[c])
        if len(v):
            print(f"  {c:9} n={len(v):3d}  median {np.median(v):.2f}s  p10 {np.percentile(v, 10):.2f}s  "
                  f"p90 {np.percentile(v, 90):.2f}s  min {v.min():.2f}s")

    print("\npassport:", bench.passport([NPZ]))


if __name__ == "__main__":
    main()
