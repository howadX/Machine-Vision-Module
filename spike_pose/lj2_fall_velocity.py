#!/usr/bin/env python3
"""E2 (LJ2): is there a measured pelvis-velocity threshold separating fall from lie-down?

The classifier emits `lying` either way; a single frame can't tell a fall from a
controlled lie-down. The only discriminating signal is kinematic: how fast the pelvis
dropped. We measure PEAK DOWNWARD pelvis velocity (m/s) per trial on CMU sequences
(60 fps, true metric height + time) and ask whether falls separate from everything else.

Honest design choice: the negative class includes `sit_stand`, not just `lie_down`.
Sitting down is ALSO a downward pelvis motion — if a brisk sit reaches fall-like
velocity, pelvis speed alone is insufficient and the endpoint (floor vs chair) matters.
E2 must expose that confound, not hide it.

Domain caveat (stated up front, carried to LJ2): all CMU falls are stunt/acrobatic
(no elderly falls) -> the measured threshold is calibrated on FAST falls and likely
bounds the true clinical threshold from above.

Run:  ../venv/bin/python3 spike_pose/lj2_fall_velocity.py
"""
import os, numpy as np
import bench

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data_probe", "mocap_transitions.npz")
SEED = 0
N_BOOT = 1000
WIN_S = 0.4         # fall timescale: peak descent measured over a 0.4 s sliding window
FLOOR_M = 0.35      # pelvis this low = the body actually reached the floor (a lying endpoint)
MEDK = 5            # median-filter kernel — kills single-frame FK spikes, keeps the ramp
TRIM = 3            # drop boundary frames (BVH first-frame settling transient)


def _medfilt(x, k):
    k |= 1; r = k // 2
    xp = np.pad(x, r, mode="edge")
    return np.array([np.median(xp[i:i + k]) for i in range(len(x))])


def peak_down_velocity(pelvis, t):
    """Robust peak DOWNWARD pelvis speed (m/s). Median-filter the height (kills single-frame
    FK spikes), trim boundary settling, then take the largest drop over a WIN_S window divided
    by WIN_S — an average descent speed immune to one-frame glitches, not a raw finite diff."""
    if len(pelvis) < 2 * TRIM + 8:
        return 0.0, float(pelvis.min())
    p = _medfilt(pelvis[TRIM:-TRIM], MEDK)
    dt = float(np.median(np.diff(t))); w = max(2, round(WIN_S / dt))
    if len(p) <= w:
        return 0.0, float(pelvis.min())
    drops = (p[:-w] - p[w:]) / (w * dt)      # >0 = pelvis went DOWN over the window
    return float(max(drops.max(), 0.0)), float(pelvis.min())


def auroc(score, label):
    """Rank-based AUROC (label 1 = positive). Ties handled by average rank."""
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score)); ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    npos = label.sum(); nneg = len(label) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def youden_threshold(score, label):
    """Threshold maximizing (TPR - FPR) over candidate cuts."""
    best_t, best_j = 0.0, -1.0
    for t in np.unique(score):
        pred = score >= t
        tpr = (pred[label == 1]).mean() if (label == 1).any() else 0
        fpr = (pred[label == 0]).mean() if (label == 0).any() else 0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, t
    return float(best_t), float(best_j)


def main():
    d = np.load(NPZ, allow_pickle=True)
    pelvis, t, trial, scen = d["pelvis"], d["t"], d["trial"], d["scenario"]
    trials = list(dict.fromkeys(trial.tolist()))   # preserve order

    rows = []   # (trial, scenario, peak_down_v, min_pelvis, is_fall)
    for tr in trials:
        m = trial == tr
        pv, mp = peak_down_velocity(pelvis[m], t[m])
        sc = scen[m][0]
        rows.append((tr, sc, pv, mp, int(sc == "fall")))

    print(f"{'trial':8} {'scenario':9} {'peak_down_v(m/s)':>16} {'min_pelvis(m)':>14}")
    for tr, sc, pv, mp, _ in sorted(rows, key=lambda r: -r[2]):
        print(f"{tr:8} {sc:9} {pv:16.2f} {mp:14.2f}")

    v = np.array([r[2] for r in rows]); mpel = np.array([r[3] for r in rows])
    lab = np.array([r[4] for r in rows]); reached_floor = mpel <= FLOOR_M

    def ci(vv, ll):
        rng = np.random.default_rng(SEED); b = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(vv), len(vv))
            if 0 < ll[idx].sum() < len(idx):
                b.append(auroc(vv[idx], ll[idx]))
        return np.percentile(b, [2.5, 97.5]) if b else (float("nan"), float("nan"))

    # ── Framing A: fall vs EVERYTHING (does high pelvis-velocity flag a fall at all?) ──
    aA = auroc(v, lab); loA, hiA = ci(v, lab)
    aE = auroc(-mpel, lab)   # endpoint (low pelvis) alone
    print(f"\n[A] fall vs all  (falls={int(lab.sum())}, non-fall={int((lab==0).sum())})")
    print(f"    AUROC velocity : {aA:.3f}  CI [{loA:.3f},{hiA:.3f}]")
    print(f"    AUROC endpoint : {aE:.3f}   (min_pelvis; falls end on the floor)")

    # ── Framing B: fall vs CONTROLLED DESCENT — the real ambiguity (both reach the floor) ──
    mB = reached_floor
    vB, lB, trB, scB = v[mB], lab[mB], [rows[i][0] for i in np.where(mB)[0]], [rows[i][1] for i in np.where(mB)[0]]
    aB = auroc(vB, lB); loB, hiB = ci(vB, lB)
    thrB, jB = youden_threshold(vB, lB)
    print(f"\n[B] fall vs controlled descent — floor-reaching only (min_pelvis<={FLOOR_M} m)")
    print(f"    population ({len(trB)}): " + ", ".join(f"{t}/{s}" for t, s in zip(trB, scB)))
    print(f"    AUROC velocity : {aB:.3f}  CI [{loB:.3f},{hiB:.3f}]   threshold {thrB:.2f} m/s (J={jB:.2f})")
    for name, mask in [("fall", lB == 1), ("controlled", lB == 0)]:
        vs = vB[mask]
        if len(vs):
            print(f"      {name:10} peak_down_v: min {vs.min():.2f} median {np.median(vs):.2f} max {vs.max():.2f} m/s (n={len(vs)})")
    print("\npassport:", bench.passport([NPZ]))


if __name__ == "__main__":
    main()
