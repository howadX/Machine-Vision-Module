#!/usr/bin/env python3
"""P4.0 (15.6): does the ABSOLUTE floor matter for fall detection, or can a floor-FREE
relative signal do the floor gate's job?

Context. The shipped E2 rule (LJ2) is: peak downward pelvis velocity >= thr AND the pelvis
reached an absolute floor height (min_pelvis <= 0.35 m). The floor gate's measured job was to
reject the ONE velocity false-positive: a playground jump (pelvis dips fast at ~1.48 m/s but
returns to standing height — it never STAYS down). Prod has no calibrated floor plane
(position_3d is camera-origin, absolute height uncalibrated), so we test whether a floor-FREE
signal recovers that rejection.

Floor-free signal: NET DROP = baseline_pelvis - end_pelvis, a DIFFERENCE of two heights so the
floor origin cancels exactly. It measures "how far the pelvis ended below its own recent max and
STAYED" — egocentric, no surveyor's frame (north-star: the eye doesn't measure the floor, it sees
you went down and stayed down). baseline = robust max over the trial; end = median of the last
WIN_S (the sustained endpoint, NOT the transient min — that's what separates a jump from a fall).

Three rules scored against ground truth (scenario == 'fall'):
  1. velocity-only            : peak_down_v >= thr_v
  2. velocity + ABS floor (E2): peak_down_v >= thr_v AND min_pelvis <= 0.35 m   (shipped)
  3. velocity + REL net-drop  : peak_down_v >= thr_v AND net_drop  >= thr_d      (floor-free)

Run:  ../venv/bin/python3 spike_pose/lj6_fall_ablation.py
"""
import os, numpy as np
import bench
from lj2_fall_velocity import peak_down_velocity, auroc, youden_threshold, WIN_S, FLOOR_M, MEDK, TRIM, _medfilt

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data_probe", "mocap_transitions.npz")
SEED = 0
N_BOOT = 1000


def net_drop(pelvis, t):
    """Floor-free 'went down and stayed down' signal: baseline (robust recent max) minus the
    sustained endpoint (median of the last WIN_S). A difference of heights -> floor cancels."""
    if len(pelvis) < 2 * TRIM + 8:
        return 0.0, float(pelvis.max()), float(pelvis[-1])
    p = _medfilt(pelvis[TRIM:-TRIM], MEDK)
    dt = float(np.median(np.diff(t))); w = max(2, round(WIN_S / dt))
    baseline = float(np.percentile(p, 95))          # robust standing height (not a lone spike)
    end = float(np.median(p[-w:]))                   # sustained endpoint over the last window
    return float(baseline - end), baseline, end


def rule_confusion(pred, lab):
    tp = int(((pred == 1) & (lab == 1)).sum()); fp = int(((pred == 1) & (lab == 0)).sum())
    fn = int(((pred == 0) & (lab == 1)).sum()); tn = int(((pred == 0) & (lab == 0)).sum())
    return tp, fp, fn, tn


def main():
    d = np.load(NPZ, allow_pickle=True)
    pelvis, t, trial, scen = d["pelvis"], d["t"], d["trial"], d["scenario"]
    trials = list(dict.fromkeys(trial.tolist()))

    rows = []   # trial, scen, peak_v, min_pelvis(abs), net_drop(rel), baseline, end, is_fall
    for tr in trials:
        m = trial == tr
        pv, mp = peak_down_velocity(pelvis[m], t[m])
        nd, base, end = net_drop(pelvis[m], t[m])
        sc = scen[m][0]
        rows.append((tr, sc, pv, mp, nd, base, end, int(sc == "fall")))

    print(f"{'trial':8} {'scenario':9} {'peak_v':>7} {'min_abs':>8} {'net_drop':>9} {'base':>6} {'end':>6}")
    for tr, sc, pv, mp, nd, base, end, _ in sorted(rows, key=lambda r: -r[2]):
        print(f"{tr:8} {sc:9} {pv:7.2f} {mp:8.2f} {nd:9.2f} {base:6.2f} {end:6.2f}")

    v = np.array([r[2] for r in rows]); mpel = np.array([r[3] for r in rows])
    nd = np.array([r[4] for r in rows]); lab = np.array([r[7] for r in rows])

    def ci(score, ll):
        rng = np.random.default_rng(SEED); b = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(score), len(score))
            if 0 < ll[idx].sum() < len(idx):
                b.append(auroc(score[idx], ll[idx]))
        return np.percentile(b, [2.5, 97.5]) if b else (float("nan"), float("nan"))

    print(f"\n=== single-signal AUROC (fall vs all; falls={int(lab.sum())}/{len(lab)}) ===")
    for name, s in [("velocity        ", v), ("abs endpoint(-min)", -mpel), ("REL net_drop     ", nd)]:
        a = auroc(s, lab); lo, hi = ci(s, lab)
        print(f"  {name}: AUROC {a:.3f}  CI [{lo:.3f},{hi:.3f}]")

    # velocity threshold from the shipped analysis (Youden on floor-reaching subset); reuse it
    thr_v, _ = youden_threshold(v[mpel <= FLOOR_M], lab[mpel <= FLOOR_M])
    thr_d, jd = youden_threshold(nd, lab)
    print(f"\nthresholds: velocity {thr_v:.2f} m/s   net_drop {thr_d:.2f} m (Youden J={jd:.2f})")

    rules = {
        "1 velocity-only            ": (v >= thr_v),
        "2 velocity + ABS floor (E2)": (v >= thr_v) & (mpel <= FLOOR_M),
        "3 velocity + REL net-drop  ": (v >= thr_v) & (nd >= thr_d),
    }
    print(f"\n=== rule-level confusion (threshold operating point) ===")
    print(f"{'rule':28} {'TP':>3} {'FP':>3} {'FN':>3} {'TN':>3}   false-positive trials")
    for name, pred in rules.items():
        tp, fp, fn, tn = rule_confusion(pred.astype(int), lab)
        fps = [rows[i][0] + "/" + rows[i][1] for i in np.where((pred) & (lab == 0))[0]]
        print(f"{name} {tp:3} {fp:3} {fn:3} {tn:3}   {', '.join(fps) if fps else '-'}")

    print("\npassport:", bench.passport([NPZ]))


if __name__ == "__main__":
    main()
