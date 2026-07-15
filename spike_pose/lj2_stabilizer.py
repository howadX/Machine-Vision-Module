#!/usr/bin/env python3
"""E1 part 2 (LJ2): transition-prior vs hand-tuned dwell on the flip-rate <-> latency frontier.

The current PostureStabilizer (mvm-core/src/posture.rs) is a symmetric, label-agnostic dwell
machine: ANY flip needs `dwell` consecutive confident frames. It can only RIDE the
flip/latency curve. A transition-prior calibrated on CMU physics (lj2_transition_matrix.py)
can SHIFT the curve, because standing<->lying is ~forbidden (0.0013) while standing->sitting
is not (0.027): the prior vetoes the impossible instant flip without taxing the possible one.

Protocol (frozen bench discipline):
  - Emitter (noise source, variant c FLOOR): a 2D-GRIP forest trained on lab+gold ONLY
    (landmarks_v2, NO mocap) -> genuine cross-domain per-frame errors on CMU projections.
    No CMU trial is in the emitter's training set (zero leakage), matching production.
  - variant c STRESS: optional landmark dropout+jitter before features (desk-occlusion-like).
  - Ground truth: kinematic label `y`, dwell-cleaned (segments >= H frames) so we score against
    REAL posture changes, not 1-frame labeler blips. Transition onset timing preserved.
  - Metrics: spurious flips / min (committed changes not explained by a real transition) and
    reaction latency (ms from true onset to committed match). Swept over each method's knob.

Run:  ../venv/bin/python3 spike_pose/lj2_stabilizer.py
"""
import os, numpy as np
from sklearn.ensemble import RandomForestClassifier
import bench
from mocap_sanity_gate import features_13

HERE = os.path.dirname(os.path.abspath(__file__))
TRANS = os.path.join(HERE, "data_probe", "mocap_transitions.npz")
LABGOLD = os.path.join(HERE, "data_probe", "landmarks_v2.npz")
CLASSES = ["lying", "sitting", "standing"]           # sklearn sorts classes alphabetically
PROD_FPS = 15
SRC_FPS = 60
H = 8                 # min frames (~0.5 s) a label must hold to count as a REAL posture
LMAX = 15             # a committed flip within LMAX frames (~1 s) after a real onset is "legit"
SEED = 0

# measured @15fps by lj2_transition_matrix.py — the physics that parameterizes the prior
A_MEASURED = np.array([
    # to:  standing  sitting   lying
    [0.9721, 0.0266, 0.0013],   # from standing
    [0.0211, 0.9724, 0.0066],   # from sitting
    [0.0000, 0.0267, 0.9733],   # from lying
])
ORDER = ["standing", "sitting", "lying"]             # A_MEASURED index order


def feat2d(lm):
    l = lm.copy(); l[:, 2] = 0.0
    return features_13(l)


def train_emitter():
    d = np.load(LABGOLD, allow_pickle=True)
    F = np.array([feat2d(x) for x in d["X"]])
    clf = RandomForestClassifier(n_estimators=60, max_depth=9, min_samples_leaf=10,
                                 random_state=SEED, n_jobs=-1).fit(F, d["y"])
    return clf


def clean_truth(y):
    """Dwell-clean ground truth: a label change commits only after it holds H frames; the
    transition ONSET (first frame of the new run) is recorded. Returns (stable_seq, onsets)
    where onsets = list of (onset_index, new_label)."""
    stable = [y[0]] * len(y); cur = y[0]; onsets = []
    i = 0
    while i < len(y):
        if y[i] != cur:
            j = i
            while j < len(y) and y[j] == y[i]:
                j += 1
            if j - i >= H:                     # a real, sustained change
                onsets.append((i, y[i])); cur = y[i]
            # else: a blip -> ignore, stable stays `cur`
            for k in range(i, j):
                stable[k] = cur
            i = j
        else:
            stable[i] = cur; i += 1
    return stable, onsets


# ---------------- stabilizers (each takes emission stream, returns committed seq) ----------
def run_raw(argmax, conf, proba):
    return list(argmax)


def run_dwell(argmax, conf, proba, dwell=3, min_conf=0.5):
    """Port of PostureStabilizer (posture.rs): confident, consecutive, dwell-gated commit."""
    committed = None; cand = None; streak = 0; out = []
    for lab, c in zip(argmax, conf):
        if c < min_conf:                       # ambiguous -> never flips, breaks streak
            cand = committed; streak = 0
        elif committed is None:                # bootstrap first confident sight
            committed = lab; cand = lab; streak = 0
        elif lab == committed:
            cand = lab; streak = 0
        else:
            if cand == lab: streak += 1
            else: cand = lab; streak = 1
            if streak >= dwell: committed = lab; streak = 0
        out.append(committed if committed is not None else lab)
    return out


def run_hybrid(argmax, conf, proba, dwell=3, sl_dwell=12):
    """The shipped form (decision A+E2): the cheap dwell machine for the general case, PLUS a
    physics veto on the ONE impossible edge — a direct standing<->lying commit needs `sl_dwell`
    consecutive frames (a fall must earn it) instead of `dwell`, so a normal lie-down is forced
    to pass through sitting. Guarantees no aggregate regression vs dwell (it IS dwell elsewhere)
    while suppressing the fall-relevant standing<->lying flip that label-agnostic dwell cannot."""
    committed = None; cand = None; streak = 0; out = []
    for lab, c in zip(argmax, conf):
        need = sl_dwell if (committed is not None and {committed, lab} == {"standing", "lying"}) else dwell
        if c < 0.5:
            cand = committed; streak = 0
        elif committed is None:
            committed = lab; cand = lab; streak = 0
        elif lab == committed:
            cand = lab; streak = 0
        else:
            if cand == lab: streak += 1
            else: cand = lab; streak = 1
            if streak >= need: committed = lab; streak = 0
        out.append(committed if committed is not None else lab)
    return out


def run_prior(argmax, conf, proba, stick=0.0):
    """Causal online forward Bayesian filter (HMM-like). Transition model = measured A,
    reshaped by signed `stick`: stick>0 blends toward identity (stickier, higher latency),
    stick<0 blends toward uniform (looser, lower latency) so the prior spans the SAME latency
    range as dwell for a fair frontier. Emission = classifier softmax; commit = argmax belief."""
    if stick >= 0:
        A = stick * np.eye(3) + (1 - stick) * A_MEASURED
    else:
        A = (1 + stick) * A_MEASURED + (-stick) * (np.ones((3, 3)) / 3)
    A = A / A.sum(1, keepdims=True)
    pmap = {c: i for i, c in enumerate(CLASSES)}       # proba column order (sklearn)
    b = np.ones(3) / 3; out = []
    for k in range(len(argmax)):
        e = np.array([proba[k][pmap[c]] for c in ORDER])   # emission in A's order
        b = (A.T @ b) * (e + 1e-9)
        b = b / b.sum()
        out.append(ORDER[int(b.argmax())])
    return out


# ----------------------------- metrics -----------------------------
def score(committed, stable, onsets):
    """Returns (spurious_total, spurious_SL, latencies). A committed change is legit if a real
    onset with the same new label occurred within the preceding LMAX frames; else spurious.
    spurious_SL isolates the standing<->lying mode — the physically-impossible flip the prior
    is specifically designed to veto (the hypothesis is about THIS mode, not adjacent confusion)."""
    spurious = spurious_sl = 0
    for i in range(1, len(committed)):
        if committed[i] != committed[i - 1]:
            legit = any(o_lab == committed[i] and 0 <= i - o_idx <= LMAX for o_idx, o_lab in onsets)
            if not legit:
                spurious += 1
                if {committed[i], committed[i - 1]} == {"standing", "lying"}:
                    spurious_sl += 1
    lats = []
    for o_idx, o_lab in onsets:
        hit = next((k for k in range(o_idx, len(committed)) if committed[k] == o_lab), None)
        lats.append((hit - o_idx) if hit is not None else LMAX * 2)   # miss penalized
    return spurious, spurious_sl, lats


def emit_stream(X, clf, rng, dropout=0.0, jitter=0.0):
    """Per-frame (argmax, conf, proba) from the emitter; optional landmark noise (stress)."""
    if dropout or jitter:
        X = X.copy()
        for x in X:
            vis = x[:, 3] > 0
            if jitter: x[vis, :2] += rng.normal(0, jitter, (vis.sum(), 2))
            if dropout:
                drop = rng.random(33) < dropout
                x[drop] = 0.0
    F = np.array([feat2d(x) for x in X])
    proba = clf.predict_proba(F)
    argmax = np.array(CLASSES)[proba.argmax(1)]
    conf = proba.max(1)
    return argmax, conf, proba


def evaluate(trials_data, method, knob, key):
    """Aggregate flip-rate (per min), standing<->lying flip-rate, and mean latency (ms)."""
    tot_spur = tot_sl = 0; tot_min = 0.0; all_lat = []
    for (argmax, conf, proba, stable, onsets, nfr) in trials_data:
        committed = method(argmax, conf, proba, **{key: knob}) if key else method(argmax, conf, proba)
        sp, sl, lats = score(committed, stable, onsets)
        tot_spur += sp; tot_sl += sl; tot_min += nfr / PROD_FPS / 60.0; all_lat += lats
    lat_ms = np.mean(all_lat) / PROD_FPS * 1000 if all_lat else 0.0
    return tot_spur / tot_min, tot_sl / tot_min, lat_ms


def main():
    clf = train_emitter()
    d = np.load(TRANS, allow_pickle=True)
    X, y, t, trial = d["X"], d["y"], d["t"], d["trial"]
    step = round(SRC_FPS / PROD_FPS)
    rng = np.random.default_rng(SEED)

    def build(dropout, jitter):
        data = []
        for tr in dict.fromkeys(trial.tolist()):
            m = trial == tr; order = np.argsort(t[m])
            Xt = X[m][order][::step]; yt = list(y[m][order][::step])
            if len(yt) < H + 2:
                continue
            argmax, conf, proba = emit_stream(Xt, clf, rng, dropout, jitter)
            stable, onsets = clean_truth(yt)
            data.append((argmax, conf, proba, stable, onsets, len(yt)))
        return data

    for tag, (dropout, jitter) in [("FLOOR (real xdomain errors)", (0.0, 0.0)),
                                   ("STRESS (dropout .06 + jitter .008, desk-occlusion-like)", (0.06, 0.008))]:
        data = build(dropout, jitter)
        n_on = sum(len(o) for *_, o, _ in data)
        r_flip, r_sl, r_lat = evaluate(data, run_raw, None, None)
        print(f"\n=== {tag} ===   trials={len(data)}  real transitions={n_on}")
        print(f"{'method':22} {'knob':>6} {'flip/min':>9} {'SL-flip/min':>12} {'latency_ms':>11}")
        print(f"{'raw argmax':22} {'-':>6} {r_flip:9.2f} {r_sl:12.2f} {r_lat:11.0f}")
        for dw in [1, 2, 3, 4, 5, 6]:
            f, sl, l = evaluate(data, run_dwell, dw, "dwell")
            print(f"{'dwell (posture.rs)':22} {dw:>6} {f:9.2f} {sl:12.2f} {l:11.0f}")
        for st in [0.0, 0.7, 0.93]:
            f, sl, l = evaluate(data, run_prior, st, "stick")
            print(f"{'prior (forward filt)':22} {st:>6.2f} {f:9.2f} {sl:12.2f} {l:11.0f}")
        hf, hsl, hl = evaluate(data, run_hybrid, 3, "dwell")   # shipped form, base dwell 3
        print(f"{'HYBRID dwell3+SL-veto':22} {'-':>6} {hf:9.2f} {hsl:12.2f} {hl:11.0f}")

        # ── bootstrap CI: the SHIPPED hybrid vs the dwell it replaces (decision A+E2) ──
        # Same base dwell -> matched latency by construction; only the SL edge differs.
        def per_trial_rates(method, knob, key):
            out = []
            for td in data:
                (argmax, conf, proba, stable, onsets, nfr) = td
                committed = method(argmax, conf, proba, **{key: knob})
                sp, sl, _ = score(committed, stable, onsets)
                out.append((sp, sl, nfr / PROD_FPS / 60.0))
            return np.array(out, float)
        A_ = per_trial_rates(run_dwell, 3, "dwell")      # dwell baseline (what we replace)
        B_ = per_trial_rates(run_hybrid, 3, "dwell")     # shipped hybrid
        rng2 = np.random.default_rng(SEED)
        d_tot, d_sl = [], []
        for _ in range(2000):
            idx = rng2.integers(0, len(data), len(data))
            mn = A_[idx, 2].sum()
            d_tot.append((B_[idx, 0].sum() - A_[idx, 0].sum()) / mn)   # hybrid - dwell, flips/min
            d_sl.append((B_[idx, 1].sum() - A_[idx, 1].sum()) / mn)
        lo_t, hi_t = np.percentile(d_tot, [2.5, 97.5]); lo_s, hi_s = np.percentile(d_sl, [2.5, 97.5])
        print(f"  Δ(hybrid − dwell3), bootstrap CI over trials (negative = hybrid better):")
        print(f"    total flip/min : {np.mean(d_tot):+.2f}  CI [{lo_t:+.2f}, {hi_t:+.2f}]  "
              f"← hybrid ≤ dwell by construction (SL-veto only removes flips)")
        print(f"    SL-flip/min    : {np.mean(d_sl):+.2f}  CI [{lo_s:+.2f}, {hi_s:+.2f}]  "
              f"{'← established' if hi_s < 0 or lo_s > 0 else '← suggestive (CI spans 0)'}")

    print("\npassport:", bench.passport([TRANS, LABGOLD]))


if __name__ == "__main__":
    main()
