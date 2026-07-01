"""P0 spike — engineered features vs raw-coords, with azimuth-robustness test.

Hypothesis (method C): gravity-preserving, azimuth-invariant geometric features
(joint angles / inclinations / ratios) fed to a shallow classifier
 (a) separate standing/sitting/lying well, and
 (b) stay accurate under camera-azimuth rotation,
where a raw-normalized-landmark classifier (current VAE-style input) degrades.

Go/no-go metric = (clean acc >= baseline) AND (azimuth-rotated acc barely drops
while baseline collapses). The clean-vs-rotated DELTA is immune to any source
imbalance in the data (same samples, only rotated), so it is a clean signal.

Run from spike_pose/ with the project venv, after spike_data.py.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.pose_encoder import normalize_landmarks
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "data_probe", "landmarks.npz")
UP = np.array([0.0, -1.0, 0.0], np.float32)   # MediaPipe: y grows downward -> up = -y
L = dict(nose=0, Lsh=11, Rsh=12, Lel=13, Rel=14, Lwr=15, Rwr=16,
         Lhip=23, Rhip=24, Lkn=25, Rkn=26, Lank=27, Rank=28)


def azimuth_rotate(lm, deg):
    """Rotate 33x4 landmarks about the vertical (y) axis by deg (camera azimuth)."""
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    out = lm.copy()
    x, z = lm[:, 0], lm[:, 2]
    out[:, 0] = x * c + z * s
    out[:, 2] = -x * s + z * c
    return out


def _ang_to_vert(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(v / n, UP), -1, 1))))


def _flex(a, b, c):
    u, w = a - b, c - b
    nu, nw = np.linalg.norm(u), np.linalg.norm(w)
    if nu < 1e-6 or nw < 1e-6:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u / nu, w / nw), -1, 1))))


def features(lm):
    """33x4 (x,y,z,vis) -> compact azimuth-invariant, gravity-preserving feature vector."""
    p = lm[:, :3].astype(np.float32)
    g = lambda k: p[L[k]]
    hip = (g("Lhip") + g("Rhip")) / 2
    sh = (g("Lsh") + g("Rsh")) / 2
    torso = sh - hip
    tlen = np.linalg.norm(torso)
    if tlen < 1e-6:
        tlen = max(np.linalg.norm(g("Lsh") - g("Rsh")), 1e-3)

    # inclinations to gravity-vertical (KEY for standing vs lying)
    torso_inc = _ang_to_vert(torso)
    thigh_L = _ang_to_vert(g("Lkn") - g("Lhip")); thigh_R = _ang_to_vert(g("Rkn") - g("Rhip"))
    shank_L = _ang_to_vert(g("Lank") - g("Lkn")); shank_R = _ang_to_vert(g("Rank") - g("Rkn"))
    # joint flexions (fully rotation-invariant) (KEY for sitting)
    knee_L = _flex(g("Lhip"), g("Lkn"), g("Lank")); knee_R = _flex(g("Rhip"), g("Rkn"), g("Rank"))
    hip_L = _flex(g("Lsh"), g("Lhip"), g("Lkn")); hip_R = _flex(g("Rsh"), g("Rhip"), g("Rkn"))
    elb_L = _flex(g("Lsh"), g("Lel"), g("Lwr")); elb_R = _flex(g("Rsh"), g("Rel"), g("Rwr"))

    # cloud shape (azimuth-invariant): vertical extent vs horizontal radial extent
    ys = p[:, 1]
    vext = (ys.max() - ys.min()) / tlen
    rad = np.sqrt((p[:, 0] - hip[0]) ** 2 + (p[:, 2] - hip[2]) ** 2)
    hext = 2 * rad.max() / tlen
    aspect = vext / (hext + 1e-6)

    # vertical placement relative to hips (signed, gravity-aware) / torso
    nose_above = (hip[1] - g("nose")[1]) / tlen
    ank_below = ((g("Lank")[1] + g("Rank")[1]) / 2 - hip[1]) / tlen
    kn_below = ((g("Lkn")[1] + g("Rkn")[1]) / 2 - hip[1]) / tlen

    return np.array([
        torso_inc,
        (thigh_L + thigh_R) / 2, abs(thigh_L - thigh_R),
        (shank_L + shank_R) / 2,
        (knee_L + knee_R) / 2, abs(knee_L - knee_R),
        (hip_L + hip_R) / 2,
        (elb_L + elb_R) / 2,
        vext, hext, aspect,
        nose_above, ank_below, kn_below,
    ], np.float32)


def raw_coords(lm):
    """Baseline input = current VAE-style: normalize_landmarks -> flat 99."""
    norm, _ = normalize_landmarks(lm)
    return norm.reshape(-1)


def build(X_raw, fn):
    return np.array([fn(x) for x in X_raw], np.float32)


def evaluate(name, Xtr, ytr, Xte, yte, Xte_rot):
    res = {}
    for mdl_name, mdl in [
        ("MLP", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=0)),
        ("RF", RandomForestClassifier(n_estimators=300, random_state=0)),
    ]:
        mdl.fit(Xtr, ytr)
        acc = accuracy_score(yte, mdl.predict(Xte))
        f1 = f1_score(yte, mdl.predict(Xte), average="macro")
        rot = {d: accuracy_score(yte, mdl.predict(Xr)) for d, Xr in Xte_rot.items()}
        res[mdl_name] = (acc, f1, rot)
        rotstr = " ".join(f"az{d}={v:.3f}" for d, v in rot.items())
        print(f"  [{name:8}/{mdl_name}] clean acc={acc:.3f} f1={f1:.3f} | {rotstr}")
    return res


if __name__ == "__main__":
    d = np.load(NPZ, allow_pickle=True)
    Xr, y, src = d["X"], d["y"], d["src"]
    import collections
    print("dataset:", dict(collections.Counter(y)), "| sources:", dict(collections.Counter(src)))

    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=0)
    Xr_tr, Xr_te = Xr[tr], Xr[te]
    ytr, yte = y[tr], y[te]

    # rotated test copies (azimuth nuisance)
    ROT = [15, 30, 45, 90]
    Xr_te_rot = {d_: np.array([azimuth_rotate(x, d_) for x in Xr_te]) for d_ in ROT}

    print("\n== FEATURES (proposed, method C) ==")
    Ftr = build(Xr_tr, features)
    Fte = build(Xr_te, features)
    Fte_rot = {d_: build(Xr_te_rot[d_], features) for d_ in ROT}
    evaluate("feat", Ftr, ytr, Fte, yte, Fte_rot)

    print("\n== RAW COORDS (baseline = current VAE-style input) ==")
    Ctr = build(Xr_tr, raw_coords)
    Cte = build(Xr_te, raw_coords)
    Cte_rot = {d_: build(Xr_te_rot[d_], raw_coords) for d_ in ROT}
    evaluate("raw", Ctr, ytr, Cte, yte, Cte_rot)

    # confusion for feature-MLP (production-relevant standing/sitting confusion)
    m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1500, random_state=0).fit(Ftr, ytr)
    labels = sorted(set(y))
    print("\nfeature-MLP confusion (rows=true):", labels)
    print(confusion_matrix(yte, m.predict(Fte), labels=labels))
