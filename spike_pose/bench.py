#!/usr/bin/env python3
"""Frozen bench protocol (phase 15.0.3) — the ONE place every LJ journal computes
metrics and stamps a reproducibility passport, so numbers are comparable across
stages and traceable to exact code+data.

Split policy (FROZEN):
  * clean core (lab+gold5) grouped 5-fold, StratifiedGroupKFold(seed=0), grouping by `grp`.
  * gold is the LOAD-BEARING held-out test (R3): never used to fit anything you report on gold.
  * azimuth bench: mocap P3D re-projected at fixed ANGLES via mocap_to_views.project — TRUE 3D,
    not the circular y-rotation of MediaPipe z.
Reproducibility = fixed seed + fixed data content. `passport()` records the git commit,
working-tree cleanliness, and sha256 of each data file, embedded in every journal's results.

Import, don't fork: from bench import balanced_accuracy, grouped_cv, bootstrap_ci, passport, ANGLES
"""
import os, hashlib, subprocess, numpy as np

CLASSES = ["standing", "sitting", "lying"]
SEED = 0
N_SPLITS = 5
ANGLES = [0, 15, 30, 45, 60, 90]     # azimuth bench, frozen
N_BOOT = 1000                        # grouped bootstrap resamples


# ------------------------------- metrics -----------------------------------
def balanced_accuracy(y, pred, classes=CLASSES):
    """Mean per-class recall (immune to class imbalance) — the headline pose metric."""
    y, pred = np.asarray(y), np.asarray(pred)
    rec = [(pred[y == c] == c).mean() for c in classes if (y == c).any()]
    return float(np.mean(rec))


def per_class_recall(y, pred, classes=CLASSES):
    y, pred = np.asarray(y), np.asarray(pred)
    return {c: (float((pred[y == c] == c).mean()) if (y == c).any() else None) for c in classes}


def confusion(y, pred, classes=CLASSES):
    y, pred = np.asarray(y), np.asarray(pred)
    idx = {c: i for i, c in enumerate(classes)}
    M = np.zeros((len(classes), len(classes)), int)
    for t, p in zip(y, pred):
        if t in idx and p in idx:
            M[idx[t], idx[p]] += 1
    return M


# --------------------------- grouped evaluation ----------------------------
def grouped_cv(X, y, groups, fit_predict, n_splits=N_SPLITS, seed=SEED):
    """StratifiedGroupKFold OOF. fit_predict(Xtr,ytr,Xte)->pred_te. Returns (per_fold_bal, oof_pred)."""
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.asarray(y)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.empty(len(y), dtype=object)
    per_fold = []
    for tr, te in sgkf.split(X, y, groups=groups):
        pred = np.asarray(fit_predict(X[tr], y[tr], X[te]))
        oof[te] = pred
        per_fold.append(balanced_accuracy(y[te], pred))
    return per_fold, oof


def bootstrap_ci(y, pred, groups, metric=balanced_accuracy, n=N_BOOT, seed=SEED, alpha=0.05):
    """Grouped bootstrap CI (resample GROUPS w/ replacement — matches NB3b). Returns (point, lo, hi)."""
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups)
    uniq = np.unique(groups)
    by = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[g] for g in pick])
        stats.append(metric(y[idx], pred[idx]))
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(metric(y, pred)), float(lo), float(hi)


def azimuth_curve(P3D, y, fit_predict_train, feat_fn, angles=ANGLES):
    """Honest azimuth bench: model fit on training features, tested on mocap re-projected at each
    angle. fit_predict_train is a fitted callable f(feature_matrix)->pred. Returns {angle: bal_acc}."""
    from mocap_to_views import project
    out = {}
    for a in angles:
        F = np.array([feat_fn(project(p, a)) for p in P3D])
        out[a] = balanced_accuracy(y, fit_predict_train(F))
    return out


# ------------------------------- passport ----------------------------------
def _sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (chunk := f.read(1 << 20)):
            h.update(chunk)
    return h.hexdigest()[:16]


def passport(data_files=(), root=None):
    """Reproducibility stamp for a journal run: git commit + dirty flag + data sha256s.
    Embed the returned dict in results/metrics.json under the journal's key."""
    root = root or os.path.dirname(os.path.abspath(__file__))
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", root, *a], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "?"
    dirty = bool(git("status", "--porcelain"))
    data = {}
    for f in data_files:
        p = f if os.path.isabs(f) else os.path.join(root, f)
        data[os.path.basename(f)] = _sha256(p) if os.path.exists(p) else "MISSING"
    return {"code_commit": git("rev-parse", "--short", "HEAD"),
            "working_tree": "dirty" if dirty else "clean", "data_sha256": data}


if __name__ == "__main__":   # smoke: baseline balanced-acc + CI on the clean core
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from mocap_sanity_gate import features_13
    from sklearn.ensemble import RandomForestClassifier
    d = np.load(os.path.join(os.path.dirname(__file__), "data_probe", "landmarks_v2.npz"), allow_pickle=True)
    src = np.array([str(s) for s in d["src"]]); m = np.isin(src, ["lab", "gold5"])
    X, y, grp = d["X"][m], np.array([str(s) for s in d["y"]])[m], d["grp"][m]

    def fp(Xtr, ytr, Xte):
        F = lambda A: np.array([features_13(a) for a in A])
        return RandomForestClassifier(n_estimators=300, random_state=SEED).fit(F(Xtr), ytr).predict(F(Xte))

    per_fold, oof = grouped_cv(X, y, grp, fp)
    pt, lo, hi = bootstrap_ci(y, oof, grp)
    print(f"clean-core features_13 RF: grouped-CV balanced-acc = {np.mean(per_fold):.3f} "
          f"(bootstrap CI [{lo:.3f},{hi:.3f}])  per-fold std={np.std(per_fold):.3f}")
    print("per-class recall:", per_class_recall(y, oof))
    print("passport:", passport(["data_probe/landmarks_v2.npz", "data_probe/mocap_views.npz"]))
