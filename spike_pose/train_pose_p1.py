"""P1 — full retrain on PUBLIC data (method C, RF head) + grouped eval + export for the Rust port.

Pipeline (final): 33x4 landmarks -> compute_features (14 azimuth-invariant, gravity-preserving) ->
RandomForest (soft-vote) -> {standing,sitting,lying}.  NO scaler (tree splits are per-feature).
VAE retrained on PUBLIC normalized-99 -> OOD/plausibility only (Tier 0.2), exported in the EXACT
byte order the existing mvm-posture/src/classifier.rs VAE loader expects.

Exports (into spike_pose/, consumed by P2):
  pose_rf_forest.bin   - forest: u32 n_trees, per tree {u32 n_nodes, nodes[...]}, node=<ifii fff>
                         (feature i32 / threshold f32 / left i32 / right i32 / prob[3] f32); leaf: left=-1
  pose_vae_public.bin  - VAE weights, same 18-tensor order as export_pose_weights.py (VAE part)
  pose_p1_config.json  - class_names (=forest prob order), feature_names, n_trees, vae dims
  golden_pose_p1.json  - parity vectors for Rust tests (features, forest probs, vae recon/logpz)

Run from spike_pose/ with the project venv, after spike_data.py.
"""
import os, sys, json, struct, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spike_feats import features, azimuth_rotate, raw_coords
from core.pose_encoder import normalize_landmarks
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import torch
from core.pose_vae import PoseVAE

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NPZ = os.path.join(HERE, "data_probe", "landmarks.npz")
FEATURE_NAMES = ["torso_inc", "thigh_inc", "thigh_asym", "shank_inc", "knee_flex", "knee_asym",
                 "hip_flex", "elbow_flex", "vext", "hext", "aspect", "nose_above", "ank_below", "kn_below"]
RF_KW = dict(n_estimators=120, max_depth=16, random_state=0)
VAE_ORDER = [  # VAE-only slice of export_pose_weights.py ORDER (must match classifier.rs loader)
    "encoder.0.weight", "encoder.0.bias", "encoder.0.running_mean", "encoder.0.running_var",
    "encoder.1.weight", "encoder.1.bias", "encoder.4.weight", "encoder.4.bias",
    "fc_mu.weight", "fc_mu.bias", "fc_logvar.weight", "fc_logvar.bias",
    "decoder.0.weight", "decoder.0.bias", "decoder.3.weight", "decoder.3.bias",
    "decoder.6.weight", "decoder.6.bias",
]


def build(X_raw, fn):
    return np.array([fn(x) for x in X_raw], np.float32)


def tilt_x(lm, deg):
    t = np.deg2rad(deg); c, s = np.cos(t), np.sin(t)
    out = lm.copy(); y, z = lm[:, 1], lm[:, 2]
    out[:, 1] = y * c - z * s; out[:, 2] = y * s + z * c
    return out


def augment(X_raw, y, grp, rng, k=4):
    """jitter + minor pitch; features are azimuth-invariant so azimuth aug is a no-op by design."""
    Xa, ya, ga = list(X_raw), list(y), list(grp)
    for x, lab, g in zip(X_raw, y, grp):
        for _ in range(k):
            j = x.copy()
            j[:, :3] += rng.normal(0, 0.01, (33, 3)).astype(np.float32)
            j = tilt_x(j, rng.uniform(-10, 10))
            Xa.append(j); ya.append(lab); ga.append(g)
    return np.array(Xa, np.float32), np.array(ya), np.array(ga)


def grouped_eval(name, X_raw, y, grp, feat_fn, head):
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
    accs, f1s = [], []; labels = sorted(set(y)); cm = np.zeros((len(labels),) * 2, int)
    for tr, te in sgkf.split(X_raw, y, grp):
        Xtr, ytr, _ = augment(X_raw[tr], y[tr], grp[tr], np.random.default_rng(0))
        m = head().fit(build(Xtr, feat_fn), ytr)
        pred = m.predict(build(X_raw[te], feat_fn))
        accs.append(accuracy_score(y[te], pred)); f1s.append(f1_score(y[te], pred, average="macro"))
        cm += confusion_matrix(y[te], pred, labels=labels)
    print(f"  [{name}] grouped 5-fold acc={np.mean(accs):.3f}±{np.std(accs):.3f} "
          f"macroF1={np.mean(f1s):.3f} folds={[round(a,3) for a in accs]}")
    print(f"    confusion (rows=true) {labels}:\n" + "\n".join("     " + str(r) for r in cm))


def train_vae(norm99, epochs=250):
    torch.manual_seed(0)  # reproducible weights → .bin, golden, and OOD threshold stay in sync
    dev = torch.device("cpu"); vae = PoseVAE(input_dim=99, latent_dim=32).to(dev)
    opt = torch.optim.Adam(vae.parameters(), lr=1e-3)
    X = torch.tensor(norm99, dtype=torch.float32, device=dev); vae.train()
    for _ in range(epochs):
        opt.zero_grad(); recon, mu, logvar = vae(X)
        rl = torch.nn.functional.mse_loss(recon, X)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        (rl + 1e-3 * kl).backward(); opt.step()
    vae.eval()
    with torch.no_grad():
        lp = vae.log_prob_z(X).cpu().numpy()
    import numpy as _np
    pct = {p: round(float(_np.percentile(lp, p)), 1) for p in (50, 90, 95, 99)}
    print(f"  VAE retrained: recon={rl.item():.4f} kl={kl.item():.3f} | in-dist log_prob_z pct={pct} max={lp.max():.1f}")
    return vae


def export_forest(rf, path):
    buf = bytearray(); buf += struct.pack("<I", len(rf.estimators_))
    for est in rf.estimators_:
        t = est.tree_; n = t.node_count; buf += struct.pack("<I", n)
        val = t.value.reshape(n, -1); val = val / val.sum(axis=1, keepdims=True)
        for i in range(n):
            buf += struct.pack("<ifii", int(t.feature[i]), float(t.threshold[i]),
                               int(t.children_left[i]), int(t.children_right[i]))
            buf += struct.pack("<3f", *[float(v) for v in val[i]])
    open(path, "wb").write(bytes(buf)); return len(buf)


def export_vae(vae, path):
    sd = vae.state_dict(); buf = bytearray()
    for k in VAE_ORDER:
        buf += sd[k].cpu().numpy().astype(np.float32).ravel().tobytes()
    open(path, "wb").write(bytes(buf)); return len(buf)


def golden(rf, vae, X_raw, y):
    dev = torch.device("cpu"); out = []
    idxs = [0, len(y) // 3, 2 * len(y) // 3, len(y) - 1]
    for i in idxs:
        lm = X_raw[i]; feat = features(lm)
        probs = rf.predict_proba(feat.reshape(1, -1))[0]
        norm99 = normalize_landmarks(lm)[0].reshape(-1)
        with torch.no_grad():
            x = torch.tensor(norm99, dtype=torch.float32, device=dev).unsqueeze(0)
            z = vae.encode_to_z(x); recon = vae.decode(z)
            mse = torch.nn.functional.mse_loss(recon, x).item()
            logpz = vae.log_prob_z(x).item()
        out.append({
            "input_33x4": lm.tolist(),
            "features": [round(float(v), 6) for v in feat],
            "forest_probs": [round(float(v), 6) for v in probs],
            "pred": list(rf.classes_)[int(np.argmax(probs))], "true": str(y[i]),
            "vae_recon_mse": round(float(mse), 6), "vae_logpz": round(float(logpz), 4),
        })
    json.dump(out, open(os.path.join(HERE, "golden_pose_p1.json"), "w"), indent=1)
    # Rust-includable literal: &[( [[f32;4];33], [f32;14], [f32;3], &str, f32, f32 )]
    def arr(v, f="{:.8}"): return "[" + ",".join(f.format(float(x)) for x in v) + "]"
    recs = []
    for g in out:
        lm_lit = "[" + ",".join(arr(row) for row in g["input_33x4"]) + "]"
        recs.append(f'({lm_lit},{arr(g["features"])},{arr(g["forest_probs"])},'
                    f'"{g["pred"]}",{g["vae_recon_mse"]:.8}f32,{g["vae_logpz"]:.6}f32)')
    open(os.path.join(HERE, "pose_p1_golden.in"), "w").write("&[\n" + ",\n".join(recs) + "\n]\n")


if __name__ == "__main__":
    d = np.load(NPZ, allow_pickle=True); X_raw, y, grp = d["X"], d["y"], d["grp"]
    import collections
    print("dataset:", dict(collections.Counter(y)), "| groups:", len(set(grp)))

    print("\n== grouped generalization (unseen people/sequences) ==")
    grouped_eval("features/RF ", X_raw, y, grp, features, lambda: RandomForestClassifier(**RF_KW))
    grouped_eval("raw/RF      ", X_raw, y, grp, raw_coords, lambda: RandomForestClassifier(**RF_KW))

    print("\n== azimuth robustness (held-out rotated), features vs raw ==")
    idx = np.arange(len(y)); tr, te = train_test_split(idx, test_size=0.25, stratify=y, random_state=0)
    for nm, fn in [("features", features), ("raw", raw_coords)]:
        m = RandomForestClassifier(**RF_KW).fit(build(X_raw[tr], fn), y[tr])
        clean = accuracy_score(y[te], m.predict(build(X_raw[te], fn)))
        rot = {r: accuracy_score(y[te], m.predict(build(np.array([azimuth_rotate(x, r) for x in X_raw[te]]), fn)))
               for r in (30, 90)}
        print(f"  [{nm:8}] clean={clean:.3f} az30={rot[30]:.3f} az90={rot[90]:.3f}")

    print("\n== final RF (all data, augmented) + VAE-OOD + export ==")
    Xa, ya, _ = augment(X_raw, y, grp, np.random.default_rng(0))
    rf = RandomForestClassifier(**RF_KW).fit(build(Xa, features), ya)
    print(f"  final RF classes={list(rf.classes_)} train acc={accuracy_score(ya, rf.predict(build(Xa, features))):.3f}")
    vae = train_vae(build(X_raw, lambda x: normalize_landmarks(x)[0].reshape(-1)))
    nf = export_forest(rf, os.path.join(HERE, "pose_rf_forest.bin"))
    nv = export_vae(vae, os.path.join(HERE, "pose_vae_public.bin"))
    json.dump({"class_names": list(rf.classes_), "feature_names": FEATURE_NAMES,
               "n_features": len(FEATURE_NAMES), "n_trees": len(rf.estimators_),
               "rf_kw": {k: v for k, v in RF_KW.items() if k != "random_state"},
               "vae": {"input_dim": 99, "latent_dim": 32}, "vae_order": VAE_ORDER,
               "forest_node_fmt": "<ifii fff = feature,threshold,left,right,prob[3]; leaf: left==-1"},
              open(os.path.join(HERE, "pose_p1_config.json"), "w"), indent=2)
    golden(rf, vae, X_raw, y)
    print(f"  exported forest={nf}B vae={nv}B + pose_p1_config.json + golden_pose_p1.json")
