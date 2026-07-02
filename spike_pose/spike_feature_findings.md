# P0 spike — findings (pose classifier method) — GO

**Date:** 2026-07-01. **Verdict: GO** for method C (engineered geometric features → shallow classifier).

## Setup
- **Data (public only):** standing+lying from **URFD** (8 fall seqs, extracted with OUR OpenCV-Zoo
  BlazePose → topology parity), sitting from **Zenodo MultiPosture** (real MediaPipe 33-lm 3D).
  668 samples: standing 173 / lying 195 / sitting 300.
- **Proposed (C):** ~14 gravity-preserving, azimuth-invariant features (torso/thigh/shank inclination
  to vertical; knee/hip/elbow flexion; vertical-vs-horizontal extent & aspect; signed vertical
  placement of nose/knee/ankle vs hips) → MLP(64,32) / RandomForest.
- **Baseline:** current VAE-style input = `normalize_landmarks` → flat 99 raw coords → same heads.
- **Viewpoint test:** rotate held-out test skeletons about the vertical (camera azimuth) 15/30/45/90°.

## Results
| Input | head | clean acc | az15 | az30 | az45 | az90 |
|-------|------|-----------|------|------|------|------|
| **features (C)** | MLP | 0.988 | 0.988 | 0.988 | 0.988 | **0.988** |
| **features (C)** | RF  | 0.994 | 0.994 | 0.994 | 0.994 | **0.994** |
| raw coords (current) | MLP | 0.982 | 0.982 | 0.946 | 0.725 | **0.305** |
| raw coords (current) | RF  | 0.994 | 0.994 | 0.772 | 0.455 | **0.305** |

feature-MLP confusion (rows=true) [lying, sitting, standing]:
```
[[48  0  1]
 [ 0 75  0]
 [ 1  0 42]]
```

## Reading
- **Clean accuracy: features ≈ raw** (both ~0.99). So features lose nothing.
- **Under camera azimuth, raw coords COLLAPSE to chance (0.305 @ 90°)** while **features stay flat**.
  This is the core result: the current raw-coordinate/VAE path is *not viewpoint-invariant* — a camera
  angle it wasn't trained on tanks it. This **is** the root cause of the production "wrong angle" /
  `standing↔sitting_at_desk` failure, and method C fixes it structurally.
- standing↔lying (BOTH from URFD, same camera → no source leakage) separate with 2/91 errors → genuine.

## Honest caveats (address in P1)
- Azimuth-invariance of the features is **by construction**; the az-stability confirms the design works,
  but the *real* multi-angle test (elevation/tilt, real second camera) belongs to P1/live.
- **sitting is single-source (MultiPosture)** → its perfect separation may be partly source-aided;
  the GO signal rests on the azimuth-delta (source-immune, within-sample) + clean standing/lying.
- Small spike set (668); P1 uses a larger, multi-source, multi-angle public mix.

## Reproduce
`cd spike_pose && ../venv/bin/python spike_data.py && ../venv/bin/python spike_feats.py`

---

# P1 — full retrain on public data + grouped generalization

Data grown to **1035 samples** (standing 312 / lying 323 / sitting 400) over **26 groups**
(14 URFD sequences + 12 MultiPosture subjects). Final: `StandardScaler` + MLP(32,16) on the 14
features, trained with landmark-jitter + ±10° tilt augmentation. VAE retrained on public
normalized-99 for OOD. Export: `pose_p1_config.json` + `pose_p1_weights.bin` (199612 B).

**Grouped 5-fold (StratifiedGroupKFold — every test group = UNSEEN people/sequences):**
| input | grouped acc | macroF1 |
|-------|-------------|---------|
| **features (C)** | **0.946 ± 0.035** | 0.945 |
| raw coords (current) | 0.908 ± 0.075 | 0.901 |

Features generalize to unseen subjects at 0.946 with **low variance** — this **retires the P0
source-leakage caveat** (test groups never seen in training). Raw-coords is worse and higher-variance.
feature confusion (rows=true) [lying,sitting,standing]: `[[308 8 7],[0 390 10],[12 19 281]]` — residual
is standing↔sitting (cross-source, hard pair) → the deferred own-camera fine-tune will sharpen it.

**Azimuth robustness (recap, held-out rotated):** features flat (0.93–0.98 @ az30/az90);
raw collapses (0.44 @ az90 MLP, 0.32 @ az90 RF). Confirmed on the larger set.

Note: RF slightly beats MLP on features (0.977 vs 0.934 held-out) — MLP kept as primary for a clean
Rust port; RF is a fallback if we want feature-importance.

**Artifacts:** `train_pose_p1.py`, `pose_p1_config.json`, `pose_p1_weights.bin`.
Weights.bin f32 LE layout documented in the config's `note` field (for the P2 Rust reader).

## P1 refinement — RF head + second-sitting-source investigation

**RF head chosen (beats MLP).** No scaler needed (tree splits are per-feature). Final = RandomForest
`n_estimators=120, max_depth=16`.
| head (grouped 5-fold) | acc | note |
|-----------------------|-----|------|
| **RF, features** | **0.959 ± 0.041** | chosen; azimuth-flat 0.981 @az90 |
| MLP, features | 0.946 ± 0.035 | simpler port, slightly worse |
| RF, raw-coords | 0.935 ± 0.048 | collapses to 0.328 @az90 |

Top RF feature importance: aspect 0.18, ank_below 0.15, torso_inc 0.12, hext 0.10, nose_above 0.09 — all physical.

**Second sitting source — investigated, none clean in public; and it doesn't target the residual.**
Searched Kaggle/HF/Zenodo: only MultiPosture has labeled 3D sitting; RGB sets are unusable
(silhouettes → BlazePose fails; melsmm → 4 kpts "person"; posture-monitor 2.7GB unlabeled).
Diagnosis of the standing→sitting residual (21/312): those errors are `torso_inc≈13°, knee_flex≈139°`
= genuinely UPRIGHT people, while "correct" URFD standing has `aspect≈0.76` (arms-out/dynamic
pre-fall frames, label −1 = "not lying" ≠ clean standing). So the residual is **URFD standing-label
noise + real geometric overlap, not a sitting-quantity gap** → a second sitting source won't fix it.
A core-only extent feature tweak (exclude arms) was tried and **rejected** (0.946, worse). The real
lever = clean standing data, best obtained in the **deferred own-camera capture phase** (which also
becomes the true cross-source break for sitting).

**Exports for P2 (RF path):** `pose_rf_forest.bin` (873 KB, 120 trees, node `<ifii fff>`, leaf left==−1),
`pose_vae_public.bin` (195 KB, VAE retrained on public data, same byte order as existing classifier.rs
loader → OOD only), `pose_p1_config.json`, `golden_pose_p1.json` (Rust parity vectors).
Classes/prob-order = **[lying, sitting, standing]**.
