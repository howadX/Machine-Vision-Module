# Checkpoint — 2026-07-16

Resume point. Branch: **`kaggle`**. Active phase: **15 — ML Refactor → v0.2.0**
(plan: `mvm/docs/phase-15-plan.md`; frozen bench: `research/BENCH-PROTOCOL.md`).

## ⭐ Where we are: 15.0 + 15.1 + 15.2 CLOSED and PUBLISHED; next = 15.3 (tracker)

| stage | status |
|---|---|
| 15.0 — lab foundation (CMU+OpenImages conveyors, frozen bench, LJ0) | ✅ closed |
| 15.1 — pose (5 experiments + integration + LJ1) | ✅ closed |
| 15.2 — transition kinematics (stabilizer + fall threshold, LJ2) | ✅ closed |
| 15.3 — tracker on true trajectories | ⬜ next |
| 15.4 — golden-scenario simulator (fusion/FSM/anomaly) | ⬜ |
| 15.5 — expert passport + diet (weight/speed) | ⬜ |
| 15.6 — Rust port + release v0.2.0 | ⬜ |

Every ML number runs through the frozen protocol (`spike_pose/bench.py`, SEED=0, grouped
bootstrap CI, data+code passport). Rust production is still **v0.1.0** — the port is 15.6.

## Published on Kaggle (private, account `howadx`)

- Notebooks (Run-All **COMPLETE**, numbers verified on Kaggle):
  `lj0-building-the-lab`, `lj1-the-pose-model-rebuilt-honestly`,
  `lj2-physics-of-posture-transitions`.
- Datasets (derived keypoints only, no images/BVH): `cmu-mocap-views`,
  `mvm-pose-landmarks` (+`landmarks_v2.npz`), `mvm-animal-ood`, `mvm-mocap-transitions`.
- Gotchas hit + baked in: dataset title ≤50 chars; kernel `id` must equal `slugify(title)`;
  Kaggle auto-runs on push so datasets must finish processing first; `_find` does a recursive
  `/kaggle/input` search (robust to dataset nesting).

## The lab (15.0) — reusable tools

- **`spike_pose/mocap_to_views.py`** — CMU BVH → FK → 13 GRIP joints → virtual eye-level camera
  (175 cm). Emits **true metric `P3D`** so azimuth is a REAL viewpoint. Subjects 111+113 span
  every posture → class ⊥ subject. Bend trials included; lying-gate needs low pelvis (deep bend =
  standing). BVH pulled per-subject from `una-dinosauria/cmu-mocap` (gitignored cache).
  `mocap_views.npz` = 4425 poses / 34 trials.
- **`mocap_sanity_gate.py`** (+`features_13` = **12 features**, 13 joints, z-free variant),
  **`bench.py`** (frozen metrics + passport), **`openimages_{relations,fetch}.py`** (relations
  inventory + S3 fetch; `--mid/--exclude` for OOD sourcing), **`research/BENCH-PROTOCOL.md`**.

## Pose (15.1) — decisions, all measured

- **Ship 2D-GRIP (z=0)**: MediaPipe z is a cross-domain liability (3D collapses 0.955 in-domain →
  0.62 on gold, sitting recall 0.01; 2D holds 0.83). Prod `mvm-posture` still uses z → real fix.
- **mocap is a TRAINING asset**, not just a bench: lab+mocap gold R3 **0.894** (vs lab-only 0.83);
  synthetic azimuth aug refuted (needs the untrustworthy z; old "0.90–0.95" was circular).
- **Flexion channel** = continuous `torso_flexion`, pure arithmetic (2D `torso_inc`, MAE ~6°
  frontal / 3.5° profile; regressor no better) → +0 model weight.
- **OOD: Mahalanobis replaces the 195 KB VAE** — on real animal-OOD 0.825 vs recon 0.702
  (Denouden blind spot). ~1.8 KB, pure arithmetic.
- **Held-out azimuth** (group-disjoint by trial): lab+mocap flat 3.0 pt, ~0.92 on unseen people.
- **Ship artifacts** (`spike_pose/ship/`, gitignored, regen via `lj1_build_ship.py`): forest
  **220 KB** + Mahalanobis 1.8 KB + flexion 0 KB. **Pose footprint 1068 KB → 222 KB (−79%).**
  Experiment scripts: `spike_pose/lj1_{2d_vs_3d,augmentation,flexion,ood,azimuth_heldout,build_ship}.py`.
- **Honest limits carried forward:** lying recall ~0.78 (scarce class); 2D-GRIP ties raw (not beats)
  cross-domain — GRIP's value is invariance + interpretability + $0-Rust.

## 15.2 done (LJ2) — decisions to port in 15.6

- **E1 stabilizer:** forward-filter prior ties hand-tuned dwell on aggregate flip-rate (H-prior
  NOT established — honest negative). Ships as a **hybrid**: dwell + one per-edge veto
  (`sl_dwell=12` on the `standing↔lying` edge, licensed by measured `A[standing,lying]≈0`) →
  drives the false standing↔lying flip to **0** under occlusion stress (Δ CI [−3.62,−0.43],
  established), no aggregate regression. That edge = the fall-relevant one.
- **E2 anomaly fall rule:** *reached floor (min pelvis ≤0.35 m)* AND *peak downward pelvis
  velocity ≳1.13 m/s* (AUROC 1.000, 6 falls vs 8 controlled; velocity alone insufficient).
  Caveat: CMU falls stunt-only → threshold bounds clinical from above.
- Tools: `spike_pose/lj2_{transitions,fall_velocity,transition_matrix,stabilizer}.py`; artifact
  `mocap_transitions.npz` (35 trials @60fps); design `research/LJ2-DESIGN.md`. Numbers in
  `results/metrics.json["lj2"]`, passport commit `e0514bed`.

## Next actions — 15.3 (tracker on true trajectories) → journal LJ3

- Offline bench from CMU projections: bbox trajectories with true identity → **reproduce walking
  fragmentation** (the live pain: up to 7 phantom tracks) as a regression test.
- Kalman process-noise from real human speed/accel distributions; re-link + Hungarian on paired
  sequences (crossings, occlusions). Gate: fragmentation ≤ agreed threshold; zero idle regressions.
- 15.2's `mocap_transitions.npz` already has walk trials (111_34/113_25/07_01/35_01 via
  `mocap_to_views`); may need a walk-focused pull + bbox synthesis from projected landmarks.

## Open debts (non-blocking)

- Kaggle assets are **private** → flip to public for portfolio when ready.
- 15.6 will port the 15.1 ship artifacts to Rust (bit-exact golden tests; posture output gains a
  `torso_flexion` field — contract change to document).
- Doc debt "0.959"/"12 КБ" fixed across skeletons + research maps (15.07).

## Key files
- Tools/experiments: `spike_pose/{mocap_to_views,mocap_sanity_gate,bench,openimages_*,lj1_*}.py`
  + `extract_images.py` (shared MPPose runner). `kctl.py` — Kaggle driver + DATASETS/NOTEBOOKS.
- Docs: `research/{BENCH-PROTOCOL,openimages-relations-inventory,DATA-EXPANSION-DESIGN}.md`,
  `mvm/docs/phase-15-plan.md`. Journals: `notebooks/{lj0,lj1}/`.
- Data (gitignored, regenerable): `data_probe/*.npz`, `mocap_bvh/`, `oi_images/`, `oi_meta/`,
  `ship/`. Gold raw images: `/home/dawoh/goldwork/{lyingv2,sitstand}/`.
