# Checkpoint — 2026-07-15

Resume point. Branch: **`kaggle`**. Active phase: **15 — ML Refactor → v0.2.0**
(plan: `mvm/docs/phase-15-plan.md`; frozen bench: `research/BENCH-PROTOCOL.md`).

## ⭐ Where we are: 15.0 "the lab" is BUILT (in draft); next = 15.1 (pose, load-bearing)

Stage 15.0 closed by code across 4 commits (`f243f2a6` mocap+bench, `f3b4a473` OpenImages,
`fb43a1c0` LJ0). Every ML number from here runs through the frozen protocol.

| 15.0 sub-stage | status |
|---|---|
| 15.0.1 CMU pipeline + sanity gate | ✅ gate PASS |
| 15.0.2 OpenImages conveyor (relations + pose-crops) | ✅ |
| 15.0.3 bench protocol (metrics + passport, frozen) | ✅ |
| 15.0.4 publish Kaggle datasets | 🟡 prepared, deferred (needs owner publish-go) |
| LJ0 "Building the lab" | ✅ draft, Run-All verified locally |

## What 15.0 built (the lab)

- **`spike_pose/mocap_to_views.py`** — CMU BVH → FK → 13 GRIP joints → virtual eye-level
  camera (175 cm). Emits **true metric `P3D`** so azimuth is a REAL viewpoint (not the circular
  y-rotation of MediaPipe z). Subjects 111+113 span every posture → class ⊥ subject. BVH pulled
  per-subject from `una-dinosauria/cmu-mocap` (gitignored cache). Artifact
  `data_probe/mocap_views.npz` = 3111 poses / 26 trials.
- **`spike_pose/mocap_sanity_gate.py`** (+`features_13`: vext/hext over the 13 GRIP joints).
- **`spike_pose/bench.py`** — frozen metrics (balanced-acc, grouped bootstrap CI, grouped-CV,
  honest azimuth curve) + reproducibility passport (code commit + data sha256). `SEED=0`.
- **`spike_pose/openimages_{relations,fetch}.py`** — relations inventory
  (`research/openimages-relations-inventory.md`, 1466 triplets → FSM-audit map) + clean
  single-person S3 fetch → same MPPose extractor.
- **`research/BENCH-PROTOCOL.md`** — frozen split policy, metric registry, LJ journal template.
- **`notebooks/lj0/`** — first Lab Journal, self-contained, Run-All.

## Key measured results (carried into 15.1)

- **Gate PASS on geometry:** angle features (torso_inc, knee/hip flex) in-distribution across
  all classes; sitting+lying 6/6.
- **⭐ Occlusion gap MEASURED (not a bug):** standing extent features OUT (`vext` 3.4 vs 2.6) —
  mocap = un-occluded truth, real lab/gold captures **truncate legs** (ankle invisible on ~7%
  standing / 22% lying). → **the joint-dropout target for 15.1**.
- **⭐ Azimuth invariance shown NON-circularly:** on true-3D re-projection the curve is FLAT
  (az0 0.630 → az90 0.669, +3.9 pt). NB3b's old azimuth test was circular by construction.
- **Frozen baseline:** clean-core (lab+gold5, 2633 poses) `features_13` RF = **0.955
  [0.933, 0.971]**, per-class recall .93/.96/.95. This is what 15.1 must beat.

## Prior pose findings still in force (from NB3b / spikes)

- **z-channel is a cross-domain liability** (train-lab→test-gold 3D-GRIP collapsed 0.93→0.66;
  2D-GRIP z=0 held 0.83 and beat raw) → **ship 2D-GRIP (z-free)**; prod `mvm-posture` still uses z.
- **Confound (class↔source), circular azimuth** — the two reasons the lab exists (LJ0 §1).
- Naive Y-heuristic weak-labeler is broken → in-the-wild OpenImages needs a **consensus gate**
  (E3 eval lives in 15.1); pose-conf is a useless quality filter.

## Next actions — 15.1 (pose, Track P, load-bearing) → journal LJ1

1. **2D-GRIP (z=0), `features_13`** as the shipped representation; view-augmentation (mocap-true
   azimuth vs z-synthesis — compare); structured joint-dropout matching the measured desk-occlusion
   (7–22% ankle truncation).
2. **First honest accuracy-vs-azimuth curve on the *gold* held-out test (R3)** — the load-bearing,
   confound-free number; mocap gives the true-3D azimuth bench.
3. **Flexion channel, not a 4th class:** output {standing/sitting/lying} + continuous `torso_flexion`
   (REBA bins downstream); ground-truth flexion angle from CMU.
4. **OOD:** class-conditional Mahalanobis on 14 features vs the 195 KB VAE vs vote-entropy, on a real
   OOD set (junk detections, animals, degraded landmarks).
5. **Gate:** gold-acc ≥ baseline 0.955 (CI), azimuth curve flat (<5 pt spread), Mahalanobis compared,
   final forest serialized. Then write **LJ1** + the NB3b research-integrity addendum (circular azimuth).

## Deferred: Kaggle publish (15.0.4)

Owner-gated (outward-facing). To finalize 15.0 on Kaggle: publish **`cmu-mocap-views`** (manifest
ready in `kctl.py`, derived keypoints only, license CC-BY-4.0 vs Other/CMU verbatim — owner to
confirm) **and** an updated **`landmarks_v2`** (LJ0 needs the `src` field for lab+gold5; current
public `mvm-pose-landmarks` is the old 1035-probe) → `kctl push lj0 --yes` → Run-All → clean passport.
Plan: batch this with LJ1's publish (fewer Kaggle versions).

## Key files

- Tools: `spike_pose/{mocap_to_views,mocap_sanity_gate,bench,openimages_relations,openimages_fetch}.py`
  + `extract_images.py` (shared MPPose runner). `kctl.py` — Kaggle driver + DATASETS/NOTEBOOKS manifest.
- Docs: `research/BENCH-PROTOCOL.md`, `research/openimages-relations-inventory.md`,
  `research/DATA-EXPANSION-DESIGN.md` (R3+R1 framing), `mvm/docs/phase-15-plan.md`.
- Data (gitignored, regenerable): `data_probe/{mocap_views,landmarks_v2}.npz`, `mocap_bvh/`,
  `oi_images/`, `oi_meta/`. Gold raw images: `/home/dawoh/goldwork/{lyingv2,sitstand}/`.
