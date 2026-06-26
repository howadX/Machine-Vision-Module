# Phase 7.5.0 — Spike findings (model de-risk)

> **Date:** 2026-06-26 · **Verdict:** 🟢 **GATE PASSED** — the one real unknown
> (BlazePose topology/preproc match + ort loadability) is killed. 7.5 can proceed.

## What was tested

The spike (`spike_pose/`, `mvm/crates/mvm-detect/examples/spike_pose_ort.rs`) proves
the two OpenCV Zoo BlazePose ONNX models are a drop-in native replacement for the
Python MediaPipe `.task` pose chain, reusing the existing PoseVAE/classifier weights.

### Models acquired (G1 ✅)
- `models/person_detection_mediapipe_2023mar.onnx` (12 MB) — input `[1,3,224,224]` NCHW,
  outputs scores `[1,2254,1]` + boxes `[1,2254,12]` (4 bbox + 4 keypoints; row[4:12] =
  mid-hip + full-body points that drive the pose ROI).
- `pose_estimation_mediapipe_2023mar.onnx` (5.5 MB) — input `[1,256,256,3]` NHWC,
  output `Identity [1,195]` = **39 landmarks × 5** (x,y,z,vis,presence; first 33 = body),
  `Identity_4 [1,117]` = 39 world landmarks, + seg mask + heatmaps.

## Gate 1 — Rust `ort` loadability ✅
`cargo run -p mvm-detect --example spike_pose_ort --release`: both models load + run a
forward pass in **ort 2.0.0-rc.12** (the same runtime as YuNet/SFace); pose Identity
output = 195 values as expected. No new runtime, no version friction.

## Gate 2 — Topology match (the main unknown) ✅
`spike_pose/spike_topology_gate.py`: OpenCV Zoo ONNX landmarks fed into the **Python
PoseVAE** (`vposer_coco33.pth`) vs the bridge's native MediaPipe `.task` on the same image.

| image | ZOO class (conf, recon) | MP class (conf, recon) | xy_diff |
|-------|-------------------------|------------------------|---------|
| p_V_01 | sitting_at_desk (.78, .0035) | sitting_at_desk (.66, .0030) | 0.013 ✓ |
| p_V_03 | standing (.85, .0099) | standing (.71, .0203) | 0.025 ✓ |
| p_V_05 | standing (.84, .0085) | sitting_at_desk (.71, .0109) | 0.027 ✗ |
| p_V    | arms_crossed (.99, .0309) | sitting (.67, .0728) | 0.245 ✗ |

**Why this passes:**
- **recon_error is in the same low band** (0.003–0.03) for both paths — often *lower*
  for ZOO. A wrong 33-landmark order would blow recon_error up 10–100×. Definitive.
- **xy_diff = 1–3%** on clean full-body frames (p_V_01/03/05) — landmarks are
  numerically the same → identical order *and* coordinate convention
  (MediaPipe-normalized: `x/w, y/h, z/w`, sigmoid on vis/presence).
- Class disagreements are **not topology**: p_V_05 has xy_diff 0.027 (landmarks agree) —
  a standing↔sitting_at_desk classifier flip on adjacent classes. p_V (640×480) has
  xy_diff 0.245 = genuinely different ROI selection on a small image.

EN labels (G7 ✅) used directly from `class_names`:
`['standing','sitting','lying_back','lying_side','arms_crossed','sitting_at_desk']`.

## Preprocessing nailed (for the 7.5.2 Rust port)
From the OpenCV Zoo reference demos (`spike_pose/mp_pose.py`, `mp_persondet.py`):
- **Pose ROI:** from person row keypoints `[4:12]` → mid-hip + full-body point →
  `full_dist = ‖mid_hip − full_body‖`, square box `mid_hip ± full_dist`, ×1.0 pre-enlarge.
- **Rotation:** align hip→body to vertical (`π/2 − atan2(...)`), warpAffine, resize to
  256² INTER_AREA, BGR→RGB, `/255`. NHWC.
- **Landmark recovery:** `Identity` → reshape 39×5; sigmoid cols 3:; unscale by
  `roi_wh/256`, de-rotate, add ROI center + pad_bias → original-image pixels.
- **z scaling:** `z *= max(scale_factor)`; z relative to hip.

## Ceiling-angle ROI — investigated and CLEARED ✅
Initial scare: the person detector returned **persons=0 on `/tmp/mvm_frame.jpg`**. We then
captured **live ceiling-cam frames with a full-body person** (`spike_pose/ceiling_capture_probe.py`
→ `spike_pose/ceiling_test/frame_*.jpg`, mirrors the bridge camera: CAP_FFMPEG + ROTATE_180):

- **Detection works at the ceiling angle:** persons=1 on all 10 frames, score 0.61–0.91.
  The `/tmp/mvm_frame.jpg` miss was simply a frame with **no full-body person**, not an
  angle problem.
- **Full pose pipeline on real frames** (`spike_topology_gate.py ceiling_test/frame_*.jpg`):
  class agreement **7/9**, **xy_diff 1.0–2.6%** (tighter than the reference photos),
  recon_error in the same band (0.014–0.031). The 2 class flips (standing↔sitting) are on
  transitional poses where MP itself is unsure (conf 0.32–0.56) and the landmarks still
  agree (xy_diff ~0.01) — classifier boundary, not topology.

**Resolution — Open decision #1 CLOSED:** use **`person_detection_mediapipe`** as the ROI
source for 7.5.2; it is validated on the actual deployment camera. The YuNet-face-bbox→body
ROI remains a noted fallback only (not needed). **Always validate the person detector with a
genuine full-body subject** — a partial/no-body frame trivially reads as persons=0.

## Status of 7.5 gates after spike
| Gate | Status |
|------|--------|
| G1 landmark models (ONNX, ort-loadable) | ✅ acquired + load + run in ort |
| G2 VAE/classifier weights to port | ✅ on disk, recon validated against ZOO landmarks |
| G7 EN labels | ✅ from `class_names` |
| Topology / preproc match | ✅ recon + xy_diff prove it |
| **ROI for ceiling angle** | ✅ validated on live full-body frames (persons 0.61–0.91; pose xy_diff 1–2.6%) → use `person_detection_mediapipe` |
| G6 pose accuracy oracle | ✅ built (7.5.5) — see below |

## 7.5.5 — pose accuracy oracle (G6)

Reproducible quantitative check of the native Rust pose vs the Python reference, on the
7 committed `mvm/models/references/p_V_*.jpg`:
- `spike_pose/golden_refs.py` → `golden_refs.json` (Python pipeline labels).
- `examples/pose_labels.rs` → native NativePose labels.
- Baked into the regression test `pose_class_oracle_vs_python` (gate: all 7
  detected+classified, class agreement ≥ 5/7).

**Result: class agreement native↔Python = 5/7 (71%).** Both disagreements are
`standing` (Python) → `sitting_at_desk` (Rust). `standing` is the weak class at the
ceiling angle: the legs are foreshortened from above and a residual landmark-z gap
nudges borderline standing poses to the sitting side. Sitting variants and
arms_crossed agree well; live runs classify clear poses correctly.

**Quality polish (this pass):** the pose ROI sampler now K×K-supersamples each output
pixel's footprint (≈ cv2 INTER_AREA, which the reference uses for the ~4.5× downscale)
instead of a single bilinear tap. Measured on p_V_03: landmark-z mean abs diff vs
Python **0.099 → 0.067 (−33%)**, max 0.30 → 0.22 (x,y already ~1.3%). Supersampling the
*detector* preprocess too was tried and reverted — no net class-agreement gain and it
destabilised a borderline standing case, at 16× the detector sampling cost.

**Remaining gap (future):** closing the last standing-class disagreements needs matching
cv2's exact letterbox/INTER_AREA in the detector (→ tighter ROI scale) or accepting the
ceiling-angle ambiguity. The Rust person-detector is also ~6× less sensitive than Python
on borderline frames.

## Artifacts
- `spike_pose/` — `probe_persondet.py`, `spike_topology_gate.py`, reference demos.
- `mvm/crates/mvm-detect/examples/spike_pose_ort.rs` — ort load/run smoke test.
- Models in `models/` ready for 7.5.1 bundle step.
