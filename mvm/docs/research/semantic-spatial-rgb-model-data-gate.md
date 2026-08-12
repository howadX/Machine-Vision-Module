# RGB semantic-spatial context — evidence for the next MVM model/data gate

> **Status:** research memo, not an implementation commitment.
> **Scope:** local, RGB-only semantic measurements; no new Rust runtime authority.
> **Decision requested:** whether to run the bounded offline Experiment 1 below after a real indoor/MVM corpus is available.

## Executive summary

The SegFormer-B0/ADE20K rejection is a rejection of a particular generic-label reducer and reference-domain recipe, not of indoor semantic segmentation as a capability. SegFormer itself is a semantic-segmentation architecture, while the released NVIDIA source is non-commercial and the public HF checkpoint is explicitly tagged `license: other`; it is therefore a **control only**, not a product-default dependency.[1][2][3]

For MVM V1, ordinary **dense semantic segmentation** is the only task that directly answers the question “which *visible pixels* look like floor/wall/ceiling/door-or-window candidate?”. A monocular-depth or normal predictor can be a later auxiliary measurement, but cannot turn a semantic mask into occupancy, free space, support, an open doorway, or passability: a single view lacks visibility behind occluders and does not establish metric clearance or agent-body constraints. Self-supervised monocular depth also depends on camera-motion/image-formation assumptions and reports a depth estimate, not an occupancy claim.[20]

**Recommendation:** do not change the MVM production pipeline. First build a scene/session-isolated *real indoor* corpus and evaluator; then run an offline three-arm comparison: (A) frozen current SegFormer control, (B) LR-ASPP MobileNetV3-Large as the small permissively licensed control, and (C) PP-LiteSeg-T/STDC1 as the compact challenger. PIDNet-S is a conditional fourth arm only if opening boundary error remains the measured bottleneck. Calibrate `withheld` on a calibration split, pin raw predictions/GT/hashes, and independently score an untouched MVM-camera holdout. A candidate that passes only the offline gate may be proposed later as **default-off shadow measurement**, never as a canonical surface/free-space source.

## 1. What RGB can measure — and what it must not assert

| Visual task | RGB output that is defensible | Role for MVM V1 | Not established by this output |
|---|---|---|---|
| **Semantic segmentation** | Per-pixel class probability/logit: floor, wall, ceiling, door/window candidate, known non-target content | **Primary task.** Map source labels to the frozen V1 taxonomy and preserve probabilities before abstention | A labelled floor is not clear floor; a door/window label is not an opening, open state, topology, or traversable portal |
| **Panoptic/instance segmentation** | Object/thing masks plus stuff masks; useful later to distinguish furniture/doors/windows from amorphous stuff | Diagnostic/hard-negative tooling only; not needed in first model gate | Empty volume, support, or a collision-free path |
| **Monocular relative depth / normals** | Image-conditioned relative depth discontinuities or local orientation hypotheses | Future corroborating expert/evaluator; **not** a V1 input requirement | Reliable metric scale, free space behind an edge, a horizontal support plane, occupancy, clearance |
| **Temporal consistency / video segmentation** | Association/smoothing of repeated visible evidence; flicker and persistence metrics | Evaluate after the single-frame expert; use only to decide when to retain/withhold a measurement | New evidence for an occluded or unobserved region; a temporal prior must not fill unknown as free |
| **Geometry with calibrated RGB-D/pose** | A different, multi-sensor evidence path for visible depth/pose-backed surfaces | Useful only as offline GT or a later separately gated expert | Permission to back-project a raw RGB class into a safe navigation fact |

This separation is important because closed-set semantic models are known to be ill-equipped for previously unseen objects; an explicit reject/abstention policy is therefore more honest than treating every low-probability pixel as negative.[21] Temperature scaling is a cheap calibration baseline, but its published evidence is classification calibration rather than an automatic proof for dense segmentation; MVM must evaluate it per class and per pixel on its own calibration data.[18] Selective classification formalizes the risk–coverage trade-off, which is the correct reporting shape for `withheld` rather than silently dropping difficult pixels.[19]

### V1 label policy

Train/evaluate five supervised semantic outputs: `floor`, `wall`, `ceiling`, `opening_candidate`, `other_non_surface`. `withheld` is a **runtime abstention state**, not a negative or a sixth GT class. Map native `door` and `window` separately in the mapper and then collapse only their approved visible-semantic mass to `opening_candidate`; do not create labels such as `open`, `passable`, `free`, `support`, or `occupied`. The ScanNet project’s NYU40 palette explicitly contains wall, floor, door and window labels, demonstrating that this mapping is a label-space reduction—not evidence of opening state.[13]

## 2. Mature model / family shortlist

“Permissive code” does not prove permissive pretrained weights or source-data rights. The **code and checkpoint columns are deliberately separate**; any blank/unclear weight licence is a procurement blocker, not an implied permission.

| Family / task | Primary source and licence status checked | Size / CPU–ONNX practicality | MVM value and limitation | Decision |
|---|---|---|---|---|
| **SegFormer-B0 ADE20K** — semantic segmentation | Paper: transformer encoder + lightweight decoder.[1] NVIDIA reference code is **NVIDIA Source Code License, non-commercial/research/evaluation only**.[2] HF `nvidia/segformer-b0-finetuned-ade-512-512` has task `image-segmentation`, ADE labels, and `license: other`.[3] | Existing MVM evidence already proves its particular ONNX scheduling/parity recipe, not quality. | Necessary **historical control** to reproduce the measured reject on the new corpus; broad ADE taxonomy/reducer is not the V1 answer. | Control only; no new product dependency. |
| **LR-ASPP + MobileNetV3-Large** — semantic segmentation | Official TorchVision API publishes LR-ASPP MobileNetV3-Large and its weight metadata.[4] TorchVision repository code is BSD-3-Clause.[5] Weight provenance/terms must be recorded from the exact downloaded artefact. | Official model card reports 3.22M parameters and 12.5 MB checkpoint.[4] Fixed-shape ONNX export is a reasonable *testable* hypothesis, not accepted until logit parity is measured. | Minimal compact RGB student/control with low engineering novelty; its stock task is VOC/COCO-style, not indoor V1. | **Candidate A** after indoor fine-tuning. |
| **PP-LiteSeg-T / STDC1** — semantic segmentation | Official PaddleSeg PP-LiteSeg README calls it a lightweight real-time semantic segmentation model and documents ONNX+TRT deployment; included public configs/checkpoints are Cityscapes/CamVid examples, not indoor proof.[6] PaddleSeg code is Apache-2.0.[7] No separate checkpoint licence statement was found in the config README. | Compact decoder/backbone family; ONNX exists in its upstream deployment path, but CPU/ORT cost and PyTorch/Paddle→ONNX parity are unmeasured. | Good compact challenger for a fixed five-class indoor head; road-scene weights should only initialize/compare, never be cited as indoor quality. | **Candidate B** after indoor fine-tuning. |
| **PIDNet-S** — real-time semantic segmentation with detail/boundary branch | Official repo describes real-time semantic segmentation and is MIT licensed.[8][9] Its README warns that original weight download links were broken and redirects to Drive; weight licence/provenance must therefore be captured independently.[8] | Potentially useful where opening boundaries, not region semantics, are the measured error; export/ORT path must be built and parity-tested. | It adds experiment cost and has no supplied indoor acceptance evidence. | **Conditional Candidate C** only after boundary-error diagnosis. |
| **MiDaS small / DPT** — monocular depth (auxiliary geometry) | Official project: “code to compute depth from a single image”, offers an ONNX path only for MiDaS v2.1, and lists MIT licence.[10][11] | Some variants are small; exact checkpoint, resolution, ONNX graph, scale behaviour and CPU latency must be measured per artefact. | Can supply a *relative-depth hypothesis* or offline diagnostic around semantic boundaries. It does not solve V1 taxonomy, has no right to create free-space, and should not join Experiment 1. | Defer: geometry-only spike after semantic acceptance. |

**Not shortlisted as a first candidate:** foundation/VLM, cloud APIs, large panoptic systems, and GPU-only claims all violate either the narrow task or CPU/privacy boundary. An instance/panoptic model can later improve hard-negative analysis, but its “thing” instances still do not prove an aperture is empty.

## 3. Indoor data and GT shortlist

| Dataset | Evidence available / target-label fit | Isolation and role | Access / legal risk |
|---|---|---|---|
| **NYUv2** | Official page: 1,449 densely labelled aligned RGB-depth pairs from 464 indoor scenes; raw data also include RGB, depth and accelerometer data.[12] Dense indoor labels make it a practical small real seed for floor/wall/ceiling and door/window-to-candidate mapping. | Use **scene** as the atomic split key. Good for a first real-data training/control split; raw video must never leak adjacent frames across train/calibration/test. | Landing page exposes downloads but the retrieved page does **not state an explicit dataset licence**. Treat product/redistribution rights as unconfirmed until separately cleared; retain acquisition/source revision evidence. |
| **ScanNet v2** | Official README: 2.5M RGB-D views, >1,500 scans, 3D camera poses, surface reconstructions, instance semantics; each sequence includes RGB-D stream, poses, 2D semantic projections and NYU40-labelled meshes.[13] Strong coverage for real moving-camera temporal evaluation. | Split by `scene<spaceId>_<scanId>` / physical space, never frame. Use for train/eval only after terms approval; it can provide consecutive-frame flicker tests and depth/pose as **offline evaluator** evidence. | Access requires a signed Terms-of-Use agreement and institutional email according to the official README.[13] Research access is not a redistribution/commercial permission. |
| **SUN RGB-D** | Official page: 10,335 RGB-D images from four sensors; densely annotated 2D polygons, 3D boxes, room layout and scene category; it explicitly aggregates NYUv2, B3DO and SUN3D.[14] | Diverse sensor/domain stress set; use supplied image/scene provenance where available and prevent source-scene overlap with NYUv2. | Landing page is not a unified licence and inherits constituent-data obligations. Do not call it commercially redistributable without a separate terms review.[14] |
| **Matterport3D** | Official page: 90 building-scale scenes, 194,400 RGB-D images, 10,800 panoramas, poses and 2D/3D semantic segmentations.[15] It is valuable for building-scale multi-view and sequential holdout. | Split by building/scan; good later temporal and camera-pose stress test, not the smallest first training dependency. | Requires a signed Terms-of-Use request.[15] Treat data as restricted research material; no weight/pixel redistribution assumption. |
| **Hypersim** | Official README: photorealistic synthetic indoor corpus, 77,400 images/461 scenes (public release 74,619), dense semantic/instance labels, complete camera information and ground-truth geometry; semantics use NYU40 IDs.[16] | Scene/trajectory is a clean split key. Use only as optional pretraining/augmentation arm with a mandatory real-data fine-tune and real-only test. | Explicit **CC BY-SA 3.0**.[16] Share-alike and synthetic-to-real gap make it unsuitable as sole data or an unreviewed product source. |
| **Structured3D** | Official page: 3.5K professionally designed house scenes with ground-truth 3D structure and photorealistic 2D images; its page distinguishes MIT **code** from dataset **Terms of Use**.[17] | Good synthetic architecture/layout stress data; split by house design. Use as augmentation only, not independent real-camera acceptance. | Dataset is under its own ToU, not MIT.[17] Obtain/review the ToU before acquisition; no redistribution inference. |

### GT that the first gate actually needs

1. **RGB + dense visible semantic mask** mapped by a versioned mapping table. Preserve source `ignore`/void outside the denominator; do not map it to `other_non_surface` or `withheld`.
2. **Scene/session/capture identity** sufficient to prevent correlated near-duplicate frames crossing splits. Dataset-provided scene IDs are mandatory; MVM data need a consented session + physical-scene ID.
3. **Opening source labels** must retain `door` vs `window` internally, even if V1 reports one `opening_candidate`; add hard negatives (cabinet fronts, posters, mirrors, curtains, glass, closed doors).
4. **Optional depth/pose/normal GT** is evaluator-only in Iteration 1: use it to stratify depth discontinuities/occlusion and future geometry work, not to claim RGB inference is depth-backed.
5. **MVM-camera holdout:** private, consented, manually annotated real frames/video, camera/session isolated from every fitting and threshold decision. Commit only manifest, annotation schema, hashes and aggregate reports—not raw pixels or identity data.

## 4. Practical protocol — two bounded iterations

### Iteration 1: make the data/evaluator gate real (no new runtime)

1. **Freeze label contract.** Emit the five supervised labels above, declared `ignore`, and a separate `withheld` policy. Make a native-label mapping fixture for every source class; unknown/ambiguous native labels fail closed to ignore/explicit review, never to floor.
2. **Prepare manifests before training.** Start with NYUv2 scene-isolated train/calibration/test; add a small consented MVM-camera test-only holdout. If MVM holdout does not exist, status is `not_measured`, not “validated by ADE20K”.
3. **Independent evaluator.** It reads GT and stored raw prediction logits/probabilities from distinct artefacts; validates SHA-256, model/preprocess revision, taxonomy and split non-overlap; then computes per-class support, IoU, precision/recall/F1, confusion, floor/opening boundary F1, ECE/Brier, risk–coverage, withheld coverage, and false-floor-on-ignore/OOD. Include negative tests for hash drift and scene leakage.
4. **Frozen controls.** Run current SegFormer-B0 reducer unchanged plus a probability-mass reducer (not just `argmax → remap`) to separate reducer from backbone failure. Pin every prediction artifact and report the historical ADE result exactly, separately from new scores.
5. **Error buckets.** Label mapping, preprocessing, rare opening support, boundary, calibration/overconfidence, domain shift, low light/glare, clutter/occlusion and temporal flicker. A model change is allowed only with a falsifiable bucket and test metric.

**Iteration-1 decision:** If the evaluator, real GT and isolated split cannot be produced, stop: no model selection claim. If they can, select at most three exact candidate artefacts for Iteration 2.

### Iteration 2: one comparable training hypothesis

* Same data manifests, input geometry, augmentation policy, evaluator, CPU host, preprocessing/output grid and exported raw-logit schema for all arms.
* Arms: LR-ASPP fine-tune, PP-LiteSeg-T/STDC1 fine-tune, and the frozen SegFormer control. Add PIDNet-S only if Iteration 1 proves boundary error is dominant and its weights/licence are independently recorded.
* Train only on train split. Choose loss/early stop/checkpoint and class-specific temperature/abstention thresholds only on calibration split. Start with class-balanced CE + Dice; add one opening-boundary loss only after a measured boundary hypothesis. No multi-change “improvement” run.
* Export selected FP32 fixed-shape ONNX; run framework-to-ORT logit/mask parity, ONNX checker, then CPU p50/p95 (preprocess + ORT + postprocess), RSS, model bytes and sparse-cadence amortization. Quantization is a separate post-acceptance arm using calibration-only data.
* Score final locked threshold once on untouched real test and once on untouched MVM-camera holdout. For RGB sequences, report raw-frame score and causal temporal score separately; a temporal filter may only preserve or withhold observed labels and must not fill unseen space.

## 5. Admission gates

These are **MVM initial hypotheses**, not claimed robotics/segmentation industry standards. Numerical targets must be pre-registered after the frozen real-data baseline and class support are known; never backfilled after test observation.

### A. Shadow-runtime admission (default-off measurement only)

All must pass:

- exact model/weights/preprocess/ONNX and source licences are recorded; any ambiguous code/weight/data licence fails the admission;
- independent evaluator passes hashes, schema and scene/session isolation; raw prediction and GT artifacts are retained;
- candidate improves the **pre-registered safety-critical profile** over frozen control: especially false-floor, opening-candidate precision/boundary and calibrated risk–coverage, rather than only mean IoU;
- `withheld` threshold is selected exclusively on calibration data and reports coverage; withheld/ignore/OOD never emit floor;
- MVM-camera holdout is scored, or admission remains `not_measured` (thus no shadow runtime); CPU parity and budget are measured on the target reference CPU;
- output remains stamped, default-off, observation-only; it cannot produce `SurfaceFact`, free-space, occupancy, passability, opening state, support or action.

**Initial threshold hypothesis (must be frozen before Iteration 2):** require non-inferiority on each floor false-positive and opening precision gate plus an improvement margin over the exact frozen control; express the margin as a confidence interval/paired scene-level comparison, not a hand-picked global mIoU. Choose allowable latency/RSS from MVM’s existing sparse-expert budget, not a paper’s GPU FPS.

### B. Canonical measurement admission (still not occupancy/navigation authority)

This stricter data-quality gate additionally requires:

- repeatable result on an independently captured MVM-camera holdout that spans camera pose, low-light, glare, rugs/thresholds/stairs, clutter/occlusion, glass/mirrors, doors/windows and non-opening lookalikes;
- per-class support adequate for a reported confidence interval; worst-scene/condition results and all failures reported, not only aggregate averages;
- stable class-specific calibration/risk–coverage and zero unreviewed `withheld|ignore|unknown → floor/free` promotions under corruption/OOD and temporal tests;
- independent annotator/adjudication protocol for the private holdout, frozen mapping revision, prediction/GT/metric hashes and a reproducible third-party evaluator invocation;
- a separate approved plan that defines only the semantic measurement envelope. Any later Fusion surface decision, geometry agreement, occupancy/free-space and passability remains a further evidence gate.

## Risks and unresolved points

- **Label semantics:** datasets label *door/window objects*, not aperture topology or state. Collapsing them is useful only as a visual candidate proposal and can increase false positives on closed doors/glass.
- **Licence chain:** code, released weights, training dataset, derivative checkpoint and MVM-collected GT are distinct rights. NVIDIA SegFormer and Niantic Monodepth2 demonstrate why a repository/model must not be assumed permissive: the former has an explicit non-commercial use limitation and Monodepth2 is explicitly non-commercial.[2][20][22]
- **Real-domain evidence gap:** NYUv2/ScanNet/SUN/Matterport sensors and framing differ from the MVM camera. Synthetic data are auxiliary, never acceptance proof. Lack of private holdout must remain visibly `not_measured`.
- **Temporal overclaim:** smoothing may make masks look stable while copying a past error across frames. Evaluate it causally, stratified by camera motion/occlusion, and preserve abstention.
- **Data scale:** NYUv2 alone is small; this is a deliberate first experiment to validate data/evaluator and failure mode, not a claim that it solves robust indoor deployment.

## Sources

[1] https://arxiv.org/abs/2105.15203
[2] https://raw.githubusercontent.com/NVlabs/SegFormer/master/LICENSE
[3] https://huggingface.co/api/models/nvidia/segformer-b0-finetuned-ade-512-512
[4] https://pytorch.org/vision/main/models/generated/torchvision.models.segmentation.lraspp_mobilenet_v3_large.html
[5] https://raw.githubusercontent.com/pytorch/vision/main/LICENSE
[6] https://raw.githubusercontent.com/PaddlePaddle/PaddleSeg/release/2.10/configs/pp_liteseg/README.md
[7] https://raw.githubusercontent.com/PaddlePaddle/PaddleSeg/release/2.10/LICENSE
[8] https://raw.githubusercontent.com/XuJiacong/PIDNet/main/README.md
[9] https://raw.githubusercontent.com/XuJiacong/PIDNet/main/LICENSE
[10] https://github.com/isl-org/MiDaS
[11] https://raw.githubusercontent.com/isl-org/MiDaS/master/LICENSE
[12] https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html
[13] https://raw.githubusercontent.com/ScanNet/ScanNet/master/README.md
[14] https://rgbd.cs.princeton.edu
[15] https://niessner.github.io/Matterport
[16] https://raw.githubusercontent.com/apple/ml-hypersim/main/README.md
[17] https://structured3d-dataset.org
[18] https://arxiv.org/abs/1706.04599
[19] https://arxiv.org/abs/1705.08500
[20] https://arxiv.org/abs/1806.01260
[21] https://arxiv.org/abs/2104.14812
[22] https://raw.githubusercontent.com/nianticlabs/monodepth2/master/LICENSE
