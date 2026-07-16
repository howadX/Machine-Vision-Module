#!/usr/bin/env python3
"""kctl — Kaggle control for the MVM ML-portfolio.

One driver for the whole loop: regenerate notebooks from their build scripts,
then push/pull them as Kaggle kernels. All kernel identity lives in the
NOTEBOOKS manifest below (single source of truth) so pushes *update* the
existing kernels instead of creating duplicates.

Each notebook lives in its own folder ``notebooks/<nb>/`` (build script + built
``.ipynb`` + design skeleton). Kaggle wants exactly one ``kernel-metadata.json``
per folder with the code file beside it, so push/pull stage each notebook into a
throwaway ``.stage/<nb>/`` folder (git-ignored) with a freshly-written metadata file.

Run everything through the tool venv:

    ./.venv/bin/python kctl.py build [nb|all]     # _build_nbX.py -> nbX_*.ipynb
    ./.venv/bin/python kctl.py status             # local vs. remote at a glance
    ./.venv/bin/python kctl.py push  <nb|all> --yes   # upload (needs --yes)
    ./.venv/bin/python kctl.py pull  <nb|all>     # fetch remote into pulled/<nb>/
    ./.venv/bin/python kctl.py list               # my kernels on Kaggle

``nb`` is a short key: nb0 nb1 nb2 nb3 nb3b nb4.  Pushing overwrites the remote
kernel's contents with the local notebook — the local build is authoritative.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _find_kaggle():
    """The authed CLI lives in the tool venv (.venv). Resolving from
    sys.executable bit us on LJ5: run kctl with the main repo venv and KAGGLE
    points at ../venv/bin/kaggle, whose kagglesdk is broken under py3.14 —
    push crashes, people fall back to raw `kaggle kernels push` and ship a
    stale .stage. Prefer the tool venv explicitly; fail loudly otherwise."""
    tool = os.path.join(HERE, ".venv", "bin", "kaggle")
    if os.path.exists(tool):
        return tool
    beside = os.path.join(os.path.dirname(sys.executable), "kaggle")
    if os.path.exists(beside):
        return beside
    sys.exit("no kaggle CLI found — expected the tool venv at "
             f"{tool} (see WORKFLOW.md 'One-time setup')")


KAGGLE = _find_kaggle()
STAGE = os.path.join(HERE, ".stage")
PULLED = os.path.join(HERE, "pulled")
RESULTS = os.path.join(HERE, "results", "metrics.json")   # source of truth for real numbers

USER = "howadx"

# --- Single source of truth ------------------------------------------------
# key -> folder, build script, built notebook, remote kernel id, human title.
# Each notebook lives under notebooks/<dir>/; `build` and `ipynb` are basenames
# inside that folder. ids for nb1/nb2/nb3 are the EXISTING slugs on Kaggle
# (do not rename, or a push forks a new kernel); nb0/nb3b/nb4 slugs are for first publish.
NB_ROOT = "notebooks"
NOTEBOOKS = {
    "nb0": dict(
        dir="nb0",
        build="_build_nb0.py",
        ipynb="nb0_emozid.ipynb",
        # NOTE: Kaggle derives the kernel slug from the TITLE; id must equal it.
        id=f"{USER}/emozid-an-svm-on-facial-geometry-2019",
        title="EmoZid: An SVM on Facial Geometry (2019)",
    ),
    "nb1": dict(
        dir="nb1",
        build="_build_nb1.py",
        ipynb="nb1_distillation.ipynb",
        id=f"{USER}/distilling-a-cnn-into-a-geometric-mlp",
        title="Distilling a CNN into a Geometric MLP",
    ),
    "nb2": dict(
        dir="nb2",
        build="_build_nb2.py",
        ipynb="nb2_posevae.ipynb",
        id=f"{USER}/a-pose-manifold-that-knows-when-it-s-surprised",
        title="A Pose Manifold That Knows When It's Surprised",
    ),
    "nb3": dict(
        dir="nb3",
        build="_build_nb3.py",
        ipynb="nb3_weak_supervision.ipynb",
        id=f"{USER}/weak-supervision-at-scale",
        title="Weak Supervision at Scale",
    ),
    "nb3b": dict(
        dir="nb3b",
        build="_build_nb3b.py",
        ipynb="nb3b_grip.ipynb",
        # Renamed from the "Method-C" spike label (an A/B/C enumeration artifact) to GRIP.
        id=f"{USER}/grip-gravity-referenced-invariant-pose-features",
        title="GRIP: Gravity-Referenced Invariant Pose Features",
        dataset_sources=[f"{USER}/mvm-pose-landmarks"],
    ),
    "nb4": dict(
        dir="nb4",
        build="_build_nb4.py",
        ipynb="nb4_pipeline_teardown.ipynb",
        # Kaggle caps kernel titles at 50 chars; the notebook's H1 keeps the full phrasing.
        id=f"{USER}/identity-embeddings-are-wrong-for-emotion",
        title="Identity Embeddings Are Wrong for Emotion",
    ),
    # Phase-15 Lab Journal series (LJ0-LJ6). Each gates one refactor stage; needs the derived
    # datasets mounted (cmu-mocap-views + the pose-landmarks set) for Run-All on Kaggle.
    "lj0": dict(
        dir="lj0",
        build="_build_lj0.py",
        ipynb="lj0_building_the_lab.ipynb",
        id=f"{USER}/lj0-building-the-lab",
        title="LJ0 — Building the Lab",
        dataset_sources=[f"{USER}/cmu-mocap-views", f"{USER}/mvm-pose-landmarks"],
    ),
    "lj1": dict(
        dir="lj1",
        build="_build_lj1.py",
        ipynb="lj1_pose_rebuilt_honestly.ipynb",
        id=f"{USER}/lj1-the-pose-model-rebuilt-honestly",   # must equal slugify(title)
        title="LJ1 — The Pose Model, Rebuilt Honestly",
        dataset_sources=[f"{USER}/cmu-mocap-views", f"{USER}/mvm-pose-landmarks", f"{USER}/mvm-animal-ood"],
    ),
    "lj2": dict(
        dir="lj2",
        build="_build_lj2.py",
        ipynb="lj2_physics_of_posture_transitions.ipynb",
        id=f"{USER}/lj2-physics-of-posture-transitions",   # must equal slugify(title)
        title="LJ2 — Physics of Posture Transitions",
        dataset_sources=[f"{USER}/mvm-mocap-transitions", f"{USER}/mvm-pose-landmarks"],
    ),
    "lj3": dict(
        dir="lj3",
        build="_build_lj3.py",
        ipynb="lj3_why_walking_people_multiply.ipynb",
        id=f"{USER}/lj3-why-walking-people-multiply",      # must equal slugify(title), <=50 chars
        title="LJ3 — Why Walking People Multiply",
        dataset_sources=[f"{USER}/mvm-mocap-tracks"],
    ),
    "lj4": dict(
        dir="lj4",
        build="_build_lj4.py",
        ipynb="lj4_a_camera_that_never_lies.ipynb",
        id=f"{USER}/lj4-a-camera-that-never-lies",         # must equal slugify(title), <=50 chars
        title="LJ4 — A Camera That Never Lies",
        # vrd-meta mounted so the S3 audit needs NO kernel internet (API-pushed runs
        # resolve no DNS even with enable_internet=true — learned on LJ4 v1).
        dataset_sources=[f"{USER}/mvm-golden-scenarios", f"{USER}/mvm-openimages-vrd-meta"],
    ),
    "lj5": dict(
        dir="lj5",
        build="_build_lj5.py",
        ipynb="lj5_weighing_the_experts.ipynb",
        id=f"{USER}/lj5-weighing-the-experts",             # must equal slugify(title), <=50 chars
        title="LJ5 — Weighing the Experts",
        # Mounted (API-pushed runs resolve no DNS): the ONNX experts, RAF-DB (emotion held-out,
        # community mirror), LFW (identity). The load-bearing findings (B1 emotion healthy, B2
        # +19.5 align headroom, C SFace-INT8 negative) re-run live off these. Act A latency is
        # reference-hardware (Kaggle-CPU echo only); Act B3 detect is presented (OI images heavy,
        # "where feasible") — so no OpenImages mount is needed for Run-All to complete.
        dataset_sources=[f"{USER}/mvm-onnx-experts", "shuvoalok/raf-db-dataset",
                         f"{USER}/mvm-lfw-verification"],
    ),
    "lj6": dict(
        dir="lj6",
        build="_build_lj6.py",
        ipynb="lj6_shipping_the_scorecard.ipynb",
        id=f"{USER}/lj6-shipping-the-scorecard",           # must equal slugify(title), <=50 chars
        title="LJ6 — Shipping the Scorecard",
        # The closer. Recomputes NOTHING — the scorecard is baked from results/metrics.json at build
        # time (LJ2–LJ5) + LJ1 literals — so Run-All completes offline with no mounts (no DNS on
        # API-pushed runs). H1 keeps the full "Shipping v0.2.0: the scorecard" phrasing.
    ),
}


def nb_path(nb, key):
    """Absolute path to a notebook's file (`build`/`ipynb` basename) in its folder."""
    return os.path.join(HERE, NB_ROOT, nb["dir"], nb[key])

# Derived datasets we publish (keypoints only, no images). Source file is read
# from the main repo; staging copies it next to a generated dataset-metadata.json.
REPO_ROOT = os.path.dirname(HERE)
DATASETS = {
    "landmarks": dict(
        id=f"{USER}/mvm-pose-landmarks",
        title="MVM Pose Landmarks (URFD + MultiPosture)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "landmarks.npz"),
               os.path.join(REPO_ROOT, "spike_pose", "data_probe", "landmarks_v2.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# MVM Pose Landmarks

33x4 MediaPipe-style pose landmarks (x, y, z, visibility) for standing / sitting / lying,
**derived** from public datasets. **Keypoints only — no images are redistributed.**

## Contents
- `landmarks.npz` — the original probe (URFD + Zenodo MultiPosture, 1035 poses). Used by NB3b.
- `landmarks_v2.npz` — expanded set used by the Lab Journal series (LJ0/LJ1): `src ∈ {lab, gold5,
  gold6}` where lab = URFD+MultiPosture and gold = human-labeled Roboflow pose datasets. Fields:
  `X` (N,33,4), `y` {standing,sitting,lying}, `grp` (group id — never split across folds), `src`.
  The clean core used in the journals is `src ∈ {lab, gold5}`.

## License & attribution
CC BY 4.0. Derivative of URFD (http://fenix.ur.edu.pl/~mkepski/ds/uf.html), the Zenodo
MultiPosture dataset, and human-labeled Roboflow posture datasets (gold). Attribute originals.

## Starter notebooks
NB3b (GRIP) uses `landmarks.npz`; LJ0/LJ1 (MVM Lab Journal) use `landmarks_v2.npz`.
""",
    ),
    "mocap": dict(
        id=f"{USER}/cmu-mocap-views",
        title="CMU Mocap Views (13-joint 3D + virtual camera)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "mocap_views.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# CMU Mocap Views

13-joint 3D skeletons + a virtual eye-level camera projection, **derived** from the CMU
Graphics Lab Motion Capture Database (via Bruce Hahne's cgspeed BVH conversion). Built for
an **honest camera-azimuth benchmark**: because mocap is true 3D, rotating the camera is a
REAL viewpoint change, not the circular y-rotation of a monocular pose estimator's noisy z.

**Derived keypoints only — no BVH files or motion data are redistributed.**

## Contents (`mocap_views.npz`)
- `P3D`: float32 (N, 13, 3) — TRUE metric 3D, subject-centered, Y-up, feet ~ y=0.
  Joint order: nose, L/R shoulder, elbow, wrist, hip, knee, ankle. Re-project at any azimuth.
- `X`: float32 (N, 33, 4) — the 13 joints projected to MediaPipe-style (x/w, y/h, z, vis) at
  azimuth 0 (GRIP indices filled, others 0), for drop-in geometric-feature code.
- `y`: str — posture {standing, sitting, lying}, labelled kinematically (pelvis height / leg
  ratio / torso inclination / knee flexion), NOT from the pose feature vector.
- `azimuth`: float — azimuth of `X` (0; `P3D` carries the truth for re-projection).
- `t`: float — time (s) within the trial;  `pelvis`: float — pelvis height (m).
- `subj`, `trial`: str — CMU subject / trial ids (grouping keys; never split a trial across folds).

## How it was built
CMU BVH -> forward kinematics -> 13 joints -> metric scale (leg length -> 0.90 m) ->
subject-centered 3D -> pinhole camera at 1.75 m, horizontal. Subjects 111 & 113 perform every
posture, so class is independent of subject (no class<->source confound). Full pipeline:
`spike_pose/mocap_to_views.py` in the MVM repo.

## License & attribution
CC BY 4.0 on these derived features. Source: **CMU Graphics Lab Motion Capture Database**
(http://mocap.cs.cmu.edu — free for research and product use with attribution; direct resale
of the data prohibited) and **Bruce Hahne's cgspeed BVH conversion**. Attribute both.

## Starter notebook
See the MVM "Lab Journal" series (LJ0 — Building the lab).
""",
    ),
    "animal": dict(
        id=f"{USER}/mvm-animal-ood",
        title="MVM Animal OOD (false-positive pose detections)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "animal_ood_poses.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# MVM Animal OOD

Out-of-distribution pose landmarks: an MPPose human-pose estimator **forced onto dog/cat images**
(no person present) — real false-positive detections. Used in LJ1 to show that a class-conditional
Mahalanobis OOD gate flags these near-manifold cases where reconstruction-error detectors (a VAE)
do not (Denouden 2018). **Derived keypoints only — no images redistributed.**

## Contents (`animal_ood_poses.npz`)
- `X`: float32 (N, 33, 4) — MediaPipe-style landmarks (the garbage skeletons)
- `y`: str — "ood";  `grp`, `src`, `conf`: provenance fields

## License & attribution
CC BY 4.0 on these derived features. Source images: OpenImages V6 Dog/Cat subset (annotations
CC BY 4.0; images CC BY 2.0, NOT redistributed here). Attribute OpenImages.

## Starter notebook
See "LJ1 — The Pose Model, Rebuilt Honestly" (MVM Lab Journal).
""",
    ),
    "mocap_transitions": dict(
        id=f"{USER}/mvm-mocap-transitions",
        title="MVM Mocap Transitions (posture, 60fps)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "mocap_transitions.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# MVM Mocap Transitions

Transition-focused CMU mocap: 35 trials (sit↔stand, lie-down/get-up, floor falls, stable
standing) as 13-joint 3D + a virtual eye-level camera projection, at **60 fps** so the pelvis
velocity of a fall is resolved. **Derived** from the CMU Graphics Lab Motion Capture Database
(via Bruce Hahne's cgspeed BVH conversion). Built for phase-15.2 (LJ2): posture-transition
kinematics, the stabiliser flip-rate↔latency benchmark, and the fall-vs-lie-down velocity threshold.

**Derived keypoints only — no BVH files or motion data are redistributed.**

## Contents (`mocap_transitions.npz`)
- `X`: float32 (N, 33, 4) — 13 joints projected MediaPipe-style at azimuth 0 (GRIP slots filled).
- `P3D`: float32 (N, 13, 3) — TRUE metric 3D, subject-centered, Y-up, feet ~ y=0.
- `y`: str — per-frame posture {standing, sitting, lying}, labelled kinematically (NOT the features).
- `t`: float — time (s) within the trial (60 fps);  `pelvis`: float — pelvis height (m).
- `scenario`: str — per-trial mechanism {sit_stand, lie_down, controlled_desc, fall, standing}.
- `subj`, `trial`: str — CMU subject / trial ids (grouping keys; never split a trial across folds).

## How it was built
CMU BVH -> forward kinematics -> 13 joints -> metric leg-length scale -> subject-centered 3D
(robust floor from a low percentile of the per-frame lowest joint, ignoring BVH settling frames)
-> pinhole camera at 1.75 m, horizontal, 60 fps. Pipeline: `spike_pose/lj2_transitions.py`.

## License & attribution
CC BY 4.0 on these derived features. Source: **CMU Graphics Lab Motion Capture Database**
(http://mocap.cs.cmu.edu — free for research and product use with attribution; resale prohibited)
and **Bruce Hahne's cgspeed BVH conversion**. Attribute both.

## Starter notebook
See "LJ2 — Physics of Posture Transitions" (MVM Lab Journal).
""",
    ),
    "mocap_tracks": dict(
        id=f"{USER}/mvm-mocap-tracks",
        title="MVM Mocap Tracks (walking bbox trajectories)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "mocap_tracks.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# MVM Mocap Tracks

CMU walking mocap projected to **bounding-box trajectories with ground-truth identity**, for the
phase-15.3 tracker benchmark (LJ3): reproduce the live "walking fragmentation" bug (one walker
splits into multiple phantom track ids) on repeatable data. 24 walk trials + 2 standing-still
(idle regression) + 2 head-on crossings (id-swap regression). **Derived** from the CMU Graphics
Lab Motion Capture Database (via Bruce Hahne's cgspeed BVH conversion).

Unlike the per-frame-centered pose sets, this one **preserves world translation** (a single global
offset, not per-frame hip centering) so the person actually moves across the frame — that motion is
what fragments the tracker.

**Derived boxes + identity only — no BVH files or motion data are redistributed.**

## Contents (`mocap_tracks.npz`, flat, grouped by `seq`)
- `seq`: str — sequence id ("07_01" | "idle_111_28" | "cross_07_01__08_02");  `kind`: {walk, idle, cross}.
- `frame`: int32 — frame index within the sequence (15 fps, the production human-chain cadence).
- `person`: int8 — TRUE identity within the sequence (0 or 1) — the tracking ground truth.
- `bbox`: float32 (M, 4) — image-normalized [x1,y1,x2,y2] (tall-narrow, YOLO-like; may exceed [0,1] at frame edges).
- `vis`: bool — bbox center within frame (a detector could plausibly see it).
- `t`: float — time (s) within the sequence;  `pelvis3d`: float32 (M, 3) — world metric pelvis (m).

## How it was built
CMU BVH -> forward kinematics -> 13 joints -> metric leg-length scale -> global offset (translation
PRESERVED) -> walk oriented laterally (worst case for the IoU gate) -> pinhole camera at 1.75 m
-> tall-narrow person bbox from the body core. Pipeline: `spike_pose/lj3_trajectories.py`.

## License & attribution
CC BY 4.0 on these derived features. Source: **CMU Graphics Lab Motion Capture Database**
(http://mocap.cs.cmu.edu — free for research and product use with attribution; resale prohibited)
and **Bruce Hahne's cgspeed BVH conversion**. Attribute both.

## Starter notebook
See "LJ3 — Why Walking People Multiply" (MVM Lab Journal).
""",
    ),
    "golden": dict(
        id=f"{USER}/mvm-golden-scenarios",
        title="MVM Golden Scenarios (fusion-layer truth)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "golden_scenarios.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# MVM Golden Scenarios

Synthetic golden scenarios for the phase-15.4 fusion-layer simulator (LJ4): timelines whose
truth is known **by construction** — walked-in→sat→works, stood-up→left, controlled lie-down /
get-up, falls, two people meeting, a lone walker — compiled from the CMU Graphics Lab Motion
Capture Database (via Bruce Hahne's cgspeed BVH conversion) in world 3D (pelvis-continuous
seams, robust percentile floor, settling frames dropped) and projected through a virtual
eye-level camera (1.75 m, 5 m rig — the floor stays in frame so a lying body remains visible).

**Derived features only — no BVH files, no motion data, no images are redistributed.**

## Contents (`golden_scenarios.npz`, flat, grouped by `inst`)
- `inst`, `group`: str — scenario instance / group (G1..G6); 28 instances @ 15 fps.
- `frame`: int32;  `t`: float32 — time (s);  `person`: int8 — TRUE identity (0/1).
- `bbox`: float32 (M,4) — normalized [x1,y1,x2,y2] tall-narrow person box;  `vis`: bool.
- `posture`: str — kinematic ground-truth label (biomechanics, NOT pose features).
- `wrists`: float32 (M,2,2) — projected wrist points (the contact truth).
- `pelvis3d`: float32 (M,3) — world metric pelvis (fall physics / the E2 rule).
- `pos3d`: float32 (M,3) — camera-frame pelvis (3D proximity truth).
- `obj_*` — synthetic static furniture (keyboard/chair) per instance.
- `meta_json`: str — per-instance expected events with tolerances (the scoring truth).

## License & attribution
CC BY 4.0 on these derived features. Source: **CMU Graphics Lab Motion Capture Database**
(http://mocap.cs.cmu.edu — free for research and product use with attribution; resale
prohibited) and **Bruce Hahne's cgspeed BVH conversion**. Attribute both.

## Starter notebook
See "LJ4 — A Camera That Never Lies" (MVM Lab Journal).
""",
    ),
    "vrdmeta": dict(
        id=f"{USER}/mvm-openimages-vrd-meta",
        title="OpenImages V6 VRD Annotations (val+test)",
        files=[os.path.join(REPO_ROOT, "spike_pose", "oi_meta", f) for f in
               ("oidv6-validation-annotations-vrd.csv", "oidv6-test-annotations-vrd.csv",
                "oidv6-class-descriptions.csv")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# OpenImages V6 VRD Annotations (val + test)

Unmodified re-hosting of three OpenImages V6 metadata files, mounted so the MVM Lab Journal
notebooks can run **without kernel internet**:

- `oidv6-validation-annotations-vrd.csv` — per-image visual-relationship instances (validation)
- `oidv6-test-annotations-vrd.csv` — per-image visual-relationship instances (test)
- `oidv6-class-descriptions.csv` — MID → human-readable class name

## License & attribution
The OpenImages annotations are licensed **CC BY 4.0** by Google LLC — redistribution with
attribution permitted. Source: https://storage.googleapis.com/openimages/web/download_v6.html
(Open Images Dataset V6, Kuznetsova et al., IJCV 2020). No images are included.

## Starter notebook
See "LJ4 — A Camera That Never Lies" (MVM Lab Journal) — the raw-activity vocabulary audit.
""",
    ),
}

# Kaggle kernel settings shared by every notebook. Internet is on for kagglehub
# dataset downloads; no accelerator (all models are CPU-sized).
KERNEL_DEFAULTS = dict(
    language="python",
    kernel_type="notebook",
    is_private=True,
    enable_gpu=False,
    enable_tpu=False,
    enable_internet=True,
    dataset_sources=[],
    kernel_sources=[],
    competition_sources=[],
    model_sources=[],
)


def resolve(keys):
    if keys == ["all"] or keys == []:
        return list(NOTEBOOKS)
    for k in keys:
        if k not in NOTEBOOKS:
            sys.exit(f"unknown notebook '{k}'. known: {', '.join(NOTEBOOKS)}")
    return keys


def build(keys):
    for k in resolve(keys):
        nb = NOTEBOOKS[k]
        script = nb_path(nb, "build")
        if not os.path.exists(script):
            print(f"  {k}: SKIP (no {nb['build']} yet)")
            continue
        print(f"  {k}: {nb['dir']}/{nb['build']} -> {nb['ipynb']}")
        subprocess.run([sys.executable, script], cwd=HERE, check=True)


def _stage(k):
    """Create .stage/<k>/ with the notebook + a fresh kernel-metadata.json."""
    nb = NOTEBOOKS[k]
    src = nb_path(nb, "ipynb")
    if not os.path.exists(src):
        sys.exit(f"{k}: {nb['ipynb']} not built yet — run: kctl.py build {k}")
    if len(nb["title"]) > 50:
        sys.exit(f"{k}: Kaggle title must be <= 50 chars (got {len(nb['title'])}): {nb['title']!r}")
    d = os.path.join(STAGE, k)
    # .stage is a throwaway build product: recreate it from scratch so nothing
    # stale ever ships (raw `kaggle kernels push -p .stage/<k>` bit us on LJ5).
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    dst = os.path.join(d, nb["ipynb"])
    with open(src) as f:
        data = f.read()
    with open(dst, "w") as f:
        f.write(data)
    meta = dict(KERNEL_DEFAULTS, id=nb["id"], title=nb["title"], code_file=nb["ipynb"])
    if nb.get("dataset_sources"):
        meta["dataset_sources"] = nb["dataset_sources"]
    with open(os.path.join(d, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return d


def push(keys, yes):
    ks = resolve(keys)
    if not yes:
        print("Refusing to publish without --yes. Would push:")
        for k in ks:
            print(f"  {k} -> {NOTEBOOKS[k]['id']}")
        sys.exit(1)
    for k in ks:
        d = _stage(k)
        print(f"== push {k} -> {NOTEBOOKS[k]['id']} ==")
        subprocess.run([KAGGLE, "kernels", "push", "-p", d], check=True)


def _stage_data(key):
    ds = DATASETS[key]
    d = os.path.join(STAGE, "data-" + key)
    shutil.rmtree(d, ignore_errors=True)   # throwaway: never version a stale file
    os.makedirs(d)
    for src in ds["files"]:
        if not os.path.exists(src):
            sys.exit(f"dataset '{key}': missing source file {src}")
        with open(src, "rb") as f, open(os.path.join(d, os.path.basename(src)), "wb") as g:
            g.write(f.read())
    meta = {"title": ds["title"], "id": ds["id"], "licenses": ds["licenses"]}
    with open(os.path.join(d, "dataset-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(d, "README.md"), "w") as f:
        f.write(ds["card"])
    return d


def push_data(key, yes, create):
    if key not in DATASETS:
        sys.exit(f"unknown dataset '{key}'. known: {', '.join(DATASETS)}")
    d = _stage_data(key)
    if not yes:
        print(f"Refusing to publish without --yes. Would {'create' if create else 'version'} "
              f"{DATASETS[key]['id']} from:\n  {d}")
        sys.exit(1)
    if create:
        subprocess.run([KAGGLE, "datasets", "create", "-p", d, "--dir-mode", "zip"], check=True)
    else:
        subprocess.run([KAGGLE, "datasets", "version", "-p", d, "-m", "update", "--dir-mode", "zip"],
                       check=True)


def pull(keys):
    for k in resolve(keys):
        nb = NOTEBOOKS[k]
        d = os.path.join(PULLED, k)
        os.makedirs(d, exist_ok=True)
        print(f"== pull {nb['id']} -> pulled/{k}/ ==")
        subprocess.run([KAGGLE, "kernels", "pull", nb["id"], "-p", d, "-m"], check=False)


def runstatus(keys):
    """Show the latest server-side run status of each kernel (a push auto-runs it)."""
    for k in resolve(keys):
        r = subprocess.run([KAGGLE, "kernels", "status", NOTEBOOKS[k]["id"]],
                           capture_output=True, text=True)
        line = (r.stdout or r.stderr).strip().replace("KernelWorkerStatus.", "")
        print(f"  {k:6} {line}")


def logs(key):
    if key not in NOTEBOOKS:
        sys.exit(f"unknown notebook '{key}'")
    subprocess.run([KAGGLE, "kernels", "logs", NOTEBOOKS[key]["id"]])


def fetch(keys):
    """Download each kernel's run output and merge its metrics.json into results/metrics.json."""
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    allm = json.load(open(RESULTS)) if os.path.exists(RESULTS) else {}
    for k in resolve(keys):
        d = os.path.join(PULLED, k)
        os.makedirs(d, exist_ok=True)
        subprocess.run([KAGGLE, "kernels", "output", NOTEBOOKS[k]["id"], "-p", d],
                       check=False, capture_output=True)
        mfile = os.path.join(d, "metrics.json")
        if os.path.exists(mfile):
            allm[k] = json.load(open(mfile))
            print(f"  {k}: merged {len(allm[k])} metric group(s)")
        else:
            print(f"  {k}: no metrics.json in output (run errored, or notebook emits none)")
    with open(RESULTS, "w") as f:
        json.dump(allm, f, indent=2, ensure_ascii=False)
    print(f"-> {os.path.relpath(RESULTS, HERE)}")


def status():
    out = subprocess.run(
        [KAGGLE, "kernels", "list", "--user", USER, "--csv"],
        capture_output=True, text=True,
    ).stdout
    remote = {line.split(",")[0] for line in out.splitlines()[1:] if line}
    print(f"{'nb':6} {'built':6} {'remote':7} id")
    print("-" * 70)
    for k, nb in NOTEBOOKS.items():
        built = "yes" if os.path.exists(nb_path(nb, "ipynb")) else "-"
        onremote = "live" if nb["id"] in remote else "-"
        print(f"{k:6} {built:6} {onremote:7} {nb['id']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "push", "pull"):
        sp = sub.add_parser(name)
        sp.add_argument("nb", nargs="*", default=["all"])
        if name == "push":
            sp.add_argument("--yes", action="store_true", help="confirm publish")
    pd = sub.add_parser("push-data")
    pd.add_argument("dataset", nargs="?", default="landmarks")
    pd.add_argument("--yes", action="store_true", help="confirm publish")
    pd.add_argument("--create", action="store_true", help="first-time create (else new version)")
    for name in ("runstatus", "fetch"):
        sub.add_parser(name).add_argument("nb", nargs="*", default=["all"])
    sub.add_parser("logs").add_argument("nb")
    sub.add_parser("status")
    sub.add_parser("list")
    a = p.parse_args()

    if a.cmd == "build":
        build(a.nb or ["all"])
    elif a.cmd == "push":
        push(a.nb or ["all"], a.yes)
    elif a.cmd == "pull":
        pull(a.nb or ["all"])
    elif a.cmd == "push-data":
        push_data(a.dataset, a.yes, a.create)
    elif a.cmd == "runstatus":
        runstatus(a.nb or ["all"])
    elif a.cmd == "fetch":
        fetch(a.nb or ["all"])
    elif a.cmd == "logs":
        logs(a.nb)
    elif a.cmd == "status":
        status()
    elif a.cmd == "list":
        subprocess.run([KAGGLE, "kernels", "list", "--user", USER])


if __name__ == "__main__":
    main()
