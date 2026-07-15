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
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KAGGLE = os.path.join(os.path.dirname(sys.executable), "kaggle")  # venv's CLI
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
        files=[os.path.join(REPO_ROOT, "spike_pose", "data_probe", "landmarks.npz")],
        licenses=[{"name": "CC-BY-4.0"}],
        card="""# MVM Pose Landmarks

33x4 MediaPipe-style pose landmarks (x, y, z, visibility) for standing / sitting / lying,
**derived** from the public URFD (fall detection) and Zenodo MultiPosture datasets.
**Keypoints only — no images are redistributed.**

## Contents (`landmarks.npz`)
- `X`: float32 (N, 33, 4) — landmarks
- `y`: str — label {lying, sitting, standing}
- `grp`: str — person/sequence group id (use for GroupKFold; frames of one group must not split)
- `src`: str — origin {urfd, multiposture}

## License & attribution
CC BY 4.0. Derivative of URFD (http://fenix.ur.edu.pl/~mkepski/ds/uf.html) and the
Zenodo MultiPosture dataset. Attribute both original sources.

## Starter notebook
See "GRIP: Gravity-Referenced Invariant Pose Features" (NB3b of the MVM ML-journey collection).
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
    os.makedirs(d, exist_ok=True)
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
    os.makedirs(d, exist_ok=True)
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
