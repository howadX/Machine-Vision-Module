#!/usr/bin/env python3
"""LJ5 Act A — latency & weight ledger for the external ONNX experts (phase 15.5).

Measures, on the reference hardware, each production ONNX expert's:
  * weight   — exact file bytes (matches mvm-cli/src/manifest.rs);
  * latency  — per-model onnxruntime CPU inference, median + p95 over timed runs,
               at two thread settings that BRACKET real deploys:
                 - "all"  = intra_op_num_threads=0 (all cores; single-model peak on this box)
                 - "one"  = intra_op_num_threads=1 (portable floor; no multicore advantage)

Why per-model onnxruntime and not the Rust hot-path: the shipped experts are these exact
ONNX bytes run by the same engine (ort crate ≙ onnxruntime). Per-model isolates the
DECOMPOSITION (which model costs what) reproducibly. The full human-chain wall-clock
(incl. preprocessing/alignment/feature-calc + concurrent depth/YOLO contention) is a
live-camera number the owner validates — cited, not fabricated here. Sum-of-required-models
is the CPU *compute budget* of the human chain (preprocessing is cheap arithmetic vs inference).

Passport: CPU model + thread config are stamped alongside the numbers (latency is
reference-hardware-bound by nature). See research/LJ5-DESIGN.md, Act A.

Usage: venv/bin/python spike_pose/lj5_latency.py [--warmup 30] [--timed 200] [--json out.json]
"""
import os, sys, json, time, argparse, platform, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "mvm", "models")

# Human chain (manifest `required=true`): runs sequentially inside the Human agent per person.
# Cadenced/optional experts run on their own threads/cadence and are NOT in the <50ms budget.
CHAIN = {"face_detection_yunet_2023mar.onnx", "face_recognition_sface_2021dec.onnx",
         "enet_b0_8_best_vgaf.onnx", "person_detection_mediapipe_2023mar.onnx",
         "pose_estimation_mediapipe_2023mar.onnx"}
# Human-readable agent labels + role, in chain order then cadenced.
ORDER = [
    ("face_detection_yunet_2023mar.onnx",      "YuNet (face detect)",      "chain"),
    ("face_recognition_sface_2021dec.onnx",    "SFace (identity)",         "chain"),
    ("enet_b0_8_best_vgaf.onnx",               "EmotiEffLib (emotion)",    "chain"),
    ("person_detection_mediapipe_2023mar.onnx","BlazePose (person det)",   "chain"),
    ("pose_estimation_mediapipe_2023mar.onnx", "BlazePose (pose)",         "chain"),
    ("yolo_object_detect.onnx",                "YOLO26n (objects)",        "cadenced"),
    ("yolo_seg.onnx",                          "YOLO-seg (masks)",         "cadenced"),
    ("depth_anything_v2_small.onnx",           "Depth-Anything (depth)",   "cadenced"),
    ("handpose_estimation_mediapipe_2023feb.onnx","MediaPipe (handpose)",  "optional"),
]
# Dynamic-dim resolution (documented): depth is patch-14, canonical 518 (mvm-depth lib.rs:90).
DYN = {"batch_size": 1, "batch": 1, "height": 518, "width": 518}


def resolve_shape(shape):
    return [DYN.get(d, d) if not isinstance(d, int) else d for d in shape]


def bench_model(path, warmup, timed, threads):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads          # 0 = all cores
    so.inter_op_num_threads = threads
    so.log_severity_level = 3                   # silence initializer-in-inputs warnings
    sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        shp = resolve_shape(i.shape)
        feeds[i.name] = np.random.rand(*shp).astype(np.float32)
    for _ in range(warmup):
        sess.run(None, feeds)
    ts = []
    for _ in range(timed):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        ts.append((time.perf_counter() - t0) * 1e3)   # ms
    ts = np.array(ts)
    in_shape = resolve_shape(sess.get_inputs()[0].shape)
    return {"median_ms": float(np.median(ts)), "p95_ms": float(np.percentile(ts, 95)),
            "min_ms": float(ts.min()), "input_shape": in_shape}


def passport():
    def git(*a):
        try:
            return subprocess.check_output(["git", "-C", ROOT, *a],
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return "?"
    return {"code_commit": git("rev-parse", "--short", "HEAD"),
            "working_tree": "dirty" if git("status", "--porcelain") else "clean",
            "cpu": platform.processor() or _cpu_name(), "logical_cores": os.cpu_count(),
            "onnxruntime": _ort_ver(), "provider": "CPUExecutionProvider"}


def _cpu_name():
    try:
        for ln in open("/proc/cpuinfo"):
            if ln.startswith("model name"):
                return ln.split(":", 1)[1].strip()
    except Exception:
        return "?"


def _ort_ver():
    import onnxruntime as ort
    return ort.__version__


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--timed", type=int, default=200)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    pp = passport()
    print(f"# LJ5 Act A — expert latency & weight ledger")
    print(f"# CPU={pp['cpu']}  cores={pp['logical_cores']}  ort={pp['onnxruntime']}  "
          f"commit={pp['code_commit']} ({pp['working_tree']})")
    print(f"# per-model onnxruntime CPU, warmup={a.warmup} timed={a.timed}\n")

    rows, results = [], {}
    for fn, label, role in ORDER:
        path = os.path.join(MODELS, fn)
        if not os.path.exists(path):
            print(f"  MISSING {fn}"); continue
        weight_kb = os.path.getsize(path) / 1024
        allc = bench_model(path, a.warmup, a.timed, 0)
        one = bench_model(path, a.warmup, a.timed, 1)
        results[fn] = {"label": label, "role": role, "weight_kb": round(weight_kb, 1),
                       "threads_all": allc, "threads_one": one}
        rows.append((label, role, weight_kb, allc["median_ms"], allc["p95_ms"], one["median_ms"]))

    print(f"{'agent':26s} {'role':9s} {'weight':>9s} {'lat(all)':>10s} {'p95':>8s} {'lat(1thr)':>10s}")
    print("-" * 76)
    for label, role, w, lat_all, p95, lat_one in rows:
        print(f"{label:26s} {role:9s} {w/1024:7.2f}MB {lat_all:8.2f}ms {p95:6.2f}ms {lat_one:8.2f}ms")

    chain = [r for r in rows if r[1] == "chain"]
    sum_all = sum(r[3] for r in chain); sum_one = sum(r[5] for r in chain)
    tot_w = sum(r[2] for r in rows)
    print("-" * 76)
    print(f"HUMAN CHAIN compute budget (sum of {len(chain)} required models):"
          f"  all-cores {sum_all:.1f}ms   1-thread {sum_one:.1f}ms")
    print(f"BUDGET GATE <50ms CPU: {'PASS' if sum_all < 50 else 'OVER'} (all-cores, single-model isolated)")
    print(f"TOTAL ONNX weight (all 9): {tot_w/1024:.2f} MB")
    print("\nNote: full human-chain wall-clock (preprocessing + concurrent depth/YOLO contention)"
          "\n      is a live-camera number (owner-validated ~16 fps engine, CLAUDE.md) — this ledger"
          "\n      is the reproducible per-model DECOMPOSITION, not the contended live figure.")

    out = {"passport": pp, "warmup": a.warmup, "timed": a.timed, "models": results,
           "human_chain_sum_ms": {"threads_all": round(sum_all, 2), "threads_one": round(sum_one, 2)},
           "total_onnx_weight_mb": round(tot_w / 1024, 3),
           "budget_gate_50ms_all": bool(sum_all < 50)}
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")
    return out


if __name__ == "__main__":
    main()
