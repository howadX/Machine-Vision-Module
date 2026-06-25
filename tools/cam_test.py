#!/usr/bin/env python3
"""Real-camera scenario test harness for the MVM native pipeline.

Runs `mvm-cli --native --dump-json` for a fixed duration, captures the Scene
JSON stream to a per-scenario log, and prints a structured summary: FPS, track
stability, identity, emotion smoothing, 3D distance, and the relational
scene-graph (zones/relations — the 6.4a feature that only exercises with 2+
entities in frame).

Usage (from project root):

    venv/bin/python3 tools/cam_test.py --scenario single_sit --secs 20
    venv/bin/python3 tools/cam_test.py --scenario two_people --secs 25
    venv/bin/python3 tools/cam_test.py --replay logs/cam_two_people.jsonl   # re-analyse a log

Logs land in mvm/logs/cam_<scenario>.jsonl. Each run is self-contained; analyse
the printed verdict against the scenario's expectation (see SCENARIOS below).
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MVM_DIR = os.path.join(ROOT, "mvm")
DEFAULT_BIN = os.path.join(MVM_DIR, "target", "release", "mvm-cli")
LOG_DIR = os.path.join(MVM_DIR, "logs")

# Scenario name → (what to do, what good looks like). Reference for the operator.
SCENARIOS = {
    "single_sit":   ("Sit still, face camera ~1.5 m", "1 stable track, 0–1 id, emotion label stable, dist ~1.5 m"),
    "walk_near_far": ("Walk 0.5 m → 2.5 m → back", "dist_m monotonic with motion, track survives"),
    "leave_return": ("In frame → leave 3 s → return", "track expires then new id; identity re-locks to Вадим"),
    "occlusion":    ("Cover face briefly with hand", "track survives via Kalman inertia (id stable)"),
    "two_people":   ("Two people, then stand close together", "2 tracks; zone merges to size 2 + a relation when near"),
    "two_objects":  ("Hold an object near your face", "(phase 7 — needs generic detector; person-only for now)"),
    "unknown_face": ("A non-gallery person in frame", "identity = unknown / low conf, not Вадим"),
}


def capture_streaming(scenario, secs, binary):
    """Stream stdout line-by-line so a kill doesn't drop the buffer."""
    if not os.path.exists(binary):
        sys.exit(f"binary not found: {binary}\n  build it: (cd mvm && cargo build --release)")
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"cam_{scenario}.jsonl")
    todo, expect = SCENARIOS.get(scenario, ("(freeform)", "(no preset expectation)"))
    print(f"\n  ┌─ scenario: {scenario}")
    print(f"  │  do    : {todo}")
    print(f"  │  expect: {expect}")
    print(f"  └─ capturing {secs}s → {os.path.relpath(log_path, ROOT)}\n", file=sys.stderr)

    cmd = ["timeout", str(secs), binary, "--native", "--dump-json", "--dump-interval", "1"]
    with open(log_path, "w") as out:
        proc = subprocess.Popen(cmd, cwd=MVM_DIR, stdout=out,
                                stderr=subprocess.DEVNULL, text=True)
        proc.wait()
    return log_path


def analyse(log_path):
    frames = []
    for line in open(log_path):
        line = line.strip()
        if line.startswith("{"):
            try:
                frames.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    if not frames:
        sys.exit(f"no frames in {log_path} — was a face in view? camera busy?")

    n = len(frames)
    occupied = [f for f in frames if f.get("status") == "occupied"]
    last_fps = frames[-1].get("fps", 0.0)

    # Per-track aggregation
    track_emos = defaultdict(list)     # id → [labels]
    track_conf = defaultdict(list)     # id → [emotion_conf]
    track_dist = defaultdict(list)     # id → [dist_m]
    track_ident = defaultdict(list)    # id → [(identity, conf)]
    max_zone = 0
    rel_frames = 0
    rel_kinds = set()
    max_faces = 0

    for f in frames:
        faces = f.get("faces", [])
        max_faces = max(max_faces, len(faces))
        for fa in faces:
            tid = fa["id"]
            if fa.get("emotion") is not None:
                track_emos[tid].append(fa["emotion"])
            if fa.get("emotion_conf") is not None:
                track_conf[tid].append(fa["emotion_conf"])
            if fa.get("dist_m") is not None:
                track_dist[tid].append(fa["dist_m"])
            if fa.get("identity") is not None:
                track_ident[tid].append((fa["identity"], fa.get("identity_conf") or 0.0))
        for z in f.get("zones", []):
            max_zone = max(max_zone, len(z))
        rels = f.get("relations", [])
        if rels:
            rel_frames += 1
            rel_kinds.update(r["kind"] for r in rels)

    print(f"\n  ══ summary: {os.path.basename(log_path)} ══")
    print(f"  frames={n}  occupied={len(occupied)} ({100*len(occupied)//max(n,1)}%)  "
          f"final_fps={last_fps:.2f}  max_faces={max_faces}")
    print(f"  tracks seen: {sorted(track_emos.keys() | track_dist.keys())}  "
          f"(count={len(track_emos.keys() | track_dist.keys())})")

    for tid in sorted(track_emos.keys() | track_dist.keys()):
        emos = track_emos.get(tid, [])
        changes = sum(1 for a, b in zip(emos, emos[1:]) if a != b)
        labels = sorted(set(emos))
        conf = track_conf.get(tid, [])
        dist = track_dist.get(tid, [])
        idents = track_ident.get(tid, [])
        top_ident = max(set(i for i, _ in idents), key=[i for i, _ in idents].count) if idents else "—"
        id_conf = [c for i, c in idents if i == top_ident]
        print(f"\n  ▸ track {tid}")
        print(f"      identity : {top_ident}  conf[{min(id_conf):.2f}..{max(id_conf):.2f}]" if id_conf
              else f"      identity : {top_ident}")
        if emos:
            print(f"      emotion  : {labels}  label_changes={changes}/{len(emos)}  "
                  f"conf[mean={statistics.mean(conf):.2f} min={min(conf):.2f} max={max(conf):.2f}]")
        if dist:
            print(f"      distance : [{min(dist):.2f} .. {max(dist):.2f}] m  "
                  f"mean={statistics.mean(dist):.2f}")

    print(f"\n  scene-graph: max_zone_size={max_zone}  "
          f"frames_with_relations={rel_frames}/{n}  relation_kinds={sorted(rel_kinds) or '—'}")
    if max_faces < 2:
        print("  ⚠ only ≤1 entity seen → relational graph (6.4a) not exercised. Use two_people.")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="single_sit", help="scenario name (see SCENARIOS)")
    ap.add_argument("--secs", type=int, default=20, help="capture duration")
    ap.add_argument("--bin", default=DEFAULT_BIN, help="mvm-cli release binary")
    ap.add_argument("--replay", help="skip capture, analyse an existing log")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    args = ap.parse_args()

    if args.list:
        print("\n  scenarios:")
        for k, (todo, expect) in SCENARIOS.items():
            print(f"    {k:<14} do: {todo}\n    {'':<14} ok: {expect}\n")
        return

    if args.replay:
        analyse(args.replay)
        return

    log_path = capture_streaming(args.scenario, args.secs, args.bin)
    analyse(log_path)


if __name__ == "__main__":
    main()
