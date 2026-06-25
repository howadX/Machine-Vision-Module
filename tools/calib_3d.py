#!/usr/bin/env python3
"""3D depth calibration helper (phase 6.2.0).

Collects face bbox heights at a *known* tape-measured distance, then back-solves
the effective ``face_height_m`` prior for the pinhole depth model:

    z = fx * real_size / pixel_size   →   real_eff = z_true * pixel_size / fx

Because the camera is ceiling-mounted (~45° down), the face bbox is foreshortened,
so the effective face height is far below the frontal 0.19 m population prior.
Run this at several distances; if real_eff is ~constant across them, a single
``face_height_m`` constant suffices. If it drifts with distance, the tilt
introduces a non-linearity that one constant cannot capture (decision point).

Usage (run from the project root — the Rust binary spawns the Python bridge
via ../venv relative paths, so we launch it with cwd=mvm/):

    venv/bin/python3 tools/calib_3d.py --distance 1.0 --frames 12
    venv/bin/python3 tools/calib_3d.py --distance 0.5
    ...
    venv/bin/python3 tools/calib_3d.py --report      # summarise the CSV

Stand a SINGLE face in view at the tape-measured distance, hold still, then run.
Each invocation appends one median row to models/calib_3d.csv.
"""

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MVM_DIR = os.path.join(ROOT, "mvm")
DEFAULT_BIN = os.path.join(MVM_DIR, "target", "release", "mvm-cli")


def csv_path(native_camera):
    """Mode-specific CSV so YuNet rows never mix with the historical SCRFD sweep
    (which back-solved 0.44). Mixing them would corrupt the recommended mean."""
    name = "calib_3d_yunet.csv" if native_camera else "calib_3d.csv"
    return os.path.join(ROOT, "models", name)

# Intrinsics — must match mvm-core Config defaults.
FX_RATIO = 0.64
WIDTH = 640
HEIGHT = 480
FX = FX_RATIO * WIDTH  # 409.6 px


def collect(distance, frames, binary, timeout, native_camera):
    """Run the native pipeline and gather median bbox_h for the closest face."""
    if not os.path.exists(binary):
        sys.exit(f"binary not found: {binary}\n  build it: (cd mvm && cargo build --release)")

    # --native-camera recalibrates for YuNet boxes (phase 7.1); --native uses the
    # Python-bridge SCRFD boxes. The back-solve below is identical — only the box
    # source differs, which is exactly what we are calibrating for.
    mode = "--native-camera" if native_camera else "--native"
    cmd = [binary, mode, "--dump-json", "--dump-interval", "1"]
    print(f"→ collecting {frames} samples at {distance} m ... (hold still)", file=sys.stderr)
    proc = subprocess.Popen(
        cmd, cwd=MVM_DIR, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
    )

    bbox_hs, native_dists = [], []
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            faces = frame.get("faces", [])
            if not faces:
                print("  · frame with no face — skipped", file=sys.stderr)
                continue
            # closest face = largest bbox height
            face = max(faces, key=lambda f: f.get("bbox_h", 0.0))
            bh = face.get("bbox_h", 0.0)
            if bh <= 0.0:
                continue
            bbox_hs.append(bh)
            nd = face.get("dist_m")
            if nd is not None:
                native_dists.append(nd)
            print(f"  · sample {len(bbox_hs)}/{frames}: bbox_h={bh:.4f} "
                  f"({bh * HEIGHT:.1f}px) native_dist={nd}", file=sys.stderr)
            if len(bbox_hs) >= frames:
                break
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()

    if not bbox_hs:
        sys.exit("no face samples collected — check the camera / lighting / framing")

    med_bh = statistics.median(bbox_hs)
    med_px = med_bh * HEIGHT
    real_eff = distance * med_px / FX
    med_native = statistics.median(native_dists) if native_dists else None

    row = {
        "distance_m": round(distance, 3),
        "n": len(bbox_hs),
        "bbox_h_norm": round(med_bh, 5),
        "bbox_h_px": round(med_px, 2),
        "real_eff_m": round(real_eff, 4),
        "native_dist_m": round(med_native, 3) if med_native is not None else "",
    }

    out_csv = csv_path(native_camera)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    new_file = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)

    print(f"\n  distance       : {distance} m  (tape-measured ground truth)")
    print(f"  median bbox_h  : {med_bh:.5f} norm  = {med_px:.1f} px")
    print(f"  effective real : {real_eff:.4f} m  ← back-solved face_height_m")
    if med_native is not None:
        print(f"  native dist_m  : {med_native:.3f} m  (current prior 0.19 → off by "
              f"{med_native / distance:.2f}×)")
    print(f"  appended → {out_csv}")


def report(native_camera):
    """Summarise the calibration CSV and recommend a constant."""
    path = csv_path(native_camera)
    if not os.path.exists(path):
        sys.exit(f"no data yet: {path}")
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit("CSV is empty")

    rows.sort(key=lambda r: float(r["distance_m"]))
    print(f"\n  Calibration data ({len(rows)} points) — {path}\n")
    print(f"  {'dist_m':>7} {'bbox_px':>9} {'real_eff_m':>11} {'native_m':>9}")
    print(f"  {'-'*7} {'-'*9} {'-'*11} {'-'*9}")
    reals = []
    for r in rows:
        reals.append(float(r["real_eff_m"]))
        nd = r["native_dist_m"] or "—"
        print(f"  {float(r['distance_m']):>7.2f} {float(r['bbox_h_px']):>9.1f} "
              f"{float(r['real_eff_m']):>11.4f} {nd:>9}")

    mean_real = statistics.mean(reals)
    spread = (max(reals) - min(reals)) / mean_real if mean_real else 0.0
    print(f"\n  mean real_eff  : {mean_real:.4f} m")
    print(f"  spread         : {spread * 100:.1f} % of mean")
    if spread < 0.15:
        print(f"  → CONSISTENT: set the constant = {mean_real:.4f}")
        print(f"    · YuNet path (--native-camera): Config.face_height_m_yunet")
        print(f"    · bridge path (--native):       Config.face_height_m")
    else:
        print(f"  → DRIFTS with distance ({spread*100:.0f}%): a single constant is")
        print(f"    insufficient — tilt introduces non-linearity. Decision needed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distance", type=float, help="tape-measured distance to face (m)")
    ap.add_argument("--frames", type=int, default=12, help="samples to collect (default 12)")
    ap.add_argument("--bin", default=DEFAULT_BIN, help="path to mvm-cli release binary")
    ap.add_argument("--timeout", type=float, default=5.0, help="process kill timeout (s)")
    ap.add_argument("--native-camera", action="store_true",
                    help="calibrate the pure-Rust YuNet path (--native-camera) — "
                         "the constant goes into Config.face_height_m_yunet")
    ap.add_argument("--report", action="store_true", help="summarise the CSV and exit")
    args = ap.parse_args()

    if args.report:
        report(args.native_camera)
        return
    if args.distance is None:
        ap.error("--distance is required (or use --report)")
    collect(args.distance, args.frames, args.bin, args.timeout, args.native_camera)


if __name__ == "__main__":
    main()
