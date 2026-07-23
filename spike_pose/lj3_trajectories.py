#!/usr/bin/env python3
"""CMU walk mocap -> WORLD-translation-preserving bbox trajectories -> mocap_tracks.npz.

Phase 15.3 lab tool (tracker on true trajectories). The point: reproduce the live
"walking fragmentation" bug (one walker splits into up to 7 phantom track ids) on an
honest offline bench with GROUND-TRUTH identity, so the fix is measured, not eyeballed.

Why a NEW extractor (not mocap_transitions.npz): mocap_to_views.py:229 re-centers the
pelvis at the origin EVERY frame (`P[:,:,[0,2]] -= hip.mean`), which annihilates the
walking translation (verified: walk trials give pelvis x/z range = 0.000 m). A tracker
bench needs bbox trajectories that actually move across the frame -- that motion is what
fragments the tracker. Here we apply a SINGLE global offset (center the whole trajectory),
preserving every inter-frame displacement.

Pipeline (reuses FK/bind_scale from mocap_to_views):
  BVH -> FK -> 13 GRIP joints -> metric scale
      -> global offset (mean hip x,z -> 0; feet -> y~0 once)  [translation PRESERVED]
      -> orient the walk LATERALLY: if the trial walks along the camera axis (z), rotate
         by azimuth 90 deg so it sweeps across the frame (worst case for the IoU gate)
      -> virtual pinhole camera (175 cm eye-level, pushed back so a person ~0.36 frame-h)
      -> bbox = min/max of the 13 projected joints + anatomical padding
  Composite crossings: two single walks, opposite azimuth -> they cross mid-frame
  (mutual occlusion -> Hungarian + re-link stress; ground-truth swap test for T2).

Artifact mocap_tracks.npz (flat, grouped by `seq`):
  seq     (M,)   str    sequence id  ("07_01" | "idle_111_28" | "cross_07_01__08_02")
  kind    (M,)   str    "walk" | "idle" | "cross"
  frame   (M,)   int32  frame index within the sequence (@15 fps)
  person  (M,)   int8   TRUE identity within the sequence (0 or 1)  <- the ground truth
  bbox    (M,4)  f32    image-normalized [x1,y1,x2,y2] (may exceed [0,1] near frame edges)
  vis     (M,)   bool   bbox center within the frame (a detector could plausibly see it)
  t       (M,)   f32    time (s) within the sequence
  pelvis3d(M,3)  f32    world metric pelvis (m) -> real speed/accel for Kalman-Q calibration

Run:  ../venv/bin/python3 spike_pose/lj3_trajectories.py
"""
import os, sys, numpy as np, collections

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mocap_to_views import parse_bvh, fk, bind_scale, BVH_NAMES, GRIP_MP_IDX, fetch, CAM_H

OUT = os.path.join(HERE, "data_probe", "mocap_tracks.npz")
FPS_OUT = 15            # production human-chain cadence (the regime the live bug lives in)
CAM_DIST = 3.0          # eye-level rig distance (~live 2.6-3 m) -> person ~0.58 of frame height
FOCAL = 1.2
MIN_WALK_M = 1.5        # trial must translate >= this to count as a real walk (else skip/idle)
PAD_TOP = 0.10          # head sits above the topmost GRIP joint (nose); pad up
PAD_SIDE = 0.02         # slight lateral pad on the body core
# A real YOLO person box is TALL and NARROW (aspect ~0.4): width from the vertical body core
# (shoulders/hips/knees/ankles), NOT the swinging wrists/elbows, whose spread would give an
# unrealistically wide, near-square box that makes the lateral IoU gate trivially easy.
CORE_W = [1, 2, 7, 8, 9, 10, 11, 12]   # GRIP idx: Lsh Rsh Lhip Rhip Lkn Rkn Lank Rank

# Walk trials (cgspeed straight-walk subjects). Fetch failures are skipped, not fatal.
WALK = ["07_01", "07_02", "07_03", "07_06", "07_07", "07_08", "07_10",
        "08_01", "08_02", "08_03", "08_05", "08_08",
        "35_01", "35_02", "16_15", "16_16", "16_21", "16_22",
        "02_01", "02_02", "12_01", "12_02", "12_03", "39_01"]
# Standing-still trials (idle regression, T2): must NOT fragment (0 excess tracks).
IDLE = ["111_28", "113_21"]
# Composite crossings (T2 non-swap test): (left->right, right->left) pairs.
CROSS = [("07_01", "08_02"), ("35_02", "16_15")]


def proj(P3D, az):
    """(13,3) metric subject-centered (Y-up) -> (13,2) image coords, y DOWN, via a pinhole
    at CAM_H, subject rotated about vertical by `az` (so a depth-walk becomes lateral)."""
    t = np.deg2rad(az); c, s = np.cos(t), np.sin(t)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Pw = P3D @ Ry.T + np.array([0, 0, CAM_DIST])
    d = Pw[:, 2].clip(0.2, None)
    xn = 0.5 + FOCAL * Pw[:, 0] / d
    yn = 0.5 - FOCAL * (Pw[:, 1] - CAM_H) / d
    return np.stack([xn, yn], 1)


def bbox_of(lm2d):
    """13 projected joints -> realistic tall-narrow person bbox [x1,y1,x2,y2] (image-normalized).
    Width from the vertical body core (YOLO-like); height from head-to-ankle over all joints."""
    core = lm2d[CORE_W]
    x1, x2 = core[:, 0].min(), core[:, 0].max()
    y1, y2 = lm2d[:, 1].min(), lm2d[:, 1].max()
    return np.array([x1 - PAD_SIDE, y1 - PAD_TOP, x2 + PAD_SIDE, y2 + PAD_SIDE], np.float32)


def track_3d(trial):
    """FK a trial into metric 13-joint 3D @FPS_OUT with world translation PRESERVED (global
    offset only). Returns (P (N,13,3), t (N,), walk_axis 'X'|'Z', span_m)."""
    joints, data, dt = parse_bvh(fetch(trial))
    scale = bind_scale(joints)
    step = max(1, round((1.0 / dt) / FPS_OUT))
    rows = list(range(0, len(data), step))
    P = np.zeros((len(rows), 13, 3), np.float32)
    for n, r in enumerate(rows):
        p = fk(joints, data[r])
        P[n] = np.array([p[b] for b in BVH_NAMES]) * scale
    hx, hz = P[:, 7:9, 0].mean(1), P[:, 7:9, 2].mean(1)      # pelvis x,z per frame (world)
    span = max(np.ptp(hx), np.ptp(hz))
    walk_axis = "Z" if np.ptp(hz) >= np.ptp(hx) else "X"
    P[:, :, 0] -= hx.mean(); P[:, :, 2] -= hz.mean()         # SINGLE global offset
    P[:, :, 1] -= P[:, :, 1].min()                            # feet -> y~0 once
    t = np.array([n * step * dt for n in range(len(rows))], np.float32)
    return P, t, walk_axis, span


def emit_single(seq, P, t, az, kind, person=0):
    """Project one person's 3D track at azimuth `az` -> per-frame rows."""
    rows = []
    for n in range(len(P)):
        lm = proj(P[n], az)
        bb = bbox_of(lm)
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        vis = bool(0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)
        rows.append((seq, kind, n, person, bb, vis, float(t[n]), P[n, 7:9, :].mean(0)))
    return rows


def main():
    R = {k: [] for k in ("seq", "kind", "frame", "person", "bbox", "vis", "t", "pelvis3d")}

    def add(rows):
        for seq, kind, fr, person, bb, vis, tt, pel in rows:
            R["seq"].append(seq); R["kind"].append(kind); R["frame"].append(fr)
            R["person"].append(person); R["bbox"].append(bb); R["vis"].append(vis)
            R["t"].append(tt); R["pelvis3d"].append(pel)

    # ---- single walkers (oriented laterally = worst case for the IoU gate) ----
    axis_of = {}
    for trial in WALK:
        try:
            P, t, walk_axis, span = track_3d(trial)
        except Exception as e:
            print(f"  !! {trial}: {e}"); continue
        if span < MIN_WALK_M:
            print(f"  -- {trial}: span {span:.2f}m < {MIN_WALK_M} -> not a walk, skip"); continue
        az = 90.0 if walk_axis == "Z" else 0.0
        axis_of[trial] = (az, span)
        rows = emit_single(trial, P, t, az, "walk")
        add(rows)
        vis = np.array([r[5] for r in rows]); cx = np.array([(r[4][0] + r[4][2]) / 2 for r in rows])
        print(f"{trial:7} walk={walk_axis} az={az:.0f} span={span:.1f}m  {len(rows)}fr  "
              f"cx=[{cx.min():.2f},{cx.max():.2f}] inframe={vis.mean()*100:.0f}%")

    # ---- idle (standing still): must produce ONE stable id ----
    for trial in IDLE:
        try:
            P, t, _, span = track_3d(trial)
        except Exception as e:
            print(f"  !! idle {trial}: {e}"); continue
        add(emit_single(f"idle_{trial}", P, t, 0.0, "idle"))
        print(f"idle_{trial:7} span={span:.2f}m  {len(P)}fr")

    # ---- composite crossings: two walkers, opposite azimuth, cross mid-frame ----
    for a, b in CROSS:
        if a not in axis_of or b not in axis_of:
            print(f"  -- cross {a}+{b}: a source missing, skip"); continue
        Pa, ta, _, _ = track_3d(a); Pb, tb, _, _ = track_3d(b)
        seq = f"cross_{a}__{b}"
        add(emit_single(seq, Pa, ta, 90.0, "cross", person=0))    # left -> right
        add(emit_single(seq, Pb, tb, 270.0, "cross", person=1))   # right -> left
        print(f"{seq}  p0={len(Pa)}fr  p1={len(Pb)}fr  (opposite sweep -> mutual occlusion)")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT, seq=np.array(R["seq"]), kind=np.array(R["kind"]),
        frame=np.array(R["frame"], np.int32), person=np.array(R["person"], np.int8),
        bbox=np.array(R["bbox"], np.float32), vis=np.array(R["vis"], bool),
        t=np.array(R["t"], np.float32), pelvis3d=np.array(R["pelvis3d"], np.float32))
    seqs = collections.Counter(R["kind"])
    print(f"\nsaved {OUT}: {len(R['seq'])} rows, "
          f"{len(set(R['seq']))} sequences, kinds={dict(seqs)}")


if __name__ == "__main__":
    main()
