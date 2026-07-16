#!/usr/bin/env python3
"""Golden-scenario compiler (phase 15.4 / LJ4): declarative scenarios -> post-detector
injection streams with known-by-construction truth -> golden_scenarios.npz.

The simulator's honest boundary: we inject AFTER the pixel detectors (bbox + posture +
wrists + 3D), so the pixel layer is out of scope; everything above it — tracker ->
infer_activities -> Activity FSM -> anomaly -> journal — runs for real (lj3_tracker.py +
lj4_fusion.py, both bit-faithful ports).

Scenario groups (plan 15.4):
  G1 enter->sit->work   walk segment + sit-down transition + typing hold (kb contact)
  G2 stood-up->left     typing hold + stand-up + walk out of frame + empty tail
  G3 lay down           lie_down trials standalone (+ controlled_desc) — anomaly MUST stay quiet
  G4 fell               fall trials standalone — anomaly MUST fire
  G5 two met            two walkers approach, stop ~1 m apart, stand together
  G6 walk solo          one walker crossing (15.3 regression: no phantoms, no events)

Composition happens in WORLD 3D before projection (the LJ3 lesson): segments are chained
pelvis-continuous (single offset per segment, translation preserved) with a short linear
crossfade over the seam, then projected through the same virtual eye-level camera as
LJ0-LJ3 (mocap_to_views CAM_H / lj3 CAM_DIST+FOCAL) -> bbox continuity by construction.
Floor is percentile-robust (the LJ2 settling-frame bug; global min floats the trial).

Truth carried per frame: true person id, bbox, vis, kinematic posture (label_pose — the
biomechanics labeller, NOT the GRIP features), camera-frame pelvis (people_together
distance), world pelvis (E2 fall physics), wrist points (contact truth). Objects are
static synthetic furniture (keyboard/chair) placed from the composed pose itself.
Expected events + tolerances are emitted at compile time into the sidecar JSON — the
scenario file IS the ground truth the bench scores against.

Run:  ../venv/bin/python3 spike_pose/lj4_scenarios.py
"""
import os, sys, json, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mocap_to_views as mv
from lj3_trajectories import bbox_of

OUT = os.path.join(HERE, "data_probe", "golden_scenarios.npz")
FPS = 15                # production human-chain cadence — the regime under test
# Scenario camera: same eye-level pinhole family as LJ0-LJ3, but pushed back to 5 m.
# At the LJ3 rig distance (3 m) the FLOOR IS OUT OF FRAME (a lying pelvis projects to
# y~1.1 > 1.0): G3/G4 bodies would vanish exactly when they lie down, silencing the
# anomaly by geometry, not by logic. At 5 m the floor enters the view (~z>4.2 m).
CAM_DIST = 5.0
FOCAL = 1.2
TRIM_FLOOR = 3          # lj2_transitions: ignore BVH settling frames for the ground plane
CROSSFADE = 8           # seam smoothing frames (~0.5 s) between composed segments
POSTURE_CONF = 0.9      # injected classifier confidence (FLOOR; STRESS degrades it)
KB_HALF = (0.06, 0.025) # synthetic keyboard bbox half-extent around the typing wrists
ANOMALY_GATE = 0.5      # scenario contract: quiet scenes stay below, a fall must exceed

# Trial casting (all reachable in mocap_bvh/ cache; fetch() pulls any missing).
# 13_xx/113_15/15_10 open standing -> clean sit-down onsets (G1); 111_xx/114_xx open
# seated -> native stand-up onsets (G2, no time-reversal needed).
G1_PAIRS = [("07_01", "13_01"), ("07_02", "113_15"), ("08_01", "15_10")]
G2_TRIPLES = [("111_09", "07_03"), ("111_10", "08_02"), ("114_04", "12_01")]  # (stand-up, walk-out)
G3_LIE = ["111_06", "111_07", "111_08", "111_12", "113_08", "114_16", "139_16"]
G3_DESC = ["01_08", "01_12", "82_05"]
G4_FALL = ["85_15", "88_04", "90_12", "90_13", "90_16", "90_18"]
G5_PAIRS = [("07_01", "08_02"), ("35_02", "16_15")]
G6_WALK = ["07_06", "07_07", "08_03", "12_02"]

TYPE_HOLD_S = 12.0      # G1 typing phase length
G2_TYPE_S = 8.0         # G2 leading typing phase
G5_HOLD_S = 8.0         # G5 standing-together phase
G5_STOP_DIST = 1.0      # stop when pelvises this close (< ACTIVITY_NEAR_M = 1.5)
G2_TAIL_S = 4.0         # empty frames after the person left


# ── trial -> world 3D (translation preserved, robust floor) ────────────────────

_CACHE = {}


def trial_3d(trial):
    """FK @FPS, SINGLE global xz offset (translation preserved — the lj3 fix), robust
    percentile floor (the lj2 settling-frame fix). Returns (P (N,13,3), walk_axis)."""
    if trial in _CACHE:
        return _CACHE[trial]
    joints, data, dt = mv.parse_bvh(mv.fetch(trial))
    scale = mv.bind_scale(joints)
    step = max(1, round((1.0 / dt) / FPS))
    # Drop the leading settling/bind frames entirely (the LJ2 lesson, taken further):
    # frame 0 is often a T-pose artifact that labels as a phantom posture and snaps the
    # bbox — on a lie-down trial it injects a fake standing->lying edge at t=0, which
    # would hand the anomaly a fall that never happened.
    rows = list(range(TRIM_FLOOR * step, len(data), step))
    P = np.zeros((len(rows), 13, 3), np.float32)
    for n, r in enumerate(rows):
        p = mv.fk(joints, data[r])
        P[n] = np.array([p[b] for b in mv.BVH_NAMES]) * scale
    hx, hz = P[:, 7:9, 0].mean(1), P[:, 7:9, 2].mean(1)
    walk_axis = "Z" if np.ptp(hz) >= np.ptp(hx) else "X"
    P[:, :, 0] -= hx.mean()
    P[:, :, 2] -= hz.mean()                                   # single global offset
    ymin = P[:, :, 1].min(axis=1)
    lo, hi = TRIM_FLOOR, max(TRIM_FLOOR + 1, len(ymin) - TRIM_FLOOR)
    P[:, :, 1] -= float(np.percentile(ymin[lo:hi], 2))        # glitch-robust floor
    _CACHE[trial] = (P, walk_axis)
    return _CACHE[trial]


def proj(P3D, az=0.0):
    """(13,3) world (Y-up) -> (13,2) normalized image coords, y DOWN — the lj3
    pinhole (CAM_H eye-level) at this bench's CAM_DIST. az kept for signature compat;
    rotation happens in world space before projection here."""
    P = rot_y(P3D, az) if az else P3D
    Pw = P + np.array([0, 0, CAM_DIST], np.float32)
    d = Pw[:, 2].clip(0.2, None)
    xn = 0.5 + FOCAL * Pw[:, 0] / d
    yn = 0.5 - FOCAL * (Pw[:, 1] - mv.CAM_H) / d
    return np.stack([xn, yn], 1)


def rot_y(P, az_deg):
    t = np.deg2rad(az_deg)
    c, s = np.cos(t), np.sin(t)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], np.float32)
    return P @ Ry.T


def pelvis_xz(P):
    """(N,13,3) or (13,3) -> pelvis (x,z)."""
    hip = P[..., 7:9, :].mean(-2)
    return hip[..., [0, 2]]


def retarget(P, target_xz, at="first"):
    """Shift the whole segment so its first (or last) pelvis lands on target_xz."""
    ref = pelvis_xz(P[0] if at == "first" else P[-1])
    Q = P.copy()
    Q[:, :, 0] += target_xz[0] - ref[0]
    Q[:, :, 2] += target_xz[1] - ref[1]
    return Q


def crossfade(A, B, k=CROSSFADE):
    """Concat A + B with k linearly interpolated seam frames (pose-space blend)."""
    if k <= 0:
        return np.concatenate([A, B])
    w = np.linspace(0, 1, k + 2)[1:-1].reshape(-1, 1, 1).astype(np.float32)
    seam = (1 - w) * A[-1][None] + w * B[0][None]
    return np.concatenate([A, seam, B])


def hold(frame, seconds):
    return np.repeat(frame[None], int(round(seconds * FPS)), axis=0)


def lateral(trial):
    """Walk trial oriented to sweep ACROSS the frame (the tracker's worst case)."""
    P, axis = trial_3d(trial)
    return rot_y(P, 90.0) if axis == "Z" else P.copy()


def labels_of(P):
    return [mv.label_pose(P[n]) for n in range(len(P))]


def find_onset(labels, frm, to, min_pre=1.0, min_post=2.0):
    """First index where a dwell-clean `frm`->`to` transition happens: >=min_pre s of
    `frm` before, >=min_post s of `to` from the onset on (1-frame blips tolerated by
    majority vote inside the windows). None if absent."""
    pre_n, post_n = int(min_pre * FPS), int(min_post * FPS)
    for i in range(pre_n, len(labels) - post_n):
        if labels[i] != to or labels[i - 1] == to:
            continue
        pre = labels[i - pre_n:i]
        post = labels[i:i + post_n]
        if (sum(l == frm for l in pre) >= 0.8 * pre_n
                and sum(l == to for l in post) >= 0.8 * post_n):
            return i
    return None


# ── frame emission ─────────────────────────────────────────────────────────────

def emit_person(P, person=0, t0=0.0, wrist_override=None):
    """Project a composed 3D segment -> per-frame person rows.
    wrist_override: {frame_range: (w1, w2)} normalized wrist points (typing truth)."""
    labels = labels_of(P)
    rows = []
    for n in range(len(P)):
        lm = proj(P[n], 0.0)                    # rotation already applied in world space
        bb = bbox_of(lm)
        cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
        vis = bool(0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0)
        wr = np.array([lm[5], lm[6]], np.float32)          # GRIP L/R wrist, projected
        if wrist_override is not None:
            for (a, b), pts in wrist_override.items():
                if a <= n < b:
                    wr = np.array(pts, np.float32)
        pel_w = P[n, 7:9, :].mean(0)                        # world pelvis (E2 truth)
        pos_c = pel_w + np.array([0, 0, CAM_DIST], np.float32)      # camera frame
        rows.append(dict(frame=n, t=t0 + n / FPS, person=person, bbox=bb, vis=vis,
                         posture=labels[n], wrists=wr, pelvis3d=pel_w.astype(np.float32),
                         pos3d=pos_c.astype(np.float32)))
    return rows


CONTACT_MARGIN = 0.03   # fusion.rs:195 — the contact predicate the injection truth mirrors
EXIT_BRIDGE = 5         # runs closer than the FSM exit dwell merge into one episode


def contact_runs(rows, bb, min_frames=3):
    """Expected activity episodes from the INJECTED wrists themselves: frames where any
    wrist lies within bb+CONTACT_MARGIN, gaps <= EXIT_BRIDGE bridged (cooling), runs
    shorter than min_frames dropped (ACTIVITY_ENTER_DWELL = 3 consecutive frames is
    exactly what confirms an instance — a 3-frame wrist swing through the keyboard IS
    an episode to the FSM, honestly). The lesson of the first
    FLOOR run: truth must be derived from what the data contains, not from the phase the
    author intended — a walking-in person's wrists hang at seated-keyboard height and 2D
    contact (prod semantics, no depth) honestly starts before the sit."""
    touch = []
    for r in rows:
        w = r["wrists"]
        hit = any(bb[0] - CONTACT_MARGIN <= x <= bb[2] + CONTACT_MARGIN
                  and bb[1] - CONTACT_MARGIN <= y <= bb[3] + CONTACT_MARGIN
                  for x, y in w)
        touch.append((r["frame"], hit))
    touch.sort()
    runs, cur = [], None
    for f, hit in touch:
        if hit:
            if cur is None:
                cur = [f, f]
            elif f - cur[1] <= EXIT_BRIDGE:
                cur[1] = f
            else:
                runs.append(cur)
                cur = [f, f]
    if cur is not None:
        runs.append(cur)
    return [(a / FPS, (b + 1) / FPS) for a, b in runs if b - a + 1 >= min_frames]


def kb_from_pose(P_frame):
    """Synthetic keyboard bbox centred on the projected wrists of a (seated) pose,
    plus the two wrist contact points inside it."""
    lm = proj(P_frame, 0.0)
    c = (lm[5] + lm[6]) / 2
    bb = np.array([c[0] - KB_HALF[0], c[1] - KB_HALF[1],
                   c[0] + KB_HALF[0], c[1] + KB_HALF[1]], np.float32)
    w1 = (c[0] - KB_HALF[0] * 0.5, c[1])
    w2 = (c[0] + KB_HALF[0] * 0.5, c[1])
    return bb, (w1, w2)


def chair_from_pose(P_frame):
    """Context furniture under the seated hips (must yield NO activity — honesty probe)."""
    lm = proj(P_frame, 0.0)
    hip = (lm[7] + lm[8]) / 2
    return np.array([hip[0] - 0.06, hip[1] - 0.02, hip[0] + 0.06, hip[1] + 0.12], np.float32)


# ── scenario builders ──────────────────────────────────────────────────────────

def build_g1(walk, sit):
    """enter -> sit -> work: walk to centre + sit-down + typing hold."""
    W = lateral(walk)
    cx = pelvis_xz(W)[:, 0]
    W = W[:int(np.argmin(np.abs(cx))) + 1]                   # stop at frame centre
    S_full, _ = trial_3d(sit)
    labels = labels_of(S_full)
    on = find_onset(labels, "standing", "sitting")
    if on is None:
        raise ValueError(f"{sit}: no clean standing->sitting onset")
    # End the segment on a frame that IS sitting (a short sitting run may end before
    # onset+3s and hand the typing hold a standing pose — the 15_10 bug).
    end = min(on + int(3.0 * FPS), len(S_full))
    while end > on + 1 and labels[end - 1] != "sitting":
        end -= 1
    S = S_full[max(0, on - int(1.0 * FPS)):end]
    S = retarget(S, pelvis_xz(W[-1]))
    body = crossfade(W, S)
    seat_pose = S[-1]
    T = hold(seat_pose, TYPE_HOLD_S)
    P = crossfade(body, T, k=0)                              # hold continues the same pose
    type_a = len(body)
    kb_bb, wrists = kb_from_pose(seat_pose)
    rows = emit_person(P, wrist_override={(type_a, len(P)): wrists})
    objects = [dict(oid=10001, cls="keyboard", bbox=kb_bb),
               dict(oid=10002, cls="chair", bbox=chair_from_pose(seat_pose))]
    expected = [
        dict(kind="activity", label="typing", person=0, obj=10001,
             t_start=a, t_end=b, tol_start=0.5, tol_end=1.0)
        for a, b in contact_runs(rows, kb_bb)
    ] + [
        dict(kind="anomaly_quiet", max=ANOMALY_GATE),
        dict(kind="tracks", person=0, n=1),
    ]
    return rows, objects, expected, len(P)


def build_g2(sit_stand, walk):
    """stood up -> left: typing hold + stand-up + walk out + empty tail."""
    S_full, _ = trial_3d(sit_stand)
    labels = labels_of(S_full)
    on = find_onset(labels, "sitting", "standing")
    reversed_seg = False
    if on is None:                                           # no stand-up in trial -> reverse
        on = find_onset(labels, "standing", "sitting")
        if on is None:
            raise ValueError(f"{sit_stand}: no usable transition")
        S_full = S_full[::-1].copy()
        labels = labels[::-1]
        on = len(labels) - on
        reversed_seg = True
    U = S_full[max(0, on - int(1.0 * FPS)):on + int(2.0 * FPS)]
    seat_pose = U[0]
    T = hold(seat_pose, G2_TYPE_S)
    W = lateral(walk)
    cx = pelvis_xz(W)[:, 0]
    W = W[int(np.argmin(np.abs(cx))):]                       # centre -> off-frame
    W = retarget(W, pelvis_xz(U[-1]))
    P = crossfade(np.concatenate([T, U]), W)
    stand_at = len(T) + (len(U) - int(2.0 * FPS))            # ~the onset inside U
    kb_bb, wrists = kb_from_pose(seat_pose)
    rows = emit_person(P, wrist_override={(0, len(T)): wrists})
    n_frames = len(P) + int(G2_TAIL_S * FPS)                 # empty tail after leaving
    objects = [dict(oid=10001, cls="keyboard", bbox=kb_bb),
               dict(oid=10002, cls="chair", bbox=chair_from_pose(seat_pose))]
    expected = [
        dict(kind="activity", label="typing", person=0, obj=10001,
             t_start=a, t_end=b, tol_start=0.5, tol_end=1.0)
        for a, b in contact_runs(rows, kb_bb)
    ] + [
        dict(kind="anomaly_quiet", max=ANOMALY_GATE),
        dict(kind="tracks", person=0, n=1),
        dict(kind="gone", person=0, t_by=(len(P)) / FPS + 1.0),
    ]
    meta = dict(reversed_standup=reversed_seg, stand_onset_t=stand_at / FPS)
    return rows, objects, expected, n_frames, meta


def build_solo(trial, group):
    """G3/G4/G6: a single trial standalone. ALL solo trials are oriented so their
    dominant translation axis is lateral — z-excursions toward the camera inflate
    the projection (magnification ∝ 1/z) and fake bbox jumps that aren't kinematics."""
    P = lateral(trial)
    rows = emit_person(P)
    lab = [r["posture"] for r in rows]
    expected = [dict(kind="tracks", person=0, n=1)]
    if group == "G4":
        lying = [i for i, l in enumerate(lab) if l == "lying"]
        onset_t = (lying[0] / FPS) if lying else None
        expected.append(dict(kind="anomaly_fire", min=ANOMALY_GATE,
                             t_from=(onset_t - 1.0) if onset_t else 0.0,
                             t_to=(onset_t + 3.0) if onset_t else len(P) / FPS))
    else:
        expected.append(dict(kind="anomaly_quiet", max=ANOMALY_GATE))
    return rows, [], expected, len(P)


def build_g5(a, b):
    """two met: walkers approach from opposite sides, stop ~1 m apart, stand together."""
    A = lateral(a)
    B = rot_y(lateral(b), 180.0)                             # opposite sweep
    # Place mirrored starts: A comes from x<0, B from x>0.
    A = retarget(A, (-abs(pelvis_xz(A[0])[0]) - 0.0, 0.0), at="first")
    B = retarget(B, (+abs(pelvis_xz(A[0])[0]), 0.3), at="first")   # slight z offset
    n = min(len(A), len(B))
    A, B = A[:n], B[:n]
    dist = np.linalg.norm(pelvis_xz(A) - pelvis_xz(B), axis=1)
    stop = int(np.argmax(dist < G5_STOP_DIST)) if (dist < G5_STOP_DIST).any() else n - 1
    A = np.concatenate([A[:stop + 1], hold(A[stop], G5_HOLD_S)])
    B = np.concatenate([B[:stop + 1], hold(B[stop], G5_HOLD_S)])
    rows = emit_person(A, person=0) + emit_person(B, person=1)
    # people_together truth from injected 3D (< ACTIVITY_NEAR_M in *3D*, incl. height).
    pa = {r["frame"]: r["pos3d"] for r in rows if r["person"] == 0}
    pb = {r["frame"]: r["pos3d"] for r in rows if r["person"] == 1}
    near = [f for f in sorted(pa) if f in pb
            and np.linalg.norm(np.asarray(pa[f]) - np.asarray(pb[f])) < 1.5]
    expected = [dict(kind="tracks", person=0, n=1), dict(kind="tracks", person=1, n=1),
                dict(kind="anomaly_quiet", max=ANOMALY_GATE), dict(kind="no_swap")]
    if near:
        expected.append(dict(kind="activity", label="people_together",
                             person=0, person2=1,
                             t_start=near[0] / FPS, t_end=(near[-1] + 1) / FPS,
                             tol_start=1.0, tol_end=1.5))
    n_frames = int(max(r["frame"] for r in rows)) + 1
    return rows, [], expected, n_frames


# ── CI export (the Rust --stub-class golden test reads THIS file) ─────────────

CI_OUT = os.path.join(os.path.dirname(HERE), "mvm", "crates", "mvm-cli",
                      "tests", "data", "golden_scenarios.json")
# One representative per group, FLOOR-clean. G3_111_07 is a prod-quiet floor
# scenario (the three prod false-alarm instances are documented in LJ4 and flip
# with the E2 port in 15.6 — CI asserts what production promises TODAY).
CI_INSTANCES = ["G1_07_01__13_01", "G2_111_09__07_03", "G3_111_07",
                "G4_90_12", "G5_07_01__08_02", "G6_07_06"]


def export_ci(R, meta):
    """Compact FLOOR-only JSON (visible rows, rounded) for the Rust golden test —
    scenario file = single source of truth for the Python bench AND the CI gate."""
    idx = collections.defaultdict(list)
    for i, inst in enumerate(R["inst"]):
        idx[inst].append(i)
    out = {"fps": FPS, "instances": []}
    for inst in CI_INSTANCES:
        m = meta[inst]
        frames = [[] for _ in range(m["n_frames"])]
        for i in idx[inst]:
            if not R["vis"][i]:
                continue                      # FLOOR emitter skips invisible rows
            r4 = lambda v: [round(float(x), 4) for x in np.asarray(v).ravel()]
            frames[int(R["frame"][i])].append(dict(
                person=int(R["person"][i]), bbox=r4(R["bbox"][i]),
                posture=str(R["posture"][i]),
                wrists=[r4(R["wrists"][i][0]), r4(R["wrists"][i][1])],
                pos3d=r4(R["pos3d"][i])))
        objs = [dict(id=int(R["obj_id"][j]), cls=str(R["obj_cls"][j]),
                     bbox=[round(float(x), 4) for x in R["obj_bbox"][j]])
                for j, oi in enumerate(R["obj_inst"]) if oi == inst]
        out["instances"].append(dict(
            id=inst, group=m["group"], n_frames=m["n_frames"],
            posture_conf=m["posture_conf"], objects=objs, frames=frames,
            expected=m["expected"]))
    os.makedirs(os.path.dirname(CI_OUT), exist_ok=True)
    with open(CI_OUT, "w") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"CI export: {CI_OUT} ({os.path.getsize(CI_OUT)//1024} KB, "
          f"{len(out['instances'])} instances)")


# ── driver ─────────────────────────────────────────────────────────────────────

def main():
    R = collections.defaultdict(list)
    meta = {}

    def add(inst, group, rows, objects, expected, n_frames, extra=None):
        for r in rows:
            R["inst"].append(inst); R["group"].append(group)
            R["frame"].append(r["frame"]); R["t"].append(r["t"])
            R["person"].append(r["person"]); R["bbox"].append(r["bbox"])
            R["vis"].append(r["vis"]); R["posture"].append(r["posture"])
            R["wrists"].append(r["wrists"]); R["pelvis3d"].append(r["pelvis3d"])
            R["pos3d"].append(r["pos3d"])
        for o in objects:
            R["obj_inst"].append(inst); R["obj_id"].append(o["oid"])
            R["obj_cls"].append(o["cls"]); R["obj_bbox"].append(o["bbox"])
        meta[inst] = dict(group=group, fps=FPS, n_frames=int(n_frames),
                          posture_conf=POSTURE_CONF, expected=expected,
                          **(extra or {}))
        lab = collections.Counter(r["posture"] for r in rows)
        print(f"{inst:24} {group}  {n_frames:4d} fr ({n_frames/FPS:5.1f} s)  "
              f"persons={len(set(r['person'] for r in rows))} objs={len(objects)}  "
              f"postures={dict(lab)}  expected={len(expected)}")

    for walk, sit in G1_PAIRS:
        rows, objs, exp, n = build_g1(walk, sit)
        add(f"G1_{walk}__{sit}", "G1", rows, objs, exp, n)
    for sit_stand, walk in G2_TRIPLES:
        rows, objs, exp, n, extra = build_g2(sit_stand, walk)
        add(f"G2_{sit_stand}__{walk}", "G2", rows, objs, exp, n, extra)
    for trial in G3_LIE:
        rows, objs, exp, n = build_solo(trial, "G3")
        add(f"G3_{trial}", "G3", rows, objs, exp, n)
    for trial in G3_DESC:
        rows, objs, exp, n = build_solo(trial, "G3")
        add(f"G3d_{trial}", "G3", rows, objs, exp, n)
    for trial in G4_FALL:
        rows, objs, exp, n = build_solo(trial, "G4")
        add(f"G4_{trial}", "G4", rows, objs, exp, n)
    for a, b in G5_PAIRS:
        rows, objs, exp, n = build_g5(a, b)
        add(f"G5_{a}__{b}", "G5", rows, objs, exp, n)
    for trial in G6_WALK:
        rows, objs, exp, n = build_solo(trial, "G6")
        add(f"G6_{trial}", "G6", rows, objs, exp, n)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(
        OUT,
        inst=np.array(R["inst"]), group=np.array(R["group"]),
        frame=np.array(R["frame"], np.int32), t=np.array(R["t"], np.float32),
        person=np.array(R["person"], np.int8), bbox=np.array(R["bbox"], np.float32),
        vis=np.array(R["vis"], bool), posture=np.array(R["posture"]),
        wrists=np.array(R["wrists"], np.float32),
        pelvis3d=np.array(R["pelvis3d"], np.float32),
        pos3d=np.array(R["pos3d"], np.float32),
        obj_inst=np.array(R["obj_inst"]), obj_id=np.array(R["obj_id"], np.int32),
        obj_cls=np.array(R["obj_cls"]), obj_bbox=np.array(R["obj_bbox"], np.float32),
        meta_json=np.array(json.dumps(meta)))
    groups = collections.Counter(m["group"] for m in meta.values())
    print(f"\nsaved {OUT}: {len(R['inst'])} person-rows, {len(meta)} instances, "
          f"groups={dict(groups)}")
    export_ci(R, meta)


if __name__ == "__main__":
    main()
