#!/usr/bin/env python3
"""Faithful Python port of MVM PersonTracker association (mvm-core: tracker.rs + kalman.rs).

Phase 15.3 lab bench. Same choice as LJ2's stabilizer port: run the production logic in
Python so we can sweep hyperparameters and bootstrap CIs cheaply. The Rust port of whatever
wins (plus a bit-exact golden test Python<->Rust) is deferred to 15.6 — there we also measure
the real Rust code on the live camera.

Ported byte-for-byte from the Rust (constants quoted at their definitions):
  - KalmanBBox: 4 independent scalar constant-velocity filters [pos, vel], F=[[1,1],[0,1]],
    H=[1,0]. Q_POS=1.0 Q_VEL=0.01 R_POS=1.0 R_SIZE=10.0 P_POS_INIT=10.0 P_VEL_INIT=100.0.
  - associate_core: predict-all -> missed+=1 -> Hungarian stage-1 (cost = -(iou + W_OCM*mom))
    gated by IOU_GATE=0.3 -> re-link cascade (cosine / label / center-distance) -> spawn ->
    ByteTrack stage-2 (low boxes) -> inertia advance -> expire at timeout.
  - lifecycle FSM next_lifecycle (reported only; with no masks it does not alter ids).

The tunable knobs the 15.3 experiments move are EXPOSED as tracker attributes (defaults =
production): iou_gate, q_pos, q_vel, r_pos, relink_center_dist, obs_vel_ema, and one structural
flag `center_relink_lowconf_id` (hypothesis A: let the geometric center re-link fire even when a
stale low-confidence identity is present, instead of skipping it).

Mask-IoU term (W_MASK) is inert here (no silhouettes in the bench) exactly as in prod on the
<2-person / no-seg path, so it is omitted. Note W_MASK is zero regardless in the mask-less path,
so `mask_cue_on` has no effect on id assignment — verified against the Rust gate.
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.optimize import linear_sum_assignment

# ── production constants (tracker.rs) ──────────────────────────────────────────
IOU_GATE = 0.3          # ln 69
W_OCM = 0.05            # ln 73
OBS_VEL_EMA = 0.5       # ln 83
DIR_EPS = 1e-4          # ln 85
CONFIRM_AGE = 3         # ln 104
LOST_AFTER_MISSED = 30  # ln 107
# kalman.rs
Q_POS, Q_VEL = 1.0, 0.01
R_POS, R_SIZE = 1.0, 10.0
P_POS_INIT, P_VEL_INIT = 10.0, 100.0
# config.rs defaults (re-link)
RELINK_WINDOW = 90
RELINK_MIN_CONF = 0.5
RELINK_COS = 0.5
RELINK_CENTER_DIST = 0.15
TIMEOUT_FRAMES = 1800


def iou(a, b):
    """a,b = [x1,y1,x2,y2]. Matches BBox::iou."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ocm_consistency(vel, delta):
    vlen = (vel[0] ** 2 + vel[1] ** 2) ** 0.5
    dlen = (delta[0] ** 2 + delta[1] ** 2) ** 0.5
    if vlen < DIR_EPS or dlen < DIR_EPS:
        return 0.0
    return (vel[0] * delta[0] + vel[1] * delta[1]) / (vlen * dlen)


class ScalarKF:
    __slots__ = ("pos", "vel", "p_pp", "p_pv", "p_vv", "q_pos", "q_vel", "r")

    def __init__(self, pos, q_pos, q_vel, r):
        self.pos = pos; self.vel = 0.0
        self.p_pp = P_POS_INIT; self.p_pv = 0.0; self.p_vv = P_VEL_INIT
        self.q_pos = q_pos; self.q_vel = q_vel; self.r = r

    def predict(self):
        self.pos += self.vel
        p_pp = self.p_pp + 2.0 * self.p_pv + self.p_vv + self.q_pos
        p_pv = self.p_pv + self.p_vv
        p_vv = self.p_vv + self.q_vel
        self.p_pp, self.p_pv, self.p_vv = p_pp, p_pv, p_vv
        return self.pos

    def update(self, observed):
        s = self.p_pp + self.r
        k_pos = self.p_pp / s
        k_vel = self.p_pv / s
        y = observed - self.pos
        self.pos += k_pos * y
        self.vel += k_vel * y
        p_pp = (1.0 - k_pos) * self.p_pp
        p_pv = (1.0 - k_pos) * self.p_pv
        p_vv = -k_vel * self.p_pv + self.p_vv
        self.p_pp, self.p_pv, self.p_vv = p_pp, p_pv, p_vv


def to_cwh(b):
    return ((b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5, b[2] - b[0], b[3] - b[1])


def from_cwh(cx, cy, w, h):
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


class KalmanBBox:
    def __init__(self, bbox, q_pos, q_vel, r_pos, r_size):
        cx, cy, w, h = to_cwh(bbox)
        self.cx = ScalarKF(cx, q_pos, q_vel, r_pos)
        self.cy = ScalarKF(cy, q_pos, q_vel, r_pos)
        self.w = ScalarKF(w, q_pos, q_vel, r_size)
        self.h = ScalarKF(h, q_pos, q_vel, r_size)

    def predict(self):
        return from_cwh(self.cx.predict(), self.cy.predict(),
                        max(0.0, self.w.predict()), max(0.0, self.h.predict()))

    def update(self, observed):
        cx, cy, w, h = to_cwh(observed)
        self.cx.update(cx); self.cy.update(cy); self.w.update(w); self.h.update(h)

    def smoothed_bbox(self):
        return from_cwh(self.cx.pos, self.cy.pos, max(0.0, self.w.pos), max(0.0, self.h.pos))


@dataclass
class Detection:
    bbox: list
    identity: Optional[str] = None
    identity_confidence: Optional[float] = None
    embedding: Optional[np.ndarray] = None


class Track:
    __slots__ = ("id", "age", "missed", "lifecycle", "kalman", "obs_velocity",
                 "last_obs_center", "identity", "bbox")

    def __init__(self, tid, bbox, kf):
        self.id = tid; self.age = 1; self.missed = 0
        self.lifecycle = "Tentative"; self.kalman = kf
        self.obs_velocity = None; self.last_obs_center = None
        self.identity = None; self.bbox = list(bbox)


def next_lifecycle(prev, age, missed, locked):
    if missed == 0:
        if age < CONFIRM_AGE:
            return "Tentative"
        return "IdLocked" if locked else "Confirmed"
    if prev in ("Tentative", "Lost"):
        return "Lost"
    if missed >= LOST_AFTER_MISSED:
        return "Lost"
    return "Occluded"


def _solve(cost):
    """Kuhn-Munkres minimising total cost; INF = forbidden, never chosen (a row with only
    forbidden options -> None). Mirrors mvm-core::assignment::solve via scipy + big-sentinel."""
    if not cost or not cost[0]:
        return [None] * len(cost)
    C = np.array(cost, float)
    finite = C[np.isfinite(C)]
    big = (finite.max() + 1.0) * (C.shape[0] * C.shape[1] + 1) if finite.size else 1.0
    Cf = np.where(np.isfinite(C), C, big)
    rows, cols = linear_sum_assignment(Cf)
    out = [None] * C.shape[0]
    for r, c in zip(rows, cols):
        if np.isfinite(C[r, c]):        # forbidden pairs (were INF) are dropped
            out[r] = int(c)
    return out


class PersonTracker:
    """Port of PersonTracker. Tunable knobs default to production values."""

    def __init__(self, iou_gate=IOU_GATE, q_pos=Q_POS, q_vel=Q_VEL, r_pos=R_POS, r_size=R_SIZE,
                 relink_center_dist=RELINK_CENTER_DIST, obs_vel_ema=OBS_VEL_EMA,
                 relink_window=RELINK_WINDOW, relink_min_conf=RELINK_MIN_CONF,
                 relink_cos=RELINK_COS, timeout=TIMEOUT_FRAMES,
                 center_relink_lowconf_id=False):
        self.tracks = {}
        self.next_id = 10          # ephemeral ids start at 10 (0-9 reserved for gallery)
        self.free_ids = []
        self.iou_gate = iou_gate
        self.q_pos, self.q_vel, self.r_pos, self.r_size = q_pos, q_vel, r_pos, r_size
        self.relink_center_dist = relink_center_dist
        self.obs_vel_ema = obs_vel_ema
        self.relink_window = relink_window
        self.relink_min_conf = relink_min_conf
        self.relink_cos = relink_cos
        self.timeout = timeout
        self.center_relink_lowconf_id = center_relink_lowconf_id  # hypothesis-A structural fix

    def _alloc_id(self):
        if self.free_ids:
            return self.free_ids.pop(0)
        i = self.next_id; self.next_id += 1
        return i

    def _record_obs(self, t, obs_c):
        if t.last_obs_center is not None:
            gap = max(1, t.missed)
            inst = [(obs_c[0] - t.last_obs_center[0]) / gap,
                    (obs_c[1] - t.last_obs_center[1]) / gap]
            if t.obs_velocity is None:
                t.obs_velocity = inst
            else:
                e = self.obs_vel_ema
                t.obs_velocity = [e * inst[0] + (1 - e) * t.obs_velocity[0],
                                  e * inst[1] + (1 - e) * t.obs_velocity[1]]
        t.last_obs_center = list(obs_c)

    def tick(self, detections, low=()):
        """One frame. detections = list[Detection]; low = list[bbox]. Returns id per detection."""
        predicted = {tid: t.kalman.predict() for tid, t in self.tracks.items()}
        for t in self.tracks.values():
            t.missed += 1

        matched_ids = []
        ids = []
        cand_ids = sorted(predicted.keys())

        # ── stage-1 Hungarian (iou + W_OCM*momentum), gated by iou_gate ──
        if not detections or not cand_ids:
            primary = [None] * len(detections)
        else:
            cost = []
            for det in detections:
                dcx = (det.bbox[0] + det.bbox[2]) * 0.5
                dcy = (det.bbox[1] + det.bbox[3]) * 0.5
                row = []
                for tid in cand_ids:
                    ov = iou(det.bbox, predicted[tid])
                    if ov <= self.iou_gate:
                        row.append(float("inf")); continue
                    t = self.tracks[tid]
                    if t.obs_velocity is not None and t.last_obs_center is not None:
                        mom = ocm_consistency(t.obs_velocity,
                                              [dcx - t.last_obs_center[0], dcy - t.last_obs_center[1]])
                    else:
                        mom = 0.0
                    row.append(-(ov + W_OCM * mom))
                cost.append(row)
            primary = [cand_ids[j] if j is not None else None for j in _solve(cost)]

        for pid in primary:
            if pid is not None:
                matched_ids.append(pid)

        # ── per-detection: primary, else re-link cascade, else spawn ──
        for d, det in enumerate(detections):
            best_id = primary[d]

            # tier 1: cosine SFace embedding
            if best_id is None and det.embedding is not None:
                best_cos = self.relink_cos
                for tid, t in self.tracks.items():
                    if tid in matched_ids: continue
                    if t.missed == 0 or t.missed > self.relink_window: continue
                    # (track embeddings not modeled in the bench)
                    _ = t
            # tier 2: label-equality re-link (no embedding)
            if best_id is None and det.embedding is None:
                if det.identity is not None and det.identity_confidence is not None \
                        and det.identity_confidence >= self.relink_min_conf:
                    best_missed = None
                    for tid, t in self.tracks.items():
                        if tid in matched_ids: continue
                        if t.missed == 0 or t.missed > self.relink_window: continue
                        if t.identity == det.identity and (best_missed is None or t.missed < best_missed):
                            best_missed = t.missed; best_id = tid

            # tier 3: center-distance re-link (anonymous geometric fallback, OCR virtual pos)
            # PROD gate: det has no embedding AND no identity.
            # Hypothesis-A fix: also allow when identity is present but LOW-confidence
            # (a stale intermittent face) — the geometric match should not be blocked by it.
            det_id_blocks = det.identity is not None
            if self.center_relink_lowconf_id:
                det_id_blocks = det.identity is not None and \
                    (det.identity_confidence is None or det.identity_confidence >= self.relink_min_conf)
            if best_id is None and det.embedding is None and not det_id_blocks:
                dcx = (det.bbox[0] + det.bbox[2]) * 0.5
                dcy = (det.bbox[1] + det.bbox[3]) * 0.5
                best_dist = self.relink_center_dist
                # names actively claimed by a CONFIDENT id on some detection THIS frame —
                # never geometrically steal a track whose person is being identified elsewhere.
                claimed = {dd.identity for dd in detections
                           if dd.identity is not None and dd.identity_confidence is not None
                           and dd.identity_confidence >= self.relink_min_conf}
                for tid, t in self.tracks.items():
                    if tid in matched_ids: continue
                    if t.missed == 0 or t.missed > self.relink_window: continue
                    # PROD: skip any identified track. FIX (hypothesis A): a body-primary walker
                    # gets ONE confident face early -> track.identity sticks -> this skip
                    # permanently disables the anonymous geometric fallback for it -> every
                    # IoU gap spawns a phantom. With the fix, an identified track is still
                    # eligible for a geometric re-link UNLESS its name is being actively
                    # claimed by a confident detection elsewhere this frame.
                    if t.identity is not None:
                        if not self.center_relink_lowconf_id or t.identity in claimed:
                            continue
                    if t.last_obs_center is not None and t.obs_velocity is not None:
                        gap = float(t.missed)
                        vcx = t.last_obs_center[0] + t.obs_velocity[0] * gap
                        vcy = t.last_obs_center[1] + t.obs_velocity[1] * gap
                    elif t.last_obs_center is not None:
                        vcx, vcy = t.last_obs_center
                    else:
                        vcx = (t.bbox[0] + t.bbox[2]) * 0.5; vcy = (t.bbox[1] + t.bbox[3]) * 0.5
                    dd = ((dcx - vcx) ** 2 + (dcy - vcy) ** 2) ** 0.5
                    if dd < best_dist:
                        best_dist = dd; best_id = tid

            if best_id is not None:
                t = self.tracks[best_id]
                t.kalman.update(det.bbox)
                t.bbox = t.kalman.smoothed_bbox()
                t.age += 1
                self._record_obs(t, [(det.bbox[0] + det.bbox[2]) * 0.5,
                                     (det.bbox[1] + det.bbox[3]) * 0.5])
                t.missed = 0
                if det.identity is not None and det.identity_confidence is not None \
                        and det.identity_confidence >= self.relink_min_conf:
                    t.identity = det.identity     # only a confident id sticks to the track
                if best_id not in matched_ids:
                    matched_ids.append(best_id)
                the_id = best_id
            else:
                the_id = self._alloc_id()
                kf = KalmanBBox(det.bbox, self.q_pos, self.q_vel, self.r_pos, self.r_size)
                nt = Track(the_id, det.bbox, kf)
                # seed the first observation anchor (apply_detection does this in prod)
                nt.last_obs_center = [(det.bbox[0] + det.bbox[2]) * 0.5,
                                      (det.bbox[1] + det.bbox[3]) * 0.5]
                if det.identity is not None and det.identity_confidence is not None \
                        and det.identity_confidence >= self.relink_min_conf:
                    nt.identity = det.identity
                self.tracks[the_id] = nt
            ids.append(the_id)

        # ── stage-2 ByteTrack low-conf recovery ──
        for lb in low:
            lcx = (lb[0] + lb[2]) * 0.5; lcy = (lb[1] + lb[3]) * 0.5
            best_score = -float("inf"); best_id = None
            for tid, t in self.tracks.items():
                if t.missed == 0 or tid in matched_ids: continue
                if tid not in predicted: continue
                ov = iou(lb, predicted[tid])
                if ov <= self.iou_gate: continue
                if t.obs_velocity is not None and t.last_obs_center is not None:
                    mom = ocm_consistency(t.obs_velocity, [lcx - t.last_obs_center[0], lcy - t.last_obs_center[1]])
                else:
                    mom = 0.0
                score = ov + W_OCM * mom
                if score > best_score:
                    best_score = score; best_id = tid
            if best_id is not None:
                t = self.tracks[best_id]
                t.kalman.update(lb); t.bbox = t.kalman.smoothed_bbox(); t.age += 1
                self._record_obs(t, [lcx, lcy]); t.missed = 0
                matched_ids.append(best_id)

        # ── inertia advance + lifecycle + expire ──
        for tid, t in self.tracks.items():
            if tid not in matched_ids and tid in predicted:
                t.bbox = predicted[tid]
            t.lifecycle = next_lifecycle(t.lifecycle, t.age, t.missed, locked=False)

        expired = [tid for tid, t in self.tracks.items() if t.missed >= self.timeout]
        for tid in expired:
            del self.tracks[tid]
            self.free_ids.append(tid); self.free_ids.sort()

        return ids


# ── self-check against known Rust unit-test scenarios ──────────────────────────
def _selfcheck():
    # 1) moving object stays one track (kalman.rs tracker_uses_predicted_bbox_for_iou)
    tk = PersonTracker()
    for i in range(5):
        x1 = 0.10 + i * 0.03
        tk.tick([Detection([x1, 0.3, x1 + 0.20, 0.7])])
    assert len(tk.tracks) == 1, f"moving object fragmented: {len(tk.tracks)} tracks"

    # 2) idle object -> exactly one stable id over many frames
    tk = PersonTracker()
    last = None
    for _ in range(50):
        ids = tk.tick([Detection([0.4, 0.2, 0.6, 0.9])])
        last = ids[0]
    assert len(tk.tracks) == 1 and last == 10, f"idle churned: {len(tk.tracks)} tracks id={last}"

    # 3) occlusion gap then re-appear nearby -> re-link (no new id), anonymous center tier
    tk = PersonTracker()
    for i in range(6):                       # establish + build obs velocity moving right
        x1 = 0.10 + i * 0.03
        tk.tick([Detection([x1, 0.3, x1 + 0.20, 0.7])])
    tk.tick([])                              # 1 missed frame
    ids = tk.tick([Detection([0.10 + 7 * 0.03, 0.3, 0.10 + 7 * 0.03 + 0.20, 0.7])])
    assert ids[0] == 10, f"occluded moving body churned a new id: {ids[0]}"
    print("selfcheck OK: moving=1 track, idle=1 stable id, occlusion re-links")


if __name__ == "__main__":
    _selfcheck()
