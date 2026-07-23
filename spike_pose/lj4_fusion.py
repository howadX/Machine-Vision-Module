#!/usr/bin/env python3
"""Faithful Python port of the MVM fusion temporal stack (phase 15.4 lab bench).

Ported byte-for-byte from the Rust (constants quoted at their definitions):
  - infer_activities        mvm-cli/src/agents/fusion.rs:365 (+ helpers 189-355)
  - ActivityJournal (FSM)   mvm-core/src/activity.rs (enter/exit dwell, theta_lo(d),
                            max-dwell cap, churn stitching 11.C6, caps)
  - AnomalyScorer (v2)      mvm-core/src/anomaly.rs (noisy-OR + decaying peak)
  - FusionStep              the advance_journal frame-gating (fusion.rs:51) — the FSM
                            ticks once per NEW frame_id, never per loop iteration.

Same lab choice as LJ2's stabilizer and LJ3's tracker ports: run the production logic
in Python so golden scenarios sweep cheaply with bootstrap CIs; the Rust port of
whatever wins (+ bit-exact golden tests) is 15.6.

Honest scope cuts (each is the prod fallback branch, not an approximation):
  - No instance masks in the bench → contact = bbox test (`point_near_bbox`), the exact
    branch prod takes when masks are off / absent (fusion.rs:241-246). Mask-contact and
    mask-centroid paths are NOT ported; scenarios must not carry masks.
  - No depth maps → `anchor_objects` is not ported; scenarios inject `position_3d`
    directly (the simulator's contract: injection AFTER the detectors).
  - Anti-pet gate (advance_journal's animal filter) not ported — scenarios carry no
    animal detections.

Self-check: `python3 lj4_fusion.py` replays the Rust unit-test scenarios from
activity.rs / anomaly.rs / fusion.rs and must print OK. A drifting port is a stop-ship.
"""
from dataclasses import dataclass, field
from typing import Optional

# ── production constants ───────────────────────────────────────────────────────
# activity.rs
ACTIVITY_ENTER_DWELL = 3        # ln 30
ACTIVITY_EXIT_DWELL = 5         # ln 34
ACTIVITY_EXIT_DWELL_YOUNG = 8   # ln 38
ACTIVITY_EXIT_DWELL_OLD = 4     # ln 42
ACTIVITY_D_MIN_S = 2.0          # ln 44
ACTIVITY_D_TYP_S = 600.0        # ln 47
ACTIVITY_CONF_LAMBDA = 0.7      # ln 49
ACTIVITY_INSTANCE_CAP = 64      # ln 53
ACTIVITY_EVENT_CAP = 256        # ln 55
# anomaly.rs
EXPECTED_OBJECTS = {
    "chair", "couch", "bed", "dining table", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "book", "clock", "vase", "potted plant", "cup",
    "bottle", "bowl", "fork", "spoon", "microwave", "oven", "toaster", "sink",
    "refrigerator",
}                               # ln 25
ANOMALY_OBJECT_WEIGHT = 0.25    # ln 33
ANOMALY_POSE_OOD = 0.3          # ln 35
ANOMALY_FALL = 0.6              # ln 37
ANOMALY_FALL_WINDOW_S = 3.0     # ln 39
ANOMALY_DECAY = 0.9             # ln 41
STANDING_MEMORY_CAP = 64        # ln 43
# fusion.rs
ACTIVITY_NEAR_M = 1.5           # ln 192
CONTACT_MARGIN = 0.03           # ln 195
FRAME_EDGE_MARGIN = 0.02        # ln 198
WORKSTATION_MARGIN = 0.10       # ln 201
WORKSTATION_CLASSES = ("dining table", "laptop", "keyboard", "mouse", "tv", "book")  # ln 208


def interaction_label(cls):
    """fusion.rs — activity label for a person↔object interaction, else None.
    Vocabulary v2 (LJ4/S3 VRD audit): the flat verbs typing/using_mouse/using_phone/
    reading (0-15 supervisable VRD instances) fold into the honest umbrella
    `using_object(<class>)`; holding_drink is kept (145 instances); watching_tv is
    dropped (head-facing, not a wrist contact; VRD 0)."""
    if cls in ("cup", "bottle", "wine glass", "bowl"):
        return "holding_drink"
    if cls in ("remote", "tv"):
        return None
    if cls in ("keyboard", "laptop", "mouse", "cell phone", "book"):
        return f"using_object({cls})"
    return None


def max_dwell_s(label):
    """activity.rs — bounded-duration cap per label (vocab v2: using_object family
    rides the transient cap by prefix)."""
    if label == "holding_drink":
        return 900.0
    if label.startswith("using_object("):
        return 1800.0
    return 3600.0


# ── injected observations (the simulator's post-detector contract) ─────────────

@dataclass
class Person:
    """PersonState subset the fusion stack reads (world.rs). bbox normalized
    [x1,y1,x2,y2]; position_3d (x,y,z) metric or None; posture (label, conf) or
    None; wrists = [[x,y],...]; hands = [[[x,y],...] per hand]."""
    track_id: int
    bbox: tuple = (0.3, 0.2, 0.5, 0.7)
    position_3d: Optional[tuple] = None
    posture: Optional[tuple] = None
    plausibility: Optional[str] = None
    wrists: list = field(default_factory=list)
    hands: list = field(default_factory=list)
    left_gesture: Optional[str] = None
    right_gesture: Optional[str] = None
    body_coverage: Optional[str] = None
    torso_attitude: Optional[str] = None
    motion_state: Optional[str] = None


@dataclass
class Obj:
    """ObjectState subset (id ≥ 10000 in prod id-space)."""
    id: int
    cls: str
    bbox: tuple
    confidence: float = 0.8


@dataclass
class Activity:
    label: str
    confidence: float
    participants: list


# ── raw cross-group inference (fusion.rs:365) ──────────────────────────────────

def _point_near_bbox(pt, b):
    """fusion.rs:226 — normalised point within b expanded by CONTACT_MARGIN."""
    return (b[0] - CONTACT_MARGIN <= pt[0] <= b[2] + CONTACT_MARGIN
            and b[1] - CONTACT_MARGIN <= pt[1] <= b[3] + CONTACT_MARGIN)


def _contact_centre(o):
    """fusion.rs:251 — bbox centre (mask centroid path out of bench scope)."""
    return ((o.bbox[0] + o.bbox[2]) * 0.5, (o.bbox[1] + o.bbox[3]) * 0.5)


class _ContactClaim:
    """fusion.rs:271 — one person's claim on a contested object."""

    def __init__(self, track_id, hand_touch, reaching_wrist_only, dist2):
        self.track_id = track_id
        self.hand_touch = hand_touch
        self.reaching_wrist_only = reaching_wrist_only
        self.dist2 = dist2

    def beats(self, other):
        if self.hand_touch != other.hand_touch:
            return self.hand_touch
        if self.reaching_wrist_only != other.reaching_wrist_only:
            return not self.reaching_wrist_only
        if self.dist2 != other.dist2:
            return self.dist2 < other.dist2
        return self.track_id < other.track_id


def _contact_claim(p, o, centre):
    """fusion.rs:295 — this person's claim on the object, or None."""
    hand_touch = wrist_touch = False
    dist2 = float("inf")

    def consider(pt, is_hand):
        nonlocal hand_touch, wrist_touch, dist2
        if _point_near_bbox(pt, o.bbox):
            if is_hand:
                hand_touch = True
            else:
                wrist_touch = True
            dx, dy = pt[0] - centre[0], pt[1] - centre[1]
            dist2 = min(dist2, dx * dx + dy * dy)

    for pt in p.wrists:
        consider(pt, False)
    for hand in p.hands:
        for pt in hand:
            consider(pt, True)
    if not hand_touch and not wrist_touch:
        return None
    reaching = p.left_gesture == "reaching" or p.right_gesture == "reaching"
    return _ContactClaim(p.track_id, hand_touch, not hand_touch and reaching, dist2)


def _dist3(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _bbox_adjacent(a, b, margin):
    """fusion.rs:336 — a expanded by margin intersects b."""
    return (a[0] - margin < b[2] and b[0] < a[2] + margin
            and a[1] - margin < b[3] and b[1] < a[3] + margin)


def _bbox_at_frame_edge(b):
    """fusion.rs:342."""
    return (b[0] <= FRAME_EDGE_MARGIN or b[1] <= FRAME_EDGE_MARGIN
            or b[2] >= 1.0 - FRAME_EDGE_MARGIN or b[3] >= 1.0 - FRAME_EDGE_MARGIN)


def _upper_upright_stationary(p):
    """fusion.rs:351."""
    return (p.body_coverage == "upper" and p.torso_attitude == "upright"
            and p.motion_state != "moving")


def infer_activities(people, objects):
    """fusion.rs:365 — raw per-tick activities, pure arithmetic."""
    acts = []

    # person ↔ person proximity (needs both torso-3D positions).
    for i in range(len(people)):
        for j in range(i + 1, len(people)):
            pa, pb = people[i].position_3d, people[j].position_3d
            if pa is not None and pb is not None:
                d = _dist3(pa, pb)
                if d < ACTIVITY_NEAR_M:
                    acts.append(Activity(
                        "people_together",
                        min(max(1.0 - d / ACTIVITY_NEAR_M, 0.0), 1.0),
                        [people[i].track_id, people[j].track_id]))

    # person ↔ interactive object by hand/wrist contact, object-major (11.A2).
    for o in objects:
        label = interaction_label(o.cls)
        if label is None:
            continue
        centre = _contact_centre(o)
        best = None
        for p in people:
            claim = _contact_claim(p, o, centre)
            if claim is not None and (best is None or claim.beats(best)):
                best = claim
        if best is not None:
            acts.append(Activity(label, min(max(o.confidence, 0.0), 1.0),
                                 [best.track_id, o.id]))

    # Occlusion contexts (occlusion-aware posture).
    for p in people:
        if _upper_upright_stationary(p):
            for o in objects:
                if (o.cls in WORKSTATION_CLASSES
                        and _bbox_adjacent(p.bbox, o.bbox, WORKSTATION_MARGIN)):
                    acts.append(Activity("seated_at_workstation",
                                         min(max(o.confidence, 0.0), 1.0),
                                         [p.track_id, o.id]))
                    break
        if (p.body_coverage is not None and p.body_coverage != "full"
                and _bbox_at_frame_edge(p.bbox)):
            acts.append(Activity("partially_in_frame", 1.0, [p.track_id]))

    return acts


# ── Activity FSM + journal (activity.rs) ───────────────────────────────────────

def _churn_class(pid):
    """activity.rs:90 — None reserved (0-9), 0 ephemeral person, 1 object."""
    if pid < 10:
        return None
    return 0 if pid < 10_000 else 1


def _one_churn_apart(a, b):
    """activity.rs:105 — sorted sets differ in exactly one same-class churnable id."""
    a_only = [x for x in a if x not in b]
    b_only = [x for x in b if x not in a]
    return (len(a_only) == 1 and len(b_only) == 1
            and _churn_class(a_only[0]) is not None
            and _churn_class(a_only[0]) == _churn_class(b_only[0]))


CANDIDATE, ACTIVE, COOLING, ENDED = "candidate", "active", "cooling", "ended"


class ActivityInstance:
    def __init__(self, key, phase, enter_streak, started_at, first_seen_frame,
                 last_seen, confidence, peak_confidence):
        self.key = key                      # (label, tuple(sorted participants))
        self.phase = phase
        self.enter_streak = enter_streak
        self.absent_streak = 0
        self.started_at = started_at
        self.first_seen_frame = first_seen_frame
        self.last_seen = last_seen
        self.confidence = confidence
        self.peak_confidence = peak_confidence


@dataclass
class ActivityEvent:
    label: str
    participants: list
    started_at: float
    ended_at: float
    duration_s: float
    peak_confidence: float
    stable_participants: bool


def _stable_participants(participants):
    """activity.rs:77 — reserved (0-9) or object (≥10000) ids only."""
    return all(pid < 10 or pid >= 10_000 for pid in participants)


def _event_from(inst):
    return ActivityEvent(
        label=inst.key[0], participants=list(inst.key[1]),
        started_at=inst.started_at, ended_at=inst.last_seen,
        duration_s=inst.last_seen - inst.started_at,
        peak_confidence=inst.peak_confidence,
        stable_participants=_stable_participants(inst.key[1]))


class ActivityJournal:
    """activity.rs:220 — temporal debounce + event journal, stitching opt-in.

    The exit-dwell knobs are exposed for the LJ4 sweep (lab pattern, like
    lj3_tracker's gates); the defaults ARE the production constants."""

    def __init__(self, stitch_churn=False, exit_dwell=ACTIVITY_EXIT_DWELL,
                 exit_young=ACTIVITY_EXIT_DWELL_YOUNG, exit_old=ACTIVITY_EXIT_DWELL_OLD):
        self.instances = []     # sorted by key — deterministic emission order
        self.completed = []
        self.stitch_churn = stitch_churn
        self.exit_dwell, self.exit_young, self.exit_old = exit_dwell, exit_young, exit_old

    def update(self, raw, t_secs, frame_id):
        # Present keys → strongest observed confidence this tick (sorted).
        present = {}
        for a in raw:
            key = (a.label, tuple(sorted(a.participants)))
            present[key] = max(present.get(key, 0.0), a.confidence)
        present = sorted(present.items())

        # Advance existing instances (in stored sorted order).
        for inst in self.instances:
            obs = None
            for i, (k, conf) in enumerate(present):
                if k == inst.key:
                    obs = conf
                    present.pop(i)
                    break
            if obs is not None:
                inst.absent_streak = 0
                inst.last_seen = t_secs
                inst.confidence = (ACTIVITY_CONF_LAMBDA * inst.confidence
                                   + (1.0 - ACTIVITY_CONF_LAMBDA) * obs)
                inst.peak_confidence = max(inst.peak_confidence, inst.confidence)
                if inst.phase == CANDIDATE:
                    inst.enter_streak += 1
                    if inst.enter_streak >= ACTIVITY_ENTER_DWELL:
                        inst.phase = ACTIVE
                elif inst.phase == COOLING:
                    inst.phase = ACTIVE
                # Bounded-duration cap (11.C2).
                if (inst.phase == ACTIVE
                        and inst.last_seen - inst.started_at >= max_dwell_s(inst.key[0])):
                    inst.phase = ENDED
                    self.completed.append(_event_from(inst))
            else:
                if inst.phase == CANDIDATE:
                    inst.phase = ENDED  # 1-frame spike never reaches the journal
                elif inst.phase in (ACTIVE, COOLING):
                    inst.phase = COOLING
                    inst.absent_streak += 1
                    d = inst.last_seen - inst.started_at
                    if d < ACTIVITY_D_MIN_S:
                        exit_dwell = self.exit_young
                    elif d > ACTIVITY_D_TYP_S:
                        exit_dwell = self.exit_old
                    else:
                        exit_dwell = self.exit_dwell
                    if inst.absent_streak >= exit_dwell:
                        inst.phase = ENDED
                        self.completed.append(_event_from(inst))

        # Reap ended/lapsed.
        self.instances = [i for i in self.instances if i.phase != ENDED]

        # Keys still unmatched are newborn candidates (bounded).
        for key, conf in present:
            if len(self.instances) >= ACTIVITY_INSTANCE_CAP:
                break
            src = None
            if self.stitch_churn and len(key[1]) >= 2:
                src = self._take_stitch_source(key)
            if src is not None:
                inst = ActivityInstance(key, ACTIVE, ACTIVITY_ENTER_DWELL,
                                        src.started_at, src.first_seen_frame,
                                        t_secs, conf, max(src.peak_confidence, conf))
            else:
                inst = ActivityInstance(key, CANDIDATE, 1, t_secs, frame_id,
                                        t_secs, conf, conf)
            # Insert keeping sort order.
            at = 0
            while at < len(self.instances) and self.instances[at].key < inst.key:
                at += 1
            self.instances.insert(at, inst)

        # Bounded event backlog even if the consumer never drains.
        if len(self.completed) > ACTIVITY_EVENT_CAP:
            del self.completed[:len(self.completed) - ACTIVITY_EVENT_CAP]

    def _take_stitch_source(self, newborn):
        """activity.rs:383 — remove + return the cooling instance this newborn
        continues (same label, one churnable participant apart)."""
        for i, inst in enumerate(self.instances):
            if (inst.phase == COOLING and inst.key[0] == newborn[0]
                    and len(inst.key[1]) == len(newborn[1])
                    and _one_churn_apart(inst.key[1], newborn[1])):
                return self.instances.pop(i)
        return None

    def active(self):
        """Stabilized activities to emit: Active + Cooling (grace-bridged)."""
        return [i for i in self.instances if i.phase in (ACTIVE, COOLING)]

    def drain_events(self):
        ev, self.completed = self.completed, []
        return ev


# ── Anomaly v2 scorer (anomaly.rs) ─────────────────────────────────────────────

class AnomalyScorer:
    """anomaly.rs:48 — noisy-OR of unexpected-object / pose-OOD / sudden-fall,
    published as a decaying peak."""

    def __init__(self):
        self.standing_seen = []     # [(track_id, t_secs last seen standing)]
        self.score = 0.0

    def update(self, people, objects, t_secs):
        components = []

        unexpected = sum(1 for o in objects if o.cls not in EXPECTED_OBJECTS)
        if unexpected > 0:
            components.append(min(unexpected * ANOMALY_OBJECT_WEIGHT, 1.0))

        for p in people:
            label = p.posture[0] if p.posture is not None else None
            if label == "standing":
                for e in self.standing_seen:
                    if e[0] == p.track_id:
                        e[1] = t_secs
                        break
                else:
                    if len(self.standing_seen) < STANDING_MEMORY_CAP:
                        self.standing_seen.append([p.track_id, t_secs])
            if label == "lying":
                recently_stood = any(
                    tid == p.track_id and t_secs - t <= ANOMALY_FALL_WINDOW_S
                    for tid, t in self.standing_seen)
                if recently_stood:
                    components.append(ANOMALY_FALL)
            if p.plausibility == "low":
                components.append(ANOMALY_POSE_OOD)

        # Forget stale standing memory (well past the window).
        self.standing_seen = [e for e in self.standing_seen
                              if t_secs - e[1] <= ANOMALY_FALL_WINDOW_S * 4.0]

        raw = 1.0
        for c in components:
            raw *= 1.0 - min(max(c, 0.0), 1.0)
        raw = 1.0 - raw
        self.score = max(raw, self.score * ANOMALY_DECAY)
        if self.score < 1e-3:
            self.score = 0.0
        return self.score


# ── FusionStep — the advance_journal frame-gating (fusion.rs:51) ───────────────

class FusionStep:
    """One world tick: raw inference every call, FSM/anomaly advanced once per
    NEW frame_id (the mesh loop spins faster than frames arrive). Stitching is
    ON as in the mesh FusionAgent (fusion.rs:40)."""

    def __init__(self, stitch_churn=True, journal_kw=None):
        self.journal = ActivityJournal(stitch_churn=stitch_churn, **(journal_kw or {}))
        self.anomaly = AnomalyScorer()
        self.last_frame = None

    def tick(self, people, objects, t_secs, frame_id):
        """Returns (activities, new_events, anomaly_score) — the stabilized set
        (Active+Cooling with EMA confidence), episodes completed this tick, and
        the decaying-peak anomaly score."""
        raw = infer_activities(people, objects)
        events = []
        if self.last_frame != frame_id:
            self.last_frame = frame_id
            self.journal.update(raw, t_secs, frame_id)
            events = self.journal.drain_events()
            self.anomaly_score = self.anomaly.update(people, objects, t_secs)
        activities = [Activity(i.key[0], i.confidence, list(i.key[1]))
                      for i in self.journal.active()]
        return activities, events, getattr(self, "anomaly_score", 0.0)


# ── self-check: replay the Rust unit-test scenarios ───────────────────────────

def _act(label, participants, conf=0.8):
    return Activity(label, conf, participants)


def _drive(j, raw, n, t0, f0):
    t, f = t0, f0
    for _ in range(n):
        j.update(raw, t, f)
        t += 0.1
        f += 1
    return t, f


def _check_fsm():
    typing = lambda: _act("typing", [0, 10001])

    # one_frame_spike_never_emits
    j = ActivityJournal()
    j.update([typing()], 0.0, 0)
    assert len(j.active()) == 0
    j.update([], 0.1, 1)
    j.update([typing()], 0.2, 2)
    j.update([], 0.3, 3)
    assert len(j.active()) == 0 and not j.drain_events()

    # enter_dwell_confirms_after_three_consecutive
    j = ActivityJournal()
    _drive(j, [typing()], 2, 0.0, 0)
    assert len(j.active()) == 0
    j.update([typing()], 0.2, 2)
    inst = j.active()
    assert len(inst) == 1 and inst[0].phase == ACTIVE
    assert inst[0].started_at == 0.0 and inst[0].first_seen_frame == 0

    # short_dropout_is_bridged_by_cooling
    j = ActivityJournal()
    t, f = _drive(j, [typing()], 3, 0.0, 0)
    t, f = _drive(j, [], 2, t, f)
    inst = j.active()
    assert len(inst) == 1 and inst[0].phase == COOLING
    j.update([typing()], t, f)
    inst = j.active()
    assert inst[0].phase == ACTIVE and inst[0].started_at == 0.0
    assert not j.drain_events()

    # exit_dwell_ends_with_honest_duration
    j = ActivityJournal()
    t, f = _drive(j, [typing()], 30, 0.0, 0)
    _drive(j, [], ACTIVITY_EXIT_DWELL, t, f)
    assert len(j.active()) == 0
    ev = j.drain_events()
    assert len(ev) == 1 and ev[0].label == "typing"
    assert ev[0].participants == [0, 10001] and ev[0].started_at == 0.0
    assert abs(ev[0].ended_at - 2.9) < 1e-6 and abs(ev[0].duration_s - 2.9) < 1e-6
    assert not j.drain_events()

    # confidence_is_ema_and_peak_tracked
    j = ActivityJournal()
    j.update([_act("typing", [0, 10001], 1.0)], 0.0, 0)
    j.update([_act("typing", [0, 10001], 0.0)], 0.1, 1)
    assert abs(j.instances[0].confidence - 0.7) < 1e-6
    assert abs(j.instances[0].peak_confidence - 1.0) < 1e-6

    # duplicate_keys_and_participant_order_collapse
    j = ActivityJournal()
    raw = [_act("typing", [10001, 0], 0.5), _act("typing", [0, 10001], 0.9)]
    _drive(j, raw, 3, 0.0, 0)
    assert len(j.active()) == 1

    # distinct_participant_sets_are_distinct_instances
    j = ActivityJournal()
    raw = [_act("typing", [0, 10001]), _act("typing", [1, 10001])]
    _drive(j, raw, 3, 0.0, 0)
    assert len(j.active()) == 2

    # emission_order_is_deterministic
    a, b = ActivityJournal(), ActivityJournal()
    _drive(a, [_act("typing", [0, 10001]), _act("reading", [1, 10002])], 3, 0.0, 0)
    _drive(b, [_act("reading", [1, 10002]), _act("typing", [0, 10001])], 3, 0.0, 0)
    assert [i.key for i in a.active()] == [i.key for i in b.active()]

    # participant_churn_ends_and_restarts (default: no stitching)
    j = ActivityJournal()
    t, f = _drive(j, [_act("typing", [10, 10001])], 6, 0.0, 0)
    _drive(j, [_act("typing", [11, 10001])], 8, t, f)
    active = j.active()
    assert len(active) == 1 and list(active[0].key[1]) == [11, 10001]
    ev = j.drain_events()
    assert len(ev) == 1 and ev[0].participants == [10, 10001]

    # stitching_bridges_ephemeral_churn_on_same_object
    j = ActivityJournal(stitch_churn=True)
    t, f = _drive(j, [_act("typing", [12, 10001])], 30, 0.0, 0)
    t, f = _drive(j, [_act("typing", [15, 10001])], 8, t, f)
    assert not j.drain_events(), "churn must not split the episode"
    inst = j.active()
    assert len(inst) == 1 and list(inst[0].key[1]) == [15, 10001]
    assert inst[0].started_at == 0.0
    _drive(j, [], ACTIVITY_EXIT_DWELL, t, f)
    ev = j.drain_events()
    assert len(ev) == 1 and ev[0].started_at == 0.0 and ev[0].duration_s > 3.0

    # stitching_does_not_bridge_across_different_objects
    j = ActivityJournal(stitch_churn=True)
    t, f = _drive(j, [_act("typing", [12, 10001])], 6, 0.0, 0)
    _drive(j, [_act("typing", [15, 10002])], 8, t, f)
    assert j.drain_events(), "two-participant churn → end+restart"

    # stitching_bridges_object_churn_under_stable_person
    j = ActivityJournal(stitch_churn=True)
    t, f = _drive(j, [_act("typing", [0, 10000])], 30, 0.0, 0)
    t, f = _drive(j, [_act("typing", [0, 10007])], 8, t, f)
    assert not j.drain_events()
    inst = j.active()
    assert len(inst) == 1 and list(inst[0].key[1]) == [0, 10007]
    assert inst[0].started_at == 0.0
    _drive(j, [], ACTIVITY_EXIT_DWELL, t, f)
    ev = j.drain_events()
    assert len(ev) == 1 and ev[0].started_at == 0.0 and ev[0].duration_s > 3.0

    # stitching_never_merges_two_gallery_people
    j = ActivityJournal(stitch_churn=True)
    t, f = _drive(j, [_act("typing", [0, 10001])], 6, 0.0, 0)
    _drive(j, [_act("typing", [5, 10001])], 8, t, f)
    assert j.drain_events(), "person 0 ≠ person 5 → honest end+restart"

    # stable_participants_reflects_id_spaces
    assert _stable_participants([0, 10001]) and not _stable_participants([42, 10001])
    j = ActivityJournal()
    t, f = _drive(j, [_act("typing", [0, 10001])], 3, 0.0, 0)
    _drive(j, [], ACTIVITY_EXIT_DWELL + 3, t, f)
    ev = j.drain_events()
    assert len(ev) == 1 and ev[0].stable_participants

    # max_dwell_caps_duration_and_restarts
    j = ActivityJournal()
    raw = [_act("typing", [0, 10001])]
    t = 0.0
    for fr in range(5000):
        j.update(raw, t, fr)
        t += 1.0
    ev = j.drain_events()
    assert ev, "cap must have force-ended at least once"
    assert all(e.duration_s <= max_dwell_s("typing") + 1.0 for e in ev)
    assert len(j.active()) == 1, "activity restarts after the forced end"

    # young_activity_survives_longer_gap
    j = ActivityJournal()
    t, f = _drive(j, [typing()], 3, 0.0, 0)
    t, f = _drive(j, [], ACTIVITY_EXIT_DWELL + 1, t, f)
    assert len(j.active()) == 1, "young instance still cooling"
    j.update([typing()], t, f)
    assert j.active()[0].started_at == 0.0 and not j.drain_events()

    # old_activity_exits_faster
    j = ActivityJournal()
    _, f = _drive(j, [typing()], 3, 0.0, 0)
    t = 100.0
    while t < ACTIVITY_D_TYP_S + 200.0:
        j.update([typing()], t, f)
        t += 100.0
        f += 1
    _drive(j, [], ACTIVITY_EXIT_DWELL_OLD, t, f)
    assert len(j.active()) == 0 and len(j.drain_events()) == 1

    # instance_cap_bounds_memory
    j = ActivityJournal()
    flood = [_act("typing", [i, 10001], 0.5) for i in range(2 * ACTIVITY_INSTANCE_CAP)]
    _drive(j, flood, 3, 0.0, 0)
    assert len(j.instances) <= ACTIVITY_INSTANCE_CAP

    # event_cap_bounds_undrained_backlog
    j = ActivityJournal()
    t, f = 0.0, 0
    for i in range(ACTIVITY_EVENT_CAP + 40):
        raw = [_act("typing", [i, 10001], 0.5)]
        for _ in range(ACTIVITY_ENTER_DWELL):
            j.update(raw, t, f)
            t += 0.1
            f += 1
        for _ in range(ACTIVITY_EXIT_DWELL):
            j.update([], t, f)
            t += 0.1
            f += 1
    assert len(j.completed) <= ACTIVITY_EVENT_CAP


def _check_anomaly():
    P = lambda tid, posture: Person(tid, posture=(posture, 0.9) if posture else None)
    O = lambda cls: Obj(10000, cls, (0.4, 0.4, 0.5, 0.5))

    # calm_scene_scores_zero
    s = AnomalyScorer()
    assert s.update([P(0, "standing")], [O("cup")], 0.0) == 0.0

    # unexpected_object_raises
    s = AnomalyScorer()
    assert abs(s.update([], [O("knife")], 0.0) - ANOMALY_OBJECT_WEIGHT) < 1e-6

    # sudden_fall_flags_but_longterm_lying_does_not
    s = AnomalyScorer()
    s.update([P(0, "standing")], [], 0.0)
    assert abs(s.update([P(0, "lying")], [], 1.0) - ANOMALY_FALL) < 1e-6
    s2 = AnomalyScorer()
    assert s2.update([P(0, "lying")], [], 0.0) == 0.0
    s3 = AnomalyScorer()
    s3.update([P(0, "standing")], [], 0.0)
    for t in range(1, 40):
        s3.update([P(0, "sitting")], [], float(t))
    assert s3.update([P(0, "lying")], [], 40.0) < 0.05

    # knife_plus_fall_scores_above_either_alone
    s = AnomalyScorer()
    s.update([P(0, "standing")], [], 0.0)
    combined = s.update([P(0, "lying")], [O("knife")], 1.0)
    expected = 1.0 - (1.0 - ANOMALY_FALL) * (1.0 - ANOMALY_OBJECT_WEIGHT)
    assert abs(combined - expected) < 1e-6
    assert ANOMALY_FALL < combined <= 1.0

    # pose_ood_contributes
    s = AnomalyScorer()
    p = P(0, "standing")
    p.plausibility = "low"
    assert abs(s.update([p], [], 0.0) - ANOMALY_POSE_OOD) < 1e-6

    # spike_decays_instead_of_vanishing
    s = AnomalyScorer()
    s.update([P(0, "standing")], [], 0.0)
    spike = s.update([P(0, "lying")], [], 1.0)
    nxt = s.update([], [], 2.0)
    assert abs(nxt - spike * ANOMALY_DECAY) < 1e-6
    last = nxt
    for t in range(3, 120):
        last = s.update([], [], float(t))
    assert last == 0.0


def _check_infer():
    def person(tid=10, z=1.0):
        return Person(tid, position_3d=(0.0, 0.0, z))

    def person_id(tid):
        return Person(tid, position_3d=None)

    # typing_when_wrist_on_keyboard
    p = person()
    p.wrists = [[0.40, 0.58]]
    kb = Obj(10001, "keyboard", (0.35, 0.55, 0.45, 0.62))
    acts = infer_activities([p], [kb])
    assert any(a.label == "typing" and a.participants == [10, 10001] for a in acts)

    # contested_object_goes_to_hand_not_reaching_wrist
    phone = Obj(10001, "cell phone", (0.5, 0.5, 0.6, 0.6))
    a = person_id(10)
    a.wrists = [[0.51, 0.55]]
    a.right_gesture = "reaching"
    b = person_id(11)
    b.hands = [[[0.58, 0.55]]]
    acts = infer_activities([a, b], [phone])
    phone_acts = [x for x in acts if x.label == "using_phone"]
    assert len(phone_acts) == 1 and phone_acts[0].participants == [11, 10001]

    # contested_object_goes_to_closest_wrist
    cup = Obj(10002, "cup", (0.4, 0.4, 0.6, 0.6))
    a = person_id(10)
    a.wrists = [[0.41, 0.5]]
    b = person_id(11)
    b.wrists = [[0.5, 0.5]]
    acts = infer_activities([a, b], [cup])
    cup_acts = [x for x in acts if x.label == "holding_drink"]
    assert len(cup_acts) == 1 and cup_acts[0].participants == [11, 10002]

    # uncontested_contact_still_fires (reaching alone must not silence it)
    a = person_id(10)
    a.wrists = [[0.41, 0.5]]
    a.right_gesture = "reaching"
    acts = infer_activities([a], [cup])
    assert any(x.label == "holding_drink" and x.participants == [10, 10002] for x in acts)

    # two_objects_two_people_attribute_independently
    kb1 = Obj(10001, "keyboard", (0.1, 0.5, 0.3, 0.6))
    kb2 = Obj(10002, "keyboard", (0.6, 0.5, 0.8, 0.6))
    a = person_id(10)
    a.wrists = [[0.2, 0.55]]
    b = person_id(11)
    b.wrists = [[0.7, 0.55]]
    acts = infer_activities([a, b], [kb1, kb2])
    assert any(x.label == "typing" and x.participants == [10, 10001] for x in acts)
    assert any(x.label == "typing" and x.participants == [11, 10002] for x in acts)

    # hand_landmark_triggers_object_use
    p = person()
    p.hands = [[[0.33, 0.57], [0.34, 0.56]]]
    mouse = Obj(10001, "mouse", (0.31, 0.54, 0.36, 0.60))
    assert any(a.label == "using_mouse" for a in infer_activities([p], [mouse]))

    # no_activity_when_landmarks_far_from_object
    p = person()
    p.wrists = [[0.2, 0.2]]
    kb = Obj(10001, "keyboard", (0.8, 0.8, 0.9, 0.85))
    assert infer_activities([p], [kb]) == []

    # no_landmarks_means_no_object_activity
    p = person()
    mouse = Obj(10001, "mouse", (0.35, 0.3, 0.45, 0.4))
    assert infer_activities([p], [mouse]) == []

    # furniture_is_context_not_activity
    p = person()
    p.wrists = [[0.40, 0.60]]
    chair = Obj(10001, "chair", (0.35, 0.55, 0.45, 0.65))
    assert infer_activities([p], [chair]) == []

    # people_together_when_near
    a, b = person(10), person(11)
    acts = infer_activities([a, b], [])
    assert any(x.label == "people_together" and x.participants == [10, 11] for x in acts)

    # seated_at_workstation: upper+upright+stationary adjacent to a workstation class
    p = person(10)
    p.body_coverage, p.torso_attitude, p.motion_state = "upper", "upright", "stationary"
    kb = Obj(10001, "keyboard", (0.45, 0.65, 0.6, 0.75))  # adjacent within 0.10 margin
    acts = infer_activities([p], [kb])
    assert any(x.label == "seated_at_workstation" and x.participants == [10, 10001]
               for x in acts)
    p.motion_state = "moving"   # locomoting → no claim
    assert not any(x.label == "seated_at_workstation"
                   for x in infer_activities([p], [kb]))

    # partially_in_frame: partial coverage + bbox clipped by the frame edge
    p = person(10)
    p.body_coverage = "upper"
    p.bbox = (0.0, 0.2, 0.2, 0.7)
    acts = infer_activities([p], [])
    assert any(x.label == "partially_in_frame" and x.participants == [10] for x in acts)


def _check_fusion_step():
    # repeated_raw_debounces_then_journals_an_event (fusion.rs mesh wiring)
    def typing_world(f):
        p = Person(10)
        p.wrists = [[0.40, 0.58]]
        kb = Obj(10001, "keyboard", (0.35, 0.55, 0.45, 0.62))
        return [p], [kb]

    step = FusionStep()
    for f in range(2):
        people, objects = typing_world(f)
        acts, _, _ = step.tick(people, objects, 0.1 * f, f)
        assert acts == [], f"frame {f}: still a candidate"
    people, objects = typing_world(2)
    acts, _, _ = step.tick(people, objects, 0.2, 2)
    assert len(acts) == 1 and acts[0].label == "typing"
    f = 3
    while f < 33:
        people, objects = typing_world(f)
        step.tick(people, objects, 0.1 * f, f)
        f += 1
    last_events = []
    for g in range(ACTIVITY_EXIT_DWELL):
        _, ev, _ = step.tick([], [], 0.1 * (f + g), f + g)
        if ev:
            last_events = ev
    assert len(last_events) == 1 and last_events[0].label == "typing"
    assert last_events[0].duration_s > 2.0

    # stale_frame_id_does_not_advance_dwell
    step = FusionStep()
    for _ in range(10):
        people, objects = typing_world(5)
        acts, _, _ = step.tick(people, objects, 0.0, 5)
        assert acts == [], "same frame repeated must not confirm"


def _selfcheck():
    _check_fsm()
    _check_anomaly()
    _check_infer()
    _check_fusion_step()
    print("selfcheck OK: FSM (18 scenarios), anomaly (6), infer_activities (13), "
          "frame-gated FusionStep (2) — all match the Rust unit tests")


if __name__ == "__main__":
    _selfcheck()
