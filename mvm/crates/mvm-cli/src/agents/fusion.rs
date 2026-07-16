//! FusionAgent (phase 8.C.7) — the cross-group step. It is the only agent that
//! sees **both** people and objects plus the shared depth map, so depth-anchoring
//! lives here (decision 8.C.5): the relative depth map is anchored to people's
//! metric torso-Z, giving objects a metric position in the same frame → real
//! person↔object `Near` edges (built later by the StateFileWriter / 8.F).
//!
//! It reads the depth handle from the shared `DepthSlot`, the frame dimensions
//! from the latest frame (the same pixel space the body-primary 3D uses), and
//! read-modify-writes `WorldState.objects` under a single write lock so a
//! concurrent `SceneAgent` swap can't clobber the anchored positions.

use crate::agents::DepthSlot;
use mvm_core::{Activity, ActivityJournal, AnomalyScorer, BBox, ObjectState, PersonState, Position3D, WorldState};
use mvm_mesh::{Agent, AgentHealth, MeshContext};

pub(crate) struct FusionAgent {
    config: mvm_core::Config,
    depth_slot: DepthSlot,
    /// Temporal debounce + event journal over the raw activities (phase 11.C3).
    journal: ActivityJournal,
    /// Anomaly v2 scorer (phase 11.B2) — same frame-gated cadence as the journal.
    anomaly: AnomalyScorer,
    /// Last frame the journal advanced on — the loop spins faster than frames
    /// arrive, and FSM dwells must count *frames*, not loop iterations.
    last_frame: Option<u64>,
    /// Zero point for the journal's monotonic clock.
    epoch: std::time::Instant,
}

impl FusionAgent {
    pub(crate) fn new(config: mvm_core::Config, depth_slot: DepthSlot) -> Self {
        let camera_height_m = config.camera_height_m;
        Self {
            config,
            depth_slot,
            // Churn stitching on (11.C6): the mesh live test (2026-07-03)
            // showed object-track id churn spamming the journal with dozens of
            // ½-second typing/holding_drink fragments for one physical desk —
            // stitching collapses a churned object-id (or anon person-id) under
            // a held anchor into one continuous episode.
            journal: ActivityJournal::new().with_stitching(true),
            anomaly: AnomalyScorer::new().with_camera_height(camera_height_m),
            last_frame: None,
            epoch: std::time::Instant::now(),
        }
    }

    /// Advance the activity journal for this world tick (phase 11.C3): re-derive
    /// the raw activities, step the FSM once per *new* frame_id, publish the
    /// stabilized set (Active+Cooling, EMA confidence) as `world.activities` and
    /// append completed episodes to `world.events`.
    fn advance_journal(&mut self, w: &mut WorldState) {
        // Drop a body overlapping a YOLO animal detection from relational
        // activities — a pet with a spurious BlazePose skeleton must never be a
        // `people_together` participant (model-corroborated anti-pet gate;
        // sync-path parity, phase 12.B3). No animals → no clone (hot path).
        let raw = if w.animals.is_empty() {
            infer_activities(w.people.as_slice(), w.objects.as_slice())
        } else {
            let people: Vec<PersonState> = w
                .people
                .iter()
                .filter(|p| !w.animals.iter().any(|a| p.bbox.iou(a) > mvm_scene::ANIMAL_IOU))
                .cloned()
                .collect();
            infer_activities(people.as_slice(), w.objects.as_slice())
        };
        if self.last_frame != Some(w.frame_id) {
            self.last_frame = Some(w.frame_id);
            let t = w.timestamp.saturating_duration_since(self.epoch).as_secs_f64();
            self.journal.update(&raw, t, w.frame_id);
            let ended = self.journal.drain_events();
            w.events.extend(ended);
            // Anomaly v2 (11.B2): decaying peak over noisy-OR'd signals —
            // ticked once per frame so the decay rate means something.
            w.anomaly_score = self.anomaly.update(&w.people, &w.objects, t);
        }
        w.activity_instances = self.journal.active().cloned().collect();
        w.activities = w
            .activity_instances
            .iter()
            .map(|i| Activity {
                label: i.key.label.clone(),
                confidence: i.confidence,
                participants: i.key.participants.clone(),
            })
            .collect();
    }
}

impl Agent for FusionAgent {
    fn name(&self) -> String {
        "fusion".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let pinhole = mvm_core::space::Pinhole::from_config(&self.config);
        let mut ticks = 0u64;
        while ctx.running() {
            // Cheaply grab the latest depth handle (clone the inner Arc, release
            // the slot lock immediately) + the frame's pixel dimensions.
            let depth = self.depth_slot.lock().ok().and_then(|g| g.clone());
            let dims = ctx
                .frames
                .latest()
                .map(|f| (f.image.width() as f32, f.image.height() as f32));
            if let Ok(mut w) = ctx.world.write() {
                {
                    let WorldState { people, objects, .. } = &mut *w;
                    // Anchor objects to metric 3D when a depth map is available
                    // (the person↔person activities don't need it — people carry
                    // torso-3D directly).
                    if let (Some(d), Some((fw, fh))) = (depth.as_ref(), dims) {
                        anchor_objects(people.as_slice(), objects.as_mut_slice(), d, &pinhole, fw, fh);
                    }
                }
                // Cross-group inference (phase 8.F) through the temporal
                // journal (phase 11.C3): raw per-tick activities are debounced
                // into stable instances + start/stop events.
                self.advance_journal(&mut w);
                ticks += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(15));
        }
        AgentHealth { name: self.name(), ticks, errors: 0 }
    }
}

/// Anchor objects to metric 3D using people as the scale reference (phase 8.E.2,
/// now on the fusion step). For each person with a metric torso-Z and a depth
/// sample, `k_i = Z_i·d_i`; the median `k` converts each object's depth value to
/// metric `Z = k / d_obj`, back-projected through the same pinhole. With no
/// person anchor, objects are left depthless (overlap-only) — no fabricated
/// distance. Mirrors the synchronous `anchor_objects_depth`.
pub(crate) fn anchor_objects(
    people: &[PersonState],
    objects: &mut [ObjectState],
    depth: &mvm_depth::DepthMap,
    pinhole: &mvm_core::space::Pinhole,
    fw: f32,
    fh: f32,
) {
    const OBJ_DEPTH_CONF: f32 = 0.4;
    const Z_MIN: f32 = 0.3;
    const Z_MAX: f32 = 8.0;

    let mut ks: Vec<f32> = Vec::new();
    for p in people {
        if let Some(q) = p.position_3d {
            if q.z > 0.0 {
                if let Some(d) = depth.median_in_bbox(&p.bbox) {
                    if d > 1e-3 {
                        ks.push(q.z * d);
                    }
                }
            }
        }
    }
    if ks.is_empty() {
        for o in objects.iter_mut() {
            o.position_3d = None;
        }
        return;
    }
    ks.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let k = ks[ks.len() / 2];

    for o in objects.iter_mut() {
        // Mask-median depth (phase 11): when a YOLO-seg mask rides the object,
        // sample depth over the silhouette (`median_at_points`) instead of the
        // bbox — the bbox bleeds background/foreground depth. Mirrors the sync
        // `anchor_objects_depth`; falls back to bbox-median when no mask.
        let obj_depth = match &o.mask {
            Some(m) if !m.is_empty() => depth.median_at_points(m.set_points()),
            _ => depth.median_in_bbox(&o.bbox),
        };
        match obj_depth {
            Some(d) if d > 1e-3 => {
                let z = (k / d).clamp(Z_MIN, Z_MAX);
                let cx = (o.bbox.x1 + o.bbox.x2) * 0.5 * fw;
                let cy = (o.bbox.y1 + o.bbox.y2) * 0.5 * fh;
                let p3 = pinhole.pixel_to_3d(cx, cy, z);
                o.position_3d = Some(Position3D { x: p3[0], y: p3[1], z: p3[2], confidence: OBJ_DEPTH_CONF });
            }
            _ => o.position_3d = None,
        }
    }
}

// ── Cross-group inference (phase 8.F) — pure arithmetic ───────────────────────

/// Two people closer than this (metric 3D) count as "together".
const ACTIVITY_NEAR_M: f32 = 1.5;
/// A hand point within this normalised margin of an object bbox counts as contact
/// (absorbs landmark jitter and tight object boxes).
const CONTACT_MARGIN: f32 = 0.03;
/// Normalised margin from the frame border within which a bbox counts as
/// clipped by the frame edge (occlusion context `partially_in_frame`).
const FRAME_EDGE_MARGIN: f32 = 0.02;
/// Expanded-bbox margin for the person↔workstation adjacency test — the desk
/// object need only be adjacent to the body, not under its hands.
const WORKSTATION_MARGIN: f32 = 0.10;

/// Objects that constitute a workstation context (the desk case). A person with
/// `body_coverage: "upper"` + upright torso + no locomotion, adjacent to one of
/// these, is *seated at it* — the relational inference that replaces the
/// withheld sitting-vs-standing classifier claim (legs occluded behind a desk
/// are geometrically mute on the difference).
const WORKSTATION_CLASSES: &[&str] =
    &["dining table", "laptop", "keyboard", "mouse", "tv", "book"];

/// The activity label a person↔object interaction yields, or `None` for objects
/// that are context only (furniture you don't "use" as an activity).
///
/// Vocabulary v2 (phase 15.6, LJ4/S3 VRD audit — contract change #2): the flat
/// verbs `typing`/`using_mouse`/`using_phone`/`reading` had **0–15** supervisable
/// VRD instances — unsupervisable *names* for what the data only labels as "a
/// person holds/uses X" (`holds`: 2153). They fold into a single honest umbrella
/// `using_object(<class>)`, the class riding as a parameter, not a fabricated
/// verb. `holding_drink` is **kept** — the one specialization with real support
/// (145 instances). `watching_tv` is **dropped** entirely: wrist-on-remote/tv is
/// not "watching" (that is head-facing, a different signal), and VRD has 0.
fn interaction_label(class: &str) -> Option<String> {
    match class {
        // The one specialization with real VRD support — kept as a named verb.
        "cup" | "bottle" | "wine glass" | "bowl" => Some("holding_drink".to_string()),
        // Watching is head-facing, not a wrist-on-object contact → no claim.
        "remote" | "tv" => None,
        // Everything else a wrist can touch is an honest generic "uses X".
        "keyboard" | "laptop" | "mouse" | "cell phone" | "book" => {
            Some(format!("using_object({class})"))
        }
        _ => None,
    }
}

/// A normalised point lies within `b` expanded by `CONTACT_MARGIN`.
fn point_near_bbox(pt: [f32; 2], b: &BBox) -> bool {
    pt[0] >= b.x1 - CONTACT_MARGIN
        && pt[0] <= b.x2 + CONTACT_MARGIN
        && pt[1] >= b.y1 - CONTACT_MARGIN
        && pt[1] <= b.y2 + CONTACT_MARGIN
}

/// Mask-contact dilation (phase 11.A1 / 10.6): grid cells of slack around the
/// silhouette, the mask analogue of `CONTACT_MARGIN` (2 cells ≈ 0.03 at 64²).
const MASK_CONTACT_CELLS: usize = 2;

/// Hand/wrist ↔ object contact. With an instance silhouette on the object the
/// test is mask-containment (background inside the bbox no longer counts —
/// phase 10.6 mask-contact); without one (masks off, no seg match, or a
/// degenerate empty mask) it falls back to the bbox test.
fn point_touches_object(pt: [f32; 2], o: &ObjectState) -> bool {
    match &o.mask {
        Some(m) if !m.is_empty() => m.contains_norm_within(pt, MASK_CONTACT_CELLS),
        _ => point_near_bbox(pt, &o.bbox),
    }
}

/// Where a contact "belongs" on the object: the silhouette centroid when a mask
/// exists, the bbox centre otherwise. The distance reference for contested
/// attribution (phase 11.A2).
fn contact_centre(o: &ObjectState) -> [f32; 2] {
    if let Some(m) = &o.mask {
        let (mut sx, mut sy, mut n) = (0.0f32, 0.0f32, 0u32);
        for [x, y] in m.set_points() {
            sx += x;
            sy += y;
            n += 1;
        }
        if n > 0 {
            return [sx / n as f32, sy / n as f32];
        }
    }
    [(o.bbox.x1 + o.bbox.x2) * 0.5, (o.bbox.y1 + o.bbox.y2) * 0.5]
}

/// One person's claim on a contested object (phase 11.A2): how they touch it
/// and how close. Ordering (strongest first): hand-landmark contact beats
/// wrist-only; a wrist-only touch while the arm is *reaching* loses to a plain
/// wrist touch (an outstretched arm crossing an object is a reach-through, not
/// a hold); then distance to the contact centre; then track id (determinism).
struct ContactClaim {
    track_id: u32,
    hand_touch: bool,
    reaching_wrist_only: bool,
    dist2: f32,
}

impl ContactClaim {
    /// `self` is a stronger claim than `other`.
    fn beats(&self, other: &ContactClaim) -> bool {
        if self.hand_touch != other.hand_touch {
            return self.hand_touch;
        }
        if self.reaching_wrist_only != other.reaching_wrist_only {
            return !self.reaching_wrist_only;
        }
        if self.dist2 != other.dist2 {
            return self.dist2 < other.dist2;
        }
        self.track_id < other.track_id
    }
}

/// This person's claim on the object, or `None` when nothing touches it.
fn contact_claim(p: &PersonState, o: &ObjectState, centre: [f32; 2]) -> Option<ContactClaim> {
    let mut hand_touch = false;
    let mut wrist_touch = false;
    let mut dist2 = f32::INFINITY;
    let mut consider = |pt: [f32; 2], is_hand: bool| {
        if point_touches_object(pt, o) {
            if is_hand {
                hand_touch = true;
            } else {
                wrist_touch = true;
            }
            let (dx, dy) = (pt[0] - centre[0], pt[1] - centre[1]);
            dist2 = dist2.min(dx * dx + dy * dy);
        }
    };
    for &pt in &p.wrists {
        consider(pt, false);
    }
    for h in &p.hands {
        for &pt in &h.landmarks {
            consider(pt, true);
        }
    }
    if !hand_touch && !wrist_touch {
        return None;
    }
    let reaching = p.left_gesture == Some("reaching") || p.right_gesture == Some("reaching");
    Some(ContactClaim {
        track_id: p.track_id,
        hand_touch,
        reaching_wrist_only: !hand_touch && reaching,
        dist2,
    })
}

fn dist3(a: Position3D, b: Position3D) -> f32 {
    let (dx, dy, dz) = (a.x - b.x, a.y - b.y, a.z - b.z);
    (dx * dx + dy * dy + dz * dz).sqrt()
}

/// Two normalised bboxes intersect once `a` is expanded by `margin` on all sides.
fn bbox_adjacent(a: &BBox, b: &BBox, margin: f32) -> bool {
    a.x1 - margin < b.x2 && b.x1 < a.x2 + margin && a.y1 - margin < b.y2 && b.y1 < a.y2 + margin
}

/// The person's bbox is clipped by the frame border (a body walking out of view
/// or standing half-outside — coverage is then structurally partial).
fn bbox_at_frame_edge(b: &BBox) -> bool {
    b.x1 <= FRAME_EDGE_MARGIN
        || b.y1 <= FRAME_EDGE_MARGIN
        || b.x2 >= 1.0 - FRAME_EDGE_MARGIN
        || b.y2 >= 1.0 - FRAME_EDGE_MARGIN
}

/// Whether this person is in the *stationary upper-body* regime the workstation
/// inference needs: lower body unseen, torso upright, not locomoting.
fn upper_upright_stationary(p: &PersonState) -> bool {
    p.body_coverage == Some("upper")
        && p.torso_attitude == Some("upright")
        && p.motion_state != Some("moving")
}

/// Infer cross-group activities from the fused world (phase 8.F). Pure arithmetic:
///   - person↔person within `ACTIVITY_NEAR_M` (metric) → "people_together";
///   - person↔interactive-object: a **hand/wrist landmark** of the person lands
///     inside the object bbox → a label (`using_object(<class>)`, or the kept
///     `holding_drink`). Body-landmark contact, not body-bbox overlap — on a
///     ceiling cam the wrists reach the desk while the body bbox (head/shoulders)
///     does not.
/// Furniture is context only (no activity).
pub(crate) fn infer_activities(people: &[PersonState], objects: &[ObjectState]) -> Vec<Activity> {
    let mut acts = Vec::new();

    // person ↔ person proximity (needs both torso-3D positions).
    for i in 0..people.len() {
        for j in (i + 1)..people.len() {
            if let (Some(pa), Some(pb)) = (people[i].position_3d, people[j].position_3d) {
                let d = dist3(pa, pb);
                if d < ACTIVITY_NEAR_M {
                    acts.push(Activity {
                        label: "people_together".into(),
                        confidence: (1.0 - d / ACTIVITY_NEAR_M).clamp(0.0, 1.0),
                        participants: vec![people[i].track_id, people[j].track_id],
                    });
                }
            }
        }
    }

    // person ↔ interactive object by hand/wrist contact. Object-major (phase
    // 11.A2): when points of several people land in one object (an outstretched
    // arm crossing a neighbour's phone), the object is attributed to exactly
    // one person — the strongest ContactClaim — instead of everyone touching.
    for o in objects {
        let Some(label) = interaction_label(&o.class) else { continue };
        let centre = contact_centre(o);
        let mut best: Option<ContactClaim> = None;
        for p in people {
            if let Some(claim) = contact_claim(p, o, centre) {
                if best.as_ref().is_none_or(|b| claim.beats(b)) {
                    best = Some(claim);
                }
            }
        }
        if let Some(winner) = best {
            acts.push(Activity {
                label,
                confidence: o.confidence.clamp(0.0, 1.0),
                participants: vec![winner.track_id, o.id],
            });
        }
    }

    // ── Occlusion contexts (occlusion-aware posture) ────────────────────────
    // When the classifier honestly withholds the posture (body_coverage
    // "upper"/"head"), fusion names the *context* of the occlusion instead —
    // that relational claim is what downstream actually wants, and it is robust
    // to any particular desk/furniture.
    for p in people {
        // Seated at a workstation: upper-body regime + upright torso + not
        // locomoting, adjacent to a workstation object. This is the desk case:
        // sitting-vs-standing is geometrically mute without legs, but "at the
        // workstation, stationary, upright" is exactly the occupancy signal.
        if upper_upright_stationary(p) {
            if let Some(o) = objects.iter().find(|o| {
                WORKSTATION_CLASSES.contains(&o.class.as_str())
                    && bbox_adjacent(&p.bbox, &o.bbox, WORKSTATION_MARGIN)
            }) {
                acts.push(Activity {
                    label: "seated_at_workstation".into(),
                    confidence: o.confidence.clamp(0.0, 1.0),
                    participants: vec![p.track_id, o.id],
                });
            }
        }
        // Clipped by the frame border with partial coverage: the body is
        // half-outside the camera's view — flag it so consumers don't read the
        // missing lower body as a scene fact.
        if p.body_coverage.is_some_and(|c| c != "full") && bbox_at_frame_edge(&p.bbox) {
            acts.push(Activity {
                label: "partially_in_frame".into(),
                confidence: 1.0,
                participants: vec![p.track_id],
            });
        }
    }

    acts
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::Config;
    use std::time::Instant;

    fn pinhole() -> mvm_core::space::Pinhole {
        mvm_core::space::Pinhole::from_config(&Config::default())
    }

    fn person(z: f32) -> PersonState {
        PersonState {
            track_id: 10,
            bbox: BBox { x1: 0.3, y1: 0.2, x2: 0.5, y2: 0.7 },
            identity: None,
            emotion: None,
            posture: None,
            position_3d: Some(Position3D { x: 0.0, y: 0.0, z, confidence: 0.5 }),
            emotion_distribution: None,
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: None,
            torso_attitude: None,
            motion_state: None,
            velocity: None,
            left_gesture: None,
            right_gesture: None,
            recon_error: None,
            plausibility: None,
            torso_flexion: None,
            head_yaw: None, head_facing: None,
            last_seen: Instant::now(),
        }
    }

    fn object() -> ObjectState {
        ObjectState {
            id: 10000,
            class: "cup".into(),
            bbox: BBox { x1: 0.45, y1: 0.45, x2: 0.6, y2: 0.6 },
            confidence: 0.8,
            age: 2,
            position_3d: Some(Position3D { x: 9.0, y: 9.0, z: 9.0, confidence: 0.4 }), // stale
            mask: None,
            remembered: false,
            last_seen: Instant::now(),
        }
    }

    /// Uniform depth map → every `median_in_bbox` returns the same value.
    fn uniform_depth(v: f32) -> mvm_depth::DepthMap {
        mvm_depth::DepthMap { data: vec![v; 64], w: 8, h: 8 }
    }

    // ── mesh journal wiring (phase 11.C3) ──

    fn fusion_agent() -> FusionAgent {
        let slot: DepthSlot = std::sync::Arc::new(std::sync::Mutex::new(None));
        FusionAgent::new(Config::default(), slot)
    }

    /// A world whose raw inference yields "using_object(keyboard)" (wrist on keyboard).
    fn typing_world(now: Instant, frame_id: u64) -> mvm_core::WorldState {
        let mut w = mvm_core::WorldState::new(now);
        w.frame_id = frame_id;
        let mut p = person(1.0);
        p.position_3d = None;
        p.wrists = vec![[0.40, 0.58]];
        w.people.push(p);
        w.objects.push(obj_class(10001, "keyboard", BBox { x1: 0.35, y1: 0.55, x2: 0.45, y2: 0.62 }));
        w
    }

    #[test]
    fn repeated_raw_debounces_then_journals_an_event() {
        let mut agent = fusion_agent();
        let t0 = Instant::now();
        // Frames 0-1: raw "using_object(keyboard)" present but below the enter dwell -> no claim.
        for f in 0..2u64 {
            let mut w = typing_world(t0 + std::time::Duration::from_millis(100 * f), f);
            agent.advance_journal(&mut w);
            assert!(w.activities.is_empty(), "frame {f}: still a candidate");
        }
        // Frame 2: third consecutive frame -> Active.
        let mut w = typing_world(t0 + std::time::Duration::from_millis(200), 2);
        agent.advance_journal(&mut w);
        assert_eq!(w.activities.len(), 1, "debounced activity emitted");
        assert_eq!(w.activities[0].label, "using_object(keyboard)");
        assert!(w.events.is_empty());
        // Keep it alive past D_MIN so the base exit dwell applies, then vanish.
        let mut f = 3u64;
        while f < 33 {
            let mut w = typing_world(t0 + std::time::Duration::from_millis(100 * f), f);
            agent.advance_journal(&mut w);
            f += 1;
        }
        let mut last_events = Vec::new();
        for g in 0..mvm_core::activity::ACTIVITY_EXIT_DWELL as u64 {
            let mut w = mvm_core::WorldState::new(t0 + std::time::Duration::from_millis(100 * (f + g)));
            w.frame_id = f + g;
            agent.advance_journal(&mut w);
            last_events = w.events.clone();
        }
        assert_eq!(last_events.len(), 1, "end of contact journals one event");
        assert_eq!(last_events[0].label, "using_object(keyboard)");
        assert!(last_events[0].duration_s > 2.0, "duration {}", last_events[0].duration_s);
    }

    #[test]
    fn stale_frame_id_does_not_advance_dwell() {
        // The fusion loop spins ~66Hz on the same frame; dwell counts frames.
        let mut agent = fusion_agent();
        let t0 = Instant::now();
        for _ in 0..10 {
            let mut w = typing_world(t0, 5);
            agent.advance_journal(&mut w);
            assert!(w.activities.is_empty(), "same frame repeated must not confirm");
        }
    }

    #[test]
    fn no_people_leaves_objects_depthless() {
        let mut objs = vec![object()];
        anchor_objects(&[], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        assert!(objs[0].position_3d.is_none(), "no anchor → depthless");
    }

    #[test]
    fn person_anchor_sets_object_metric_z() {
        // person Z=2.0, d_person=0.5 → k=1.0; object d=0.5 → Z = 1.0/0.5 = 2.0.
        let mut objs = vec![object()];
        anchor_objects(&[person(2.0)], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        let q = objs[0].position_3d.expect("object anchored");
        assert!((q.z - 2.0).abs() < 1e-3, "metric Z from people anchor, got {}", q.z);
        assert!((q.confidence - 0.4).abs() < 1e-6);
    }

    #[test]
    fn degenerate_object_bbox_yields_no_position() {
        // A zero-area object bbox → median_in_bbox None → depthless even with anchor.
        let mut o = object();
        o.bbox = BBox { x1: 0.5, y1: 0.5, x2: 0.5, y2: 0.5 };
        let mut objs = vec![o];
        anchor_objects(&[person(2.0)], &mut objs, &uniform_depth(0.5), &pinhole(), 640.0, 480.0);
        assert!(objs[0].position_3d.is_none(), "degenerate bbox → None");
    }

    #[test]
    fn mask_median_depth_beats_bbox_on_mesh() {
        // Phase 11 mesh mask-aware depth: a two-region map (left half depth 1.0,
        // right half 0.25). Person + object bbox sit in the LEFT region (bbox
        // central-half samples 1.0). A silhouette mask whose set cells fall in the
        // RIGHT region makes `anchor_objects` sample 0.25 via `median_at_points` —
        // a different metric Z than the bbox path, proving the mask branch is live.
        let mut data = vec![0.0f32; 64];
        for gy in 0..8 {
            for gx in 0..8 {
                data[gy * 8 + gx] = if gx < 4 { 1.0 } else { 0.25 };
            }
        }
        let depth = mvm_depth::DepthMap { data, w: 8, h: 8 };

        // Mask cells in the right region (normalized x ≥ 0.5 → depth 0.25).
        let mut bits = vec![0u8; mvm_core::mask::MASK_RES * mvm_core::mask::MASK_RES];
        for gy in 32..40 {
            for gx in 32..40 {
                bits[gy * mvm_core::mask::MASK_RES + gx] = 1;
            }
        }
        let mask = mvm_core::mask::InstanceMask::from_bits(bits);

        let mut o = object();
        o.bbox = BBox { x1: 0.1, y1: 0.1, x2: 0.4, y2: 0.4 }; // left region
        let people = [person(1.0)]; // k = 1.0·1.0 = 1.0

        // Bbox path: central-half samples 1.0 → Z = k/1.0 = 1.0.
        let mut bbox_objs = vec![o.clone()];
        anchor_objects(&people, &mut bbox_objs, &depth, &pinhole(), 1280.0, 720.0);
        let z_bbox = bbox_objs[0].position_3d.expect("bbox anchored").z;
        assert!((z_bbox - 1.0).abs() < 1e-3, "bbox-median Z, got {z_bbox}");

        // Mask path: silhouette samples 0.25 → Z = k/0.25 = 4.0.
        o.mask = Some(mask);
        let mut mask_objs = vec![o];
        anchor_objects(&people, &mut mask_objs, &depth, &pinhole(), 1280.0, 720.0);
        let z_mask = mask_objs[0].position_3d.expect("mask anchored").z;
        assert!((z_mask - 4.0).abs() < 1e-3, "mask-median Z, got {z_mask}");
    }

    // ── activities + anomaly (phase 8.F) ──

    fn obj_class(id: u32, class: &str, bbox: BBox) -> ObjectState {
        ObjectState {
            id,
            class: class.into(),
            bbox,
            confidence: 0.8,
            age: 1,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: Instant::now(),
        }
    }

    #[test]
    fn typing_when_wrist_on_keyboard() {
        // Wrist landmark lands inside the keyboard bbox — body-landmark contact,
        // not body-bbox overlap.
        let mut p = person(1.0);
        let kb = obj_class(10001, "keyboard", BBox { x1: 0.35, y1: 0.55, x2: 0.45, y2: 0.62 });
        p.wrists = vec![[0.40, 0.58]];
        let acts = infer_activities(&[p], &[kb]);
        assert!(
            acts.iter().any(|a| a.label == "using_object(keyboard)" && a.participants == vec![10, 10001]),
            "got {acts:?}"
        );
    }

    // ── mask-contact (phase 11.A1 / 10.6) ──

    /// A mask whose set cells are exactly those with centres inside `r`.
    fn region_mask(r: &BBox) -> mvm_core::InstanceMask {
        use mvm_core::MASK_RES;
        let mut bits = vec![0u8; MASK_RES * MASK_RES];
        for gy in 0..MASK_RES {
            for gx in 0..MASK_RES {
                let nx = (gx as f32 + 0.5) / MASK_RES as f32;
                let ny = (gy as f32 + 0.5) / MASK_RES as f32;
                if nx >= r.x1 && nx <= r.x2 && ny >= r.y1 && ny <= r.y2 {
                    bits[gy * MASK_RES + gx] = 1;
                }
            }
        }
        mvm_core::InstanceMask::from_bits(bits)
    }

    #[test]
    fn mask_contact_cuts_bbox_background() {
        // Wrist inside the object's *bbox* but far outside its *silhouette* —
        // the background contact the bbox test wrongly accepted. With a mask
        // attached, the claim must not fire.
        let mut p = person(1.0);
        let bbox = BBox { x1: 0.2, y1: 0.5, x2: 0.8, y2: 0.7 };
        let mut kb = obj_class(10001, "keyboard", bbox);
        kb.mask = Some(region_mask(&BBox { x1: 0.2, y1: 0.5, x2: 0.45, y2: 0.7 }));
        p.wrists = vec![[0.7, 0.6]]; // in bbox, ~0.25 right of the silhouette
        assert!(
            infer_activities(&[p], &[kb]).iter().all(|a| a.label != "using_object(keyboard)"),
            "bbox-only background contact must be cut by the mask"
        );
    }

    #[test]
    fn mask_contact_fires_on_silhouette() {
        let mut p = person(1.0);
        let bbox = BBox { x1: 0.2, y1: 0.5, x2: 0.8, y2: 0.7 };
        let mut kb = obj_class(10001, "keyboard", bbox);
        kb.mask = Some(region_mask(&BBox { x1: 0.2, y1: 0.5, x2: 0.45, y2: 0.7 }));
        p.wrists = vec![[0.3, 0.6]]; // on the silhouette
        let acts = infer_activities(&[p], &[kb]);
        assert!(
            acts.iter().any(|a| a.label == "using_object(keyboard)" && a.participants == vec![10, 10001]),
            "got {acts:?}"
        );
    }

    // ── contact attribution (phase 11.A2) ──

    fn person_id(id: u32) -> PersonState {
        let mut p = person(1.0);
        p.track_id = id;
        p.position_3d = None; // keep people_together out of these tests
        p
    }

    fn hand_at(pt: [f32; 2]) -> mvm_core::HandState {
        mvm_core::HandState {
            landmarks: vec![pt],
            presence: 0.9,
            side: "right",
            handedness: 0.9,
            gesture: None,
        }
    }

    #[test]
    fn contested_object_goes_to_hand_not_reaching_wrist() {
        // A's outstretched wrist crosses B's phone while B actually holds it
        // (hand landmarks on the object) — the activity belongs to B, and only B.
        let phone = obj_class(10001, "cell phone", BBox { x1: 0.5, y1: 0.5, x2: 0.6, y2: 0.6 });
        let mut a = person_id(10);
        a.wrists = vec![[0.51, 0.55]]; // even closer to the centre than B's hand
        a.right_gesture = Some("reaching");
        let mut b = person_id(11);
        b.hands = vec![hand_at([0.58, 0.55])];
        let acts = infer_activities(&[a, b], &[phone]);
        let phone_acts: Vec<_> = acts.iter().filter(|x| x.label == "using_object(cell phone)").collect();
        assert_eq!(phone_acts.len(), 1, "one attribution per contested object, got {acts:?}");
        assert_eq!(phone_acts[0].participants, vec![11, 10001], "holder wins over reacher");
    }

    #[test]
    fn contested_object_goes_to_closest_wrist() {
        // Both touch wrist-only, nobody reaching → the wrist nearest the object
        // centre wins.
        let cup = obj_class(10002, "cup", BBox { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 });
        let mut a = person_id(10);
        a.wrists = vec![[0.41, 0.5]]; // at the edge
        let mut b = person_id(11);
        b.wrists = vec![[0.5, 0.5]]; // dead centre
        let acts = infer_activities(&[a, b], &[cup]);
        let cup_acts: Vec<_> = acts.iter().filter(|x| x.label == "holding_drink").collect();
        assert_eq!(cup_acts.len(), 1, "got {acts:?}");
        assert_eq!(cup_acts[0].participants, vec![11, 10002]);
    }

    #[test]
    fn uncontested_contact_still_fires() {
        // A single person touching stays attributed exactly as before 11.A2.
        let cup = obj_class(10002, "cup", BBox { x1: 0.4, y1: 0.4, x2: 0.6, y2: 0.6 });
        let mut a = person_id(10);
        a.wrists = vec![[0.41, 0.5]];
        a.right_gesture = Some("reaching"); // reaching alone must not silence it
        let acts = infer_activities(&[a], &[cup]);
        assert!(acts.iter().any(|x| x.label == "holding_drink" && x.participants == vec![10, 10002]),
            "got {acts:?}");
    }

    #[test]
    fn two_objects_two_people_attribute_independently() {
        let kb1 = obj_class(10001, "keyboard", BBox { x1: 0.1, y1: 0.5, x2: 0.3, y2: 0.6 });
        let kb2 = obj_class(10002, "keyboard", BBox { x1: 0.6, y1: 0.5, x2: 0.8, y2: 0.6 });
        let mut a = person_id(10);
        a.wrists = vec![[0.2, 0.55]];
        let mut b = person_id(11);
        b.wrists = vec![[0.7, 0.55]];
        let acts = infer_activities(&[a, b], &[kb1, kb2]);
        assert!(acts.iter().any(|x| x.label == "using_object(keyboard)" && x.participants == vec![10, 10001]));
        assert!(acts.iter().any(|x| x.label == "using_object(keyboard)" && x.participants == vec![11, 10002]));
    }

    #[test]
    fn empty_mask_falls_back_to_bbox_contact() {
        // A degenerate (all-empty) mask must not silence the interaction — fall
        // back to the bbox test rather than claim "no silhouette anywhere".
        let mut p = person(1.0);
        let kb = obj_class(10001, "keyboard", BBox { x1: 0.35, y1: 0.55, x2: 0.45, y2: 0.62 });
        let mut kb = kb;
        kb.mask = Some(mvm_core::InstanceMask::empty());
        p.wrists = vec![[0.40, 0.58]];
        assert!(
            infer_activities(&[p], &[kb]).iter().any(|a| a.label == "using_object(keyboard)"),
            "empty mask → honest bbox fallback"
        );
    }

    #[test]
    fn hand_landmark_triggers_object_use() {
        // No wrists, but a detected hand landmark on the mouse → "using_object(mouse)".
        let mut p = person(1.0);
        p.hands = vec![mvm_core::HandState {
            landmarks: vec![[0.33, 0.57], [0.34, 0.56]],
            presence: 0.9,
            side: "right",
            handedness: 0.9,
            gesture: None,
        }];
        let mouse = obj_class(10001, "mouse", BBox { x1: 0.31, y1: 0.54, x2: 0.36, y2: 0.60 });
        let acts = infer_activities(&[p], &[mouse]);
        assert!(acts.iter().any(|a| a.label == "using_object(mouse)"), "got {acts:?}");
    }

    #[test]
    fn no_activity_when_landmarks_far_from_object() {
        let mut p = person(1.0);
        let kb = obj_class(10001, "keyboard", BBox { x1: 0.8, y1: 0.8, x2: 0.9, y2: 0.85 });
        p.wrists = vec![[0.2, 0.2]];
        assert!(infer_activities(&[p], &[kb]).is_empty());
    }

    #[test]
    fn no_landmarks_means_no_object_activity() {
        // Object sits inside the body bbox, but the person has no wrists/hands →
        // we don't fabricate a contact claim from bbox overlap alone.
        let p = person(1.0); // wrists/hands empty
        let mouse = obj_class(10001, "mouse", BBox { x1: 0.35, y1: 0.3, x2: 0.45, y2: 0.4 });
        assert!(infer_activities(&[p], &[mouse]).is_empty());
    }

    #[test]
    fn furniture_is_context_not_activity() {
        let mut p = person(1.0);
        p.wrists = vec![[0.40, 0.60]];
        let chair = obj_class(10001, "chair", BBox { x1: 0.35, y1: 0.55, x2: 0.45, y2: 0.65 });
        assert!(infer_activities(&[p], &[chair]).is_empty(), "chair is context, not an activity");
    }

    #[test]
    fn people_together_when_near() {
        let mut a = person(1.0);
        a.track_id = 10;
        let mut b = person(1.0);
        b.track_id = 11;
        let acts = infer_activities(&[a, b], &[]);
        assert!(
            acts.iter().any(|x| x.label == "people_together" && x.participants == vec![10, 11]),
            "got {acts:?}"
        );
    }

    #[test]
    fn anti_pet_gate_drops_animal_body_from_people_together() {
        // Mesh parity (12.B3): a body overlapping a YOLO animal detection (a pet
        // with a spurious BlazePose skeleton) must never be a `people_together`
        // participant. Control run (no animals) confirms the pair would fire.
        let t0 = std::time::Instant::now();
        let mut a = person(1.0);
        a.track_id = 10;
        a.bbox = BBox { x1: 0.1, y1: 0.2, x2: 0.3, y2: 0.7 };
        a.position_3d = Some(Position3D { x: 0.0, y: 0.0, z: 1.0, confidence: 0.5 });
        let mut b = person(1.0);
        b.track_id = 11;
        b.bbox = BBox { x1: 0.6, y1: 0.5, x2: 0.95, y2: 0.9 }; // low-wide pet-shaped box
        b.position_3d = Some(Position3D { x: 0.3, y: 0.0, z: 1.0, confidence: 0.5 });

        let together_seen = |animals: Vec<BBox>| {
            let mut agent = fusion_agent();
            let mut seen = false;
            for f in 0..5u64 {
                let mut w = mvm_core::WorldState::new(t0 + std::time::Duration::from_millis(100 * f));
                w.frame_id = f;
                w.people = vec![a.clone(), b.clone()];
                w.animals = animals.clone();
                agent.advance_journal(&mut w);
                if w.activities.iter().any(|x| x.label == "people_together") {
                    seen = true;
                }
            }
            seen
        };

        assert!(together_seen(vec![]), "control: the pair should fire without animals");
        assert!(
            !together_seen(vec![b.bbox]),
            "gate: pet-overlapping body must not join people_together"
        );
    }

    // ── occlusion contexts (occlusion-aware posture) ──

    /// The live desk scenario: upper coverage, upright torso, idle, next to a
    /// workstation object → `seated_at_workstation`.
    fn desk_person() -> PersonState {
        let mut p = person(1.0);
        p.body_coverage = Some("upper");
        p.torso_attitude = Some("upright");
        p.motion_state = Some("idle");
        p
    }

    #[test]
    fn seated_at_workstation_inferred_at_desk() {
        let laptop = obj_class(10001, "laptop", BBox { x1: 0.45, y1: 0.55, x2: 0.65, y2: 0.75 });
        let acts = infer_activities(&[desk_person()], &[laptop]);
        assert!(
            acts.iter().any(|a| a.label == "seated_at_workstation"
                && a.participants == vec![10, 10001]),
            "got {acts:?}"
        );
    }

    #[test]
    fn no_workstation_claim_when_moving() {
        // Walking past the desk with clipped legs is not "seated at it".
        let mut p = desk_person();
        p.motion_state = Some("moving");
        let laptop = obj_class(10001, "laptop", BBox { x1: 0.45, y1: 0.55, x2: 0.65, y2: 0.75 });
        assert!(infer_activities(&[p], &[laptop])
            .iter()
            .all(|a| a.label != "seated_at_workstation"));
    }

    #[test]
    fn no_workstation_claim_with_full_coverage() {
        // Full-body view → the classifier already claims the posture; the
        // relational fallback stays quiet.
        let mut p = desk_person();
        p.body_coverage = Some("full");
        let laptop = obj_class(10001, "laptop", BBox { x1: 0.45, y1: 0.55, x2: 0.65, y2: 0.75 });
        assert!(infer_activities(&[p], &[laptop])
            .iter()
            .all(|a| a.label != "seated_at_workstation"));
    }

    #[test]
    fn no_workstation_claim_without_adjacent_object() {
        // Upper coverage but nothing desk-like nearby (occluded by other
        // furniture) → no fabricated workstation claim.
        let far = obj_class(10001, "laptop", BBox { x1: 0.85, y1: 0.85, x2: 0.95, y2: 0.95 });
        let chair = obj_class(10002, "chair", BBox { x1: 0.45, y1: 0.55, x2: 0.65, y2: 0.75 });
        assert!(infer_activities(&[desk_person()], &[far, chair])
            .iter()
            .all(|a| a.label != "seated_at_workstation"));
    }

    #[test]
    fn partially_in_frame_flagged_at_edge() {
        let mut p = desk_person();
        p.bbox = BBox { x1: 0.0, y1: 0.2, x2: 0.2, y2: 0.7 }; // clipped left
        let acts = infer_activities(&[p], &[]);
        assert!(
            acts.iter().any(|a| a.label == "partially_in_frame" && a.participants == vec![10]),
            "got {acts:?}"
        );
        // Full coverage in the middle of the frame → no flag.
        let clean = person(1.0);
        assert!(infer_activities(&[clean], &[])
            .iter()
            .all(|a| a.label != "partially_in_frame"));
    }

    #[test]
    fn anomaly_rises_for_unexpected_object() {
        // v2 (11.B2): the scorer rides the same frame gate as the journal.
        let mut agent = fusion_agent();
        let t0 = Instant::now();
        let mut calm = mvm_core::WorldState::new(t0);
        calm.objects.push(obj_class(10001, "cup", BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 }));
        agent.advance_journal(&mut calm);
        assert_eq!(calm.anomaly_score, 0.0, "cup is expected → no anomaly");

        let mut odd = mvm_core::WorldState::new(t0 + std::time::Duration::from_millis(100));
        odd.frame_id = 1;
        odd.objects.push(obj_class(10002, "knife", BBox { x1: 0.1, y1: 0.1, x2: 0.2, y2: 0.2 }));
        agent.advance_journal(&mut odd);
        assert!(odd.anomaly_score > 0.0, "knife is unexpected → anomaly");
    }
}
