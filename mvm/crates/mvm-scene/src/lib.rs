//! mvm-scene — object detection via an ultralytics YOLO ONNX model (80-class COCO).
//!
//! Phase 8.E.1 (G₂ Scene group). Produces `EntityKind::Object(class)` scene-graph
//! nodes so the existing object-agnostic `build_scene` (mvm-fusion) wires
//! person↔object relations — the "context outward from the human" signal.
//!
//! Model-agnostic by design — two ultralytics output layouts are handled, chosen
//! at runtime from the tensor shape:
//!   - **end-to-end / NMS-free** `[1, N, 6]` rows `[x1,y1,x2,y2,score,class]`
//!     (YOLOv10, **YOLO26** — the shipped model); already deduplicated, no NMS.
//!   - **raw anchor grid** `[1, 4+num_classes, num_anchors]` (YOLOv8/v11); needs
//!     argmax-decode + class-aware NMS.
//!
//! So YOLO26n ↔ YOLOv8n is a pure file swap.
//!
//! The decode / letterbox / NMS math is split into pure functions so the chain is
//! fully unit-tested on synthetic tensors with no model file present.

use anyhow::Context;
use image::DynamicImage;
use image::GenericImageView;
use mvm_core::types::LifecycleState;
use mvm_core::{
    AssociationObservation, BBox, TrackedAssociationGeometry, TrackerDecision, TrackerDecisionKind,
    TrackerDiagnostics, TrackerDiagnosticsSnapshot, TrackerDomain, TrackerFrameBinding,
};
use ort::{session::Session, value::Tensor};
use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub mod mask;
pub use mask::{InstanceMask, MASK_RES, assemble_mask};

/// COCO 80-class names, in the canonical ultralytics order.
pub const COCO_CLASSES: [&str; 80] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// One detected object. `bbox` is normalized [0,1] in the **original** image frame
/// (letterbox padding already undone), consistent with every other MVM detector.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjectDetection {
    pub class_id: usize,
    pub class_name: &'static str,
    pub bbox: BBox,
    pub score: f32,
}

/// Scene-graph id base for object nodes (phase 8.E). Person ids are 0–9 (reserved
/// users) and ≥10 (ephemeral bodies); objects sit far above to avoid collision in
/// zones/relations. The `ObjectTracker` allocates stable ids from here upward.
pub const OBJECT_ID_BASE: u32 = 10_000;

/// Indoor / interaction-relevant COCO classes (phase 8.E filter C). Excludes
/// `person` (owned by the body-primary tracker, group G₁), plus outdoor / vehicle
/// / animal / sports / perishable-food classes that are noise for a desk-mounted
/// camera. Edit this list to tune what reaches the scene-graph.
pub const RELEVANT_CLASSES: &[&str] = &[
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

/// Whether a COCO class is an interaction-relevant object worth a scene node.
pub fn is_relevant(class_name: &str) -> bool {
    RELEVANT_CLASSES.contains(&class_name)
}

/// COCO animal classes. Not scene nodes (excluded from `RELEVANT_CLASSES`), but
/// detected all the same so a body track overlapping one can be flagged as a
/// likely animal — a **model-corroborated** anti-phantom gate (phase 10.x): an
/// independent YOLO "dog"/"cat"/"bird" over the body box is strong evidence the
/// body is the pet, and a fallen human never lands inside a "dog" detection.
pub const ANIMAL_CLASSES: &[&str] = &[
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
];

/// Whether a COCO class is an animal (see [`ANIMAL_CLASSES`]).
pub fn is_animal(class_name: &str) -> bool {
    ANIMAL_CLASSES.contains(&class_name)
}

/// Body↔animal-detection bbox-IoU above which a body is flagged `likely_animal`
/// and dropped from relational fusion (phase 10.x anti-pet gate). Single source
/// for both the sync and mesh paths.
pub const ANIMAL_IOU: f32 = 0.4;

/// Large, visually-similar furniture prone to low-confidence flicker
/// (couch ↔ bed ↔ chair confusion). These need a higher score bar than small
/// interaction objects, which a single global threshold can't express.
const HIGH_BAR_CLASSES: &[&str] = &[
    "couch",
    "bed",
    "chair",
    "dining table",
    "potted plant",
    "tv",
    "refrigerator",
    "oven",
    "microwave",
    "sink",
    "toaster",
];

/// Static desk fixtures that do not move during use and are routinely occluded
/// by the user's own hands while in use (phase 11 object-permanence). Once
/// confirmed, the tracker keeps emitting them across a long hand-occlusion so a
/// keyboard doesn't vanish — and `typing` with it — the moment the hands land on
/// it. Kept deliberately narrow: things a person operates in place, not
/// furniture they sit on / carry.
pub const STATIC_CLASSES: &[&str] = &[
    "keyboard",
    "mouse",
    "laptop",
    "tv",
    "dining table",
    "remote",
];

/// Whether a class is a static desk fixture eligible for object-permanence hold.
pub fn is_static(class_name: &str) -> bool {
    STATIC_CLASSES.contains(&class_name)
}

/// Static desk tools the user's hands routinely cover *while operating them*
/// (phase 11): give them a slightly lower detection floor than the base so the
/// partial, hand-occluded view that first seeds the permanence hold still clears
/// the gate. Distinct from furniture, which needs a *higher* bar.
const LOW_BAR_CLASSES: &[&str] = &["keyboard", "mouse"];

/// Per-class minimum score (phase 8.E filter A, per-class). Small interaction
/// objects (cup, keyboard, mouse, phone…) keep the low `base` floor so they
/// survive a ceiling-angle / partial view; flicker-prone large furniture is held
/// to a higher bar; hand-covered desk tools (keyboard/mouse) get a slightly
/// lower floor to seed the permanence hold. `base` is `Config.object_score_thresh`.
pub fn class_score_threshold(class_name: &str, base: f32) -> f32 {
    if HIGH_BAR_CLASSES.contains(&class_name) {
        base.max(0.50)
    } else if LOW_BAR_CLASSES.contains(&class_name) {
        (base - 0.08).max(0.20)
    } else {
        base
    }
}

/// A temporally-tracked object: the same physical object keeps `id` across frames.
#[derive(Debug, Clone, PartialEq)]
pub struct TrackedObject {
    /// Stable scene-graph id (≥ OBJECT_ID_BASE), reused via a free-list.
    pub id: u32,
    pub class_name: &'static str,
    pub bbox: BBox,
    pub score: f32,
    /// Consecutive frames this track has been seen.
    pub age: u32,
    /// Consecutive frames missed (0 when currently visible).
    pub missed: u32,
    /// Explicit publication lifecycle. Objects never use `IdLocked`.
    pub lifecycle: LifecycleState,
    /// Metric 3D position [x,y,z], set by the CLI from depth-anchoring (phase
    /// 8.E.2) — `None` when no depth map / no metric anchor is available. The
    /// tracker itself never sets this (it has no depth); it is filled in after
    /// tracking.
    pub pos_3d: Option<[f32; 3]>,
    /// Confidence of `pos_3d` (0 = none → overlap-only in the scene-graph).
    pub pos_conf: f32,
    /// Instance mask from the YOLO-seg expert (phase 10.4), attached by
    /// [`attach_masks`] on the seg cadence and refreshed each tick — `None` when
    /// masks are disabled (`--masks` off) or no seg detection matched this object.
    /// The tracker never sets this (it has no seg model); it rides on the object
    /// so depth-by-mask (`median_at_points`) and mask-contact fusion can read it.
    pub mask: Option<InstanceMask>,
}

/// Lightweight greedy-IoU object tracker (phase 8.E filter B).
///
/// Gives objects **stable ids across cadence frames** so `id 10000` is the same
/// physical object over time (unlike the old per-frame positional id). Matching is
/// class-aware; ids are reused from a free-list on expiry. Far simpler than the
/// person Kalman tracker — objects are largely static, so per-frame IoU + a short
/// miss-timeout is enough, and it absorbs the one-frame flicker YOLO produces.
pub struct ObjectTracker {
    tracks: Vec<TrackedObject>,
    next_id: u32,
    free_ids: BinaryHeap<Reverse<u32>>,
    iou_thresh: f32,
    timeout: u32,
    ema_alpha: f32,
    /// Object-permanence hold for confirmed *static* fixtures (phase 11): missed
    /// detection-frames a keyboard/mouse/… is still emitted (as `remembered`)
    /// while occluded by the operating hands, far beyond `timeout`. A static
    /// object doesn't walk away, so holding its last box is safe and keeps
    /// hand-on-keyboard `typing` attributable.
    static_hold: u32,
    diagnostics: TrackerDiagnostics,
    frame_binding: TrackerFrameBinding,
    compatibility_frame_id: u64,
    association_geometry: Vec<TrackedAssociationGeometry>,
    retirements: Vec<u32>,
}

/// Consecutive observations required before an object can leave tracker-local
/// diagnostics and become a downstream scene entity.
pub const OBJECT_CONFIRM_AGE: u32 = 2;

/// Confirmations (matched detection-frames) a static object needs before the
/// permanence hold trusts it — one clean flicker must not mint a multi-second
/// phantom keyboard.
pub const STATIC_HOLD_MIN_AGE: u32 = 2;
/// Default permanence hold in missed detection-frames. At `object_cadence=3` /
/// ~15 fps this is ~15 s — long enough to survive a typing burst, short enough
/// that a genuinely removed fixture clears within a sensible window.
pub const DEFAULT_STATIC_HOLD: u32 = 75;

impl Default for ObjectTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl ObjectTracker {
    /// Defaults: IoU 0.3, expire after 5 missed detection-frames, bbox EMA 0.5.
    pub fn new() -> Self {
        Self::with_params(0.3, 5, 0.5)
    }

    pub fn with_params(iou_thresh: f32, timeout: u32, ema_alpha: f32) -> Self {
        Self {
            tracks: Vec::new(),
            next_id: OBJECT_ID_BASE,
            free_ids: BinaryHeap::new(),
            iou_thresh,
            timeout,
            ema_alpha,
            static_hold: DEFAULT_STATIC_HOLD,
            diagnostics: TrackerDiagnostics::default(),
            frame_binding: TrackerFrameBinding::rgb_only(0),
            compatibility_frame_id: 0,
            association_geometry: Vec::new(),
            retirements: Vec::new(),
        }
    }

    /// Override the static-object permanence hold (phase 11). `0` disables it
    /// (pre-11 behaviour: static objects expire on `timeout` like everything else).
    pub fn with_static_hold(mut self, static_hold: u32) -> Self {
        self.static_hold = static_hold;
        self
    }

    /// The retention/emission horizon for a track: the long permanence hold for a
    /// confirmed static fixture, else the ordinary `timeout`.
    fn hold_limit(&self, t: &TrackedObject) -> u32 {
        if self.static_hold > 0 && is_static(t.class_name) && t.age >= STATIC_HOLD_MIN_AGE {
            self.static_hold
        } else {
            self.timeout
        }
    }

    fn alloc_id(&mut self) -> u32 {
        if let Some(Reverse(id)) = self.free_ids.pop() {
            id
        } else {
            let id = self.next_id;
            self.next_id += 1;
            id
        }
    }

    /// Update with this frame's detections; return the currently-visible tracks
    /// (those matched or created this frame). Call once per detection (cadence) frame.
    pub fn update(&mut self, dets: &[ObjectDetection]) -> Vec<TrackedObject> {
        self.compatibility_frame_id = self.compatibility_frame_id.saturating_add(1);
        self.frame_binding = TrackerFrameBinding::rgb_only(self.compatibility_frame_id);
        self.update_current(dets, None)
    }

    /// Update from one exact live/replay RGB or paired RGB-D packet.
    pub fn update_bound(
        &mut self,
        binding: &TrackerFrameBinding,
        dets: &[ObjectDetection],
    ) -> Vec<TrackedObject> {
        self.frame_binding = binding.clone();
        self.update_current(dets, None)
    }

    pub fn update_observed_bound(
        &mut self,
        binding: &TrackerFrameBinding,
        observations: &[AssociationObservation],
        dets: &[ObjectDetection],
    ) -> Result<Vec<TrackedObject>, &'static str> {
        if observations.len() != dets.len() {
            return Err("association observation count differs from object detections");
        }
        let mut keys = std::collections::HashSet::with_capacity(observations.len());
        for (observation, detection) in observations.iter().zip(dets) {
            observation.validate()?;
            if &observation.source != binding
                || observation.bbox != detection.bbox
                || observation.detector_score != detection.score
            {
                return Err("association observation differs from object detection");
            }
            if !keys.insert(observation.detection_key) {
                return Err("duplicate object detection key");
            }
        }
        self.frame_binding = binding.clone();
        Ok(self.update_current(dets, Some(observations)))
    }

    pub fn set_diagnostic_capacity(&mut self, capacity: usize) {
        self.diagnostics.set_capacity(capacity);
    }

    pub fn diagnostics(&self) -> TrackerDiagnosticsSnapshot {
        self.diagnostics.snapshot()
    }

    /// Current internal track table for image-free replay/reporting.
    pub fn all(&self) -> &[TrackedObject] {
        &self.tracks
    }

    pub fn association_geometry(&self) -> &[TrackedAssociationGeometry] {
        &self.association_geometry
    }

    pub fn occluded(&self) -> impl Iterator<Item = &TrackedObject> {
        self.tracks
            .iter()
            .filter(|track| track.lifecycle == LifecycleState::Occluded)
    }

    pub fn retained_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .tracks
            .iter()
            .filter(|track| {
                track.age >= OBJECT_CONFIRM_AGE && track.lifecycle != LifecycleState::Tentative
            })
            .map(|track| track.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    pub fn retirements(&self) -> &[u32] {
        &self.retirements
    }

    fn update_current(
        &mut self,
        dets: &[ObjectDetection],
        observations: Option<&[AssociationObservation]>,
    ) -> Vec<TrackedObject> {
        self.retirements.clear();
        self.diagnostics.record(TrackerDecision::new(
            &self.frame_binding,
            TrackerDomain::Object,
            TrackerDecisionKind::Admission,
            "accepted",
        ));
        if self.diagnostics.is_tracing() {
            for track in &self.tracks {
                let mut event = TrackerDecision::new(
                    &self.frame_binding,
                    TrackerDomain::Object,
                    TrackerDecisionKind::Prediction,
                    "not_supported_greedy_baseline",
                );
                event.track_id = Some(track.id);
                self.diagnostics.record(event);
            }
        }
        let mut track_used = vec![false; self.tracks.len()];
        let mut det_used = vec![false; dets.len()];
        let mut det_track_ids = vec![None; dets.len()];

        // Process detections high-score first for stable greedy matching.
        let mut order: Vec<usize> = (0..dets.len()).collect();
        order.sort_by(|&a, &b| dets[b].score.partial_cmp(&dets[a].score).unwrap());

        for &di in &order {
            let d = &dets[di];
            let mut best = self.iou_thresh;
            let mut best_t = None;
            for (ti, t) in self.tracks.iter().enumerate() {
                if track_used[ti] {
                    continue;
                }
                if t.class_name != d.class_name {
                    if self.diagnostics.is_tracing() {
                        let mut event = TrackerDecision::new(
                            &self.frame_binding,
                            TrackerDomain::Object,
                            TrackerDecisionKind::Candidate,
                            "vetoed",
                        );
                        event.detection_index = Some(di);
                        event.track_id = Some(t.id);
                        event.reason = Some("class_mismatch".into());
                        event.cue = Some("class_iou".into());
                        self.diagnostics.record(event);
                    }
                    continue;
                }
                let iou = t.bbox.iou(&d.bbox);
                if self.diagnostics.is_tracing() {
                    let mut event = TrackerDecision::new(
                        &self.frame_binding,
                        TrackerDomain::Object,
                        TrackerDecisionKind::Candidate,
                        if iou >= self.iou_thresh {
                            "eligible"
                        } else {
                            "vetoed"
                        },
                    );
                    event.detection_index = Some(di);
                    event.track_id = Some(t.id);
                    event.iou = Some(iou);
                    event.score = (iou >= self.iou_thresh).then_some(iou);
                    event.cue = Some("class_iou".into());
                    event.reason = (iou < self.iou_thresh).then(|| "iou_gate".into());
                    self.diagnostics.record(event);
                }
                if iou >= best {
                    best = iou;
                    best_t = Some(ti);
                }
            }
            if let Some(ti) = best_t {
                track_used[ti] = true;
                det_used[di] = true;
                let a = self.ema_alpha;
                let t = &mut self.tracks[ti];
                t.bbox = ema_bbox(&t.bbox, &d.bbox, a);
                t.score = d.score;
                t.age += 1;
                t.missed = 0;
                t.lifecycle = if t.age >= OBJECT_CONFIRM_AGE {
                    LifecycleState::Confirmed
                } else {
                    LifecycleState::Tentative
                };
                det_track_ids[di] = Some(t.id);
                let mut event = TrackerDecision::new(
                    &self.frame_binding,
                    TrackerDomain::Object,
                    TrackerDecisionKind::Assignment,
                    "matched_greedy_baseline",
                );
                event.detection_index = Some(di);
                event.track_id = Some(t.id);
                event.iou = Some(best);
                self.diagnostics.record(event);
            }
        }

        // Unmatched detections → new tracks.
        for (di, d) in dets.iter().enumerate() {
            if det_used[di] {
                continue;
            }
            let id = self.alloc_id();
            self.tracks.push(TrackedObject {
                id,
                class_name: d.class_name,
                bbox: d.bbox,
                score: d.score,
                age: 1,
                missed: 0,
                lifecycle: LifecycleState::Tentative,
                pos_3d: None,
                pos_conf: 0.0,
                mask: None,
            });
            det_track_ids[di] = Some(id);
            let mut event = TrackerDecision::new(
                &self.frame_binding,
                TrackerDomain::Object,
                TrackerDecisionKind::Birth,
                "born",
            );
            event.detection_index = Some(di);
            event.track_id = Some(id);
            self.diagnostics.record(event);
        }

        // Unmatched tracks age out; expired ids return to the free-list.
        for (ti, used) in track_used.iter().enumerate() {
            if !used {
                self.tracks[ti].missed += 1;
                let mut event = TrackerDecision::new(
                    &self.frame_binding,
                    TrackerDomain::Object,
                    TrackerDecisionKind::Miss,
                    "remembered_or_aging",
                );
                event.track_id = Some(self.tracks[ti].id);
                self.diagnostics.record(event);
            }
        }
        let lifecycle_limits: Vec<(u32, u32)> = self
            .tracks
            .iter()
            .map(|track| (self.hold_limit(track), self.timeout))
            .collect();
        for (track, (hold_limit, timeout)) in self.tracks.iter_mut().zip(lifecycle_limits) {
            let before = track.lifecycle;
            track.lifecycle = if track.missed == 0 {
                if track.age >= OBJECT_CONFIRM_AGE {
                    LifecycleState::Confirmed
                } else {
                    LifecycleState::Tentative
                }
            } else if track.age < OBJECT_CONFIRM_AGE {
                LifecycleState::Lost
            } else if hold_limit > timeout && track.missed <= hold_limit {
                LifecycleState::Occluded
            } else if track.missed >= timeout {
                LifecycleState::Lost
            } else {
                LifecycleState::Occluded
            };
            if before != track.lifecycle {
                let mut event = TrackerDecision::new(
                    &self.frame_binding,
                    TrackerDomain::Object,
                    TrackerDecisionKind::Lifecycle,
                    "transition",
                );
                event.track_id = Some(track.id);
                event.lifecycle_before = Some(before.as_str().into());
                event.lifecycle_after = Some(track.lifecycle.as_str().into());
                self.diagnostics.record(event);
            }
        }
        // Expiry: a confirmed static fixture rides the long permanence hold
        // (phase 11); everything else the ordinary timeout.
        let holds: Vec<u32> = self.tracks.iter().map(|t| self.hold_limit(t)).collect();
        let mut freed = Vec::new();
        let mut i = 0;
        self.tracks.retain(|t| {
            let keep = t.missed <= holds[i];
            if !keep {
                freed.push(t.id);
            }
            i += 1;
            keep
        });
        for id in freed {
            self.retirements.push(id);
            self.free_ids.push(Reverse(id));
            let mut event = TrackerDecision::new(
                &self.frame_binding,
                TrackerDomain::Object,
                TrackerDecisionKind::Expire,
                "expired",
            );
            event.track_id = Some(id);
            self.diagnostics.record(event);
        }

        // Emit currently-seen tracks plus *remembered* static fixtures — a
        // keyboard held under the operating hands still anchors `typing`
        // (phase 11 object-permanence). `TrackedObject.missed > 0` marks the
        // remembered ones so consumers can tell "held" from "seen".
        let emitted: Vec<TrackedObject> = self
            .tracks
            .iter()
            .filter(|track| track.missed == 0 && track.lifecycle == LifecycleState::Confirmed)
            .cloned()
            .collect();
        for track in &self.tracks {
            let mut event = TrackerDecision::new(
                &self.frame_binding,
                TrackerDomain::Object,
                TrackerDecisionKind::Publication,
                if track.missed == 0 && track.lifecycle == LifecycleState::Confirmed {
                    "observed"
                } else if track.lifecycle == LifecycleState::Tentative {
                    "tentative_suppressed"
                } else if track.lifecycle == LifecycleState::Occluded {
                    "occluded_handoff"
                } else {
                    "not_current"
                },
            );
            event.track_id = Some(track.id);
            event.lifecycle_after = Some(track.lifecycle.as_str().into());
            self.diagnostics.record(event);
        }
        self.association_geometry = observations.map_or_else(Vec::new, |observations| {
            observations
                .iter()
                .zip(det_track_ids)
                .filter_map(|(observation, track_id)| {
                    track_id.map(|track_id| TrackedAssociationGeometry {
                        track_id,
                        detection_key: observation.detection_key,
                        source_frame_id: observation.source.source_frame_id,
                        geometry: observation.geometry.clone(),
                    })
                })
                .collect()
        });
        emitted
    }
}

/// EMA-smooth a bbox toward a new observation (reduces per-frame jitter).
fn ema_bbox(prev: &BBox, new: &BBox, alpha: f32) -> BBox {
    let f = |p: f32, n: f32| p * (1.0 - alpha) + n * alpha;
    BBox {
        x1: f(prev.x1, new.x1),
        y1: f(prev.y1, new.y1),
        x2: f(prev.x2, new.x2),
        y2: f(prev.y2, new.y2),
    }
}

/// Attach YOLO-seg instance masks onto the tracked objects (phase 10.4 / 10.5).
///
/// YOLO-seg runs as an **additional** cadenced expert over YOLO26n (which stays
/// the tracking authority — spike §3.2: seg is not a drop-in, its small-object
/// recall is weaker). Each tick this maps the seg detections' masks onto the
/// objects the tracker already owns, so masks ride the stable track ids without
/// changing who tracks what. Matching is greedy and class-aware by bbox-IoU, with
/// the seg mask's IoU against the object's **previous** mask as a lexicographic
/// tiebreak — the instance-separation signal (phase 10.5) that disambiguates two
/// overlapping same-class objects whose bboxes match a seg equally well.
///
/// Every object's mask is reset first, so a stale mask never outlives the seg
/// detection that produced it — a mask is only ever as fresh as this tick's seg.
pub fn attach_masks(objects: &mut [TrackedObject], segs: &[SegDetection], iou_thresh: f32) {
    // Candidates `(obj, seg, bbox_iou, mask_iou)`. mask_iou uses the object's
    // *previous* mask (read before the reset below) as the tiebreak.
    let mut cands: Vec<(usize, usize, f32, f32)> = Vec::new();
    for (oi, o) in objects.iter().enumerate() {
        for (si, s) in segs.iter().enumerate() {
            if o.class_name != s.det.class_name {
                continue;
            }
            let bi = o.bbox.iou(&s.det.bbox);
            if bi < iou_thresh {
                continue;
            }
            let mi = o.mask.as_ref().map_or(0.0, |m| m.iou(&s.mask));
            cands.push((oi, si, bi, mi));
        }
    }
    // A mask is only as fresh as this tick's seg — never carry a stale silhouette.
    for o in objects.iter_mut() {
        o.mask = None;
    }
    // Greedy: highest bbox-IoU first, mask-IoU breaks ties; one seg per object.
    cands.sort_by(|a, b| {
        b.2.partial_cmp(&a.2)
            .unwrap()
            .then(b.3.partial_cmp(&a.3).unwrap())
    });
    let mut obj_used = vec![false; objects.len()];
    let mut seg_used = vec![false; segs.len()];
    for (oi, si, _, _) in cands {
        if obj_used[oi] || seg_used[si] {
            continue;
        }
        obj_used[oi] = true;
        seg_used[si] = true;
        objects[oi].mask = Some(segs[si].mask.clone());
    }
}

/// Letterbox transform: how the original WxH maps onto a square `target`xtarget
/// canvas (aspect-preserving resize + centred grey padding, the ultralytics
/// convention). `scale` is original→canvas; `pad_x`/`pad_y` are the left/top
/// borders in canvas pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
}

/// Compute the letterbox mapping for an original image onto `target`x`target`.
pub fn compute_letterbox(orig_w: u32, orig_h: u32, target: u32) -> Letterbox {
    let t = target as f32;
    let scale = (t / orig_w as f32).min(t / orig_h as f32);
    let new_w = orig_w as f32 * scale;
    let new_h = orig_h as f32 * scale;
    Letterbox {
        scale,
        pad_x: (t - new_w) * 0.5,
        pad_y: (t - new_h) * 0.5,
    }
}

/// A raw decoded box in **canvas** (letterboxed input) pixel space, corner form.
#[derive(Debug, Clone, Copy, PartialEq)]
struct RawBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    score: f32,
    class_id: usize,
}

/// IoU of two corner-form boxes (any consistent coordinate space).
fn iou_xyxy(a: &RawBox, b: &RawBox) -> f32 {
    let ix1 = a.x1.max(b.x1);
    let iy1 = a.y1.max(b.y1);
    let ix2 = a.x2.min(b.x2);
    let iy2 = a.y2.min(b.y2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    if inter == 0.0 {
        return 0.0;
    }
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    inter / (area_a + area_b - inter).max(1e-10)
}

/// Decode a raw ultralytics output tensor into candidate boxes (canvas pixels).
///
/// `raw` is channel-major flat `[num_attrs * num_anchors]`: attribute `c`, anchor
/// `a` lives at `raw[c * num_anchors + a]`. Attributes 0..4 are `cx,cy,w,h` (in
/// canvas pixels); attributes 4.. are per-class scores (post-sigmoid in [0,1]).
/// For each anchor the best class is taken; below `score_thresh` it is dropped.
fn decode_yolo(raw: &[f32], num_classes: usize, score_thresh: f32) -> Vec<RawBox> {
    let num_attrs = num_classes + 4;
    debug_assert_eq!(
        raw.len() % num_attrs,
        0,
        "tensor length not divisible by attrs"
    );
    let num_anchors = raw.len() / num_attrs;
    let mut out = Vec::new();

    for a in 0..num_anchors {
        // Best class for this anchor.
        let mut best_score = score_thresh;
        let mut best_class = None;
        for k in 0..num_classes {
            let s = raw[(4 + k) * num_anchors + a];
            if s > best_score {
                best_score = s;
                best_class = Some(k);
            }
        }
        let Some(class_id) = best_class else { continue };

        let cx = raw[a];
        let cy = raw[num_anchors + a];
        let w = raw[2 * num_anchors + a];
        let h = raw[3 * num_anchors + a];
        out.push(RawBox {
            x1: cx - w * 0.5,
            y1: cy - h * 0.5,
            x2: cx + w * 0.5,
            y2: cy + h * 0.5,
            score: best_score,
            class_id,
        });
    }
    out
}

/// Decode the **end-to-end (NMS-free)** ultralytics output `[1, N, 6]`: each row
/// is `[x1, y1, x2, y2, score, class_id]` in canvas pixels, already top-k filtered
/// and deduplicated — so NO NMS is applied. YOLOv10 and **YOLO26** export this form
/// (the slots below `score_thresh` are padding and are dropped here).
fn decode_end2end(raw: &[f32], score_thresh: f32, num_classes: usize) -> Vec<RawBox> {
    let mut out = Vec::new();
    for row in raw.chunks_exact(6) {
        let score = row[4];
        if score < score_thresh {
            continue;
        }
        let class_id = row[5].round() as usize;
        if class_id >= num_classes {
            continue;
        }
        out.push(RawBox {
            x1: row[0],
            y1: row[1],
            x2: row[2],
            y2: row[3],
            score,
            class_id,
        });
    }
    out
}

/// Number of prototype masks / mask coefficients a YOLO-seg head emits.
pub const N_PROTO: usize = 32;

/// A raw seg detection: a decoded box plus its 32 mask coefficients (still in
/// canvas space, pre-NMS). Kept next to [`RawBox`] so the seg NMS reuses the
/// same suppression, carrying the coeffs of the survivors.
#[derive(Debug, Clone)]
struct RawSeg {
    boxb: RawBox,
    coeffs: [f32; N_PROTO],
}

/// Decode the YOLO-seg output `[1, 4+C+32, A]` (channel-major): boxes + class
/// scores exactly like [`decode_yolo`], plus the trailing 32 mask coefficients
/// per anchor. Same anchor/threshold logic; the coeffs ride along for the
/// surviving boxes so the prototype combination can be assembled after NMS.
fn decode_yolo_seg(raw: &[f32], num_classes: usize, score_thresh: f32) -> Vec<RawSeg> {
    let num_attrs = num_classes + 4 + N_PROTO;
    debug_assert_eq!(
        raw.len() % num_attrs,
        0,
        "seg tensor length not divisible by attrs"
    );
    let num_anchors = raw.len() / num_attrs;
    let coeff_base = 4 + num_classes;
    let mut out = Vec::new();
    for a in 0..num_anchors {
        let mut best_score = score_thresh;
        let mut best_class = None;
        for k in 0..num_classes {
            let s = raw[(4 + k) * num_anchors + a];
            if s > best_score {
                best_score = s;
                best_class = Some(k);
            }
        }
        let Some(class_id) = best_class else { continue };
        let cx = raw[a];
        let cy = raw[num_anchors + a];
        let w = raw[2 * num_anchors + a];
        let h = raw[3 * num_anchors + a];
        let mut coeffs = [0.0f32; N_PROTO];
        for (k, c) in coeffs.iter_mut().enumerate() {
            *c = raw[(coeff_base + k) * num_anchors + a];
        }
        out.push(RawSeg {
            boxb: RawBox {
                x1: cx - w * 0.5,
                y1: cy - h * 0.5,
                x2: cx + w * 0.5,
                y2: cy + h * 0.5,
                score: best_score,
                class_id,
            },
            coeffs,
        });
    }
    out
}

/// Class-aware greedy NMS over seg detections (mirrors [`nms`], keeps coeffs).
fn nms_seg(mut segs: Vec<RawSeg>, iou_thresh: f32) -> Vec<RawSeg> {
    segs.sort_by(|a, b| b.boxb.score.partial_cmp(&a.boxb.score).unwrap());
    let mut kept: Vec<RawSeg> = Vec::new();
    for cand in segs {
        let suppressed = kept.iter().any(|k| {
            k.boxb.class_id == cand.boxb.class_id && iou_xyxy(&k.boxb, &cand.boxb) > iou_thresh
        });
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// A seg detection surfaced to callers: the object box + its instance mask.
#[derive(Debug, Clone, PartialEq)]
pub struct SegDetection {
    pub det: ObjectDetection,
    pub mask: InstanceMask,
}

/// Class-aware greedy NMS: sort by score desc, suppress same-class overlaps.
fn nms(mut boxes: Vec<RawBox>, iou_thresh: f32) -> Vec<RawBox> {
    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut kept: Vec<RawBox> = Vec::new();
    for cand in boxes {
        let suppressed = kept
            .iter()
            .any(|k| k.class_id == cand.class_id && iou_xyxy(k, &cand) > iou_thresh);
        if !suppressed {
            kept.push(cand);
        }
    }
    kept
}

/// Map a canvas-pixel box back to a normalized [0,1] bbox in the original frame.
fn canvas_to_norm(b: &RawBox, lb: &Letterbox, orig_w: u32, orig_h: u32) -> BBox {
    let undo = |v: f32, pad: f32, dim: u32| (((v - pad) / lb.scale) / dim as f32).clamp(0.0, 1.0);
    BBox {
        x1: undo(b.x1, lb.pad_x, orig_w),
        y1: undo(b.y1, lb.pad_y, orig_h),
        x2: undo(b.x2, lb.pad_x, orig_w),
        y2: undo(b.y2, lb.pad_y, orig_h),
    }
}

/// YOLO ONNX object detector (ultralytics anchor-free family).
pub struct YoloDetector {
    session: Session,
    input_size: u32,
    score_threshold: f32,
    nms_threshold: f32,
}

impl YoloDetector {
    const INPUT_NAME: &'static str = "images";

    /// Load from a file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load YOLO from {}", path.display()))?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    /// Load from ONNX model bytes (embedded mode).
    #[cfg(feature = "embed-model")]
    pub fn from_bytes(model_bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(model_bytes)
            .context("Failed to load YOLO from embedded bytes")?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    pub fn score_threshold(&self) -> f32 {
        self.score_threshold
    }

    pub fn set_score_threshold(&mut self, t: f32) {
        self.score_threshold = t;
    }

    /// Detect objects. Returns normalized [0,1] bboxes in the original frame.
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<ObjectDetection>> {
        let (orig_w, orig_h) = image.dimensions();
        let lb = compute_letterbox(orig_w, orig_h, self.input_size);
        let input = self.preprocess(image, &lb)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;

        let (shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
        // Branch on the output layout (model-agnostic): end-to-end NMS-free
        // `[1, N, 6]` (YOLOv10 / YOLO26) vs raw anchor grid `[1, 4+C, A]` (YOLOv8).
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let is_end2end = dims.len() == 3 && *dims.last().unwrap_or(&0) == 6;
        let kept = if is_end2end {
            decode_end2end(raw, self.score_threshold, COCO_CLASSES.len())
        } else {
            let proposals = decode_yolo(raw, COCO_CLASSES.len(), self.score_threshold);
            nms(proposals, self.nms_threshold)
        };

        Ok(kept
            .into_iter()
            .map(|b| ObjectDetection {
                class_id: b.class_id,
                class_name: COCO_CLASSES[b.class_id],
                bbox: canvas_to_norm(&b, &lb, orig_w, orig_h),
                score: b.score,
            })
            .collect())
    }

    /// Letterbox + RGB + NCHW + [0,1] normalization (ultralytics preprocessing).
    fn preprocess(&self, image: &DynamicImage, lb: &Letterbox) -> anyhow::Result<Tensor<f32>> {
        let sz = self.input_size as usize;
        let (orig_w, orig_h) = image.dimensions();
        let new_w = (orig_w as f32 * lb.scale).round() as u32;
        let new_h = (orig_h as f32 * lb.scale).round() as u32;
        let resized = image.resize_exact(
            new_w.max(1),
            new_h.max(1),
            image::imageops::FilterType::Triangle,
        );
        let resized = resized.to_rgb8();

        // Grey (114) padded canvas, NCHW, channels R,G,B, normalized to [0,1].
        let pad_val = 114.0f32 / 255.0;
        let mut flat = vec![pad_val; 3 * sz * sz];
        let off_x = lb.pad_x.round() as usize;
        let off_y = lb.pad_y.round() as usize;
        for y in 0..(new_h as usize).min(sz.saturating_sub(off_y)) {
            for x in 0..(new_w as usize).min(sz.saturating_sub(off_x)) {
                let px = resized.get_pixel(x as u32, y as u32);
                let cy = off_y + y;
                let cx = off_x + x;
                let idx = cy * sz + cx;
                flat[idx] = px[0] as f32 / 255.0; // R
                flat[sz * sz + idx] = px[1] as f32 / 255.0; // G
                flat[2 * sz * sz + idx] = px[2] as f32 / 255.0; // B
            }
        }
        let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
        let array =
            ndarray::Array::from_shape_vec(shape, flat).context("Failed to create input array")?;
        Tensor::from_array(array).context("Failed to create tensor")
    }
}

/// YOLO-seg ONNX detector (phase 10.4) — the sanctioned +1 ONNX expert.
///
/// Emits two tensors: `output0 [1, 4+80+32, A]` (boxes + class scores + mask
/// coefficients) and `output1 [1, 32, ph, pw]` (the shared prototype bank). Per
/// spike §3.2 this is an **additional** cadenced expert, not a replacement for
/// the YOLO26n detector — its masks are matched onto tracked objects, its raw
/// boxes are not the tracking authority. Reuses `YoloDetector`'s letterbox +
/// preprocess so both models see an identical canvas.
pub struct YoloSegDetector {
    session: Session,
    input_size: u32,
    score_threshold: f32,
    nms_threshold: f32,
}

impl YoloSegDetector {
    const INPUT_NAME: &'static str = "images";

    /// Load from a file path.
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load YOLO-seg from {}", path.display()))?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    /// Load from ONNX model bytes (embedded mode).
    #[cfg(feature = "embed-model")]
    pub fn from_bytes(model_bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(model_bytes)
            .context("Failed to load YOLO-seg from embedded bytes")?;
        Ok(Self {
            session,
            input_size: 640,
            score_threshold: 0.35,
            nms_threshold: 0.45,
        })
    }

    pub fn set_score_threshold(&mut self, t: f32) {
        self.score_threshold = t;
    }

    /// Detect objects **with instance masks**. Boxes are normalized [0,1] in the
    /// original frame; each mask is a [`InstanceMask`] over the same frame.
    pub fn detect(&mut self, image: &DynamicImage) -> anyhow::Result<Vec<SegDetection>> {
        let (orig_w, orig_h) = image.dimensions();
        let lb = compute_letterbox(orig_w, orig_h, self.input_size);
        // Reuse YoloDetector's identical preprocess by building a throwaway view is
        // overkill; inline the same letterbox+NCHW here via a shared helper.
        let input = preprocess_letterbox(image, &lb, self.input_size)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;

        // output0: boxes+scores+coeffs; output1: prototype bank.
        let (s0, raw0) = outputs[0].try_extract_tensor::<f32>()?;
        let (s1, proto) = outputs[1].try_extract_tensor::<f32>()?;
        let d0: Vec<usize> = s0.iter().map(|&d| d as usize).collect();
        let d1: Vec<usize> = s1.iter().map(|&d| d as usize).collect();
        // proto dims [1, n_proto, ph, pw].
        let (n_proto, ph, pw) = match d1.as_slice() {
            [_, n, h, w] => (*n, *h, *w),
            _ => anyhow::bail!("unexpected proto shape {d1:?}"),
        };
        anyhow::ensure!(
            n_proto == N_PROTO,
            "expected {N_PROTO} prototypes, got {n_proto}"
        );
        // output0 may be [1, attrs, A] or [1, A, attrs]; decode_yolo_seg wants
        // channel-major [attrs, A]. Detect and transpose if needed.
        let raw0 = to_channel_major(raw0, &d0, COCO_CLASSES.len() + 4 + N_PROTO);

        let segs = nms_seg(
            decode_yolo_seg(&raw0, COCO_CLASSES.len(), self.score_threshold),
            self.nms_threshold,
        );

        Ok(segs
            .into_iter()
            .map(|s| {
                let bbox = canvas_to_norm(&s.boxb, &lb, orig_w, orig_h);
                let mask = assemble_mask(
                    proto,
                    n_proto,
                    ph,
                    pw,
                    &s.coeffs,
                    &bbox,
                    lb,
                    orig_w,
                    orig_h,
                    self.input_size,
                );
                SegDetection {
                    det: ObjectDetection {
                        class_id: s.boxb.class_id,
                        class_name: COCO_CLASSES[s.boxb.class_id],
                        bbox,
                        score: s.boxb.score,
                    },
                    mask,
                }
            })
            .collect())
    }
}

/// Ensure a `[1, A, attrs]`-vs-`[1, attrs, A]` output is returned channel-major
/// `[attrs, A]` (what `decode_yolo_seg` reads). ONNX exports differ; we pick the
/// axis whose length equals `attrs`.
fn to_channel_major(raw: &[f32], dims: &[usize], attrs: usize) -> Vec<f32> {
    // dims == [1, d1, d2]. If d1==attrs it is already channel-major.
    if dims.len() != 3 {
        return raw.to_vec();
    }
    let (d1, d2) = (dims[1], dims[2]);
    if d1 == attrs {
        raw.to_vec()
    } else {
        // [1, A, attrs] → transpose to [attrs, A].
        let (a, at) = (d1, d2);
        let mut out = vec![0.0f32; a * at];
        for i in 0..a {
            for c in 0..at {
                out[c * a + i] = raw[i * at + c];
            }
        }
        out
    }
}

/// Shared letterbox + RGB + NCHW + [0,1] preprocess (used by both YOLO detectors).
fn preprocess_letterbox(
    image: &DynamicImage,
    lb: &Letterbox,
    input_size: u32,
) -> anyhow::Result<Tensor<f32>> {
    let sz = input_size as usize;
    let (orig_w, orig_h) = image.dimensions();
    let new_w = (orig_w as f32 * lb.scale).round() as u32;
    let new_h = (orig_h as f32 * lb.scale).round() as u32;
    let resized = image
        .resize_exact(
            new_w.max(1),
            new_h.max(1),
            image::imageops::FilterType::Triangle,
        )
        .to_rgb8();
    let pad_val = 114.0f32 / 255.0;
    let mut flat = vec![pad_val; 3 * sz * sz];
    let off_x = lb.pad_x.round() as usize;
    let off_y = lb.pad_y.round() as usize;
    for y in 0..(new_h as usize).min(sz.saturating_sub(off_y)) {
        for x in 0..(new_w as usize).min(sz.saturating_sub(off_x)) {
            let px = resized.get_pixel(x as u32, y as u32);
            let idx = (off_y + y) * sz + (off_x + x);
            flat[idx] = px[0] as f32 / 255.0;
            flat[sz * sz + idx] = px[1] as f32 / 255.0;
            flat[2 * sz * sz + idx] = px[2] as f32 / 255.0;
        }
    }
    let shape = ndarray::IxDyn(&[1, 3, sz, sz]);
    let array = ndarray::Array::from_shape_vec(shape, flat).context("seg input array")?;
    Tensor::from_array(array).context("seg tensor")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── seg decode (pure, synthetic tensor) ────────────────────────────────

    #[test]
    fn decode_yolo_seg_reads_box_class_and_coeffs() {
        // 2 anchors, num_classes=2, 32 coeffs → attrs = 4+2+32 = 38, channel-major.
        let nc = 2;
        let attrs = 4 + nc + N_PROTO;
        let anchors = 2;
        let mut raw = vec![0.0f32; attrs * anchors];
        let put = |raw: &mut [f32], c: usize, a: usize, v: f32| raw[c * anchors + a] = v;
        // anchor 0: box (cx,cy,w,h)=(100,120,40,60), class 1 score 0.9, coeff[5]=2.0
        put(&mut raw, 0, 0, 100.0);
        put(&mut raw, 1, 0, 120.0);
        put(&mut raw, 2, 0, 40.0);
        put(&mut raw, 3, 0, 60.0);
        put(&mut raw, 4 + 1, 0, 0.9);
        put(&mut raw, 4 + nc + 5, 0, 2.0);
        // anchor 1: below threshold (max class score 0.1) → dropped.
        put(&mut raw, 4, 1, 0.1);
        let segs = decode_yolo_seg(&raw, nc, 0.35);
        assert_eq!(segs.len(), 1, "only anchor 0 survives");
        let s = &segs[0];
        assert_eq!(s.boxb.class_id, 1);
        assert!((s.boxb.x1 - 80.0).abs() < 1e-4 && (s.boxb.y2 - 150.0).abs() < 1e-4);
        assert!((s.coeffs[5] - 2.0).abs() < 1e-6, "coeff 5 carried through");
    }

    #[test]
    fn to_channel_major_transposes_anchor_last() {
        // [1, A=2, attrs=3] row-major → channel-major [attrs, A].
        let raw = vec![
            1.0, 2.0, 3.0, // anchor 0: attrs
            4.0, 5.0, 6.0, // anchor 1: attrs
        ];
        let cm = to_channel_major(&raw, &[1, 2, 3], 3);
        // channel-major: c0=[1,4], c1=[2,5], c2=[3,6]
        assert_eq!(cm, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn to_channel_major_passthrough_when_already_channel_major() {
        // [1, attrs=3, A=2] already channel-major → unchanged.
        let raw = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
        assert_eq!(to_channel_major(&raw, &[1, 3, 2], 3), raw);
    }

    /// Model-gated: run the real YOLO-seg on the live rig frame and assert a
    /// person instance mask with a sensible silhouette coverage (phase 10.4).
    #[test]
    fn seg_model_finds_person_mask_on_live_frame() {
        let model = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/yolo_seg.onnx"
        ));
        let frame =
            std::path::PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/live0.jpg"));
        if !model.exists() || !frame.exists() {
            eprintln!("Skipping seg live test: model or frame absent");
            return;
        }
        let mut seg = YoloSegDetector::from_path(&model).unwrap();
        seg.set_score_threshold(0.25);
        let img = image::open(&frame).unwrap();
        let dets = seg.detect(&img).unwrap();
        eprintln!("seg live: {} instances", dets.len());
        for d in &dets {
            eprintln!(
                "  {} score={:.2} mask_cov={:.3}",
                d.det.class_name,
                d.det.score,
                d.mask.frame_coverage()
            );
        }
        let person = dets
            .iter()
            .find(|d| d.det.class_name == "person")
            .expect("person detected");
        assert!(!person.mask.is_empty(), "person mask non-empty");
        // The mask must be tighter than its bbox (a real silhouette, not the box).
        let b = &person.det.bbox;
        let bbox_area = (b.x2 - b.x1) * (b.y2 - b.y1);
        assert!(
            person.mask.frame_coverage() < bbox_area,
            "mask tighter than bbox"
        );
    }

    // ── letterbox (pure) ───────────────────────────────────────────────────

    #[test]
    fn letterbox_square_is_identity_scale() {
        let lb = compute_letterbox(640, 640, 640);
        assert!((lb.scale - 1.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6 && lb.pad_y.abs() < 1e-6);
    }

    #[test]
    fn letterbox_landscape_pads_vertically() {
        // 640x480 → scale 1.0, height 480 → pad top/bottom (640-480)/2 = 80.
        let lb = compute_letterbox(640, 480, 640);
        assert!((lb.scale - 1.0).abs() < 1e-6);
        assert!(lb.pad_x.abs() < 1e-6);
        assert!((lb.pad_y - 80.0).abs() < 1e-6);
    }

    #[test]
    fn letterbox_portrait_pads_horizontally() {
        // 480x640 → scale 1.0 → pad left/right 80.
        let lb = compute_letterbox(480, 640, 640);
        assert!((lb.pad_x - 80.0).abs() < 1e-6);
        assert!(lb.pad_y.abs() < 1e-6);
    }

    // ── canvas_to_norm inverse (pure) ──────────────────────────────────────

    #[test]
    fn canvas_to_norm_undoes_letterbox_padding() {
        // 640x480 letterboxed into 640: a box that in canvas spans y 80..560
        // (the full image band) should map back to the full original height.
        let lb = compute_letterbox(640, 480, 640);
        let b = RawBox {
            x1: 0.0,
            y1: 80.0,
            x2: 640.0,
            y2: 560.0,
            score: 0.9,
            class_id: 0,
        };
        let n = canvas_to_norm(&b, &lb, 640, 480);
        assert!((n.x1 - 0.0).abs() < 1e-4 && (n.x2 - 1.0).abs() < 1e-4);
        assert!(
            (n.y1 - 0.0).abs() < 1e-4 && (n.y2 - 1.0).abs() < 1e-4,
            "y={:?}",
            (n.y1, n.y2)
        );
    }

    // ── decode (pure, synthetic tensor — no model file) ────────────────────

    /// Build a channel-major raw tensor for `num_anchors` anchors, `num_classes`
    /// classes. Anchor 0 gets a strong detection of `hot_class`; the rest are noise.
    fn synth_raw(num_anchors: usize, num_classes: usize, hot_class: usize) -> Vec<f32> {
        let num_attrs = num_classes + 4;
        let mut raw = vec![0.0f32; num_attrs * num_anchors];
        let put = |raw: &mut Vec<f32>, attr: usize, anchor: usize, v: f32| {
            raw[attr * num_anchors + anchor] = v;
        };
        // Anchor 0: box centred at (100,100), 40x40, hot_class score 0.9.
        put(&mut raw, 0, 0, 100.0);
        put(&mut raw, 1, 0, 100.0);
        put(&mut raw, 2, 0, 40.0);
        put(&mut raw, 3, 0, 40.0);
        put(&mut raw, 4 + hot_class, 0, 0.9);
        raw
    }

    #[test]
    fn decode_picks_best_class_above_threshold() {
        let raw = synth_raw(10, 80, 42);
        let boxes = decode_yolo(&raw, 80, 0.35);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].class_id, 42);
        assert!((boxes[0].score - 0.9).abs() < 1e-6);
        // corner form of centre (100,100) size 40 → 80..120.
        assert!((boxes[0].x1 - 80.0).abs() < 1e-4);
        assert!((boxes[0].x2 - 120.0).abs() < 1e-4);
    }

    #[test]
    fn decode_drops_everything_below_threshold() {
        let mut raw = synth_raw(10, 80, 42);
        // Lower the only hot score under threshold.
        raw[(4 + 42) * 10] = 0.10;
        let boxes = decode_yolo(&raw, 80, 0.35);
        assert!(boxes.is_empty());
    }

    #[test]
    fn decode_maps_class_to_coco_name() {
        let raw = synth_raw(5, 80, 56); // 56 = "chair"
        let boxes = decode_yolo(&raw, 80, 0.35);
        assert_eq!(COCO_CLASSES[boxes[0].class_id], "chair");
    }

    // ── end-to-end decode (pure, synthetic — YOLO26/v10 format) ────────────

    #[test]
    fn decode_end2end_filters_by_score_and_maps_class() {
        // Two rows: one strong "cup" (41), one weak below threshold.
        let raw = vec![
            80.0, 80.0, 120.0, 120.0, 0.92, 41.0, // keep
            10.0, 10.0, 20.0, 20.0, 0.05, 7.0, // drop (padding)
        ];
        let boxes = decode_end2end(&raw, 0.35, 80);
        assert_eq!(boxes.len(), 1);
        assert_eq!(COCO_CLASSES[boxes[0].class_id], "cup");
        assert!((boxes[0].x1 - 80.0).abs() < 1e-4 && (boxes[0].x2 - 120.0).abs() < 1e-4);
        assert!((boxes[0].score - 0.92).abs() < 1e-6);
    }

    #[test]
    fn decode_end2end_drops_all_padding() {
        let raw = vec![0.0; 300 * 6]; // black-image style: all-zero, score 0
        assert!(decode_end2end(&raw, 0.35, 80).is_empty());
    }

    #[test]
    fn decode_end2end_rejects_out_of_range_class() {
        let raw = vec![0.0, 0.0, 10.0, 10.0, 0.9, 999.0];
        assert!(decode_end2end(&raw, 0.35, 80).is_empty());
    }

    // ── NMS (pure) ─────────────────────────────────────────────────────────

    fn rb(x1: f32, y1: f32, x2: f32, y2: f32, score: f32, class_id: usize) -> RawBox {
        RawBox {
            x1,
            y1,
            x2,
            y2,
            score,
            class_id,
        }
    }

    #[test]
    fn nms_suppresses_same_class_overlap() {
        let boxes = vec![
            rb(0.0, 0.0, 40.0, 40.0, 0.9, 1),
            rb(2.0, 2.0, 42.0, 42.0, 0.8, 1), // ~heavy overlap, same class → suppressed
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].score - 0.9).abs() < 1e-6);
    }

    #[test]
    fn nms_keeps_overlapping_different_classes() {
        let boxes = vec![
            rb(0.0, 0.0, 40.0, 40.0, 0.9, 1),
            rb(2.0, 2.0, 42.0, 42.0, 0.8, 2), // same box, different class → both kept
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn nms_keeps_disjoint_same_class() {
        let boxes = vec![
            rb(0.0, 0.0, 40.0, 40.0, 0.9, 1),
            rb(100.0, 100.0, 140.0, 140.0, 0.8, 1),
        ];
        let kept = nms(boxes, 0.45);
        assert_eq!(kept.len(), 2);
    }

    // ── allow-list (filter C) ──────────────────────────────────────────────

    #[test]
    fn is_relevant_keeps_interaction_objects_drops_noise() {
        assert!(is_relevant("cup") && is_relevant("laptop") && is_relevant("chair"));
        assert!(
            !is_relevant("person"),
            "person owned by body-primary tracker"
        );
        assert!(!is_relevant("car") && !is_relevant("dog") && !is_relevant("banana"));
    }

    #[test]
    fn is_animal_matches_pets_and_is_disjoint_from_relevant() {
        assert!(is_animal("dog") && is_animal("cat") && is_animal("bird"));
        assert!(!is_animal("person") && !is_animal("cup") && !is_animal("chair"));
        // Animals must never also be scene-node objects (disjoint sets).
        assert!(ANIMAL_CLASSES.iter().all(|c| !is_relevant(c)));
    }

    #[test]
    fn class_score_threshold_raises_bar_for_flicker_furniture() {
        // Small interaction objects keep the low base floor.
        assert!((class_score_threshold("cup", 0.35) - 0.35).abs() < 1e-6);
        // Flicker-prone furniture is held to ≥0.50.
        assert!((class_score_threshold("couch", 0.35) - 0.50).abs() < 1e-6);
        assert!((class_score_threshold("bed", 0.35) - 0.50).abs() < 1e-6);
        // A base above the furniture bar wins (never lowers it).
        assert!((class_score_threshold("couch", 0.60) - 0.60).abs() < 1e-6);
    }

    #[test]
    fn class_score_threshold_lowers_bar_for_hand_covered_desk_tools() {
        // keyboard/mouse get a slightly lower floor (phase 11) so the partial,
        // hand-occluded view that seeds object-permanence still clears the gate.
        assert!((class_score_threshold("keyboard", 0.30) - 0.22).abs() < 1e-6);
        assert!((class_score_threshold("mouse", 0.30) - 0.22).abs() < 1e-6);
        // …but never below the 0.20 hard floor.
        assert!((class_score_threshold("keyboard", 0.25) - 0.20).abs() < 1e-6);
    }

    // ── object tracker (filter B) ──────────────────────────────────────────

    fn od(
        class_name: &'static str,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        score: f32,
    ) -> ObjectDetection {
        ObjectDetection {
            class_id: 0,
            class_name,
            bbox: BBox { x1, y1, x2, y2 },
            score,
        }
    }

    #[test]
    fn tracker_first_detection_gets_id_at_base() {
        let mut t = ObjectTracker::new();
        let out = t.update(&[od("cup", 0.4, 0.4, 0.5, 0.55, 0.8)]);
        assert!(out.is_empty(), "birth is tentative and not published");
        assert_eq!(t.all()[0].id, OBJECT_ID_BASE);
        assert_eq!(t.all()[0].age, 1);
    }

    #[test]
    fn single_frame_object_detection_is_not_published() {
        let mut tracker = ObjectTracker::new();
        let detection = ObjectDetection {
            class_id: 41,
            class_name: "cup",
            bbox: BBox {
                x1: 0.2,
                y1: 0.2,
                x2: 0.4,
                y2: 0.5,
            },
            score: 0.9,
        };
        assert!(tracker.update(&[detection]).is_empty());
        assert_eq!(
            tracker.all().len(),
            1,
            "tentative hypothesis stays internal"
        );
    }

    #[test]
    fn object_publication_handoff_confirms_coasts_and_retires() {
        let mut tracker = ObjectTracker::with_params(0.3, 1, 0.5).with_static_hold(0);
        let detection = ObjectDetection {
            class_id: 41,
            class_name: "cup",
            bbox: BBox {
                x1: 0.2,
                y1: 0.2,
                x2: 0.4,
                y2: 0.5,
            },
            score: 0.9,
        };
        assert!(tracker.update(std::slice::from_ref(&detection)).is_empty());
        let confirmed = tracker.update(std::slice::from_ref(&detection));
        assert_eq!(confirmed.len(), 1);
        let id = confirmed[0].id;
        assert_eq!(confirmed[0].lifecycle, LifecycleState::Confirmed);
        assert_eq!(tracker.retained_ids(), vec![id]);

        assert!(tracker.update(&[]).is_empty());
        assert_eq!(tracker.all()[0].lifecycle, LifecycleState::Lost);
        assert_eq!(tracker.retained_ids(), vec![id]);
        assert!(tracker.retirements().is_empty());

        assert!(tracker.update(&[]).is_empty());
        assert!(tracker.retained_ids().is_empty());
        assert_eq!(tracker.retirements(), &[id]);
    }

    #[test]
    fn observed_object_step_maps_same_packet_geometry_to_track() {
        let binding = TrackerFrameBinding::rgb_only(12);
        let detection = od("cup", 0.4, 0.4, 0.5, 0.55, 0.8);
        let observation =
            AssociationObservation::unavailable(3, &binding, detection.bbox, detection.score);
        let mut tracker = ObjectTracker::new();
        let out = tracker
            .update_observed_bound(&binding, &[observation], &[detection])
            .unwrap();
        assert!(out.is_empty());
        assert_eq!(tracker.association_geometry()[0].track_id, OBJECT_ID_BASE);
        assert_eq!(tracker.association_geometry()[0].source_frame_id, 12);
    }

    #[test]
    fn tracker_same_object_keeps_id_across_frames() {
        let mut t = ObjectTracker::new();
        t.update(&[od("cup", 0.40, 0.40, 0.50, 0.55, 0.8)]);
        let id1 = t.all()[0].id;
        // Slightly moved next frame → high IoU, same class → same id.
        let f2 = t.update(&[od("cup", 0.41, 0.41, 0.51, 0.56, 0.8)]);
        assert_eq!(f2[0].id, id1);
        assert_eq!(f2[0].age, 2);
    }

    #[test]
    fn tracker_same_box_different_class_gets_new_id() {
        let mut t = ObjectTracker::new();
        t.update(&[od("cup", 0.4, 0.4, 0.5, 0.55, 0.8)]);
        let cup = t.all()[0].id;
        t.update(&[od("keyboard", 0.4, 0.4, 0.5, 0.55, 0.8)]);
        let kbd = t
            .all()
            .iter()
            .find(|track| track.class_name == "keyboard")
            .unwrap()
            .id;
        assert_ne!(cup, kbd, "class-aware matching must not merge cup↔keyboard");
    }

    #[test]
    fn tracker_two_objects_distinct_ids() {
        let mut t = ObjectTracker::new();
        let detections = [
            od("cup", 0.1, 0.1, 0.2, 0.2, 0.8),
            od("laptop", 0.6, 0.6, 0.9, 0.9, 0.8),
        ];
        assert!(t.update(&detections).is_empty());
        let out = t.update(&detections);
        assert_eq!(out.len(), 2);
        assert_ne!(out[0].id, out[1].id);
    }

    #[test]
    fn tracker_disappeared_object_expires_and_frees_id() {
        let mut t = ObjectTracker::with_params(0.3, 2, 0.5); // timeout 2
        let cup = od("cup", 0.4, 0.4, 0.5, 0.55, 0.8);
        t.update(std::slice::from_ref(&cup));
        let id1 = t.update(std::slice::from_ref(&cup))[0].id;
        // Absent for > timeout frames → track expires.
        for _ in 0..4 {
            let visible = t.update(&[]);
            assert!(visible.is_empty(), "no object visible while absent");
        }
        // New object reuses the freed id.
        t.update(&[od("bottle", 0.1, 0.1, 0.2, 0.3, 0.7)]);
        let id2 = t.all()[0].id;
        assert_eq!(id2, id1, "freed id should be reused");
    }

    // ── object-permanence for static fixtures (phase 11) ──────────────────

    #[test]
    fn static_fixture_held_under_hand_occlusion_then_resumes() {
        // Confirm a keyboard (age≥2), then occlude it for many frames > timeout:
        // it stays in the explicit occluded handoff, keeps its id, and resumes
        // on re-detection without becoming current interaction evidence.
        let mut t = ObjectTracker::with_params(0.3, 3, 0.5).with_static_hold(20);
        t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)]);
        let id = t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)])[0].id; // age 2 → confirmed
        for k in 0..10 {
            // absent (hands cover it) — well past timeout=3.
            let vis = t.update(&[]);
            assert!(
                vis.is_empty(),
                "occluded object is not current at miss {}",
                k + 1
            );
            let held = t.occluded().next().expect("held in occluded handoff");
            assert_eq!(held.id, id, "same id");
            assert!(held.missed > 0);
        }
        let back = t.update(&[od("keyboard", 0.41, 0.4, 0.61, 0.55, 0.6)]);
        assert_eq!(back[0].id, id, "re-detection resumes the same track");
        assert_eq!(back[0].missed, 0, "no longer remembered — seen again");
    }

    #[test]
    fn static_fixture_needs_confirmation_before_hold() {
        // A single-frame keyboard flicker (age 1) must NOT mint a held phantom.
        let mut t = ObjectTracker::with_params(0.3, 2, 0.5).with_static_hold(20);
        t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)]); // age 1 only
        for _ in 0..4 {
            assert!(
                t.update(&[]).is_empty(),
                "unconfirmed static object is not held"
            );
        }
    }

    #[test]
    fn static_hold_eventually_expires() {
        let mut t = ObjectTracker::with_params(0.3, 2, 0.5).with_static_hold(5);
        t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)]);
        t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)]); // confirmed
        for _ in 0..5 {
            assert!(t.update(&[]).is_empty(), "not published while occluded");
            assert_eq!(t.occluded().count(), 1, "retained within the window");
        }
        assert!(t.update(&[]).is_empty(), "expires past the static hold");
        assert!(t.all().is_empty());
    }

    #[test]
    fn non_static_object_is_not_held() {
        // A cup is a moved-around object — it must expire on the ordinary
        // timeout, never ride the permanence hold.
        let mut t = ObjectTracker::with_params(0.3, 2, 0.5).with_static_hold(20);
        t.update(&[od("cup", 0.4, 0.4, 0.5, 0.55, 0.8)]);
        t.update(&[od("cup", 0.4, 0.4, 0.5, 0.55, 0.8)]);
        for _ in 0..3 {
            t.update(&[]);
        }
        assert!(t.update(&[]).is_empty(), "cup expires on timeout, not held");
    }

    #[test]
    fn static_hold_zero_restores_pre11_behaviour() {
        let mut t = ObjectTracker::with_params(0.3, 2, 0.5).with_static_hold(0);
        t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)]);
        t.update(&[od("keyboard", 0.4, 0.4, 0.6, 0.55, 0.5)]);
        assert!(
            t.update(&[]).is_empty(),
            "hold disabled → keyboard not emitted while absent"
        );
    }

    #[test]
    fn tracker_brief_flicker_keeps_id() {
        let mut t = ObjectTracker::with_params(0.3, 5, 0.5);
        let cup = od("cup", 0.4, 0.4, 0.5, 0.55, 0.8);
        t.update(std::slice::from_ref(&cup));
        let id1 = t.update(std::slice::from_ref(&cup))[0].id;
        let _ = t.update(&[]); // missed one frame (within timeout)
        let back = t.update(&[od("cup", 0.4, 0.4, 0.5, 0.55, 0.8)]);
        assert_eq!(
            back[0].id, id1,
            "object reappearing within timeout keeps its id"
        );
    }

    // ── attach_masks (phase 10.4 / 10.5, synthetic masks) ──────────────────

    /// A full-frame proto-plane, all logits `v` (uniform "inside").
    fn uniform_proto() -> Vec<f32> {
        vec![10.0f32; MASK_RES * MASK_RES]
    }
    /// A proto whose left half is strongly inside, right half outside (coeff sign
    /// then selects which half a mask covers).
    fn left_half_proto() -> Vec<f32> {
        let mut p = vec![0.0f32; MASK_RES * MASK_RES];
        for y in 0..MASK_RES {
            for x in 0..MASK_RES {
                p[y * MASK_RES + x] = if x < MASK_RES / 2 { 10.0 } else { -10.0 };
            }
        }
        p
    }
    fn id_lb() -> Letterbox {
        Letterbox {
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }
    fn mk_mask(proto: &[f32], coeff: f32, b: &BBox) -> InstanceMask {
        let r = MASK_RES as u32;
        assemble_mask(proto, 1, MASK_RES, MASK_RES, &[coeff], b, id_lb(), r, r, r)
    }
    /// A seg detection with a full-bbox uniform mask.
    fn full_seg(class: &'static str, b: BBox) -> SegDetection {
        SegDetection {
            det: ObjectDetection {
                class_id: 0,
                class_name: class,
                bbox: b,
                score: 0.8,
            },
            mask: mk_mask(&uniform_proto(), 1.0, &b),
        }
    }
    fn trk(id: u32, class: &'static str, b: BBox) -> TrackedObject {
        TrackedObject {
            id,
            class_name: class,
            bbox: b,
            score: 0.8,
            age: 2,
            missed: 0,
            lifecycle: LifecycleState::Confirmed,
            pos_3d: None,
            pos_conf: 0.0,
            mask: None,
        }
    }

    #[test]
    fn attach_masks_matches_overlapping_same_class() {
        let b = BBox {
            x1: 0.4,
            y1: 0.4,
            x2: 0.6,
            y2: 0.6,
        };
        let mut objs = vec![trk(10000, "cup", b)];
        attach_masks(&mut objs, &[full_seg("cup", b)], 0.3);
        assert!(
            objs[0].mask.as_ref().is_some_and(|m| !m.is_empty()),
            "cup got its mask"
        );
    }

    #[test]
    fn attach_masks_ignores_class_mismatch() {
        let b = BBox {
            x1: 0.4,
            y1: 0.4,
            x2: 0.6,
            y2: 0.6,
        };
        let mut objs = vec![trk(10000, "cup", b)];
        attach_masks(&mut objs, &[full_seg("laptop", b)], 0.3);
        assert!(objs[0].mask.is_none(), "class mismatch → no mask");
    }

    #[test]
    fn attach_masks_ignores_disjoint_boxes() {
        let mut objs = vec![trk(
            10000,
            "cup",
            BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.1,
                y2: 0.1,
            },
        )];
        let far = full_seg(
            "cup",
            BBox {
                x1: 0.8,
                y1: 0.8,
                x2: 0.9,
                y2: 0.9,
            },
        );
        attach_masks(&mut objs, &[far], 0.3);
        assert!(objs[0].mask.is_none(), "no bbox overlap → no attach");
    }

    #[test]
    fn attach_masks_resets_stale_mask_when_no_seg() {
        let b = BBox {
            x1: 0.4,
            y1: 0.4,
            x2: 0.6,
            y2: 0.6,
        };
        let mut objs = vec![trk(10000, "cup", b)];
        objs[0].mask = Some(mk_mask(&uniform_proto(), 1.0, &b)); // stale mask
        attach_masks(&mut objs, &[], 0.3); // no seg this tick
        assert!(
            objs[0].mask.is_none(),
            "stale mask cleared when no seg supports it"
        );
    }

    #[test]
    fn attach_masks_one_seg_per_object_no_double_assign() {
        // Two distinct cups, two segs → each cup gets its own seg's mask.
        let ba = BBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.2,
            y2: 0.2,
        };
        let bb = BBox {
            x1: 0.7,
            y1: 0.7,
            x2: 0.8,
            y2: 0.8,
        };
        let mut objs = vec![trk(10000, "cup", ba), trk(10001, "cup", bb)];
        attach_masks(&mut objs, &[full_seg("cup", ba), full_seg("cup", bb)], 0.3);
        assert!(
            objs[0].mask.is_some() && objs[1].mask.is_some(),
            "both cups masked"
        );
    }

    #[test]
    fn attach_masks_mask_iou_breaks_bbox_ties() {
        // Two overlapping same-class objects with an *identical* bbox → equal
        // bbox-IoU to both segs. Each carries a prior silhouette (obj A = left
        // half, obj B = right half); two segs carry the matching halves. The
        // mask-IoU tiebreak must route the left seg to A and the right seg to B
        // (phase 10.5 instance separation).
        let full = BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        let mut a = trk(10000, "book", full);
        let mut b = trk(10001, "book", full);
        a.mask = Some(mk_mask(&left_half_proto(), 1.0, &full)); // left silhouette
        b.mask = Some(mk_mask(&left_half_proto(), -1.0, &full)); // right silhouette
        let mut objs = vec![a, b];
        let seg_left = SegDetection {
            det: ObjectDetection {
                class_id: 0,
                class_name: "book",
                bbox: full,
                score: 0.8,
            },
            mask: mk_mask(&left_half_proto(), 1.0, &full),
        };
        let seg_right = SegDetection {
            det: ObjectDetection {
                class_id: 0,
                class_name: "book",
                bbox: full,
                score: 0.8,
            },
            mask: mk_mask(&left_half_proto(), -1.0, &full),
        };
        attach_masks(&mut objs, &[seg_left, seg_right], 0.3);
        // obj A should now carry a left-half mask (points all x<0.5).
        let am = objs[0].mask.as_ref().expect("A masked");
        assert!(
            am.set_points().all(|p| p[0] < 0.5),
            "A got the LEFT seg (mask-IoU tiebreak)"
        );
        let bm = objs[1].mask.as_ref().expect("B masked");
        assert!(bm.set_points().all(|p| p[0] >= 0.5), "B got the RIGHT seg");
    }

    // ── coco table sanity ──────────────────────────────────────────────────

    #[test]
    fn coco_table_is_80_classes() {
        assert_eq!(COCO_CLASSES.len(), 80);
        assert_eq!(COCO_CLASSES[0], "person");
        assert_eq!(COCO_CLASSES[79], "toothbrush");
    }

    // ── end-to-end model load (gated on file present, like mvm-detect) ──────

    #[test]
    fn model_loads_and_runs_when_present() {
        let model_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/yolo_object_detect.onnx"
        ));
        if !model_path.exists() {
            eprintln!(
                "Skipping: {} not found (download bootstrap weights)",
                model_path.display()
            );
            return;
        }
        let mut det = YoloDetector::from_path(&model_path).unwrap();
        let img = DynamicImage::new_rgb8(640, 480);
        let result = det.detect(&img).unwrap();
        eprintln!("black image → {} objects", result.len());
        // Black image: typically zero detections, but the chain must not panic.
    }

    #[test]
    fn detects_person_on_reference_photo() {
        let model_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/yolo_object_detect.onnx"
        ));
        let ref_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/references/p_V_01.jpg"
        ));
        if !model_path.exists() || !ref_path.exists() {
            eprintln!("Skipping: model or reference photo not present");
            return;
        }
        let mut det = YoloDetector::from_path(&model_path).unwrap();
        let img = image::open(&ref_path).unwrap();
        let result = det.detect(&img).unwrap();
        for o in &result {
            eprintln!(
                "  {} ({:.2}) @ [{:.2},{:.2},{:.2},{:.2}]",
                o.class_name, o.score, o.bbox.x1, o.bbox.y1, o.bbox.x2, o.bbox.y2
            );
        }
        // A portrait photo must yield at least one "person" — validates that the
        // letterbox + RGB + /255 preprocessing matches the model's expectations.
        assert!(
            result.iter().any(|o| o.class_name == "person"),
            "expected a 'person' detection on the reference portrait, got {result:?}"
        );
    }
}
