//! HumanAgent (phase 8.C.5) — group G₁. Wraps a (human-only) `BodyPrimaryPipeline`
//! and owns its own `PersonTracker`. Every frame: process → track → project the
//! active tracks into `WorldState.people`. The pipeline is constructed with no
//! scene/depth detector (those are their own agents now), so it does only the
//! body-primary chain; objects/depth come from `SceneAgent`/`DepthAgent`.

use crate::agents::{PersonMaskSlot, project_people};
use crate::pipeline::BodyPrimaryPipeline;
use mvm_core::{BBox, Detection, mask::InstanceMask};
use mvm_mesh::{Agent, AgentHealth, MeshContext};

/// Bbox-IoU floor for binding a published person silhouette to a body detection
/// (11.A3). The seg person-bbox and the body-primary bbox track the same person,
/// so a real match overlaps strongly; a loose floor tolerates the cadence lag.
const PERSON_MASK_IOU: f32 = 0.3;

/// Greedily bind published person silhouettes to this frame's body detections by
/// bbox-IoU (phase 11.A3), one mask per detection. Each bound mask rides into the
/// tracker cost as the association's silhouette cue. Pure geometry — a mask whose
/// bbox overlaps no detection above the floor is simply dropped (the person may be
/// off the body cadence this frame). Mirrors `mvm_scene::attach_masks`.
fn attach_person_masks(
    dets: &mut [Detection],
    person_masks: &[(BBox, InstanceMask)],
    iou_thresh: f32,
) {
    let mut used = vec![false; person_masks.len()];
    for det in dets.iter_mut() {
        let mut best: Option<(usize, f32)> = None;
        for (i, (mbbox, _)) in person_masks.iter().enumerate() {
            if used[i] {
                continue;
            }
            let iou = det.bbox.iou(mbbox);
            if iou > iou_thresh && best.is_none_or(|(_, b)| iou > b) {
                best = Some((i, iou));
            }
        }
        if let Some((i, _)) = best {
            used[i] = true;
            det.mask = Some(person_masks[i].1.clone());
        }
    }
}

pub(crate) struct HumanAgent {
    pipeline: BodyPrimaryPipeline,
    tracker: mvm_core::PersonTracker,
    /// Shared person-mask slot written by the `SceneAgent` (11.A3). Read each
    /// frame and matched onto body detections before the tracker tick. Holds
    /// `None` until the first seg cadence hit (or forever when `--masks` is off).
    person_masks: PersonMaskSlot,
}

impl HumanAgent {
    pub(crate) fn new(
        pipeline: BodyPrimaryPipeline,
        tracker: mvm_core::PersonTracker,
        person_masks: PersonMaskSlot,
    ) -> Self {
        Self {
            pipeline,
            tracker,
            person_masks,
        }
    }
}

impl Agent for HumanAgent {
    fn name(&self) -> String {
        "human".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0u64;
        let mut last_id: Option<u64> = None;
        let mut epoch: Option<std::time::Instant> = None; // motion timestamp origin (9.8)
        while ctx.running() {
            if let Some(frame) = ctx.frames.latest()
                && last_id != Some(frame.id)
            {
                last_id = Some(frame.id);
                let prev_tracks: Vec<mvm_core::BBox> =
                    self.tracker.active().map(|t| t.bbox).collect();
                // Id-locked tracks (phase 9.6) → their bodies skip SFace this frame.
                let locked_tracks: Vec<mvm_core::BBox> = self
                    .tracker
                    .active()
                    .filter(|t| t.is_id_locked())
                    .map(|t| t.bbox)
                    .collect();
                // Emotion-fresh tracks (phase 9.7) → their bodies skip EmotiEffLib.
                let emotion_fresh: Vec<mvm_core::BBox> = self
                    .tracker
                    .active()
                    .filter(|t| !t.needs_emotion())
                    .map(|t| t.bbox)
                    .collect();
                // Id-locked tracks (phase 9.9) → pose-ROI recovery if the detector
                // drops them. Locked-only (not merely established) so a recognised /
                // cosine-stable body is sustained through a detector gap, but an anon
                // fragment is NOT propped up into a coexisting phantom.
                let predicted_rois: Vec<mvm_core::BBox> = self
                    .tracker
                    .active()
                    .filter(|t| t.is_id_locked())
                    .map(|t| t.bbox)
                    .collect();
                let mut result = self.pipeline.process(
                    frame.image.as_ref(),
                    frame.id,
                    &prev_tracks,
                    &locked_tracks,
                    &emotion_fresh,
                    &predicted_rois,
                );
                // 11.A3: bind the SceneAgent's latest person silhouettes onto
                // this frame's body detections (bbox-IoU) so the tracker cost
                // can use them. Lagged (seg cadence) and cross-thread — a race
                // that misses just leaves detections maskless (term degrades to
                // zero). The ≥2-people gate lives in the tracker.
                if let Some(masks) = self.person_masks.lock().ok().and_then(|s| s.clone()) {
                    attach_person_masks(&mut result.detections, &masks, PERSON_MASK_IOU);
                }
                // Monotonic seconds since the first frame → motion velocities (9.8).
                let t_secs = frame
                    .captured
                    .duration_since(*epoch.get_or_insert(frame.captured))
                    .as_secs_f64();
                self.tracker.tick_motion(
                    &result.detections,
                    &result.low_conf,
                    &result.landmarks,
                    t_secs,
                );
                project_people(
                    &ctx.world,
                    &self.tracker,
                    &result.bodies,
                    frame.id,
                    frame.captured,
                );
                ticks += 1;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{Config, Detection, WorldState};
    use std::sync::{Arc, RwLock};
    use std::time::Instant;

    /// An anonymous body (no matched face) projects to a person with no identity
    /// — the body-primary FRONTIER invariant (track without a frontal face).
    #[test]
    fn anonymous_body_has_no_identity() {
        let config = Config::default();
        let mut tracker = mvm_core::PersonTracker::with_config(&config);
        tracker.tick(&[Detection::bare(0.2, 0.2, 0.4, 0.6)]);

        let world = Arc::new(RwLock::new(WorldState::new(Instant::now())));
        project_people(&world, &tracker, &[], 1, Instant::now());

        let w = world.read().unwrap();
        assert_eq!(w.people.len(), 1);
        assert!(
            w.people[0].identity.is_none(),
            "anonymous body → identity None"
        );
    }

    /// Two distinct bodies → two independent people in the world (multi-persona).
    #[test]
    fn two_bodies_project_two_people() {
        let config = Config::default();
        let mut tracker = mvm_core::PersonTracker::with_config(&config);
        tracker.tick(&[
            Detection::bare(0.05, 0.2, 0.25, 0.7),
            Detection::bare(0.65, 0.2, 0.85, 0.7),
        ]);

        let world = Arc::new(RwLock::new(WorldState::new(Instant::now())));
        project_people(&world, &tracker, &[], 2, Instant::now());

        assert_eq!(world.read().unwrap().people.len(), 2);
    }

    /// A distinguishable non-empty mask (a single set cell at `i`) so tests can
    /// tell which silhouette bound to which detection.
    fn mask_bit(i: usize) -> InstanceMask {
        let mut bits = vec![0u8; mvm_core::mask::MASK_RES * mvm_core::mask::MASK_RES];
        bits[i] = 1;
        InstanceMask::from_bits(bits)
    }

    #[test]
    fn person_masks_bind_to_overlapping_detections() {
        // 11.A3 matcher: each body detection takes the bbox-overlapping silhouette,
        // exclusively; a non-overlapping mask binds to nobody.
        let mut dets = vec![
            Detection::bare(0.05, 0.2, 0.25, 0.7), // left body
            Detection::bare(0.65, 0.2, 0.85, 0.7), // right body
        ];
        let left_mask = mask_bit(0);
        let right_mask = mask_bit(100);
        let person_masks = vec![
            (
                BBox {
                    x1: 0.06,
                    y1: 0.2,
                    x2: 0.26,
                    y2: 0.7,
                },
                left_mask.clone(),
            ), // → det 0
            (
                BBox {
                    x1: 0.64,
                    y1: 0.2,
                    x2: 0.84,
                    y2: 0.7,
                },
                right_mask.clone(),
            ), // → det 1
            (
                BBox {
                    x1: 0.40,
                    y1: 0.2,
                    x2: 0.50,
                    y2: 0.7,
                },
                mask_bit(200),
            ), // matches neither
        ];
        attach_person_masks(&mut dets, &person_masks, PERSON_MASK_IOU);
        assert!(dets[0].mask.is_some(), "left body got a silhouette");
        assert!(dets[1].mask.is_some(), "right body got a silhouette");
        // Exclusivity: the two bound masks are the two overlapping ones.
        assert!(dets[0].mask.as_ref().unwrap().iou(&left_mask) > 0.99);
        assert!(dets[1].mask.as_ref().unwrap().iou(&right_mask) > 0.99);
    }

    #[test]
    fn non_overlapping_mask_leaves_detection_bare() {
        // A far-away silhouette must not attach (no false silhouette on a body).
        let mut dets = vec![Detection::bare(0.05, 0.2, 0.25, 0.7)];
        let person_masks = vec![(
            BBox {
                x1: 0.7,
                y1: 0.2,
                x2: 0.9,
                y2: 0.7,
            },
            mask_bit(0),
        )];
        attach_person_masks(&mut dets, &person_masks, PERSON_MASK_IOU);
        assert!(dets[0].mask.is_none(), "no overlap → no mask");
    }
}
