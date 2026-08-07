//! Runtime-only stamped expert observations (phase 18.1).
//!
//! These packets preserve the camera frame that produced each asynchronous
//! expert result. Raw depth and masks remain evidence handles rather than
//! serializable canonical scene facts.

use mvm_core::{BBox, ObjectState, PersonState, mask::InstanceMask};
use mvm_fusion::canonical::AnimalObservation;
use mvm_mesh::{Frame, LatestValue};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObservationStamp {
    pub source_frame_id: u64,
    pub captured_at: Instant,
    pub produced_at: Instant,
    pub captured_at_unix_ms: i64,
    pub produced_at_unix_ms: i64,
}

impl ObservationStamp {
    pub(crate) fn new(
        source_frame_id: u64,
        captured_at: Instant,
        captured_at_unix_ms: i64,
    ) -> Self {
        Self {
            source_frame_id,
            captured_at,
            produced_at: Instant::now(),
            captured_at_unix_ms,
            produced_at_unix_ms: unix_ms_now(),
        }
    }

    pub(crate) fn from_frame(frame: &Frame) -> Self {
        Self::new(frame.id, frame.captured, frame.captured_at_unix_ms)
    }

    pub(crate) fn age_at(self, camera_frame_id: u64) -> Option<u64> {
        camera_frame_id.checked_sub(self.source_frame_id)
    }
}

fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
pub(crate) struct HumanObservationFrame {
    pub stamp: ObservationStamp,
    pub people: Arc<Vec<PersonState>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SceneObservationFrame {
    pub stamp: ObservationStamp,
    pub objects: Arc<Vec<ObjectState>>,
    pub animals: Arc<Vec<AnimalObservation>>,
    pub person_masks: Arc<Vec<(BBox, InstanceMask)>>,
}

#[derive(Debug, Clone)]
pub(crate) struct DepthObservationFrame {
    pub stamp: ObservationStamp,
    pub map: Arc<mvm_depth::DepthMap>,
}

#[derive(Debug, Clone)]
pub(crate) struct MotionObservationFrame {
    pub stamp: ObservationStamp,
    pub camera_motion: mvm_core::canonical::SceneFact<mvm_core::canonical::RelativeCameraMotion>,
}

#[derive(Clone)]
pub(crate) struct FrameEvidenceHandles {
    pub rgb: Arc<image::DynamicImage>,
    pub depth: Option<DepthObservationFrame>,
    pub human: Option<HumanObservationFrame>,
    pub scene: Option<SceneObservationFrame>,
}

#[derive(Clone)]
pub(crate) struct FusedFramePacket {
    pub scene: Arc<mvm_core::canonical::FusedSceneFrame>,
    pub evidence: FrameEvidenceHandles,
}

pub(crate) type HumanObservationBus = LatestValue<HumanObservationFrame>;
pub(crate) type SceneObservationBus = LatestValue<SceneObservationFrame>;
pub(crate) type DepthObservationBus = LatestValue<DepthObservationFrame>;
pub(crate) type MotionObservationBus = LatestValue<MotionObservationFrame>;
pub(crate) type FusedFrameBus = LatestValue<FusedFramePacket>;
/// Runtime output of the optional Cortex adapter. It is intentionally separate
/// from canonical facts: Cortex publishes an interpretation, never a mutation.
pub(crate) type CortexFrameBus = LatestValue<mvm_cortex::CortexFrame>;
pub(crate) type AttentionPolicyBus = LatestValue<mvm_cortex::AttentionPolicy>;
pub(crate) type PolicyOutcomeBus = LatestValue<mvm_cortex::PolicyOutcomeFrame>;

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::canonical::{
        EventTailView, FactReason, FrameStamp, FusedSceneFrame, SceneFact, SceneQuality,
    };

    fn fused_scene(frame_id: u64) -> FusedSceneFrame {
        let timestamp = i64::try_from(frame_id).unwrap() * 100;
        FusedSceneFrame {
            stamp: FrameStamp::new("run", frame_id, timestamp, timestamp + 1).unwrap(),
            sources: Vec::new(),
            entities: Vec::new(),
            relations: Vec::new(),
            hand_interactions: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            spatial: Default::default(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn observation_age_rejects_future_results() {
        let now = Instant::now();
        let stamp = ObservationStamp {
            source_frame_id: 12,
            captured_at: now,
            produced_at: now,
            captured_at_unix_ms: 1_000,
            produced_at_unix_ms: 1_001,
        };
        assert_eq!(stamp.age_at(15), Some(3));
        assert_eq!(stamp.age_at(11), None);
    }

    #[test]
    fn typed_bus_keeps_source_stamp_with_depth_handle() {
        let bus = DepthObservationBus::new();
        let frame = Frame::new_stamped(
            7,
            image::DynamicImage::new_rgb8(2, 2),
            Instant::now(),
            1_000,
        );
        let stamp = ObservationStamp::from_frame(&frame);
        bus.publish(DepthObservationFrame {
            stamp,
            map: Arc::new(mvm_depth::DepthMap {
                data: vec![0.5; 4],
                w: 2,
                h: 2,
            }),
        });

        let observed = bus.latest().expect("published depth observation");
        assert_eq!(observed.stamp.source_frame_id, 7);
        assert_eq!(observed.stamp.captured_at_unix_ms, 1_000);
        assert_eq!(observed.map.w, 2);
    }

    #[test]
    fn publishing_late_result_cannot_mutate_previous_fused_packet() {
        let bus = FusedFrameBus::new();
        let image = Arc::new(image::DynamicImage::new_rgb8(2, 2));
        let evidence = FrameEvidenceHandles {
            rgb: Arc::clone(&image),
            depth: None,
            human: None,
            scene: None,
        };
        bus.publish(FusedFramePacket {
            scene: Arc::new(fused_scene(1)),
            evidence: evidence.clone(),
        });
        let first = bus.latest().expect("first packet");
        let serialized = serde_json::to_string(first.scene.as_ref()).unwrap();

        bus.publish(FusedFramePacket {
            scene: Arc::new(fused_scene(2)),
            evidence,
        });

        assert_eq!(first.scene.stamp.frame_id, 1);
        assert_eq!(
            serde_json::to_string(first.scene.as_ref()).unwrap(),
            serialized
        );
        assert_eq!(bus.latest().unwrap().scene.stamp.frame_id, 2);
    }
}
