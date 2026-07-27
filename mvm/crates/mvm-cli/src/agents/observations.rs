//! Runtime-only stamped expert observations (phase 18.1).
//!
//! These packets preserve the camera frame that produced each asynchronous
//! expert result. Raw depth and masks remain evidence handles rather than
//! serializable canonical scene facts.

use mvm_core::{BBox, ObjectState, PersonState, mask::InstanceMask};
use mvm_mesh::LatestValue;
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObservationStamp {
    pub source_frame_id: u64,
    pub captured_at: Instant,
    pub produced_at: Instant,
}

impl ObservationStamp {
    pub(crate) fn new(source_frame_id: u64, captured_at: Instant) -> Self {
        Self {
            source_frame_id,
            captured_at,
            produced_at: Instant::now(),
        }
    }

    pub(crate) fn age_at(self, camera_frame_id: u64) -> Option<u64> {
        camera_frame_id.checked_sub(self.source_frame_id)
    }
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
    pub animals: Arc<Vec<BBox>>,
    pub person_masks: Arc<Vec<(BBox, InstanceMask)>>,
}

#[derive(Debug, Clone)]
pub(crate) struct DepthObservationFrame {
    pub stamp: ObservationStamp,
    pub map: Arc<mvm_depth::DepthMap>,
}

pub(crate) type HumanObservationBus = LatestValue<HumanObservationFrame>;
pub(crate) type SceneObservationBus = LatestValue<SceneObservationFrame>;
pub(crate) type DepthObservationBus = LatestValue<DepthObservationFrame>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observation_age_rejects_future_results() {
        let now = Instant::now();
        let stamp = ObservationStamp {
            source_frame_id: 12,
            captured_at: now,
            produced_at: now,
        };
        assert_eq!(stamp.age_at(15), Some(3));
        assert_eq!(stamp.age_at(11), None);
    }

    #[test]
    fn typed_bus_keeps_source_stamp_with_depth_handle() {
        let bus = DepthObservationBus::new();
        let stamp = ObservationStamp::new(7, Instant::now());
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
        assert_eq!(observed.map.w, 2);
    }
}
