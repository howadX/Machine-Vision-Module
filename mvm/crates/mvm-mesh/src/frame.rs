//! Frame + the generic single-slot "latest value wins" channel.
//!
//! Proven by spike B (`spikes/mesh-spike/src/bin/latest_frame.rs`): a fast producer
//! (camera) feeds slow consumers (agents); each consumer always reads the FRESHEST
//! frame, intermediate frames silently dropped — no backlog, no catch-up-from-oldest
//! (which `tokio::broadcast` would do on lag). std-only: `Arc<Mutex<Option<Frame>>>`
//! (an `ArcSwap` would be the lock-free production variant; semantics identical).

use image::DynamicImage;
use mvm_core::{
    CalibrationRef, DepthStatus, MetricDepthFrame, RegisteredDepthOwners, SensorHealth,
    SensorSourceId, SensorStamp,
};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One captured frame, cheaply cloneable (`image` behind an `Arc`).
#[derive(Clone)]
pub struct Frame {
    pub id: u64,
    pub image: Arc<DynamicImage>,
    pub captured: Instant,
    pub captured_at_unix_ms: i64,
    /// Sensor provenance/evidence attached to the RGB clock tick.
    pub sensor: Arc<SensorFrameBundle>,
}

/// Immutable sensor packet. RGB stays directly accessible on [`Frame`] for
/// compatibility; optional metric evidence is carried here without promotion to
/// canonical scene facts.
#[derive(Clone)]
pub struct SensorFrameBundle {
    pub source: SensorSourceId,
    pub rgb_stamp: SensorStamp,
    pub depth_status: DepthStatus,
    pub metric_depth: Option<Arc<MetricDepthFrame>>,
    pub registered_depth_owners: Option<Arc<RegisteredDepthOwners>>,
    pub calibration: Option<CalibrationRef>,
    pub health: SensorHealth,
}

impl Frame {
    pub fn new(id: u64, image: DynamicImage, captured: Instant) -> Self {
        let captured_at_unix_ms = unix_ms(std::time::SystemTime::now());
        Self::new_stamped(id, image, captured, captured_at_unix_ms)
    }

    pub fn new_stamped(
        id: u64,
        image: DynamicImage,
        captured: Instant,
        captured_at_unix_ms: i64,
    ) -> Self {
        let image = Arc::new(image);
        let host_monotonic_ns = monotonic_ns();
        Self {
            id,
            image,
            captured,
            captured_at_unix_ms,
            sensor: Arc::new(SensorFrameBundle {
                source: SensorSourceId::v4l2("compatibility"),
                rgb_stamp: SensorStamp {
                    source_sequence: id,
                    host_monotonic_ns,
                    wall_clock_unix_ms: captured_at_unix_ms,
                    clock_domain: mvm_core::SensorClockDomain::HostMonotonic,
                    device_timestamp_raw: None,
                },
                depth_status: DepthStatus::Unavailable,
                metric_depth: None,
                registered_depth_owners: None,
                calibration: None,
                health: SensorHealth::default(),
            }),
        }
    }

    pub fn from_sensor(
        id: u64,
        image: Arc<DynamicImage>,
        captured: Instant,
        bundle: SensorFrameBundle,
    ) -> Self {
        Self {
            id,
            image,
            captured,
            captured_at_unix_ms: bundle.rgb_stamp.wall_clock_unix_ms,
            sensor: Arc::new(bundle),
        }
    }
}

fn unix_ms(time: std::time::SystemTime) -> i64 {
    time.duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

fn monotonic_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX)
        .max(1)
}

/// Generic single-slot latest-value cell shared between one producer and any
/// number of consumers. Publishing replaces the previous value without a
/// backlog; cloning the cell shares the same slot.
///
/// Values are cloned out, so callers cannot mutate a value after publication
/// through this channel. Phase 18 observation packets use immutable structs and
/// cheap `Arc` handles for large evidence.
pub struct LatestValue<T> {
    slot: Arc<Mutex<Option<T>>>,
}

impl<T> Clone for LatestValue<T> {
    fn clone(&self) -> Self {
        Self {
            slot: Arc::clone(&self.slot),
        }
    }
}

impl<T> Default for LatestValue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> LatestValue<T> {
    pub fn new() -> Self {
        Self {
            slot: Arc::new(Mutex::new(None)),
        }
    }

    /// Overwrite with the newest value (producer).
    pub fn publish(&self, value: T) {
        *self.slot.lock().unwrap() = Some(value);
    }
}

impl<T: Clone> LatestValue<T> {
    /// Read the freshest value, if any. The stored value remains available to
    /// every consumer and is never drained by a read.
    pub fn latest(&self) -> Option<T> {
        self.slot.lock().unwrap().clone()
    }
}

/// Compatibility name for the capture-frame channel.
pub type LatestFrame = LatestValue<Frame>;

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(id: u64) -> Frame {
        Frame::new(id, DynamicImage::new_rgb8(8, 8), Instant::now())
    }

    #[test]
    fn empty_latest_is_none() {
        assert!(LatestFrame::new().latest().is_none());
    }

    #[test]
    fn publish_then_latest_returns_newest_and_drops_intermediate() {
        let lf = LatestFrame::new();
        lf.publish(frame(1));
        lf.publish(frame(2));
        lf.publish(frame(3));
        // Only the freshest survives — intermediate (2) is dropped, no backlog.
        assert_eq!(lf.latest().unwrap().id, 3);
    }

    #[test]
    fn clone_shares_the_same_slot() {
        let a = LatestFrame::new();
        let b = a.clone();
        a.publish(frame(7));
        assert_eq!(b.latest().unwrap().id, 7, "clone sees producer's publish");
    }

    #[test]
    fn generic_latest_value_supports_typed_observations() {
        #[derive(Clone, Debug, PartialEq, Eq)]
        struct Observation {
            source_frame_id: u64,
            labels: Vec<&'static str>,
        }

        let observations = LatestValue::new();
        observations.publish(Observation {
            source_frame_id: 12,
            labels: vec!["person"],
        });

        assert_eq!(
            observations.latest(),
            Some(Observation {
                source_frame_id: 12,
                labels: vec!["person"],
            })
        );
    }

    #[test]
    fn reads_are_non_destructive_for_independent_consumers() {
        let producer = LatestValue::new();
        let consumer_a = producer.clone();
        let consumer_b = producer.clone();
        producer.publish(42_u64);

        assert_eq!(consumer_a.latest(), Some(42));
        assert_eq!(consumer_b.latest(), Some(42));
        assert_eq!(consumer_a.latest(), Some(42));
    }

    #[test]
    fn compatibility_frame_is_explicitly_rgb_only() {
        let frame = frame(9);
        assert_eq!(frame.sensor.depth_status, DepthStatus::Unavailable);
        assert!(frame.sensor.metric_depth.is_none());
        assert_eq!(frame.sensor.rgb_stamp.source_sequence, 9);
        assert_eq!(frame.image.width(), 8);
    }

    #[test]
    fn cloned_frame_shares_immutable_sensor_payload() {
        let frame = frame(4);
        let cloned = frame.clone();
        assert!(Arc::ptr_eq(&frame.image, &cloned.image));
        assert!(Arc::ptr_eq(&frame.sensor, &cloned.sensor));
    }
}
