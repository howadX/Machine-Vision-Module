//! Frame + the generic single-slot "latest value wins" channel.
//!
//! Proven by spike B (`spikes/mesh-spike/src/bin/latest_frame.rs`): a fast producer
//! (camera) feeds slow consumers (agents); each consumer always reads the FRESHEST
//! frame, intermediate frames silently dropped — no backlog, no catch-up-from-oldest
//! (which `tokio::broadcast` would do on lag). std-only: `Arc<Mutex<Option<Frame>>>`
//! (an `ArcSwap` would be the lock-free production variant; semantics identical).

use image::DynamicImage;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One captured frame, cheaply cloneable (`image` behind an `Arc`).
#[derive(Clone)]
pub struct Frame {
    pub id: u64,
    pub image: Arc<DynamicImage>,
    pub captured: Instant,
}

impl Frame {
    pub fn new(id: u64, image: DynamicImage, captured: Instant) -> Self {
        Self {
            id,
            image: Arc::new(image),
            captured,
        }
    }
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
}
