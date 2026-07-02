//! Frame + the single-slot "latest frame wins" channel (phase 8.C).
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
        Self { id, image: Arc::new(image), captured }
    }
}

/// Single-slot latest-value cell shared between the capture agent (producer) and
/// all consumer agents. Cloning shares the same slot.
#[derive(Clone, Default)]
pub struct LatestFrame {
    slot: Arc<Mutex<Option<Frame>>>,
}

impl LatestFrame {
    pub fn new() -> Self {
        Self { slot: Arc::new(Mutex::new(None)) }
    }

    /// Overwrite with the newest frame (producer).
    pub fn publish(&self, frame: Frame) {
        *self.slot.lock().unwrap() = Some(frame);
    }

    /// Read the freshest frame, if any (consumer). Clones out (cheap — Arc image).
    pub fn latest(&self) -> Option<Frame> {
        self.slot.lock().unwrap().clone()
    }
}

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
}
