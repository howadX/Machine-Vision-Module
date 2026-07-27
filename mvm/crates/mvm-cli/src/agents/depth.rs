//! DepthAgent (phase 8.C.5) — group G₄ (environment). Owns the heavy
//! Depth-Anything estimator (~230ms), runs on a sparse cadence (default every
//! 30th frame), and publishes the raw relative `DepthMap` to a **shared slot**.
//!
//! It does NOT anchor objects — anchoring needs both people and objects and runs
//! on the fusion step (8.C.7), which reads this same slot. The slot lives in
//! `mvm-cli` (not `MeshContext`) so `mvm-mesh` need not depend on `mvm-depth`.

use crate::agents::cadence_due;
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::sync::{Arc, Mutex};

/// Shared latest-depth slot: the `DepthAgent` writes it, the fusion step reads it.
/// The map is behind an inner `Arc` so the reader (fusion) clones the handle
/// cheaply and releases the slot lock before doing work — no ~1 MB map copy.
pub(crate) type DepthSlot = Arc<Mutex<Option<Arc<mvm_depth::DepthMap>>>>;

pub(crate) struct DepthAgent {
    estimator: mvm_depth::DepthEstimator,
    slot: DepthSlot,
    cadence: u32,
}

impl DepthAgent {
    pub(crate) fn new(estimator: mvm_depth::DepthEstimator, slot: DepthSlot, cadence: u32) -> Self {
        Self {
            estimator,
            slot,
            cadence,
        }
    }
}

impl Agent for DepthAgent {
    fn name(&self) -> String {
        "depth".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0u64;
        let mut errors = 0u64;
        let mut last_id: Option<u64> = None;
        while ctx.running() {
            if let Some(frame) = ctx.frames.latest() {
                if last_id != Some(frame.id) && cadence_due(frame.id, self.cadence) {
                    last_id = Some(frame.id);
                    match self.estimator.estimate(frame.image.as_ref()) {
                        Ok(map) => {
                            if let Ok(mut slot) = self.slot.lock() {
                                *slot = Some(Arc::new(map));
                            }
                            ticks += 1;
                        }
                        Err(e) => {
                            eprintln!("  depth: estimate error: {e}");
                            errors += 1;
                        }
                    }
                }
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_starts_empty_and_accepts_a_map() {
        let slot: DepthSlot = Arc::new(Mutex::new(None));
        assert!(slot.lock().unwrap().is_none());
        // A reader (fusion) and writer (agent) share the same Arc.
        let writer = slot.clone();
        *writer.lock().unwrap() = Some(Arc::new(mvm_depth::DepthMap {
            data: vec![0.5; 4],
            w: 2,
            h: 2,
        }));
        let got = slot.lock().unwrap();
        assert!(got.is_some(), "fusion sees the published map");
        assert_eq!(got.as_ref().unwrap().w, 2);
    }

    #[test]
    fn depth_cadence_is_sparse() {
        // Default depth cadence (30) is due rarely.
        assert!(cadence_due(0, 30));
        assert!(!cadence_due(1, 30));
        assert!(!cadence_due(29, 30));
        assert!(cadence_due(30, 30));
    }
}
