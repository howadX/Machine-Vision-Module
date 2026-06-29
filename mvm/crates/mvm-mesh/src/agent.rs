//! The `Agent` contract + the shared context the mesh hands each agent (phase 8.C).
//!
//! Decision 8.0.1: **`std::thread`, not tokio.** Each agent owns a thread and runs
//! `run(&ctx)` to completion — it loops at its OWN cadence (human ~every frame,
//! depth every Nth), pulling the latest frame and writing its slice of the shared
//! `WorldState`. A slow agent never blocks a fast one (no head-of-line blocking) —
//! the whole reason the mesh exists once heavy detectors (depth ~230ms) landed.

use crate::frame::LatestFrame;
use mvm_core::WorldState;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

/// A recoverable per-step agent error (logged, counted, does not kill the thread).
#[derive(Debug, Clone)]
pub struct AgentError(pub String);

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for AgentError {}

/// Liveness summary an agent returns when it stops (joined by the orchestrator).
#[derive(Debug, Clone, PartialEq)]
pub struct AgentHealth {
    pub name: String,
    /// Successful work iterations.
    pub ticks: u64,
    /// Recoverable errors encountered.
    pub errors: u64,
}

/// Shared handles every agent gets: the latest-frame channel, the blackboard, and
/// the cooperative stop flag.
#[derive(Clone)]
pub struct MeshContext {
    pub frames: LatestFrame,
    pub world: Arc<RwLock<WorldState>>,
    pub stop: Arc<AtomicBool>,
}

impl MeshContext {
    /// True while the mesh is running (agents loop on this).
    pub fn running(&self) -> bool {
        !self.stop.load(Ordering::Relaxed)
    }

    /// Signal all agents to stop (used by the capture agent when its frame budget
    /// is exhausted, or by the CLI on shutdown).
    pub fn request_stop(&self) {
        self.stop.store(true, Ordering::Relaxed);
    }
}

/// A mesh participant. One thread per agent; `run` blocks until `ctx.running()`
/// is false, then returns its health. Must be `Send` (moved into a thread).
pub trait Agent: Send {
    fn name(&self) -> String;
    fn run(&mut self, ctx: &MeshContext) -> AgentHealth;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    fn ctx() -> MeshContext {
        MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(WorldState::new(Instant::now()))),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn running_reflects_stop_flag() {
        let c = ctx();
        assert!(c.running());
        c.request_stop();
        assert!(!c.running());
    }

    #[test]
    fn agent_error_displays() {
        let e = AgentError("boom".into());
        assert_eq!(e.to_string(), "boom");
    }

    #[test]
    fn agent_health_equality() {
        let a = AgentHealth { name: "x".into(), ticks: 3, errors: 0 };
        let b = AgentHealth { name: "x".into(), ticks: 3, errors: 0 };
        assert_eq!(a, b);
    }
}
