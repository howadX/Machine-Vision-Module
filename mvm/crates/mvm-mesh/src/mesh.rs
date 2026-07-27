//! `AgentMesh` orchestrator (phase 8.C): spawns one `std::thread` per agent over a
//! shared `LatestFrame` + `Arc<RwLock<WorldState>>` blackboard + `AtomicBool` stop.
//! No tokio, no runtime (decision 8.0.1). The mesh will eventually subsume
//! `--native-camera` (it becomes a thin mesh with one HumanAgent).

use crate::agent::{Agent, AgentHealth, MeshContext};
use crate::frame::LatestFrame;
use mvm_core::WorldState;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, RwLock};
use std::time::Instant;

pub struct AgentMesh {
    agents: Vec<Box<dyn Agent>>,
    world: Arc<RwLock<WorldState>>,
    frames: LatestFrame,
    stop: Arc<AtomicBool>,
}

impl AgentMesh {
    pub fn new(now: Instant) -> Self {
        Self {
            agents: Vec::new(),
            world: Arc::new(RwLock::new(WorldState::new(now))),
            frames: LatestFrame::new(),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn add(&mut self, agent: Box<dyn Agent>) -> &mut Self {
        self.agents.push(agent);
        self
    }

    /// Shared blackboard handle — clone it BEFORE `run()` (which consumes `self`)
    /// to read the fused world after the mesh stops.
    pub fn world(&self) -> Arc<RwLock<WorldState>> {
        self.world.clone()
    }

    /// The stop flag — set it (or let the capture agent set it) to wind the mesh down.
    pub fn stop_flag(&self) -> Arc<AtomicBool> {
        self.stop.clone()
    }

    /// Spawn a thread per agent and block until all return (stop flag set), then
    /// collect each agent's health. Consumes `self`.
    pub fn run(self) -> Vec<AgentHealth> {
        let handles: Vec<_> = self
            .agents
            .into_iter()
            .map(|mut agent| {
                let ctx = MeshContext {
                    frames: self.frames.clone(),
                    world: self.world.clone(),
                    stop: self.stop.clone(),
                };
                std::thread::spawn(move || agent.run(&ctx))
            })
            .collect();
        handles.into_iter().filter_map(|h| h.join().ok()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stubs::{FrameCounterAgent, StubCaptureAgent};
    use std::time::Duration;

    #[test]
    fn capture_feeds_consumer_and_world_advances() {
        let mut mesh = AgentMesh::new(Instant::now());
        // Fast synthetic capture (1000fps → 1ms) of 10 frames, then auto-stop.
        mesh.add(Box::new(StubCaptureAgent::new(1000.0, 10)));
        mesh.add(Box::new(FrameCounterAgent::new()));
        let world = mesh.world();

        let health = mesh.run(); // returns when capture exhausts budget + sets stop

        // Capture produced exactly its budget.
        let cap = health.iter().find(|h| h.name == "stub-capture").unwrap();
        assert_eq!(cap.ticks, 10);
        // Consumer saw at least one frame and advanced the blackboard.
        let cons = health.iter().find(|h| h.name == "frame-counter").unwrap();
        assert!(
            cons.ticks >= 1,
            "consumer should process ≥1 frame, got {}",
            cons.ticks
        );
        assert!(
            world.read().unwrap().frame_id >= 1,
            "world.frame_id should advance"
        );
    }

    #[test]
    fn mesh_shuts_down_cleanly_when_stop_set_externally() {
        let mut mesh = AgentMesh::new(Instant::now());
        mesh.add(Box::new(FrameCounterAgent::with_poll(
            Duration::from_millis(1),
        )));
        let stop = mesh.stop_flag();
        // No capture agent → consumer would loop forever; stop it from outside.
        let stopper = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        });
        let health = mesh.run(); // must return once stop is set
        stopper.join().unwrap();
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].name, "frame-counter");
    }
}
