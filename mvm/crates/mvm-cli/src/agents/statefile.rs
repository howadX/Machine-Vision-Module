//! StateFileWriter (phase 8.C.7) — the mesh sink. Periodically snapshots the
//! `WorldState` and serialises it to the Hermes StateFile (+ optional Scene JSON
//! line), reusing the same contract as the synchronous path. It owns no models;
//! it reads the blackboard the agents fill and computes FPS from the frame_id
//! advance between writes.

use crate::state::StateFile;
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::time::{Duration, Instant};

pub(crate) struct StateFileWriter {
    state_file: StateFile,
    dump_json: bool,
    interval: Duration,
    /// Stop after this many writes (tests / bounded runs). `None` = until stopped.
    max_writes: Option<u64>,
    /// (frame_id, instant) of the previous write — for FPS.
    last: Option<(u64, Instant)>,
}

impl StateFileWriter {
    pub(crate) fn new(
        state_file: StateFile,
        dump_json: bool,
        interval: Duration,
        max_writes: Option<u64>,
    ) -> Self {
        Self { state_file, dump_json, interval, max_writes, last: None }
    }
}

impl Agent for StateFileWriter {
    fn name(&self) -> String {
        "statefile".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut writes = 0u64;
        let mut errors = 0u64;
        while ctx.running() {
            // Snapshot under the lock, then release before doing IO. A write
            // lock because completed activity events (phase 11.C3) are consumed
            // exclusively — each event belongs to exactly one written frame.
            let snapshot = match ctx.world.write() {
                Ok(mut w) => {
                    let s = w.clone();
                    w.events.clear();
                    s
                }
                Err(_) => {
                    errors += 1;
                    break;
                }
            };

            let now = Instant::now();
            let fps = fps_between(self.last, snapshot.frame_id, now);
            self.last = Some((snapshot.frame_id, now));

            if let Err(e) = self.state_file.write_world(&snapshot, fps) {
                eprintln!("  statefile: write error: {e}");
                errors += 1;
            }
            if self.dump_json {
                println!("{}", self.state_file.to_json_line_world(&snapshot, fps));
            }
            writes += 1;

            if let Some(max) = self.max_writes {
                if writes >= max {
                    break;
                }
            }
            std::thread::sleep(self.interval);
        }
        AgentHealth { name: self.name(), ticks: writes, errors }
    }
}

/// FPS from the frame_id advance since the previous write. `None` (first write)
/// or a non-positive interval → 0.0.
fn fps_between(prev: Option<(u64, Instant)>, frame_id: u64, now: Instant) -> f64 {
    match prev {
        Some((pf, pt)) => {
            let dt = now.duration_since(pt).as_secs_f64();
            if dt > 0.0 {
                frame_id.saturating_sub(pf) as f64 / dt
            } else {
                0.0
            }
        }
        None => 0.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{BBox, PersonState, Position3D, WorldState};
    use mvm_mesh::LatestFrame;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, RwLock};

    #[test]
    fn fps_from_frame_advance() {
        let now = Instant::now();
        let prev = Some((0u64, now - Duration::from_secs(1)));
        let fps = fps_between(prev, 30, now);
        assert!((fps - 30.0).abs() < 1.0, "≈30 fps over 1s / 30 frames, got {fps}");
    }

    #[test]
    fn first_write_has_zero_fps() {
        assert_eq!(fps_between(None, 5, Instant::now()), 0.0);
    }

    #[test]
    fn writes_statefile_from_world() {
        // A world with one person → bounded run writes the StateFile once.
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 7;
        world.people.push(PersonState {
            track_id: 10,
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 },
            identity: None,
            emotion: Some(("Neutral".into(), 0.4)),
            posture: Some(("sitting".into(), 0.9)),
            position_3d: Some(Position3D { x: 0.1, y: 0.2, z: 1.5, confidence: 0.6 }),
            emotion_distribution: None,
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: None,
            torso_attitude: None,
            motion_state: None,
            velocity: None,
            left_gesture: None,
            right_gesture: None,
            recon_error: None,
            plausibility: None,
            head_yaw: None, head_facing: None,
            last_seen: now,
        });
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(world)),
            stop: Arc::new(AtomicBool::new(false)),
        };

        let path = std::env::temp_dir()
            .join(format!("mvm_sfw_{}.md", std::process::id()));
        let sf = StateFile::open_at(path.clone()).unwrap();
        let mut agent = StateFileWriter::new(sf, false, Duration::from_millis(1), Some(1));
        let health = agent.run(&ctx);

        assert_eq!(health.ticks, 1, "exactly one write");
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("- Status: occupied"), "got:\n{content}");
        assert!(content.contains("sitting"));
        let _ = std::fs::remove_file(&path);
    }
}
