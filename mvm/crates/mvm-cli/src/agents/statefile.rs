//! StateFileWriter (phase 8.C.7) — the mesh sink. Periodically snapshots the
//! `WorldState` and serialises it to the Hermes StateFile (+ optional Scene JSON
//! line), reusing the same contract as the synchronous path. It owns no models;
//! it reads the blackboard the agents fill and computes FPS from the frame_id
//! advance between writes.

use crate::state::{SceneLog, StateFile};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::time::{Duration, Instant};

pub(crate) struct StateFileWriter {
    state_file: StateFile,
    dump_json: bool,
    interval: Duration,
    /// Stop after this many writes (tests / bounded runs). `None` = until stopped.
    max_writes: Option<u64>,
    /// Optional durable JSONL frame history (phase 12.C0, `--scene-log`).
    scene_log: Option<SceneLog>,
    /// Optional durable completed-episode log (phase 12.C1) — same `--scene-log`
    /// dir, `events-DATE.jsonl`. One line per drained `WorldState::events` entry.
    events_log: Option<SceneLog>,
    /// (frame_id, instant) of the previous write — for FPS.
    last: Option<(u64, Instant)>,
}

impl StateFileWriter {
    pub(crate) fn new(
        state_file: StateFile,
        dump_json: bool,
        interval: Duration,
        max_writes: Option<u64>,
        scene_log: Option<SceneLog>,
        events_log: Option<SceneLog>,
    ) -> Self {
        Self { state_file, dump_json, interval, max_writes, scene_log, events_log, last: None }
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
            // Build the JSON line once when either sink wants it (stdout dump
            // and/or the durable scene log, phase 12.C0).
            if self.dump_json || self.scene_log.is_some() {
                let line = self.state_file.to_json_line_world(&snapshot, fps);
                if self.dump_json {
                    println!("{line}");
                }
                if let Some(log) = self.scene_log.as_mut() {
                    if let Err(e) = log.append(&line) {
                        eprintln!("  scene-log: append error: {e}");
                    }
                }
            }
            // Drain completed episodes to the events log (phase 12.C1). The
            // snapshot holds this write's events (cleared from the shared world
            // above), so each episode is logged exactly once.
            if let Some(elog) = self.events_log.as_mut() {
                if !snapshot.events.is_empty() {
                    for line in self.state_file.events_log_lines(&snapshot.events) {
                        if let Err(e) = elog.append(&line) {
                            eprintln!("  events-log: append error: {e}");
                        }
                    }
                }
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
        let mut agent = StateFileWriter::new(sf, false, Duration::from_millis(1), Some(1), None, None);
        let health = agent.run(&ctx);

        assert_eq!(health.ticks, 1, "exactly one write");
        let content = std::fs::read_to_string(&path).unwrap();
        // Snapshot v2 (phase 12.B3): markdown table, "· occupied ·" in the header.
        assert!(content.starts_with("# Scene "), "v2 header:\n{content}");
        assert!(content.contains("· occupied ·"), "got:\n{content}");
        assert!(content.contains("sitting"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn writer_appends_to_scene_log() {
        // Phase 12.C0: a StateFileWriter with a scene log appends one valid JSON
        // line per write to the day's rotating file (independent of --dump-json).
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 3;
        world.people.push(PersonState {
            track_id: 0,
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.6, y2: 0.8 },
            identity: Some(("Вадим".into(), 0.9)),
            emotion: None,
            posture: None,
            position_3d: Some(Position3D { x: 0.1, y: 0.2, z: 1.0, confidence: 0.6 }),
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
            head_yaw: None,
            head_facing: None,
            last_seen: now,
        });
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(world)),
            stop: Arc::new(AtomicBool::new(false)),
        };
        let sf_path = std::env::temp_dir().join(format!("mvm_sflog_sf_{}.md", std::process::id()));
        let log_dir = std::env::temp_dir().join(format!("mvm_sflog_dir_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&log_dir);
        let sf = StateFile::open_at(sf_path.clone()).unwrap();
        let log = sf.open_scene_log(log_dir.clone());
        // dump_json = false → proves the log is fed independently of stdout.
        let mut agent = StateFileWriter::new(sf, false, Duration::from_millis(1), Some(1), Some(log), None);
        agent.run(&ctx);

        let files: Vec<_> = std::fs::read_dir(&log_dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1, "one day's scene-*.jsonl written");
        let fname = files[0].file_name().into_string().unwrap();
        assert!(fname.starts_with("scene-") && fname.ends_with(".jsonl"), "got {fname}");
        let content = std::fs::read_to_string(files[0].path()).unwrap();
        assert_eq!(content.lines().count(), 1, "one line per write");
        let v: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(v["faces"][0]["identity"], "Вадим");
        let _ = std::fs::remove_file(&sf_path);
        let _ = std::fs::remove_dir_all(&log_dir);
    }

    #[test]
    fn writer_drains_events_to_events_log() {
        // Phase 12.C1: a completed episode in world.events is written as one line
        // to events-DATE.jsonl (stamped ts + the episode fields), exactly once.
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 9;
        world.events.push(mvm_core::ActivityEvent {
            label: "typing".into(),
            participants: vec![0, 10001],
            started_at: 1.0,
            ended_at: 13.5,
            duration_s: 12.5,
            peak_confidence: 0.9,
            stable_participants: true,
        });
        let ctx = MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(world)),
            stop: Arc::new(AtomicBool::new(false)),
        };
        let sf_path = std::env::temp_dir().join(format!("mvm_evlog_sf_{}.md", std::process::id()));
        let log_dir = std::env::temp_dir().join(format!("mvm_evlog_dir_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&log_dir);
        let sf = StateFile::open_at(sf_path.clone()).unwrap();
        let elog = sf.open_events_log(log_dir.clone());
        let mut agent = StateFileWriter::new(sf, false, Duration::from_millis(1), Some(1), None, Some(elog));
        agent.run(&ctx);

        let files: Vec<_> = std::fs::read_dir(&log_dir).unwrap().filter_map(|e| e.ok()).collect();
        assert_eq!(files.len(), 1, "one day's events-*.jsonl");
        let fname = files[0].file_name().into_string().unwrap();
        assert!(fname.starts_with("events-") && fname.ends_with(".jsonl"), "got {fname}");
        let content = std::fs::read_to_string(files[0].path()).unwrap();
        assert_eq!(content.lines().count(), 1, "one line per episode");
        let v: serde_json::Value = serde_json::from_str(content.lines().next().unwrap()).unwrap();
        assert_eq!(v["label"], "typing");
        assert!((v["duration_s"].as_f64().unwrap() - 12.5).abs() < 1e-9);
        assert_eq!(v["stable_participants"], true);
        assert!(v["ts"].is_string(), "events log line carries a ts");
        let _ = std::fs::remove_file(&sf_path);
        let _ = std::fs::remove_dir_all(&log_dir);
    }
}
