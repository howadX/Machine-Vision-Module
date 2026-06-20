//! sightd — MVM daemon. Continuous person tracking and state observation.
//!
//! Architecture:
//!   Python bridge (stdin) → BBox detections → Tracker (IoU) → State file
//!
//! v0.2: real camera via Python subprocess bridge.

mod camera;
mod tracker;
mod state;

use anyhow::Result;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    eprintln!("sightd v0.2.0 — starting");

    // --cpu flag uses OpenCV YuNet CPU detector instead of SCRFD GPU
    let use_cpu = std::env::args().any(|a| a == "--cpu");

    let mut cam = camera::Camera::open(use_cpu)?;
    let mut tracker = tracker::PersonTracker::new();
    let state_file = state::StateFile::open()?;
    let mut last_dump = Instant::now();
    let dump_interval = Duration::from_secs(180);

    let mut frame_count: u64 = 0;
    let start = Instant::now();

    loop {
        let detections = cam.read()?;
        frame_count += 1;

        tracker.tick(&detections);

        if last_dump.elapsed() >= dump_interval {
            let fps = frame_count as f64 / start.elapsed().as_secs_f64();
            eprintln!(
                "  tick {frame_count}: {} tracks, {fps:.1} fps",
                tracker.track_count()
            );
            state_file.write(&tracker, fps)?;
            last_dump = Instant::now();
        }

        // Small sleep to avoid busy-waiting; actual frame rate
        // is governed by the Python bridge's camera loop.
        std::thread::sleep(Duration::from_millis(10));
    }
}
