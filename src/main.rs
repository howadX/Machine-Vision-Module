//! sightd — MVM daemon. Continuous person tracking and state observation.
//!
//! Architecture:
//!   Python bridge (stdin) → BBox detections → Tracker (IoU) → State file
//!
//! v0.1: tracker infrastructure working. Real camera detections via Python bridge in v0.2.

mod camera;
mod tracker;
mod state;

use anyhow::Result;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    eprintln!("sightd v0.1.0 — starting");

    let cam = camera::Camera::open()?;
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
            eprintln!("  tick {frame_count}: {} tracks, {fps:.1} fps", tracker.track_count());
            state_file.write(&tracker, fps)?;
            last_dump = Instant::now();
        }

        // ~30 fps with no real camera load
        std::thread::sleep(Duration::from_millis(33));
    }
}
