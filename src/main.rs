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

fn parse_args() -> (bool, u64, bool) {
    let args: Vec<String> = std::env::args().collect();
    let mut use_cpu = false;
    let mut dump_interval = 180u64;
    let mut once = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--cpu" => use_cpu = true,
            "--once" => once = true,
            "--dump-interval" => {
                if i + 1 < args.len() {
                    dump_interval = args[i + 1].parse().unwrap_or(180);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    (use_cpu, dump_interval, once)
}

fn main() -> Result<()> {
    let (use_cpu, dump_secs, once) = parse_args();

    eprintln!("sightd v0.2.0 — starting (dump every {dump_secs}s, {}mode)",
        if once { "once, " } else { "" });

    let mut cam = camera::Camera::open(use_cpu)?;
    let mut tracker = tracker::PersonTracker::new();
    let state_file = state::StateFile::open()?;
    let mut last_dump = Instant::now();
    let dump_interval = Duration::from_secs(dump_secs);

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

            if once {
                eprintln!("  once: done after {frame_count} frames");
                break;
            }
        }

        std::thread::sleep(Duration::from_millis(10));
    }

    Ok(())
}
