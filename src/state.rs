//! State file writer. Produces key: value format compatible with
//! Hermes Sight agent and sight-journal skill.

use crate::tracker::PersonTracker;
use anyhow::Result;
use chrono::{Local, FixedOffset};
use std::fs;
use std::path::PathBuf;

/// Path to the Sight state file.
const STATE_PATH: &str = ".hermes/profiles/sight/workspace/current_state.md";

pub struct StateFile {
    path: PathBuf,
    tz: FixedOffset,
}

impl StateFile {
    pub fn open() -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let path = PathBuf::from(home).join(STATE_PATH);

        // Ensure directory exists
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        // MSK timezone offset
        let tz = FixedOffset::east_opt(3 * 3600).unwrap();

        Ok(Self { path, tz })
    }

    /// Write current state to file.
    pub fn write(&self, tracker: &PersonTracker, fps: f64) -> Result<()> {
        let now = Local::now().with_timezone(&self.tz);
        let timestamp = now.format("%Y-%m-%dT%H:%M%:z").to_string();

        let active_tracks: Vec<_> = tracker.active().collect();
        let face_count = active_tracks.len();

        let status = if face_count > 0 { "occupied" } else { "empty" };

        // Build key: value output
        let mut lines = vec![
            "# Camera State (sightd v0.1)".to_string(),
            format!("- Last checked: {timestamp}"),
            format!("- Camera: /dev/video0"),
            format!("- Uptime: —"),
            format!("- FPS: {fps:.1}"),
            format!("- Face count: {face_count}"),
        ];

        if face_count > 0 {
            let names: Vec<String> = active_tracks
                .iter()
                .map(|t| format!("track_{}", t.id))
                .collect();
            lines.push(format!("- Persons visible: {}", names.join(", ")));

            for (i, t) in active_tracks.iter().enumerate() {
                lines.push(format!("- Person_{}:", i + 1));
                lines.push(format!("  - id: {}", t.id));
                lines.push(format!("  - identity: track_{}", t.id));
                lines.push(format!("  - bbox: {:.2},{:.2},{:.2},{:.2}",
                    t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2));
                lines.push(format!("  - age: {} frames", t.age));
                lines.push(format!("  - emotion: none"));
                lines.push(format!("  - posture: none"));
            }
        } else {
            lines.push("- Persons visible: none".to_string());
        }

        lines.push(format!("- Status: {status}"));
        lines.push("".to_string());

        fs::write(&self.path, lines.join("\n"))?;

        eprintln!("  state written: {face_count} faces, {:.1} fps", fps);
        Ok(())
    }
}
