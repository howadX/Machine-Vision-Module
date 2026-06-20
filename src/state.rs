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
    /// Production: uses HOME/.hermes/profiles/sight/workspace/current_state.md
    pub fn open() -> Result<Self> {
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let path = PathBuf::from(home).join(STATE_PATH);
        Self::open_at(path)
    }

    /// Test constructor: writes to an arbitrary path.
    pub fn open_at(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let tz = FixedOffset::east_opt(3 * 3600).unwrap();
        Ok(Self { path, tz })
    }

    /// Write current state to file.
    pub fn write(&self, tracker: &PersonTracker, fps: f64) -> Result<()> {
        let now = Local::now().with_timezone(&self.tz);
        let timestamp = now.format("%Y-%m-%dT%H:%M%:z").to_string();

        let mut active_tracks: Vec<_> = tracker.active().collect();
        active_tracks.sort_by_key(|t| t.id);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracker::{BBox, PersonTracker};

    fn temp_state_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("sightd_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn read_state(path: &PathBuf) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    #[test]
    fn state_empty_room() {
        let path = temp_state_path("test_empty.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 30.0).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Status: empty"));
        assert!(content.contains("- Face count: 0"));
        assert!(content.contains("- Persons visible: none"));
        assert!(content.contains("- FPS: 30.0"));
        // Should NOT contain person entries
        assert!(!content.contains("Person_1:"));
    }

    #[test]
    fn state_occupied_one_person() {
        let path = temp_state_path("test_occupied.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.tick(&[BBox { x1: 0.2, y1: 0.3, x2: 0.5, y2: 0.7 }]);

        sf.write(&tracker, 25.5).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Status: occupied"));
        assert!(content.contains("- Face count: 1"));
        assert!(content.contains("- Persons visible: track_0"));
        assert!(content.contains("- Person_1:"));
        assert!(content.contains("  - id: 0"));
        assert!(content.contains("  - identity: track_0"));
        assert!(content.contains("  - bbox: 0.20,0.30,0.50,0.70"));
        assert!(content.contains("  - age: 1 frames"));
        assert!(content.contains("  - emotion: none"));
        assert!(content.contains("  - posture: none"));
        assert!(content.contains("- FPS: 25.5"));
        // Should have last_checked timestamp
        assert!(content.contains("Last checked:"));
    }

    #[test]
    fn state_multiple_people() {
        let path = temp_state_path("test_multi.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.tick(&[
            BBox { x1: 0.1, y1: 0.1, x2: 0.3, y2: 0.5 },
            BBox { x1: 0.6, y1: 0.2, x2: 0.9, y2: 0.7 },
        ]);

        sf.write(&tracker, 29.9).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Face count: 2"));
        assert!(content.contains("- Persons visible: track_0, track_1"));
        assert!(content.contains("- Person_1:"));
        assert!(content.contains("  - id: 0"));
        assert!(content.contains("- Person_2:"));
        assert!(content.contains("  - id: 1"));
    }

    #[test]
    fn state_timestamp_is_iso8601() {
        let path = temp_state_path("test_ts.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 0.0).unwrap();

        let content = read_state(&path);
        // Format: 2026-06-20T15:04+03:00
        let ts_line = content
            .lines()
            .find(|l| l.starts_with("- Last checked:"))
            .unwrap();
        let ts = ts_line.strip_prefix("- Last checked: ").unwrap();
        // Basic sanity: contains T, ends with timezone offset
        assert!(ts.contains('T'));
        assert!(ts.ends_with("+03:00") || ts.ends_with("+0300"));
    }

    #[test]
    fn state_file_creates_parent_dirs() {
        let path = std::env::temp_dir()
            .join("sightd_test_nested")
            .join("a")
            .join("b")
            .join("test_nested.md");
        // Parent dirs don't exist yet
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        // write() actually creates the file
        sf.write(&tracker, 0.0).unwrap();
        assert!(path.exists());
        // Cleanup
        let _ = std::fs::remove_dir_all(path.parent().unwrap().parent().unwrap().parent().unwrap());
    }

    #[test]
    fn state_header_present() {
        let path = temp_state_path("test_header.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 15.0).unwrap();

        let content = read_state(&path);
        assert!(content.starts_with("# Camera State"));
        assert!(content.contains("- Camera: /dev/video0"));
        assert!(content.contains("- Uptime: —"));
    }
}
