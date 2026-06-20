//! Camera capture via Python bridge subprocess.
//!
//! Architecture: Python `tools/sight_bridge.py` runs face detection (SCRFD GPU
//! or YuNet CPU) and writes JSON lines to stdout. Rust sightd spawns it as a
//! subprocess, reads and parses the JSON, and feeds bboxes to the tracker.
//!
//! v0.2: real camera via subprocess bridge.

use crate::tracker::BBox;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

/// A single detection from the Python bridge.
#[derive(Debug, Deserialize)]
struct Detection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

/// Top-level JSON message from bridge.
#[derive(Debug, Deserialize)]
struct BridgeMsg {
    #[allow(dead_code)]
    frame: u64,
    #[allow(dead_code)]
    elapsed: f64,
    detections: Vec<Detection>,
}

/// Manages the Python bridge subprocess.
#[allow(dead_code)]
pub struct Camera {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
    width: u32,
    height: u32,
}

impl Camera {
    /// Spawn the Python bridge subprocess.
    ///
    /// Expects `tools/sight_bridge.py` to exist at the project root.
    /// Pass `--cpu` to use OpenCV YuNet instead of SCRFD GPU.
    pub fn open(cpu: bool) -> Result<Self> {
        // Locate the bridge script
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| ".".into());
        let bridge_path = std::path::Path::new(&manifest_dir)
            .join("tools")
            .join("sight_bridge.py");

        // Find venv python
        let venv_python = std::path::Path::new(&manifest_dir)
            .join("venv")
            .join("bin")
            .join("python3");

        let python = if venv_python.exists() {
            venv_python
        } else {
            // Fallback: system python3
            std::path::PathBuf::from("python3")
        };

        let mut cmd = Command::new(&python);
        cmd.arg(&bridge_path)
            .arg("--json-only") // suppress stderr
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        if cpu {
            cmd.arg("--cpu");
        }

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn bridge: {} {}", python.display(), bridge_path.display()))?;

        let stdout = child.stdout.take()
            .context("Failed to capture bridge stdout")?;

        let reader = BufReader::new(stdout);

        eprintln!("  camera: bridge started (pid {})", child.id());

        Ok(Self {
            child,
            reader,
            width: 640,
            height: 480,
        })
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Parse a single JSON line from the bridge into BBoxes.
    ///
    /// Public for testing. Returns empty vec for blank lines.
    pub fn parse_line(line: &str) -> Result<Vec<BBox>> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(vec![]);
        }

        let msg: BridgeMsg = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse bridge JSON: {line}"))?;

        let bboxes: Vec<BBox> = msg
            .detections
            .into_iter()
            .map(|d| BBox {
                x1: d.x1.clamp(0.0, 1.0),
                y1: d.y1.clamp(0.0, 1.0),
                x2: d.x2.clamp(0.0, 1.0),
                y2: d.y2.clamp(0.0, 1.0),
            })
            .collect();

        Ok(bboxes)
    }

    /// Read one JSON message from the bridge, return bboxes.
    ///
    /// Blocks until a valid line is received. Returns error on EOF
    /// (bridge died).
    pub fn read(&mut self) -> Result<Vec<BBox>> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => {
                return Err(anyhow::anyhow!("Bridge subprocess exited unexpectedly"));
            }
            Ok(_) => {}
            Err(e) => {
                return Err(anyhow::anyhow!("Failed to read from bridge: {}", e));
            }
        }

        Self::parse_line(&line)
    }
}

impl Drop for Camera {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        eprintln!("  camera: bridge shut down");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── JSON parsing ─────────────────────────────────────

    #[test]
    fn parse_empty_detections() {
        let bboxes = Camera::parse_line(
            r#"{"frame":1,"elapsed":0.033,"detections":[]}"#,
        )
        .unwrap();
        assert!(bboxes.is_empty());
    }

    #[test]
    fn parse_single_detection() {
        let bboxes = Camera::parse_line(
            r#"{"frame":42,"elapsed":1.5,"detections":[{"x1":0.2,"y1":0.3,"x2":0.5,"y2":0.7}]}"#,
        )
        .unwrap();
        assert_eq!(bboxes.len(), 1);
        assert!((bboxes[0].x1 - 0.2).abs() < 1e-6);
        assert!((bboxes[0].y2 - 0.7).abs() < 1e-6);
    }

    #[test]
    fn parse_multiple_detections() {
        let bboxes = Camera::parse_line(
            r#"{"frame":10,"elapsed":0.35,"detections":[
                {"x1":0.1,"y1":0.1,"x2":0.3,"y2":0.5},
                {"x1":0.6,"y1":0.2,"x2":0.9,"y2":0.7}
            ]}"#,
        )
        .unwrap();
        assert_eq!(bboxes.len(), 2);
    }

    #[test]
    fn parse_clamps_out_of_range() {
        let bboxes = Camera::parse_line(
            r#"{"frame":1,"elapsed":0.0,"detections":[{"x1":-0.5,"y1":-0.2,"x2":1.5,"y2":2.0}]}"#,
        )
        .unwrap();
        assert_eq!(bboxes.len(), 1);
        assert_eq!(bboxes[0].x1, 0.0);
        assert_eq!(bboxes[0].y1, 0.0);
        assert_eq!(bboxes[0].x2, 1.0);
        assert_eq!(bboxes[0].y2, 1.0);
    }

    #[test]
    fn parse_blank_line() {
        let bboxes = Camera::parse_line("").unwrap();
        assert!(bboxes.is_empty());

        let bboxes = Camera::parse_line("   \n  ").unwrap();
        assert!(bboxes.is_empty());
    }

    #[test]
    fn parse_invalid_json_is_error() {
        let result = Camera::parse_line("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn parse_missing_detections_field() {
        let result = Camera::parse_line(r#"{"frame":1,"elapsed":0.0}"#);
        assert!(result.is_err());
    }

    // ── Integration: parse → tracker → state ─────────────

    use crate::state::StateFile;
    use crate::tracker::PersonTracker;

    fn temp_state(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("sightd_integration_test");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    /// Feed a stream of JSON frames through the full pipeline.
    fn feed_pipeline(frames: &[&str]) -> (PersonTracker, String) {
        let mut tracker = PersonTracker::new();
        for line in frames {
            let detections = Camera::parse_line(line).unwrap();
            tracker.tick(&detections);
        }

        let path = temp_state("integration_test.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        sf.write(&tracker, 30.0).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        (tracker, content)
    }

    #[test]
    fn integration_empty_stream() {
        let (_tracker, content) = feed_pipeline(&[]);
        assert!(content.contains("- Status: empty"));
        assert!(content.contains("Face count: 0"));
    }

    #[test]
    fn integration_one_person_moving() {
        let frames = &[
            r#"{"frame":1,"elapsed":0.0,"detections":[{"x1":0.3,"y1":0.3,"x2":0.5,"y2":0.5}]}"#,
            r#"{"frame":2,"elapsed":0.033,"detections":[{"x1":0.32,"y1":0.31,"x2":0.52,"y2":0.51}]}"#,
            r#"{"frame":3,"elapsed":0.066,"detections":[{"x1":0.34,"y1":0.32,"x2":0.54,"y2":0.52}]}"#,
        ];

        let (tracker, content) = feed_pipeline(frames);

        // Single track maintained across all 3 frames
        assert_eq!(tracker.track_count(), 1);
        let active: Vec<_> = tracker.active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].age, 3);

        // State reflects it
        assert!(content.contains("Status: occupied"));
        assert!(content.contains("Face count: 1"));
        assert!(content.contains("age: 3 frames"));
    }

    #[test]
    fn integration_person_appears_and_leaves() {
        let frames = &[
            r#"{"frame":1,"elapsed":0.0,"detections":[{"x1":0.3,"y1":0.3,"x2":0.5,"y2":0.5}]}"#,
            r#"{"frame":2,"elapsed":0.033,"detections":[]}"#,
            r#"{"frame":3,"elapsed":0.066,"detections":[]}"#,
            r#"{"frame":4,"elapsed":0.1,"detections":[]}"#,
        ];

        let (tracker, _content) = feed_pipeline(frames);

        // Person appeared then left — still tracked (not expired yet)
        assert_eq!(tracker.track_count(), 1);
        let active: Vec<_> = tracker.active().collect();
        assert_eq!(active.len(), 0); // not active (missed > 0)
    }

    #[test]
    fn integration_two_people_crossing() {
        // Two people appear, then cross positions
        let frames = &[
            // Frame 1: person A left, person B right
            r#"{"frame":1,"elapsed":0.0,"detections":[
                {"x1":0.1,"y1":0.2,"x2":0.3,"y2":0.6},
                {"x1":0.7,"y1":0.2,"x2":0.9,"y2":0.6}
            ]}"#,
            // Frame 2: they cross — A moves right, B moves left
            r#"{"frame":2,"elapsed":0.033,"detections":[
                {"x1":0.6,"y1":0.2,"x2":0.8,"y2":0.6},
                {"x1":0.2,"y1":0.2,"x2":0.4,"y2":0.6}
            ]}"#,
            // Frame 3: both move further
            r#"{"frame":3,"elapsed":0.066,"detections":[
                {"x1":0.65,"y1":0.21,"x2":0.85,"y2":0.61},
                {"x1":0.15,"y1":0.21,"x2":0.35,"y2":0.61}
            ]}"#,
        ];

        let (tracker, _content) = feed_pipeline(frames);
        assert_eq!(tracker.track_count(), 2);
        // Both still tracked (no new IDs created)
    }
}
