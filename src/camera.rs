//! Camera capture via Python bridge subprocess.
//!
//! Architecture: Python `tools/sight_bridge.py` runs face detection (SCRFD GPU
//! or YuNet CPU) and writes JSON lines to stdout. Rust sightd spawns it as a
//! subprocess, reads and parses the JSON, and feeds bboxes to the tracker.
//!
//! v0.3: enriched detections (identity, emotion, posture).

use crate::tracker::{BBox, Detection};
use anyhow::{Context, Result};
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::process::{Child, Command, Stdio};

/// Raw detection from Python bridge JSON (optional fields may be absent).
#[derive(Debug, Deserialize)]
struct RawDetection {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
    #[serde(default)]
    identity: Option<String>,
    #[serde(default)]
    identity_confidence: Option<f32>,
    #[serde(default)]
    emotion: Option<String>,
    #[serde(default)]
    emotion_confidence: Option<f32>,
    #[serde(default)]
    posture: Option<String>,
    #[serde(default)]
    posture_confidence: Option<f32>,
    // 3D Workspace fields (optional, only when body matched)
    #[serde(default)]
    distance: Option<f32>,
    #[serde(default)]
    position_x: Option<f32>,
    #[serde(default)]
    position_y: Option<f32>,
    #[serde(default)]
    position_z: Option<f32>,
    #[serde(default)]
    ws_confidence: Option<f32>,
    // Posture quality fields
    #[serde(default)]
    recon_error: Option<f32>,
    #[serde(default)]
    plausibility: Option<String>,
}

/// Top-level JSON message from bridge.
#[derive(Debug, Deserialize)]
struct BridgeMsg {
    #[allow(dead_code)]
    frame: u64,
    #[allow(dead_code)]
    elapsed: f64,
    detections: Vec<RawDetection>,
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
    pub fn open(cpu: bool) -> Result<Self> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .unwrap_or_else(|_| ".".into());
        let bridge_path = std::path::Path::new(&manifest_dir)
            .join("tools")
            .join("sight_bridge.py");

        let venv_python = std::path::Path::new(&manifest_dir)
            .join("venv")
            .join("bin")
            .join("python3");

        let python = if venv_python.exists() {
            venv_python
        } else {
            std::path::PathBuf::from("python3")
        };

        let mut cmd = Command::new(&python);
        cmd.arg(&bridge_path)
            .arg("--json-only")
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

        Ok(Self { child, reader, width: 640, height: 480 })
    }

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }

    /// Parse a single JSON line from the bridge into Detections.
    pub fn parse_line(line: &str) -> Result<Vec<Detection>> {
        let line = line.trim();
        if line.is_empty() {
            return Ok(vec![]);
        }

        let msg: BridgeMsg = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse bridge JSON: {line}"))?;

        let detections: Vec<Detection> = msg
            .detections
            .into_iter()
            .map(|d| Detection {
                bbox: BBox {
                    x1: d.x1.clamp(0.0, 1.0),
                    y1: d.y1.clamp(0.0, 1.0),
                    x2: d.x2.clamp(0.0, 1.0),
                    y2: d.y2.clamp(0.0, 1.0),
                },
                identity: d.identity,
                identity_confidence: d.identity_confidence,
                emotion: d.emotion,
                emotion_confidence: d.emotion_confidence,
                posture: d.posture,
                posture_confidence: d.posture_confidence,
                distance: d.distance,
                position_x: d.position_x,
                position_y: d.position_y,
                position_z: d.position_z,
                ws_confidence: d.ws_confidence,
                recon_error: d.recon_error,
                plausibility: d.plausibility,
            })
            .collect();

        Ok(detections)
    }

    /// Read one JSON message from the bridge, return detections.
    pub fn read(&mut self) -> Result<Vec<Detection>> {
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

    #[test]
    fn parse_bare_detection() {
        let dets = Camera::parse_line(
            r#"{"frame":1,"elapsed":0.0,"detections":[{"x1":0.2,"y1":0.3,"x2":0.5,"y2":0.7}]}"#,
        ).unwrap();
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].identity, None);
        assert_eq!(dets[0].emotion, None);
        assert_eq!(dets[0].posture, None);
    }

    #[test]
    fn parse_enriched_detection() {
        let json = r#"{"frame":42,"elapsed":1.5,"detections":[{
            "x1":0.2,"y1":0.3,"x2":0.5,"y2":0.7,
            "identity":"Вадим","identity_confidence":0.72,
            "emotion":"Нейтральная","emotion_confidence":0.65,
            "posture":"сидит","posture_confidence":0.95
        }]}"#;
        let dets = Camera::parse_line(json).unwrap();
        assert_eq!(dets.len(), 1);
        let d = &dets[0];
        assert_eq!(d.identity.as_deref(), Some("Вадим"));
        assert!((d.identity_confidence.unwrap() - 0.72).abs() < 0.001);
        assert_eq!(d.emotion.as_deref(), Some("Нейтральная"));
        assert_eq!(d.posture.as_deref(), Some("сидит"));
        assert!((d.posture_confidence.unwrap() - 0.95).abs() < 0.001);
    }

    #[test]
    fn parse_partial_enrichment() {
        // Only identity, no emotion/posture
        let json = r#"{"frame":1,"elapsed":0.0,"detections":[{
            "x1":0.2,"y1":0.3,"x2":0.5,"y2":0.7,
            "identity":"Вадим","identity_confidence":0.5
        }]}"#;
        let dets = Camera::parse_line(json).unwrap();
        assert_eq!(dets[0].identity.as_deref(), Some("Вадим"));
        assert_eq!(dets[0].emotion, None);
        assert_eq!(dets[0].posture, None);
    }

    #[test]
    fn parse_empty_detections() {
        let dets = Camera::parse_line(
            r#"{"frame":1,"elapsed":0.033,"detections":[]}"#,
        ).unwrap();
        assert!(dets.is_empty());
    }

    #[test]
    fn parse_clamps_out_of_range() {
        let dets = Camera::parse_line(
            r#"{"frame":1,"elapsed":0.0,"detections":[{"x1":-0.5,"y1":-0.2,"x2":1.5,"y2":2.0}]}"#,
        ).unwrap();
        assert_eq!(dets[0].bbox.x1, 0.0);
        assert_eq!(dets[0].bbox.x2, 1.0);
    }

    #[test]
    fn parse_multiple_detections() {
        let dets = Camera::parse_line(
            r#"{"frame":10,"elapsed":0.35,"detections":[
                {"x1":0.1,"y1":0.1,"x2":0.3,"y2":0.5},
                {"x1":0.6,"y1":0.2,"x2":0.9,"y2":0.7}
            ]}"#,
        ).unwrap();
        assert_eq!(dets.len(), 2);
    }

    // ── Integration tests ────────────────────────────────

    use crate::state::StateFile;
    use crate::tracker::PersonTracker;

    fn temp_state(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("sightd_int_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn feed_pipeline(file: &str, frames: &[&str]) -> (PersonTracker, String) {
        let mut tracker = PersonTracker::new();
        for line in frames {
            let detections = Camera::parse_line(line).unwrap();
            tracker.tick(&detections);
        }
        let path = temp_state(file);
        let sf = StateFile::open_at(path.clone()).unwrap();
        sf.write(&tracker, 30.0).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        (tracker, content)
    }

    #[test]
    fn integration_enriched_pipeline() {
        let frames = &[
            r#"{"frame":1,"elapsed":0.0,"detections":[{
                "x1":0.3,"y1":0.3,"x2":0.5,"y2":0.5,
                "identity":"Вадим","identity_confidence":0.72,
                "emotion":"Нейтральная","emotion_confidence":0.65,
                "posture":"сидит","posture_confidence":0.95
            }]}"#,
            r#"{"frame":2,"elapsed":0.033,"detections":[{
                "x1":0.32,"y1":0.31,"x2":0.52,"y2":0.51,
                "identity":"Вадим","identity_confidence":0.70,
                "emotion":"Радость","emotion_confidence":0.55
            }]}"#,
        ];

        let (_tracker, content) = feed_pipeline("int_rich.md", frames);
        // Final state: track matched both frames, last emotion wins
        assert!(content.contains("identity: Вадим"));
        assert!(content.contains("emotion: Радость (0.55)"));
        assert!(content.contains("posture: сидит (0.95)"));
        // Track persists across frames (matched, not duplicated)
        assert!(content.contains("age: 2 frames"));
        // Confidence present for identity
        assert!(content.contains("identity: Вадим (0.72)"));
    }
}
