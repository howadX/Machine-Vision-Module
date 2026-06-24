//! Camera capture via Python bridge subprocess.
//!
//! Spawns `sight_bridge.py` from the old MVM repo and reads JSON detections.
//! Ported from sightd v0.3 `src/camera.rs`.

use anyhow::{Context, Result};
use mvm_core::Detection;
use serde::Deserialize;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

/// Raw detection from Python bridge JSON.
#[derive(Debug, Deserialize)]
struct RawDetection {
    x1: f32, y1: f32, x2: f32, y2: f32,
    #[serde(default)] identity: Option<String>,
    #[serde(default)] identity_confidence: Option<f32>,
    #[serde(default)] emotion: Option<String>,
    #[serde(default)] emotion_confidence: Option<f32>,
    #[serde(default)] posture: Option<String>,
    #[serde(default)] posture_confidence: Option<f32>,
    #[serde(default)] distance: Option<f32>,
    #[serde(default)] position_x: Option<f32>,
    #[serde(default)] position_y: Option<f32>,
    #[serde(default)] position_z: Option<f32>,
    #[serde(default)] ws_confidence: Option<f32>,
    #[serde(default)] recon_error: Option<f32>,
    #[serde(default)] plausibility: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BridgeMsg {
    #[allow(dead_code)]
    frame: u64,
    #[allow(dead_code)]
    elapsed: f64,
    detections: Vec<RawDetection>,
}

/// Camera configuration.
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Path to the old MVM repo root (contains venv/ and tools/mvm_bridge.py).
    pub mvm_repo: PathBuf,
    /// Python binary (default: mvm_repo/venv/bin/python3).
    pub python: Option<PathBuf>,
    /// Use CPU mode (--cpu flag).
    pub cpu: bool,
    /// Camera device (default: /dev/video0).
    pub device: String,
    /// Write frame JPEG to this path for native enrichment.
    pub frame_output: Option<PathBuf>,
}

impl Default for CameraConfig {
    fn default() -> Self {
        let home = std::env::var("HOME").unwrap_or_default();
        Self {
        mvm_repo: PathBuf::from(home).join("repos/Machine Vision Module"),
        python: None,
        cpu: false,
        device: "/dev/video0".into(),
        frame_output: None,
        }
    }
}

/// Manages the Python bridge subprocess.
pub struct Camera {
    child: Child,
    reader: BufReader<std::process::ChildStdout>,
}

impl Camera {
    /// Spawn the Python bridge subprocess.
    pub fn open(config: &CameraConfig) -> Result<Self> {
        let bridge_path = config.mvm_repo.join("tools/mvm_bridge.py");

        let python = config.python.clone().unwrap_or_else(|| {
            config.mvm_repo.join("venv/bin/python3")
        });

        let mut cmd = Command::new(&python);
        cmd.arg(&bridge_path)
            .arg("--json-only")
            .stdout(Stdio::piped())
            .stderr(Stdio::null());

        if config.cpu {
            cmd.arg("--cpu");
        }

        if let Some(ref frame_path) = config.frame_output {
            cmd.arg("--write-frame");
            cmd.arg(frame_path);
        }

        let mut child = cmd.spawn().with_context(|| {
            format!(
                "Failed to spawn bridge: {} {}\n  (is the old MVM venv set up?)\n  repo: {}",
                python.display(),
                bridge_path.display(),
                config.mvm_repo.display(),
            )
        })?;

        let stdout = child.stdout.take()
            .context("Failed to capture bridge stdout")?;

        let reader = BufReader::new(stdout);
        eprintln!("  camera: bridge started (pid {})", child.id());

        Ok(Self { child, reader })
    }

    /// Parse a single JSON line into Detections.
    pub fn parse_line(line: &str) -> Result<Vec<Detection>> {
        let line = line.trim();
        if line.is_empty() { return Ok(vec![]); }

        let msg: BridgeMsg = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse bridge JSON: {line}"))?;

        Ok(msg.detections.into_iter().map(|d| Detection {
            bbox: mvm_core::BBox {
                x1: d.x1.clamp(0.0, 1.0), y1: d.y1.clamp(0.0, 1.0),
                x2: d.x2.clamp(0.0, 1.0), y2: d.y2.clamp(0.0, 1.0),
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
            emotion_distribution: None, // Python bridge doesn't output distributions
        }).collect())
    }

    /// Read one JSON line from the bridge, skipping non-JSON noise.
    pub fn read(&mut self) -> Result<Vec<Detection>> {
        loop {
            let mut line = String::new();
            match self.reader.read_line(&mut line) {
                Ok(0) => return Err(anyhow::anyhow!("Bridge subprocess exited unexpectedly")),
                Ok(_) => {
                    let trimmed = line.trim();
                    if trimmed.is_empty() { continue; }
                    if !trimmed.starts_with('{') {
                        eprintln!("  bridge log: {trimmed}");
                        continue;
                    }
                    return Self::parse_line(trimmed);
                }
                Err(e) => return Err(anyhow::anyhow!("Failed to read from bridge: {}", e)),
            }
        }
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
}
