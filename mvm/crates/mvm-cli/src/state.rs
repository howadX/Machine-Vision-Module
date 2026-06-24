//! State file writer. Produces key: value format compatible with
//! Hermes Sight agent and sight-journal skill.
//!
//! Ported from sightd v0.3 `src/state.rs`.

use anyhow::Result;
use chrono::{FixedOffset, Local};
use mvm_core::PersonTracker;
use serde::Serialize;
use std::fs;
use std::path::PathBuf;

/// Path to the Sight state file.
const STATE_PATH: &str = ".hermes/profiles/sight/workspace/current_state.md";

#[derive(Serialize)]
struct JsonFace {
    id: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    identity_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    emotion_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    posture: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    posture_conf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dist_m: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pos: Option<[f32; 3]>,
}

#[derive(Serialize)]
struct JsonFrame<'a> {
    ts: String,
    fps: f64,
    status: &'a str,
    faces: Vec<JsonFace>,
}

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
        let timestamp = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();

        let mut active_tracks: Vec<_> = tracker.active().collect();
        active_tracks.sort_by_key(|t| t.id);
        let face_count = active_tracks.len();

        let status = if face_count > 0 {
            "occupied"
        } else {
            "empty"
        };

        let dominant_emotion = active_tracks
            .iter()
            .filter_map(|t| t.emotion.as_deref())
            .next()
            .unwrap_or("none");

        let mut lines = vec![
            "# Camera State (mvm v0.4)".to_string(),
            format!("- Last checked: {timestamp}"),
            "- Camera: /dev/video0".to_string(),
            "- Uptime: —".to_string(),
            format!("- FPS: {fps:.1}"),
            format!("- Face count: {face_count}"),
            format!("- Dominant emotion: {dominant_emotion}"),
        ];

        if face_count > 0 {
            let names: Vec<String> = active_tracks
                .iter()
                .map(|t| t.identity.as_deref().unwrap_or("unknown").to_string())
                .collect();
            lines.push(format!("- Persons visible: {}", names.join(", ")));

            for (i, t) in active_tracks.iter().enumerate() {
                let identity = t.identity.as_deref().unwrap_or("unknown");
                let id_conf = t
                    .identity_confidence
                    .map(|c| format!(" ({c:.2})"))
                    .unwrap_or_default();
                let emotion = t.emotion.as_deref().unwrap_or("none");
                let emo_conf = t
                    .emotion_confidence
                    .map(|c| format!(" ({c:.2})"))
                    .unwrap_or_default();
                let posture = t.posture.as_deref().unwrap_or("—");
                let pos_conf = t
                    .posture_confidence
                    .map(|c| format!(" ({c:.2})"))
                    .unwrap_or_default();

                lines.push(format!("- Person_{}:", i + 1));
                lines.push(format!("  - id: {}", t.id));
                lines.push(format!("  - identity: {identity}{id_conf}"));
                lines.push(format!(
                    "  - bbox: {:.2},{:.2},{:.2},{:.2}",
                    t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2
                ));
                lines.push(format!("  - age: {} frames", t.age));
                lines.push(format!("  - emotion: {emotion}{emo_conf}"));
                lines.push(format!("  - posture: {posture}{pos_conf}"));
                if let Some(d) = t.distance {
                    lines.push(format!("  - distance: {d:.2}"));
                }
                if let (Some(x), Some(y), Some(z)) =
                    (t.position_x, t.position_y, t.position_z)
                {
                    lines.push(format!("  - position: ({x:.2}, {y:.2}, {z:.2})m"));
                }
                if let Some(ws) = t.ws_confidence {
                    lines.push(format!("  - ws_confidence: {ws:.2}"));
                }
                if let Some(re) = t.recon_error {
                    lines.push(format!("  - recon_error: {re:.4}"));
                }
                if let Some(ref pl) = t.plausibility {
                    lines.push(format!("  - plausibility: {pl}"));
                }
            }
        } else {
            lines.push("- Persons visible: none".to_string());
        }

        lines.push(format!("- Status: {status}"));
        lines.push(String::new());

        fs::write(&self.path, lines.join("\n"))?;

        eprintln!(
            "  state written: {face_count} faces, {fps:.1} fps"
        );
        Ok(())
    }

    /// Serialise current tracker state to a single JSON line.
    pub fn to_json_line(&self, tracker: &PersonTracker, fps: f64) -> String {
        let now = Local::now().with_timezone(&self.tz);
        let ts = now.format("%Y-%m-%dT%H:%M:%S%:z").to_string();

        let mut active: Vec<_> = tracker.active().collect();
        active.sort_by_key(|t| t.id);
        let status = if active.is_empty() { "empty" } else { "occupied" };

        let faces: Vec<JsonFace> = active.iter().map(|t| {
            let pos = match (t.position_x, t.position_y, t.position_z) {
                (Some(x), Some(y), Some(z)) => Some([x, y, z]),
                _ => None,
            };
            JsonFace {
                id: t.id,
                identity: t.identity.clone(),
                identity_conf: t.identity_confidence,
                emotion: t.emotion.clone(),
                emotion_conf: t.emotion_confidence,
                posture: t.posture.clone(),
                posture_conf: t.posture_confidence,
                dist_m: t.distance,
                pos,
            }
        }).collect();

        let frame = JsonFrame { ts, fps, status, faces };
        serde_json::to_string(&frame).unwrap_or_else(|_| "{}".to_string())
    }

    /// Print a formatted table of all tracked persons to stderr.
    pub fn dump_table(&self, tracker: &PersonTracker, fps: f64) {
        let now = Local::now().with_timezone(&self.tz);
        let ts = now.format("%Y-%m-%dT%H:%M:%S%:z");

        let active: Vec<_> = tracker.active().collect();
        let face_count = active.len();
        let status = if face_count > 0 { "occupied" } else { "empty" };

        eprintln!();
        eprintln!("  ╔══════════════════════════════════════════════════════════════════════╗");
        eprintln!("  ║  MVM Rust Pipeline — Frame Analysis                               ║");
        eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");
        eprintln!("  ║  Time: {ts} | FPS: {fps:.1} | Status: {status:<8} | Faces: {face_count:<3}         ║", 
            ts = ts, fps = fps, status = status, face_count = face_count);
        eprintln!("  ╠══════════════════════════════════════════════════════════════════════╣");

        if face_count > 0 {
            eprintln!("  ║ {:>3} │ {:<12} │ {:>5} │ {:<12} │ {:>5} │ {:<13} │ {:>5} │ {:>6} │ {:>18} ║",
                "ID", "Identity", "Conf", "Emotion", "Conf", "Posture", "Conf", "Dist,m", "Position (x,y,z)m");
            eprintln!("  ╟─────┼──────────────┼───────┼──────────────┼───────┼───────────────┼───────┼────────┼────────────────────╢");

            for t in &active {
                let id = t.id;
                let identity = t.identity.as_deref().unwrap_or("—");
                let id_conf = t.identity_confidence.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
                let emotion = t.emotion.as_deref().unwrap_or("—");
                let emo_conf = t.emotion_confidence.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
                let posture = t.posture.as_deref().unwrap_or("—");
                let pos_conf = t.posture_confidence.map(|c| format!("{c:.2}")).unwrap_or_else(|| "—".into());
                let dist = t.distance.map(|d| format!("{d:.2}")).unwrap_or_else(|| "—".into());
                let position = match (t.position_x, t.position_y, t.position_z) {
                    (Some(x), Some(y), Some(z)) => format!("({x:.2}, {y:.2}, {z:.2})"),
                    _ => "—".into(),
                };

                eprintln!("  ║ {:>3} │ {:<12} │ {:>5} │ {:<12} │ {:>5} │ {:<13} │ {:>5} │ {:>6} │ {:>18} ║",
                    id, identity, id_conf, emotion, emo_conf, posture, pos_conf, dist, position);
            }
        } else {
            eprintln!("  ║  No faces detected                                                ║");
        }

        eprintln!("  ╚══════════════════════════════════════════════════════════════════════╝");
        eprintln!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{Detection, PersonTracker};

    fn temp_state_path(name: &str) -> PathBuf {
        let dir = std::env::temp_dir()
            .join(format!("mvm_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(name)
    }

    fn read_state(path: &PathBuf) -> String {
        std::fs::read_to_string(path).unwrap()
    }

    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> Detection {
        Detection::bare(x1, y1, x2, y2)
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
        assert!(content.contains("- Dominant emotion: none"));
        assert!(content.contains("- Persons visible: none"));
        assert!(!content.contains("Person_1:"));
    }

    #[test]
    fn state_enriched_person() {
        let path = temp_state_path("test_rich.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.tick(&[Detection {
            bbox: mvm_core::BBox {
                x1: 0.2,
                y1: 0.3,
                x2: 0.5,
                y2: 0.7,
            },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: Some("Нейтральная".into()),
            emotion_confidence: Some(0.65),
            posture: Some("сидит".into()),
            posture_confidence: Some(0.95),
            distance: None,
            position_x: None,
            position_y: None,
            position_z: None,
            ws_confidence: None,
            recon_error: None,
            plausibility: None,
        }]);

        sf.write(&tracker, 25.5).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Status: occupied"));
        assert!(content.contains("- Dominant emotion: Нейтральная"));
        assert!(content.contains("- Persons visible: Вадим"));
        assert!(content.contains("  - identity: Вадим (0.72)"));
        assert!(content.contains("  - bbox: 0.20,0.30,0.50,0.70"));
        assert!(content.contains("  - emotion: Нейтральная (0.65)"));
        assert!(content.contains("  - posture: сидит (0.95)"));
    }

    #[test]
    fn state_unknown_person() {
        let path = temp_state_path("test_unknown.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.tick(&[det(0.2, 0.3, 0.5, 0.7)]);

        sf.write(&tracker, 30.0).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Persons visible: unknown"));
        assert!(content.contains("  - identity: unknown"));
        assert!(content.contains("  - emotion: none"));
        assert!(content.contains("  - posture: —"));
    }

    #[test]
    fn state_multiple_people() {
        let path = temp_state_path("test_multi.md");
        let sf = StateFile::open_at(path.clone()).unwrap();

        let mut tracker = PersonTracker::new();
        tracker.tick(&[
            Detection {
                bbox: mvm_core::BBox {
                    x1: 0.1,
                    y1: 0.1,
                    x2: 0.3,
                    y2: 0.5,
                },
                identity: Some("Вадим".into()),
                identity_confidence: Some(0.8),
                emotion: Some("Нейтральная".into()),
                emotion_confidence: Some(0.6),
                posture: Some("сидит".into()),
                posture_confidence: Some(0.9),
                distance: None,
                position_x: None,
                position_y: None,
                position_z: None,
                ws_confidence: None,
                recon_error: None,
                plausibility: None,
            },
            Detection {
                bbox: mvm_core::BBox {
                    x1: 0.6,
                    y1: 0.2,
                    x2: 0.9,
                    y2: 0.7,
                },
                identity: Some("Даня".into()),
                identity_confidence: Some(0.75),
                emotion: Some("Радость".into()),
                emotion_confidence: Some(0.7),
                posture: Some("стоит".into()),
                posture_confidence: Some(0.85),
                distance: None,
                position_x: None,
                position_y: None,
                position_z: None,
                ws_confidence: None,
                recon_error: None,
                plausibility: None,
            },
        ]);

        sf.write(&tracker, 29.9).unwrap();

        let content = read_state(&path);
        assert!(content.contains("- Face count: 2"));
        assert!(content.contains("- Persons visible: Вадим, Даня"));
        assert!(content.contains("  - identity: Вадим (0.80)"));
        assert!(content.contains("  - identity: Даня (0.75)"));
        assert!(content.contains("  - emotion: Радость (0.70)"));
        assert!(content.contains("  - posture: стоит (0.85)"));
    }

    #[test]
    fn state_timestamp_is_iso8601() {
        let path = temp_state_path("test_ts.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 0.0).unwrap();

        let content = read_state(&path);
        let ts_line = content
            .lines()
            .find(|l| l.starts_with("- Last checked:"))
            .unwrap();
        let ts = ts_line.strip_prefix("- Last checked: ").unwrap();
        assert!(ts.contains('T'));
        assert!(ts.ends_with("+03:00") || ts.ends_with("+0300"));
    }

    #[test]
    fn state_file_creates_parent_dirs() {
        let path = std::env::temp_dir()
            .join("mvm_test_nested")
            .join("a")
            .join("b")
            .join("test_nested.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 0.0).unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_dir_all(
            path.parent()
                .unwrap()
                .parent()
                .unwrap()
                .parent()
                .unwrap(),
        );
    }

    #[test]
    fn state_header_present() {
        let path = temp_state_path("test_header.md");
        let sf = StateFile::open_at(path.clone()).unwrap();
        let tracker = PersonTracker::new();
        sf.write(&tracker, 15.0).unwrap();

        let content = read_state(&path);
        assert!(content.contains("Camera: /dev/video0"));
    }

    #[test]
    fn json_line_is_valid_json() {
        let sf = StateFile::open_at(temp_state_path("test_json.md")).unwrap();
        let tracker = PersonTracker::new();
        let line = sf.to_json_line(&tracker, 30.0);
        let v: serde_json::Value = serde_json::from_str(&line).expect("must be valid JSON");
        assert!(v["ts"].is_string());
        assert!(v["fps"].is_number());
        assert!(v["status"].is_string());
        assert!(v["faces"].is_array());
    }

    #[test]
    fn json_line_contains_face_fields() {
        let sf = StateFile::open_at(temp_state_path("test_json_face.md")).unwrap();
        let mut tracker = PersonTracker::new();
        tracker.tick(&[Detection {
            bbox: mvm_core::BBox { x1: 0.2, y1: 0.3, x2: 0.5, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.69),
            emotion: Some("Радость".into()),
            emotion_confidence: Some(1.0),
            posture: Some("сидит_за_ПК".into()),
            posture_confidence: Some(0.89),
            distance: Some(2.75),
            position_x: Some(0.75),
            position_y: Some(0.36),
            position_z: Some(2.75),
            ws_confidence: Some(0.9),
            recon_error: None,
            plausibility: None,
        }]);
        let line = sf.to_json_line(&tracker, 29.9);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        let face = &v["faces"][0];
        assert_eq!(face["identity"], "Вадим");
        assert_eq!(face["emotion"], "Радость");
        assert_eq!(face["posture"], "сидит_за_ПК");
        assert!((face["identity_conf"].as_f64().unwrap() - 0.69).abs() < 0.01);
        assert!((face["emotion_conf"].as_f64().unwrap() - 1.0).abs() < 0.01);
        assert!((face["dist_m"].as_f64().unwrap() - 2.75).abs() < 0.01);
        let pos = &face["pos"];
        assert!(pos.is_array());
        assert_eq!(pos.as_array().unwrap().len(), 3);
    }

    #[test]
    fn json_line_empty_room() {
        let sf = StateFile::open_at(temp_state_path("test_json_empty.md")).unwrap();
        let tracker = PersonTracker::new();
        let line = sf.to_json_line(&tracker, 0.0);
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        assert_eq!(v["status"], "empty");
        assert_eq!(v["faces"].as_array().unwrap().len(), 0);
    }
}
