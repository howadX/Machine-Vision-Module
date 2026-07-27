//! Data-quality gates (§16.6-A) — the engine behind `mvm dataset check`.
//!
//! Pure over a session/dataset on disk: parses `labels.jsonl`, verifies structural
//! integrity + the semantic contract (binding keys, referenced artifacts, mask RLE,
//! depth range, privacy) and folds quality-as-data into a stats block. Returns a
//! typed report so the CLI can print it and gate CI on the error count.
//!
//! **No pixel decoding.** `mvm-dataset` links no image crate (invariant #2), so this
//! checks that referenced files *exist* and that label-encoded integers are sound;
//! decoding frames/depth/masks back to arrays is the Python `validate.py` side.
//!
//! Honesty rules it enforces (§16.0.2): a `withheld` posture is counted as a
//! withheld gap, never as an error and never as a class. Absence is data.

use crate::observation::CanonicalObservation;
use crate::observation::SCHEMA_VERSION;
use crate::session::{list_sessions, read_meta, sessions_root};
use mvm_core::mask::MASK_RES;
use std::collections::BTreeMap;
use std::path::Path;

/// Attribute keys that must never appear in `labels.jsonl` — raw face descriptors
/// are identity-grade PII and belong in the (separately gated) embeddings sidecar,
/// not the self-describing label stream. Guards a future regression that would dump
/// SFace vectors inline regardless of privacy mode.
const EMBEDDING_KEYS: [&str; 4] = [
    "embedding",
    "descriptor",
    "sface_embedding",
    "face_embedding",
];

/// A finding's severity. Only [`Severity::Error`] fails the gate (unless `--strict`
/// also promotes warnings — that policy lives in the CLI, not here).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// A broken contract: fails `mvm dataset check`.
    Error,
    /// A soft anomaly worth surfacing but not gate-failing on its own.
    Warn,
}

/// One issue found in a session, anchored to the frame it came from when known.
#[derive(Debug, Clone)]
pub struct Finding {
    pub severity: Severity,
    /// `frame_id` of the offending observation, or `None` for whole-session issues.
    pub frame_id: Option<u64>,
    /// Stable machine-readable slug (e.g. `"missing-frame"`, `"bad-mask-rle"`).
    pub code: &'static str,
    /// Human-readable detail.
    pub msg: String,
}

/// Quality-as-data rollup for a session — the numbers `check` prints alongside
/// findings. Not a pass/fail signal itself; a skewed balance or high withheld-rate
/// is information the curator acts on (feeds 16.6-B hard-example mining).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CheckStats {
    pub frames: u64,
    pub people: u64,
    pub objects: u64,
    /// People carrying a gallery `identity` label.
    pub gallery_people: u64,
    /// People with no identity (anonymous).
    pub anon_people: u64,
    /// Person-frames whose posture is honestly `withheld` (§16.0.2), not a class.
    pub withheld_posture: u64,
    /// Frames that reference a depth map.
    pub with_depth: u64,
    /// Object instance masks encountered (and RLE-validated).
    pub masks: u64,
    pub posture_hist: BTreeMap<String, u64>,
    pub emotion_hist: BTreeMap<String, u64>,
    pub object_hist: BTreeMap<String, u64>,
}

impl CheckStats {
    /// Fraction of person-frames whose posture was withheld, `0.0` if no people.
    pub fn withheld_rate(&self) -> f32 {
        if self.people == 0 {
            0.0
        } else {
            self.withheld_posture as f32 / self.people as f32
        }
    }
}

/// Result of checking one session.
#[derive(Debug, Clone)]
pub struct SessionReport {
    pub id: String,
    pub findings: Vec<Finding>,
    pub stats: CheckStats,
}

impl SessionReport {
    pub fn errors(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Error)
            .count()
    }
    pub fn warns(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == Severity::Warn)
            .count()
    }
}

/// Result of checking a whole dataset root (one report per session).
#[derive(Debug, Clone)]
pub struct DatasetReport {
    pub sessions: Vec<SessionReport>,
}

impl DatasetReport {
    pub fn errors(&self) -> usize {
        self.sessions.iter().map(SessionReport::errors).sum()
    }
    pub fn warns(&self) -> usize {
        self.sessions.iter().map(SessionReport::warns).sum()
    }
    /// The gate passes when no session has an error.
    pub fn ok(&self) -> bool {
        self.errors() == 0
    }
}

/// Check every session under `<root>/sessions`.
pub fn check_dataset(root: &Path) -> anyhow::Result<DatasetReport> {
    let sessions = list_sessions(root)?;
    let mut out = Vec::with_capacity(sessions.len());
    for si in &sessions {
        out.push(check_session(&sessions_root(root).join(&si.id), &si.id));
    }
    Ok(DatasetReport { sessions: out })
}

/// Check one session directory. `id` is the expected `session_id` binding key
/// (normally the folder name). Pure over the filesystem — never mutates.
pub fn check_session(session_dir: &Path, id: &str) -> SessionReport {
    let mut findings = Vec::new();
    let mut stats = CheckStats::default();

    let labels_path = session_dir.join("labels.jsonl");
    let text = match std::fs::read_to_string(&labels_path) {
        Ok(t) => t,
        Err(_) => {
            findings.push(Finding {
                severity: Severity::Warn,
                frame_id: None,
                code: "no-labels",
                msg: format!("no labels.jsonl under {}", session_dir.display()),
            });
            return SessionReport {
                id: id.to_string(),
                findings,
                stats,
            };
        }
    };

    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let obs = match CanonicalObservation::from_jsonl_line(line) {
            Ok(o) => o,
            Err(e) => {
                findings.push(Finding {
                    severity: Severity::Error,
                    frame_id: None,
                    code: "parse",
                    msg: format!("labels.jsonl line {}: {e}", i + 1),
                });
                continue;
            }
        };
        stats.frames += 1;
        check_observation(session_dir, id, &obs, &mut findings, &mut stats);
    }

    // Cross-check the closed-session summary against what's actually on disk.
    if let Ok(Some(meta)) = read_meta(session_dir) {
        if meta.session.saved_frames != stats.frames {
            findings.push(Finding {
                severity: Severity::Warn,
                frame_id: None,
                code: "frame-count-mismatch",
                msg: format!(
                    "session.toml saved_frames={} but {} labels on disk",
                    meta.session.saved_frames, stats.frames
                ),
            });
        }
    }

    SessionReport {
        id: id.to_string(),
        findings,
        stats,
    }
}

/// Push an error finding anchored to a frame.
fn err(findings: &mut Vec<Finding>, fid: u64, code: &'static str, msg: String) {
    findings.push(Finding {
        severity: Severity::Error,
        frame_id: Some(fid),
        code,
        msg,
    });
}

/// Push a warning finding anchored to a frame.
fn warn(findings: &mut Vec<Finding>, fid: u64, code: &'static str, msg: String) {
    findings.push(Finding {
        severity: Severity::Warn,
        frame_id: Some(fid),
        code,
        msg,
    });
}

/// Validate one observation's binding keys, referenced artifacts, and entities;
/// accumulate stats. Split out so the per-line loop stays readable.
fn check_observation(
    session_dir: &Path,
    id: &str,
    obs: &CanonicalObservation,
    findings: &mut Vec<Finding>,
    stats: &mut CheckStats,
) {
    let fid = obs.frame_id;

    // Binding keys (§16.0.1).
    if obs.session_id != id {
        err(
            findings,
            fid,
            "session-id-mismatch",
            format!("session_id={:?} but folder is {id:?}", obs.session_id),
        );
    }
    if obs.schema_version != SCHEMA_VERSION {
        warn(
            findings,
            fid,
            "schema-drift",
            format!(
                "schema_version={:?}, checker expects {SCHEMA_VERSION:?}",
                obs.schema_version
            ),
        );
    }

    // Referenced RGB frame must exist on disk.
    if let Some(rel) = &obs.frame.frame_path {
        if !session_dir.join(rel).is_file() {
            err(
                findings,
                fid,
                "missing-frame",
                format!("frame_path {rel:?} not on disk"),
            );
        }
    }

    // Depth: file must exist and carry a reversible range (§16.2.2).
    if let Some(rel) = &obs.frame.depth_path {
        stats.with_depth += 1;
        if !session_dir.join(rel).is_file() {
            err(
                findings,
                fid,
                "missing-depth",
                format!("depth_path {rel:?} not on disk"),
            );
        }
        match obs.frame.depth_range {
            None => err(
                findings,
                fid,
                "bad-depth-range",
                "depth_path set but depth_range missing".into(),
            ),
            Some([lo, hi]) if !(lo.is_finite() && hi.is_finite() && lo <= hi) => {
                err(
                    findings,
                    fid,
                    "bad-depth-range",
                    format!("depth_range [{lo}, {hi}] is not a valid [min ≤ max]"),
                );
            }
            Some(_) => {}
        }
    } else if obs.frame.depth_range.is_some() {
        warn(
            findings,
            fid,
            "orphan-depth-range",
            "depth_range present without a depth_path".into(),
        );
    }

    // Entities.
    let mut person_entities = 0u32;
    let mut object_entities = 0u32;
    for e in &obs.entities {
        if e.entity_id.trim().is_empty() {
            err(
                findings,
                fid,
                "empty-entity-id",
                "entity has an empty entity_id".into(),
            );
        }
        // Privacy: no raw face descriptor may ride inline in the labels.
        if let Some(map) = e.attributes.as_object() {
            for k in EMBEDDING_KEYS {
                if map.contains_key(k) {
                    err(
                        findings,
                        fid,
                        "embedding-leak",
                        format!(
                            "entity {:?} carries a raw {k:?} — face descriptors must not sit in labels.jsonl",
                            e.entity_id
                        ),
                    );
                }
            }
        }
        match e.kind.as_str() {
            "person" => {
                person_entities += 1;
                stats.people += 1;
                accumulate_person(&e.attributes, stats);
            }
            "object" => {
                object_entities += 1;
                stats.objects += 1;
                accumulate_object(e, fid, findings, stats);
            }
            other => warn(
                findings,
                fid,
                "unknown-kind",
                format!("entity {:?} has unknown kind {other:?}", e.entity_id),
            ),
        }
    }

    // Quality block must agree with the entities it summarizes.
    if obs.quality.people_count != person_entities {
        warn(
            findings,
            fid,
            "quality-count-mismatch",
            format!(
                "quality.people_count={} but {person_entities} person entities",
                obs.quality.people_count
            ),
        );
    }
    if obs.quality.object_count != object_entities {
        warn(
            findings,
            fid,
            "quality-count-mismatch",
            format!(
                "quality.object_count={} but {object_entities} object entities",
                obs.quality.object_count
            ),
        );
    }
}

/// Fold a person's observed attributes into stats (identity, posture balance,
/// withheld-rate, emotion balance). Never fails — a person's shape is not a gate.
fn accumulate_person(attrs: &serde_json::Value, stats: &mut CheckStats) {
    if attrs.get("identity").is_some() {
        stats.gallery_people += 1;
    } else {
        stats.anon_people += 1;
    }
    if let Some(label) = attrs
        .get("posture")
        .and_then(|p| p.get("label"))
        .and_then(|l| l.as_str())
    {
        *stats.posture_hist.entry(label.to_string()).or_default() += 1;
    }
    // A withheld posture is honest absence (§16.0.2) — track the rate, never error.
    if attrs.get("posture_status").and_then(|s| s.as_str()) == Some("withheld") {
        stats.withheld_posture += 1;
    }
    if let Some(label) = attrs
        .get("emotion")
        .and_then(|e| e.get("label"))
        .and_then(|l| l.as_str())
    {
        *stats.emotion_hist.entry(label.to_string()).or_default() += 1;
    }
}

/// Fold an object into stats and validate its mask RLE if present.
fn accumulate_object(
    e: &crate::observation::EntityRecord,
    fid: u64,
    findings: &mut Vec<Finding>,
    stats: &mut CheckStats,
) {
    if let Some(class) = e.attributes.get("class").and_then(|c| c.as_str()) {
        *stats.object_hist.entry(class.to_string()).or_default() += 1;
    }
    if let Some(mask) = e.attributes.get("mask") {
        stats.masks += 1;
        validate_mask_rle(e, mask, fid, findings);
    }
}

/// A mask RLE must be a run-length list covering exactly `res²` cells (§16.2.3).
fn validate_mask_rle(
    e: &crate::observation::EntityRecord,
    mask: &serde_json::Value,
    fid: u64,
    findings: &mut Vec<Finding>,
) {
    let res = mask
        .get("res")
        .and_then(|r| r.as_u64())
        .unwrap_or(MASK_RES as u64);
    if res != MASK_RES as u64 {
        err(
            findings,
            fid,
            "mask-res-drift",
            format!(
                "object {:?} mask res={res}, checker expects {MASK_RES}",
                e.entity_id
            ),
        );
    }
    let Some(rle) = mask.get("rle").and_then(|r| r.as_array()) else {
        err(
            findings,
            fid,
            "bad-mask-rle",
            format!("object {:?} mask has no rle array", e.entity_id),
        );
        return;
    };
    let sum: u64 = rle.iter().filter_map(|v| v.as_u64()).sum();
    let expected = res * res;
    if sum != expected {
        err(
            findings,
            fid,
            "bad-mask-rle",
            format!(
                "object {:?} mask RLE covers {sum} cells, expected {expected}",
                e.entity_id
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{EntityRecord, SemanticGroup};
    use std::path::PathBuf;

    fn temp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mvm_val_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// A minimal person entity with the given attributes.
    fn person(id: &str, track: u64, attrs: serde_json::Value) -> EntityRecord {
        EntityRecord {
            entity_id: id.into(),
            track_id: Some(track),
            group: SemanticGroup::Human,
            kind: "person".into(),
            attributes: attrs,
        }
    }

    fn object(id: &str, attrs: serde_json::Value) -> EntityRecord {
        EntityRecord {
            entity_id: id.into(),
            track_id: Some(10_001),
            group: SemanticGroup::Scene,
            kind: "object".into(),
            attributes: attrs,
        }
    }

    /// Build an observation for session `sid`, frame `fid`, with the given entities;
    /// quality counts are set to match so the happy path stays warning-free.
    fn obs(sid: &str, fid: u64, entities: Vec<EntityRecord>) -> CanonicalObservation {
        let mut o = CanonicalObservation::new("ds", sid, fid, "2026-07-23T00:00:00", fid);
        o.quality.people_count = entities.iter().filter(|e| e.kind == "person").count() as u32;
        o.quality.object_count = entities.iter().filter(|e| e.kind == "object").count() as u32;
        o.entities = entities;
        o
    }

    /// Write observations as a session on disk; create every referenced frame/depth
    /// file so the happy path finds them. Returns the session dir.
    fn write_session(root: &Path, sid: &str, frames: &[CanonicalObservation]) -> PathBuf {
        let dir = sessions_root(root).join(sid);
        std::fs::create_dir_all(dir.join("frames")).unwrap();
        std::fs::create_dir_all(dir.join("depth")).unwrap();
        let mut body = String::new();
        for o in frames {
            if let Some(rel) = &o.frame.frame_path {
                std::fs::write(dir.join(rel), b"jpg").unwrap();
            }
            if let Some(rel) = &o.frame.depth_path {
                std::fs::write(dir.join(rel), b"png").unwrap();
            }
            body.push_str(&o.to_jsonl_line().unwrap());
            body.push('\n');
        }
        std::fs::write(dir.join("labels.jsonl"), body).unwrap();
        dir
    }

    /// A full RLE over the 64² grid: one all-background run of exactly res² cells.
    fn full_bg_rle() -> serde_json::Value {
        serde_json::json!({ "res": MASK_RES, "coverage": 0.0, "rle": [MASK_RES * MASK_RES] })
    }

    #[test]
    fn clean_session_has_no_errors_and_folds_stats() {
        let root = temp_root("clean");
        let mut f0 = obs(
            "s",
            0,
            vec![
                person(
                    "p1",
                    0,
                    serde_json::json!({ "identity": {"label":"Вадим","confidence":0.9}, "posture": {"label":"sitting","confidence":0.95}, "posture_status":"observed", "emotion":{"label":"Neutral","confidence":0.8} }),
                ),
                person(
                    "p10",
                    10,
                    serde_json::json!({ "posture_status":"withheld" }),
                ),
                object(
                    "o10001",
                    serde_json::json!({ "class":"cup", "confidence":0.7, "mask": full_bg_rle() }),
                ),
            ],
        );
        f0.frame.frame_path = Some("frames/000000.jpg".into());
        f0.frame.depth_path = Some("depth/000000.png".into());
        f0.frame.depth_range = Some([0.3, 9.0]);
        let dir = write_session(&root, "s", &[f0]);

        let r = check_session(&dir, "s");
        assert_eq!(
            r.errors(),
            0,
            "clean session must not error: {:?}",
            r.findings
        );
        assert_eq!(
            r.warns(),
            0,
            "clean session must not warn: {:?}",
            r.findings
        );
        assert_eq!(r.stats.frames, 1);
        assert_eq!(r.stats.people, 2);
        assert_eq!(r.stats.gallery_people, 1);
        assert_eq!(r.stats.anon_people, 1);
        assert_eq!(r.stats.withheld_posture, 1);
        assert_eq!(r.stats.withheld_rate(), 0.5);
        assert_eq!(r.stats.objects, 1);
        assert_eq!(r.stats.masks, 1);
        assert_eq!(r.stats.with_depth, 1);
        assert_eq!(r.stats.posture_hist.get("sitting"), Some(&1));
        assert_eq!(r.stats.object_hist.get("cup"), Some(&1));
    }

    #[test]
    fn missing_frame_file_is_error() {
        let root = temp_root("missframe");
        let mut f = obs("s", 0, vec![]);
        f.frame.frame_path = Some("frames/000000.jpg".into());
        // Write the session but delete the referenced frame file.
        let dir = write_session(&root, "s", &[f]);
        std::fs::remove_file(dir.join("frames/000000.jpg")).unwrap();
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 1);
        assert_eq!(r.findings[0].code, "missing-frame");
    }

    #[test]
    fn bad_mask_rle_is_error() {
        let root = temp_root("badrle");
        let bad = object(
            "o10001",
            serde_json::json!({ "class":"cup", "mask": {"res":MASK_RES,"coverage":0.0,"rle":[10, 20]} }),
        );
        let dir = write_session(&root, "s", &[obs("s", 0, vec![bad])]);
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 1, "{:?}", r.findings);
        assert_eq!(r.findings[0].code, "bad-mask-rle");
        // The mask was still counted (integrity failed, presence noted).
        assert_eq!(r.stats.masks, 1);
    }

    #[test]
    fn depth_without_range_is_error() {
        let root = temp_root("badrange");
        let mut f = obs("s", 0, vec![]);
        f.frame.depth_path = Some("depth/000000.png".into());
        f.frame.depth_range = None; // set path, forgot the reversible range
        let dir = write_session(&root, "s", &[f]);
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 1);
        assert_eq!(r.findings[0].code, "bad-depth-range");
    }

    #[test]
    fn inverted_depth_range_is_error() {
        let root = temp_root("invrange");
        let mut f = obs("s", 0, vec![]);
        f.frame.depth_path = Some("depth/000000.png".into());
        f.frame.depth_range = Some([9.0, 0.3]); // min > max
        let dir = write_session(&root, "s", &[f]);
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 1);
        assert_eq!(r.findings[0].code, "bad-depth-range");
    }

    #[test]
    fn session_id_mismatch_is_error() {
        let root = temp_root("idmismatch");
        // Observation claims a different session_id than its folder.
        let dir = write_session(&root, "folder_id", &[obs("other_id", 0, vec![])]);
        let r = check_session(&dir, "folder_id");
        assert_eq!(r.errors(), 1);
        assert_eq!(r.findings[0].code, "session-id-mismatch");
    }

    #[test]
    fn embedding_in_labels_is_a_privacy_leak() {
        let root = temp_root("leak");
        let leaky = person("p0", 0, serde_json::json!({ "embedding": [0.1, 0.2, 0.3] }));
        let dir = write_session(&root, "s", &[obs("s", 0, vec![leaky])]);
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 1);
        assert_eq!(r.findings[0].code, "embedding-leak");
    }

    #[test]
    fn unparseable_line_is_error() {
        let root = temp_root("garbage");
        let dir = sessions_root(&root).join("s");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("labels.jsonl"), "{ not json }\n").unwrap();
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 1);
        assert_eq!(r.findings[0].code, "parse");
        assert_eq!(r.stats.frames, 0, "a broken line is not a counted frame");
    }

    #[test]
    fn frame_count_mismatch_with_session_toml_warns() {
        use crate::session::{
            AnomaliesBlock, ObjectsBlock, ParticipantsBlock, SessionBlock, SessionMeta,
        };
        let root = temp_root("countmismatch");
        let dir = write_session(&root, "s", &[obs("s", 0, vec![]), obs("s", 1, vec![])]);
        // Summary claims 5 saved frames; only 2 labels exist.
        let meta = SessionMeta {
            session: SessionBlock {
                id: "s".into(),
                started: "a".into(),
                ended: "b".into(),
                saved_frames: 5,
            },
            participants: ParticipantsBlock {
                known: vec![],
                anonymous: 0,
            },
            activities: Default::default(),
            anomalies: AnomaliesBlock {
                max: 0.0,
                mean: 0.0,
            },
            objects: ObjectsBlock { top: vec![] },
        };
        std::fs::write(
            dir.join("session.toml"),
            toml::to_string_pretty(&meta).unwrap(),
        )
        .unwrap();
        let r = check_session(&dir, "s");
        assert_eq!(r.errors(), 0);
        assert_eq!(r.warns(), 1);
        assert_eq!(r.findings[0].code, "frame-count-mismatch");
    }

    #[test]
    fn check_dataset_reports_each_session() {
        let root = temp_root("dataset");
        write_session(
            &root,
            "2026-07-23_100000",
            &[obs("2026-07-23_100000", 0, vec![])],
        );
        // Second session has a leak.
        write_session(
            &root,
            "2026-07-23_110000",
            &[obs(
                "2026-07-23_110000",
                0,
                vec![person("p0", 0, serde_json::json!({ "descriptor": [1.0] }))],
            )],
        );
        let rep = check_dataset(&root).unwrap();
        assert_eq!(rep.sessions.len(), 2);
        assert!(!rep.ok());
        assert_eq!(rep.errors(), 1);
    }
}
