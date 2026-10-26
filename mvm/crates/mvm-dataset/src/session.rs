//! Session metadata model + on-disk readers (§16.3).
//!
//! [`SessionMeta`] is the `session.toml` schema — written by [`crate::writer`] on
//! close and read back here for `mvm dataset list/info/stats/prune`. The readers
//! are pure over a filesystem root so they unit-test against a tempdir.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// `session.toml` — the closed-session summary (§16.3.2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session: SessionBlock,
    pub participants: ParticipantsBlock,
    /// Activity label → number of saved frames it appeared in.
    #[serde(default)]
    pub activities: BTreeMap<String, u32>,
    pub anomalies: AnomaliesBlock,
    pub objects: ObjectsBlock,
    #[serde(default)]
    pub depth: DepthBlock,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SessionBlock {
    pub id: String,
    pub started: String,
    pub ended: String,
    pub saved_frames: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ParticipantsBlock {
    #[serde(default)]
    pub known: Vec<String>,
    #[serde(default)]
    pub anonymous: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnomaliesBlock {
    pub max: f32,
    pub mean: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ObjectsBlock {
    #[serde(default)]
    pub top: Vec<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DepthBlock {
    #[serde(default)]
    pub encodings: Vec<String>,
    #[serde(default)]
    pub calibration_revisions: Vec<String>,
}

/// One session as seen on disk. `meta` is `None` for an in-progress session whose
/// `session.toml` hasn't been finalized yet; `saved_frames_on_disk` (labels.jsonl
/// line count) is always available.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionInfo {
    pub id: String,
    pub meta: Option<SessionMeta>,
    pub saved_frames_on_disk: u64,
}

/// `<root>/sessions`.
pub fn sessions_root(root: &Path) -> PathBuf {
    root.join("sessions")
}

/// Read a single session's `session.toml`, if present.
pub fn read_meta(session_dir: &Path) -> anyhow::Result<Option<SessionMeta>> {
    let path = session_dir.join("session.toml");
    match std::fs::read_to_string(&path) {
        Ok(text) => Ok(Some(toml::from_str(&text)?)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(anyhow::anyhow!("read {}: {e}", path.display())),
    }
}

/// Count saved frames from `labels.jsonl` (one line per frame). 0 if absent.
pub fn labels_line_count(session_dir: &Path) -> u64 {
    match std::fs::read_to_string(session_dir.join("labels.jsonl")) {
        Ok(text) => text.lines().filter(|l| !l.trim().is_empty()).count() as u64,
        Err(_) => 0,
    }
}

/// List all sessions under `<root>/sessions`, sorted by id (chronological, since
/// ids are `YYYY-MM-DD_HHMMSS`). Empty (not an error) when the dir is absent.
pub fn list_sessions(root: &Path) -> anyhow::Result<Vec<SessionInfo>> {
    let dir = sessions_root(root);
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(anyhow::anyhow!("read {}: {e}", dir.display())),
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let p = entry.path();
        if !p.is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().into_owned();
        out.push(SessionInfo {
            meta: read_meta(&p)?,
            saved_frames_on_disk: labels_line_count(&p),
            id,
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

/// Aggregate stats over the whole dataset (§16.4.3 `mvm dataset stats`).
#[derive(Debug, Clone, PartialEq)]
pub struct DatasetStats {
    pub session_count: usize,
    pub total_saved_frames: u64,
    pub participants: BTreeSet<String>,
    pub anon_sessions: u32,
    pub activities: BTreeMap<String, u32>,
    pub anomaly_max: f32,
}

/// Fold a session list into dataset-wide stats.
pub fn aggregate(sessions: &[SessionInfo]) -> DatasetStats {
    let mut s = DatasetStats {
        session_count: sessions.len(),
        total_saved_frames: 0,
        participants: BTreeSet::new(),
        anon_sessions: 0,
        activities: BTreeMap::new(),
        anomaly_max: 0.0,
    };
    for si in sessions {
        s.total_saved_frames += si.saved_frames_on_disk;
        if let Some(m) = &si.meta {
            for k in &m.participants.known {
                s.participants.insert(k.clone());
            }
            if m.participants.anonymous > 0 {
                s.anon_sessions += 1;
            }
            for (act, c) in &m.activities {
                *s.activities.entry(act.clone()).or_default() += c;
            }
            s.anomaly_max = s.anomaly_max.max(m.anomalies.max);
        }
    }
    s
}

/// Retention policy for `mvm dataset prune`.
#[derive(Debug, Clone, PartialEq)]
pub enum PrunePolicy {
    /// Keep the N most recent sessions, remove older.
    KeepLast(usize),
    /// Keep sessions on/after the cutoff date `YYYY-MM-DD`, remove older.
    KeepSince(String),
}

/// Partition sorted session ids into (keep, remove) under a policy. Pure — the CLI
/// does the filesystem removal. Ids must be the chronological `YYYY-MM-DD_HHMMSS`.
pub fn partition_prune(sorted_ids: &[String], policy: &PrunePolicy) -> (Vec<String>, Vec<String>) {
    match policy {
        PrunePolicy::KeepLast(n) => {
            let cut = sorted_ids.len().saturating_sub(*n);
            (sorted_ids[cut..].to_vec(), sorted_ids[..cut].to_vec())
        }
        PrunePolicy::KeepSince(cutoff) => sorted_ids.iter().cloned().partition(|id| {
            id.as_str()
                .get(..10)
                .map(|d| d >= cutoff.as_str())
                .unwrap_or(true)
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn meta(id: &str, frames: u64, known: &[&str], anon: u32) -> SessionMeta {
        SessionMeta {
            session: SessionBlock {
                id: id.into(),
                started: "s".into(),
                ended: "e".into(),
                saved_frames: frames,
            },
            participants: ParticipantsBlock {
                known: known.iter().map(|s| s.to_string()).collect(),
                anonymous: anon,
            },
            activities: BTreeMap::from([("typing".to_string(), 3)]),
            anomalies: AnomaliesBlock {
                max: 0.5,
                mean: 0.1,
            },
            objects: ObjectsBlock {
                top: vec!["cup".into()],
            },
            depth: DepthBlock::default(),
        }
    }

    fn write_session(root: &Path, id: &str, m: Option<&SessionMeta>, lines: u64) {
        let d = sessions_root(root).join(id);
        std::fs::create_dir_all(&d).unwrap();
        if let Some(m) = m {
            std::fs::write(d.join("session.toml"), toml::to_string_pretty(m).unwrap()).unwrap();
        }
        let body: String = (0..lines).map(|_| "{}\n").collect();
        std::fs::write(d.join("labels.jsonl"), body).unwrap();
    }

    fn temp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("mvm_sess_{}_{}", tag, std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn session_meta_round_trips_through_toml() {
        let m = meta("2026-07-17_100000", 10, &["Вадим"], 2);
        let s = toml::to_string_pretty(&m).unwrap();
        let back: SessionMeta = toml::from_str(&s).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    fn list_sessions_sorted_with_in_progress() {
        let root = temp_root("list");
        write_session(
            &root,
            "2026-07-17_120000",
            Some(&meta("2026-07-17_120000", 5, &["Вадим"], 1)),
            5,
        );
        // In-progress: no session.toml, but labels present.
        write_session(&root, "2026-07-17_090000", None, 3);
        let list = list_sessions(&root).unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, "2026-07-17_090000", "sorted chronologically");
        assert!(list[0].meta.is_none(), "in-progress → no meta");
        assert_eq!(list[0].saved_frames_on_disk, 3);
        assert!(list[1].meta.is_some());
    }

    #[test]
    fn list_missing_root_is_empty_not_error() {
        let root = temp_root("empty");
        assert!(list_sessions(&root).unwrap().is_empty());
    }

    #[test]
    fn aggregate_sums_frames_and_unions_participants() {
        let root = temp_root("agg");
        write_session(
            &root,
            "2026-07-17_100000",
            Some(&meta("a", 10, &["Вадим"], 2)),
            10,
        );
        write_session(
            &root,
            "2026-07-17_110000",
            Some(&meta("b", 7, &["Маша"], 0)),
            7,
        );
        let st = aggregate(&list_sessions(&root).unwrap());
        assert_eq!(st.session_count, 2);
        assert_eq!(st.total_saved_frames, 17);
        assert_eq!(
            st.participants,
            BTreeSet::from(["Вадим".to_string(), "Маша".to_string()])
        );
        assert_eq!(st.anon_sessions, 1);
        assert_eq!(st.activities["typing"], 6);
    }

    #[test]
    fn prune_keep_last_n() {
        let ids: Vec<String> = ["s1", "s2", "s3", "s4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let (keep, remove) = partition_prune(&ids, &PrunePolicy::KeepLast(2));
        assert_eq!(keep, vec!["s3", "s4"]);
        assert_eq!(remove, vec!["s1", "s2"]);
        // Keeping more than exist removes nothing.
        let (k, r) = partition_prune(&ids, &PrunePolicy::KeepLast(10));
        assert_eq!(k.len(), 4);
        assert!(r.is_empty());
    }

    #[test]
    fn prune_keep_since_date() {
        let ids: Vec<String> = [
            "2026-07-15_100000",
            "2026-07-17_090000",
            "2026-07-18_120000",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let (keep, remove) = partition_prune(&ids, &PrunePolicy::KeepSince("2026-07-17".into()));
        assert_eq!(keep, vec!["2026-07-17_090000", "2026-07-18_120000"]);
        assert_eq!(remove, vec!["2026-07-15_100000"]);
    }
}
