//! `mvm dataset <list|info|stats|prune|export>` (§16.3) — inspect and maintain the
//! Dataset Factory sessions on disk. Thin CLI over the pure readers in
//! `mvm-dataset::session`; formatting only, no session logic here.

use crate::config_file::ProductConfig;
use mvm_dataset::session::{
    DatasetStats, PrunePolicy, SessionInfo, aggregate, list_sessions, partition_prune,
};
use mvm_dataset::validate::{DatasetReport, SessionReport, Severity, check_dataset, check_session};
use std::path::{Path, PathBuf};

const USAGE: &str = "\
mvm dataset — inspect & maintain Dataset Factory sessions

USAGE:
    mvm dataset list                 list recorded sessions
    mvm dataset info <session_id>    details for one session
    mvm dataset stats                dataset-wide totals
    mvm dataset check [session_id]   data-quality gate (--strict fails on warnings)
    mvm dataset prune --keep <N|Nd>  remove old sessions (dry-run unless --yes)
    mvm dataset export ...           → Python exporters (phase 16.4)
";

/// Dispatch `mvm dataset <sub>`.
pub fn run_dataset(product: &ProductConfig, raw: &[String]) -> anyhow::Result<()> {
    let root = PathBuf::from(&product.dataset.dir);
    match raw.get(2).map(String::as_str) {
        Some("list") => cmd_list(&root),
        Some("info") => cmd_info(&root, raw.get(3).map(String::as_str)),
        Some("stats") => cmd_stats(&root),
        Some("check") => cmd_check(&root, raw),
        Some("prune") => cmd_prune(&root, raw),
        Some("export") => {
            cmd_export(&root);
            Ok(())
        }
        _ => {
            print!("{USAGE}");
            Ok(())
        }
    }
}

/// One-line participants summary: `Вадим +2 anon` / `— +1 anon` / `—`.
fn participants_line(si: &SessionInfo) -> String {
    match &si.meta {
        None => "recording…".to_string(),
        Some(m) => {
            let mut parts = Vec::new();
            if !m.participants.known.is_empty() {
                parts.push(m.participants.known.join(", "));
            }
            if m.participants.anonymous > 0 {
                parts.push(format!("+{} anon", m.participants.anonymous));
            }
            if parts.is_empty() {
                "—".to_string()
            } else {
                parts.join(" ")
            }
        }
    }
}

fn cmd_list(root: &Path) -> anyhow::Result<()> {
    let sessions = list_sessions(root)?;
    if sessions.is_empty() {
        println!("No sessions under {}", root.join("sessions").display());
        return Ok(());
    }
    println!(
        "{:<20}  {:>6}  {:<9}  {}",
        "SESSION", "FRAMES", "STATUS", "PARTICIPANTS"
    );
    for si in &sessions {
        let status = if si.meta.is_some() {
            "closed"
        } else {
            "recording"
        };
        let frames = si
            .meta
            .as_ref()
            .map(|m| m.session.saved_frames)
            .unwrap_or(si.saved_frames_on_disk);
        println!(
            "{:<20}  {:>6}  {:<9}  {}",
            si.id,
            frames,
            status,
            participants_line(si)
        );
    }
    println!("\n{} session(s) · {}", sessions.len(), root.display());
    Ok(())
}

fn cmd_info(root: &Path, id: Option<&str>) -> anyhow::Result<()> {
    let Some(id) = id else {
        anyhow::bail!("usage: mvm dataset info <session_id>");
    };
    let dir = root.join("sessions").join(id);
    if !dir.is_dir() {
        anyhow::bail!("no session {id} under {}", root.join("sessions").display());
    }
    let frames_on_disk = mvm_dataset::session::labels_line_count(&dir);
    println!("Session {id}");
    println!("  path:   {}", dir.display());
    println!("  frames: {frames_on_disk} on disk");
    match mvm_dataset::session::read_meta(&dir)? {
        None => println!("  status: recording (no session.toml yet)"),
        Some(m) => {
            println!("  status: closed");
            println!("  span:   {} → {}", m.session.started, m.session.ended);
            println!("  saved:  {} frames", m.session.saved_frames);
            let anon = if m.participants.anonymous > 0 {
                format!(" +{} anon", m.participants.anonymous)
            } else {
                String::new()
            };
            println!("  people: {}{anon}", m.participants.known.join(", "));
            if !m.activities.is_empty() {
                let mut acts: Vec<_> = m.activities.iter().collect();
                acts.sort_by(|a, b| b.1.cmp(a.1));
                let s: Vec<String> = acts.iter().map(|(k, c)| format!("{k}×{c}")).collect();
                println!("  activities: {}", s.join(", "));
            }
            if !m.objects.top.is_empty() {
                println!("  objects: {}", m.objects.top.join(", "));
            }
            println!(
                "  anomaly: max {:.3} · mean {:.3}",
                m.anomalies.max, m.anomalies.mean
            );
        }
    }
    Ok(())
}

fn cmd_stats(root: &Path) -> anyhow::Result<()> {
    let st: DatasetStats = aggregate(&list_sessions(root)?);
    println!("Dataset {}", root.display());
    println!("  sessions:     {}", st.session_count);
    println!("  saved frames: {}", st.total_saved_frames);
    println!(
        "  participants: {} ({})",
        st.participants.len(),
        if st.participants.is_empty() {
            "—".into()
        } else {
            st.participants
                .iter()
                .cloned()
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!("  anon sessions: {}", st.anon_sessions);
    if !st.activities.is_empty() {
        let mut acts: Vec<_> = st.activities.iter().collect();
        acts.sort_by(|a, b| b.1.cmp(a.1));
        let s: Vec<String> = acts
            .iter()
            .take(8)
            .map(|(k, c)| format!("{k}×{c}"))
            .collect();
        println!("  activities:   {}", s.join(", "));
    }
    println!("  anomaly max:  {:.3}", st.anomaly_max);
    Ok(())
}

/// `mvm dataset check [session_id] [--strict]` — data-quality gate (§16.6-A).
/// Prints findings + quality stats, then fails (non-zero exit) when the gate does:
/// any error, or — with `--strict` — any warning too. CI-gateable.
fn cmd_check(root: &Path, raw: &[String]) -> anyhow::Result<()> {
    let strict = raw.iter().any(|a| a == "--strict");
    // First non-flag positional after `check` is an optional session id.
    let session_id = raw
        .get(3..)
        .and_then(|rest| rest.iter().find(|a| !a.starts_with('-')));

    let report: DatasetReport = if let Some(id) = session_id {
        let dir = root.join("sessions").join(id);
        if !dir.is_dir() {
            anyhow::bail!("no session {id} under {}", root.join("sessions").display());
        }
        DatasetReport {
            sessions: vec![check_session(&dir, id)],
        }
    } else {
        check_dataset(root)?
    };

    if report.sessions.is_empty() {
        println!(
            "No sessions under {} — nothing to check.",
            root.join("sessions").display()
        );
        return Ok(());
    }
    for sr in &report.sessions {
        print_session_report(sr);
    }

    let (errs, warns) = (report.errors(), report.warns());
    println!(
        "\n{} session(s) · {errs} error(s) · {warns} warning(s)",
        report.sessions.len()
    );

    if errs > 0 {
        anyhow::bail!("data-quality check FAILED: {errs} error(s)");
    }
    if strict && warns > 0 {
        anyhow::bail!("data-quality check FAILED (--strict): {warns} warning(s)");
    }
    println!("OK — data-quality check passed.");
    Ok(())
}

fn print_session_report(sr: &SessionReport) {
    let st = &sr.stats;
    println!("── {} ──", sr.id);
    for f in &sr.findings {
        let tag = match f.severity {
            Severity::Error => "E",
            Severity::Warn => "W",
        };
        match f.frame_id {
            Some(fid) => println!("  [{tag}] {:<22} frame {fid}: {}", f.code, f.msg),
            None => println!("  [{tag}] {:<22} {}", f.code, f.msg),
        }
    }
    println!(
        "  stats: {} frames · {} people ({} gallery / {} anon) · {} objects · {} masks · {} depth",
        st.frames,
        st.people,
        st.gallery_people,
        st.anon_people,
        st.objects,
        st.masks,
        st.with_depth
    );
    println!(
        "         withheld-posture {:.0}% · postures {} · emotions {} · objects {}",
        st.withheld_rate() * 100.0,
        fmt_hist(&st.posture_hist),
        fmt_hist(&st.emotion_hist),
        fmt_hist(&st.object_hist),
    );
}

/// Compact `a×3 b×1` rendering of a class histogram (desc by count), `—` if empty.
fn fmt_hist(h: &std::collections::BTreeMap<String, u64>) -> String {
    if h.is_empty() {
        return "—".to_string();
    }
    let mut v: Vec<(&String, &u64)> = h.iter().collect();
    v.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
    v.iter()
        .take(6)
        .map(|(k, c)| format!("{k}×{c}"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn cmd_prune(root: &Path, raw: &[String]) -> anyhow::Result<()> {
    // Parse --keep <N|Nd> and --yes.
    let mut keep: Option<String> = None;
    let mut confirmed = false;
    let mut i = 3;
    while i < raw.len() {
        match raw[i].as_str() {
            "--keep" => {
                keep = raw.get(i + 1).cloned();
                i += 1;
            }
            "--yes" | "-y" => confirmed = true,
            _ => {}
        }
        i += 1;
    }
    let Some(keep) = keep else {
        anyhow::bail!("usage: mvm dataset prune --keep <N|Nd> [--yes]");
    };
    let policy = parse_keep(&keep)?;

    let sessions = list_sessions(root)?;
    let ids: Vec<String> = sessions.iter().map(|s| s.id.clone()).collect();
    let (_, remove) = partition_prune(&ids, &policy);

    if remove.is_empty() {
        println!(
            "Nothing to prune ({} session(s), policy keeps all).",
            ids.len()
        );
        return Ok(());
    }
    if !confirmed {
        println!(
            "Would remove {} session(s) (dry-run — pass --yes to delete):",
            remove.len()
        );
        for id in &remove {
            println!("  {id}");
        }
        return Ok(());
    }
    for id in &remove {
        let dir = root.join("sessions").join(id);
        std::fs::remove_dir_all(&dir)
            .map_err(|e| anyhow::anyhow!("remove {}: {e}", dir.display()))?;
        println!("removed {id}");
    }
    println!("Pruned {} session(s).", remove.len());
    Ok(())
}

/// ML export lives in Python (PyTorch/TF ecosystems); Rust never links them
/// (invariant #2). Print ready-to-run commands with the real dataset dir filled in.
fn cmd_export(root: &Path) {
    let root = root.display();
    println!(
        "ML export runs in Python (prototype/pipeline/dataset/) — Rust core stays framework-free.\n"
    );
    println!("From the repo:\n  cd prototype/pipeline/dataset\n");
    println!("PyTorch (smoke-iterate the dataset):");
    println!("  python export_pytorch.py {root} --config dataset_config.yaml --limit 8\n");
    println!("WebDataset shards (no extra deps to write):");
    println!("  python export_webdataset.py {root} /tmp/mvm_wds --config dataset_config.yaml\n");
    println!("TFRecord shards (needs tensorflow):");
    println!("  python export_tfrecord.py {root} /tmp/mvm_tfr --config dataset_config.yaml\n");
    println!("Filters (task presets) live in dataset_config.yaml. See the README there.");
}

/// Parse `--keep` value: `5` → keep last 5; `7d` → keep sessions on/after today−7d.
fn parse_keep(v: &str) -> anyhow::Result<PrunePolicy> {
    if let Some(days) = v.strip_suffix('d') {
        let n: i64 = days
            .parse()
            .map_err(|_| anyhow::anyhow!("bad --keep days: {v}"))?;
        let cutoff = chrono::Local::now().date_naive() - chrono::Duration::days(n);
        Ok(PrunePolicy::KeepSince(
            cutoff.format("%Y-%m-%d").to_string(),
        ))
    } else {
        let n: usize = v
            .parse()
            .map_err(|_| anyhow::anyhow!("bad --keep count: {v}"))?;
        Ok(PrunePolicy::KeepLast(n))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_keep_count_and_days() {
        assert_eq!(parse_keep("5").unwrap(), PrunePolicy::KeepLast(5));
        match parse_keep("7d").unwrap() {
            PrunePolicy::KeepSince(d) => assert_eq!(d.len(), 10, "YYYY-MM-DD cutoff, got {d}"),
            _ => panic!("7d should be a KeepSince policy"),
        }
        assert!(parse_keep("xx").is_err());
    }
}
