//! `recap` subcommand (phase 12.C3) — offline session summary over the durable
//! logs (`scene-*.jsonl` + `events-*.jsonl`, phase 12.C0/C1).
//!
//! Parses the JSONL into the lightweight observations [`mvm_core::recap`] expects,
//! optionally windows them (`--from`/`--to`/`--last`), builds the [`Recap`], and
//! prints the markdown from [`crate::render::render_recap`]. All the aggregation
//! is pure arithmetic in `mvm-core`; this file only does IO + parsing + windowing.

use anyhow::{bail, Context, Result};
use chrono::DateTime;
use mvm_core::recap::{EpisodeObs, FrameObs, PersonObs, Recap};
use std::path::{Path, PathBuf};

pub struct RecapArgs {
    /// A directory holding `scene-*.jsonl`/`events-*.jsonl`, or a single scene file.
    input: PathBuf,
    /// Explicit events file (when `input` is a single file).
    events: Option<PathBuf>,
    /// Inclusive ISO8601 window bounds.
    from: Option<String>,
    to: Option<String>,
    /// Keep only the last <dur> before the latest entry (e.g. `10m`, `90s`, `2h`).
    last: Option<String>,
}

pub fn parse_recap_args(raw: &[String]) -> RecapArgs {
    let mut a = RecapArgs { input: PathBuf::from("."), events: None, from: None, to: None, last: None };
    let mut i = 2; // skip program name + "recap"
    while i < raw.len() {
        match raw[i].as_str() {
            "--input" => take(raw, &mut i, |v| a.input = PathBuf::from(v)),
            "--events" => take(raw, &mut i, |v| a.events = Some(PathBuf::from(v))),
            "--from" => take(raw, &mut i, |v| a.from = Some(v.to_string())),
            "--to" => take(raw, &mut i, |v| a.to = Some(v.to_string())),
            "--last" => take(raw, &mut i, |v| a.last = Some(v.to_string())),
            _ => {}
        }
        i += 1;
    }
    a
}

fn take(raw: &[String], i: &mut usize, mut set: impl FnMut(&str)) {
    if *i + 1 < raw.len() {
        set(&raw[*i + 1]);
        *i += 1;
    }
}

pub fn run_recap(args: RecapArgs) -> Result<()> {
    print!("{}", render_recap_string(&args)?);
    Ok(())
}

/// Build + render the recap markdown for a whole log directory, no window — the
/// rolling `journal.md` view of the current day (phase 12.D1).
pub fn recap_markdown_from_dir(dir: &Path) -> Result<String> {
    let args = RecapArgs {
        input: dir.to_path_buf(),
        events: None,
        from: None,
        to: None,
        last: None,
    };
    render_recap_string(&args)
}

/// Parse the logs, window them, build the recap and render markdown (shared by
/// the `recap` subcommand and the rolling journal).
fn render_recap_string(args: &RecapArgs) -> Result<String> {
    let (scene_files, events_files) = resolve_files(args)?;

    let mut frames: Vec<(String, FrameObs)> = Vec::new();
    for f in &scene_files {
        let text = std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if let Some(obs) = parse_frame(line) {
                frames.push(obs);
            }
        }
    }
    let mut episodes: Vec<(String, EpisodeObs)> = Vec::new();
    for f in &events_files {
        let text = std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if let Some(obs) = parse_episode(line) {
                episodes.push(obs);
            }
        }
    }

    // Sort by ts so windowing + presence order are stable regardless of file order.
    frames.sort_by(|a, b| a.0.cmp(&b.0));
    episodes.sort_by(|a, b| a.0.cmp(&b.0));

    let window = resolve_window(args, &frames)?;
    let kept_frames: Vec<FrameObs> =
        frames.iter().filter(|(ts, _)| in_window(ts, &window)).map(|(_, o)| o.clone()).collect();
    let kept_eps: Vec<EpisodeObs> =
        episodes.iter().filter(|(ts, _)| in_window(ts, &window)).map(|(_, o)| o.clone()).collect();

    if kept_frames.is_empty() && kept_eps.is_empty() {
        bail!("no log entries in the selected window (checked {} scene files)", scene_files.len());
    }

    let recap = Recap::build(&kept_frames, &kept_eps);
    let title = window_title(&frames, &window);
    Ok(crate::render::render_recap(&recap, &title))
}

/// Resolve the scene/events files from `--input` (dir → glob; file → itself).
fn resolve_files(args: &RecapArgs) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    if args.input.is_dir() {
        let mut scene = Vec::new();
        let mut events = Vec::new();
        for entry in std::fs::read_dir(&args.input)? {
            let p = entry?.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("scene-") && name.ends_with(".jsonl") {
                scene.push(p.clone());
            } else if name.starts_with("events-") && name.ends_with(".jsonl") {
                events.push(p.clone());
            }
        }
        scene.sort();
        events.sort();
        if scene.is_empty() {
            bail!("no scene-*.jsonl in {}", args.input.display());
        }
        Ok((scene, events))
    } else {
        let events = args.events.clone().into_iter().collect();
        Ok((vec![args.input.clone()], events))
    }
}

fn parse_frame(line: &str) -> Option<(String, FrameObs)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let ts = v.get("ts")?.as_str()?.to_string();
    let people = v
        .get("faces")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|f| {
                    Some(PersonObs {
                        id: f.get("id")?.as_u64()? as u32,
                        dist_m: f.get("dist_m").and_then(|x| x.as_f64()).map(|x| x as f32),
                        posture: f.get("posture").and_then(|x| x.as_str()).map(String::from),
                        identity: f.get("identity").and_then(|x| x.as_str()).map(String::from),
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    let object_classes = distinct_strings(v.get("objects"), "class");
    let zone_labels = distinct_strings(v.get("stable_zones"), "label");
    let anomaly = v.get("anomaly_score").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32;
    Some((ts, FrameObs { people, object_classes, zone_labels, anomaly }))
}

/// Distinct string values of `field` across an array node (dedup within a frame).
fn distinct_strings(node: Option<&serde_json::Value>, field: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(arr) = node.and_then(|n| n.as_array()) {
        for item in arr {
            if let Some(s) = item.get(field).and_then(|x| x.as_str()) {
                if !out.iter().any(|e| e == s) {
                    out.push(s.to_string());
                }
            }
        }
    }
    out
}

fn parse_episode(line: &str) -> Option<(String, EpisodeObs)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    let ts = v.get("ts")?.as_str()?.to_string();
    let participants = v
        .get("participants")?
        .as_array()?
        .iter()
        .filter_map(|x| x.as_u64().map(|n| n as u32))
        .collect();
    Some((
        ts,
        EpisodeObs {
            label: v.get("label")?.as_str()?.to_string(),
            participants,
            duration_s: v.get("duration_s").and_then(|x| x.as_f64()).unwrap_or(0.0),
            peak_confidence: v.get("peak_confidence").and_then(|x| x.as_f64()).unwrap_or(0.0) as f32,
            stable_participants: v.get("stable_participants").and_then(|x| x.as_bool()).unwrap_or(false),
        },
    ))
}

/// A resolved [from, to] window as raw ISO strings (lexicographic-comparable
/// because they share the same UTC offset in one session).
struct Window {
    from: Option<String>,
    to: Option<String>,
}

fn resolve_window(args: &RecapArgs, frames: &[(String, FrameObs)]) -> Result<Window> {
    if let Some(dur) = &args.last {
        let secs = parse_duration_secs(dur)?;
        // Anchor to the latest frame ts (deterministic + meaningful offline).
        let latest = frames.last().map(|(ts, _)| ts.clone());
        let from = match &latest {
            Some(ts) => {
                let dt = DateTime::parse_from_rfc3339(ts)
                    .with_context(|| format!("parse ts {ts}"))?;
                let cutoff = dt - chrono::Duration::seconds(secs);
                Some(cutoff.format("%Y-%m-%dT%H:%M:%S%:z").to_string())
            }
            None => None,
        };
        Ok(Window { from, to: None })
    } else {
        Ok(Window { from: args.from.clone(), to: args.to.clone() })
    }
}

fn in_window(ts: &str, w: &Window) -> bool {
    if let Some(f) = &w.from {
        if ts < f.as_str() {
            return false;
        }
    }
    if let Some(t) = &w.to {
        if ts > t.as_str() {
            return false;
        }
    }
    true
}

fn parse_duration_secs(s: &str) -> Result<i64> {
    let s = s.trim();
    let (num, mult) = if let Some(n) = s.strip_suffix('h') {
        (n, 3600)
    } else if let Some(n) = s.strip_suffix('m') {
        (n, 60)
    } else if let Some(n) = s.strip_suffix('s') {
        (n, 1)
    } else {
        (s, 1)
    };
    let n: i64 = num.trim().parse().with_context(|| format!("bad duration '{s}'"))?;
    Ok(n * mult)
}

/// A time-of-day span label from the kept frames (e.g. `18:55:01–18:57:39`).
fn window_title(frames: &[(String, FrameObs)], w: &Window) -> String {
    let kept: Vec<&String> = frames.iter().map(|(ts, _)| ts).filter(|ts| in_window(ts, w)).collect();
    match (kept.first(), kept.last()) {
        (Some(a), Some(b)) => format!("{}–{}", hms(a), hms(b)),
        _ => "—".to_string(),
    }
}

/// The `HH:MM:SS` slice of an ISO ts (`2026-07-05T18:55:01+03:00` → `18:55:01`).
fn hms(ts: &str) -> &str {
    ts.get(11..19).unwrap_or(ts)
}
