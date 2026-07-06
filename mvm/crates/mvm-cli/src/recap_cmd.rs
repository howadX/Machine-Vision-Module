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

/// Default log source when `--input` is absent (phase 14.0.4): the same
/// data_dir a default `mvm run` writes to. An unreadable product config falls
/// back to cwd — recap is read-only, and `run`/`serve` already fail loudly.
fn default_input() -> PathBuf {
    crate::config_file::load().map(|p| p.data_dir).unwrap_or_else(|_| PathBuf::from("."))
}

pub fn parse_recap_args(raw: &[String]) -> RecapArgs {
    let mut a = RecapArgs { input: default_input(), events: None, from: None, to: None, last: None };
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
    recap_markdown(dir, None, None, None)
}

/// Programmatic recap entry (also used by the MCP `mvm_recap` tool, 12.D2):
/// recap markdown for `dir` with an optional window.
pub fn recap_markdown(
    dir: &Path,
    last: Option<String>,
    from: Option<String>,
    to: Option<String>,
) -> Result<String> {
    let args = RecapArgs { input: dir.to_path_buf(), events: None, from, to, last };
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
    resolve_files_at(&args.input, args.events.as_ref())
}

/// A dir → globbed `scene-*.jsonl` / `events-*.jsonl` (sorted); a file → itself
/// (+ optional explicit events). Shared by `recap` and `inspect`.
fn resolve_files_at(input: &Path, events: Option<&PathBuf>) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    if input.is_dir() {
        let mut scene = Vec::new();
        let mut ev = Vec::new();
        for entry in std::fs::read_dir(input)? {
            let p = entry?.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with("scene-") && name.ends_with(".jsonl") {
                scene.push(p.clone());
            } else if name.starts_with("events-") && name.ends_with(".jsonl") {
                ev.push(p.clone());
            }
        }
        scene.sort();
        ev.sort();
        if scene.is_empty() {
            bail!("no scene-*.jsonl in {}", input.display());
        }
        Ok((scene, ev))
    } else {
        let ev = events.cloned().into_iter().collect();
        Ok((vec![input.to_path_buf()], ev))
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

// ── inspect subcommand (phase 12.E1) — the "show me the raw slice" zoom ──────
// Level 3 of the pull model: when the recap/snapshot flags something, return the
// exact logged frames for a narrow window (± seconds around a ts) and/or a single
// track id — full logged precision, not aggregated. Compact markdown per-frame
// lines by default (matches the MCP `mvm_inspect` tool); `--raw` emits raw JSONL.

pub struct InspectArgs {
    input: PathBuf,
    around: Option<String>,
    window_s: f64,
    id: Option<u32>,
    md: bool,
}

pub fn parse_inspect_args(raw: &[String]) -> InspectArgs {
    // md defaults true — compact markdown, aligned with the MCP `mvm_inspect`
    // tool (phase 13.A4). `--raw`/`--json` opts back into raw JSONL.
    let mut a = InspectArgs { input: default_input(), around: None, window_s: 5.0, id: None, md: true };
    let mut i = 2; // skip program name + "inspect"
    while i < raw.len() {
        match raw[i].as_str() {
            "--input" => take(raw, &mut i, |v| a.input = PathBuf::from(v)),
            "--around" => take(raw, &mut i, |v| a.around = Some(v.to_string())),
            "--window" => take(raw, &mut i, |v| a.window_s = v.parse().unwrap_or(5.0)),
            "--id" => take(raw, &mut i, |v| a.id = v.parse().ok()),
            "--md" => a.md = true,
            "--raw" | "--json" => a.md = false,
            _ => {}
        }
        i += 1;
    }
    a
}

pub fn run_inspect(args: InspectArgs) -> Result<()> {
    print!("{}", inspect_collect(&args)?);
    Ok(())
}

/// Programmatic inspect entry (also the MCP `mvm_inspect` tool, 12.D2): the raw
/// (or `--md`) slice as a single newline-joined string.
pub fn inspect_output(
    input: &Path,
    around: Option<String>,
    window_s: f64,
    id: Option<u32>,
    md: bool,
) -> Result<String> {
    inspect_collect(&InspectArgs { input: input.to_path_buf(), around, window_s, id, md })
}

fn inspect_collect(args: &InspectArgs) -> Result<String> {
    let (scene_files, _events) = resolve_files_at(&args.input, None)?;
    let mut lines: Vec<(String, String)> = Vec::new();
    for f in &scene_files {
        let text = std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            if let Some(ts) = serde_json::from_str::<serde_json::Value>(line)
                .ok()
                .and_then(|v| v.get("ts").and_then(|x| x.as_str()).map(String::from))
            {
                lines.push((ts, line.to_string()));
            }
        }
    }
    lines.sort_by(|a, b| a.0.cmp(&b.0));

    let kept = inspect_select(&lines, args.around.as_deref(), args.window_s, args.id)?;
    if kept.is_empty() {
        bail!("no frames in the selected slice");
    }
    let mut out = String::new();
    for raw in &kept {
        if args.md {
            out.push_str(&inspect_md_line(raw));
        } else {
            out.push_str(raw);
        }
        out.push('\n');
    }
    Ok(out)
}

/// Pure selection core (testable): keep raw lines within `window_s` of `around`
/// (if given) that contain face `id` (if given). Returns the raw lines in order.
fn inspect_select(
    lines: &[(String, String)],
    around: Option<&str>,
    window_s: f64,
    id: Option<u32>,
) -> Result<Vec<String>> {
    let around_dt = around
        .map(|a| DateTime::parse_from_rfc3339(a).with_context(|| format!("parse --around {a}")))
        .transpose()?;
    let mut out = Vec::new();
    for (ts, raw) in lines {
        if let Some(adt) = around_dt {
            let dt = DateTime::parse_from_rfc3339(ts).with_context(|| format!("parse ts {ts}"))?;
            let diff = (dt - adt).num_milliseconds().abs() as f64 / 1000.0;
            if diff > window_s {
                continue;
            }
        }
        if let Some(want) = id {
            let has = serde_json::from_str::<serde_json::Value>(raw)
                .ok()
                .and_then(|v| {
                    v.get("faces").and_then(|f| f.as_array()).map(|arr| {
                        arr.iter()
                            .any(|f| f.get("id").and_then(|x| x.as_u64()) == Some(want as u64))
                    })
                })
                .unwrap_or(false);
            if !has {
                continue;
            }
        }
        out.push(raw.clone());
    }
    Ok(out)
}

/// A compact one-line-per-frame view of a raw scene line (`--md`).
fn inspect_md_line(raw: &str) -> String {
    let v: serde_json::Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return raw.to_string(),
    };
    let ts = v.get("ts").and_then(|x| x.as_str()).unwrap_or("?");
    let people = v
        .get("faces")
        .and_then(|f| f.as_array())
        .map(|arr| {
            arr.iter()
                .map(|f| {
                    let id = f.get("id").and_then(|x| x.as_u64()).unwrap_or(0);
                    let who = f.get("identity").and_then(|x| x.as_str()).unwrap_or("anon");
                    let dist = f
                        .get("dist_m")
                        .and_then(|x| x.as_f64())
                        .map(|d| format!("{d:.1}m"))
                        .unwrap_or_else(|| "—".into());
                    let posture = f.get("posture").and_then(|x| x.as_str()).unwrap_or("—");
                    let emo = f.get("emotion").and_then(|x| x.as_str()).unwrap_or("—");
                    format!("id{id} {who} {dist} {posture} {emo}")
                })
                .collect::<Vec<_>>()
                .join(" | ")
        })
        .unwrap_or_default();
    format!("{} · {}", hms(ts), people)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(ts: &str, id: u32) -> (String, String) {
        (ts.into(), format!(r#"{{"ts":"{ts}","faces":[{{"id":{id}}}]}}"#))
    }

    #[test]
    fn inspect_window_around_ts() {
        let lines = vec![
            line("2026-07-05T18:55:00+03:00", 0),
            line("2026-07-05T18:55:03+03:00", 0),
            line("2026-07-05T18:55:10+03:00", 0),
        ];
        let kept = inspect_select(&lines, Some("2026-07-05T18:55:00+03:00"), 5.0, None).unwrap();
        assert_eq!(kept.len(), 2, "±5s keeps t+0 and t+3, drops t+10");
    }

    #[test]
    fn inspect_filters_by_track_id() {
        let lines = vec![
            line("2026-07-05T18:55:00+03:00", 0),
            line("2026-07-05T18:55:01+03:00", 12),
            line("2026-07-05T18:55:02+03:00", 0),
        ];
        let kept = inspect_select(&lines, None, 5.0, Some(12)).unwrap();
        assert_eq!(kept.len(), 1);
        assert!(kept[0].contains("\"id\":12"));
    }

    /// Phase 13.A4: `inspect` defaults to compact markdown (aligned with the MCP
    /// `mvm_inspect` tool, whose `md` defaults true); `--raw`/`--json` opts out.
    #[test]
    fn inspect_md_defaults_true_and_raw_opts_out() {
        let base: Vec<String> = ["mvm", "inspect"].iter().map(|s| s.to_string()).collect();
        assert!(parse_inspect_args(&base).md, "markdown is the default");
        let raw: Vec<String> = ["mvm", "inspect", "--raw"].iter().map(|s| s.to_string()).collect();
        assert!(!parse_inspect_args(&raw).md, "--raw switches to raw JSONL");
        let json: Vec<String> = ["mvm", "inspect", "--json"].iter().map(|s| s.to_string()).collect();
        assert!(!parse_inspect_args(&json).md, "--json is an alias of --raw");
    }

    #[test]
    fn inspect_md_line_is_compact() {
        let raw = r#"{"ts":"2026-07-05T18:55:19+03:00","faces":[{"id":0,"identity":"Вадим","dist_m":1.02,"posture":"standing","emotion":"Neutral"}]}"#;
        assert_eq!(inspect_md_line(raw), "18:55:19 · id0 Вадим 1.0m standing Neutral");
    }
}
