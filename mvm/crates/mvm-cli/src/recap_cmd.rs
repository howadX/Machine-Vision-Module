//! `recap` subcommand (phase 12.C3) — offline session summary over the durable
//! logs (`scene-*.jsonl` + `events-*.jsonl`, phase 12.C0/C1) **or** semantic
//! session `labels.jsonl` files (phase 16.7).
//!
//! Parses the JSONL into the lightweight observations [`mvm_core::recap`] expects,
//! optionally windows them (`--from`/`--to`/`--last`), builds the [`Recap`], and
//! prints the markdown from [`crate::render::render_recap`]. All the aggregation
//! is pure arithmetic in `mvm-core`; this file only does IO + parsing + windowing.
//!
//! Two sources feed the same aggregation (see [`Source`]): the flat per-day logs,
//! and semantic sessions whose observations are *projected* to the flat scene/event
//! shape ([`scene_value_from_semantic`]) so recap/inspect stay one code path. A
//! directory prefers flat logs when present and falls back to semantic sessions —
//! keeping recap working once phase 3 makes text-only sessions the default.

use anyhow::{Context, Result, bail};
use chrono::DateTime;
use mvm_core::recap::{EpisodeObs, FrameObs, PersonObs, Recap};
use std::path::{Path, PathBuf};

type TimestampedFrameObs = (String, FrameObs);
type TimestampedEpisodeObs = (String, EpisodeObs);
type CollectedObservations = (Vec<TimestampedFrameObs>, Vec<TimestampedEpisodeObs>);

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
    crate::config_file::load()
        .map(|p| p.data_dir)
        .unwrap_or_else(|_| PathBuf::from("."))
}

pub fn parse_recap_args(raw: &[String]) -> RecapArgs {
    let mut a = RecapArgs {
        input: default_input(),
        events: None,
        from: None,
        to: None,
        last: None,
    };
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
    let args = RecapArgs {
        input: dir.to_path_buf(),
        events: None,
        from,
        to,
        last,
    };
    render_recap_string(&args)
}

/// Parse the logs, window them, build the recap and render markdown (shared by
/// the `recap` subcommand and the rolling journal).
fn render_recap_string(args: &RecapArgs) -> Result<String> {
    let source = resolve_source(&args.input, args.events.as_ref())?;
    let (mut frames, mut episodes) = collect_frames_and_episodes(&source)?;

    // Sort by ts so windowing + presence order are stable regardless of file order.
    frames.sort_by(|a, b| a.0.cmp(&b.0));
    episodes.sort_by(|a, b| a.0.cmp(&b.0));

    let window = resolve_window(args, &frames)?;
    let kept_frames: Vec<FrameObs> = frames
        .iter()
        .filter(|(ts, _)| in_window(ts, &window))
        .map(|(_, o)| o.clone())
        .collect();
    let kept_eps: Vec<EpisodeObs> = episodes
        .iter()
        .filter(|(ts, _)| in_window(ts, &window))
        .map(|(_, o)| o.clone())
        .collect();

    if kept_frames.is_empty() && kept_eps.is_empty() {
        bail!(
            "no log entries in the selected window ({})",
            source.describe()
        );
    }

    let recap = Recap::build(&kept_frames, &kept_eps);
    let title = window_title(&frames, &window);
    Ok(crate::render::render_recap(&recap, &title))
}

/// Where recap/inspect read frames from — flat per-day logs (`scene-*.jsonl` +
/// `events-*.jsonl`) or semantic session `labels.jsonl` files (phase 16.7).
enum Source {
    Flat {
        scene: Vec<PathBuf>,
        events: Vec<PathBuf>,
    },
    Semantic {
        labels: Vec<PathBuf>,
    },
}

impl Source {
    fn describe(&self) -> String {
        match self {
            Source::Flat { scene, .. } => format!("{} scene file(s)", scene.len()),
            Source::Semantic { labels } => format!("{} semantic session(s)", labels.len()),
        }
    }
}

/// Resolve `--input` to a [`Source`]. A single `labels.jsonl` → semantic; a single
/// other file → a flat scene file. A directory prefers semantic sessions
/// (`labels.jsonl` directly, or under `sessions/` / `datasets/sessions/`) and
/// falls back to flat logs only when no semantic sessions exist. This prevents
/// retired flat logs from shadowing the current default-on session store.
fn resolve_source(input: &Path, events: Option<&PathBuf>) -> Result<Source> {
    if input.is_file() {
        if input.file_name().and_then(|n| n.to_str()) == Some("labels.jsonl") {
            return Ok(Source::Semantic {
                labels: vec![input.to_path_buf()],
            });
        }
        return Ok(Source::Flat {
            scene: vec![input.to_path_buf()],
            events: events.cloned().into_iter().collect(),
        });
    }
    let labels = find_session_labels(input);
    if !labels.is_empty() {
        return Ok(Source::Semantic { labels });
    }
    let (scene, ev) = glob_flat(input)?;
    if !scene.is_empty() {
        return Ok(Source::Flat { scene, events: ev });
    }
    bail!(
        "no scene-*.jsonl or semantic sessions (labels.jsonl) in {}",
        input.display()
    );
}

/// Globbed `scene-*.jsonl` / `events-*.jsonl` in `dir` (sorted; may be empty).
fn glob_flat(dir: &Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>)> {
    let mut scene = Vec::new();
    let mut ev = Vec::new();
    for entry in std::fs::read_dir(dir)? {
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
    Ok((scene, ev))
}

/// Semantic session `labels.jsonl` files reachable from `dir`: the dir itself (a
/// single session), or every session under `dir/sessions/` or
/// `dir/datasets/sessions/` (the data_dir layout). Sorted; empty when none.
fn find_session_labels(dir: &Path) -> Vec<PathBuf> {
    let direct = dir.join("labels.jsonl");
    if direct.is_file() {
        return vec![direct];
    }
    for sub in ["sessions", "datasets/sessions"] {
        let base = dir.join(sub);
        if !base.is_dir() {
            continue;
        }
        let mut out = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&base) {
            for e in rd.flatten() {
                let l = e.path().join("labels.jsonl");
                if l.is_file() {
                    out.push(l);
                }
            }
        }
        out.sort();
        if !out.is_empty() {
            return out;
        }
    }
    Vec::new()
}

/// Read a [`Source`] into timestamped frame + episode observations. The semantic
/// path projects each observation into the flat scene/event shape and reuses the
/// exact same parsers — so a recap over a semantic session equals a recap over the
/// flat logs of the same frames (see the golden-parity test).
fn collect_frames_and_episodes(src: &Source) -> Result<CollectedObservations> {
    let mut frames: Vec<TimestampedFrameObs> = Vec::new();
    let mut episodes: Vec<TimestampedEpisodeObs> = Vec::new();
    match src {
        Source::Flat { scene, events } => {
            for f in scene {
                let text =
                    std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    if let Some(o) = parse_frame(line) {
                        frames.push(o);
                    }
                }
            }
            for f in events {
                let text =
                    std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    if let Some(o) = parse_episode(line) {
                        episodes.push(o);
                    }
                }
            }
        }
        Source::Semantic { labels } => {
            for f in labels {
                let text =
                    std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let Ok(obs) = serde_json::from_str::<serde_json::Value>(line) else {
                        continue;
                    };
                    if let Some(sv) = scene_value_from_semantic(&obs)
                        && let Some(o) = parse_frame_value(&sv)
                    {
                        frames.push(o);
                    }
                    for ev in event_values_from_semantic(&obs) {
                        if let Some(o) = parse_episode_value(&ev) {
                            episodes.push(o);
                        }
                    }
                }
            }
        }
    }
    Ok((frames, episodes))
}

// ── semantic session shim (phase 16.7) ───────────────────────────────────────
// A semantic `labels.jsonl` line (Canonical Observation) is projected to the flat
// scene/event JSON shape the recap + inspect readers already understand, so the
// aggregation stays a single, parity-tested code path.

/// Numeric id from an `entity_id` string (`"p12"`→12, `"o10001"`→10001).
fn entity_num(s: &str) -> Option<u64> {
    s.get(1..)?.parse().ok()
}

/// Project a semantic observation into the flat scene-JSON shape (ts + faces +
/// objects + stable_zones + anomaly_score) consumed by [`parse_frame_value`] and
/// [`inspect_md_line`]. `None` when the line lacks a frame timestamp.
fn scene_value_from_semantic(obs: &serde_json::Value) -> Option<serde_json::Value> {
    let ts = obs.get("frame")?.get("timestamp")?.as_str()?;
    let entities = obs.get("entities").and_then(|e| e.as_array());
    let attr_label = |attrs: Option<&serde_json::Value>, key: &str| -> Option<String> {
        attrs?.get(key)?.get("label")?.as_str().map(String::from)
    };

    let mut faces = Vec::new();
    let mut objects = Vec::new();
    for e in entities.into_iter().flatten() {
        let attrs = e.get("attributes");
        match e.get("kind").and_then(|k| k.as_str()).unwrap_or("") {
            "person" => {
                let id = e
                    .get("entity_id")
                    .and_then(|x| x.as_str())
                    .and_then(entity_num)
                    .unwrap_or(0);
                let mut face = serde_json::Map::new();
                face.insert("id".into(), serde_json::json!(id));
                if let Some(name) = attr_label(attrs, "identity") {
                    face.insert("identity".into(), serde_json::json!(name));
                }
                if let Some(post) = attr_label(attrs, "posture") {
                    face.insert("posture".into(), serde_json::json!(post));
                }
                if let Some(emo) = attr_label(attrs, "emotion") {
                    face.insert("emotion".into(), serde_json::json!(emo));
                }
                // dist_m mirrors the flat log: the Z of the 3D position (not Euclidean).
                if let Some(z) = attrs
                    .and_then(|a| a.get("position_3d"))
                    .and_then(|p| p.get("z"))
                    .and_then(|x| x.as_f64())
                {
                    face.insert("dist_m".into(), serde_json::json!(z));
                }
                faces.push(serde_json::Value::Object(face));
            }
            "object" => {
                if let Some(class) = attrs.and_then(|a| a.get("class")).and_then(|x| x.as_str()) {
                    objects.push(serde_json::json!({ "class": class }));
                }
            }
            _ => {}
        }
    }

    // Zone records with a label → `stable_zones[{label}]`; anonymous null-label
    // components carry no zone label (matches the flat `stable_zones` contract).
    let mut stable_zones = Vec::new();
    if let Some(rels) = obs.get("relations").and_then(|r| r.as_array()) {
        for r in rels {
            if r.get("kind").and_then(|k| k.as_str()) == Some("zone")
                && let Some(label) = r.get("label").and_then(|x| x.as_str())
            {
                stable_zones.push(serde_json::json!({ "label": label }));
            }
        }
    }

    let anomaly = obs
        .get("quality")
        .and_then(|q| q.get("anomaly_score"))
        .and_then(|x| x.as_f64())
        .unwrap_or(0.0);
    Some(serde_json::json!({
        "ts": ts,
        "faces": faces,
        "objects": objects,
        "stable_zones": stable_zones,
        "anomaly_score": anomaly,
    }))
}

/// Extract completed episodes inlined in a semantic observation as flat
/// `events-*.jsonl`-shaped values (ts from the frame; participants→u32; the
/// semantic `stable` field → the flat `stable_participants`).
fn event_values_from_semantic(obs: &serde_json::Value) -> Vec<serde_json::Value> {
    let ts = obs
        .get("frame")
        .and_then(|f| f.get("timestamp"))
        .and_then(|x| x.as_str())
        .unwrap_or("");
    let mut out = Vec::new();
    for ev in obs
        .get("events")
        .and_then(|e| e.as_array())
        .into_iter()
        .flatten()
    {
        if ev.get("kind").and_then(|k| k.as_str()) != Some("episode") {
            continue;
        }
        let participants: Vec<u64> = ev
            .get("participants")
            .and_then(|p| p.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str())
                    .filter_map(entity_num)
                    .collect()
            })
            .unwrap_or_default();
        let mut m = serde_json::Map::new();
        m.insert("ts".into(), serde_json::json!(ts));
        if let Some(label) = ev.get("label") {
            m.insert("label".into(), label.clone());
        }
        m.insert("participants".into(), serde_json::json!(participants));
        if let Some(d) = ev.get("duration_s") {
            m.insert("duration_s".into(), d.clone());
        }
        if let Some(pc) = ev.get("peak_confidence") {
            m.insert("peak_confidence".into(), pc.clone());
        }
        if let Some(s) = ev.get("stable") {
            m.insert("stable_participants".into(), s.clone());
        }
        out.push(serde_json::Value::Object(m));
    }
    out
}

fn parse_frame(line: &str) -> Option<(String, FrameObs)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    parse_frame_value(&v)
}

/// Parse an already-decoded flat scene-JSON value into a `FrameObs`. Shared by the
/// flat-log path and the semantic-session shim (which projects an observation into
/// this same shape — see [`scene_value_from_semantic`]).
fn parse_frame_value(v: &serde_json::Value) -> Option<(String, FrameObs)> {
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
    let anomaly = v
        .get("anomaly_score")
        .and_then(|x| x.as_f64())
        .unwrap_or(0.0) as f32;
    Some((
        ts,
        FrameObs {
            people,
            object_classes,
            zone_labels,
            anomaly,
        },
    ))
}

/// Distinct string values of `field` across an array node (dedup within a frame).
fn distinct_strings(node: Option<&serde_json::Value>, field: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    if let Some(arr) = node.and_then(|n| n.as_array()) {
        for item in arr {
            if let Some(s) = item.get(field).and_then(|x| x.as_str())
                && !out.iter().any(|e| e == s)
            {
                out.push(s.to_string());
            }
        }
    }
    out
}

fn parse_episode(line: &str) -> Option<(String, EpisodeObs)> {
    let v: serde_json::Value = serde_json::from_str(line).ok()?;
    parse_episode_value(&v)
}

/// Parse an already-decoded flat `events-*.jsonl` value into an `EpisodeObs`.
/// Shared by the flat-log path and the semantic-session shim.
fn parse_episode_value(v: &serde_json::Value) -> Option<(String, EpisodeObs)> {
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
            peak_confidence: v
                .get("peak_confidence")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0) as f32,
            stable_participants: v
                .get("stable_participants")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
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
                let dt =
                    DateTime::parse_from_rfc3339(ts).with_context(|| format!("parse ts {ts}"))?;
                let cutoff = dt - chrono::Duration::seconds(secs);
                Some(cutoff.format("%Y-%m-%dT%H:%M:%S%:z").to_string())
            }
            None => None,
        };
        Ok(Window { from, to: None })
    } else {
        Ok(Window {
            from: args.from.clone(),
            to: args.to.clone(),
        })
    }
}

fn in_window(ts: &str, w: &Window) -> bool {
    if let Some(f) = &w.from
        && ts < f.as_str()
    {
        return false;
    }
    if let Some(t) = &w.to
        && ts > t.as_str()
    {
        return false;
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
    let n: i64 = num
        .trim()
        .parse()
        .with_context(|| format!("bad duration '{s}'"))?;
    Ok(n * mult)
}

/// A time-of-day span label from the kept frames (e.g. `18:55:01–18:57:39`).
fn window_title(frames: &[(String, FrameObs)], w: &Window) -> String {
    let kept: Vec<&String> = frames
        .iter()
        .map(|(ts, _)| ts)
        .filter(|ts| in_window(ts, w))
        .collect();
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
    let mut a = InspectArgs {
        input: default_input(),
        around: None,
        window_s: 5.0,
        id: None,
        md: true,
    };
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
    inspect_collect(&InspectArgs {
        input: input.to_path_buf(),
        around,
        window_s,
        id,
        md,
    })
}

fn inspect_collect(args: &InspectArgs) -> Result<String> {
    let source = resolve_source(&args.input, None)?;
    let mut lines: Vec<(String, String)> = Vec::new();
    match &source {
        Source::Flat { scene, .. } => {
            for f in scene {
                let text =
                    std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    if let Some(ts) = serde_json::from_str::<serde_json::Value>(line)
                        .ok()
                        .and_then(|v| v.get("ts").and_then(|x| x.as_str()).map(String::from))
                    {
                        lines.push((ts, line.to_string()));
                    }
                }
            }
        }
        // Semantic sessions are projected to the flat scene shape so `inspect`'s
        // md/filter/raw paths work unchanged (`--raw` then shows that projection).
        Source::Semantic { labels } => {
            for f in labels {
                let text =
                    std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
                for line in text.lines().filter(|l| !l.trim().is_empty()) {
                    let Ok(obs) = serde_json::from_str::<serde_json::Value>(line) else {
                        continue;
                    };
                    if let Some(sv) = scene_value_from_semantic(&obs)
                        && let Some(ts) = sv.get("ts").and_then(|x| x.as_str()).map(String::from)
                    {
                        lines.push((ts, sv.to_string()));
                    }
                }
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
        (
            ts.into(),
            format!(r#"{{"ts":"{ts}","faces":[{{"id":{id}}}]}}"#),
        )
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
        let raw: Vec<String> = ["mvm", "inspect", "--raw"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(!parse_inspect_args(&raw).md, "--raw switches to raw JSONL");
        let json: Vec<String> = ["mvm", "inspect", "--json"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(!parse_inspect_args(&json).md, "--json is an alias of --raw");
    }

    #[test]
    fn inspect_md_line_is_compact() {
        let raw = r#"{"ts":"2026-07-05T18:55:19+03:00","faces":[{"id":0,"identity":"Вадим","dist_m":1.02,"posture":"standing","emotion":"Neutral"}]}"#;
        assert_eq!(
            inspect_md_line(raw),
            "18:55:19 · id0 Вадим 1.0m standing Neutral"
        );
    }

    // ── §16.7 semantic-session reading (golden-parity with the flat logs) ──────

    use crate::state::StateFile;
    use mvm_core::BBox;
    use mvm_core::world::{ObjectState, PersonState, Position3D, WorldState};
    use mvm_dataset::{RouteContext, SceneRelations, ZoneRel, route_with_relations};
    use std::time::Instant;

    fn tmp_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("mvm_recap_{}_{}", tag, std::process::id()))
    }

    fn full_person(id: u32) -> PersonState {
        PersonState {
            track_id: id,
            bbox: BBox {
                x1: 0.25,
                y1: 0.1,
                x2: 0.55,
                y2: 0.9,
            },
            face_bbox: None,
            identity: Some(("Вадим".into(), 0.8)),
            emotion: Some(("Neutral".into(), 0.7)),
            posture: Some(("sitting".into(), 0.9)),
            // z = 1.5 survives the flat log's 3-dp rounding, so dist_m matches exactly.
            position_3d: Some(Position3D {
                x: 0.1,
                y: 0.2,
                z: 1.5,
                confidence: 0.6,
            }),
            emotion_distribution: None,
            wrists: Vec::new(),
            hands: Vec::new(),
            body_coverage: None,
            torso_attitude: None,
            motion_state: None,
            velocity: None,
            left_gesture: None,
            right_gesture: None,
            recon_error: None,
            plausibility: None,
            torso_flexion: None,
            head_yaw: None,
            head_facing: None,
            last_seen: Instant::now(),
        }
    }

    fn ctx_at(ts: &'static str) -> RouteContext<'static> {
        RouteContext {
            dataset_id: "d",
            session_id: "s",
            timestamp: ts,
            seq_in_session: 0,
        }
    }

    #[test]
    fn semantic_and_flat_frame_yield_equal_frameobs() {
        // The core Phase 2 guarantee: the same world, emitted as a flat scene line
        // and as a semantic observation, parses to the identical recap FrameObs.
        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.frame_id = 3;
        world.people.push(full_person(1));
        world.objects.push(ObjectState {
            id: 10001,
            class: "keyboard".into(),
            bbox: BBox {
                x1: 0.3,
                y1: 0.4,
                x2: 0.5,
                y2: 0.6,
            }, // overlaps the person → one zone
            confidence: 0.8,
            age: 5,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: now,
        });
        world.anomaly_score = 0.25;

        // Flat: the real scene-log emitter (stable_zones via its own ZoneTracker).
        let sf = StateFile::open_at(tmp_path("golden_flat").with_extension("md")).unwrap();
        let flat_line = sf.to_json_line_world(&world, 30.0);
        let (_, flat) = parse_frame(&flat_line).expect("flat frame parses");

        // Semantic: the real router with the matching persistent zone.
        let rel = SceneRelations {
            zones: vec![ZoneRel {
                persistent_id: Some(1),
                label: Some("workstation".into()),
                members: vec![1, 10001],
            }],
            edges: vec![],
        };
        let obs = route_with_relations(&world, &ctx_at("2026-07-17T15:30:00.000+03:00"), &rel);
        let v = serde_json::to_value(&obs).unwrap();
        let sv = scene_value_from_semantic(&v).expect("scene projection");
        let (_, sem) = parse_frame_value(&sv).expect("semantic frame parses");

        assert_eq!(
            flat, sem,
            "FrameObs must match between flat and semantic sources"
        );
        // Sanity: the shared content is actually populated (not an empty-equals-empty pass).
        assert_eq!(sem.people[0].identity.as_deref(), Some("Вадим"));
        assert_eq!(sem.people[0].dist_m, Some(1.5));
        assert_eq!(sem.people[0].posture.as_deref(), Some("sitting"));
        assert_eq!(sem.object_classes, vec!["keyboard".to_string()]);
        assert_eq!(sem.zone_labels, vec!["workstation".to_string()]);
    }

    #[test]
    fn semantic_and_flat_episode_yield_equal_episodeobs() {
        // Episodes: the flat events-log line vs the router's inline `{kind:"episode"}`
        // both parse to the same EpisodeObs (participants string→u32, stable mapped).
        let ev = mvm_core::ActivityEvent {
            label: "seated_at_workstation".into(),
            participants: vec![0, 10001],
            started_at: 10.0,
            ended_at: 90.0,
            duration_s: 80.0,
            peak_confidence: 0.9,
            stable_participants: true,
        };
        let sf = StateFile::open_at(tmp_path("golden_ep").with_extension("md")).unwrap();
        let flat_line = &sf.events_log_lines(std::slice::from_ref(&ev))[0];
        let (_, flat) = parse_episode(flat_line).expect("flat episode parses");

        let mut world = WorldState::new(Instant::now());
        world.events.push(ev);
        let obs = route_with_relations(
            &world,
            &ctx_at("2026-07-17T15:30:00.000+03:00"),
            &SceneRelations::default(),
        );
        let v = serde_json::to_value(&obs).unwrap();
        let evs = event_values_from_semantic(&v);
        assert_eq!(evs.len(), 1, "one inline episode");
        let (_, sem) = parse_episode_value(&evs[0]).expect("semantic episode parses");

        assert_eq!(
            flat, sem,
            "EpisodeObs must match between flat and semantic sources"
        );
        assert_eq!(sem.participants, vec![0, 10001]);
        assert!(sem.stable_participants);
    }

    #[test]
    fn resolve_source_prefers_fresh_semantic_over_stale_flat() {
        let root = tmp_path("resolve_semantic_first");
        let _ = std::fs::remove_dir_all(&root);
        let sess = root.join("datasets/sessions/2026-07-17_153000");
        std::fs::create_dir_all(&sess).unwrap();
        std::fs::write(sess.join("labels.jsonl"), "{}\n").unwrap();
        std::fs::write(root.join("scene-2026-07-16.jsonl"), "{}\n").unwrap();
        std::fs::write(root.join("events-2026-07-16.jsonl"), "{}\n").unwrap();

        match resolve_source(&root, None).unwrap() {
            Source::Semantic { labels } => assert_eq!(labels.len(), 1),
            _ => panic!("fresh semantic sessions must take precedence over stale flat logs"),
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn inspect_over_a_semantic_session_projects_md_lines() {
        // `inspect` (level-3 zoom) must also read semantic sessions — the observation
        // is projected to the flat scene shape so the compact md line renders.
        let root = tmp_path("inspect_sem");
        let _ = std::fs::remove_dir_all(&root);
        let sess = root.join("datasets/sessions/2026-07-17_153000");
        std::fs::create_dir_all(&sess).unwrap();

        let mut world = WorldState::new(Instant::now());
        world.people.push(full_person(1));
        let obs = route_with_relations(
            &world,
            &ctx_at("2026-07-17T15:30:00.000+03:00"),
            &SceneRelations::default(),
        );
        std::fs::write(
            sess.join("labels.jsonl"),
            format!("{}\n", serde_json::to_string(&obs).unwrap()),
        )
        .unwrap();

        let md = inspect_output(&root, None, 5.0, None, true).expect("inspect reads the session");
        assert!(
            md.contains("id1 Вадим"),
            "compact md line from the semantic obs:\n{md}"
        );
        assert!(
            md.contains("1.5m sitting"),
            "dist + posture projected:\n{md}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn recap_over_a_semantic_session_renders_markdown() {
        // End-to-end: a real semantic session on disk → recap markdown via the dir entry.
        let root = tmp_path("e2e");
        let _ = std::fs::remove_dir_all(&root);
        let sess = root.join("datasets/sessions/2026-07-17_153000");
        std::fs::create_dir_all(&sess).unwrap();

        let now = Instant::now();
        let mut world = WorldState::new(now);
        world.people.push(full_person(1));
        world.objects.push(ObjectState {
            id: 10001,
            class: "keyboard".into(),
            bbox: BBox {
                x1: 0.3,
                y1: 0.4,
                x2: 0.5,
                y2: 0.6,
            },
            confidence: 0.8,
            age: 5,
            position_3d: None,
            mask: None,
            remembered: false,
            last_seen: now,
        });
        let rel = SceneRelations {
            zones: vec![ZoneRel {
                persistent_id: Some(1),
                label: Some("workstation".into()),
                members: vec![1, 10001],
            }],
            edges: vec![],
        };
        let mut out = String::new();
        for (i, ts) in [
            "2026-07-17T15:30:00.000+03:00",
            "2026-07-17T15:30:01.000+03:00",
        ]
        .iter()
        .enumerate()
        {
            let mut ctx = ctx_at("");
            ctx.timestamp = ts;
            ctx.seq_in_session = i as u64;
            let obs = route_with_relations(&world, &ctx, &rel);
            out.push_str(&serde_json::to_string(&obs).unwrap());
            out.push('\n');
        }
        std::fs::write(sess.join("labels.jsonl"), out).unwrap();

        let md = recap_markdown_from_dir(&root).expect("recap builds from the semantic session");
        assert!(
            md.contains("Вадим"),
            "recap should surface the person:\n{md}"
        );
        assert!(
            md.contains("workstation"),
            "recap should surface the zone:\n{md}"
        );
        let _ = std::fs::remove_dir_all(&root);
    }
}
