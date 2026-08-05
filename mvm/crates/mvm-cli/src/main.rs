//! mvm-cli — MVM binary entry point.
//!
//! Subcommands (phase 13 CLI tree; `mvm <cmd> --help` for flags):
//!   mvm run        live vision mesh — production (alias: `mesh`, hidden)
//!   mvm snapshot   print the current scene (current_state.md) once
//!   mvm recap      summarise a period from the durable logs
//!   mvm inspect    raw per-frame slices at full precision
//!   mvm serve --mcp   MCP stdio server (snapshot/recap/inspect as tools)
//!   mvm --help / --version
//!
//! Compatibility flags (hidden, not subcommands):
//!   mvm --native-camera        deprecated alias of canonical `mvm run`
//!   mvm --native               deprecated Python-bridge sync pipeline
//!   mvm --once / --cpu / --bridge-path <DIR> / --no-camera

mod agents;
mod config_file;
mod cortex_report;
mod dataset_cmd;
mod doctor;
mod gallery;
mod help;
mod manifest;
mod mcp;
mod mcp_install;
mod paths;
mod pipeline;
mod recap_cmd;
mod render;
mod scene_debug;
mod service;
mod setup;
mod state;

use anyhow::{Context, Result};
use mvm_core::Detection;
use mvm_core::Module;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// Returns true when the Rust emotion result should replace the current (Python bridge) prediction.
/// Comparison uses `calibrated_conf` (prob × (1 - entropy)) — a high-entropy prediction
/// loses even if its raw probability looks reasonable.
pub(crate) fn emotion_wins(rust: &mvm_emotion::EmotionResult, py_conf: Option<f32>) -> bool {
    rust.calibrated_conf > py_conf.unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_emotion::EmotionResult;

    fn make_result(calibrated_conf: f32) -> EmotionResult {
        EmotionResult {
            label: "test".into(),
            prob: calibrated_conf,
            distribution: [0.0; 8],
            entropy: 0.0,
            uncertainty: 0.0,
            calibrated_conf,
        }
    }

    #[test]
    fn emotion_wins_when_no_python_conf() {
        assert!(emotion_wins(&make_result(0.6), None));
    }

    #[test]
    fn emotion_wins_when_rust_higher() {
        assert!(emotion_wins(&make_result(0.8), Some(0.5)));
    }

    #[test]
    fn emotion_loses_when_python_higher() {
        assert!(!emotion_wins(&make_result(0.5), Some(0.8)));
    }

    #[test]
    fn emotion_loses_on_tie() {
        assert!(!emotion_wins(&make_result(0.7), Some(0.7)));
    }

    /// High-entropy result loses even if raw prob would win.
    #[test]
    fn emotion_wins_uses_calibrated_conf() {
        // Rust prob=0.8, but high entropy → calibrated_conf = 0.8 * (1-0.9) = 0.08
        let uncertain = EmotionResult {
            label: "test".into(),
            prob: 0.8,
            distribution: [0.0; 8],
            entropy: 0.9,
            uncertainty: 0.9,
            calibrated_conf: 0.08,
        };
        // Python only has 0.1 confidence, but uncertain Rust should still lose
        assert!(
            !emotion_wins(&uncertain, Some(0.1)),
            "uncertain prediction should lose"
        );
    }

    // ── mesh subcommand (phase 8.D) ─────────────────────────────────────────

    #[test]
    fn mesh_parse_stub_and_max_frames() {
        let raw: Vec<String> = ["mvm", "mesh", "--stub", "--max-frames", "100"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let a = parse_mesh_args(&raw);
        assert!(a.stub);
        assert_eq!(a.max_frames, Some(100));
    }

    #[test]
    fn mesh_parse_defaults() {
        let raw: Vec<String> = ["mvm", "mesh"].iter().map(|s| s.to_string()).collect();
        let a = parse_mesh_args(&raw);
        assert!(!a.stub);
        assert_eq!(a.max_frames, None);
        // No flag → resolved from the product config ([cadence] snapshot_ms).
        assert_eq!(a.interval_ms, None);
        assert!(!a.common.dump_json);
        assert!(
            !a.common.masks,
            "masks opt-in — off by default on the production mesh path"
        );
    }

    #[test]
    fn mesh_parse_masks_flag() {
        let raw: Vec<String> = ["mvm", "mesh", "--masks"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(parse_mesh_args(&raw).common.masks);
    }

    /// Phase 13.A1: the common flags parse identically on the legacy sync root
    /// path and the mesh/`run` path — one `parse_common`, no drift.
    #[test]
    fn common_flags_parse_identically_on_both_paths() {
        let sync: Vec<String> = [
            "mvm",
            "--dump-json",
            "--masks",
            "--dump-full-graph",
            "--scene-log",
            "/x",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let mesh: Vec<String> = [
            "mvm",
            "run",
            "--dump-json",
            "--masks",
            "--dump-full-graph",
            "--scene-log",
            "/x",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let a = parse_args(&sync);
        let m = parse_mesh_args(&mesh);
        assert!(a.common.dump_json && a.common.masks && a.common.dump_full_graph);
        assert_eq!(
            a.common.scene_log.as_deref(),
            Some(std::path::Path::new("/x"))
        );
        assert_eq!(a.common.dump_json, m.common.dump_json);
        assert_eq!(a.common.masks, m.common.masks);
        assert_eq!(a.common.dump_full_graph, m.common.dump_full_graph);
        assert_eq!(a.common.scene_log, m.common.scene_log);
    }

    #[test]
    fn mesh_parse_dump_json_and_interval() {
        let raw: Vec<String> = ["mvm", "mesh", "--dump-json", "--interval-ms", "50"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let a = parse_mesh_args(&raw);
        assert!(a.common.dump_json);
        assert_eq!(a.interval_ms, Some(50));
    }

    #[test]
    fn mesh_parse_scheduling_mode_for_spike_runs() {
        let raw: Vec<String> = ["mvm", "run", "--scheduling-mode", "scene"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(
            parse_mesh_args(&raw).scheduling_mode,
            agents::SchedulingMode::Scene
        );
    }

    #[test]
    fn native_camera_alias_maps_legacy_cadence_to_canonical_mesh() {
        let raw: Vec<String> = [
            "mvm",
            "--once",
            "--native-camera",
            "--dump-interval",
            "3",
            "--dump-json",
            "--record",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let args = parse_native_camera_alias(&raw).unwrap();
        assert_eq!(args.max_frames, Some(1));
        assert_eq!(args.interval_ms, Some(3_000));
        assert!(args.common.dump_json);
        assert!(args.common.record);
    }

    #[test]
    fn native_camera_alias_rejects_retired_sync_only_debug_flags() {
        for flag in ["--scene-debug", "--detect-ab"] {
            let raw = vec!["mvm".into(), "--native-camera".into(), flag.into()];
            let error = parse_native_camera_alias(&raw).err().unwrap().to_string();
            assert!(error.contains("retired synchronous native-camera loop"));
        }
    }

    /// The model-free stub mesh actually runs on threads and writes a valid
    /// StateFile — the CI smoke test of the mesh plumbing (8.D acceptance).
    #[test]
    fn mesh_stub_runs_and_writes_statefile() {
        let path = std::env::temp_dir().join(format!("mvm_mesh_stub_{}.md", std::process::id()));
        let sf = state::StateFile::open_at(path.clone()).unwrap();
        let mesh = build_stub_mesh(sf, false, Duration::from_millis(1), 20, None, None, None);
        let health = mesh.run();

        let cap = health
            .iter()
            .find(|h| h.name == "stub-capture")
            .expect("capture present");
        assert_eq!(cap.ticks, 20, "stub capture produced its frame budget");
        assert!(path.exists(), "StateFile written");
        let content = std::fs::read_to_string(&path).unwrap();
        // Snapshot v2 (phase 12.B3): empty stub world → the empty-scene header.
        assert!(
            content.starts_with("# Scene ") && content.contains("· empty ·"),
            "valid v2 StateFile, got:\n{content}"
        );
        let _ = std::fs::remove_file(&path);
    }
}

/// Flags shared by the live paths — declared once here and parsed by
/// [`parse_common`] so `run`/`mesh` (mesh) and the legacy sync root path can't
/// drift apart (phase 13.A1). Downstream code reads them as `args.common.<f>`.
#[derive(Default)]
struct CommonFlags {
    /// --dump-json: print one JSON line per tick to stdout.
    dump_json: bool,
    /// --emotion-debug: show A/B table (EmotiEffLib vs HSEmotion) and add emotion_debug to JSON.
    emotion_debug: bool,
    /// --masks: run the YOLO-seg expert (phase 10.4, +1 ONNX) *in addition* to
    /// YOLO26n on the object cadence, attaching instance masks to tracked objects
    /// (→ mask-median depth, mask contact). Opt-in / default off — zero perf hit
    /// unless requested (spike §3.3: owner A/Bs the ~+47ms cadence cost live).
    masks: bool,
    /// --scene-log <dir>: write the **legacy** per-day flat logs
    /// (`scene-YYYY-MM-DD.jsonl` + `events-*.jsonl`) to <dir> (phase 12.C0).
    /// Retired from the default path in 16.7-3 (the semantic session replaced them);
    /// pass this to keep emitting them for back-compat. Independent of --dump-json.
    scene_log: Option<PathBuf>,
    /// --dump-distributions: include the 8-class `emotion_distribution` in the
    /// Scene JSON (phase 12.A2 opt-in; off by default — summary is enough).
    dump_distributions: bool,
    /// --dump-full-graph: include object↔object `near` edges (phase 12.A3 opt-in;
    /// off by default — static desk geometry).
    dump_full_graph: bool,
    /// --record: enable the **pixel** Dataset Factory recorder (phase 16.1.2b).
    /// Overrides `[dataset].enabled = false`; every N-th frame is saved with full
    /// labels + RGB/depth/masks. Without it, `mvm run` still writes a text-only
    /// semantic session (§16.7-3); this flag upgrades it to a pixel dataset.
    record: bool,
    /// --no-vision-log: disable the always-on text-only semantic session
    /// (§16.7-3). Turns off durable vision history entirely (unless `--record`).
    no_vision_log: bool,
    /// --privacy <safe|full|anonymous>: override `[dataset].privacy` for this run
    /// (phase 16.5). Raw string; parsed at mesh build.
    privacy: Option<String>,
}

/// Try to consume the flag at `raw[*i]` as a common flag. Returns `true` if it
/// matched (and advances `*i` past any value token); the caller's loop then does
/// its own trailing `i += 1`, exactly as before. Single source of truth so the
/// two live-path parsers stay byte-identical on these flags.
fn parse_common(raw: &[String], i: &mut usize, c: &mut CommonFlags) -> bool {
    match raw[*i].as_str() {
        "--dump-json" => c.dump_json = true,
        "--emotion-debug" => c.emotion_debug = true,
        "--masks" => c.masks = true,
        "--dump-distributions" => c.dump_distributions = true,
        "--dump-full-graph" => c.dump_full_graph = true,
        "--record" => c.record = true,
        "--no-vision-log" => c.no_vision_log = true,
        "--privacy" => {
            if *i + 1 < raw.len() {
                c.privacy = Some(raw[*i + 1].clone());
                *i += 1;
            }
        }
        "--scene-log" => {
            if *i + 1 < raw.len() {
                c.scene_log = Some(PathBuf::from(&raw[*i + 1]));
                *i += 1;
            }
        }
        _ => return false,
    }
    true
}

struct Args {
    use_cpu: bool,
    dump_interval: u64,
    once: bool,
    no_camera: bool,
    native: bool,
    bridge_path: Option<PathBuf>,
    /// --detect-ab: run native YuNet on each frame and log coverage vs Python
    /// SCRFD (phase 7.1, G4). Measurement only — does NOT change the detection
    /// source or 3D; that switch waits on the operator recalibration (7.1 gate).
    detect_ab: bool,
    /// --native-camera: full Python-light hot-path (phase 7.4, Variant A).
    /// Pure-Rust v4l2 capture + YuNet detection + native enrichment; ZERO Python,
    /// no pose (until 7.3). Implies --native. Frame stays in memory (no disk
    /// round-trip). NOTE: 3D uses face_height_m calibrated for SCRFD boxes —
    /// provisional until recalibrated for YuNet boxes (operator, 7.1 gate).
    native_camera: bool,
    /// --scene-debug: write an annotated JPG (object + person boxes, stable-id
    /// numbers) to /tmp/mvm_scene.jpg each dump and print a class/colour legend.
    /// Lets you SEE which objects the scene detector finds on real frames.
    scene_debug: bool,
    /// Flags shared with the mesh/`run` path (phase 13.A1).
    common: CommonFlags,
}

fn parse_args(raw: &[String]) -> Args {
    let mut args = Args {
        use_cpu: false,
        dump_interval: 180,
        once: false,
        no_camera: false,
        native: false,
        bridge_path: None,
        detect_ab: false,
        native_camera: false,
        scene_debug: false,
        common: CommonFlags::default(),
    };

    let mut i = 1;
    while i < raw.len() {
        if parse_common(raw, &mut i, &mut args.common) {
            i += 1;
            continue;
        }
        match raw[i].as_str() {
            "--cpu" => args.use_cpu = true,
            "--once" => args.once = true,
            "--no-camera" => args.no_camera = true,
            "--native" => args.native = true,
            "--detect-ab" => args.detect_ab = true,
            "--native-camera" => {
                args.native_camera = true;
                args.native = true;
            }
            "--scene-debug" => args.scene_debug = true,
            "--bridge-path" => {
                if i + 1 < raw.len() {
                    args.bridge_path = Some(PathBuf::from(&raw[i + 1]));
                    i += 1;
                }
            }
            "--dump-interval" => {
                if i + 1 < raw.len() {
                    args.dump_interval = raw[i + 1].parse().unwrap_or(180);
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }

    args
}

/// The pure-Rust `--native-camera` loop (phase 8.C-R). Drives a
/// [`BodyPrimaryPipeline`](pipeline::BodyPrimaryPipeline) that owns every model;
/// the `PersonTracker` and `StateFile` stay here. Behaviour matches the previous
/// in-`main` native path exactly: process → track → depth-anchor → dump.
fn run_native_loop(
    mut native_camera: Box<dyn mvm_pipeline::source::CameraSource>,
    mut pipeline: pipeline::BodyPrimaryPipeline,
    mut tracker: mvm_core::PersonTracker,
    state_file: state::StateFile,
    args: &Args,
    dump_interval: Duration,
) -> Result<()> {
    let mut last_dump = Instant::now();
    let mut frame_count: u64 = 0;
    let start = Instant::now();
    // Durable JSONL history (phase 12.C0); reuses the StateFile timezone.
    let mut scene_log = args
        .common
        .scene_log
        .as_ref()
        .map(|d| state_file.open_scene_log(d.clone()));

    loop {
        let frame = match native_camera.read() {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  native camera error: {e}");
                break;
            }
        };

        // Last frame's active track bboxes feed A1 anti-flip suppression.
        let prev_tracks: Vec<mvm_core::BBox> = tracker.active().map(|t| t.bbox).collect();
        // Id-locked tracks (phase 9.6) → their bodies skip SFace this frame.
        let locked_tracks: Vec<mvm_core::BBox> = tracker
            .active()
            .filter(|t| t.is_id_locked())
            .map(|t| t.bbox)
            .collect();
        // Emotion-fresh tracks (phase 9.7) → their bodies skip EmotiEffLib.
        let emotion_fresh: Vec<mvm_core::BBox> = tracker
            .active()
            .filter(|t| !t.needs_emotion())
            .map(|t| t.bbox)
            .collect();
        // Id-locked tracks (phase 9.9) → pose-ROI recovery if the detector drops
        // them. Locked-only so an anon fragment isn't propped into a phantom.
        let predicted_rois: Vec<mvm_core::BBox> = tracker
            .active()
            .filter(|t| t.is_id_locked())
            .map(|t| t.bbox)
            .collect();
        let pipeline::FrameResult {
            detections,
            low_conf,
            landmarks,
            bodies,
            overlay,
            emotion_debug_entries,
        } = pipeline.process(
            &frame,
            frame_count,
            &prev_tracks,
            &locked_tracks,
            &emotion_fresh,
            &predicted_rois,
        );

        frame_count += 1;
        // Monotonic seconds since loop start → motion velocities (phase 9.8).
        let t_secs = start.elapsed().as_secs_f64();
        tracker.tick_motion(&detections, &low_conf, &landmarks, t_secs);
        // Depth-anchoring needs the post-tick tracker (people's metric torso-Z).
        pipeline.anchor_depth(&tracker, &frame);

        if last_dump.elapsed() >= dump_interval {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = if elapsed > 0.0 {
                frame_count as f64 / elapsed
            } else {
                0.0
            };
            eprintln!(
                "  tick {frame_count}: {} tracks, {fps:.1} fps",
                tracker.track_count()
            );

            state_file.write(&tracker, fps, pipeline.objects())?;
            if args.common.emotion_debug {
                state_file.dump_table_debug(&tracker, fps, &emotion_debug_entries);
            } else {
                state_file.dump_table(&tracker, fps);
            }
            if args.common.dump_json || scene_log.is_some() {
                let line = if args.common.emotion_debug {
                    state_file.to_json_line_debug(&tracker, fps, &emotion_debug_entries)
                } else {
                    state_file.to_json_line(
                        &tracker,
                        fps,
                        pipeline.objects(),
                        &bodies,
                        pipeline.animals(),
                    )
                };
                if args.common.dump_json {
                    println!("{line}");
                }
                if let Some(log) = scene_log.as_mut()
                    && let Err(e) = log.append(&line)
                {
                    eprintln!("  scene-log: append error: {e}");
                }
            }

            // Scene visual debug (phase 8.E): annotated JPG + printed legend.
            if args.scene_debug {
                let person_bboxes: Vec<mvm_core::BBox> = tracker.active().map(|t| t.bbox).collect();
                match scene_debug::write_annotated(
                    "/tmp/mvm_scene.jpg",
                    &frame,
                    &person_bboxes,
                    pipeline.objects(),
                    &overlay,
                ) {
                    Ok(legend) => {
                        eprintln!("  scene-debug → /tmp/mvm_scene.jpg");
                        for line in &legend {
                            eprintln!("    {line}");
                        }
                    }
                    Err(e) => eprintln!("  scene-debug: write failed: {e}"),
                }
            }
            last_dump = Instant::now();

            if args.once {
                eprintln!("  once: done after {frame_count} frames");
                break;
            }
        }

        // Camera already paces itself; just a tiny yield.
        std::thread::sleep(Duration::from_millis(1));
    }

    Ok(())
}

// ── `mesh` subcommand (phase 8.D) ───────────────────────────────────────────

/// Args for `mvm-cli mesh [--stub] [--max-frames N] [--dump-json] [--dump-landmarks] [--interval-ms N]`.
struct MeshArgs {
    /// Build a model-free mesh (StubCapture + FrameCounter + StateFileWriter) —
    /// CI-safe smoke test of the threading + StateFile plumbing.
    stub: bool,
    /// Stop the producer after N frames (bounded run). `None` = until Ctrl-C.
    max_frames: Option<u64>,
    /// --dump-landmarks: include raw 21-point hand landmarks in the Scene JSON
    /// (phase 10.0 opt-in — heavy payload).
    dump_landmarks: bool,
    /// StateFileWriter snapshot interval (ms). `None` = take it from the
    /// product config (`[cadence] snapshot_ms`, default 1000) — CLI > config.
    interval_ms: Option<u64>,
    /// Experimental 18.8b.1 diagnostic selector. It only gates one-shot
    /// permits; fixed expert cadence remains unchanged.
    scheduling_mode: agents::SchedulingMode,
    /// Flags shared with the legacy sync root path (phase 13.A1).
    common: CommonFlags,
}

fn parse_mesh_args(raw: &[String]) -> MeshArgs {
    let mut a = MeshArgs {
        stub: false,
        max_frames: None,
        dump_landmarks: false,
        interval_ms: None,
        scheduling_mode: agents::SchedulingMode::SceneAndDepth,
        common: CommonFlags::default(),
    };
    let mut i = 2; // skip program name + subcommand (`mesh` / `run`)
    while i < raw.len() {
        if parse_common(raw, &mut i, &mut a.common) {
            i += 1;
            continue;
        }
        match raw[i].as_str() {
            "--stub" => a.stub = true,
            "--dump-landmarks" => a.dump_landmarks = true,
            "--max-frames" => {
                if i + 1 < raw.len() {
                    a.max_frames = raw[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--interval-ms" => {
                if i + 1 < raw.len() {
                    a.interval_ms = raw[i + 1].parse().ok();
                    i += 1;
                }
            }
            "--scheduling-mode" => {
                if let Some(mode) = raw
                    .get(i + 1)
                    .and_then(|value| agents::SchedulingMode::parse(value))
                {
                    a.scheduling_mode = mode;
                    i += 1;
                }
            }
            _ => {}
        }
        i += 1;
    }
    a
}

/// Translate the hidden root-level `--native-camera` compatibility surface to
/// the canonical AgentMesh arguments. The Rust v4l2 source already backs
/// `mvm run`; this adapter prevents the legacy flag from selecting the old
/// synchronous state pipeline.
fn parse_native_camera_alias(raw: &[String]) -> Result<MeshArgs> {
    let mut args = MeshArgs {
        stub: false,
        max_frames: None,
        dump_landmarks: false,
        interval_ms: None,
        scheduling_mode: agents::SchedulingMode::SceneAndDepth,
        common: CommonFlags::default(),
    };
    let mut i = 1;
    while i < raw.len() {
        if parse_common(raw, &mut i, &mut args.common) {
            i += 1;
            continue;
        }
        match raw[i].as_str() {
            "--native-camera" | "--native" | "--cpu" => {}
            "--once" => args.max_frames = Some(1),
            "--dump-landmarks" => args.dump_landmarks = true,
            "--max-frames" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--max-frames requires a value"))?;
                args.max_frames = Some(value.parse().context("invalid --max-frames value")?);
                i += 1;
            }
            "--interval-ms" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--interval-ms requires a value"))?;
                args.interval_ms = Some(value.parse().context("invalid --interval-ms value")?);
                i += 1;
            }
            "--dump-interval" => {
                let value = raw
                    .get(i + 1)
                    .ok_or_else(|| anyhow::anyhow!("--dump-interval requires seconds"))?;
                let seconds: u64 = value.parse().context("invalid --dump-interval value")?;
                args.interval_ms = Some(
                    seconds
                        .checked_mul(1_000)
                        .ok_or_else(|| anyhow::anyhow!("--dump-interval is too large"))?,
                );
                i += 1;
            }
            "--scene-debug" | "--detect-ab" => {
                anyhow::bail!(
                    "{} belonged to the retired synchronous native-camera loop; \
                     use `mvm run --dump-json` on the canonical AgentMesh path",
                    raw[i]
                );
            }
            "--no-camera" => {
                anyhow::bail!("--no-camera conflicts with the --native-camera compatibility alias");
            }
            "--bridge-path" => {
                anyhow::bail!("--bridge-path applies only to the deprecated Python --native path");
            }
            _ => {}
        }
        i += 1;
    }
    Ok(args)
}

fn report_health(health: &[mvm_mesh::AgentHealth]) {
    eprintln!("  mesh stopped — agent health:");
    for h in health {
        eprintln!(
            "    {:<10} ticks={:<6} errors={}",
            h.name, h.ticks, h.errors
        );
    }
}

/// Model-free stub mesh: synthetic capture + frame counter + StateFileWriter.
/// Exercises the threading and StateFile plumbing without models or a camera.
#[allow(clippy::too_many_arguments)]
fn build_stub_mesh(
    state_file: state::StateFile,
    dump_json: bool,
    interval: Duration,
    max_frames: u64,
    scene_log: Option<state::SceneLog>,
    events_log: Option<state::SceneLog>,
    journal: Option<agents::JournalCfg>,
) -> mvm_mesh::AgentMesh {
    let mut mesh = mvm_mesh::AgentMesh::new(Instant::now());
    mesh.add(Box::new(mvm_mesh::StubCaptureAgent::new(
        1000.0, max_frames,
    )));
    mesh.add(Box::new(mvm_mesh::FrameCounterAgent::new()));
    mesh.add(Box::new(agents::StateFileWriter::new(
        state_file, dump_json, interval, None, scene_log, events_log, journal,
    )));
    mesh
}

/// Required models (from the manifest) absent in `dir` — the human chain
/// cannot start without these five. Optional detectors degrade gracefully.
fn missing_required_models(dir: &std::path::Path) -> Vec<&'static str> {
    manifest::required_files()
        .into_iter()
        .filter(|f| !dir.join(f).exists())
        .collect()
}

fn load_yolo(config: &mvm_core::Config) -> Option<mvm_scene::YoloDetector> {
    let p = config.models_path.join("yolo_object_detect.onnx");
    if !p.exists() {
        eprintln!(
            "  scene: no YOLO model at {} — object agent disabled",
            p.display()
        );
        return None;
    }
    match mvm_scene::YoloDetector::from_path(&p) {
        Ok(mut d) => {
            d.set_score_threshold(config.object_score_thresh);
            eprintln!("  scene: YOLO loaded ({})", p.display());
            Some(d)
        }
        Err(e) => {
            eprintln!("  scene: YOLO load failed: {e} (object agent disabled)");
            None
        }
    }
}

fn load_depth(config: &mvm_core::Config) -> Option<mvm_depth::DepthEstimator> {
    let p = config.models_path.join("depth_anything_v2_small.onnx");
    if !p.exists() {
        eprintln!(
            "  depth: no model at {} — depth agent disabled",
            p.display()
        );
        return None;
    }
    match mvm_depth::DepthEstimator::from_path(&p) {
        Ok(d) => {
            eprintln!("  depth: Depth-Anything loaded ({})", p.display());
            Some(d)
        }
        Err(e) => {
            eprintln!("  depth: load failed: {e} (depth agent disabled)");
            None
        }
    }
}

/// YOLO-seg expert (phase 10.4, +1 ONNX). Gated on the model file like the other
/// heavy detectors; the `--masks` flag gate is applied by the caller. On the mesh
/// (production) path this rides inside the SceneAgent — one seg forward yields
/// both object masks (→ mask-contact/mask-depth) and person masks (→ 11.A3 tracker
/// cue). Absent model → `None`, masks silently off.
fn load_seg(config: &mvm_core::Config) -> Option<mvm_scene::YoloSegDetector> {
    let p = config.models_path.join("yolo_seg.onnx");
    if !p.exists() {
        eprintln!(
            "  seg: --masks set but no model at {} — masks disabled",
            p.display()
        );
        return None;
    }
    match mvm_scene::YoloSegDetector::from_path(&p) {
        Ok(mut s) => {
            s.set_score_threshold(config.object_score_thresh);
            eprintln!(
                "  seg: YOLO-seg loaded ({}), cadence every {} frames",
                p.display(),
                config.object_cadence
            );
            Some(s)
        }
        Err(e) => {
            eprintln!("  seg: YOLO-seg load failed: {e} (masks disabled)");
            None
        }
    }
}

fn load_hands(config: &mvm_core::Config) -> Option<mvm_hands::HandLandmarker> {
    let p = config
        .models_path
        .join("handpose_estimation_mediapipe_2023feb.onnx");
    if !p.exists() {
        return None;
    }
    match mvm_hands::HandLandmarker::from_path(&p) {
        Ok(h) => {
            eprintln!("  hands: handpose loaded ({})", p.display());
            Some(h)
        }
        Err(e) => {
            eprintln!("  hands: load failed: {e} (hands disabled)");
            None
        }
    }
}

/// Run the Agent Mesh (phase 8.D). `--stub` builds a model-free mesh; otherwise
/// loads every model + the native camera and assembles
/// Capture → {Human, Scene, Depth} → Fusion → StateFileWriter on threads.
fn run_mesh(args: MeshArgs) -> Result<()> {
    // Product config (phase 14.0.2): file + env resolved once; CLI flags below
    // override the resolved values.
    let product = config_file::load()?;

    if args.stub {
        // One private (0700) data dir holds state + logs + journal (14.0.4).
        paths::ensure_dir_private(&product.data_dir)?;
        let mut state_file = state::StateFile::open_at(product.state_path.clone())?;
        state_file.dump_landmarks = args.dump_landmarks;
        state_file.dump_distributions = args.common.dump_distributions;
        state_file.dump_full_graph = args.common.dump_full_graph;
        let interval = Duration::from_millis(args.interval_ms.unwrap_or(product.snapshot_ms));
        eprintln!("mvm mesh — starting (stub, model-free)");
        let mesh = build_stub_mesh(
            state_file,
            args.common.dump_json,
            interval,
            args.max_frames.unwrap_or(100),
            None,
            None,
            None,
        );
        let health = mesh.run();
        report_health(&health);
        return Ok(());
    }

    let mesh = build_live_mesh(&product, &args)?;
    eprintln!("  mesh: running agents on threads (Ctrl-C or --max-frames to stop)");
    let health = mesh.run();
    report_health(&health);
    Ok(())
}

/// Parse a `--privacy` value into a [`mvm_dataset::Privacy`]; `None` on anything
/// unrecognised (the file/default value is kept).
fn parse_privacy(s: &str) -> Option<mvm_dataset::Privacy> {
    match s.to_ascii_lowercase().as_str() {
        "safe" => Some(mvm_dataset::Privacy::Safe),
        "full" => Some(mvm_dataset::Privacy::Full),
        "anonymous" | "anon" => Some(mvm_dataset::Privacy::Anonymous),
        _ => None,
    }
}

/// Assemble the full live production mesh (state/logs/journal + camera + every
/// model + writer) from the resolved config. Shared by `run` and
/// `serve --with-camera` (14.3), which spawns it in a background thread while it
/// serves MCP over stdio.
fn build_live_mesh(
    product: &config_file::ProductConfig,
    args: &MeshArgs,
) -> Result<mvm_mesh::AgentMesh> {
    // One private (0700) data dir holds state + logs + journal (phase 14.0.4) —
    // `serve`/`snapshot`/`recap` read the same place `run` writes by default.
    paths::ensure_dir_private(&product.data_dir)?;
    let mut state_file = state::StateFile::open_at(product.state_path.clone())?;
    state_file.dump_landmarks = args.dump_landmarks;
    state_file.dump_distributions = args.common.dump_distributions;
    state_file.dump_full_graph = args.common.dump_full_graph;
    let interval = Duration::from_millis(args.interval_ms.unwrap_or(product.snapshot_ms));
    // Flat JSONL logs are RETIRED by default (phase 16.7-3): the durable vision
    // history is now the text-only semantic session (labels.jsonl), and the journal
    // reads it via the recap resolver. `--scene-log <dir>` still writes the legacy
    // `scene-*.jsonl`/`events-*.jsonl` when explicitly requested (back-compat).
    let scene_log = args
        .common
        .scene_log
        .as_ref()
        .map(|d| state_file.open_scene_log(d.clone()));
    let events_log = args
        .common
        .scene_log
        .as_ref()
        .map(|d| state_file.open_events_log(d.clone()));
    // Rolling journal.md beside current_state.md (phase 12.D1). Its source dir is
    // data_dir by default (holds `datasets/sessions/` → the recap resolver finds
    // the semantic sessions), or the explicit `--scene-log` dir for the legacy logs.
    let log_dir = args
        .common
        .scene_log
        .clone()
        .unwrap_or_else(|| product.data_dir.clone());
    let journal = Some(agents::JournalCfg {
        path: state_file.journal_path(),
        log_dir: log_dir.clone(),
        cadence: 60,
    });

    eprintln!("mvm mesh — starting (live)");
    let mut config = mvm_core::Config::default();
    config.models_path = product.models_dir.clone();
    config.face_height_m = config.face_height_m_yunet; // YuNet-calibrated 3D
    // Dataset Factory session grouper reads its timeout from the core Config.
    config.session_timeout_secs = product.dataset.session_timeout_secs;

    // First-launch auto-setup (phase 14.1c): if required models are absent,
    // fetch them once before starting — so "поставил → mvm run → работает"
    // holds without a separate `mvm setup` step. Optional detectors stay lazy.
    let missing = missing_required_models(&config.models_path);
    if !missing.is_empty() {
        eprintln!(
            "  setup: {} required model(s) missing — fetching once into {}",
            missing.len(),
            config.models_path.display()
        );
        setup::fetch_specific(&config.models_path, &missing).with_context(|| {
            "auto-setup failed — run `mvm setup` manually, or point [models].dir \
                 in ~/.config/mvm/config.toml (or MVM_MODELS_DIR) at a populated dir"
                .to_string()
        })?;
    }

    // Required models (human chain).
    eprintln!("  detect: loading YuNet");
    let mut detector = mvm_detect::YunetDetector::init(&config)?;
    eprintln!("  emotion: loading EmotiEffLib B0");
    let emotion = mvm_emotion::EmotionModule::init(&config)?;
    eprintln!("  identity: loading SFace");
    let mut identity = mvm_identity::IdentityModule::init(&config)?;
    gallery::load_into(&mut identity, &mut detector, &product.gallery_dir)?;
    detector.set_score_threshold(0.5); // undo the loose gallery threshold
    eprintln!("  posture: loading native BlazePose");
    let native_pose = mvm_posture::NativePose::init(&config)?;
    let hands = load_hands(&config);

    // Optional heavy detectors (own agents).
    let yolo = load_yolo(&config);
    let depth = load_depth(&config);
    // YOLO-seg masks (opt-in --masks) ride the SceneAgent on the production path.
    let seg = if args.common.masks {
        load_seg(&config)
    } else {
        None
    };

    // Person tracker (body-primary: torso-3D from detections, no face pinhole).
    let mut person_tracker = mvm_core::PersonTracker::with_config(&config);
    person_tracker.pinhole = None;
    let mut names = identity.gallery_names();
    names.sort();
    for name in &names {
        let uid = person_tracker.register_user(name);
        eprintln!("  identity: user '{name}' → reserved id {uid}");
    }

    // Human-only pipeline (scene/depth are their own agents → None here). The
    // object_tracker field is unused in this mode but the type requires a value.
    let pipeline = pipeline::BodyPrimaryPipeline::new(
        config.clone(),
        args.common.emotion_debug,
        detector,
        Some(native_pose),
        Some(emotion),
        Some(identity),
        hands,
        None,
        None, // seg_detector — mesh path masks are a follow-up (phase 10.4)
        mvm_scene::ObjectTracker::with_params(0.3, config.object_track_timeout, 0.5),
        None,
    );

    // Typed latest-wins observation channels (phase 18.1). Expert packets keep
    // their source frame; Fusion is their only facts assembler.
    let human_observations = agents::HumanObservationBus::new();
    let scene_observations = agents::SceneObservationBus::new();
    let depth_observations = agents::DepthObservationBus::new();
    let fused_frames = agents::FusedFrameBus::new();
    let cortex_frames = agents::CortexFrameBus::new();
    let attention_policies = agents::AttentionPolicyBus::new();
    let policy_outcomes = agents::PolicyOutcomeBus::new();
    let scene_control = agents::SceneExecutionControl::default();
    let depth_control = agents::DepthExecutionControl::default();
    let scheduling_metrics = agents::SchedulingMetrics::new(args.scheduling_mode);
    let scene_available = yolo.is_some();
    let depth_available = depth.is_some();
    let run_id = format!(
        "{}-{}",
        chrono::Utc::now().format("%Y%m%dT%H%M%S%.3fZ"),
        std::process::id()
    );

    // Camera through the platform seam (14.0.3): "auto" scans /dev/video*,
    // an explicit [camera].device is opened literally.
    let camera = mvm_pipeline::source::open_source(
        &product.camera_device,
        product.camera_width,
        product.camera_height,
    )?;
    let (cam_w, cam_h) = camera.dimensions();
    eprintln!(
        "▶ camera {} ({cam_w}×{cam_h} MJPG) · data {} · snapshot {}ms",
        camera.device(),
        product.data_dir.display(),
        interval.as_millis()
    );

    // Assemble the mesh: capture → {human, scene, depth} → fusion → statefile.
    let mut mesh = mvm_mesh::AgentMesh::new(Instant::now());
    mesh.add(Box::new(agents::CaptureAgent::new(
        Box::new(camera),
        args.max_frames,
    )));
    mesh.add(Box::new(agents::HumanAgent::new(
        pipeline,
        person_tracker,
        scene_observations.clone(),
        human_observations.clone(),
        config.object_cadence,
        scheduling_metrics.clone(),
    )));
    if let Some(yolo) = yolo {
        mesh.add(Box::new(agents::SceneAgent::new(
            yolo,
            seg,
            scene_observations.clone(),
            mvm_scene::ObjectTracker::with_params(0.3, config.object_track_timeout, 0.5),
            config.object_score_thresh,
            config.object_min_area,
            config.object_cadence,
            scene_control.clone(),
            scheduling_metrics.clone(),
        )));
    }
    if let Some(depth) = depth {
        mesh.add(Box::new(agents::DepthAgent::new(
            depth,
            depth_observations.clone(),
            config.depth_cadence,
            depth_control.clone(),
            scheduling_metrics.clone(),
        )));
    }
    mesh.add(Box::new(agents::FusionAgent::new(
        config.clone(),
        human_observations,
        scene_observations.clone(),
        depth_observations.clone(),
        fused_frames.clone(),
        run_id,
        scheduling_metrics.clone(),
    )));
    mesh.add(Box::new(agents::CortexAgent::new(
        fused_frames.clone(),
        cortex_frames.clone(),
        attention_policies.clone(),
        config.clone(),
        mvm_cortex::CortexConfig { enabled: true },
    )));
    mesh.add(Box::new(agents::PolicySchedulerAgent::new(
        agents::PolicySchedulerWiring {
            policies: attention_policies.clone(),
            scenes: scene_observations.clone(),
            depths: depth_observations.clone(),
            outcomes: policy_outcomes.clone(),
            scene_control,
            depth_control,
            scene_available,
            depth_available,
            mode: args.scheduling_mode,
            metrics: scheduling_metrics,
        },
    )));
    mesh.add(Box::new(agents::CortexReportWriter::new(
        cortex_frames.clone(),
        attention_policies.clone(),
        policy_outcomes,
        product.data_dir.join("current_cortex.json"),
    )));
    mesh.add(Box::new(
        agents::StateFileWriter::new(
            state_file,
            args.common.dump_json,
            interval,
            None,
            scene_log,
            events_log,
            journal,
        )
        .with_fused_frames(fused_frames.clone()),
    ));

    // Dataset Factory recorder (phase 16.1.2b / 16.7-3). Default-on model: a
    // text-only semantic session unless `--record` upgrades it to a pixel dataset,
    // or `--no-vision-log` disables it. Reads WorldState like any other sink; the
    // vision core is untouched. CLI overrides win over the file.
    let mut ds_cfg = product.dataset.clone();
    ds_cfg.apply_cli_overrides(
        args.common.record.then_some(true),
        None,
        args.common.privacy.as_deref().and_then(parse_privacy),
    );
    if args.common.no_vision_log {
        ds_cfg.vision_log = false;
    }
    if let Some(cfg) = ds_cfg.effective_recorder() {
        let root = PathBuf::from(&cfg.dir);
        paths::ensure_dir_private(&root)?;
        let mode = if cfg.save_frames {
            "pixel dataset"
        } else {
            "text-only vision log"
        };
        eprintln!(
            "  dataset: {mode} → {} (cadence {}, privacy {:?})",
            root.display(),
            cfg.frame_cadence.max(1),
            cfg.privacy,
        );
        // Wire the stamped depth bus so a pixel dataset can save 16-bit depth PNGs
        // (empty until DepthAgent warms up → no depth path recorded meanwhile).
        mesh.add(Box::new(
            agents::DatasetWriterAgent::new(cfg, root, Some(depth_observations.clone()))
                .with_fused_frames(fused_frames.clone()),
        ));
    }

    Ok(mesh)
}

/// `mvm snapshot [--state <PATH>]` (phase 13.A3) — print the current scene once,
/// the same markdown the MCP `mvm_snapshot` tool returns. Reads the StateFile a
/// running `mvm run` writes (no camera of its own).
fn run_snapshot(raw: &[String]) -> Result<()> {
    let mut state = config_file::load()?.state_path;
    let mut i = 2; // skip program name + "snapshot"
    while i < raw.len() {
        if raw[i] == "--state" && i + 1 < raw.len() {
            state = PathBuf::from(&raw[i + 1]);
            i += 1;
        }
        i += 1;
    }
    let text = std::fs::read_to_string(&state)
        .with_context(|| format!("read snapshot {} (is `mvm run` running?)", state.display()))?;
    print!("{text}");
    Ok(())
}

/// `mvm cortex` (phase 18.9) — render the latest atomic cognitive artifact.
fn run_cortex(raw: &[String]) -> Result<()> {
    let product = config_file::load()?;
    let args = cortex_report::parse_args(raw)?;
    let path = args
        .input
        .unwrap_or_else(|| product.data_dir.join("current_cortex.json"));
    let output = cortex_report::render_file(
        &path,
        args.view,
        args.top_k,
        args.horizon_seconds,
        args.raw_json,
    )?;
    println!("{output}");
    Ok(())
}

/// `mvm serve --with-camera` (phase 14.3) — one process: the producer mesh runs
/// on its own threads writing state/logs/journal to `data_dir`, while the MCP
/// server runs on the main thread serving snapshot/recap/inspect over *those*
/// fresh artifacts. No separate `mvm run` needed. The MCP defaults already point
/// at `data_dir` (14.0.4), so producer and server line up automatically.
fn run_serve_with_camera(serve_args: mcp::ServeArgs, raw: &[String]) -> Result<()> {
    let product = config_file::load()?;
    let mesh = build_live_mesh(&product, &parse_mesh_args(raw))?;
    let stop = mesh.stop_flag();
    // Producer on background threads; MCP owns the main thread (stdin/stdout).
    let handle = std::thread::spawn(move || mesh.run());

    eprintln!("mvm serve --with-camera — MCP over the live mesh's own artifacts");
    let serve_result = mcp::run_serve(serve_args);

    // stdin closed (client gone) → wind the mesh down and reap it.
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    if let Ok(health) = handle.join() {
        report_health(&health);
    }
    serve_result
}

fn main() -> Result<()> {
    let raw: Vec<String> = std::env::args().collect();
    let sub = raw.get(1).map(String::as_str);

    // Top-level version / help — intercepted before the hand-rolled parsers,
    // which otherwise ignore unknown flags (`-h` included). Bare `mvm` (no args)
    // prints the overview instead of silently launching the legacy bridge.
    if help::wants_version(sub) {
        println!("mvm {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if raw.len() == 1 || matches!(sub, Some("--help") | Some("-h") | Some("help")) {
        print!("{}", help::MAIN);
        return Ok(());
    }

    // Subcommand tree. `run` is the canonical live path; `mesh` is a hidden alias
    // (phase 13.A2). Per-command `--help` prints that command's screen.
    if let Some(name) = sub
        && let Some(screen) = help::subcommand(name)
        && help::wants_help(&raw)
    {
        print!("{screen}");
        return Ok(());
    }
    if matches!(sub, Some("run") | Some("mesh")) {
        return run_mesh(parse_mesh_args(&raw));
    }
    if sub == Some("snapshot") {
        return run_snapshot(&raw);
    }
    if sub == Some("cortex") {
        return run_cortex(&raw);
    }
    if sub == Some("recap") {
        return recap_cmd::run_recap(recap_cmd::parse_recap_args(&raw));
    }
    if sub == Some("inspect") {
        return recap_cmd::run_inspect(recap_cmd::parse_inspect_args(&raw));
    }
    if sub == Some("serve") {
        let serve_args = mcp::parse_serve_args(&raw)?;
        if serve_args.with_camera {
            return run_serve_with_camera(serve_args, &raw);
        }
        return mcp::run_serve(serve_args);
    }
    if sub == Some("doctor") {
        let product = config_file::load()?;
        doctor::print_report(&product);
        return Ok(());
    }
    if sub == Some("setup") {
        let product = config_file::load()?;
        return setup::run_setup(&product, setup::parse_setup_args(&raw));
    }
    if sub == Some("gallery") {
        let product = config_file::load()?;
        return gallery::run_gallery(&product.gallery_dir, &raw);
    }
    if sub == Some("mcp") {
        // `mvm mcp install --host <host>` (phase 14.4). Only `install` today.
        if raw.get(2).map(String::as_str) != Some("install") {
            print!("{}", help::MCP);
            return Ok(());
        }
        let product = config_file::load()?;
        return mcp_install::run_install(&product, mcp_install::parse_install_args(&raw)?);
    }
    if sub == Some("service") {
        // `mvm service <install [--now]|uninstall|status>` (phase 14.5).
        return service::run_service(&raw);
    }
    if sub == Some("dataset") {
        // `mvm dataset <list|info|stats|prune|export>` (phase 16.3).
        let product = config_file::load()?;
        return dataset_cmd::run_dataset(&product, &raw);
    }

    // Phase 18.1 legacy transition: root-level `--native-camera` is now only a
    // compatibility spelling for the same Rust capture backend and canonical
    // AgentMesh used by `mvm run`, regardless of flag order.
    if raw.iter().skip(1).any(|arg| arg == "--native-camera") {
        eprintln!(
            "warning: root-level `--native-camera` is deprecated; routing to canonical `mvm run`"
        );
        return run_mesh(parse_native_camera_alias(&raw)?);
    }

    let args = parse_args(&raw);
    // Product config (phase 14.0.2) — models dir for the legacy sync path too.
    let product = config_file::load()?;
    let dump_secs = args.dump_interval;
    let once = args.once;
    let native = args.native;
    if native && !args.native_camera {
        eprintln!(
            "warning: `--native` Python bridge is deprecated for v0.4; use `mvm run` for the canonical Rust AgentMesh path"
        );
    }

    let mode_label = if native { "native, " } else { "" };
    let json_label = if args.common.dump_json { "json, " } else { "" };
    eprintln!(
        "mvm v{} — starting (dump every {dump_secs}s, {mode_label}{json_label}{}mode)",
        env!("CARGO_PKG_VERSION"),
        if once { "once, " } else { "" }
    );

    // Init detector (YuNet)
    #[cfg(feature = "embed-models")]
    let mut _detector = {
        let model_bytes = include_bytes!("../../../models/face_detection_yunet_2023mar.onnx");
        eprintln!(
            "  detect: YuNet from embedded bytes ({} bytes)",
            model_bytes.len()
        );
        mvm_detect::YunetDetector::from_bytes(model_bytes)?
    };

    #[cfg(not(feature = "embed-models"))]
    let mut _detector = {
        let config = mvm_core::Config {
            models_path: product.models_dir.clone(),
            ..Default::default()
        };
        eprintln!(
            "  detect: loading YuNet from {}",
            config.models_path.display()
        );
        mvm_detect::YunetDetector::init(&config)?
    };

    // Camera. Two mutually-exclusive sources:
    //   --native-camera → pure-Rust v4l2 (phase 7.4-light, Variant A), ZERO Python.
    //   default         → Python bridge (full pipeline incl. pose).
    let mut camera: Option<mvm_pipeline::Camera> = None;
    let mut native_camera: Option<Box<dyn mvm_pipeline::source::CameraSource>> = None;
    if args.no_camera {
        eprintln!("  camera: disabled (--no-camera)");
    } else if args.native_camera {
        // Platform seam (14.0.3). rotate_180 = false: camera is mounted upright
        // at eye level (the ceiling mount that needed cv2.ROTATE_180 is gone).
        match mvm_pipeline::source::open_source(
            &product.camera_device,
            product.camera_width,
            product.camera_height,
        ) {
            Ok(cam) => {
                let (w, h) = cam.dimensions();
                eprintln!(
                    "  camera: native v4l2 MJPG {} {w}x{h} (Python-light, no pose)",
                    cam.device()
                );
                native_camera = Some(cam);
            }
            Err(e) => {
                eprintln!("  camera: native open failed: {e}");
                eprintln!("  camera: falling back to stub mode");
            }
        }
    } else {
        let mut cfg = mvm_pipeline::CameraConfig::default();
        if let Some(ref bp) = args.bridge_path {
            cfg.mvm_repo = bp.clone();
        }
        if args.use_cpu {
            cfg.cpu = true;
        }

        if native {
            cfg.frame_output = Some(PathBuf::from("/tmp/mvm_frame.jpg"));
        }

        match mvm_pipeline::Camera::open(&cfg) {
            Ok(cam) => {
                eprintln!("  camera: connected via Python bridge");
                camera = Some(cam);
            }
            Err(e) => {
                eprintln!("  camera: {e}");
                eprintln!("  camera: falling back to stub mode (--no-camera)");
            }
        }
    }

    // Native inference modules
    let mut config = mvm_core::Config {
        models_path: product.models_dir.clone(),
        ..Default::default()
    };
    // Native (YuNet) capture uses YuNet-calibrated face height for 3D — the
    // SCRFD-tuned 0.44 would bias depth (phase 7.1 recalibration target).
    if args.native_camera {
        config.face_height_m = config.face_height_m_yunet;
    }
    let mut emotion_mod: Option<mvm_emotion::EmotionModule> = None;
    let mut identity_mod: Option<mvm_identity::IdentityModule> = None;

    if native {
        eprintln!("  emotion: loading EmotiEffLib B0 ONNX");
        emotion_mod = Some(mvm_emotion::EmotionModule::init(&config)?);
        eprintln!("  identity: loading SFace ONNX");
        let mut id = mvm_identity::IdentityModule::init(&config)?;
        gallery::load_into(&mut id, &mut _detector, &product.gallery_dir)?;
        identity_mod = Some(id);
    }

    // Native posture (phase 7.5.4) — only the pure-Rust --native-camera path. The
    // bridge path already gets pose from Python (Variant A). person_detection ROI
    // (the validated source) → BlazePose → VAE/classifier, multi-persona (fixes #3).
    let mut native_pose: Option<mvm_posture::NativePose> = None;
    if args.native_camera {
        eprintln!("  posture: loading native BlazePose (person det + pose + VAE)");
        #[cfg(feature = "embed-models")]
        {
            native_pose = Some(mvm_posture::NativePose::from_embedded()?);
        }
        #[cfg(not(feature = "embed-models"))]
        {
            native_pose = Some(mvm_posture::NativePose::init(&config)?);
        }
    }

    // Scene objects (phase 8.E, G₂ Scene group) — heavy YOLO detector on the
    // pure-Rust --native-camera path only. Gated on the model file: if the
    // bootstrap weights aren't downloaded yet, object nodes are simply disabled
    // (chain still runs, like a missing optional detector).
    let mut scene_detector: Option<mvm_scene::YoloDetector> = None;
    let object_tracker =
        mvm_scene::ObjectTracker::with_params(0.3, config.object_track_timeout, 0.5);
    let latest_objects: Vec<mvm_scene::TrackedObject> = Vec::new();
    if args.native_camera {
        let model_path = config.models_path.join("yolo_object_detect.onnx");
        if model_path.exists() {
            match mvm_scene::YoloDetector::from_path(&model_path) {
                Ok(mut d) => {
                    // Filter A: accept only confident objects (above the flicker band).
                    d.set_score_threshold(config.object_score_thresh);
                    eprintln!(
                        "  scene: loading YOLO object detector ({}), cadence every {} frames, score≥{}",
                        model_path.display(),
                        config.object_cadence,
                        config.object_score_thresh
                    );
                    scene_detector = Some(d);
                }
                Err(e) => eprintln!("  scene: YOLO load failed: {e} (object nodes disabled)"),
            }
        } else {
            eprintln!(
                "  scene: no YOLO model at {} — object nodes disabled (download bootstrap weights)",
                model_path.display()
            );
        }
    }

    // YOLO-seg instance masks (phase 10.4, +1 ONNX) — opt-in via --masks, an
    // *additional* cadenced expert over YOLO26n (not a drop-in; spike §3.2). Its
    // masks are attached onto the tracked objects for mask-median depth + contact.
    // Gated on both the flag and the model file (absent → masks silently off).
    let mut seg_detector: Option<mvm_scene::YoloSegDetector> = None;
    if args.native_camera && args.common.masks {
        let model_path = config.models_path.join("yolo_seg.onnx");
        if model_path.exists() {
            match mvm_scene::YoloSegDetector::from_path(&model_path) {
                Ok(mut s) => {
                    s.set_score_threshold(config.object_score_thresh);
                    eprintln!(
                        "  seg: loading YOLO-seg masks ({}), cadence every {} frames (over YOLO26n)",
                        model_path.display(),
                        config.object_cadence
                    );
                    seg_detector = Some(s);
                }
                Err(e) => eprintln!("  seg: YOLO-seg load failed: {e} (masks disabled)"),
            }
        } else {
            eprintln!(
                "  seg: --masks set but no model at {} — masks disabled (see phase-10 recipe)",
                model_path.display()
            );
        }
    }

    // Monocular depth (phase 8.E.2, G₄ Environment) — heavy (~230ms), low cadence.
    // Gives objects metric 3D by anchoring the relative depth map to a person's
    // torso-Z. Gated on the model file like the scene detector.
    let mut depth_estimator: Option<mvm_depth::DepthEstimator> = None;
    if args.native_camera {
        let model_path = config.models_path.join("depth_anything_v2_small.onnx");
        if model_path.exists() {
            match mvm_depth::DepthEstimator::from_path(&model_path) {
                Ok(d) => {
                    eprintln!(
                        "  depth: loading Depth-Anything-V2-Small ({}), cadence every {} frames",
                        model_path.display(),
                        config.depth_cadence
                    );
                    depth_estimator = Some(d);
                }
                Err(e) => eprintln!("  depth: load failed: {e} (depth disabled)"),
            }
        } else {
            eprintln!(
                "  depth: no model at {} — objects stay overlap-only",
                model_path.display()
            );
        }
    }

    // Hand landmarks (phase 8.E.4, completes G₁ Human) — body-primary wrist ROIs.
    // Gated on the model file like the other detectors.
    let mut hand_landmarker: Option<mvm_hands::HandLandmarker> = None;
    if args.native_camera {
        let model_path = config
            .models_path
            .join("handpose_estimation_mediapipe_2023feb.onnx");
        if model_path.exists() {
            match mvm_hands::HandLandmarker::from_path(&model_path) {
                Ok(h) => {
                    eprintln!(
                        "  hands: loading MediaPipe handpose ({})",
                        model_path.display()
                    );
                    hand_landmarker = Some(h);
                }
                Err(e) => eprintln!("  hands: load failed: {e} (hands disabled)"),
            }
        } else {
            eprintln!(
                "  hands: no model at {} — hands disabled",
                model_path.display()
            );
        }
    }

    let mut tracker = mvm_core::PersonTracker::with_config(&config);
    // Body-primary (phase 8): 3D comes from the torso in `build_body_primary`
    // (set on each Detection), so disable the tracker's face-bbox pinhole — it
    // would otherwise overwrite torso-3D by treating the body bbox as a face.
    // The detection-provided 3D then passes through unchanged. Bridge path keeps
    // native pinhole 3D (face-keyed) as before.
    if args.native_camera {
        tracker.pinhole = None;
    }
    // Register gallery users → reserved stable ids (phase 8 identity model): a
    // recognised body adopts a per-person id (Вадим=0, …) while it holds the
    // user; anonymous bodies get ephemeral ids ≥ 10.
    if let Some(id) = &identity_mod {
        let mut names = id.gallery_names();
        names.sort(); // deterministic reserved-id assignment
        for name in names {
            let uid = tracker.register_user(name);
            eprintln!("  identity: user '{name}' → reserved id {uid}");
        }
    }
    paths::ensure_dir_private(&product.data_dir)?;
    let mut state_file = state::StateFile::open_at(product.state_path.clone())?;
    state_file.dump_distributions = args.common.dump_distributions;
    state_file.dump_full_graph = args.common.dump_full_graph;
    state_file.camera_device = native_camera.as_ref().map(|c| c.device().to_string());
    let mut last_dump = Instant::now();
    let dump_interval = Duration::from_secs(dump_secs);

    let mut frame_count: u64 = 0;
    let start = Instant::now();

    // A/B detector coverage accumulators (phase 7.1, G4).
    let mut ab_cov = mvm_detect::Coverage::default();
    let mut ab_frames: u32 = 0;
    if native && (args.detect_ab || args.native_camera) {
        // Use YuNet's standard live threshold, not the loose gallery-build
        // threshold (0.3) left over from build_gallery.
        _detector.set_score_threshold(0.5);
    }
    if native && args.detect_ab {
        eprintln!("  detect-ab: logging native YuNet coverage vs Python SCRFD (measurement only)");
    }

    // ── Native-camera path (phase 8.C-R) ───────────────────────────────────
    // Move every model into a self-owning `BodyPrimaryPipeline` and run a
    // dedicated loop. The bridge path (`--native` / default) stays below — the
    // early return means its references to the moved values are never reached
    // (flow-sensitive), and `.take()` leaves `native_camera = None` on the
    // open-failed fallthrough so the old "idle stub" behaviour is preserved.
    if args.native_camera
        && let Some(nc) = native_camera.take()
    {
        let pipeline = pipeline::BodyPrimaryPipeline::new(
            config,
            args.common.emotion_debug,
            _detector,
            native_pose,
            emotion_mod,
            identity_mod,
            hand_landmarker,
            scene_detector,
            seg_detector,
            object_tracker,
            depth_estimator,
        );
        return run_native_loop(nc, pipeline, tracker, state_file, &args, dump_interval);
    }
    // Native camera failed to open → fall through to the idle bridge loop.

    loop {
        // ── Frame + detections acquisition ──────────────────────────────────
        // `tick_frame` is the working image for this tick (enrichment + A/B).
        // Native-camera path (Variant A): capture in memory + YuNet detection,
        // no Python, no pose. Bridge path: detections from JSON, frame on disk.
        let mut emotion_debug_entries: Vec<state::EmotionDebugEntry> = Vec::new();
        let overlay = scene_debug::DetectorOverlay::default();
        let mut tick_frame: Option<image::DynamicImage> = None;
        let mut detections: Vec<Detection> = if let Some(ncam) = &mut native_camera {
            // Body-primary (phase 8): grab the frame here; detections are built
            // below from the person detector (bodies), not from YuNet faces.
            match ncam.read() {
                Ok(frame) => {
                    tick_frame = Some(frame);
                    Vec::new()
                }
                Err(e) => {
                    eprintln!("  native camera error: {e}");
                    break;
                }
            }
        } else {
            match &mut camera {
                Some(cam) => match cam.read() {
                    Ok(dets) => dets,
                    Err(e) => {
                        eprintln!("  camera error: {e}");
                        break;
                    }
                },
                None => vec![],
            }
        };

        // Body-primary enrichment + object/depth cadence (phase 8) now live in
        // the native-camera path (`run_native_loop` + `BodyPrimaryPipeline`),
        // reached via early return above. This bridge loop only handles the
        // Python-bridge `--native` path, which carries no object/depth nodes.

        // Bridge path: the frame lives on disk (Python --write-frame). Load it
        // once for A/B + enrichment. Native path already holds it in memory.
        if tick_frame.is_none() && native && (args.detect_ab || !detections.is_empty()) {
            tick_frame = image::open("/tmp/mvm_frame.jpg").ok();
        }

        // A/B detector coverage (phase 7.1, G4) — bridge path only (the native
        // path already IS YuNet, so there's nothing to compare against).
        if native
            && args.detect_ab
            && native_camera.is_none()
            && let Some(frame) = &tick_frame
            && let Ok(yunet) = _detector.detect(frame)
        {
            let cov = mvm_detect::compare_coverage(&yunet, &detections, 0.5);
            ab_cov.matched += cov.matched;
            ab_cov.a_only += cov.a_only;
            ab_cov.b_only += cov.b_only;
            ab_frames += 1;
        }

        // Bridge-path enrichment: emotion + identity per face (the native-camera
        // path is handled above by build_body_primary).
        if native
            && native_camera.is_none()
            && !detections.is_empty()
            && let Some(frame) = &tick_frame
        {
            let (fw, fh) = (frame.width(), frame.height());
            for (det_idx, det) in detections.iter_mut().enumerate() {
                // Crop face
                let x1 = (det.bbox.x1 * fw as f32) as u32;
                let y1 = (det.bbox.y1 * fh as f32) as u32;
                let x2 = (det.bbox.x2 * fw as f32) as u32;
                let y2 = (det.bbox.y2 * fh as f32) as u32;
                let face = frame.crop_imm(x1, y1, x2 - x1, y2 - y1);

                // Emotion: keep higher calibrated-confidence prediction (Rust or Python bridge)
                if let Some(ref mut emo) = emotion_mod
                    && let Ok(result) = emo.classify(&face)
                {
                    let py_emotion = det.emotion.clone();
                    let py_conf = det.emotion_confidence;
                    // Always store the full distribution for PoE fusion (6.4).
                    det.emotion_distribution = Some(result.distribution);
                    let winner = if emotion_wins(&result, det.emotion_confidence) {
                        det.emotion = Some(result.label.clone());
                        det.emotion_confidence = Some(result.calibrated_conf);
                        "rust"
                    } else {
                        "python"
                    };
                    if args.common.emotion_debug {
                        emotion_debug_entries.push(state::EmotionDebugEntry {
                            id: det_idx as u32,
                            rust_emotion: result.label,
                            rust_conf: result.calibrated_conf,
                            py_emotion,
                            py_conf,
                            winner,
                        });
                    }
                }

                // Identity + embedding (phase 7.5).
                // Always embed the face (for cosine re-link #6, even unknown
                // faces). Apply the SFace label only when the Python bridge
                // didn't already set one (ArcFace > SFace), and only through
                // the margin gate (fixes #5: stranger no longer → "Вадим").
                if let Some(ref mut id) = identity_mod
                    && let Ok((name, sim, emb)) = id.identify_margin(
                        &face,
                        config.identity_abs_thresh,
                        config.identity_margin_thresh,
                    )
                {
                    det.embedding = Some(emb);
                    if det.identity.is_none() && name != "unknown" {
                        det.identity = Some(name);
                        det.identity_confidence = Some(sim);
                    }
                }
            }
        }

        // Native posture for the native-camera path is now produced body-primary
        // (in build_body_primary, per body) — the old face-primary match-by-bbox
        // block was removed in phase 8. Bridge-path posture comes from Python.

        frame_count += 1;
        tracker.tick(&detections);

        if last_dump.elapsed() >= dump_interval {
            let elapsed = start.elapsed().as_secs_f64();
            let fps = if elapsed > 0.0 {
                frame_count as f64 / elapsed
            } else {
                0.0
            };
            eprintln!(
                "  tick {frame_count}: {} tracks, {fps:.1} fps",
                tracker.track_count()
            );
            if args.detect_ab && ab_frames > 0 {
                // YuNet ≥ SCRFD coverage gate: a_only ≥ b_only means YuNet finds
                // at least as many faces as SCRFD. b_only = SCRFD-found YuNet-missed.
                eprintln!(
                    "  detect-ab[{ab_frames}f]: matched={} yunet_only={} scrfd_only={} \
                     → YuNet {} SCRFD",
                    ab_cov.matched,
                    ab_cov.a_only,
                    ab_cov.b_only,
                    if ab_cov.b_only == 0 {
                        "covers"
                    } else {
                        "MISSES some"
                    },
                );
            }
            state_file.write(&tracker, fps, &latest_objects)?;
            if args.common.emotion_debug {
                state_file.dump_table_debug(&tracker, fps, &emotion_debug_entries);
            } else {
                state_file.dump_table(&tracker, fps);
            }
            if args.common.dump_json {
                let line = if args.common.emotion_debug {
                    state_file.to_json_line_debug(&tracker, fps, &emotion_debug_entries)
                } else {
                    state_file.to_json_line(&tracker, fps, &latest_objects, &[], &[])
                };
                println!("{line}");
            }

            // Scene visual debug (phase 8.E): annotated JPG + printed legend so the
            // operator can SEE which objects are detected on real frames.
            if args.scene_debug {
                if let Some(frame) = &tick_frame {
                    let person_bboxes: Vec<mvm_core::BBox> =
                        tracker.active().map(|t| t.bbox).collect();
                    match scene_debug::write_annotated(
                        "/tmp/mvm_scene.jpg",
                        frame,
                        &person_bboxes,
                        &latest_objects,
                        &overlay,
                    ) {
                        Ok(legend) => {
                            eprintln!("  scene-debug → /tmp/mvm_scene.jpg");
                            for line in &legend {
                                eprintln!("    {line}");
                            }
                        }
                        Err(e) => eprintln!("  scene-debug: write failed: {e}"),
                    }
                } else {
                    eprintln!("  scene-debug: no frame this tick (skipped)");
                }
            }
            last_dump = Instant::now();

            if once {
                eprintln!("  once: done after {frame_count} frames");
                break;
            }
        }

        // Camera already paces itself; just a tiny yield
        std::thread::sleep(Duration::from_millis(1));
    }

    Ok(())
}
