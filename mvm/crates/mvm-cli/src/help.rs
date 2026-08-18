//! `--help` / `-h` / `--version` text (phase 13.B) — static, hand-rolled, +0-dep.
//!
//! One `const` per screen so the strings unit-test as goldens and can't silently
//! drift from the real flags. `main()` intercepts help/version *before* the
//! hand-rolled parsers (which otherwise ignore unknown flags, `-h` included).

/// True if `--help` or `-h` appears anywhere in the args (subcommand help).
pub fn wants_help(raw: &[String]) -> bool {
    raw.iter().any(|a| a == "--help" || a == "-h")
}

/// True if `--version` or `-V` appears as the first argument (top-level only).
pub fn wants_version(sub: Option<&str>) -> bool {
    matches!(sub, Some("--version") | Some("-V"))
}

/// Help text for a subcommand, or `None` if the name has no dedicated screen.
pub fn subcommand(name: &str) -> Option<&'static str> {
    Some(match name {
        "run" | "mesh" => RUN,
        "snapshot" => SNAPSHOT,
        "cortex" => CORTEX,
        "recap" => RECAP,
        "inspect" => INSPECT,
        "serve" => SERVE,
        "doctor" => DOCTOR,
        "sensor" => SENSOR,
        "setup" => SETUP,
        "gallery" => GALLERY,
        "mcp" => MCP,
        "service" => SERVICE,
        "dataset" => DATASET,
        _ => return None,
    })
}

pub const MAIN: &str = "\
mvm — Machine Vision Module: local, CPU-only structured vision.

MVM turns a camera stream into evidence-linked scene facts. It is an ensemble of
specialist models, not a general visual-question-answering system: use it for
people, objects, posture, motion, relations, zones and activities, and preserve
its confidence, freshness, `withheld`, and `unknown` results.

USAGE:
    mvm <command> [flags]

START HERE:
    1. mvm run                         Start the camera producer (leave it running)
    2. mvm snapshot                    Read canonical facts about the current scene
    3. mvm cortex --view attention     Read the separate cognitive view when needed

COMMANDS:
    run        Start live production vision: camera → experts → canonical scene facts
    snapshot   Read canonical facts now, once (current_state.md)
    cortex     Read attention, bounded forecast, self-model or uncertainty, once
    recap      Summarise durable semantic history over a time window
    inspect    Inspect raw recorded evidence around a time or track
    serve      Start the read-only MCP stdio server (for an LLM host)
    setup      Fetch the ONNX models into the models dir (checksum-verified)
    gallery    Manage the identity gallery (add a person's photos, list people)
    mcp        Register MVM as an MCP server with an LLM host (install)
    service    Run the producer as a systemd user service (install/uninstall/status)
    dataset    Inspect & maintain recorded Dataset Factory sessions (list/info/stats/prune)
    doctor     Check the setup: camera, models, data dir, config
    sensor     Bounded sensor diagnostics (no models, no image files)

READING THE SYSTEM:
    snapshot  → canonical facts now; begin every live-scene question here
    cortex    → evidence-linked cognitive interpretation; never replaces facts
    recap     → what happened over a window from durable labels.jsonl history
    inspect   → raw recorded slices that adjudicate a specific recap claim

ARTIFACTS AND AUTHORITY:
    current_state.md     Canonical current-scene facts, written atomically by run
    labels.jsonl         Sole durable semantic history (text-only by default)
    current_cortex.json  Atomic, runtime cognitive view; not facts or history

MCP:
    mvm serve --mcp                  Read a separately running producer
    mvm serve --mcp --with-camera    Run producer and MCP server together
    Tools: mvm_snapshot, mvm_cortex, mvm_recap, mvm_inspect (all read-only)

EXAMPLES:
    mvm run
    mvm snapshot
    mvm cortex --view uncertainty
    mvm recap --last 10m
    mvm inspect --id 12

FLAGS:
    -h, --help       Show help (add to any command for its flags)
    -V, --version    Print version

Run `mvm <command> --help` for exact flags and examples. `mvm run` is the
production path; `mvm run --stub` is a model-free CI smoke test, not live vision.
";

pub const RUN: &str = "\
mvm run — live vision mesh (production). Camera → {human, scene, depth} → fusion → StateFile.
`mvm mesh` is a hidden alias of this command.

By default `mvm run` keeps a durable, text-only semantic session log under
[dataset].dir/sessions/ (labels.jsonl — no pixels); recap/snapshot/inspect read
it automatically. --record upgrades to a pixel dataset; --no-vision-log disables it.

USAGE:
    mvm run [flags]

FLAGS:
    --record                Upgrade to a full pixel dataset (RGB + depth + masks +
                            labels → [dataset].dir; privacy-safe). See mvm dataset.
    --no-vision-log         Disable the always-on text-only semantic session log
    --scene-log <DIR>       Also write the LEGACY flat logs (scene-/events-*.jsonl)
                            to <DIR> — retired from the default path in 16.7
    --dump-json             Print one Scene-JSON line per tick to stdout
    --masks                 Run the YOLO-seg expert (+1 ONNX): instance masks →
                            mask-contact + mask-median depth (opt-in)
    --privacy <MODE>        Override [dataset].privacy for this run: safe (blur anon
                            faces) | anonymous (blur all) | full (blur none)
    --interval-ms <MS>      StateFile snapshot interval, milliseconds
                            (default: [cadence] snapshot_ms in config.toml, 1000)
    --max-frames <N>        Stop after N frames (default: run until Ctrl-C)
    --tracker-capture <DIR> Write exact Human/Scene tracker-tick calibration evidence
                            to a fresh DIR (requires --record and explicit --privacy)
    --sensor-backend <NAME> Override [sensor].backend: auto | v4l2 | kinect_v1
    --stub                  Model-free stub mesh (CI smoke; no camera/models)
    --emotion-debug         Show the emotion A/B table and add emotion_debug to JSON
    --dump-distributions    Include the 8-class emotion_distribution in Scene JSON
    --dump-full-graph       Include object↔object `near` edges in Scene JSON
    --dump-landmarks        Include raw 21-point hand landmarks in Scene JSON (heavy)

EXAMPLES:
    mvm run                                  # production: camera auto-detect, history → ~/.local/share/mvm
    mvm run --masks --dump-json              # live JSON with instance masks
    mvm run --stub --max-frames 100          # CI smoke, no hardware

Camera, paths and cadence come from ~/.config/mvm/config.toml (flags override).
";

pub const SENSOR: &str = "\
mvm sensor — bounded sensor-only diagnostics. No ONNX models or image files.

USAGE:
    mvm sensor smoke [--frames <N>]

The configured [sensor] backend is opened explicitly. For Kinect this reports
source identity, calibration revision, RGB-D pairing/skew, valid-depth ratio and
cumulative stream health, then cleanly releases the camera. Default: 90 frames.
";

pub const DATASET: &str = "\
mvm dataset — inspect & maintain recorded Dataset Factory sessions (phase 16).

`mvm run` records a text-only semantic session by default (labels.jsonl only);
`--record` upgrades to a pixel dataset (RGB + depth + masks) under [dataset].dir
(default ~/.local/share/mvm/datasets). Each session is self-describing:
[frames/ + depth/ +] labels.jsonl + session.toml. Prune old ones with `dataset prune`.

USAGE:
    mvm dataset list                 List recorded sessions (frames, status, people)
    mvm dataset info <session_id>    Details for one session
    mvm dataset stats                Dataset-wide totals (frames, participants, activities)
    mvm dataset check [session_id]   Data-quality gate: integrity, privacy, balance
                                     (--strict also fails on warnings; non-zero exit → CI)
    mvm dataset prune --keep <N|Nd>  Remove old sessions — keep last N, or on/after today−Nd
                                     (dry-run unless --yes)
    mvm dataset export ...           Convert to PyTorch/TFRecord/WebDataset (Python, phase 16.4)

EXAMPLES:
    mvm dataset list
    mvm dataset info 2026-07-17_153721
    mvm dataset check                     # gate the whole dataset (exit 1 on any error)
    mvm dataset check 2026-07-17_153721 --strict
    mvm dataset prune --keep 7d            # dry-run: what falls outside the last 7 days
    mvm dataset prune --keep 5 --yes       # actually keep the 5 newest, delete the rest

Privacy: `safe` (default) blurs anonymous faces, keeps gallery people clear;
`anonymous` blurs everyone; `full` saves everything. Set [dataset].privacy.
";

pub const SNAPSHOT: &str = "\
mvm snapshot — print the current scene once (same markdown as the MCP `mvm_snapshot` tool).

USAGE:
    mvm snapshot [--state <PATH>]

FLAGS:
    --state <PATH>   StateFile to read
                     (default: ~/.local/share/mvm/current_state.md, or [output] state_path)

Reads the file a running `mvm run` writes — start `mvm run` (or `mvm mesh`) first.
";

pub const CORTEX: &str = "\
mvm cortex — read the latest Visual Cortex report (read-only).

Use after `mvm snapshot` when the question concerns current attention, a bounded
forecast, the visual self-model, uncertainty, affordances, or advisory
re-observation. This is an evidence-linked cognitive interpretation: it never
replaces canonical scene facts and it is not durable history.

USAGE:
    mvm cortex [--view <full|attention|forecast|self|uncertainty>] [flags]

FLAGS:
    --view <VIEW>          Report section (default: full)
    --top-k <N>            Attention items, bounded to 1..20 (default: 5)
    --horizon-seconds <S>  Requested forecast horizon (Cortex remains bounded by evidence)
    --input <PATH>         Explicit current_cortex.json path
    --json                 Explicit raw JSON output; default is compact human-readable text

Reads the atomic artifact written by a running `mvm run` or `mvm serve --with-camera`.
`attention` ranks focus items; `forecast` reports only the horizon supported by
evidence; `self` reports camera/world-pose knowledge; `uncertainty` preserves
withheld/unknown status and reasons. Use `--json` only for an explicit raw-data need.
";

pub const RECAP: &str = "\
mvm recap — summarise a period from the durable vision history (semantic sessions,
or legacy scene-/events-*.jsonl).

USAGE:
    mvm recap [--input <PATH>] [--last <DUR> | --from <ISO> --to <ISO>]

FLAGS:
    --input <PATH>   Data dir, a session dir/labels.jsonl, or a legacy scene file
                     (default: ~/.local/share/mvm — where `mvm run` writes). Prefers
                     semantic sessions; legacy flat logs are fallback only.
    --events <PATH>  Explicit events file (when --input is a single file)
    --last <DUR>     Keep only the last window before the latest entry: 30s, 10m, 2h
    --from <ISO>     Inclusive ISO8601 lower bound
    --to <ISO>       Inclusive ISO8601 upper bound

EXAMPLES:
    mvm recap --input ~/vision/logs --last 10m
    mvm recap --input ~/vision/logs --from 2026-07-05T17:00:00 --to 2026-07-05T17:30:00
";

pub const INSPECT: &str = "\
mvm inspect — raw per-frame slices at full logged precision (level 3 of the pull model).

USAGE:
    mvm inspect [--input <PATH>] [--around <ISO>] [--window <SECS>] [--id <N>] [--raw]

FLAGS:
    --input <PATH>    Log directory or single scene file
                      (default: ~/.local/share/mvm — where `mvm run` writes)
    --around <ISO>    ISO8601 timestamp to centre the slice on
    --window <SECS>   ± seconds around --around (default 5)
    --id <N>          Keep only frames with this track id
                      (0–9 gallery · ≥10 anon people · ≥10000 objects)
    --raw             Raw JSONL instead of the default compact markdown lines (--md forces markdown)

EXAMPLES:
    mvm inspect --input ~/vision/logs --around 2026-07-05T17:51:23 --window 5
    mvm inspect --input ~/vision/logs --id 12
";

pub const SERVE: &str = "\
mvm serve --mcp — MCP stdio server exposing scene and Cortex tools (read-only).

USAGE:
    mvm serve --mcp [--with-camera] [--state <PATH>] [--logs <DIR>]

FLAGS:
    --mcp            Select the MCP stdio transport (the only transport)
    --with-camera    Also run the producer mesh in-process (camera + models),
                     so the tools serve this server's OWN fresh artifacts —
                     no separate `mvm run` needed (phase 14.3)
    --state <PATH>   Snapshot source StateFile (default: ~/.local/share/mvm/current_state.md)
    --logs <DIR>     Recap/inspect log dir (default: ~/.local/share/mvm — same data_dir `mvm run` writes)

Without --with-camera: read-only over the artifacts a separate `mvm run` writes
(producer and server decoupled). With --with-camera: one self-contained process.
The server exposes `mvm_snapshot`, `mvm_cortex`, `mvm_recap`, and `mvm_inspect`.
`mvm_cortex` reads current_cortex.json as a cognitive view; it does not alter
current_state.md or labels.jsonl.
";

pub const DOCTOR: &str = "\
mvm doctor — check the setup and show how everything resolves (why doesn't it work?).

USAGE:
    mvm doctor

CHECKS:
    sensor       opens v4l2, or verifies configured Kinect serial and ownership
    runtime      Kinect only: pinned local libfreenect 0.7.5
    calibration  Kinect only: schema, serial and exact profile revision
    models       required + optional ONNX models present in the models dir
    data dir     ~/.local/share/mvm (or [output].data_dir) exists and is writable
    config       which ~/.config/mvm/config.toml was read (or pure defaults)

Every path/device shown is the value the other commands will actually use.
";

pub const SETUP: &str = "\
mvm setup — fetch the ONNX models into the models dir, verifying each SHA-256.

USAGE:
    mvm setup [--force]

FLAGS:
    --force    Re-download even models already present with a matching checksum

Downloads to [models].dir (default: ~/.local/share/mvm/models) from each model's
canonical source (OpenCV Zoo / HuggingFace / project). Interrupted transfers
resume; only checksum-verified files are committed. Two YOLO models (objects,
masks) have no public URL yet — provide them manually; the rest set up the full
core chain. `mvm run` auto-fetches the required models on first launch.
";

pub const GALLERY: &str = "\
mvm gallery — manage the identity gallery: who the camera should recognise by name.

USAGE:
    mvm gallery add <name> <photo>…    Add reference photo(s) for a person
    mvm gallery list                   List known people and their photo counts

The gallery lives at [gallery].dir (default: ~/.config/mvm/gallery) as one folder
per person — the folder name IS the display name (Unicode, e.g. «Вадим»). `add`
copies photos in; `run` detects and crops the face from each on load. You can also
just drop photos into a `<gallery>/<Name>/` folder by hand — same effect.

EXAMPLES:
    mvm gallery add Вадим ~/photos/me1.jpg ~/photos/me2.jpg
    mvm gallery list
";

pub const MCP: &str = "\
mvm mcp install — register MVM as an MCP server with an LLM host.

USAGE:
    mvm mcp install --host <claude-desktop|claude-code|hermes> [--with-camera] [--write]

FLAGS:
    --host <HOST>    Which host to configure:
                       claude-desktop  → ~/.config/Claude/claude_desktop_config.json (JSON)
                       claude-code     → ~/.claude.json (JSON)
                       hermes          → ~/.hermes/config.yaml (YAML)
    --with-camera    Register `serve --mcp --with-camera` (self-contained: the
                     server also runs the producer). Default assumes a separate
                     `mvm run` / service produces the artifacts.
    --write          Merge the block into the file (JSON hosts only; backs up to
                     <file>.bak, preserves other keys). Hermes is YAML — printed
                     for you to paste manually.

Default is a dry run: prints the exact block and the file it belongs in. The
block carries this machine's mvm path and data dir (→ --logs/--state), so the
host launches a server that reads the same place `mvm run` writes.

EXAMPLES:
    mvm mcp install --host claude-code                 # show the block + path
    mvm mcp install --host claude-code --write         # merge it in (with .bak)
    mvm mcp install --host hermes --with-camera        # print YAML to paste
";

pub const SERVICE: &str = "\
mvm service — run the vision producer (`mvm run`) as a systemd user service (no root).

USAGE:
    mvm service install [--now]    Write ~/.config/systemd/user/mvm.service + daemon-reload
    mvm service uninstall          Disable, stop, and remove the unit
    mvm service status             Show whether the unit is present / enabled / active

FLAGS:
    --now    On install, also `enable --now` (start the producer immediately and
             on every login). Without it, the unit is written but not started.

The unit runs this machine's `mvm run`, writing state/logs/journal to the data dir
so `mvm serve --mcp` (or a registered MCP host) reads fresh artifacts. Logs go to
the journal: journalctl --user -u mvm.service -f

EXAMPLES:
    mvm service install --now      # install and start now (+ at login)
    mvm service status
    mvm service uninstall
";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wants_help_detects_both_spellings() {
        let long: Vec<String> = ["mvm", "run", "--help"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let short: Vec<String> = ["mvm", "recap", "-h"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let none: Vec<String> = ["mvm", "run", "--masks"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert!(wants_help(&long));
        assert!(wants_help(&short));
        assert!(!wants_help(&none));
    }

    #[test]
    fn version_only_at_top_level() {
        assert!(wants_version(Some("--version")));
        assert!(wants_version(Some("-V")));
        assert!(!wants_version(Some("run")));
        assert!(!wants_version(None));
    }

    #[test]
    fn mesh_is_a_hidden_alias_of_run_help() {
        // `mvm mesh --help` and `mvm run --help` show the same screen.
        assert_eq!(subcommand("mesh"), subcommand("run"));
        assert_eq!(subcommand("run"), Some(RUN));
    }

    /// Every advertised subcommand has a help screen, and each screen names its
    /// own command — a golden that breaks if a screen is dropped or mislabelled.
    #[test]
    fn every_subcommand_has_help() {
        for (name, needle) in [
            ("run", "mvm run"),
            ("snapshot", "mvm snapshot"),
            ("recap", "mvm recap"),
            ("inspect", "mvm inspect"),
            ("serve", "mvm serve"),
        ] {
            let text = subcommand(name).unwrap_or_else(|| panic!("no help for {name}"));
            assert!(text.contains(needle), "{name} help must name `{needle}`");
        }
        assert!(subcommand("frobnicate").is_none());
    }

    /// The top-level screen must list every command so `mvm --help` is a complete map.
    #[test]
    fn main_help_lists_all_commands() {
        for cmd in ["run", "snapshot", "cortex", "recap", "inspect", "serve"] {
            assert!(MAIN.contains(cmd), "top-level help must list `{cmd}`");
        }
    }

    #[test]
    fn main_help_explains_the_producer_reader_authority_boundary() {
        for needle in [
            "START HERE:",
            "mvm run",
            "current_state.md",
            "labels.jsonl",
            "current_cortex.json",
            "never replaces facts",
            "mvm_snapshot, mvm_cortex, mvm_recap, mvm_inspect",
        ] {
            assert!(
                MAIN.contains(needle),
                "top-level help must explain `{needle}`"
            );
        }
    }

    #[test]
    fn cortex_and_mcp_help_preserve_the_cognitive_boundary() {
        assert!(CORTEX.contains("never\nreplaces canonical scene facts"));
        assert!(CORTEX.contains("withheld/unknown"));
        assert!(SERVE.contains("mvm_cortex"));
        assert!(SERVE.contains("does not alter\ncurrent_state.md or labels.jsonl"));
    }
}
