//! MCP stdio server (phase 12.D2) — `mvm-cli serve --mcp`.
//!
//! Lets an LLM host (Hermes / Alter / Claude) call MVM's vision as tools instead
//! of reading files: `mvm_snapshot` (what now), `mvm_recap` (what happened),
//! `mvm_inspect` (raw slice). Hand-rolled JSON-RPC 2.0 over newline-delimited
//! stdin/stdout — **+0 deps** (serde_json only), no HTTP/tokio. **Read-only**:
//! the server owns no camera, it reads the artifacts a running mesh writes
//! (StateFile + scene/events logs), so producer and serving stay decoupled.

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::PathBuf;

pub struct ServeArgs {
    /// Snapshot source — the StateFile (`current_state.md`).
    state: PathBuf,
    /// Recap/inspect source — the `--scene-log` dir (scene-/events-*.jsonl).
    logs: PathBuf,
}

pub fn parse_serve_args(raw: &[String]) -> ServeArgs {
    let mut state = crate::state::default_state_path();
    let mut logs: Option<PathBuf> = None;
    let mut i = 2; // skip program name + "serve"
    while i < raw.len() {
        match raw[i].as_str() {
            "--state" if i + 1 < raw.len() => {
                state = PathBuf::from(&raw[i + 1]);
                i += 1;
            }
            "--logs" if i + 1 < raw.len() => {
                logs = Some(PathBuf::from(&raw[i + 1]));
                i += 1;
            }
            _ => {} // `--mcp` (only transport) and unknown flags are ignored
        }
        i += 1;
    }
    // Default the log dir to the StateFile's directory (journal.md lives there).
    let logs = logs.unwrap_or_else(|| {
        state.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
    });
    ServeArgs { state, logs }
}

pub fn run_serve(args: ServeArgs) -> Result<()> {
    use std::io::{BufRead, Write};
    eprintln!(
        "mvm serve --mcp — reading stdin (state={}, logs={})",
        args.state.display(),
        args.logs.display()
    );
    let stdin = std::io::stdin();
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // ignore malformed frames
        };
        if let Some(resp) = handle(&req, &args) {
            writeln!(out, "{}", serde_json::to_string(&resp)?)?;
            out.flush()?;
        }
    }
    Ok(())
}

fn result(id: Value, result: Value) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "result": result })
}

fn error(id: Value, code: i64, message: &str) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

/// Handle one JSON-RPC message. Returns `None` for notifications (no `id` → no
/// reply). Pure over `(req, args)` so it unit-tests without stdio.
pub(crate) fn handle(req: &Value, args: &ServeArgs) -> Option<Value> {
    let method = req.get("method").and_then(|m| m.as_str())?;
    // Notifications (e.g. `notifications/initialized`) carry no id → no response.
    let id = req.get("id").cloned()?;
    match method {
        "initialize" => Some(result(
            id,
            json!({
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "mvm-vision", "version": env!("CARGO_PKG_VERSION") }
            }),
        )),
        "tools/list" => Some(result(id, json!({ "tools": tool_defs() }))),
        "tools/call" => {
            let params = req.get("params");
            let name = params.and_then(|p| p.get("name")).and_then(|n| n.as_str()).unwrap_or("");
            let arguments =
                params.and_then(|p| p.get("arguments")).cloned().unwrap_or_else(|| json!({}));
            let call = dispatch_tool(name, &arguments, args);
            Some(result(
                id,
                match call {
                    Ok(text) => json!({ "content": [{ "type": "text", "text": text }] }),
                    // Tool failures are results with isError, not protocol errors.
                    Err(e) => json!({
                        "content": [{ "type": "text", "text": format!("error: {e}") }],
                        "isError": true
                    }),
                },
            ))
        }
        "ping" => Some(result(id, json!({}))),
        _ => Some(error(id, -32601, "method not found")),
    }
}

fn tool_defs() -> Value {
    // These three tools are the pull model, cheap → expensive: snapshot (now) →
    // recap (a window) → inspect (raw frames). The descriptions teach an agent
    // when to reach for each and how to read the output; keep them in sync with
    // mvm/AGENT.md (single source of truth, phase 13.C/D).
    json!([
        {
            "name": "mvm_snapshot",
            "title": "Snapshot — current scene",
            "description": "What's happening RIGHT NOW: who is present, where (metric 3D), and what each person is doing — a compact markdown table (~300 tokens). Start here; it's the cheapest call. Confidence is stated in words (low/medium/high), and ⚠️ flags phantom / withheld / low-plausibility notes — never trust a bare label under a ⚠️. Track ids: 0–9 = gallery (known people), ≥10 = anonymous people, ≥10000 = objects. No arguments.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "mvm_recap",
            "title": "Recap — period summary",
            "description": "What HAPPENED over a period, reconstructed from the durable logs: per-person presence spans, activities rolled up across object-id churn, scene objects/zones, and phantom notes. The middle tier — call after a snapshot to summarise a window without reading raw frames. With no arguments it covers the whole current day; narrow it with `last` or `from`/`to`.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "last": { "type": "string", "description": "Window before the latest entry, e.g. '30s', '10m', '2h'. Preferred for 'what just happened'." },
                    "from": { "type": "string", "description": "ISO8601 lower bound (inclusive), e.g. '2026-07-05T17:00:00'." },
                    "to":   { "type": "string", "description": "ISO8601 upper bound (inclusive)." }
                }
            }
        },
        {
            "name": "mvm_inspect",
            "title": "Inspect — raw frames",
            "description": "Raw per-frame slices at full logged precision, around a timestamp and/or for one track id — the expensive ZOOM to adjudicate something snapshot/recap flagged (a phantom, an unstable id, a brief event). Use only after a cheaper call points you at a moment; not for browsing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "around": { "type": "string", "description": "ISO8601 timestamp to centre the slice on, e.g. '2026-07-05T17:51:23'." },
                    "window": { "type": "number", "description": "± seconds around `around` (default 5)." },
                    "id":     { "type": "integer", "description": "Keep only frames with this track id (0–9 gallery · ≥10 anon · ≥10000 objects)." },
                    "md":     { "type": "boolean", "description": "Compact markdown lines (default true); false = raw JSON." }
                }
            }
        }
    ])
}

fn dispatch_tool(name: &str, a: &Value, args: &ServeArgs) -> Result<String> {
    match name {
        "mvm_snapshot" => std::fs::read_to_string(&args.state)
            .with_context(|| format!("read snapshot {}", args.state.display())),
        "mvm_recap" => {
            let last = a.get("last").and_then(|x| x.as_str()).map(String::from);
            let from = a.get("from").and_then(|x| x.as_str()).map(String::from);
            let to = a.get("to").and_then(|x| x.as_str()).map(String::from);
            crate::recap_cmd::recap_markdown(&args.logs, last, from, to)
        }
        "mvm_inspect" => {
            let around = a.get("around").and_then(|x| x.as_str()).map(String::from);
            let window = a.get("window").and_then(|x| x.as_f64()).unwrap_or(5.0);
            let id = a.get("id").and_then(|x| x.as_u64()).map(|n| n as u32);
            let md = a.get("md").and_then(|x| x.as_bool()).unwrap_or(true);
            crate::recap_cmd::inspect_output(&args.logs, around, window, id, md)
        }
        _ => bail!("unknown tool '{name}'"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with_state(path: PathBuf) -> ServeArgs {
        let logs = path.parent().unwrap().to_path_buf();
        ServeArgs { state: path, logs }
    }

    #[test]
    fn initialize_reports_server_info() {
        let a = args_with_state(PathBuf::from("/tmp/nope.md"));
        let req = json!({"jsonrpc":"2.0","id":1,"method":"initialize"});
        let resp = handle(&req, &a).unwrap();
        assert_eq!(resp["result"]["serverInfo"]["name"], "mvm-vision");
        assert!(resp["result"]["capabilities"]["tools"].is_object());
    }

    #[test]
    fn tools_list_advertises_three_tools() {
        let a = args_with_state(PathBuf::from("/tmp/nope.md"));
        let req = json!({"jsonrpc":"2.0","id":2,"method":"tools/list"});
        let resp = handle(&req, &a).unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"mvm_snapshot"));
        assert!(names.contains(&"mvm_recap"));
        assert!(names.contains(&"mvm_inspect"));
    }

    /// Phase 13.D: tool descriptions must stay self-teaching — each carries its
    /// place in the pull model and every tool has a human title. A regression to
    /// terse one-liners breaks this.
    #[test]
    fn tool_descriptions_teach_the_pull_model() {
        let a = args_with_state(PathBuf::from("/tmp/nope.md"));
        let req = json!({"jsonrpc":"2.0","id":9,"method":"tools/list"});
        let resp = handle(&req, &a).unwrap();
        let tools = resp["result"]["tools"].as_array().unwrap();
        let tool = |name: &str| tools.iter().find(|t| t["name"] == name).cloned().unwrap();

        for name in ["mvm_snapshot", "mvm_recap", "mvm_inspect"] {
            assert!(tool(name)["title"].is_string(), "{name} needs a title");
        }
        let snap = tool("mvm_snapshot");
        let snap = snap["description"].as_str().unwrap();
        assert!(snap.contains("NOW") && snap.contains("cheapest"), "snapshot frames itself as the cheap now-call");
        assert!(snap.contains("0–9") && snap.contains("⚠️"), "snapshot explains id ranges + honesty notes");
        let recap = tool("mvm_recap");
        assert!(
            recap["description"].as_str().unwrap().contains("HAPPENED")
                && recap["description"].as_str().unwrap().contains("after a snapshot"),
            "recap frames itself as the middle tier"
        );
        let insp = tool("mvm_inspect");
        assert!(
            insp["description"].as_str().unwrap().contains("ZOOM")
                && insp["description"].as_str().unwrap().contains("adjudicate"),
            "inspect frames itself as the expensive zoom"
        );
    }

    #[test]
    fn notification_gets_no_response() {
        let a = args_with_state(PathBuf::from("/tmp/nope.md"));
        // No `id` → a notification → no reply.
        let req = json!({"jsonrpc":"2.0","method":"notifications/initialized"});
        assert!(handle(&req, &a).is_none());
    }

    #[test]
    fn tools_call_snapshot_returns_file_content() {
        let path = std::env::temp_dir().join(format!("mvm_mcp_snap_{}.md", std::process::id()));
        std::fs::write(&path, "# Scene 17:51:19 · occupied · anomaly 0.00 · 18 fps\n").unwrap();
        let a = args_with_state(path.clone());
        let req = json!({
            "jsonrpc":"2.0","id":3,"method":"tools/call",
            "params":{"name":"mvm_snapshot","arguments":{}}
        });
        let resp = handle(&req, &a).unwrap();
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("# Scene 17:51:19"), "got: {text}");
        assert!(resp["result"].get("isError").is_none());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn tools_call_missing_snapshot_is_tool_error() {
        let a = args_with_state(PathBuf::from("/tmp/definitely_absent_mvm.md"));
        let req = json!({
            "jsonrpc":"2.0","id":4,"method":"tools/call",
            "params":{"name":"mvm_snapshot","arguments":{}}
        });
        let resp = handle(&req, &a).unwrap();
        // A missing file is an isError result, not a JSON-RPC protocol error.
        assert_eq!(resp["result"]["isError"], true);
        assert!(resp.get("error").is_none());
    }

    #[test]
    fn unknown_method_is_protocol_error() {
        let a = args_with_state(PathBuf::from("/tmp/nope.md"));
        let req = json!({"jsonrpc":"2.0","id":5,"method":"frobnicate"});
        let resp = handle(&req, &a).unwrap();
        assert_eq!(resp["error"]["code"], -32601);
    }
}
