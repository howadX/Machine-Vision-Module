//! `mvm mcp install --host <host>` (phase 14.4) — register MVM as an MCP server
//! with an LLM host so a human never hand-edits JSON/YAML.
//!
//! Each host has a real, verified config location and shape (checked 2026-07-06):
//!   * **claude-desktop** — `~/.config/Claude/claude_desktop_config.json`,
//!     JSON `{ "mcpServers": { "<name>": { command, args } } }`.
//!   * **claude-code** — `~/.claude.json`, JSON top-level `mcpServers` (same shape).
//!   * **hermes** — `~/.hermes/config.yaml`, YAML top-level `mcp_servers:` with
//!     `command / args / transport / timeout / connect_timeout`.
//!
//! Default is a **dry run**: print the exact block and the file it belongs in.
//! `--write` merges it in for the JSON hosts (backing the file up to `.bak`,
//! preserving every other key). Hermes config is YAML and the owner has asked
//! that it not be edited without confirmation, so `--write` there prints and
//! points at the block instead of touching the file.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use serde_json::{json, Value};

use crate::config_file::ProductConfig;

#[derive(Clone, Copy, PartialEq)]
pub enum Host {
    ClaudeDesktop,
    ClaudeCode,
    Hermes,
}

impl Host {
    fn parse(s: &str) -> Result<Host> {
        Ok(match s {
            "claude-desktop" => Host::ClaudeDesktop,
            "claude-code" => Host::ClaudeCode,
            "hermes" => Host::Hermes,
            other => bail!(
                "unknown host {other:?} — use claude-desktop | claude-code | hermes"
            ),
        })
    }

    fn is_json(self) -> bool {
        matches!(self, Host::ClaudeDesktop | Host::ClaudeCode)
    }

    /// The config file this host reads MCP servers from.
    fn config_path(self, home: &Path) -> PathBuf {
        match self {
            // Linux path (macOS is ~/Library/Application Support/Claude/…) — the
            // paths seam stays Linux-first, matching `paths.rs`.
            Host::ClaudeDesktop => home.join(".config/Claude/claude_desktop_config.json"),
            Host::ClaudeCode => home.join(".claude.json"),
            Host::Hermes => home.join(".hermes/config.yaml"),
        }
    }

    fn label(self) -> &'static str {
        match self {
            Host::ClaudeDesktop => "claude-desktop",
            Host::ClaudeCode => "claude-code",
            Host::Hermes => "hermes",
        }
    }
}

pub struct InstallArgs {
    host: Host,
    /// Embed the producer mesh in the server (`serve --mcp --with-camera`), so
    /// the host launches a self-contained vision module.
    with_camera: bool,
    /// Actually merge into the config file (JSON hosts only).
    write: bool,
}

pub fn parse_install_args(raw: &[String]) -> Result<InstallArgs> {
    // raw = ["mvm", "mcp", "install", "--host", "<h>", ...]
    let mut host: Option<Host> = None;
    let mut with_camera = false;
    let mut write = false;
    let mut i = 3.min(raw.len());
    while i < raw.len() {
        match raw[i].as_str() {
            "--host" if i + 1 < raw.len() => {
                host = Some(Host::parse(&raw[i + 1])?);
                i += 1;
            }
            "--with-camera" => with_camera = true,
            "--write" => write = true,
            _ => {}
        }
        i += 1;
    }
    let host = host.context("missing --host (claude-desktop | claude-code | hermes)")?;
    Ok(InstallArgs { host, with_camera, write })
}

/// The `serve` invocation the host should launch. Explicit `--logs`/`--state`
/// make it env-independent (the host may run with a different HOME).
fn server_args(product: &ProductConfig, with_camera: bool) -> Vec<String> {
    let mut a = vec!["serve".into(), "--mcp".into()];
    if with_camera {
        a.push("--with-camera".into());
    }
    a.push("--logs".into());
    a.push(product.data_dir.display().to_string());
    a.push("--state".into());
    a.push(product.state_path.display().to_string());
    a
}

/// Entry point for `mvm mcp install`.
pub fn run_install(product: &ProductConfig, args: InstallArgs) -> Result<()> {
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
    let path = args.host.config_path(&home);
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "mvm".into());
    let srv_args = server_args(product, args.with_camera);

    if args.host.is_json() {
        let block = render_json_block(&exe, &srv_args);
        eprintln!("mvm mcp install — host {} → {}", args.host.label(), path.display());
        eprintln!("{block}");
        if args.write {
            merge_json_file(&path, "mvm", &exe, &srv_args)?;
            eprintln!("✓ merged into {} (backup at {})", path.display(), backup_path(&path).display());
        } else {
            eprintln!("(dry run — pass --write to merge this into the file above)");
        }
    } else {
        // Hermes YAML — print only (owner keeps Hermes config hand-edited).
        let block = render_hermes_block(&exe, &srv_args);
        eprintln!("mvm mcp install — host hermes → {}", path.display());
        eprintln!("Add under the top-level `mcp_servers:` key:\n{block}");
        if args.write {
            bail!(
                "hermes config is YAML — paste the block above under `mcp_servers:` \
                 in {} (auto-write is intentionally not supported for Hermes)",
                path.display()
            );
        }
    }
    Ok(())
}

/// The server entry as a serde_json object — the shared shape for both JSON
/// hosts and the source of truth for the merge.
fn server_entry(exe: &str, args: &[String]) -> Value {
    json!({ "command": exe, "args": args })
}

/// Pretty JSON block a human can paste (the whole `mcpServers` wrapper).
fn render_json_block(exe: &str, args: &[String]) -> String {
    let v = json!({ "mcpServers": { "mvm": server_entry(exe, args) } });
    serde_json::to_string_pretty(&v).unwrap_or_default()
}

/// YAML block for Hermes' `mcp_servers:` (hand-rendered — no yaml dep).
fn render_hermes_block(exe: &str, args: &[String]) -> String {
    let args_yaml = args.iter().map(|a| format!("\"{a}\"")).collect::<Vec<_>>().join(", ");
    format!(
        "  mvm:\n    command: {exe}\n    args: [{args_yaml}]\n    transport: stdio\n    timeout: 120\n    connect_timeout: 30"
    )
}

/// Merge `{mcpServers: {name: entry}}` into the JSON file at `path`, preserving
/// every other key. Backs the original up to `<path>.bak` first. Creating a new
/// file (host config absent) is fine.
fn merge_json_file(path: &Path, name: &str, exe: &str, args: &[String]) -> Result<()> {
    let existing = std::fs::read_to_string(path).ok();
    if let Some(text) = &existing {
        std::fs::write(backup_path(path), text)
            .with_context(|| format!("back up {}", path.display()))?;
    } else if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut root: Value = match existing.as_deref() {
        Some(t) if !t.trim().is_empty() => {
            serde_json::from_str(t).with_context(|| format!("parse {}", path.display()))?
        }
        _ => json!({}),
    };
    merge_entry(&mut root, name, exe, args)?;
    let out = serde_json::to_string_pretty(&root)?;
    std::fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

/// `<path>.bak` — appended, not extension-replaced, so `.claude.json` backs up
/// to `.claude.json.bak` (not `.claude.bak`).
fn backup_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".bak");
    PathBuf::from(s)
}

/// Insert/replace `root.mcpServers.<name>` = entry. Pure — the unit under test.
fn merge_entry(root: &mut Value, name: &str, exe: &str, args: &[String]) -> Result<()> {
    if !root.is_object() {
        bail!("host config root is not a JSON object");
    }
    let obj = root.as_object_mut().unwrap();
    let servers = obj
        .entry("mcpServers")
        .or_insert_with(|| json!({}));
    if !servers.is_object() {
        bail!("`mcpServers` in host config is not an object");
    }
    servers
        .as_object_mut()
        .unwrap()
        .insert(name.to_string(), server_entry(exe, args));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_parse_and_paths() {
        let home = Path::new("/home/u");
        assert!(Host::parse("nope").is_err());
        assert_eq!(
            Host::ClaudeDesktop.config_path(home),
            PathBuf::from("/home/u/.config/Claude/claude_desktop_config.json")
        );
        assert_eq!(Host::ClaudeCode.config_path(home), PathBuf::from("/home/u/.claude.json"));
        assert_eq!(Host::Hermes.config_path(home), PathBuf::from("/home/u/.hermes/config.yaml"));
        assert!(Host::ClaudeCode.is_json() && !Host::Hermes.is_json());
    }

    #[test]
    fn server_args_are_data_dir_aware() {
        let mut product = crate::config_file::ProductConfig {
            camera_device: "auto".into(),
            camera_width: 1280,
            camera_height: 720,
            data_dir: PathBuf::from("/d"),
            state_path: PathBuf::from("/d/current_state.md"),
            models_dir: PathBuf::from("/d/models"),
            gallery_dir: PathBuf::from("/g"),
            snapshot_ms: 1000,
        };
        let a = server_args(&product, false);
        assert_eq!(a, vec!["serve", "--mcp", "--logs", "/d", "--state", "/d/current_state.md"]);
        product.snapshot_ms = 500;
        let a = server_args(&product, true);
        assert!(a.contains(&"--with-camera".to_string()));
    }

    #[test]
    fn merge_into_empty_creates_mcpservers() {
        let mut root = json!({});
        merge_entry(&mut root, "mvm", "/bin/mvm", &["serve".into()]).unwrap();
        assert_eq!(root["mcpServers"]["mvm"]["command"], "/bin/mvm");
        assert_eq!(root["mcpServers"]["mvm"]["args"][0], "serve");
    }

    #[test]
    fn merge_preserves_other_keys_and_servers() {
        let mut root = json!({
            "numStartups": 42,
            "mcpServers": { "other": { "command": "other-server" } }
        });
        merge_entry(&mut root, "mvm", "/bin/mvm", &["serve".into()]).unwrap();
        // Unrelated top-level key untouched.
        assert_eq!(root["numStartups"], 42);
        // Existing server untouched, new one added.
        assert_eq!(root["mcpServers"]["other"]["command"], "other-server");
        assert_eq!(root["mcpServers"]["mvm"]["command"], "/bin/mvm");
    }

    #[test]
    fn merge_replaces_existing_mvm_idempotently() {
        let mut root = json!({ "mcpServers": { "mvm": { "command": "old", "args": [] } } });
        merge_entry(&mut root, "mvm", "/new/mvm", &["serve".into(), "--mcp".into()]).unwrap();
        assert_eq!(root["mcpServers"]["mvm"]["command"], "/new/mvm");
        assert_eq!(root["mcpServers"].as_object().unwrap().len(), 1, "no duplicate");
    }

    #[test]
    fn merge_rejects_non_object_mcpservers() {
        let mut root = json!({ "mcpServers": "oops" });
        assert!(merge_entry(&mut root, "mvm", "/bin/mvm", &[]).is_err());
    }

    #[test]
    fn merge_file_backs_up_and_writes() {
        let dir = std::env::temp_dir().join(format!("mvm_mcpinstall_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // Use a double-extension name like the real .claude.json to catch the
        // with_extension vs append-.bak bug.
        let path = dir.join(".claude.json");
        std::fs::write(&path, r#"{"numStartups":7}"#).unwrap();
        merge_json_file(&path, "mvm", "/bin/mvm", &["serve".into()]).unwrap();
        // Backup is <name>.bak (appended), holds the original.
        assert_eq!(std::fs::read_to_string(backup_path(&path)).unwrap(), r#"{"numStartups":7}"#);
        let merged: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(merged["numStartups"], 7);
        assert_eq!(merged["mcpServers"]["mvm"]["command"], "/bin/mvm");
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn hermes_block_is_valid_shape() {
        let b = render_hermes_block("/bin/mvm", &["serve".into(), "--mcp".into()]);
        assert!(b.contains("mvm:"));
        assert!(b.contains("command: /bin/mvm"));
        assert!(b.contains("transport: stdio"));
        assert!(b.contains(r#"["serve", "--mcp"]"#));
    }
}
