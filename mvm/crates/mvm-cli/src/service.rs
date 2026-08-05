//! `mvm service <install|uninstall|status>` (phase 14.5) — run the vision
//! producer as a **systemd user** service (no root).
//!
//! `install` writes `~/.config/systemd/user/mvm.service` (ExecStart = this
//! machine's `mvm run`) and reloads the user daemon. By default it does NOT
//! start the camera — pass `--now` to `enable --now`. `uninstall` disables +
//! removes the unit; `status` shows whether it's enabled/active.
//!
//! systemd absent (non-Linux, or no `systemctl --user`) → `install` still
//! writes the unit and prints the manual commands, so nothing is silently lost.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail};

const UNIT_NAME: &str = "mvm.service";

/// The systemd user unit contents for `mvm run` at `exe`.
///
/// The executable path is double-quoted: systemd splits `ExecStart` on
/// whitespace, and this project's install path can contain spaces (e.g.
/// `.../Machine Vision Module/...`) — unquoted, systemd would try to exec the
/// first word only. `run` stays outside the quotes as a separate argument.
pub fn unit_content(exe: &str) -> String {
    format!(
        "[Unit]\n\
         Description=MVM — Machine Vision Module (vision producer)\n\
         After=graphical-session.target\n\
         \n\
         [Service]\n\
         Type=simple\n\
         ExecStart=\"{exe}\" run\n\
         Restart=on-failure\n\
         RestartSec=5\n\
         StandardOutput=journal\n\
         StandardError=journal\n\
         \n\
         [Install]\n\
         WantedBy=default.target\n"
    )
}

pub fn unit_path(home: &Path) -> PathBuf {
    home.join(".config/systemd/user").join(UNIT_NAME)
}

pub fn run_service(raw: &[String]) -> Result<()> {
    let home = std::env::var("HOME").map(PathBuf::from).unwrap_or_default();
    match raw.get(2).map(String::as_str) {
        Some("install") => install(&home, raw.iter().any(|a| a == "--now")),
        Some("uninstall") => uninstall(&home),
        Some("status") => status(&home),
        other => bail!(
            "unknown service command {:?} — use install [--now] | uninstall | status",
            other.unwrap_or("")
        ),
    }
}

fn install(home: &Path, now: bool) -> Result<()> {
    let exe = std::env::current_exe()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "mvm".into());
    let path = unit_path(home);
    std::fs::create_dir_all(path.parent().unwrap())
        .with_context(|| format!("create {}", path.parent().unwrap().display()))?;
    std::fs::write(&path, unit_content(&exe))
        .with_context(|| format!("write {}", path.display()))?;
    eprintln!("✓ wrote {}", path.display());

    if !systemctl_available() {
        eprintln!(
            "systemctl --user not available — enable it yourself with:\n  \
             systemctl --user daemon-reload && systemctl --user enable --now {UNIT_NAME}"
        );
        return Ok(());
    }

    run_systemctl(&["daemon-reload"])?;
    if now {
        run_systemctl(&["enable", "--now", UNIT_NAME])?;
        eprintln!("✓ enabled + started {UNIT_NAME} (mvm run is producing now)");
        eprintln!("  logs: journalctl --user -u {UNIT_NAME} -f");
    } else {
        eprintln!("service installed but not started. Start it with:");
        eprintln!("  systemctl --user enable --now {UNIT_NAME}");
        eprintln!("  (or re-run `mvm service install --now`)");
    }
    Ok(())
}

fn uninstall(home: &Path) -> Result<()> {
    let path = unit_path(home);
    if systemctl_available() {
        // Best-effort: the unit may not be enabled/running.
        let _ = Command::new("systemctl")
            .args(["--user", "disable", "--now", UNIT_NAME])
            .status();
    }
    if path.exists() {
        std::fs::remove_file(&path).with_context(|| format!("remove {}", path.display()))?;
        eprintln!("✓ removed {}", path.display());
    } else {
        eprintln!("no unit at {} (nothing to remove)", path.display());
    }
    if systemctl_available() {
        let _ = run_systemctl(&["daemon-reload"]);
    }
    Ok(())
}

fn status(home: &Path) -> Result<()> {
    let path = unit_path(home);
    eprintln!(
        "unit: {} ({})",
        path.display(),
        if path.exists() { "present" } else { "absent" }
    );
    if !systemctl_available() {
        eprintln!("systemctl --user not available");
        return Ok(());
    }
    for prop in ["is-enabled", "is-active"] {
        let out = Command::new("systemctl")
            .args(["--user", prop, UNIT_NAME])
            .output();
        let val = out
            .ok()
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into());
        eprintln!("  {prop}: {val}");
    }
    Ok(())
}

fn systemctl_available() -> bool {
    Command::new("systemctl")
        .args(["--user", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn run_systemctl(args: &[&str]) -> Result<()> {
    let status = Command::new("systemctl")
        .arg("--user")
        .args(args)
        .status()
        .with_context(|| format!("run systemctl --user {}", args.join(" ")))?;
    if !status.success() {
        bail!("systemctl --user {} failed", args.join(" "));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_content_runs_mvm_run_and_installs_to_default_target() {
        // Path is quoted so a space in the install dir (this repo has one!)
        // doesn't make systemd exec only the first word.
        let u = unit_content("/opt/with space/mvm");
        assert!(u.contains("ExecStart=\"/opt/with space/mvm\" run\n"), "{u}");
        assert!(u.contains("WantedBy=default.target"), "{u}");
        assert!(u.contains("Restart=on-failure"), "{u}");
        // Valid unit shape: the three sections present, in order.
        let unit = u.find("[Unit]").unwrap();
        let service = u.find("[Service]").unwrap();
        let install = u.find("[Install]").unwrap();
        assert!(unit < service && service < install, "sections ordered");
    }

    #[test]
    fn unit_path_is_user_systemd_dir() {
        assert_eq!(
            unit_path(Path::new("/home/u")),
            PathBuf::from("/home/u/.config/systemd/user/mvm.service")
        );
    }
}
