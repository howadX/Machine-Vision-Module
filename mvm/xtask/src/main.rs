//! xtask — build tasks for MVM.
//!
//! Usage:
//!   cargo xtask bundle    — build mvm with embedded models (~260KB binary + models)

use std::process::Command;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(|s| s.as_str()) {
        Some("bundle") => bundle(),
        Some(cmd) => {
            eprintln!("Unknown xtask command: {cmd}");
            eprintln!("Available: bundle");
            std::process::exit(1);
        }
        None => {
            eprintln!("Usage: cargo xtask <command>");
            eprintln!("Available: bundle");
        }
    }
}

fn bundle() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let workspace_root = std::path::Path::new(&manifest_dir).parent().unwrap();

    eprintln!("xtask: building mvm with embedded models...");

    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "--features", "embed-models",
            "-p", "mvm-cli",
        ])
        .current_dir(workspace_root)
        .status()
        .expect("Failed to run cargo build");

    if !status.success() {
        eprintln!("xtask: build failed");
        std::process::exit(1);
    }

    let dist_dir = workspace_root.join("dist");
    std::fs::create_dir_all(&dist_dir).unwrap();

    // The mvm-cli package builds a bin named `mvm` (phase 14.0.5).
    let src = workspace_root.join("target/release/mvm");
    let dst = dist_dir.join("mvm");
    std::fs::copy(&src, &dst).unwrap();

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&dst).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&dst, perms).unwrap();
    }

    let size = std::fs::metadata(&dst).unwrap().len();
    eprintln!("xtask: bundled → {} ({size} bytes)", dst.display());
    println!("Done: dist/mvm ({size} bytes)");
}

#[cfg(test)]
mod tests {
    #[test]
    fn xtask_compiles() {
        // Ensures xtask crate is valid
    }
}
