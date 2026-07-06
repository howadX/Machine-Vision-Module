//! `mvm doctor` (phase 14.0.6) — answer "why doesn't it work" in one screen,
//! and prove the de-hardcoding resolves correctly: which camera, which models
//! dir (and what's missing), where data goes, which config was read.

use std::path::Path;

use crate::config_file::ProductConfig;

/// Optional experts — absent models degrade gracefully (agent disabled), so
/// they count toward the tally but never fail the check.
const OPTIONAL_MODELS: &[&str] = &[
    "yolo_object_detect.onnx",
    "yolo_seg.onnx",
    "depth_anything_v2_small.onnx",
    "handpose_estimation_mediapipe_2023feb.onnx",
];

pub enum Status {
    Ok,
    Warn,
    Fail,
}

pub struct Check {
    pub name: &'static str,
    pub status: Status,
    pub detail: String,
}

/// Render the report exactly as printed — pure, golden-testable.
pub fn render(checks: &[Check]) -> String {
    let mut out = String::from("mvm doctor\n");
    for c in checks {
        let icon = match c.status {
            Status::Ok => "✅",
            Status::Warn => "⚠️ ",
            Status::Fail => "❌",
        };
        out.push_str(&format!("  {:<10} {icon} {}\n", c.name, c.detail));
    }
    out
}

/// Count (present, total) over required + optional models in `dir`.
pub fn models_tally(dir: &Path, required: &[&str]) -> (usize, usize, Vec<String>) {
    let mut present = 0;
    let mut missing_required: Vec<String> = Vec::new();
    let total = required.len() + OPTIONAL_MODELS.len();
    for f in required {
        if dir.join(f).exists() {
            present += 1;
        } else {
            missing_required.push((*f).to_string());
        }
    }
    for f in OPTIONAL_MODELS {
        if dir.join(f).exists() {
            present += 1;
        }
    }
    (present, total, missing_required)
}

/// Probe everything against the resolved product config and build the report.
/// Camera probing actually opens the device (the honest check) — a busy camera
/// (another `mvm run`) reports as such rather than pretending.
pub fn run_doctor(product: &ProductConfig, required_models: &[&str]) -> Vec<Check> {
    let mut checks = Vec::new();

    // Camera: auto → scan; explicit → open that one.
    checks.push(
        match mvm_pipeline::source::open_source(
            &product.camera_device,
            product.camera_width,
            product.camera_height,
        ) {
            Ok(cam) => {
                let (w, h) = cam.dimensions();
                Check {
                    name: "camera",
                    status: Status::Ok,
                    detail: format!("{} ({w}×{h} MJPG)", cam.device()),
                }
            }
            Err(e) => Check {
                name: "camera",
                status: Status::Fail,
                detail: format!("{e:#}"),
            },
        },
    );

    // Models: required missing → warn with the setup pointer.
    let (present, total, missing) = models_tally(&product.models_dir, required_models);
    checks.push(if missing.is_empty() {
        Check {
            name: "models",
            status: Status::Ok,
            detail: format!("{present}/{total} present in {}", product.models_dir.display()),
        }
    } else {
        Check {
            name: "models",
            status: Status::Warn,
            detail: format!(
                "{present}/{total} present in {} — missing required: {} (run `mvm setup`)",
                product.models_dir.display(),
                missing.join(", ")
            ),
        }
    });

    // Data dir: creatable + writable (probe with a real write).
    checks.push(match data_dir_writable(&product.data_dir) {
        Ok(()) => Check {
            name: "data dir",
            status: Status::Ok,
            detail: format!("{} (writable)", product.data_dir.display()),
        },
        Err(e) => Check {
            name: "data dir",
            status: Status::Fail,
            detail: format!("{}: {e}", product.data_dir.display()),
        },
    });

    // Config: present file vs pure defaults — both healthy, but say which.
    let config_path = crate::paths::config_dir().join("config.toml");
    checks.push(Check {
        name: "config",
        status: Status::Ok,
        detail: if config_path.exists() {
            format!("{}", config_path.display())
        } else {
            format!("defaults (no {})", config_path.display())
        },
    });

    checks
}

fn data_dir_writable(dir: &Path) -> std::io::Result<()> {
    crate::paths::ensure_dir_private(dir)?;
    let probe = dir.join(".doctor_write_probe");
    std::fs::write(&probe, b"ok")?;
    std::fs::remove_file(&probe)
}

pub fn print_report(product: &ProductConfig, required_models: &[&str]) {
    print!("{}", render(&run_doctor(product, required_models)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn render_formats_aligned_rows_with_icons() {
        let checks = vec![
            Check { name: "camera", status: Status::Ok, detail: "/dev/video2 (1280×720 MJPG)".into() },
            Check { name: "models", status: Status::Warn, detail: "3/9 present (run `mvm setup`)".into() },
            Check { name: "data dir", status: Status::Fail, detail: "/nope: permission denied".into() },
        ];
        let out = render(&checks);
        assert!(out.starts_with("mvm doctor\n"), "{out}");
        assert!(out.contains("  camera     ✅ /dev/video2 (1280×720 MJPG)\n"), "{out}");
        assert!(out.contains("models"), "{out}");
        assert!(out.contains("⚠️"), "{out}");
        assert!(out.contains("❌ /nope"), "{out}");
    }

    #[test]
    fn models_tally_counts_required_and_optional() {
        let dir = std::env::temp_dir().join(format!("mvm_doctor_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let required = ["a.onnx", "b.onnx"];
        // One required + one optional present.
        std::fs::write(dir.join("a.onnx"), b"x").unwrap();
        std::fs::write(dir.join("yolo_seg.onnx"), b"x").unwrap();
        let (present, total, missing) = models_tally(&dir, &required);
        assert_eq!(present, 2);
        assert_eq!(total, 2 + OPTIONAL_MODELS.len());
        assert_eq!(missing, vec!["b.onnx".to_string()]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn models_tally_missing_dir_is_all_missing() {
        let (present, _, missing) =
            models_tally(&PathBuf::from("/definitely/not/here"), &["a.onnx"]);
        assert_eq!(present, 0);
        assert_eq!(missing, vec!["a.onnx".to_string()]);
    }
}
