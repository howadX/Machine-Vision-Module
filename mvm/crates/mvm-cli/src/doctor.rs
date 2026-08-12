//! `mvm doctor` (phase 14.0.6) — answer "why doesn't it work" in one screen,
//! and prove the de-hardcoding resolves correctly: which camera, which models
//! dir (and what's missing), where data goes, which config was read.

use std::path::Path;

use crate::config_file::ProductConfig;
use crate::manifest::MODELS;

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

/// Count (present, total) over all manifest models in `dir`, plus the list of
/// missing *required* ones.
pub fn models_tally(dir: &Path) -> (usize, usize, Vec<String>) {
    let mut present = 0;
    let mut missing_required: Vec<String> = Vec::new();
    for m in MODELS {
        if dir.join(m.file).exists() {
            present += 1;
        } else if m.required {
            missing_required.push(m.file.to_string());
        }
    }
    (present, MODELS.len(), missing_required)
}

/// Probe everything against the resolved product config and build the report.
/// Camera probing actually opens the device (the honest check) — a busy camera
/// (another `mvm run`) reports as such rather than pretending.
pub fn run_doctor(product: &ProductConfig) -> Vec<Check> {
    let mut checks = Vec::new();

    checks.extend(sensor_checks(product));

    // Models: required missing → warn with the setup pointer.
    let (present, total, missing) = models_tally(&product.models_dir);
    checks.push(if missing.is_empty() {
        Check {
            name: "models",
            status: Status::Ok,
            detail: format!(
                "{present}/{total} present in {}",
                product.models_dir.display()
            ),
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

fn sensor_checks(product: &ProductConfig) -> Vec<Check> {
    if product.sensor_backend != "kinect_v1" {
        return vec![match mvm_pipeline::source::open_source(
            &product.camera_device,
            product.camera_width,
            product.camera_height,
        ) {
            Ok(cam) => {
                let (w, h) = cam.dimensions();
                Check {
                    name: "sensor",
                    status: Status::Ok,
                    detail: format!("v4l2 {} ({w}×{h} MJPG)", cam.device()),
                }
            }
            Err(e) => Check {
                name: "sensor",
                status: Status::Fail,
                detail: format!("v4l2: {e:#}"),
            },
        }];
    }

    #[cfg(not(feature = "kinect-v1"))]
    return vec![Check {
        name: "sensor",
        status: Status::Fail,
        detail: "kinect_v1 configured but binary lacks feature kinect-v1".into(),
    }];

    #[cfg(feature = "kinect-v1")]
    {
        let config = product.kinect_v1_config();
        let mut checks = match mvm_pipeline::kinect_v1::probe(&config) {
            Ok(probe) => vec![
                Check {
                    name: "runtime",
                    status: Status::Ok,
                    detail: format!("libfreenect 0.7.5 at {}", probe.library_path.display()),
                },
                Check {
                    name: "calibration",
                    status: Status::Ok,
                    detail: format!(
                        "serial {} revision {}",
                        probe.calibration.camera_serial,
                        &probe.calibration.revision_sha256[..12]
                    ),
                },
                Check {
                    name: "usb",
                    status: if probe.usb_camera_present {
                        Status::Ok
                    } else {
                        Status::Fail
                    },
                    detail: if probe.usb_camera_present {
                        format!("camera serial {} present", config.serial)
                    } else {
                        format!("camera serial {} not present", config.serial)
                    },
                },
            ],
            Err(error) => {
                return vec![Check {
                    name: "sensor",
                    status: Status::Fail,
                    detail: format!("kinect_v1: {error:#}"),
                }];
            }
        };
        checks.push(
            match mvm_pipeline::kinect_v1::KinectV1Source::open(config) {
                Ok(source) => {
                    drop(source);
                    Check {
                        name: "ownership",
                        status: Status::Ok,
                        detail:
                            "camera opened by serial and cleanly released; motor/audio untouched"
                                .into(),
                    }
                }
                Err(error) => Check {
                    name: "ownership",
                    status: Status::Fail,
                    detail: format!("camera unavailable or busy: {error:#}"),
                },
            },
        );
        checks
    }
}

fn data_dir_writable(dir: &Path) -> std::io::Result<()> {
    crate::paths::ensure_dir_private(dir)?;
    let probe = dir.join(".doctor_write_probe");
    std::fs::write(&probe, b"ok")?;
    std::fs::remove_file(&probe)
}

pub fn print_report(product: &ProductConfig) {
    print!("{}", render(&run_doctor(product)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn render_formats_aligned_rows_with_icons() {
        let checks = vec![
            Check {
                name: "camera",
                status: Status::Ok,
                detail: "/dev/video2 (1280×720 MJPG)".into(),
            },
            Check {
                name: "models",
                status: Status::Warn,
                detail: "3/9 present (run `mvm setup`)".into(),
            },
            Check {
                name: "data dir",
                status: Status::Fail,
                detail: "/nope: permission denied".into(),
            },
        ];
        let out = render(&checks);
        assert!(out.starts_with("mvm doctor\n"), "{out}");
        assert!(
            out.contains("  camera     ✅ /dev/video2 (1280×720 MJPG)\n"),
            "{out}"
        );
        assert!(out.contains("models"), "{out}");
        assert!(out.contains("⚠️"), "{out}");
        assert!(out.contains("❌ /nope"), "{out}");
    }

    #[test]
    fn models_tally_counts_against_the_manifest() {
        let dir = std::env::temp_dir().join(format!("mvm_doctor_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        // One required + one optional model present, from the real manifest.
        std::fs::write(dir.join("face_recognition_sface_2021dec.onnx"), b"x").unwrap();
        std::fs::write(dir.join("yolo_seg.onnx"), b"x").unwrap();
        let (present, total, missing) = models_tally(&dir);
        assert_eq!(present, 2);
        assert_eq!(total, MODELS.len());
        // The other four required models are reported missing (order = manifest).
        assert!(missing.contains(&"face_detection_yunet_2023mar.onnx".to_string()));
        assert!(
            !missing.contains(&"yolo_seg.onnx".to_string()),
            "optional not flagged"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn models_tally_missing_dir_is_all_required_missing() {
        let (present, total, missing) = models_tally(&PathBuf::from("/definitely/not/here"));
        assert_eq!(present, 0);
        assert_eq!(total, MODELS.len());
        assert_eq!(missing.len(), MODELS.iter().filter(|m| m.required).count());
    }
}
