//! `CameraSource` — the platform camera seam (phase 14.0.3).
//!
//! The CLI opens cameras through this trait, never through a concrete backend.
//! Today the only backend is [`V4l2Source`] (the phase-7 `NativeCamera`);
//! macOS (`AvFoundationSource`) and Windows (`MediaFoundationSource`) later
//! land here without touching any call site.

use anyhow::{Result, bail};
use image::DynamicImage;

use crate::native::NativeCamera;

/// A camera the pipeline can read frames from. Object-safe so the CLI holds a
/// `Box<dyn CameraSource>`.
pub trait CameraSource: Send {
    /// Capture and decode the next frame (blocking at the camera's tempo).
    fn read(&mut self) -> Result<DynamicImage>;
    /// Negotiated (width, height).
    fn dimensions(&self) -> (u32, u32);
    /// Human-readable device name (e.g. "/dev/video2") for logs/doctor.
    fn device(&self) -> &str;
}

/// The Linux v4l2 MJPG backend — wraps [`NativeCamera`] and remembers which
/// device it opened.
pub struct V4l2Source {
    cam: NativeCamera,
    device: String,
}

impl V4l2Source {
    pub fn open(device: &str, width: u32, height: u32, rotate_180: bool) -> Result<Self> {
        let cam = NativeCamera::open(device, width, height, rotate_180)?;
        Ok(Self {
            cam,
            device: device.to_string(),
        })
    }
}

impl CameraSource for V4l2Source {
    fn read(&mut self) -> Result<DynamicImage> {
        self.cam.read()
    }
    fn dimensions(&self) -> (u32, u32) {
        self.cam.dimensions()
    }
    fn device(&self) -> &str {
        &self.device
    }
}

/// Open `device` as a boxed [`CameraSource`]. `"auto"` scans `/dev/video*` in
/// numeric order and takes the first device that negotiates MJPG; anything
/// else is opened literally.
pub fn open_source(device: &str, width: u32, height: u32) -> Result<Box<dyn CameraSource>> {
    if device == "auto" {
        let (_, source) = pick_first(&v4l2_candidates(), |d| {
            V4l2Source::open(d, width, height, false)
        })?;
        Ok(Box::new(source))
    } else {
        Ok(Box::new(V4l2Source::open(device, width, height, false)?))
    }
}

/// `/dev/video*` device paths in numeric order (video0, video1, … video10).
pub fn v4l2_candidates() -> Vec<String> {
    let names = std::fs::read_dir("/dev")
        .map(|rd| {
            rd.flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|n| {
                    n.strip_prefix("video")
                        .is_some_and(|s| s.parse::<u32>().is_ok())
                })
                .map(|n| format!("/dev/{n}"))
                .collect()
        })
        .unwrap_or_default();
    order_candidates(names)
}

/// Sort device paths by their numeric suffix so `video10` follows `video9`
/// (lexicographic order would put it after `video1`).
fn order_candidates(mut names: Vec<String>) -> Vec<String> {
    let num = |s: &str| -> u32 {
        s.rfind(|c: char| !c.is_ascii_digit())
            .map(|i| s[i + 1..].parse().unwrap_or(u32::MAX))
            .unwrap_or(u32::MAX)
    };
    names.sort_by_key(|s| num(s));
    names
}

/// Try `opener` over `candidates`, returning the first success along with the
/// device that worked. Pure with respect to hardware — tested with a mock.
fn pick_first<T>(candidates: &[String], opener: impl Fn(&str) -> Result<T>) -> Result<(String, T)> {
    let mut errors: Vec<String> = Vec::new();
    for dev in candidates {
        match opener(dev) {
            Ok(v) => return Ok((dev.clone(), v)),
            Err(e) => errors.push(format!("{dev}: {e}")),
        }
    }
    if candidates.is_empty() {
        bail!("no /dev/video* devices found — is a camera connected?");
    }
    bail!(
        "no working MJPG camera among {} device(s):\n  {}\n(set [camera].device in ~/.config/mvm/config.toml to pick one explicitly)",
        candidates.len(),
        errors.join("\n  ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidates_sort_numerically_not_lexicographically() {
        let names = vec![
            "/dev/video10".to_string(),
            "/dev/video2".to_string(),
            "/dev/video0".to_string(),
        ];
        assert_eq!(
            order_candidates(names),
            vec!["/dev/video0", "/dev/video2", "/dev/video10"]
        );
    }

    #[test]
    fn pick_first_takes_first_live_device() {
        let candidates: Vec<String> = ["/dev/video0", "/dev/video2", "/dev/video4"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        // video0 is dead (e.g. a metadata node), video2 opens.
        let (dev, val) = pick_first(&candidates, |d| {
            if d == "/dev/video2" {
                Ok(42)
            } else {
                bail!("not MJPG")
            }
        })
        .unwrap();
        assert_eq!(dev, "/dev/video2");
        assert_eq!(val, 42);
    }

    #[test]
    fn pick_first_reports_every_failure() {
        let candidates: Vec<String> = vec!["/dev/video0".into(), "/dev/video1".into()];
        let err = pick_first(&candidates, |_| -> Result<()> { bail!("busy") }).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("/dev/video0") && msg.contains("/dev/video1"),
            "{msg}"
        );
        assert!(
            msg.contains("[camera].device"),
            "points at the config override: {msg}"
        );
    }

    #[test]
    fn pick_first_empty_list_is_a_clear_error() {
        let err = pick_first(&[], |_| -> Result<()> { unreachable!() }).unwrap_err();
        assert!(err.to_string().contains("no /dev/video*"), "{err}");
    }
}
