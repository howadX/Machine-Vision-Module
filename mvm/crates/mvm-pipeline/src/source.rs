//! Generic sensor source seam. RGB-only V4L2 sources publish explicit
//! unavailable metric-depth evidence through the same packet contract.
//!
//! The CLI opens cameras through this trait, never through a concrete backend.
//! Today the only backend is [`V4l2Source`] (the phase-7 `NativeCamera`);
//! macOS (`AvFoundationSource`) and Windows (`MediaFoundationSource`) later
//! land here without touching any call site.

use anyhow::{Result, bail};
use image::DynamicImage;
use mvm_core::{
    CalibrationRef, DepthStatus, MetricDepthFrame, RegisteredDepthOwners, SensorClockDomain,
    SensorHealth, SensorSourceId, SensorStamp,
};
use std::path::Path;
use std::sync::Arc;

use crate::native::NativeCamera;

/// One immutable packet returned by a sensor backend.
pub struct CapturedSensorFrame {
    pub rgb: Arc<DynamicImage>,
    pub rgb_stamp: SensorStamp,
    pub source: SensorSourceId,
    pub depth_status: DepthStatus,
    pub metric_depth: Option<Arc<MetricDepthFrame>>,
    /// Runtime-only exact registered-pixel to native-depth-ray association.
    pub registered_depth_owners: Option<Arc<RegisteredDepthOwners>>,
    pub calibration: Option<CalibrationRef>,
    pub health: SensorHealth,
}

impl CapturedSensorFrame {
    pub fn rgb_only(image: DynamicImage, device: impl Into<String>, sequence: u64) -> Self {
        let device = device.into();
        Self {
            rgb: Arc::new(image),
            rgb_stamp: SensorStamp {
                source_sequence: sequence,
                host_monotonic_ns: monotonic_ns(),
                wall_clock_unix_ms: unix_ms_now(),
                clock_domain: SensorClockDomain::HostMonotonic,
                device_timestamp_raw: None,
            },
            source: SensorSourceId::v4l2(device),
            depth_status: DepthStatus::Unavailable,
            metric_depth: None,
            registered_depth_owners: None,
            calibration: None,
            health: SensorHealth {
                rgb_only: sequence.saturating_add(1),
                ..SensorHealth::default()
            },
        }
    }
}

pub trait SensorSource: Send {
    /// Capture the next immutable sensor packet (blocking at the source tempo).
    fn read(&mut self) -> Result<CapturedSensorFrame>;
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
    sequence: u64,
}

impl V4l2Source {
    pub fn open(device: &str, width: u32, height: u32, rotate_180: bool) -> Result<Self> {
        let cam = NativeCamera::open(device, width, height, rotate_180)?;
        Ok(Self {
            cam,
            device: device.to_string(),
            sequence: 0,
        })
    }
}

impl SensorSource for V4l2Source {
    fn read(&mut self) -> Result<CapturedSensorFrame> {
        let image = self.cam.read()?;
        let sequence = self.sequence;
        self.sequence = self.sequence.saturating_add(1);
        Ok(CapturedSensorFrame {
            rgb: Arc::new(image),
            rgb_stamp: SensorStamp {
                source_sequence: sequence,
                host_monotonic_ns: monotonic_ns(),
                wall_clock_unix_ms: unix_ms_now(),
                clock_domain: SensorClockDomain::HostMonotonic,
                device_timestamp_raw: None,
            },
            source: SensorSourceId::v4l2(self.device.clone()),
            depth_status: DepthStatus::Unavailable,
            metric_depth: None,
            registered_depth_owners: None,
            calibration: None,
            health: SensorHealth {
                rgb_only: sequence.saturating_add(1),
                ..SensorHealth::default()
            },
        })
    }
    fn dimensions(&self) -> (u32, u32) {
        self.cam.dimensions()
    }
    fn device(&self) -> &str {
        &self.device
    }
}

/// Open `device` as a boxed [`SensorSource`]. `"auto"` scans `/dev/video*` in
/// numeric order and takes the first device that negotiates MJPG; anything
/// else is opened literally.
pub fn open_source(device: &str, width: u32, height: u32) -> Result<Box<dyn SensorSource>> {
    if device == "auto" {
        let (_, source) = pick_first(&v4l2_candidates(), |d| {
            V4l2Source::open(d, width, height, false)
        })?;
        Ok(Box::new(source))
    } else {
        Ok(Box::new(V4l2Source::open(device, width, height, false)?))
    }
}

pub(crate) fn monotonic_ns() -> u64 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let epoch = EPOCH.get_or_init(std::time::Instant::now);
    epoch
        .elapsed()
        .as_nanos()
        .try_into()
        .unwrap_or(u64::MAX)
        .max(1)
}

pub(crate) fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
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
                .filter(|device| !is_kinect_v1_video_node(device))
                .collect()
        })
        .unwrap_or_default();
    order_candidates(names)
}

fn is_kinect_v1_video_node(device: &str) -> bool {
    let Some(name) = Path::new(device).file_name() else {
        return false;
    };
    let uevent = Path::new("/sys/class/video4linux")
        .join(name)
        .join("device/uevent");
    std::fs::read_to_string(uevent)
        .map(|text| {
            text.lines().any(|line| {
                line.strip_prefix("PRODUCT=")
                    .is_some_and(|product| product.to_ascii_lowercase().starts_with("45e/2ae/"))
            })
        })
        .unwrap_or(false)
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
    fn rgb_only_packet_is_explicit_and_self_describing() {
        let packet = CapturedSensorFrame::rgb_only(DynamicImage::new_rgb8(4, 3), "/dev/video7", 12);
        assert_eq!(packet.rgb_stamp.source_sequence, 12);
        assert_eq!(packet.source.backend, "v4l2");
        assert_eq!(packet.source.logical_name, "/dev/video7");
        assert_eq!(packet.depth_status, DepthStatus::Unavailable);
        assert!(packet.metric_depth.is_none());
        assert!(packet.calibration.is_none());
        assert_eq!(packet.rgb.width(), 4);
        packet.rgb_stamp.validate().unwrap();
    }

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
