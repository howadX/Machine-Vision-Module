//! Image-free sensor contracts shared by capture, mesh, depth and Dataset layers.
//!
//! Raw device timestamps are intentionally scoped to one stream. Cross-stream
//! pairing uses `host_monotonic_ns`; wall clock exists only for durable provenance.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorSourceId {
    pub backend: String,
    pub logical_name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub serial: Option<String>,
}

impl SensorSourceId {
    pub fn v4l2(device: impl Into<String>) -> Self {
        Self {
            backend: "v4l2".into(),
            logical_name: device.into(),
            serial: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensorClockDomain {
    HostMonotonic,
    KinectV1Rgb,
    KinectV1Depth,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorStamp {
    pub source_sequence: u64,
    pub host_monotonic_ns: u64,
    pub wall_clock_unix_ms: i64,
    pub clock_domain: SensorClockDomain,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub device_timestamp_raw: Option<u32>,
}

impl SensorStamp {
    pub fn validate(&self) -> Result<(), SensorContractError> {
        if self.host_monotonic_ns == 0 {
            return Err(SensorContractError::MissingHostTimestamp);
        }
        if self.wall_clock_unix_ms < 0 {
            return Err(SensorContractError::InvalidWallClock);
        }
        if self.device_timestamp_raw.is_some()
            && matches!(
                self.clock_domain,
                SensorClockDomain::HostMonotonic | SensorClockDomain::Unavailable
            )
        {
            return Err(SensorContractError::DeviceTimestampDomainMismatch);
        }
        Ok(())
    }

    pub fn host_skew_ns(&self, other: &Self) -> u64 {
        self.host_monotonic_ns.abs_diff(other.host_monotonic_ns)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthGeometry {
    RegisteredToRgb,
    NativeDepthCamera,
    UnknownRegistration,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DepthStatus {
    Paired,
    Unavailable,
    Stale,
    Unregistered,
    CalibrationMismatch,
    Corrupt,
}

/// Runtime-only association from a registered RGB-grid depth sample to the
/// native depth-camera pixel that won registration's nearest-depth z-buffer.
///
/// This intentionally has no serde implementation: durable RGB-D recordings
/// keep the lossless registered millimetre map, while exact ray ownership is
/// rebuilt by the pinned capture backend.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredDepthOwners {
    pub registered_width: u32,
    pub registered_height: u32,
    pub native_width: u32,
    pub native_height: u32,
    /// Factory-calibrated native depth-camera intrinsics. They travel with the
    /// runtime-only owner association so registered samples can be unprojected
    /// through the exact native ray that won registration.
    pub native_fx_px: f32,
    pub native_fy_px: f32,
    pub native_cx_px: f32,
    pub native_cy_px: f32,
    /// Row-major native pixel indices; [`Self::NO_OWNER`] means unknown.
    pub owner_index: Vec<u32>,
    pub calibration_revision: String,
}

impl RegisteredDepthOwners {
    pub const NO_OWNER: u32 = u32::MAX;

    pub fn validate(&self) -> Result<(), SensorContractError> {
        if self.registered_width == 0
            || self.registered_height == 0
            || self.native_width == 0
            || self.native_height == 0
            || !self.native_fx_px.is_finite()
            || !self.native_fy_px.is_finite()
            || self.native_fx_px <= 0.0
            || self.native_fy_px <= 0.0
            || !self.native_cx_px.is_finite()
            || !self.native_cy_px.is_finite()
        {
            return Err(SensorContractError::InvalidDimensions);
        }
        let registered_len = checked_pixels(self.registered_width, self.registered_height)?;
        if self.owner_index.len() != registered_len {
            return Err(SensorContractError::BufferLengthMismatch);
        }
        let native_len = checked_pixels(self.native_width, self.native_height)?;
        if self.owner_index.iter().any(|&owner| {
            owner != Self::NO_OWNER && usize::try_from(owner).map_or(true, |v| v >= native_len)
        }) {
            return Err(SensorContractError::InvalidOwnerIndex);
        }
        validate_revision(&self.calibration_revision)
    }

    pub fn native_pixel(&self, registered_x: u32, registered_y: u32) -> Option<[u32; 2]> {
        if registered_x >= self.registered_width || registered_y >= self.registered_height {
            return None;
        }
        let index = registered_y as usize * self.registered_width as usize + registered_x as usize;
        let owner = self.owner_index[index];
        (owner != Self::NO_OWNER).then_some([owner % self.native_width, owner / self.native_width])
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricDepthFrame {
    pub width: u32,
    pub height: u32,
    pub depth_mm: Vec<u16>,
    pub invalid_value: u16,
    pub stamp: SensorStamp,
    pub geometry: DepthGeometry,
    pub calibration_revision: String,
    pub valid_ratio: f32,
    pub rgb_depth_skew_ns: u64,
}

impl MetricDepthFrame {
    pub fn validate(&self) -> Result<(), SensorContractError> {
        self.stamp.validate()?;
        if self.width == 0 || self.height == 0 {
            return Err(SensorContractError::InvalidDimensions);
        }
        let expected = checked_pixels(self.width, self.height)?;
        if self.depth_mm.len() != expected {
            return Err(SensorContractError::BufferLengthMismatch);
        }
        if !self.valid_ratio.is_finite() || !(0.0..=1.0).contains(&self.valid_ratio) {
            return Err(SensorContractError::InvalidRatio);
        }
        validate_revision(&self.calibration_revision)?;
        Ok(())
    }
}

fn checked_pixels(width: u32, height: u32) -> Result<usize, SensorContractError> {
    usize::try_from(width)
        .ok()
        .and_then(|w| usize::try_from(height).ok().and_then(|h| w.checked_mul(h)))
        .ok_or(SensorContractError::InvalidDimensions)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CalibrationRef {
    pub schema: String,
    pub camera_serial: String,
    /// Lower-case SHA-256 of the exact profile bytes loaded by the backend.
    pub revision_sha256: String,
}

impl CalibrationRef {
    pub fn validate(&self) -> Result<(), SensorContractError> {
        if self.schema.trim().is_empty() || self.camera_serial.trim().is_empty() {
            return Err(SensorContractError::MissingIdentity);
        }
        validate_revision(&self.revision_sha256)
    }
}

fn validate_revision(value: &str) -> Result<(), SensorContractError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(SensorContractError::InvalidCalibrationRevision);
    }
    Ok(())
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorHealth {
    pub paired: u64,
    pub rgb_only: u64,
    pub depth_dropped: u64,
    pub rgb_gaps: u64,
    pub depth_gaps: u64,
    pub duplicate_timestamps: u64,
    pub out_of_order_timestamps: u64,
    pub resync_errors: u64,
}

/// Image-free replay envelope used to prove admission rules before a hardware
/// backend is wired into the runtime. Dense RGB bytes are deliberately absent.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensorReplayEnvelope {
    pub case_id: String,
    pub source: SensorSourceId,
    pub rgb_width: u32,
    pub rgb_height: u32,
    pub rgb_stamp: SensorStamp,
    pub depth_status: DepthStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metric_depth: Option<MetricDepthFrame>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration: Option<CalibrationRef>,
    #[serde(default)]
    pub health: SensorHealth,
}

impl SensorReplayEnvelope {
    pub fn validate(&self, max_pair_skew_ns: u64) -> Result<(), SensorContractError> {
        if self.case_id.trim().is_empty() || self.rgb_width == 0 || self.rgb_height == 0 {
            return Err(SensorContractError::InvalidDimensions);
        }
        self.rgb_stamp.validate()?;
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
            if self.source.serial.as_deref() != Some(calibration.camera_serial.as_str()) {
                return Err(SensorContractError::SourceCalibrationMismatch);
            }
        }
        match (&self.depth_status, &self.metric_depth) {
            (DepthStatus::Paired, Some(depth)) => {
                depth.validate()?;
                if depth.geometry != DepthGeometry::RegisteredToRgb {
                    return Err(SensorContractError::UnregisteredRgbAssociation);
                }
                let calibration = self
                    .calibration
                    .as_ref()
                    .ok_or(SensorContractError::MissingCalibration)?;
                if depth.calibration_revision != calibration.revision_sha256 {
                    return Err(SensorContractError::SourceCalibrationMismatch);
                }
                let measured = self.rgb_stamp.host_skew_ns(&depth.stamp);
                if measured != depth.rgb_depth_skew_ns || measured > max_pair_skew_ns {
                    return Err(SensorContractError::PairSkewExceeded);
                }
                if depth.width != self.rgb_width || depth.height != self.rgb_height {
                    return Err(SensorContractError::GeometryMismatch);
                }
            }
            (DepthStatus::Paired, None) => return Err(SensorContractError::MissingDepth),
            (_, Some(_)) => return Err(SensorContractError::DepthStatusMismatch),
            (_, None) => {}
        }
        Ok(())
    }

    pub fn admits_metric_rgb_association(&self, max_pair_skew_ns: u64) -> bool {
        self.validate(max_pair_skew_ns).is_ok() && matches!(self.depth_status, DepthStatus::Paired)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorContractError {
    MissingHostTimestamp,
    InvalidWallClock,
    DeviceTimestampDomainMismatch,
    InvalidDimensions,
    BufferLengthMismatch,
    InvalidRatio,
    MissingIdentity,
    InvalidCalibrationRevision,
    SourceCalibrationMismatch,
    MissingCalibration,
    MissingDepth,
    DepthStatusMismatch,
    PairSkewExceeded,
    GeometryMismatch,
    UnregisteredRgbAssociation,
    InvalidOwnerIndex,
}

impl std::fmt::Display for SensorContractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for SensorContractError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(domain: SensorClockDomain) -> SensorStamp {
        SensorStamp {
            source_sequence: 7,
            host_monotonic_ns: 1_000_000,
            wall_clock_unix_ms: 1_700_000_000_000,
            clock_domain: domain,
            device_timestamp_raw: Some(42),
        }
    }

    #[test]
    fn sensor_stamp_round_trip_preserves_clock_domain() {
        let value = stamp(SensorClockDomain::KinectV1Depth);
        let json = serde_json::to_string(&value).unwrap();
        assert_eq!(serde_json::from_str::<SensorStamp>(&json).unwrap(), value);
    }

    #[test]
    fn raw_device_timestamp_requires_device_clock_domain() {
        assert_eq!(
            stamp(SensorClockDomain::HostMonotonic).validate(),
            Err(SensorContractError::DeviceTimestampDomainMismatch)
        );
    }

    #[test]
    fn metric_depth_rejects_wrong_buffer_and_non_finite_ratio() {
        let mut depth = MetricDepthFrame {
            width: 2,
            height: 2,
            depth_mm: vec![0; 3],
            invalid_value: 0,
            stamp: stamp(SensorClockDomain::KinectV1Depth),
            geometry: DepthGeometry::RegisteredToRgb,
            calibration_revision: "a".repeat(64),
            valid_ratio: 0.5,
            rgb_depth_skew_ns: 4,
        };
        assert_eq!(
            depth.validate(),
            Err(SensorContractError::BufferLengthMismatch)
        );
        depth.depth_mm.push(1);
        depth.valid_ratio = f32::NAN;
        assert_eq!(depth.validate(), Err(SensorContractError::InvalidRatio));
    }

    #[test]
    fn calibration_revision_is_exact_lowercase_sha256() {
        let mut reference = CalibrationRef {
            schema: "kinect-v1-sensor-calibration/1.0".into(),
            camera_serial: "serial".into(),
            revision_sha256: "f".repeat(64),
        };
        assert!(reference.validate().is_ok());
        reference.revision_sha256 = "F".repeat(64);
        assert_eq!(
            reference.validate(),
            Err(SensorContractError::InvalidCalibrationRevision)
        );
    }

    #[test]
    fn unknown_json_fields_fail_closed() {
        let json = r#"{"source_sequence":1,"host_monotonic_ns":1,"wall_clock_unix_ms":1,"clock_domain":"unavailable","device_timestamp_raw":null,"surprise":true}"#;
        assert!(serde_json::from_str::<SensorStamp>(json).is_err());
    }

    #[test]
    fn registered_owner_map_validates_bounds_and_resolves_native_pixel() {
        let mut owners = RegisteredDepthOwners {
            registered_width: 2,
            registered_height: 2,
            native_width: 4,
            native_height: 3,
            native_fx_px: 575.0,
            native_fy_px: 575.0,
            native_cx_px: 2.0,
            native_cy_px: 1.5,
            owner_index: vec![0, 6, RegisteredDepthOwners::NO_OWNER, 11],
            calibration_revision: "a".repeat(64),
        };
        owners.validate().unwrap();
        assert_eq!(owners.native_pixel(1, 0), Some([2, 1]));
        assert_eq!(owners.native_pixel(0, 1), None);
        owners.owner_index[0] = 12;
        assert_eq!(
            owners.validate(),
            Err(SensorContractError::InvalidOwnerIndex)
        );
    }

    #[test]
    fn registered_owner_map_is_runtime_only_not_part_of_depth_json() {
        let depth = MetricDepthFrame {
            width: 1,
            height: 1,
            depth_mm: vec![1_000],
            invalid_value: 0,
            stamp: stamp(SensorClockDomain::KinectV1Depth),
            geometry: DepthGeometry::RegisteredToRgb,
            calibration_revision: "a".repeat(64),
            valid_ratio: 1.0,
            rgb_depth_skew_ns: 0,
        };
        assert!(!serde_json::to_string(&depth).unwrap().contains("owner"));
    }
}
