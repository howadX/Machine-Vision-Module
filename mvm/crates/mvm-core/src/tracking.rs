//! Exact frame provenance and bounded, image-free tracker diagnostics.
//!
//! The binding is shared by live AgentMesh ticks and production-parity replay.
//! It deliberately carries no pixels, masks, embeddings or durable semantic state.

use crate::sensor::{
    CalibrationRef, DepthGeometry, DepthStatus, MetricDepthFrame, SensorContractError,
    SensorSourceId, SensorStamp,
};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackerFrameBinding {
    pub source_frame_id: u64,
    pub source: SensorSourceId,
    pub rgb_width: u32,
    pub rgb_height: u32,
    pub rgb_stamp: SensorStamp,
    pub depth_status: DepthStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub depth_stamp: Option<SensorStamp>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rgb_depth_skew_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub calibration_revision: Option<String>,
}

impl TrackerFrameBinding {
    pub fn from_sensor_parts(
        source_frame_id: u64,
        source: &SensorSourceId,
        rgb_width: u32,
        rgb_height: u32,
        rgb_stamp: &SensorStamp,
        depth_status: &DepthStatus,
        metric_depth: Option<&MetricDepthFrame>,
        calibration: Option<&CalibrationRef>,
        max_pair_skew_ns: u64,
    ) -> Result<Self, SensorContractError> {
        if rgb_width == 0 || rgb_height == 0 {
            return Err(SensorContractError::InvalidDimensions);
        }
        rgb_stamp.validate()?;
        if let Some(calibration) = calibration {
            calibration.validate()?;
            if source.serial.as_deref() != Some(calibration.camera_serial.as_str()) {
                return Err(SensorContractError::SourceCalibrationMismatch);
            }
        }
        let (depth_stamp, skew, revision) = match (depth_status, metric_depth) {
            (DepthStatus::Paired, Some(depth)) => {
                depth.validate()?;
                if depth.geometry != DepthGeometry::RegisteredToRgb {
                    return Err(SensorContractError::UnregisteredRgbAssociation);
                }
                if depth.width != rgb_width || depth.height != rgb_height {
                    return Err(SensorContractError::GeometryMismatch);
                }
                let calibration = calibration.ok_or(SensorContractError::MissingCalibration)?;
                if depth.calibration_revision != calibration.revision_sha256 {
                    return Err(SensorContractError::SourceCalibrationMismatch);
                }
                let measured = rgb_stamp.host_skew_ns(&depth.stamp);
                if measured != depth.rgb_depth_skew_ns || measured > max_pair_skew_ns {
                    return Err(SensorContractError::PairSkewExceeded);
                }
                (
                    Some(depth.stamp.clone()),
                    Some(measured),
                    Some(calibration.revision_sha256.clone()),
                )
            }
            (DepthStatus::Paired, None) => return Err(SensorContractError::MissingDepth),
            (_, Some(_)) => return Err(SensorContractError::DepthStatusMismatch),
            (_, None) => (None, None, calibration.map(|c| c.revision_sha256.clone())),
        };
        Ok(Self {
            source_frame_id,
            source: source.clone(),
            rgb_width,
            rgb_height,
            rgb_stamp: rgb_stamp.clone(),
            depth_status: depth_status.clone(),
            depth_stamp,
            rgb_depth_skew_ns: skew,
            calibration_revision: revision,
        })
    }

    /// Compatibility binding for direct tracker callers and legacy unit tests.
    pub fn rgb_only(frame_id: u64) -> Self {
        Self {
            source_frame_id: frame_id,
            source: SensorSourceId::v4l2("tracker-compatibility"),
            rgb_width: 1,
            rgb_height: 1,
            rgb_stamp: SensorStamp {
                source_sequence: frame_id,
                host_monotonic_ns: frame_id.saturating_add(1),
                wall_clock_unix_ms: 0,
                clock_domain: crate::sensor::SensorClockDomain::HostMonotonic,
                device_timestamp_raw: None,
            },
            depth_status: DepthStatus::Unavailable,
            depth_stamp: None,
            rgb_depth_skew_ns: None,
            calibration_revision: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackerDomain {
    Person,
    Object,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrackerDecisionKind {
    Admission,
    Prediction,
    Candidate,
    Assignment,
    Birth,
    LowRecovery,
    Relink,
    Lifecycle,
    Miss,
    Expire,
    Publication,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackerDecision {
    pub binding: TrackerFrameBinding,
    pub domain: TrackerDomain,
    pub kind: TrackerDecisionKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detection_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub track_id: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub iou: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub momentum: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mask_iou: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub score: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cue: Option<String>,
    pub outcome: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle_before: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lifecycle_after: Option<String>,
}

impl TrackerDecision {
    pub fn new(
        binding: &TrackerFrameBinding,
        domain: TrackerDomain,
        kind: TrackerDecisionKind,
        outcome: impl Into<String>,
    ) -> Self {
        Self {
            binding: binding.clone(),
            domain,
            kind,
            detection_index: None,
            track_id: None,
            iou: None,
            momentum: None,
            mask_iou: None,
            score: None,
            cue: None,
            outcome: outcome.into(),
            reason: None,
            lifecycle_before: None,
            lifecycle_after: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrackerDiagnosticsSnapshot {
    pub capacity: usize,
    pub dropped_events: u64,
    pub counters: BTreeMap<TrackerDecisionKind, u64>,
    pub events: Vec<TrackerDecision>,
}

#[derive(Debug, Clone)]
pub struct TrackerDiagnostics {
    capacity: usize,
    dropped_events: u64,
    counters: BTreeMap<TrackerDecisionKind, u64>,
    events: VecDeque<TrackerDecision>,
}

impl Default for TrackerDiagnostics {
    fn default() -> Self {
        Self::with_capacity(0)
    }
}

impl TrackerDiagnostics {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            dropped_events: 0,
            counters: BTreeMap::new(),
            events: VecDeque::with_capacity(capacity),
        }
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        self.capacity = capacity;
        while self.events.len() > capacity {
            self.events.pop_front();
            self.dropped_events += 1;
        }
    }

    pub fn is_tracing(&self) -> bool {
        self.capacity > 0
    }

    pub fn record(&mut self, event: TrackerDecision) {
        *self.counters.entry(event.kind).or_insert(0) += 1;
        if self.capacity == 0 {
            return;
        }
        if self.events.len() == self.capacity {
            self.events.pop_front();
            self.dropped_events += 1;
        }
        self.events.push_back(event);
    }

    pub fn snapshot(&self) -> TrackerDiagnosticsSnapshot {
        let mut events: Vec<TrackerDecision> = self.events.iter().cloned().collect();
        events.sort_by(|a, b| {
            (
                a.binding.source_frame_id,
                a.binding.rgb_stamp.host_monotonic_ns,
                a.kind,
                a.detection_index,
                a.track_id,
                a.cue.as_deref(),
                a.outcome.as_str(),
                a.reason.as_deref(),
            )
                .cmp(&(
                    b.binding.source_frame_id,
                    b.binding.rgb_stamp.host_monotonic_ns,
                    b.kind,
                    b.detection_index,
                    b.track_id,
                    b.cue.as_deref(),
                    b.outcome.as_str(),
                    b.reason.as_deref(),
                ))
        });
        TrackerDiagnosticsSnapshot {
            capacity: self.capacity,
            dropped_events: self.dropped_events,
            counters: self.counters.clone(),
            events,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_trace_counts_without_creating_a_durable_store() {
        let binding = TrackerFrameBinding::rgb_only(7);
        let mut diagnostics = TrackerDiagnostics::with_capacity(1);
        diagnostics.record(TrackerDecision::new(
            &binding,
            TrackerDomain::Person,
            TrackerDecisionKind::Admission,
            "accepted",
        ));
        diagnostics.record(TrackerDecision::new(
            &binding,
            TrackerDomain::Person,
            TrackerDecisionKind::Publication,
            "current_baseline",
        ));
        let snapshot = diagnostics.snapshot();
        assert_eq!(snapshot.events.len(), 1);
        assert_eq!(snapshot.dropped_events, 1);
        assert_eq!(snapshot.counters.values().sum::<u64>(), 2);
    }
}
