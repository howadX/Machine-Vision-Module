//! Runtime-only stamped expert observations (phase 18.1).
//!
//! These packets preserve the camera frame that produced each asynchronous
//! expert result. Metric and relative depth remain distinct evidence handles rather than
//! serializable canonical scene facts.

use mvm_core::{BBox, ObjectState, PersonState, mask::InstanceMask};
use mvm_fusion::canonical::AnimalObservation;
use mvm_mesh::{Frame, LatestValue};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ObservationStamp {
    pub source_frame_id: u64,
    pub captured_at: Instant,
    pub produced_at: Instant,
    pub captured_at_unix_ms: i64,
    pub produced_at_unix_ms: i64,
}

impl ObservationStamp {
    pub(crate) fn new(
        source_frame_id: u64,
        captured_at: Instant,
        captured_at_unix_ms: i64,
    ) -> Self {
        Self {
            source_frame_id,
            captured_at,
            produced_at: Instant::now(),
            captured_at_unix_ms,
            produced_at_unix_ms: unix_ms_now(),
        }
    }

    pub(crate) fn from_frame(frame: &Frame) -> Self {
        Self::new(frame.id, frame.captured, frame.captured_at_unix_ms)
    }

    pub(crate) fn age_at(self, camera_frame_id: u64) -> Option<u64> {
        camera_frame_id.checked_sub(self.source_frame_id)
    }
}

fn unix_ms_now() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
        .unwrap_or(0)
}

#[derive(Debug, Clone)]
pub(crate) struct HumanObservationFrame {
    pub stamp: ObservationStamp,
    /// Confirmed/IdLocked observations from this Human tick.
    pub people: Arc<Vec<PersonState>>,
    /// Established tracks currently coasting; never current-presence evidence.
    pub occluded_people: Arc<Vec<PersonState>>,
    pub retained_track_ids: Arc<Vec<u32>>,
    pub retirements: Arc<Vec<u32>>,
    pub metric_geometry: Arc<Vec<mvm_core::TrackedAssociationGeometry>>,
}

#[derive(Debug, Clone)]
pub(crate) struct SceneObservationFrame {
    pub stamp: ObservationStamp,
    /// Confirmed objects observed on this Scene tick.
    pub objects: Arc<Vec<ObjectState>>,
    /// Established object projections without a current detection.
    pub occluded_objects: Arc<Vec<ObjectState>>,
    pub retained_track_ids: Arc<Vec<u32>>,
    pub retirements: Arc<Vec<u32>>,
    pub animals: Arc<Vec<AnimalObservation>>,
    pub person_masks: Arc<Vec<(BBox, InstanceMask)>>,
    pub metric_geometry: Arc<Vec<mvm_core::TrackedAssociationGeometry>>,
}

#[derive(Debug, Clone)]
pub(crate) struct DepthObservationFrame {
    pub stamp: ObservationStamp,
    pub evidence: DepthEvidence,
}

#[derive(Debug, Clone)]
pub(crate) enum DepthEvidence {
    Relative(Arc<mvm_depth::RelativeDepthMap>),
    Metric(MetricDepthEvidence),
}

#[derive(Debug, Clone)]
pub(crate) struct MetricDepthEvidence {
    pub depth: mvm_depth::RegisteredMetricDepth,
    pub source: mvm_core::SensorSourceId,
    pub rgb_stamp: mvm_core::SensorStamp,
}

impl DepthObservationFrame {
    pub(crate) fn relative_map(&self) -> Option<&Arc<mvm_depth::RelativeDepthMap>> {
        match &self.evidence {
            DepthEvidence::Relative(map) => Some(map),
            DepthEvidence::Metric(_) => None,
        }
    }

    pub(crate) fn metric(&self) -> Option<&mvm_depth::RegisteredMetricDepth> {
        match &self.evidence {
            DepthEvidence::Metric(metric) => Some(&metric.depth),
            DepthEvidence::Relative(_) => None,
        }
    }
}

pub(crate) fn registered_metric_from_frame(
    frame: &Frame,
    max_pair_skew_ns: u64,
) -> Option<DepthObservationFrame> {
    let metric = frame.sensor.metric_depth.as_ref()?;
    let calibration = frame.sensor.calibration.as_ref()?;
    let admitted = mvm_depth::RegisteredMetricDepth::admit(
        frame.image.width(),
        frame.image.height(),
        &frame.sensor.rgb_stamp,
        &frame.sensor.depth_status,
        Arc::clone(metric),
        calibration,
        max_pair_skew_ns,
    )
    .ok()?;
    Some(DepthObservationFrame {
        stamp: ObservationStamp::from_frame(frame),
        evidence: DepthEvidence::Metric(MetricDepthEvidence {
            depth: admitted,
            source: frame.sensor.source.clone(),
            rgb_stamp: frame.sensor.rgb_stamp.clone(),
        }),
    })
}

pub(crate) fn tracker_binding_from_frame(
    frame: &Frame,
    max_pair_skew_ns: u64,
) -> Result<mvm_core::TrackerFrameBinding, mvm_core::sensor::SensorContractError> {
    mvm_core::TrackerFrameBinding::from_sensor_parts(
        frame.id,
        &frame.sensor.source,
        frame.image.width(),
        frame.image.height(),
        &frame.sensor.rgb_stamp,
        &frame.sensor.depth_status,
        frame.sensor.metric_depth.as_deref(),
        frame.sensor.calibration.as_ref(),
        max_pair_skew_ns,
    )
}

pub(crate) struct AssociationMeasurementContext {
    binding: mvm_core::TrackerFrameBinding,
    depth: Option<mvm_depth::RegisteredMetricDepth>,
    owners: Option<Arc<mvm_core::RegisteredDepthOwners>>,
    forced_reason: Option<mvm_core::GeometryReason>,
}

impl AssociationMeasurementContext {
    pub(crate) fn from_frame(
        frame: &Frame,
        max_pair_skew_ns: u64,
    ) -> Result<Self, mvm_core::sensor::SensorContractError> {
        if frame.image.width() == 0 || frame.image.height() == 0 {
            return Err(mvm_core::sensor::SensorContractError::InvalidDimensions);
        }
        // RGB is the detector clock and remains fail-closed. Once it validates,
        // failures from the strict combined binding belong to optional metric
        // evidence and may degrade to a typed unavailable geometry.
        frame.sensor.rgb_stamp.validate()?;
        let (binding, forced_reason) = match tracker_binding_from_frame(frame, max_pair_skew_ns) {
            Ok(binding) => (binding, None),
            Err(error) => {
                let metric = frame.sensor.metric_depth.as_deref();
                let reason = match error {
                    mvm_core::sensor::SensorContractError::PairSkewExceeded => {
                        mvm_core::GeometryReason::DepthStale
                    }
                    mvm_core::sensor::SensorContractError::UnregisteredRgbAssociation
                    | mvm_core::sensor::SensorContractError::GeometryMismatch => {
                        mvm_core::GeometryReason::DepthUnregistered
                    }
                    mvm_core::sensor::SensorContractError::SourceCalibrationMismatch
                    | mvm_core::sensor::SensorContractError::MissingCalibration
                    | mvm_core::sensor::SensorContractError::InvalidCalibrationRevision => {
                        mvm_core::GeometryReason::CalibrationMismatch
                    }
                    _ => mvm_core::GeometryReason::DepthUnavailable,
                };
                (
                    mvm_core::TrackerFrameBinding {
                        source_frame_id: frame.id,
                        source: frame.sensor.source.clone(),
                        rgb_width: frame.image.width(),
                        rgb_height: frame.image.height(),
                        rgb_stamp: frame.sensor.rgb_stamp.clone(),
                        depth_status: frame.sensor.depth_status.clone(),
                        depth_stamp: metric.map(|depth| depth.stamp.clone()),
                        rgb_depth_skew_ns: metric.map(|depth| depth.rgb_depth_skew_ns),
                        calibration_revision: frame
                            .sensor
                            .calibration
                            .as_ref()
                            .map(|calibration| calibration.revision_sha256.clone()),
                    },
                    Some(reason),
                )
            }
        };
        let depth = forced_reason
            .is_none()
            .then(|| {
                registered_metric_from_frame(frame, max_pair_skew_ns)
                    .and_then(|observation| observation.metric().cloned())
            })
            .flatten();
        Ok(Self {
            binding,
            depth,
            owners: frame.sensor.registered_depth_owners.clone(),
            forced_reason,
        })
    }

    pub(crate) fn binding(&self) -> &mvm_core::TrackerFrameBinding {
        &self.binding
    }

    pub(crate) fn observation(
        &self,
        detection_key: u32,
        bbox: BBox,
        detector_score: f32,
        current_mask: Option<&InstanceMask>,
    ) -> mvm_core::AssociationObservation {
        mvm_core::AssociationObservation {
            detection_key,
            source: self.binding.clone(),
            bbox,
            detector_score,
            geometry: self.forced_reason.map_or_else(
                || {
                    mvm_depth::measure_association_geometry(
                        &self.binding,
                        self.depth.as_ref(),
                        self.owners.as_deref(),
                        bbox,
                        current_mask,
                    )
                },
                |reason| {
                    let mut geometry =
                        mvm_core::AssociationGeometry::unavailable(&self.binding, reason);
                    if matches!(
                        reason,
                        mvm_core::GeometryReason::DepthUnregistered
                            | mvm_core::GeometryReason::CalibrationMismatch
                    ) {
                        geometry.status = mvm_core::AssociationGeometryStatus::Withheld;
                    }
                    geometry
                },
            ),
        }
    }
}

#[derive(Clone)]
pub(crate) struct FrameEvidenceHandles {
    pub rgb: Arc<image::DynamicImage>,
    pub depth: Option<DepthObservationFrame>,
    pub human: Option<HumanObservationFrame>,
    pub scene: Option<SceneObservationFrame>,
}

#[derive(Clone)]
pub(crate) struct FusedFramePacket {
    pub scene: Arc<mvm_core::canonical::FusedSceneFrame>,
    pub evidence: FrameEvidenceHandles,
}

pub(crate) type HumanObservationBus = LatestValue<HumanObservationFrame>;
pub(crate) type SceneObservationBus = LatestValue<SceneObservationFrame>;
pub(crate) type DepthObservationBus = LatestValue<DepthObservationFrame>;
pub(crate) type FusedFrameBus = LatestValue<FusedFramePacket>;
/// Runtime output of the optional Cortex adapter. It is intentionally separate
/// from canonical facts: Cortex publishes an interpretation, never a mutation.
pub(crate) type CortexFrameBus = LatestValue<mvm_cortex::CortexFrame>;
pub(crate) type AttentionPolicyBus = LatestValue<mvm_cortex::AttentionPolicy>;
pub(crate) type PolicyOutcomeBus = LatestValue<mvm_cortex::PolicyOutcomeFrame>;

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::canonical::{
        EventTailView, FactReason, FrameStamp, FusedSceneFrame, SceneFact, SceneQuality,
    };
    use mvm_mesh::SensorFrameBundle;

    fn paired_frame(declared_skew_ns: u64) -> Frame {
        let revision = "a".repeat(64);
        let rgb_stamp = mvm_core::SensorStamp {
            source_sequence: 70,
            host_monotonic_ns: 1_000_000,
            wall_clock_unix_ms: 1_700_000_000_000,
            clock_domain: mvm_core::SensorClockDomain::KinectV1Rgb,
            device_timestamp_raw: Some(10),
        };
        let depth_stamp = mvm_core::SensorStamp {
            source_sequence: 7,
            host_monotonic_ns: 1_001_000,
            wall_clock_unix_ms: 1_700_000_000_001,
            clock_domain: mvm_core::SensorClockDomain::KinectV1Depth,
            device_timestamp_raw: Some(11),
        };
        let calibration = mvm_core::CalibrationRef {
            schema: "test/1".into(),
            camera_serial: "serial".into(),
            revision_sha256: revision.clone(),
        };
        let metric_depth = Arc::new(mvm_core::MetricDepthFrame {
            width: 64,
            height: 64,
            depth_mm: vec![2_000; 64 * 64],
            invalid_value: 0,
            stamp: depth_stamp,
            geometry: mvm_core::DepthGeometry::RegisteredToRgb,
            calibration_revision: revision.clone(),
            valid_ratio: 1.0,
            rgb_depth_skew_ns: declared_skew_ns,
        });
        let owners = Arc::new(mvm_core::RegisteredDepthOwners {
            registered_width: 64,
            registered_height: 64,
            native_width: 64,
            native_height: 64,
            native_fx_px: 50.0,
            native_fy_px: 50.0,
            native_cx_px: 32.0,
            native_cy_px: 32.0,
            owner_index: (0..64 * 64).collect(),
            calibration_revision: revision,
        });
        Frame::from_sensor(
            7,
            Arc::new(image::DynamicImage::new_rgb8(64, 64)),
            Instant::now(),
            SensorFrameBundle {
                source: mvm_core::SensorSourceId {
                    backend: "kinect_v1".into(),
                    logical_name: "test".into(),
                    serial: Some("serial".into()),
                },
                rgb_stamp,
                depth_status: mvm_core::DepthStatus::Paired,
                metric_depth: Some(metric_depth),
                registered_depth_owners: Some(owners),
                calibration: Some(calibration),
                health: mvm_core::SensorHealth::default(),
            },
        )
    }

    fn fused_scene(frame_id: u64) -> FusedSceneFrame {
        let timestamp = i64::try_from(frame_id).unwrap() * 100;
        FusedSceneFrame {
            stamp: FrameStamp::new("run", frame_id, timestamp, timestamp + 1).unwrap(),
            sources: Vec::new(),
            entities: Vec::new(),
            relations: Vec::new(),
            hand_interactions: Vec::new(),
            zones: Vec::new(),
            activities: Vec::new(),
            events: EventTailView::default(),
            anomaly: SceneFact::unknown(FactReason::MissingSource),
            quality: SceneQuality::default(),
        }
    }

    #[test]
    fn observation_age_rejects_future_results() {
        let now = Instant::now();
        let stamp = ObservationStamp {
            source_frame_id: 12,
            captured_at: now,
            produced_at: now,
            captured_at_unix_ms: 1_000,
            produced_at_unix_ms: 1_001,
        };
        assert_eq!(stamp.age_at(15), Some(3));
        assert_eq!(stamp.age_at(11), None);
    }

    #[test]
    fn typed_bus_keeps_source_stamp_with_depth_handle() {
        let bus = DepthObservationBus::new();
        let frame = Frame::new_stamped(
            7,
            image::DynamicImage::new_rgb8(2, 2),
            Instant::now(),
            1_000,
        );
        let stamp = ObservationStamp::from_frame(&frame);
        bus.publish(DepthObservationFrame {
            stamp,
            evidence: DepthEvidence::Relative(Arc::new(mvm_depth::RelativeDepthMap {
                data: vec![0.5; 4],
                w: 2,
                h: 2,
            })),
        });

        let observed = bus.latest().expect("published depth observation");
        assert_eq!(observed.stamp.source_frame_id, 7);
        assert_eq!(observed.stamp.captured_at_unix_ms, 1_000);
        assert_eq!(observed.relative_map().unwrap().w, 2);
    }

    #[test]
    fn publishing_late_result_cannot_mutate_previous_fused_packet() {
        let bus = FusedFrameBus::new();
        let image = Arc::new(image::DynamicImage::new_rgb8(2, 2));
        let evidence = FrameEvidenceHandles {
            rgb: Arc::clone(&image),
            depth: None,
            human: None,
            scene: None,
        };
        bus.publish(FusedFramePacket {
            scene: Arc::new(fused_scene(1)),
            evidence: evidence.clone(),
        });
        let first = bus.latest().expect("first packet");
        let serialized = serde_json::to_string(first.scene.as_ref()).unwrap();

        bus.publish(FusedFramePacket {
            scene: Arc::new(fused_scene(2)),
            evidence,
        });

        assert_eq!(first.scene.stamp.frame_id, 1);
        assert_eq!(
            serde_json::to_string(first.scene.as_ref()).unwrap(),
            serialized
        );
        assert_eq!(bus.latest().unwrap().scene.stamp.frame_id, 2);
    }

    #[test]
    fn measurement_context_binds_mesh_frame_to_its_exact_rgb_depth_pair() {
        let frame = paired_frame(1_000);
        let context = AssociationMeasurementContext::from_frame(&frame, 20_000).unwrap();
        let observation = context.observation(
            0,
            BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            0.9,
            None,
        );
        assert_eq!(observation.source.source_frame_id, 7);
        assert_eq!(observation.source.rgb_stamp.source_sequence, 70);
        assert_eq!(
            observation.geometry.status,
            mvm_core::AssociationGeometryStatus::Observed
        );
        assert_eq!(observation.geometry.median_z_m, Some(2.0));
        observation.validate().unwrap();
    }

    #[test]
    fn bad_depth_skew_withholds_geometry_without_dropping_rgb_observation() {
        let frame = paired_frame(2_000);
        let context = AssociationMeasurementContext::from_frame(&frame, 20_000).unwrap();
        let observation = context.observation(
            0,
            BBox {
                x1: 0.1,
                y1: 0.1,
                x2: 0.9,
                y2: 0.9,
            },
            0.9,
            None,
        );
        assert_eq!(observation.source.source_frame_id, 7);
        assert_eq!(
            observation.geometry.reason,
            Some(mvm_core::GeometryReason::DepthStale)
        );
        assert!(observation.geometry.centroid_xyz_m.is_none());
        observation.validate().unwrap();
    }

    #[test]
    fn malformed_rgb_stamp_remains_fail_closed() {
        let mut frame = paired_frame(1_000);
        Arc::make_mut(&mut frame.sensor).rgb_stamp.host_monotonic_ns = 0;
        assert!(matches!(
            AssociationMeasurementContext::from_frame(&frame, 20_000),
            Err(mvm_core::sensor::SensorContractError::MissingHostTimestamp)
        ));
    }
}
