//! Same-packet metric measurement for tracker association and canonical fusion.

use mvm_core::{
    AssociationGeometry, AssociationGeometryStatus, AxisAlignedExtent, BBox, GeometryReason,
    GeometrySampling, InstanceMask, RegisteredDepthOwners, TrackerFrameBinding,
};

use crate::roi::{
    sample_registered_owner_mask_validated, sample_registered_owner_roi_validated,
    validate_owner_spec,
};
use crate::{
    CameraIntrinsics, GeometryError, MetricRoi, ReferenceFrame, RegisteredMetricDepth,
    SamplingSpec, measure_roi,
};

const INNER_BBOX_INSET: f32 = 0.20;
const MIN_SAMPLES: usize = 16;
const SPEC: SamplingSpec = SamplingSpec {
    stride: 4,
    max_samples: crate::MAX_GEOMETRY_SAMPLES,
};

/// Compute one foreground-oriented measurement from the exact RGB-D packet.
/// The returned value is retained through association and consumed by Fusion;
/// callers must not resample another frame for the same observation.
pub fn measure_association_geometry(
    binding: &TrackerFrameBinding,
    depth: Option<&RegisteredMetricDepth>,
    owners: Option<&RegisteredDepthOwners>,
    bbox: BBox,
    current_mask: Option<&InstanceMask>,
) -> AssociationGeometry {
    measure_association_geometries(binding, depth, owners, &[(bbox, current_mask)])
        .pop()
        .expect("one association geometry input")
}

/// Batch variant for one exact packet. Expensive owner-map validation is done
/// once, then every high/low candidate reuses the admitted packet contract.
pub fn measure_association_geometries(
    binding: &TrackerFrameBinding,
    depth: Option<&RegisteredMetricDepth>,
    owners: Option<&RegisteredDepthOwners>,
    inputs: &[(BBox, Option<&InstanceMask>)],
) -> Vec<AssociationGeometry> {
    let repeat = |status, reason| {
        inputs
            .iter()
            .map(|_| unavailable(binding, status, reason))
            .collect()
    };
    let Some(depth) = depth else {
        return repeat(
            AssociationGeometryStatus::Unknown,
            GeometryReason::DepthUnavailable,
        );
    };
    if binding.depth_stamp.as_ref() != Some(&depth.frame().stamp) {
        return repeat(
            AssociationGeometryStatus::Unknown,
            GeometryReason::DepthStale,
        );
    }
    if binding.calibration_revision.as_deref() != Some(depth.calibration().revision_sha256.as_str())
    {
        return repeat(
            AssociationGeometryStatus::Withheld,
            GeometryReason::CalibrationMismatch,
        );
    }
    let Some(owners) = owners else {
        return repeat(
            AssociationGeometryStatus::Withheld,
            GeometryReason::MissingOwnerAssociation,
        );
    };
    let intrinsics = match CameraIntrinsics::new(
        owners.native_width,
        owners.native_height,
        owners.native_fx_px,
        owners.native_fy_px,
        owners.native_cx_px,
        owners.native_cy_px,
        owners.calibration_revision.clone(),
        ReferenceFrame::CameraOptical,
    ) {
        Ok(value) => value,
        Err(_) => {
            return repeat(
                AssociationGeometryStatus::Withheld,
                GeometryReason::InvalidOwnerAssociation,
            );
        }
    };

    if let Err(error) = validate_owner_spec(depth, owners, &intrinsics, SPEC) {
        return repeat(AssociationGeometryStatus::Withheld, geometry_reason(error));
    }

    inputs
        .iter()
        .map(|(bbox, current_mask)| {
            measure_validated(binding, depth, owners, &intrinsics, *bbox, *current_mask)
        })
        .collect()
}

fn measure_validated(
    binding: &TrackerFrameBinding,
    depth: &RegisteredMetricDepth,
    owners: &RegisteredDepthOwners,
    intrinsics: &CameraIntrinsics,
    bbox: BBox,
    current_mask: Option<&InstanceMask>,
) -> AssociationGeometry {
    let (sampling, samples) = match current_mask {
        Some(mask) if mask.is_empty() => {
            return unavailable_with_sampling(
                binding,
                AssociationGeometryStatus::Withheld,
                GeometryReason::EmptyMask,
                GeometrySampling::Mask,
            );
        }
        Some(mask) => (
            GeometrySampling::Mask,
            sample_registered_owner_mask_validated(depth, owners, intrinsics, mask, SPEC),
        ),
        None => {
            let Some(roi) = inner_bbox_roi(bbox, owners.registered_width, owners.registered_height)
            else {
                return unavailable(
                    binding,
                    AssociationGeometryStatus::Withheld,
                    GeometryReason::InvalidBbox,
                );
            };
            (
                GeometrySampling::InnerBbox,
                sample_registered_owner_roi_validated(depth, owners, intrinsics, roi, SPEC),
            )
        }
    };
    let samples = match samples {
        Ok(value) => value,
        Err(error) => {
            return unavailable_with_sampling(
                binding,
                AssociationGeometryStatus::Withheld,
                geometry_reason(error),
                sampling,
            );
        }
    };
    let value = match measure_roi(&samples, MIN_SAMPLES) {
        Ok(value) => value,
        Err(error) => {
            return unavailable_with_sampling(
                binding,
                AssociationGeometryStatus::Withheld,
                geometry_reason(error),
                sampling,
            );
        }
    };
    let weak = value.valid_ratio < 0.5 || value.boundary_contamination > 0.5;
    AssociationGeometry {
        centroid_xyz_m: Some(value.centroid_m),
        median_z_m: Some(value.median_z_m),
        extent_m: Some(AxisAlignedExtent {
            min_m: value.aabb_min_m,
            max_m: value.aabb_max_m,
        }),
        valid_ratio: value.valid_ratio,
        depth_mad_m: Some(value.median_absolute_deviation_m),
        boundary_contamination: value.boundary_contamination,
        sample_count: u32::try_from(value.sample_count).unwrap_or(u32::MAX),
        rgb_stamp: binding.rgb_stamp.clone(),
        depth_stamp: binding.depth_stamp.clone(),
        pairing_skew_ns: binding.rgb_depth_skew_ns,
        calibration: Some(depth.calibration().clone()),
        sampling,
        status: if weak {
            AssociationGeometryStatus::Weak
        } else {
            AssociationGeometryStatus::Observed
        },
        reason: if value.boundary_contamination > 0.5 {
            Some(GeometryReason::HighBoundaryContamination)
        } else if value.valid_ratio < 0.5 {
            Some(GeometryReason::LowValidRatio)
        } else {
            None
        },
    }
}

fn inner_bbox_roi(bbox: BBox, width: u32, height: u32) -> Option<MetricRoi> {
    if width == 0
        || height == 0
        || ![bbox.x1, bbox.y1, bbox.x2, bbox.y2]
            .iter()
            .all(|v| v.is_finite())
        || bbox.x2 <= bbox.x1
        || bbox.y2 <= bbox.y1
    {
        return None;
    }
    let inset_x = (bbox.x2 - bbox.x1) * INNER_BBOX_INSET;
    let inset_y = (bbox.y2 - bbox.y1) * INNER_BBOX_INSET;
    let x1 = ((bbox.x1 + inset_x).clamp(0.0, 1.0) * width as f32).floor() as u32;
    let y1 = ((bbox.y1 + inset_y).clamp(0.0, 1.0) * height as f32).floor() as u32;
    let x2 = ((bbox.x2 - inset_x).clamp(0.0, 1.0) * width as f32).ceil() as u32;
    let y2 = ((bbox.y2 - inset_y).clamp(0.0, 1.0) * height as f32).ceil() as u32;
    (x2 > x1 && y2 > y1).then(|| MetricRoi::pixels(x1, y1, x2 - x1, y2 - y1))
}

fn geometry_reason(error: GeometryError) -> GeometryReason {
    match error {
        GeometryError::CalibrationMismatch => GeometryReason::CalibrationMismatch,
        GeometryError::InvalidOwnerAssociation | GeometryError::DimensionsMismatch => {
            GeometryReason::InvalidOwnerAssociation
        }
        GeometryError::InsufficientSamples => GeometryReason::InsufficientSamples,
        _ => GeometryReason::InvalidBbox,
    }
}

fn unavailable(
    binding: &TrackerFrameBinding,
    status: AssociationGeometryStatus,
    reason: GeometryReason,
) -> AssociationGeometry {
    unavailable_with_sampling(binding, status, reason, GeometrySampling::None)
}

fn unavailable_with_sampling(
    binding: &TrackerFrameBinding,
    status: AssociationGeometryStatus,
    reason: GeometryReason,
    sampling: GeometrySampling,
) -> AssociationGeometry {
    let mut value = AssociationGeometry::unavailable(binding, reason);
    value.status = status;
    value.sampling = sampling;
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::{
        CalibrationRef, DepthGeometry, DepthStatus, MetricDepthFrame, RegisteredDepthOwners,
        SensorClockDomain, SensorSourceId, SensorStamp,
    };
    use std::sync::Arc;

    fn fixture() -> (
        TrackerFrameBinding,
        RegisteredMetricDepth,
        RegisteredDepthOwners,
    ) {
        let revision = "a".repeat(64);
        let rgb_stamp = SensorStamp {
            source_sequence: 7,
            host_monotonic_ns: 1_000_000,
            wall_clock_unix_ms: 1_700_000_000_000,
            clock_domain: SensorClockDomain::KinectV1Rgb,
            device_timestamp_raw: Some(10),
        };
        let depth_stamp = SensorStamp {
            source_sequence: 7,
            host_monotonic_ns: 1_001_000,
            wall_clock_unix_ms: 1_700_000_000_001,
            clock_domain: SensorClockDomain::KinectV1Depth,
            device_timestamp_raw: Some(11),
        };
        let calibration = CalibrationRef {
            schema: "test/1".into(),
            camera_serial: "serial".into(),
            revision_sha256: revision.clone(),
        };
        let frame = Arc::new(MetricDepthFrame {
            width: 64,
            height: 64,
            depth_mm: vec![2_000; 64 * 64],
            invalid_value: 0,
            stamp: depth_stamp,
            geometry: DepthGeometry::RegisteredToRgb,
            calibration_revision: revision.clone(),
            valid_ratio: 1.0,
            rgb_depth_skew_ns: 1_000,
        });
        let depth = RegisteredMetricDepth::admit(
            64,
            64,
            &rgb_stamp,
            &DepthStatus::Paired,
            Arc::clone(&frame),
            &calibration,
            20_000,
        )
        .unwrap();
        let source = SensorSourceId {
            backend: "kinect_v1".into(),
            logical_name: "test".into(),
            serial: Some("serial".into()),
        };
        let binding = TrackerFrameBinding::from_sensor_parts(
            7,
            &source,
            64,
            64,
            &rgb_stamp,
            &DepthStatus::Paired,
            Some(frame.as_ref()),
            Some(&calibration),
            20_000,
        )
        .unwrap();
        let owners = RegisteredDepthOwners {
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
        };
        (binding, depth, owners)
    }

    #[test]
    fn bbox_measurement_uses_inner_bbox_and_exact_pair() {
        let (binding, depth, owners) = fixture();
        let value = measure_association_geometry(
            &binding,
            Some(&depth),
            Some(&owners),
            BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            None,
        );
        assert_eq!(value.status, AssociationGeometryStatus::Observed);
        assert_eq!(value.sampling, GeometrySampling::InnerBbox);
        assert_eq!(value.depth_stamp, binding.depth_stamp);
        assert_eq!(value.median_z_m, Some(2.0));
        value.validate_for(&binding).unwrap();

        let mask = InstanceMask::from_bits(vec![1; mvm_core::MASK_RES * mvm_core::MASK_RES]);
        let masked = measure_association_geometry(
            &binding,
            Some(&depth),
            Some(&owners),
            BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            },
            Some(&mask),
        );
        assert_eq!(masked.status, AssociationGeometryStatus::Observed);
        assert_eq!(masked.sampling, GeometrySampling::Mask);
        assert_eq!(masked.median_z_m, Some(2.0));
        masked.validate_for(&binding).unwrap();
    }

    #[test]
    fn batch_measurement_matches_individual_results() {
        let (binding, depth, owners) = fixture();
        let left = BBox {
            x1: 0.0,
            y1: 0.0,
            x2: 0.5,
            y2: 1.0,
        };
        let right = BBox {
            x1: 0.5,
            y1: 0.0,
            x2: 1.0,
            y2: 1.0,
        };
        let batch = measure_association_geometries(
            &binding,
            Some(&depth),
            Some(&owners),
            &[(left, None), (right, None)],
        );
        assert_eq!(
            batch,
            vec![
                measure_association_geometry(&binding, Some(&depth), Some(&owners), left, None,),
                measure_association_geometry(&binding, Some(&depth), Some(&owners), right, None,),
            ]
        );
    }

    #[test]
    fn missing_owner_and_empty_mask_fail_soft_with_typed_reasons() {
        let (binding, depth, owners) = fixture();
        let bbox = BBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.9,
            y2: 0.9,
        };
        let ownerless = measure_association_geometry(&binding, Some(&depth), None, bbox, None);
        assert_eq!(ownerless.status, AssociationGeometryStatus::Withheld);
        assert_eq!(
            ownerless.reason,
            Some(GeometryReason::MissingOwnerAssociation)
        );

        let empty = InstanceMask::empty();
        let masked =
            measure_association_geometry(&binding, Some(&depth), Some(&owners), bbox, Some(&empty));
        assert_eq!(masked.status, AssociationGeometryStatus::Withheld);
        assert_eq!(masked.reason, Some(GeometryReason::EmptyMask));
    }

    #[test]
    fn stale_depth_and_owner_calibration_mismatch_fail_soft() {
        let (binding, depth, mut owners) = fixture();
        let bbox = BBox {
            x1: 0.1,
            y1: 0.1,
            x2: 0.9,
            y2: 0.9,
        };

        let mut stale_binding = binding.clone();
        stale_binding.depth_stamp.as_mut().unwrap().source_sequence += 1;
        let stale =
            measure_association_geometry(&stale_binding, Some(&depth), Some(&owners), bbox, None);
        assert_eq!(stale.status, AssociationGeometryStatus::Unknown);
        assert_eq!(stale.reason, Some(GeometryReason::DepthStale));

        owners.calibration_revision = "b".repeat(64);
        let mismatched =
            measure_association_geometry(&binding, Some(&depth), Some(&owners), bbox, None);
        assert_eq!(mismatched.status, AssociationGeometryStatus::Withheld);
        assert_eq!(mismatched.reason, Some(GeometryReason::CalibrationMismatch));
    }
}
