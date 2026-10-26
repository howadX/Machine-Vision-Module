use std::sync::Arc;

use mvm_core::{
    CalibrationRef, DepthGeometry, DepthStatus, InstanceMask, MASK_RES, MetricDepthFrame,
    RegisteredDepthOwners, SensorClockDomain, SensorStamp,
};
use mvm_depth::{
    CameraIntrinsics, DepthVisibility, GeometryStatus, MetricRoi, MountTransform, PlaneFitConfig,
    ReferenceFrame, SamplingSpec, fit_plane, measure_roi, sample_mask, sample_roi,
};

fn stamp(ns: u64) -> SensorStamp {
    SensorStamp {
        source_sequence: 1,
        host_monotonic_ns: ns,
        wall_clock_unix_ms: 1_700_000_000_000,
        clock_domain: SensorClockDomain::KinectV1Depth,
        device_timestamp_raw: Some(1),
    }
}

fn admitted(width: u32, height: u32, depth_mm: Vec<u16>) -> mvm_depth::RegisteredMetricDepth {
    let revision = "a".repeat(64);
    let calibration = CalibrationRef {
        schema: "test-camera/1".into(),
        camera_serial: "serial".into(),
        revision_sha256: revision.clone(),
    };
    let depth = Arc::new(MetricDepthFrame {
        width,
        height,
        valid_ratio: depth_mm.iter().filter(|&&z| z != 0).count() as f32 / depth_mm.len() as f32,
        depth_mm,
        invalid_value: 0,
        stamp: stamp(1_000_000),
        geometry: DepthGeometry::RegisteredToRgb,
        calibration_revision: revision,
        rgb_depth_skew_ns: 0,
    });
    mvm_depth::RegisteredMetricDepth::admit(
        width,
        height,
        &stamp(1_000_000),
        &DepthStatus::Paired,
        depth,
        &calibration,
        20_000_000,
    )
    .unwrap()
}

fn intrinsics(width: u32, height: u32) -> CameraIntrinsics {
    CameraIntrinsics::new(
        width,
        height,
        100.0,
        100.0,
        width as f32 / 2.0,
        height as f32 / 2.0,
        "a".repeat(64),
        ReferenceFrame::CameraOptical,
    )
    .unwrap()
}

#[test]
fn project_round_trip_preserves_signs_units_and_revision() {
    let camera = intrinsics(8, 6);
    let p = camera.unproject(6.0, 1.0, 2_000).unwrap();
    assert!(p.xyz_m[0] > 0.0, "right pixel is +X");
    assert!(p.xyz_m[1] < 0.0, "upper pixel is -Y");
    assert_eq!(p.xyz_m[2], 2.0, "millimetres become metres");
    assert_eq!(p.calibration_revision, "a".repeat(64));
    let pixel = camera.project(&p).unwrap();
    assert!((pixel[0] - 6.0).abs() < 1e-5);
    assert!((pixel[1] - 1.0).abs() < 1e-5);
}

#[test]
fn invalid_depth_is_absent_and_sampling_is_bounded() {
    let depth = admitted(
        8,
        6,
        (0..48)
            .map(|i| if i % 5 == 0 { 0 } else { 1_000 })
            .collect(),
    );
    assert!(
        depth
            .unproject_pixel(0, 0, &intrinsics(8, 6))
            .unwrap()
            .is_none()
    );
    let samples = sample_roi(
        &depth,
        &intrinsics(8, 6),
        MetricRoi::pixels(0, 0, 8, 6),
        SamplingSpec {
            stride: 1,
            max_samples: 7,
        },
    )
    .unwrap();
    assert!(samples.points.len() <= 7);
    assert!(samples.invalid_pixels > 0);
    assert_eq!(
        sample_roi(
            &depth,
            &intrinsics(8, 6),
            MetricRoi::pixels(0, 0, 8, 6),
            SamplingSpec {
                stride: 1,
                max_samples: mvm_depth::MAX_GEOMETRY_SAMPLES + 1
            },
        ),
        Err(mvm_depth::GeometryError::InvalidConfig)
    );
}

#[test]
fn geometry_rejects_unverified_dimensions_and_calibration() {
    let depth = admitted(8, 6, vec![1_000; 48]);
    let wrong_dimensions = intrinsics(7, 6);
    assert_eq!(
        depth.unproject_pixel(0, 0, &wrong_dimensions),
        Err(mvm_depth::GeometryError::DimensionsMismatch)
    );
    let mut wrong_revision = intrinsics(8, 6);
    wrong_revision.calibration_revision = "b".repeat(64);
    assert_eq!(
        depth.unproject_pixel(0, 0, &wrong_revision),
        Err(mvm_depth::GeometryError::CalibrationMismatch)
    );
}

#[test]
fn registered_sample_unprojects_through_exact_native_owner_ray() {
    let depth = admitted(4, 3, vec![0, 0, 0, 0, 0, 2_000, 0, 0, 0, 0, 0, 0]);
    let mut owner_index = vec![RegisteredDepthOwners::NO_OWNER; 12];
    owner_index[5] = 11;
    let owners = RegisteredDepthOwners {
        registered_width: 4,
        registered_height: 3,
        native_width: 4,
        native_height: 3,
        native_fx_px: 200.0,
        native_fy_px: 200.0,
        native_cx_px: 1.0,
        native_cy_px: 1.0,
        owner_index,
        calibration_revision: "a".repeat(64),
    };
    let point = depth
        .unproject_registered_owner(1, 1, &owners, &intrinsics(4, 3))
        .unwrap()
        .unwrap();
    assert_eq!(point.xyz_m, [0.02, 0.01, 2.0]);
    assert_ne!(
        point.xyz_m,
        depth
            .unproject_pixel(1, 1, &intrinsics(4, 3))
            .unwrap()
            .unwrap()
            .xyz_m
    );
}

#[test]
fn valid_registered_depth_without_owner_fails_closed() {
    let depth = admitted(2, 1, vec![1_000, 0]);
    let owners = RegisteredDepthOwners {
        registered_width: 2,
        registered_height: 1,
        native_width: 2,
        native_height: 1,
        native_fx_px: 200.0,
        native_fy_px: 200.0,
        native_cx_px: 1.0,
        native_cy_px: 1.0,
        owner_index: vec![RegisteredDepthOwners::NO_OWNER; 2],
        calibration_revision: "a".repeat(64),
    };
    assert_eq!(
        depth.unproject_registered_owner(0, 0, &owners, &intrinsics(2, 1)),
        Err(mvm_depth::GeometryError::InvalidOwnerAssociation)
    );
}

#[test]
fn mask_sampling_excludes_bbox_background_and_measurement_is_robust() {
    let mut map = vec![4_000u16; 64 * 64];
    let mut bits = vec![0u8; MASK_RES * MASK_RES];
    for y in 24..40 {
        for x in 24..40 {
            map[y * 64 + x] = if (x + y) % 13 == 0 { 0 } else { 1_500 };
            bits[y * MASK_RES + x] = 1;
        }
    }
    let depth = admitted(64, 64, map);
    let mask = InstanceMask::from_bits(bits);
    let samples = sample_mask(
        &depth,
        &intrinsics(64, 64),
        &mask,
        SamplingSpec {
            stride: 1,
            max_samples: 512,
        },
    )
    .unwrap();
    let measurement = measure_roi(&samples, 16).unwrap();
    assert_eq!(measurement.status, GeometryStatus::Measured);
    assert!((measurement.median_z_m - 1.5).abs() < 1e-6);
    assert!(measurement.valid_ratio < 1.0);
    assert!(measurement.centroid_m[2] > 1.49 && measurement.centroid_m[2] < 1.51);
}

#[test]
fn plane_fit_is_deterministic_with_holes_and_outlier() {
    let mut data = Vec::new();
    for y in 0..20 {
        for x in 0..20 {
            if (x + y) % 17 == 0 {
                data.push(0);
            } else if x == 19 && y == 19 {
                data.push(4_000);
            } else {
                data.push(2_000);
            }
        }
    }
    let depth = admitted(20, 20, data);
    let samples = sample_roi(
        &depth,
        &intrinsics(20, 20),
        MetricRoi::pixels(0, 0, 20, 20),
        SamplingSpec {
            stride: 1,
            max_samples: 400,
        },
    )
    .unwrap();
    let cfg = PlaneFitConfig {
        residual_threshold_m: 0.01,
        min_inliers: 100,
        max_candidates: 64,
    };
    let a = fit_plane(&samples.points, cfg).unwrap();
    let b = fit_plane(&samples.points, cfg).unwrap();
    assert_eq!(a, b);
    assert!(a.normal[2].abs() > 0.999);
    assert!(a.rmse_m < 1e-5);
    assert!(a.inliers >= 350);
}

#[test]
fn mount_transform_propagates_reference_and_revision() {
    let camera = intrinsics(8, 6);
    let p = camera.unproject(4.0, 3.0, 1_000).unwrap();
    let mount = MountTransform::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        [1.0, 2.0, 3.0],
        "mount-r1",
    )
    .unwrap();
    let world = mount.camera_to_local_world(&p).unwrap();
    assert_eq!(world.reference_frame, ReferenceFrame::LocalWorld);
    assert_eq!(world.transform_revision.as_deref(), Some("mount-r1"));
    assert_eq!(world.xyz_m, [1.0, 2.0, 4.0]);
}

#[test]
fn mount_transform_rejects_reflection() {
    assert_eq!(
        MountTransform::new(
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [0.0; 3],
            "reflection",
        ),
        Err(mvm_depth::GeometryError::InvalidTransform)
    );
}

#[test]
fn visibility_is_typed_and_never_invents_free_space() {
    let depth = admitted(2, 2, vec![0, 1_000, 2_000, 3_000]);
    assert_eq!(
        depth.visibility_at(0, 0, Some(1_000), 50).unwrap(),
        DepthVisibility::Unknown
    );
    assert_eq!(
        depth.visibility_at(1, 0, Some(1_020), 50).unwrap(),
        DepthVisibility::Visible
    );
    assert_eq!(
        depth.visibility_at(1, 0, Some(1_500), 50).unwrap(),
        DepthVisibility::Occluded
    );
    assert_eq!(
        depth.visibility_at(1, 0, Some(800), 50).unwrap(),
        DepthVisibility::OccupiedSurface
    );
}
