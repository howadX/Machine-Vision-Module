use mvm_core::{InstanceMask, MASK_RES, RegisteredDepthOwners};

use crate::{
    CameraIntrinsics, GeometryError, GeometryMeasurement, GeometryStatus, MAX_GEOMETRY_SAMPLES,
    MetricPoint, ReferenceFrame, RegisteredMetricDepth,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricRoi {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl MetricRoi {
    pub const fn pixels(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SamplingSpec {
    pub stride: u32,
    pub max_samples: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricSamples {
    pub points: Vec<MetricPoint>,
    pub considered_pixels: usize,
    pub invalid_pixels: usize,
    pub boundary_valid_pixels: usize,
    pub boundary_considered_pixels: usize,
    pub reference_frame: ReferenceFrame,
    pub calibration_revision: String,
}

pub fn sample_roi(
    depth: &RegisteredMetricDepth,
    intrinsics: &CameraIntrinsics,
    roi: MetricRoi,
    spec: SamplingSpec,
) -> Result<MetricSamples, GeometryError> {
    validate_spec(depth, intrinsics, spec)?;
    if roi.width == 0 || roi.height == 0 {
        return Err(GeometryError::InvalidConfig);
    }
    let x_end = roi.x.saturating_add(roi.width).min(intrinsics.width);
    let y_end = roi.y.saturating_add(roi.height).min(intrinsics.height);
    if roi.x >= x_end || roi.y >= y_end {
        return Err(GeometryError::InvalidPixel);
    }
    sample_coordinates(
        depth,
        intrinsics,
        spec,
        (roi.y..y_end)
            .step_by(spec.stride as usize)
            .flat_map(move |y| {
                (roi.x..x_end).step_by(spec.stride as usize).map(move |x| {
                    let boundary = x == roi.x
                        || y == roi.y
                        || x.saturating_add(spec.stride) >= x_end
                        || y.saturating_add(spec.stride) >= y_end;
                    (x, y, boundary)
                })
            }),
    )
}

pub fn sample_mask(
    depth: &RegisteredMetricDepth,
    intrinsics: &CameraIntrinsics,
    mask: &InstanceMask,
    spec: SamplingSpec,
) -> Result<MetricSamples, GeometryError> {
    validate_spec(depth, intrinsics, spec)?;
    let delta = 1.0 / MASK_RES as f32;
    let coordinates = mask.set_points().enumerate().filter_map(|(ordinal, p)| {
        if !ordinal.is_multiple_of(spec.stride as usize) {
            return None;
        }
        let x = (p[0] * intrinsics.width as f32) as u32;
        let y = (p[1] * intrinsics.height as f32) as u32;
        let boundary = !mask.contains_norm([p[0] - delta, p[1]])
            || !mask.contains_norm([p[0] + delta, p[1]])
            || !mask.contains_norm([p[0], p[1] - delta])
            || !mask.contains_norm([p[0], p[1] + delta]);
        Some((
            x.min(intrinsics.width - 1),
            y.min(intrinsics.height - 1),
            boundary,
        ))
    });
    sample_coordinates(depth, intrinsics, spec, coordinates)
}

pub fn sample_registered_owner_roi(
    depth: &RegisteredMetricDepth,
    owners: &RegisteredDepthOwners,
    native_intrinsics: &CameraIntrinsics,
    roi: MetricRoi,
    spec: SamplingSpec,
) -> Result<MetricSamples, GeometryError> {
    validate_owner_spec(depth, owners, native_intrinsics, spec)?;
    if roi.width == 0 || roi.height == 0 {
        return Err(GeometryError::InvalidConfig);
    }
    let x_end = roi.x.saturating_add(roi.width).min(owners.registered_width);
    let y_end = roi
        .y
        .saturating_add(roi.height)
        .min(owners.registered_height);
    if roi.x >= x_end || roi.y >= y_end {
        return Err(GeometryError::InvalidPixel);
    }
    sample_owner_coordinates(
        depth,
        owners,
        native_intrinsics,
        spec,
        (roi.y..y_end)
            .step_by(spec.stride as usize)
            .flat_map(move |y| {
                (roi.x..x_end).step_by(spec.stride as usize).map(move |x| {
                    let boundary = x == roi.x
                        || y == roi.y
                        || x.saturating_add(spec.stride) >= x_end
                        || y.saturating_add(spec.stride) >= y_end;
                    (x, y, boundary)
                })
            }),
    )
}

pub fn sample_registered_owner_mask(
    depth: &RegisteredMetricDepth,
    owners: &RegisteredDepthOwners,
    native_intrinsics: &CameraIntrinsics,
    mask: &InstanceMask,
    spec: SamplingSpec,
) -> Result<MetricSamples, GeometryError> {
    validate_owner_spec(depth, owners, native_intrinsics, spec)?;
    let delta = 1.0 / MASK_RES as f32;
    let coordinates = mask.set_points().enumerate().filter_map(|(ordinal, p)| {
        if !ordinal.is_multiple_of(spec.stride as usize) {
            return None;
        }
        let x = (p[0] * owners.registered_width as f32) as u32;
        let y = (p[1] * owners.registered_height as f32) as u32;
        Some((
            x.min(owners.registered_width - 1),
            y.min(owners.registered_height - 1),
            !mask.contains_norm([p[0] - delta, p[1]])
                || !mask.contains_norm([p[0] + delta, p[1]])
                || !mask.contains_norm([p[0], p[1] - delta])
                || !mask.contains_norm([p[0], p[1] + delta]),
        ))
    });
    sample_owner_coordinates(depth, owners, native_intrinsics, spec, coordinates)
}

fn validate_owner_spec(
    depth: &RegisteredMetricDepth,
    owners: &RegisteredDepthOwners,
    intrinsics: &CameraIntrinsics,
    spec: SamplingSpec,
) -> Result<(), GeometryError> {
    if spec.stride == 0 || spec.max_samples == 0 || spec.max_samples > MAX_GEOMETRY_SAMPLES {
        return Err(GeometryError::InvalidConfig);
    }
    owners
        .validate()
        .map_err(|_| GeometryError::InvalidOwnerAssociation)?;
    intrinsics.validate()?;
    if depth.frame().width != owners.registered_width
        || depth.frame().height != owners.registered_height
        || owners.native_width != intrinsics.width
        || owners.native_height != intrinsics.height
    {
        return Err(GeometryError::DimensionsMismatch);
    }
    if depth.frame().calibration_revision != owners.calibration_revision
        || depth.frame().calibration_revision != intrinsics.calibration_revision
    {
        return Err(GeometryError::CalibrationMismatch);
    }
    Ok(())
}

fn sample_owner_coordinates(
    depth: &RegisteredMetricDepth,
    owners: &RegisteredDepthOwners,
    intrinsics: &CameraIntrinsics,
    spec: SamplingSpec,
    coordinates: impl Iterator<Item = (u32, u32, bool)>,
) -> Result<MetricSamples, GeometryError> {
    let mut points = Vec::with_capacity(spec.max_samples.min(MAX_GEOMETRY_SAMPLES));
    let mut considered = 0;
    let mut invalid = 0;
    let mut boundary_valid = 0;
    let mut boundary_considered = 0;
    for (x, y, boundary) in coordinates.take(spec.max_samples) {
        considered += 1;
        boundary_considered += usize::from(boundary);
        match depth.unproject_registered_owner(x, y, owners, intrinsics)? {
            Some(point) => {
                boundary_valid += usize::from(boundary);
                points.push(point);
            }
            None => invalid += 1,
        }
    }
    Ok(MetricSamples {
        points,
        considered_pixels: considered,
        invalid_pixels: invalid,
        boundary_valid_pixels: boundary_valid,
        boundary_considered_pixels: boundary_considered,
        reference_frame: intrinsics.reference_frame,
        calibration_revision: intrinsics.calibration_revision.clone(),
    })
}

fn validate_spec(
    depth: &RegisteredMetricDepth,
    intrinsics: &CameraIntrinsics,
    spec: SamplingSpec,
) -> Result<(), GeometryError> {
    if spec.stride == 0 || spec.max_samples == 0 || spec.max_samples > MAX_GEOMETRY_SAMPLES {
        return Err(GeometryError::InvalidConfig);
    }
    if depth.frame().width != intrinsics.width || depth.frame().height != intrinsics.height {
        return Err(GeometryError::DimensionsMismatch);
    }
    if depth.frame().calibration_revision != intrinsics.calibration_revision {
        return Err(GeometryError::CalibrationMismatch);
    }
    Ok(())
}

fn sample_coordinates(
    depth: &RegisteredMetricDepth,
    intrinsics: &CameraIntrinsics,
    spec: SamplingSpec,
    coordinates: impl Iterator<Item = (u32, u32, bool)>,
) -> Result<MetricSamples, GeometryError> {
    let mut points = Vec::with_capacity(spec.max_samples.min(4_096));
    let mut considered = 0;
    let mut invalid = 0;
    let mut boundary_valid = 0;
    let mut boundary_considered = 0;
    for (x, y, boundary) in coordinates.take(spec.max_samples) {
        considered += 1;
        if boundary {
            boundary_considered += 1;
        }
        match depth.unproject_pixel(x, y, intrinsics)? {
            Some(point) => {
                if boundary {
                    boundary_valid += 1;
                }
                points.push(point);
            }
            None => invalid += 1,
        }
    }
    Ok(MetricSamples {
        points,
        considered_pixels: considered,
        invalid_pixels: invalid,
        boundary_valid_pixels: boundary_valid,
        boundary_considered_pixels: boundary_considered,
        reference_frame: intrinsics.reference_frame,
        calibration_revision: intrinsics.calibration_revision.clone(),
    })
}

pub fn measure_roi(
    samples: &MetricSamples,
    min_samples: usize,
) -> Result<GeometryMeasurement, GeometryError> {
    if min_samples == 0 || samples.points.len() < min_samples || samples.considered_pixels == 0 {
        return Err(GeometryError::InsufficientSamples);
    }
    let mut ordered: Vec<&MetricPoint> = samples.points.iter().collect();
    ordered.sort_by(|a, b| a.xyz_m[2].total_cmp(&b.xyz_m[2]));
    let median_z = ordered[ordered.len() / 2].xyz_m[2];
    let mut deviations: Vec<f32> = ordered
        .iter()
        .map(|point| (point.xyz_m[2] - median_z).abs())
        .collect();
    deviations.sort_by(f32::total_cmp);
    let mad = deviations[deviations.len() / 2];
    let trim = ordered.len() / 10;
    let robust = &ordered[trim..ordered.len().saturating_sub(trim).max(trim + 1)];
    let mut centroid = [0.0; 3];
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    for point in robust {
        for axis in 0..3 {
            centroid[axis] += point.xyz_m[axis];
            min[axis] = min[axis].min(point.xyz_m[axis]);
            max[axis] = max[axis].max(point.xyz_m[axis]);
        }
    }
    for value in &mut centroid {
        *value /= robust.len() as f32;
    }
    let boundary_contamination = if samples.boundary_considered_pixels == 0 {
        0.0
    } else {
        1.0 - samples.boundary_valid_pixels as f32 / samples.boundary_considered_pixels as f32
    };
    Ok(GeometryMeasurement {
        status: GeometryStatus::Measured,
        sample_count: samples.points.len(),
        valid_ratio: samples.points.len() as f32 / samples.considered_pixels as f32,
        median_z_m: median_z,
        median_absolute_deviation_m: mad,
        centroid_m: centroid,
        aabb_min_m: min,
        aabb_max_m: max,
        nearest_surface_m: ordered.first().unwrap().xyz_m[2],
        farthest_surface_m: ordered.last().unwrap().xyz_m[2],
        boundary_contamination,
        reference_frame: samples.reference_frame,
        calibration_revision: samples.calibration_revision.clone(),
    })
}
