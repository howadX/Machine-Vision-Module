use crate::{GeometryError, MAX_GEOMETRY_SAMPLES, MetricPoint, ReferenceFrame};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlaneFitConfig {
    pub residual_threshold_m: f32,
    pub min_inliers: usize,
    pub max_candidates: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlaneMeasurement {
    pub normal: [f32; 3],
    pub distance_m: f32,
    pub inliers: usize,
    pub coverage: f32,
    pub rmse_m: f32,
    pub p95_abs_m: f32,
    pub reference_frame: ReferenceFrame,
    pub calibration_revision: String,
    pub transform_revision: Option<String>,
}

pub fn fit_plane(
    points: &[MetricPoint],
    config: PlaneFitConfig,
) -> Result<PlaneMeasurement, GeometryError> {
    if points.len() < 3
        || points.len() > MAX_GEOMETRY_SAMPLES
        || config.min_inliers < 3
        || config.max_candidates == 0
        || !config.residual_threshold_m.is_finite()
        || config.residual_threshold_m <= 0.0
    {
        return Err(GeometryError::InvalidConfig);
    }
    let first = &points[0];
    if points.iter().any(|point| {
        point.reference_frame != first.reference_frame
            || point.calibration_revision != first.calibration_revision
            || point.transform_revision != first.transform_revision
            || point.xyz_m.iter().any(|value| !value.is_finite())
    }) {
        return Err(GeometryError::ReferenceFrameMismatch);
    }

    let mut best: Option<([f32; 3], f32, Vec<f32>, f32)> = None;
    let mut candidates = 0usize;
    'outer: for i in 0..points.len().saturating_sub(2) {
        for j in (i + 1)..points.len().saturating_sub(1) {
            for k in (j + 1)..points.len() {
                let Some((normal, distance)) =
                    plane_from_three(points[i].xyz_m, points[j].xyz_m, points[k].xyz_m)
                else {
                    continue;
                };
                candidates += 1;
                let mut residuals = Vec::new();
                let mut squared_sum = 0.0;
                for point in points {
                    let residual = signed_distance(normal, distance, point.xyz_m).abs();
                    if residual <= config.residual_threshold_m {
                        residuals.push(residual);
                        squared_sum += residual * residual;
                    }
                }
                let score = if residuals.is_empty() {
                    f32::INFINITY
                } else {
                    squared_sum / residuals.len() as f32
                };
                let replace = best.as_ref().is_none_or(|(_, _, current, current_score)| {
                    residuals.len() > current.len()
                        || (residuals.len() == current.len() && score < *current_score)
                });
                if replace {
                    best = Some((normal, distance, residuals, score));
                }
                if candidates >= config.max_candidates {
                    break 'outer;
                }
            }
        }
    }
    let Some((mut normal, mut distance, _, _)) = best else {
        return Err(GeometryError::DegenerateGeometry);
    };

    // Refit once from the inlier centroid and deterministic farthest-pair basis.
    let inlier_points: Vec<[f32; 3]> = points
        .iter()
        .filter(|point| {
            signed_distance(normal, distance, point.xyz_m).abs() <= config.residual_threshold_m
        })
        .map(|point| point.xyz_m)
        .collect();
    if inlier_points.len() < config.min_inliers {
        return Err(GeometryError::InsufficientSamples);
    }
    if let Some((refined_normal, refined_distance)) = refit_from_inliers(&inlier_points) {
        normal = refined_normal;
        distance = refined_distance;
    }
    canonicalize_plane(&mut normal, &mut distance);

    let mut residuals: Vec<f32> = inlier_points
        .iter()
        .map(|point| signed_distance(normal, distance, *point).abs())
        .filter(|residual| *residual <= config.residual_threshold_m)
        .collect();
    if residuals.len() < config.min_inliers {
        return Err(GeometryError::InsufficientSamples);
    }
    residuals.sort_by(f32::total_cmp);
    let rmse = (residuals.iter().map(|v| v * v).sum::<f32>() / residuals.len() as f32).sqrt();
    let p95_index = ((residuals.len() - 1) as f32 * 0.95).round() as usize;
    Ok(PlaneMeasurement {
        normal,
        distance_m: distance,
        inliers: residuals.len(),
        coverage: residuals.len() as f32 / points.len() as f32,
        rmse_m: rmse,
        p95_abs_m: residuals[p95_index],
        reference_frame: first.reference_frame,
        calibration_revision: first.calibration_revision.clone(),
        transform_revision: first.transform_revision.clone(),
    })
}

fn refit_from_inliers(points: &[[f32; 3]]) -> Option<([f32; 3], f32)> {
    let mut centroid = [0.0; 3];
    for point in points {
        for axis in 0..3 {
            centroid[axis] += point[axis];
        }
    }
    for value in &mut centroid {
        *value /= points.len() as f32;
    }
    let mut first = (0usize, 0.0f32);
    for (index, point) in points.iter().enumerate() {
        let distance = squared_distance(*point, centroid);
        if distance > first.1 {
            first = (index, distance);
        }
    }
    let mut second = (0usize, 0.0f32);
    for (index, point) in points.iter().enumerate() {
        let distance = squared_distance(*point, points[first.0]);
        if distance > second.1 {
            second = (index, distance);
        }
    }
    let axis = sub(points[second.0], points[first.0]);
    let mut widest = (0usize, 0.0f32);
    for (index, point) in points.iter().enumerate() {
        let area = norm(cross(axis, sub(*point, points[first.0])));
        if area > widest.1 {
            widest = (index, area);
        }
    }
    plane_from_three(points[first.0], points[second.0], points[widest.0]).map(|(normal, _)| {
        let distance = -dot(normal, centroid);
        (normal, distance)
    })
}

fn plane_from_three(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> Option<([f32; 3], f32)> {
    let raw = cross(sub(b, a), sub(c, a));
    let length = norm(raw);
    if !length.is_finite() || length <= 1e-8 {
        return None;
    }
    let normal = [raw[0] / length, raw[1] / length, raw[2] / length];
    Some((normal, -dot(normal, a)))
}

fn canonicalize_plane(normal: &mut [f32; 3], distance: &mut f32) {
    let first_non_zero = normal.iter().copied().find(|value| value.abs() > 1e-8);
    if first_non_zero.is_some_and(|value| value < 0.0) {
        for value in normal {
            *value = -*value;
        }
        *distance = -*distance;
    }
}

fn signed_distance(normal: [f32; 3], distance: f32, point: [f32; 3]) -> f32 {
    dot(normal, point) + distance
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn norm(value: [f32; 3]) -> f32 {
    dot(value, value).sqrt()
}

fn squared_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    dot(sub(a, b), sub(a, b))
}
