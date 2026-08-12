//! mvm-depth — typed relative and metric depth evidence.
//!
//! Phase 8.E.2 (G₄ Environment group). Produces a per-frame relative depth map so
//! scene objects — which carry no metric depth of their own (phase 8.E gives them
//! `pos_3d=[0,0,0]`) — can be placed in 3D by **anchoring** the relative map to a
//! person's known metric Z (torso-3D). That turns the object↔person relation from
//! 2D-overlap-only into a real 3D `Near` edge.
//!
//! The network (DINOv2 ViT-S) outputs an affine-relative inverse-depth-like map:
//! **larger value = nearer**. It is NOT metric on its own — the anchoring step
//! (in the CLI) maps it to metres using a person as the reference point.
//!
//! Heavy on CPU (~230 ms quantized), so it runs at a **low cadence** (every Nth
//! frame), reusing the last map in between. This is exactly the head-of-line
//! blocking that will later justify the threads-mesh (phase 8.E.3+).

use anyhow::Context;
use image::DynamicImage;
use mvm_core::{BBox, CalibrationRef, DepthGeometry, DepthStatus, MetricDepthFrame, SensorStamp};
use ort::{session::Session, value::Tensor};
use std::sync::Arc;

mod metric;
mod plane;
mod roi;
mod unproject;
mod visibility;

pub use metric::{
    GeometryError, GeometryMeasurement, GeometryStatus, MAX_GEOMETRY_SAMPLES, MetricPoint,
    ReferenceFrame,
};
pub use plane::{PlaneFitConfig, PlaneMeasurement, fit_plane};
pub use roi::{
    MetricRoi, MetricSamples, SamplingSpec, measure_roi, sample_mask, sample_registered_owner_mask,
    sample_registered_owner_roi, sample_roi,
};
pub use unproject::{CameraIntrinsics, MountTransform};
pub use visibility::DepthVisibility;

/// ImageNet normalization (Depth-Anything preprocessing): pixel/255 → (v-mean)/std.
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// A relative depth map. `data` is row-major `h*w`; larger = nearer. Values are
/// affine-relative (not metres) until anchored to a metric reference.
#[derive(Debug, Clone)]
pub struct RelativeDepthMap {
    pub data: Vec<f32>,
    pub w: usize,
    pub h: usize,
}

impl RelativeDepthMap {
    /// Nearest-pixel depth at a normalized [0,1] coordinate.
    pub fn at_normalized(&self, nx: f32, ny: f32) -> f32 {
        let ix = ((nx * self.w as f32) as isize).clamp(0, self.w as isize - 1) as usize;
        let iy = ((ny * self.h as f32) as isize).clamp(0, self.h as isize - 1) as usize;
        self.data[iy * self.w + ix]
    }

    /// Robust depth for a normalized bbox: the median over a grid sampled inside
    /// the **central half** of the box (avoids edge/background bleed). Returns
    /// `None` for a degenerate box.
    pub fn median_in_bbox(&self, b: &BBox) -> Option<f32> {
        let cx = (b.x1 + b.x2) * 0.5;
        let cy = (b.y1 + b.y2) * 0.5;
        let hw = (b.x2 - b.x1).abs() * 0.25; // quarter-width → central half span
        let hh = (b.y2 - b.y1).abs() * 0.25;
        if hw <= 0.0 || hh <= 0.0 {
            return None;
        }
        let mut samples = Vec::with_capacity(25);
        for gy in 0..5 {
            for gx in 0..5 {
                let nx = cx + (gx as f32 / 4.0 - 0.5) * 2.0 * hw;
                let ny = cy + (gy as f32 / 4.0 - 0.5) * 2.0 * hh;
                samples.push(self.at_normalized(nx, ny));
            }
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(samples[samples.len() / 2])
    }

    /// Robust depth over an arbitrary set of normalized sample points — the
    /// mask-aware twin of [`median_in_bbox`](Self::median_in_bbox) (phase 10.4).
    /// An instance mask supplies its set-cell centres, so background inside the
    /// bbox (which `median_in_bbox` samples) is excluded → less depth bleed on
    /// non-convex / partially-occluded objects. `None` when no points are given.
    pub fn median_at_points(&self, points: impl Iterator<Item = [f32; 2]>) -> Option<f32> {
        let mut samples: Vec<f32> = points.map(|p| self.at_normalized(p[0], p[1])).collect();
        if samples.is_empty() {
            return None;
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(samples[samples.len() / 2])
    }
}

/// Fresh, calibrated metric depth registered to the current RGB frame.
///
/// Construction is intentionally fallible: callers cannot obtain this type
/// from stale, unregistered, mismatched, or malformed sensor evidence. It is
/// distinct from [`RelativeDepthMap`], so relative model output cannot be
/// interpreted as millimetres by accident.
#[derive(Debug, Clone)]
pub struct RegisteredMetricDepth {
    frame: Arc<MetricDepthFrame>,
    calibration: CalibrationRef,
}

impl RegisteredMetricDepth {
    pub fn admit(
        rgb_width: u32,
        rgb_height: u32,
        rgb_stamp: &SensorStamp,
        status: &DepthStatus,
        frame: Arc<MetricDepthFrame>,
        calibration: &CalibrationRef,
        max_pair_skew_ns: u64,
    ) -> Result<Self, MetricDepthAdmissionError> {
        if *status != DepthStatus::Paired {
            return Err(MetricDepthAdmissionError::NotPaired);
        }
        frame
            .validate()
            .map_err(|_| MetricDepthAdmissionError::InvalidFrame)?;
        calibration
            .validate()
            .map_err(|_| MetricDepthAdmissionError::InvalidCalibration)?;
        if frame.geometry != DepthGeometry::RegisteredToRgb {
            return Err(MetricDepthAdmissionError::Unregistered);
        }
        if frame.width != rgb_width || frame.height != rgb_height {
            return Err(MetricDepthAdmissionError::GeometryMismatch);
        }
        if frame.calibration_revision != calibration.revision_sha256 {
            return Err(MetricDepthAdmissionError::CalibrationMismatch);
        }
        let measured_skew = rgb_stamp.host_skew_ns(&frame.stamp);
        if measured_skew != frame.rgb_depth_skew_ns || measured_skew > max_pair_skew_ns {
            return Err(MetricDepthAdmissionError::Stale);
        }
        Ok(Self {
            frame,
            calibration: calibration.clone(),
        })
    }

    pub fn frame(&self) -> &Arc<MetricDepthFrame> {
        &self.frame
    }

    pub fn calibration(&self) -> &CalibrationRef {
        &self.calibration
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDepthAdmissionError {
    NotPaired,
    InvalidFrame,
    InvalidCalibration,
    Unregistered,
    GeometryMismatch,
    CalibrationMismatch,
    Stale,
}

/// Depth-Anything-V2-Small ONNX estimator.
pub struct DepthEstimator {
    session: Session,
    input_size: u32,
}

impl DepthEstimator {
    const INPUT_NAME: &'static str = "pixel_values";

    /// Load from a file path. Input is fixed to 518×518 (the canonical size; the
    /// model is patch-14 so it must be a multiple of 14 — 518 = 14×37).
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load Depth-Anything from {}", path.display()))?;
        Ok(Self {
            session,
            input_size: 518,
        })
    }

    /// Estimate a relative depth map for the frame (resized to the model input;
    /// normalized coordinates are preserved, so sampling by normalized bbox works
    /// regardless of the aspect squash).
    pub fn estimate(&mut self, image: &DynamicImage) -> anyhow::Result<RelativeDepthMap> {
        let input = self.preprocess(image)?;
        let outputs = self.session.run(ort::inputs![Self::INPUT_NAME => input])?;
        let (shape, raw) = outputs[0].try_extract_tensor::<f32>()?;
        // Output [1, H, W] (H=W=input_size). Derive h,w from the shape tail.
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let (h, w) = match dims.as_slice() {
            [_, h, w] => (*h, *w),
            [_, _, h, w] => (*h, *w),
            _ => (self.input_size as usize, self.input_size as usize),
        };
        Ok(RelativeDepthMap {
            data: raw.to_vec(),
            w,
            h,
        })
    }

    /// Resize to input_size², RGB, /255, ImageNet-normalize, NCHW.
    fn preprocess(&self, image: &DynamicImage) -> anyhow::Result<Tensor<f32>> {
        let sz = self.input_size as usize;
        let resized = image
            .resize_exact(
                self.input_size,
                self.input_size,
                image::imageops::FilterType::Triangle,
            )
            .to_rgb8();
        let mut flat = vec![0.0f32; 3 * sz * sz];
        for y in 0..sz {
            for x in 0..sz {
                let px = resized.get_pixel(x as u32, y as u32);
                let idx = y * sz + x;
                for c in 0..3 {
                    flat[c * sz * sz + idx] = (px[c] as f32 / 255.0 - MEAN[c]) / STD[c];
                }
            }
        }
        let array = ndarray::Array::from_shape_vec(ndarray::IxDyn(&[1, 3, sz, sz]), flat)
            .context("Failed to create depth input array")?;
        Tensor::from_array(array).context("Failed to create depth tensor")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stamp(ns: u64) -> SensorStamp {
        SensorStamp {
            source_sequence: 1,
            host_monotonic_ns: ns,
            wall_clock_unix_ms: 1_700_000_000_000,
            clock_domain: mvm_core::SensorClockDomain::KinectV1Depth,
            device_timestamp_raw: Some(1),
        }
    }

    fn metric(geometry: DepthGeometry, skew_ns: u64) -> Arc<MetricDepthFrame> {
        Arc::new(MetricDepthFrame {
            width: 2,
            height: 2,
            depth_mm: vec![1_000; 4],
            invalid_value: 0,
            stamp: stamp(1_000_000 + skew_ns),
            geometry,
            calibration_revision: "a".repeat(64),
            valid_ratio: 1.0,
            rgb_depth_skew_ns: skew_ns,
        })
    }

    fn calibration() -> CalibrationRef {
        CalibrationRef {
            schema: "kinect-v1-sensor-calibration/1.0".into(),
            camera_serial: "serial".into(),
            revision_sha256: "a".repeat(64),
        }
    }

    #[test]
    fn registered_metric_is_admitted_with_mm_and_provenance_intact() {
        let admitted = RegisteredMetricDepth::admit(
            2,
            2,
            &stamp(1_000_000),
            &DepthStatus::Paired,
            metric(DepthGeometry::RegisteredToRgb, 10_000_000),
            &calibration(),
            20_000_000,
        )
        .unwrap();
        assert_eq!(admitted.frame().depth_mm, vec![1_000; 4]);
        assert_eq!(admitted.calibration().revision_sha256, "a".repeat(64));
    }

    #[test]
    fn unregistered_metric_cannot_bind_to_rgb() {
        let error = RegisteredMetricDepth::admit(
            2,
            2,
            &stamp(1_000_000),
            &DepthStatus::Paired,
            metric(DepthGeometry::NativeDepthCamera, 10_000_000),
            &calibration(),
            20_000_000,
        )
        .unwrap_err();
        assert_eq!(error, MetricDepthAdmissionError::Unregistered);
    }

    #[test]
    fn stale_metric_is_rejected() {
        let error = RegisteredMetricDepth::admit(
            2,
            2,
            &stamp(1_000_000),
            &DepthStatus::Paired,
            metric(DepthGeometry::RegisteredToRgb, 20_000_001),
            &calibration(),
            20_000_000,
        )
        .unwrap_err();
        assert_eq!(error, MetricDepthAdmissionError::Stale);
    }

    // ── RelativeDepthMap sampling (pure) ─────────────────────────────────────

    /// 4×4 map with a horizontal gradient (col index as value): near on the right.
    fn grad_map() -> RelativeDepthMap {
        let (w, h) = (4usize, 4usize);
        let mut data = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                data[y * w + x] = x as f32; // 0,1,2,3 across
            }
        }
        RelativeDepthMap { data, w, h }
    }

    #[test]
    fn at_normalized_maps_corners() {
        let m = grad_map();
        assert_eq!(m.at_normalized(0.0, 0.0), 0.0);
        assert_eq!(m.at_normalized(0.99, 0.0), 3.0); // right column
        assert_eq!(m.at_normalized(0.99, 0.99), 3.0);
    }

    #[test]
    fn at_normalized_clamps_out_of_range() {
        let m = grad_map();
        assert_eq!(m.at_normalized(-1.0, -1.0), 0.0);
        assert_eq!(m.at_normalized(5.0, 5.0), 3.0);
    }

    #[test]
    fn median_in_bbox_returns_central_value() {
        let m = grad_map();
        // Box centred on the map → samples around the middle columns (1..3).
        let med = m
            .median_in_bbox(&BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            })
            .unwrap();
        assert!(
            (1.0..=2.0).contains(&med),
            "median {med} should be mid-range"
        );
    }

    #[test]
    fn median_in_bbox_rejects_degenerate() {
        let m = grad_map();
        assert!(
            m.median_in_bbox(&BBox {
                x1: 0.5,
                y1: 0.5,
                x2: 0.5,
                y2: 0.5
            })
            .is_none()
        );
    }

    #[test]
    fn median_in_bbox_right_box_is_nearer() {
        let m = grad_map();
        let left = m
            .median_in_bbox(&BBox {
                x1: 0.0,
                y1: 0.0,
                x2: 0.4,
                y2: 1.0,
            })
            .unwrap();
        let right = m
            .median_in_bbox(&BBox {
                x1: 0.6,
                y1: 0.0,
                x2: 1.0,
                y2: 1.0,
            })
            .unwrap();
        assert!(
            right > left,
            "right box (nearer) should have larger depth value"
        );
    }

    #[test]
    fn median_at_points_reads_the_sampled_cells() {
        // grad_map depth = x-column value (0..3). Points on the right columns →
        // larger median than points on the left, and it ignores everything else.
        let m = grad_map();
        let right = m
            .median_at_points([[0.9, 0.5], [0.85, 0.2], [0.95, 0.8]].into_iter())
            .unwrap();
        let left = m
            .median_at_points([[0.1, 0.5], [0.05, 0.2], [0.15, 0.8]].into_iter())
            .unwrap();
        assert!(
            right > left,
            "right points nearer: right={right} left={left}"
        );
    }

    #[test]
    fn median_at_points_empty_is_none() {
        let m = grad_map();
        assert!(m.median_at_points(std::iter::empty()).is_none());
    }

    // ── end-to-end model load (gated on file present) ──────────────────────

    #[test]
    fn model_loads_and_runs_when_present() {
        let model_path = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/depth_anything_v2_small.onnx"
        ));
        if !model_path.exists() {
            eprintln!("Skipping: {} not found", model_path.display());
            return;
        }
        let mut est = DepthEstimator::from_path(&model_path).unwrap();
        let img = DynamicImage::new_rgb8(640, 480);
        let dm = est.estimate(&img).unwrap();
        assert_eq!(dm.w, 518);
        assert_eq!(dm.h, 518);
        assert_eq!(dm.data.len(), 518 * 518);
        eprintln!(
            "depth map {}x{}, sample center = {:.3}",
            dm.w,
            dm.h,
            dm.at_normalized(0.5, 0.5)
        );
    }
}
