use crate::{GeometryError, MetricPoint, ReferenceFrame, RegisteredMetricDepth};
use mvm_core::RegisteredDepthOwners;

#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    pub width: u32,
    pub height: u32,
    pub fx_px: f32,
    pub fy_px: f32,
    pub cx_px: f32,
    pub cy_px: f32,
    pub calibration_revision: String,
    pub reference_frame: ReferenceFrame,
}

impl CameraIntrinsics {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        fx_px: f32,
        fy_px: f32,
        cx_px: f32,
        cy_px: f32,
        calibration_revision: String,
        reference_frame: ReferenceFrame,
    ) -> Result<Self, GeometryError> {
        let value = Self {
            width,
            height,
            fx_px,
            fy_px,
            cx_px,
            cy_px,
            calibration_revision,
            reference_frame,
        };
        value.validate()?;
        Ok(value)
    }

    pub fn validate(&self) -> Result<(), GeometryError> {
        if self.width == 0
            || self.height == 0
            || !self.fx_px.is_finite()
            || !self.fy_px.is_finite()
            || self.fx_px <= 0.0
            || self.fy_px <= 0.0
            || !self.cx_px.is_finite()
            || !self.cy_px.is_finite()
            || self.calibration_revision.len() != 64
            || !self
                .calibration_revision
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            || self.reference_frame != ReferenceFrame::CameraOptical
        {
            return Err(GeometryError::InvalidIntrinsics);
        }
        Ok(())
    }

    pub fn unproject(&self, u: f32, v: f32, depth_mm: u16) -> Result<MetricPoint, GeometryError> {
        self.validate()?;
        if !u.is_finite()
            || !v.is_finite()
            || !(0.0..self.width as f32).contains(&u)
            || !(0.0..self.height as f32).contains(&v)
        {
            return Err(GeometryError::InvalidPixel);
        }
        if depth_mm == 0 {
            return Err(GeometryError::InvalidDepth);
        }
        let z = depth_mm as f32 / 1_000.0;
        Ok(MetricPoint {
            xyz_m: [
                (u - self.cx_px) * z / self.fx_px,
                (v - self.cy_px) * z / self.fy_px,
                z,
            ],
            reference_frame: self.reference_frame,
            calibration_revision: self.calibration_revision.clone(),
            transform_revision: None,
        })
    }

    pub fn project(&self, point: &MetricPoint) -> Result<[f32; 2], GeometryError> {
        self.validate()?;
        if point.reference_frame != ReferenceFrame::CameraOptical {
            return Err(GeometryError::ReferenceFrameMismatch);
        }
        if point.calibration_revision != self.calibration_revision {
            return Err(GeometryError::CalibrationMismatch);
        }
        let [x, y, z] = point.xyz_m;
        if !x.is_finite() || !y.is_finite() || !z.is_finite() || z <= 0.0 {
            return Err(GeometryError::InvalidDepth);
        }
        Ok([
            self.fx_px * x / z + self.cx_px,
            self.fy_px * y / z + self.cy_px,
        ])
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MountTransform {
    rotation: [[f32; 3]; 3],
    translation_m: [f32; 3],
    revision: String,
}

impl MountTransform {
    pub fn new(
        rotation: [[f32; 3]; 3],
        translation_m: [f32; 3],
        revision: impl Into<String>,
    ) -> Result<Self, GeometryError> {
        let value = Self {
            rotation,
            translation_m,
            revision: revision.into(),
        };
        if value.revision.trim().is_empty()
            || value.rotation.iter().flatten().any(|v| !v.is_finite())
            || value.translation_m.iter().any(|v| !v.is_finite())
        {
            return Err(GeometryError::InvalidTransform);
        }
        for row in &value.rotation {
            let norm = row.iter().map(|v| v * v).sum::<f32>();
            if (norm - 1.0).abs() > 1e-3 {
                return Err(GeometryError::InvalidTransform);
            }
        }
        for a in 0..3 {
            for b in (a + 1)..3 {
                let dot = (0..3)
                    .map(|i| value.rotation[a][i] * value.rotation[b][i])
                    .sum::<f32>();
                if dot.abs() > 1e-3 {
                    return Err(GeometryError::InvalidTransform);
                }
            }
        }
        let determinant = value.rotation[0][0]
            * (value.rotation[1][1] * value.rotation[2][2]
                - value.rotation[1][2] * value.rotation[2][1])
            - value.rotation[0][1]
                * (value.rotation[1][0] * value.rotation[2][2]
                    - value.rotation[1][2] * value.rotation[2][0])
            + value.rotation[0][2]
                * (value.rotation[1][0] * value.rotation[2][1]
                    - value.rotation[1][1] * value.rotation[2][0]);
        if (determinant - 1.0).abs() > 1e-3 {
            return Err(GeometryError::InvalidTransform);
        }
        Ok(value)
    }

    pub fn camera_to_local_world(&self, point: &MetricPoint) -> Result<MetricPoint, GeometryError> {
        if point.reference_frame != ReferenceFrame::CameraOptical {
            return Err(GeometryError::ReferenceFrameMismatch);
        }
        if point.xyz_m.iter().any(|value| !value.is_finite()) {
            return Err(GeometryError::InvalidDepth);
        }
        let mut xyz = self.translation_m;
        for (row, value) in xyz.iter_mut().enumerate() {
            *value += (0..3)
                .map(|col| self.rotation[row][col] * point.xyz_m[col])
                .sum::<f32>();
        }
        Ok(MetricPoint {
            xyz_m: xyz,
            reference_frame: ReferenceFrame::LocalWorld,
            calibration_revision: point.calibration_revision.clone(),
            transform_revision: Some(self.revision.clone()),
        })
    }
}

impl RegisteredMetricDepth {
    pub fn unproject_pixel(
        &self,
        x: u32,
        y: u32,
        intrinsics: &CameraIntrinsics,
    ) -> Result<Option<MetricPoint>, GeometryError> {
        intrinsics.validate()?;
        let frame = self.frame();
        if frame.width != intrinsics.width || frame.height != intrinsics.height {
            return Err(GeometryError::DimensionsMismatch);
        }
        if frame.calibration_revision != intrinsics.calibration_revision {
            return Err(GeometryError::CalibrationMismatch);
        }
        if x >= frame.width || y >= frame.height {
            return Err(GeometryError::InvalidPixel);
        }
        let index = y as usize * frame.width as usize + x as usize;
        let depth_mm = frame.depth_mm[index];
        if depth_mm == frame.invalid_value {
            return Ok(None);
        }
        intrinsics.unproject(x as f32, y as f32, depth_mm).map(Some)
    }

    /// Exact Kinect-style unprojection for a registered RGB-grid sample. The
    /// depth value comes from the registered map, while the ray comes from the
    /// native depth pixel retained by registration's nearest-depth z-buffer.
    pub fn unproject_registered_owner(
        &self,
        registered_x: u32,
        registered_y: u32,
        owners: &RegisteredDepthOwners,
        native_intrinsics: &CameraIntrinsics,
    ) -> Result<Option<MetricPoint>, GeometryError> {
        owners
            .validate()
            .map_err(|_| GeometryError::InvalidOwnerAssociation)?;
        native_intrinsics.validate()?;
        let frame = self.frame();
        if frame.width != owners.registered_width
            || frame.height != owners.registered_height
            || owners.native_width != native_intrinsics.width
            || owners.native_height != native_intrinsics.height
        {
            return Err(GeometryError::DimensionsMismatch);
        }
        if frame.calibration_revision != owners.calibration_revision
            || frame.calibration_revision != native_intrinsics.calibration_revision
        {
            return Err(GeometryError::CalibrationMismatch);
        }
        if registered_x >= frame.width || registered_y >= frame.height {
            return Err(GeometryError::InvalidPixel);
        }
        let index = registered_y as usize * frame.width as usize + registered_x as usize;
        let depth_mm = frame.depth_mm[index];
        if depth_mm == frame.invalid_value {
            return Ok(None);
        }
        let [native_x, native_y] = owners
            .native_pixel(registered_x, registered_y)
            .ok_or(GeometryError::InvalidOwnerAssociation)?;
        native_intrinsics
            .unproject(native_x as f32, native_y as f32, depth_mm)
            .map(Some)
    }
}
