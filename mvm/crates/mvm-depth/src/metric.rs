//! Image-free metric geometry contracts. These are measurements, not semantic or
//! canonical facts; callers must provide an explicitly validated camera model.

pub const MAX_GEOMETRY_SAMPLES: usize = 4_096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceFrame {
    CameraOptical,
    LocalWorld,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetricPoint {
    pub xyz_m: [f32; 3],
    pub reference_frame: ReferenceFrame,
    pub calibration_revision: String,
    pub transform_revision: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryStatus {
    Measured,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeometryError {
    InvalidIntrinsics,
    InvalidTransform,
    InvalidPixel,
    InvalidDepth,
    DimensionsMismatch,
    CalibrationMismatch,
    ReferenceFrameMismatch,
    InsufficientSamples,
    DegenerateGeometry,
    InvalidConfig,
    InvalidOwnerAssociation,
}

impl std::fmt::Display for GeometryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for GeometryError {}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometryMeasurement {
    pub status: GeometryStatus,
    pub sample_count: usize,
    pub valid_ratio: f32,
    pub median_z_m: f32,
    pub median_absolute_deviation_m: f32,
    pub centroid_m: [f32; 3],
    pub aabb_min_m: [f32; 3],
    pub aabb_max_m: [f32; 3],
    pub nearest_surface_m: f32,
    pub farthest_surface_m: f32,
    pub boundary_contamination: f32,
    pub reference_frame: ReferenceFrame,
    pub calibration_revision: String,
}
