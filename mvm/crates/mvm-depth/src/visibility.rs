use crate::{GeometryError, RegisteredMetricDepth};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepthVisibility {
    Visible,
    OccupiedSurface,
    Occluded,
    Unknown,
    Stale,
    OutOfRange,
}

impl RegisteredMetricDepth {
    pub fn visibility_at(
        &self,
        x: u32,
        y: u32,
        target_depth_mm: Option<u16>,
        tolerance_mm: u16,
    ) -> Result<DepthVisibility, GeometryError> {
        let frame = self.frame();
        if x >= frame.width || y >= frame.height {
            return Ok(DepthVisibility::OutOfRange);
        }
        let observed = frame.depth_mm[y as usize * frame.width as usize + x as usize];
        if observed == frame.invalid_value {
            return Ok(DepthVisibility::Unknown);
        }
        let Some(target) = target_depth_mm else {
            return Ok(DepthVisibility::OccupiedSurface);
        };
        if target == frame.invalid_value {
            return Ok(DepthVisibility::Unknown);
        }
        let delta = i32::from(target) - i32::from(observed);
        if delta.unsigned_abs() <= u32::from(tolerance_mm) {
            Ok(DepthVisibility::Visible)
        } else if delta > 0 {
            Ok(DepthVisibility::Occluded)
        } else {
            Ok(DepthVisibility::OccupiedSurface)
        }
    }
}
