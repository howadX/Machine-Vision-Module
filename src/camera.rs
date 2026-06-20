//! Camera capture via Python bridge (stdin pipe).
//!
//! Architecture: Python `sight.py --stream` writes bbox/landmark JSON lines
//! to stdout. Rust sightd reads them and runs the tracker + state file.
//!
//! v0.1: placeholder — returns empty detections. The loop infrastructure
//! (tracker, state, timing) works without real camera data.

use anyhow::Result;

#[allow(dead_code)]
pub struct Camera {
    width: u32,
    height: u32,
}

#[allow(dead_code)]
impl Camera {
    /// Open the camera bridge. v0.1: no-op placeholder.
    pub fn open() -> Result<Self> {
        eprintln!("  camera: bridge mode (no real capture in v0.1)");
        Ok(Self { width: 640, height: 480 })
    }

    pub fn width(&self) -> u32 { self.width }
    pub fn height(&self) -> u32 { self.height }

    /// v0.1: returns no detections. When bridge is ready, reads JSON from
    /// Python subprocess stdin and parses bboxes.
    pub fn read(&self) -> Result<Vec<crate::tracker::BBox>> {
        // Placeholder: no real detections yet
        Ok(vec![])
    }
}
