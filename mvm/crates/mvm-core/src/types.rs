//! MVM core types — bounding boxes, detections, person tracks.
//!
//! These types are shared across all MVM modules. They carry identity,
//! emotion, posture, and 3D workspace data with confidence scores.

/// A bounding box in normalised frame coordinates (0.0..1.0).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl BBox {
    pub fn area(&self) -> f32 {
        let w = (self.x2 - self.x1).max(0.0);
        let h = (self.y2 - self.y1).max(0.0);
        w * h
    }

    pub fn iou(&self, other: &BBox) -> f32 {
        let xi1 = self.x1.max(other.x1);
        let yi1 = self.y1.max(other.y1);
        let xi2 = self.x2.min(other.x2);
        let yi2 = self.y2.min(other.y2);
        let inter = ((xi2 - xi1).max(0.0)) * ((yi2 - yi1).max(0.0));
        let union = self.area() + other.area() - inter;
        if union < 1e-6 { 0.0 } else { inter / union }
    }
}

/// A single detection from the vision pipeline, with optional enrichment.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    // Identity
    pub identity: Option<String>,
    pub identity_confidence: Option<f32>,
    // Emotion
    pub emotion: Option<String>,
    pub emotion_confidence: Option<f32>,
    // Posture
    pub posture: Option<String>,
    pub posture_confidence: Option<f32>,
    // 3D Workspace
    pub distance: Option<f32>,
    pub position_x: Option<f32>,
    pub position_y: Option<f32>,
    pub position_z: Option<f32>,
    pub ws_confidence: Option<f32>,
    // Posture quality
    pub recon_error: Option<f32>,
    pub plausibility: Option<String>,
}

impl Detection {
    /// Convenience constructor: bbox-only detection, all enrichment None.
    pub fn bare(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            bbox: BBox { x1, y1, x2, y2 },
            identity: None, identity_confidence: None,
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }
    }
}

/// A tracked person — persistent across frames.
#[derive(Debug, Clone)]
pub struct PersonTrack {
    pub id: u32,
    pub bbox: BBox,
    pub age: u64,
    pub missed: u64,
    /// Best identity seen for this track (highest confidence).
    pub identity: Option<String>,
    pub identity_confidence: Option<f32>,
    /// Most recent emotion.
    pub emotion: Option<String>,
    pub emotion_confidence: Option<f32>,
    /// Most recent posture.
    pub posture: Option<String>,
    pub posture_confidence: Option<f32>,
    /// Most recent 3D position.
    pub distance: Option<f32>,
    pub position_x: Option<f32>,
    pub position_y: Option<f32>,
    pub position_z: Option<f32>,
    pub ws_confidence: Option<f32>,
    /// Posture quality.
    pub recon_error: Option<f32>,
    pub plausibility: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── BBox ──────────────────────────────────────────────

    #[test]
    fn bbox_area_unit_square() {
        let b = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert!((b.area() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bbox_area_half_frame() {
        let b = BBox { x1: 0.25, y1: 0.25, x2: 0.75, y2: 0.75 };
        assert!((b.area() - 0.25).abs() < 1e-6);
    }

    #[test]
    fn bbox_area_zero_width() {
        let b = BBox { x1: 0.5, y1: 0.0, x2: 0.5, y2: 1.0 };
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn bbox_area_inverted() {
        let b = BBox { x1: 0.8, y1: 0.8, x2: 0.2, y2: 0.2 };
        assert_eq!(b.area(), 0.0);
    }

    #[test]
    fn iou_identical() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let b = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert!((a.iou(&b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn iou_half_overlap() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 0.6, y2: 1.0 };
        let b = BBox { x1: 0.4, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert!((a.iou(&b) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn iou_no_overlap() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 0.4, y2: 1.0 };
        let b = BBox { x1: 0.6, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_one_inside_another() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let b = BBox { x1: 0.2, y1: 0.2, x2: 0.8, y2: 0.8 };
        assert!((a.iou(&b) - 0.36).abs() < 1e-5);
    }

    #[test]
    fn iou_edge_touching() {
        let a = BBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 1.0 };
        let b = BBox { x1: 0.5, y1: 0.0, x2: 1.0, y2: 1.0 };
        assert_eq!(a.iou(&b), 0.0);
    }

    #[test]
    fn iou_symmetric() {
        let a = BBox { x1: 0.1, y1: 0.2, x2: 0.7, y2: 0.9 };
        let b = BBox { x1: 0.3, y1: 0.1, x2: 0.9, y2: 0.8 };
        assert_eq!(a.iou(&b), b.iou(&a));
    }
}
