//! Instance-mask decode + geometry (phase 10.4, pure arithmetic).
//!
//! YOLO-seg emits, per instance, 32 mask **coefficients**; the model also emits a
//! shared bank of 32 **prototype** masks at 1/4 input resolution (160² for a 640
//! input). The instance mask is `sigmoid(Σ coeff_k · proto_k)`, then cropped to the
//! detection bbox (the ultralytics convention). This module does that assembly and
//! stores the result as a small fixed-resolution occupancy grid over the **original
//! frame** in normalized coords — so mask-IoU (tracker), point-containment (fusion
//! contact) and median-in-mask (depth) are all cheap grid ops, decoupled from ONNX
//! and unit-tested on synthetic prototypes.

use crate::Letterbox;
use mvm_core::BBox;

/// Occupancy-grid resolution over the original frame (each axis). 64² = 4096 cells
/// — enough to separate overlapping instances and read mask-median depth, small
/// enough to keep every op O(1) per object.
pub const MASK_RES: usize = 64;

/// A low-resolution instance mask over the **original frame**, normalized coords.
/// `bits[gy*MASK_RES + gx]` is 1 when the cell centred at
/// `((gx+0.5)/RES, (gy+0.5)/RES)` lies inside the instance.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceMask {
    bits: Vec<u8>,
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl InstanceMask {
    /// An all-empty mask (no cells set).
    pub fn empty() -> Self {
        Self { bits: vec![0u8; MASK_RES * MASK_RES] }
    }

    /// Assemble an instance mask from the prototype bank + this instance's
    /// coefficients, cropped to `bbox`.
    ///
    /// - `proto`: row-major `[n_proto, ph, pw]` prototype logits (over the square
    ///   letterboxed canvas of side `canvas`, at proto resolution `pw×ph`).
    /// - `coeffs`: this instance's `n_proto` coefficients.
    /// - `bbox`: normalized detection box in the original frame — cells outside it
    ///   are never set (ultralytics crops the mask to the box).
    /// - `lb` / `orig_w` / `orig_h` / `canvas`: the letterbox mapping original↔canvas.
    ///
    /// A cell is set when the assembled logit at its back-projected proto location,
    /// passed through the sigmoid, clears 0.5.
    #[allow(clippy::too_many_arguments)]
    pub fn assemble(
        proto: &[f32],
        n_proto: usize,
        ph: usize,
        pw: usize,
        coeffs: &[f32],
        bbox: &BBox,
        lb: Letterbox,
        orig_w: u32,
        orig_h: u32,
        canvas: u32,
    ) -> Self {
        debug_assert_eq!(proto.len(), n_proto * ph * pw);
        debug_assert_eq!(coeffs.len(), n_proto);
        let plane = ph * pw;
        let mut bits = vec![0u8; MASK_RES * MASK_RES];
        let (ow, oh, cv) = (orig_w as f32, orig_h as f32, canvas as f32);
        for gy in 0..MASK_RES {
            let ny = (gy as f32 + 0.5) / MASK_RES as f32;
            if ny < bbox.y1 || ny > bbox.y2 {
                continue;
            }
            for gx in 0..MASK_RES {
                let nx = (gx as f32 + 0.5) / MASK_RES as f32;
                if nx < bbox.x1 || nx > bbox.x2 {
                    continue;
                }
                // original-norm → canvas px → proto cell.
                let cx = nx * ow * lb.scale + lb.pad_x;
                let cy = ny * oh * lb.scale + lb.pad_y;
                let px = ((cx * pw as f32 / cv) as isize).clamp(0, pw as isize - 1) as usize;
                let py = ((cy * ph as f32 / cv) as isize).clamp(0, ph as isize - 1) as usize;
                let off = py * pw + px;
                let mut logit = 0.0f32;
                for k in 0..n_proto {
                    logit += coeffs[k] * proto[k * plane + off];
                }
                if sigmoid(logit) >= 0.5 {
                    bits[gy * MASK_RES + gx] = 1;
                }
            }
        }
        Self { bits }
    }

    /// Number of set cells.
    pub fn count(&self) -> usize {
        self.bits.iter().filter(|&&b| b != 0).count()
    }

    pub fn is_empty(&self) -> bool {
        self.count() == 0
    }

    /// Fraction of the whole frame the instance covers (set cells / all cells).
    pub fn frame_coverage(&self) -> f32 {
        self.count() as f32 / (MASK_RES * MASK_RES) as f32
    }

    /// Whether the normalized point falls in a set cell.
    pub fn contains_norm(&self, p: [f32; 2]) -> bool {
        if !(0.0..1.0).contains(&p[0]) || !(0.0..1.0).contains(&p[1]) {
            return false;
        }
        let gx = (p[0] * MASK_RES as f32) as usize;
        let gy = (p[1] * MASK_RES as f32) as usize;
        let (gx, gy) = (gx.min(MASK_RES - 1), gy.min(MASK_RES - 1));
        self.bits[gy * MASK_RES + gx] != 0
    }

    /// Intersection-over-union of two instance masks (grid cells). 0 when either is
    /// empty. The instance-separation signal for the object tracker (phase 10.5).
    pub fn iou(&self, other: &InstanceMask) -> f32 {
        let mut inter = 0u32;
        let mut union = 0u32;
        for (a, b) in self.bits.iter().zip(&other.bits) {
            let (a, b) = (*a != 0, *b != 0);
            if a && b {
                inter += 1;
            }
            if a || b {
                union += 1;
            }
        }
        if union == 0 { 0.0 } else { inter as f32 / union as f32 }
    }

    /// Normalized centres of the set cells — the sample points for
    /// `DepthMap::median_at_points` (mask-median depth, phase 10.4).
    pub fn set_points(&self) -> impl Iterator<Item = [f32; 2]> + '_ {
        (0..MASK_RES * MASK_RES).filter_map(move |i| {
            if self.bits[i] != 0 {
                let gx = i % MASK_RES;
                let gy = i / MASK_RES;
                Some([(gx as f32 + 0.5) / MASK_RES as f32, (gy as f32 + 0.5) / MASK_RES as f32])
            } else {
                None
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No letterbox: square frame → identity canvas mapping (scale 1, no pad) with
    /// canvas == proto resolution, so proto cell = original grid cell directly.
    fn identity_lb() -> Letterbox {
        Letterbox { scale: 1.0, pad_x: 0.0, pad_y: 0.0 }
    }

    /// One prototype, `res×res`, filled so the left half is strongly positive
    /// (inside) and the right half strongly negative (outside). coeff = 1.
    fn left_half_proto(res: usize) -> Vec<f32> {
        let mut p = vec![0.0f32; res * res];
        for y in 0..res {
            for x in 0..res {
                p[y * res + x] = if x < res / 2 { 10.0 } else { -10.0 };
            }
        }
        p
    }

    #[test]
    fn assemble_left_half_sets_left_cells_only() {
        let res = MASK_RES;
        let proto = left_half_proto(res);
        let full = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let m = InstanceMask::assemble(
            &proto, 1, res, res, &[1.0], &full,
            identity_lb(), res as u32, res as u32, res as u32,
        );
        assert!(m.contains_norm([0.25, 0.5]), "left half is inside");
        assert!(!m.contains_norm([0.75, 0.5]), "right half is outside");
        // ~half the frame.
        assert!((m.frame_coverage() - 0.5).abs() < 0.02, "cov {}", m.frame_coverage());
    }

    #[test]
    fn assemble_crops_to_bbox() {
        let res = MASK_RES;
        let proto = left_half_proto(res);
        // bbox covers only the top-left quadrant → even the "inside" left half is
        // clipped to y<0.5.
        let bbox = BBox { x1: 0.0, y1: 0.0, x2: 0.5, y2: 0.5 };
        let m = InstanceMask::assemble(
            &proto, 1, res, res, &[1.0], &bbox,
            identity_lb(), res as u32, res as u32, res as u32,
        );
        assert!(m.contains_norm([0.25, 0.25]), "inside bbox + inside proto");
        assert!(!m.contains_norm([0.25, 0.75]), "below bbox → cropped out");
    }

    #[test]
    fn negative_coeff_flips_the_mask() {
        let res = MASK_RES;
        let proto = left_half_proto(res);
        let full = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let m = InstanceMask::assemble(
            &proto, 1, res, res, &[-1.0], &full,
            identity_lb(), res as u32, res as u32, res as u32,
        );
        assert!(!m.contains_norm([0.25, 0.5]), "coeff sign flip → left now outside");
        assert!(m.contains_norm([0.75, 0.5]), "right now inside");
    }

    #[test]
    fn iou_identical_is_one_disjoint_is_zero() {
        let res = MASK_RES;
        let proto = left_half_proto(res);
        let full = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let left = InstanceMask::assemble(&proto, 1, res, res, &[1.0], &full, identity_lb(), res as u32, res as u32, res as u32);
        let left2 = left.clone();
        let right = InstanceMask::assemble(&proto, 1, res, res, &[-1.0], &full, identity_lb(), res as u32, res as u32, res as u32);
        assert!((left.iou(&left2) - 1.0).abs() < 1e-6, "identical → 1");
        assert!(left.iou(&right) < 1e-6, "disjoint halves → 0");
    }

    #[test]
    fn iou_half_overlap() {
        // Two boxes-as-masks: [0,0.6] and [0.4,1.0] on x, full y. Overlap 0.2 wide,
        // union 1.0 wide → IoU = 0.2/1.0 = 0.2 (within grid quantization).
        let res = MASK_RES;
        let uniform = vec![10.0f32; res * res]; // everything "inside"
        let a = InstanceMask::assemble(&uniform, 1, res, res, &[1.0], &BBox{x1:0.0,y1:0.0,x2:0.6,y2:1.0}, identity_lb(), res as u32, res as u32, res as u32);
        let b = InstanceMask::assemble(&uniform, 1, res, res, &[1.0], &BBox{x1:0.4,y1:0.0,x2:1.0,y2:1.0}, identity_lb(), res as u32, res as u32, res as u32);
        let iou = a.iou(&b);
        assert!((iou - 0.2).abs() < 0.05, "expected ~0.2, got {iou}");
    }

    #[test]
    fn set_points_lie_in_mask() {
        let res = MASK_RES;
        let proto = left_half_proto(res);
        let full = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let m = InstanceMask::assemble(&proto, 1, res, res, &[1.0], &full, identity_lb(), res as u32, res as u32, res as u32);
        let pts: Vec<_> = m.set_points().collect();
        assert_eq!(pts.len(), m.count());
        assert!(pts.iter().all(|p| m.contains_norm(*p)), "every set point is inside");
        assert!(pts.iter().all(|p| p[0] < 0.5), "left-half mask → all points x<0.5");
    }

    #[test]
    fn letterbox_pad_maps_correctly() {
        // Wide 2:1 frame onto a square canvas → vertical letterbox (pad_y > 0).
        // A uniform "all inside" proto still fills the whole bbox regardless of pad,
        // proving the mapping stays in-bounds (no panic, coverage sensible).
        let res = MASK_RES;
        let uniform = vec![10.0f32; res * res];
        let lb = crate::compute_letterbox(200, 100, res as u32);
        let full = BBox { x1: 0.0, y1: 0.0, x2: 1.0, y2: 1.0 };
        let m = InstanceMask::assemble(&uniform, 1, res, res, &[1.0], &full, lb, 200, 100, res as u32);
        assert!(m.frame_coverage() > 0.9, "uniform-inside proto fills the box, cov {}", m.frame_coverage());
    }
}
