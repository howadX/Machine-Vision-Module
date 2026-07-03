//! Instance-mask occupancy grid (phase 10.4 core, pure arithmetic).
//!
//! The grid container lives in `mvm-core` (phase 11.A1) so the mask can ride
//! `ObjectState` into fusion (`InstanceMask::contains_norm` = mask-contact)
//! without `mvm-core` learning anything about ONNX or letterboxing. The
//! YOLO-seg **assembly** (prototype bank × coefficients → grid) stays in
//! `mvm-scene::mask::assemble_mask`, which needs the letterbox mapping.

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

impl InstanceMask {
    /// An all-empty mask (no cells set).
    pub fn empty() -> Self {
        Self { bits: vec![0u8; MASK_RES * MASK_RES] }
    }

    /// Build a mask from a raw `MASK_RES²` occupancy grid (row-major, non-zero =
    /// inside). The constructor the out-of-crate assembler uses.
    pub fn from_bits(bits: Vec<u8>) -> Self {
        assert_eq!(bits.len(), MASK_RES * MASK_RES, "mask grid must be MASK_RES²");
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

    /// `contains_norm` with a dilation of `cells` grid cells: true when any set
    /// cell lies within a `(2·cells+1)²` neighbourhood of the point's cell. The
    /// mask-contact analogue of the bbox `CONTACT_MARGIN` — absorbs landmark
    /// jitter and grid quantization without re-admitting the whole bbox
    /// background (1 cell ≈ 0.016 normalized).
    pub fn contains_norm_within(&self, p: [f32; 2], cells: usize) -> bool {
        if !(0.0..1.0).contains(&p[0]) || !(0.0..1.0).contains(&p[1]) {
            return false;
        }
        let gx = ((p[0] * MASK_RES as f32) as usize).min(MASK_RES - 1) as isize;
        let gy = ((p[1] * MASK_RES as f32) as usize).min(MASK_RES - 1) as isize;
        let r = cells as isize;
        for dy in -r..=r {
            for dx in -r..=r {
                let (x, y) = (gx + dx, gy + dy);
                if x < 0 || y < 0 || x >= MASK_RES as isize || y >= MASK_RES as isize {
                    continue;
                }
                if self.bits[y as usize * MASK_RES + x as usize] != 0 {
                    return true;
                }
            }
        }
        false
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

    /// A mask with exactly one grid cell set.
    fn one_cell(gx: usize, gy: usize) -> InstanceMask {
        let mut bits = vec![0u8; MASK_RES * MASK_RES];
        bits[gy * MASK_RES + gx] = 1;
        InstanceMask::from_bits(bits)
    }

    #[test]
    fn from_bits_roundtrips_contains() {
        let m = one_cell(10, 20);
        assert_eq!(m.count(), 1);
        let p = [(10.0 + 0.5) / MASK_RES as f32, (20.0 + 0.5) / MASK_RES as f32];
        assert!(m.contains_norm(p));
        assert!(!m.contains_norm([0.9, 0.9]));
    }

    #[test]
    #[should_panic(expected = "MASK_RES")]
    fn from_bits_rejects_wrong_len() {
        let _ = InstanceMask::from_bits(vec![0u8; 3]);
    }

    #[test]
    fn dilated_contains_reaches_neighbours_only() {
        let m = one_cell(32, 32);
        // Two cells away from the set cell.
        let p = [(34.0 + 0.5) / MASK_RES as f32, (32.0 + 0.5) / MASK_RES as f32];
        assert!(!m.contains_norm(p), "exact containment misses");
        assert!(!m.contains_norm_within(p, 1), "1-cell dilation still misses");
        assert!(m.contains_norm_within(p, 2), "2-cell dilation reaches");
        assert!(!m.contains_norm_within([0.9, 0.9], 2), "far point never contained");
    }

    #[test]
    fn dilated_contains_rejects_out_of_frame() {
        let m = one_cell(0, 0);
        assert!(!m.contains_norm_within([-0.01, 0.0], 2));
        assert!(!m.contains_norm_within([0.0, 1.01], 2));
    }
}
