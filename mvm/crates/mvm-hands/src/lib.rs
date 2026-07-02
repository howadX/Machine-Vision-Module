//! mvm-hands — hand landmarks via MediaPipe handpose (OpenCV Zoo) ONNX.
//!
//! Phase 8.E.4 (completes the G₁ Human group → HumanAgent). **Body-primary**: the
//! hand ROI is derived from a body's BlazePose wrist + finger landmarks (the
//! MediaPipe-Holistic approach), so NO palm detector / anchors are needed — hands
//! are children of the body that already drives the chain. One model:
//! `handpose_estimation_mediapipe` (input NHWC [1,224,224,3]; outputs 21×3 px
//! landmarks, handedness, a presence score that gates bad ROIs, world landmarks).
//!
//! ROIs are axis-aligned squares here (no rotation): the presence score filters
//! poorly-oriented crops, and a person at a desk holds hands roughly upright.
//! Rotation-from-pose is a future refinement if recall proves too low.

use anyhow::Context;
use image::RgbImage;
use mvm_core::BBox;
use ort::{session::Session, value::Tensor};

const IN: usize = 224;
/// 21 MediaPipe hand landmarks.
pub const N_HAND_LM: usize = 21;

/// Which hand, from the body side the ROI came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandSide {
    Left,
    Right,
}

/// A detected hand: 21 landmarks in original-image pixels (x, y, relative z).
#[derive(Debug, Clone)]
pub struct Hand {
    pub landmarks_px: [[f32; 3]; N_HAND_LM],
    /// Model handedness output (raw, ~[0,1]); `side` is the authoritative source
    /// (it comes from which body wrist drove the ROI).
    pub handedness: f32,
    /// Presence/confidence score in [0,1].
    pub score: f32,
    pub side: HandSide,
}

impl Hand {
    /// Normalized [0,1] bbox over the 21 landmarks, given the frame dimensions.
    pub fn bbox_norm(&self, fw: f32, fh: f32) -> BBox {
        let mut x1 = f32::MAX;
        let mut y1 = f32::MAX;
        let mut x2 = f32::MIN;
        let mut y2 = f32::MIN;
        for p in &self.landmarks_px {
            x1 = x1.min(p[0]);
            y1 = y1.min(p[1]);
            x2 = x2.max(p[0]);
            y2 = y2.max(p[1]);
        }
        BBox {
            x1: (x1 / fw).clamp(0.0, 1.0),
            y1: (y1 / fh).clamp(0.0, 1.0),
            x2: (x2 / fw).clamp(0.0, 1.0),
            y2: (y2 / fh).clamp(0.0, 1.0),
        }
    }

    /// The wrist landmark (index 0) in pixels — the natural hand anchor point.
    pub fn wrist_px(&self) -> (f32, f32) {
        (self.landmarks_px[0][0], self.landmarks_px[0][1])
    }
}

/// A square hand ROI in original-image pixels, with the body side it came from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HandRoi {
    pub cx: f32,
    pub cy: f32,
    pub side: f32,
    pub which: HandSide,
    /// Wrist visibility that produced this ROI [0,1] (phase 10.2). Below the
    /// classic 0.5 gate the resulting hand's score is downweighted by
    /// `vis / 0.5` instead of the ROI being skipped outright — recovers desk /
    /// frame-edge hands without fabricating confident ones.
    pub vis: f32,
}

// ── Hand-gesture classification (phase 10.2, pure arithmetic) ────────────────

/// A coarse static hand gesture from the 21 landmarks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandGesture {
    /// All four fingers curled.
    Fist,
    /// Three or more fingers extended.
    Open,
    /// Only the index finger extended.
    Point,
    /// Thumb and index tips touching while the remaining fingers stay open.
    Pinch,
}

impl HandGesture {
    pub fn as_str(self) -> &'static str {
        match self {
            HandGesture::Fist => "fist",
            HandGesture::Open => "open",
            HandGesture::Point => "point",
            HandGesture::Pinch => "pinch",
        }
    }
}

/// Classify a static gesture from the 21 hand landmarks (pixels; z ignored).
///
/// Scale-free: every threshold is relative to the palm length (wrist → middle
/// MCP). A finger counts as *extended* when its tip sits farther from the wrist
/// than its PIP by ≥15% — curled fingers fold the tip back toward the palm.
/// Rules (checked in order, first match wins):
/// - **pinch**: thumb-tip↔index-tip < 0.3·palm while ≥2 of middle/ring/pinky
///   are extended (distinguishes it from a fist, where the thumb rests on the
///   curled index);
/// - **fist**: no finger extended;
/// - **point**: index extended, middle/ring/pinky all curled;
/// - **open**: ≥3 fingers extended.
/// `None` when the shape matches nothing cleanly or the palm is degenerate.
pub fn classify_gesture(lm: &[[f32; 3]; N_HAND_LM]) -> Option<HandGesture> {
    let p = |i: usize| (lm[i][0], lm[i][1]);
    let wrist = p(0);
    let palm = dist(wrist, p(9));
    if palm < 1e-3 {
        return None;
    }
    // (PIP, TIP) per finger: index, middle, ring, pinky.
    let extended =
        |pip: usize, tip: usize| dist(wrist, p(tip)) > dist(wrist, p(pip)) * 1.15;
    let index = extended(6, 8);
    let rest = [extended(10, 12), extended(14, 16), extended(18, 20)];
    let rest_ext = rest.iter().filter(|&&e| e).count();
    let n_ext = rest_ext + index as usize;

    if dist(p(4), p(8)) < palm * 0.3 && rest_ext >= 2 {
        return Some(HandGesture::Pinch);
    }
    if n_ext == 0 {
        return Some(HandGesture::Fist);
    }
    if index && rest_ext == 0 {
        return Some(HandGesture::Point);
    }
    if n_ext >= 3 {
        return Some(HandGesture::Open);
    }
    None
}

// BlazePose-33 indices for hands.
const L_WRIST: usize = 15;
const R_WRIST: usize = 16;
const L_PINKY: usize = 17;
const R_PINKY: usize = 18;
const L_INDEX: usize = 19;
const R_INDEX: usize = 20;
const L_THUMB: usize = 21;
const R_THUMB: usize = 22;

fn dist(a: (f32, f32), b: (f32, f32)) -> f32 {
    let (dx, dy) = (a.0 - b.0, a.1 - b.1);
    (dx * dx + dy * dy).sqrt()
}

/// Derive axis-aligned hand ROIs from a body's BlazePose landmarks (pixels).
///
/// For each hand, the wrist + finger tips bound the hand; we centre a square just
/// past the wrist toward the fingers and pad it (×2.4) so the whole hand fits.
/// Hands whose wrist visibility is below `vis_min` are skipped (occluded / behind);
/// the ROI records the visibility so a low-vis hand (phase 10.2 relaxation:
/// callers may pass a floor below the classic 0.5) gets its score downweighted
/// downstream instead of silently looking fully trusted.
pub fn hand_rois_from_pose(lm: &[[f32; 4]; 33], vis_min: f32) -> Vec<HandRoi> {
    let mut rois = Vec::new();
    for (wrist, pinky, index, thumb, which) in [
        (L_WRIST, L_PINKY, L_INDEX, L_THUMB, HandSide::Left),
        (R_WRIST, R_PINKY, R_INDEX, R_THUMB, HandSide::Right),
    ] {
        if lm[wrist][3] < vis_min {
            continue;
        }
        let w = (lm[wrist][0], lm[wrist][1]);
        let i = (lm[index][0], lm[index][1]);
        let p = (lm[pinky][0], lm[pinky][1]);
        let t = (lm[thumb][0], lm[thumb][1]);
        // Direction wrist → finger centroid; centre the ROI a bit past the knuckles.
        let fc = ((i.0 + p.0) * 0.5, (i.1 + p.1) * 0.5);
        let center = (w.0 + (fc.0 - w.0) * 1.1, w.1 + (fc.1 - w.1) * 1.1);
        let span = dist(w, i).max(dist(w, p)).max(dist(w, t)).max(dist(w, fc));
        let side = (span * 2.4).max(24.0);
        rois.push(HandRoi { cx: center.0, cy: center.1, side, which, vis: lm[wrist][3] });
    }
    rois
}

/// Bilinear sample of an RgbImage at fractional (x, y); out-of-bounds → black.
fn sample_bilinear(img: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let (w, h) = (img.width() as i32, img.height() as i32);
    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let mut out = [0f32; 3];
    for (wy, yy) in [(1.0 - fy, y0), (fy, y0 + 1)] {
        for (wx, xx) in [(1.0 - fx, x0), (fx, x0 + 1)] {
            let wgt = wx * wy;
            if wgt <= 0.0 || xx < 0 || yy < 0 || xx >= w || yy >= h {
                continue;
            }
            let px = img.get_pixel(xx as u32, yy as u32);
            out[0] += wgt * px[0] as f32;
            out[1] += wgt * px[1] as f32;
            out[2] += wgt * px[2] as f32;
        }
    }
    out
}

/// MediaPipe handpose ONNX landmarker.
pub struct HandLandmarker {
    session: Session,
    conf_threshold: f32,
}

impl HandLandmarker {
    pub fn from_path(path: &std::path::Path) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_file(path)
            .with_context(|| format!("Failed to load handpose from {}", path.display()))?;
        Ok(Self { session, conf_threshold: 0.5 })
    }

    #[cfg(feature = "embed-model")]
    pub fn from_bytes(bytes: &[u8]) -> anyhow::Result<Self> {
        let session = Session::builder()?
            .commit_from_memory(bytes)
            .context("Failed to load handpose from embedded bytes")?;
        Ok(Self { session, conf_threshold: 0.5 })
    }

    pub fn set_conf_threshold(&mut self, t: f32) {
        self.conf_threshold = t;
    }

    /// Run handpose on a body's hand ROIs; returns the hands that pass the presence
    /// gate, with landmarks mapped back to original-image pixels.
    pub fn infer_rois(&mut self, img: &RgbImage, rois: &[HandRoi]) -> Vec<Hand> {
        rois.iter().filter_map(|r| self.infer_square(img, r).ok().flatten()).collect()
    }

    /// Run handpose on one axis-aligned square ROI.
    pub fn infer_square(&mut self, img: &RgbImage, roi: &HandRoi) -> anyhow::Result<Option<Hand>> {
        if roi.side < 2.0 {
            return Ok(None);
        }
        let x1 = roi.cx - roi.side / 2.0;
        let y1 = roi.cy - roi.side / 2.0;
        let s = roi.side / IN as f32; // original px per output px
        // NHWC, normalized to [0,1].
        let mut data = vec![0f32; IN * IN * 3];
        for dy in 0..IN {
            let oy = y1 + (dy as f32 + 0.5) * s;
            for dx in 0..IN {
                let ox = x1 + (dx as f32 + 0.5) * s;
                let rgb = sample_bilinear(img, ox, oy);
                let o = (dy * IN + dx) * 3;
                data[o] = rgb[0] / 255.0;
                data[o + 1] = rgb[1] / 255.0;
                data[o + 2] = rgb[2] / 255.0;
            }
        }
        let input = Tensor::from_array(([1usize, IN, IN, 3], data))?;
        let outputs = self.session.run(ort::inputs!["input_1" => input])?;
        let score = outputs["Identity_2"].try_extract_tensor::<f32>()?.1[0];
        if score < self.conf_threshold {
            return Ok(None);
        }
        let handedness = outputs["Identity_1"].try_extract_tensor::<f32>()?.1[0];
        let lm = outputs["Identity"].try_extract_tensor::<f32>()?.1; // [63] = 21*3, in 224 px

        let mut landmarks_px = [[0f32; 3]; N_HAND_LM];
        for k in 0..N_HAND_LM {
            let b = k * 3;
            landmarks_px[k][0] = x1 + lm[b] * s;
            landmarks_px[k][1] = y1 + lm[b + 1] * s;
            landmarks_px[k][2] = lm[b + 2] * s; // relative z
        }
        // Phase 10.2 relaxation: a hand whose ROI came from a low-vis wrist keeps
        // the model presence gate above, but its emitted score is honestly
        // downweighted by vis/0.5 (1.0 at/above the classic gate).
        let vis_weight = (roi.vis / 0.5).clamp(0.0, 1.0);
        let score = score * vis_weight;
        Ok(Some(Hand { landmarks_px, handedness, score, side: roi.which }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a 33-landmark pose with a visible left hand at a known location.
    fn pose_with_left_hand() -> [[f32; 4]; 33] {
        let mut lm = [[0.0f32, 0.0, 0.0, 0.0]; 33];
        // left wrist at (100,100), fingers up-left; all visible.
        lm[L_WRIST] = [100.0, 100.0, 0.0, 0.9];
        lm[L_INDEX] = [110.0, 70.0, 0.0, 0.9];
        lm[L_PINKY] = [90.0, 72.0, 0.0, 0.9];
        lm[L_THUMB] = [120.0, 90.0, 0.0, 0.9];
        // right wrist not visible.
        lm[R_WRIST] = [300.0, 100.0, 0.0, 0.1];
        lm
    }

    #[test]
    fn roi_from_visible_wrist_is_centered_past_wrist() {
        let lm = pose_with_left_hand();
        let rois = hand_rois_from_pose(&lm, 0.5);
        assert_eq!(rois.len(), 1, "only the visible (left) hand");
        let r = rois[0];
        assert_eq!(r.which, HandSide::Left);
        // Centre is pushed from the wrist (100,100) toward the fingers (~y 71).
        assert!(r.cy < 100.0, "centre should move toward the fingers, got {}", r.cy);
        assert!(r.side >= 24.0);
    }

    #[test]
    fn invisible_wrist_yields_no_roi() {
        let mut lm = pose_with_left_hand();
        lm[L_WRIST][3] = 0.2; // now both wrists low-vis
        let rois = hand_rois_from_pose(&lm, 0.5);
        assert!(rois.is_empty());
    }

    #[test]
    fn both_hands_when_both_visible() {
        let mut lm = pose_with_left_hand();
        lm[R_WRIST] = [300.0, 100.0, 0.0, 0.9];
        lm[R_INDEX] = [290.0, 70.0, 0.0, 0.9];
        lm[R_PINKY] = [310.0, 72.0, 0.0, 0.9];
        lm[R_THUMB] = [280.0, 90.0, 0.0, 0.9];
        let rois = hand_rois_from_pose(&lm, 0.5);
        assert_eq!(rois.len(), 2);
    }

    #[test]
    fn bbox_norm_spans_landmarks() {
        // All landmarks inside [64,96]..[192,240] so min/max are well-defined.
        let mut landmarks_px = [[64.0f32, 96.0, 0.0]; N_HAND_LM];
        landmarks_px[1] = [192.0, 240.0, 0.0];
        let hand = Hand { landmarks_px, handedness: 0.9, score: 0.9, side: HandSide::Left };
        let b = hand.bbox_norm(640.0, 480.0);
        assert!((b.x1 - 0.1).abs() < 1e-4 && (b.x2 - 0.3).abs() < 1e-4);
        assert!((b.y1 - 0.2).abs() < 1e-4 && (b.y2 - 0.5).abs() < 1e-4);
    }

    #[test]
    fn sample_bilinear_out_of_bounds_is_black() {
        let img = RgbImage::new(10, 10);
        assert_eq!(sample_bilinear(&img, -5.0, -5.0), [0.0, 0.0, 0.0]);
        assert_eq!(sample_bilinear(&img, 100.0, 100.0), [0.0, 0.0, 0.0]);
    }

    // ── gesture classification (phase 10.2, pure) ──

    /// A canonical upright hand: wrist at (100,200), palm length 40 px.
    /// `ext[f]` extends finger f (index/middle/ring/pinky); curled tips fold
    /// back toward the palm.
    fn hand_lm(ext: [bool; 4], thumb_on_index: bool) -> [[f32; 3]; N_HAND_LM] {
        let mut lm = [[0.0f32; 3]; N_HAND_LM];
        lm[0] = [100.0, 200.0, 0.0]; // wrist
        // Finger columns at x = 90/100/110/120, MCP y=160, PIP y=145.
        for (f, &(mcp, pip, dip, tip)) in
            [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
                .iter()
                .enumerate()
        {
            let x = 90.0 + f as f32 * 10.0;
            lm[mcp] = [x, 160.0, 0.0];
            lm[pip] = [x, 145.0, 0.0];
            if ext[f] {
                lm[dip] = [x, 130.0, 0.0];
                lm[tip] = [x, 115.0, 0.0]; // far above the PIP → extended
            } else {
                lm[dip] = [x, 150.0, 0.0];
                lm[tip] = [x, 165.0, 0.0]; // folded back to the palm
            }
        }
        // Thumb: 1-4 out to the left; tip either resting away or on the index tip.
        lm[1] = [90.0, 190.0, 0.0];
        lm[2] = [80.0, 180.0, 0.0];
        lm[3] = [72.0, 172.0, 0.0];
        lm[4] = if thumb_on_index {
            [lm[8][0] - 2.0, lm[8][1] + 2.0, 0.0]
        } else {
            [60.0, 165.0, 0.0]
        };
        lm
    }

    #[test]
    fn gesture_open_hand() {
        let lm = hand_lm([true; 4], false);
        assert_eq!(classify_gesture(&lm), Some(HandGesture::Open));
    }

    #[test]
    fn gesture_fist() {
        let lm = hand_lm([false; 4], false);
        assert_eq!(classify_gesture(&lm), Some(HandGesture::Fist));
    }

    #[test]
    fn gesture_point_index_only() {
        let lm = hand_lm([true, false, false, false], false);
        assert_eq!(classify_gesture(&lm), Some(HandGesture::Point));
    }

    #[test]
    fn gesture_pinch_thumb_on_index_others_open() {
        // The "OK" shape: thumb tip on the index tip, middle/ring/pinky open.
        let lm = hand_lm([true, true, true, true], true);
        assert_eq!(classify_gesture(&lm), Some(HandGesture::Pinch));
    }

    #[test]
    fn gesture_fist_with_thumb_on_curled_index_is_not_pinch() {
        // In a fist the thumb rests on the curled index — must stay a fist.
        let lm = hand_lm([false; 4], true);
        assert_eq!(classify_gesture(&lm), Some(HandGesture::Fist));
    }

    #[test]
    fn gesture_degenerate_palm_is_none() {
        let lm = [[5.0f32, 5.0, 0.0]; N_HAND_LM];
        assert_eq!(classify_gesture(&lm), None);
    }

    #[test]
    fn gesture_labels() {
        assert_eq!(HandGesture::Fist.as_str(), "fist");
        assert_eq!(HandGesture::Open.as_str(), "open");
        assert_eq!(HandGesture::Point.as_str(), "point");
        assert_eq!(HandGesture::Pinch.as_str(), "pinch");
    }

    // ── vis relaxation (phase 10.2) ──

    #[test]
    fn low_vis_wrist_yields_roi_with_recorded_vis() {
        let mut lm = pose_with_left_hand();
        lm[L_WRIST][3] = 0.3; // below the classic 0.5 gate
        assert!(hand_rois_from_pose(&lm, 0.5).is_empty(), "0.5 floor still skips");
        let rois = hand_rois_from_pose(&lm, 0.2);
        assert_eq!(rois.len(), 1, "relaxed floor recovers the hand");
        assert!((rois[0].vis - 0.3).abs() < 1e-6, "ROI records the wrist vis");
    }

    // model-gated end-to-end load
    #[test]
    fn model_loads_when_present() {
        let p = std::path::PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../models/handpose_estimation_mediapipe_2023feb.onnx"
        ));
        if !p.exists() {
            eprintln!("Skipping: {} not found", p.display());
            return;
        }
        let mut hl = HandLandmarker::from_path(&p).unwrap();
        let img = RgbImage::new(640, 480);
        // Black ROI → low presence → None (must not panic).
        let r = hl.infer_square(&img, &HandRoi { cx: 320.0, cy: 240.0, side: 200.0, which: HandSide::Left, vis: 0.9 }).unwrap();
        eprintln!("black ROI hand = {:?}", r.map(|h| h.score));
    }
}
