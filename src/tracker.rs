//! Continuous person tracker with IoU-based identity persistence.
//!
//! Key insight: a person is the same person across frames if their bounding
//! boxes overlap significantly. We accumulate confidence over time instead of
//! re-identifying from scratch each tick.

use std::collections::HashMap;

/// A bounding box in frame coordinates (normalised 0..1).
#[derive(Debug, Clone, Copy)]
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

    /// Intersection-over-Union with another bbox.
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

/// A tracked person — persistent across frames.
#[derive(Debug, Clone)]
pub struct PersonTrack {
    pub id: u32,
    pub bbox: BBox,
    /// How many consecutive frames this track has been seen.
    pub age: u64,
    /// Frames since last match (used for timeout).
    pub missed: u64,
}

/// Multi-person tracker.
pub struct PersonTracker {
    next_id: u32,
    tracks: HashMap<u32, PersonTrack>,
    /// Remove tracks unseen for this many frames (~30s).
    timeout_frames: u64,
}

impl PersonTracker {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            tracks: HashMap::new(),
            timeout_frames: 1800, // 30s at 60fps
        }
    }

    /// Process one frame's detections. Matches new bboxes to existing tracks
    /// via IoU. Expires stale tracks.
    pub fn tick(&mut self, detections: &[BBox]) {
        // Increment missed counter for all existing tracks
        for t in self.tracks.values_mut() {
            t.missed += 1;
        }

        // Greedy IoU matching (simple for now, Hungarian later if needed)
        let mut matched_ids: Vec<u32> = Vec::new();

        for det in detections {
            let mut best_iou = 0.3_f32; // minimum IoU to consider a match
            let mut best_id = None;

            for (id, track) in &self.tracks {
                if matched_ids.contains(id) {
                    continue;
                }
                let iou = det.iou(&track.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_id = Some(*id);
                }
            }

            if let Some(id) = best_id {
                // Update existing track
                if let Some(t) = self.tracks.get_mut(&id) {
                    // EMA-smooth the bbox
                    let alpha = 0.3;
                    t.bbox.x1 = alpha * det.x1 + (1.0 - alpha) * t.bbox.x1;
                    t.bbox.y1 = alpha * det.y1 + (1.0 - alpha) * t.bbox.y1;
                    t.bbox.x2 = alpha * det.x2 + (1.0 - alpha) * t.bbox.x2;
                    t.bbox.y2 = alpha * det.y2 + (1.0 - alpha) * t.bbox.y2;
                    t.age += 1;
                    t.missed = 0;
                    matched_ids.push(id);
                }
            } else {
                // New track
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.insert(id, PersonTrack {
                    id,
                    bbox: *det,
                    age: 1,
                    missed: 0,
                });
            }
        }

        // Remove stale tracks
        self.tracks.retain(|_, t| t.missed < self.timeout_frames);
    }

    /// Currently active tracks (seen in the last frame).
    pub fn active(&self) -> impl Iterator<Item = &PersonTrack> {
        self.tracks.values().filter(|t| t.missed == 0)
    }

    /// All tracks (including recently lost ones).
    #[allow(dead_code)]
    pub fn all(&self) -> &HashMap<u32, PersonTrack> {
        &self.tracks
    }

    pub fn track_count(&self) -> usize {
        self.tracks.len()
    }
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
        // x2 < x1, y2 < y1 — should clamp to 0
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
        // inter = 0.2*1.0 = 0.2, union = 0.6+0.6-0.2 = 1.0, iou = 0.2
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
        // b полностью внутри a. inter = 0.36, union = 1.0
        assert!((a.iou(&b) - 0.36).abs() < 1e-5);
    }

    #[test]
    fn iou_edge_touching() {
        // касаются границей, без перекрытия
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

    // ── PersonTracker ─────────────────────────────────────

    /// Helper: single detection at centre of frame.
    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> Vec<BBox> {
        vec![BBox { x1, y1, x2, y2 }]
    }

    #[test]
    fn tracker_creates_new_track_on_first_detection() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, 0);
        assert_eq!(active[0].age, 1);
        assert_eq!(active[0].missed, 0);
    }

    #[test]
    fn tracker_matches_by_iou() {
        let mut t = PersonTracker::new();
        // Frame 1
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        // Frame 2: slight shift, should match same track
        t.tick(&det(0.32, 0.31, 0.72, 0.71));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, 0);
        assert_eq!(active[0].age, 2);
    }

    #[test]
    fn tracker_creates_separate_tracks_for_non_overlapping() {
        let mut t = PersonTracker::new();
        // Two distinct people
        t.tick(&vec![
            BBox { x1: 0.1, y1: 0.1, x2: 0.3, y2: 0.5 },
            BBox { x1: 0.7, y1: 0.1, x2: 0.9, y2: 0.5 },
        ]);
        assert_eq!(t.track_count(), 2);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 2);
        assert_ne!(active[0].id, active[1].id);
    }

    #[test]
    fn tracker_does_not_match_below_threshold() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.0, 0.0, 0.2, 0.2));
        // Far away detection — below 0.3 IoU threshold
        t.tick(&det(0.7, 0.7, 0.9, 0.9));
        assert_eq!(t.track_count(), 2); // new track, not a match
    }

    #[test]
    fn tracker_increments_missed_when_not_seen() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&[]); // empty frame — person left
        let all: Vec<_> = t.all().values().collect();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].missed, 1);
        assert_eq!(all[0].age, 1); // age doesn't grow when unmatched
    }

    #[test]
    fn tracker_expires_stale_tracks() {
        let mut t = PersonTracker::new();
        // Manually lower timeout for testing
        t.timeout_frames = 3;
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&[]);
        t.tick(&[]);
        t.tick(&[]); // 3 missed → expired
        assert_eq!(t.track_count(), 0);
    }

    #[test]
    fn tracker_ema_smoothing() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.5, 0.5));
        // Slight shift right (overlapping). alpha=0.3:
        //   x1: 0.3*0.35 + 0.7*0.3 = 0.105 + 0.21 = 0.315
        //   x2: 0.3*0.55 + 0.7*0.5 = 0.165 + 0.35 = 0.515
        t.tick(&det(0.35, 0.3, 0.55, 0.5));
        let active: Vec<_> = t.active().collect();
        let b = active[0].bbox;
        assert!((b.x1 - 0.315).abs() < 0.01);
        assert!((b.x2 - 0.515).abs() < 0.01);
        // y unchanged
        assert!((b.y1 - 0.3).abs() < 0.01);
    }

    #[test]
    fn tracker_active_filters_missed() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.1, 0.1, 0.3, 0.3));
        t.tick(&det(0.7, 0.7, 0.9, 0.9)); // new person, old one missed
        // First track (id=0) has missed=1, second (id=1) has missed=0
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, 1);
    }

    #[test]
    fn tracker_id_increments() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.0, 0.0, 0.2, 0.2));
        // empty frame — track 0 becomes missed
        t.tick(&[]);
        // new detection far away — gets id=1
        t.tick(&det(0.5, 0.5, 0.7, 0.7));
        assert_eq!(t.track_count(), 2);
        let ids: Vec<u32> = t.all().keys().copied().collect();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
    }

    #[test]
    fn tracker_empty_ticks() {
        let mut t = PersonTracker::new();
        for _ in 0..100 {
            t.tick(&[]);
        }
        assert_eq!(t.track_count(), 0);
        assert_eq!(t.active().count(), 0);
    }
}
