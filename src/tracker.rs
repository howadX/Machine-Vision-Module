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
