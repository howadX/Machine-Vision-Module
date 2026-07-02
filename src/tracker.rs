//! Continuous person tracker with IoU-based identity persistence.
//!
//! v0.3: enriched detections carry identity, emotion, and posture metadata.
//! These are propagated through tracking: the track remembers the best
//! identity and most recent emotion/posture.

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

/// A single detection from the bridge, with optional enrichment.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub identity: Option<String>,
    pub identity_confidence: Option<f32>,
    pub emotion: Option<String>,
    pub emotion_confidence: Option<f32>,
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
    /// Most recent emotion (snapshot, not EMA).
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
    /// Posture quality (latest snapshot).
    pub recon_error: Option<f32>,
    pub plausibility: Option<String>,
}

/// Multi-person tracker.
pub struct PersonTracker {
    next_id: u32,
    tracks: HashMap<u32, PersonTrack>,
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

    /// Process one frame's detections. Matches new detections to existing
    /// tracks via IoU, propagates metadata, expires stale tracks.
    pub fn tick(&mut self, detections: &[Detection]) {
        for t in self.tracks.values_mut() {
            t.missed += 1;
        }

        let mut matched_ids: Vec<u32> = Vec::new();

        for det in detections {
            let mut best_iou = 0.3_f32;
            let mut best_id = None;

            for (id, track) in &self.tracks {
                if matched_ids.contains(id) {
                    continue;
                }
                let iou = det.bbox.iou(&track.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_id = Some(*id);
                }
            }

            if let Some(id) = best_id {
                if let Some(t) = self.tracks.get_mut(&id) {
                    // EMA-smooth the bbox
                    let alpha = 0.3;
                    t.bbox.x1 = alpha * det.bbox.x1 + (1.0 - alpha) * t.bbox.x1;
                    t.bbox.y1 = alpha * det.bbox.y1 + (1.0 - alpha) * t.bbox.y1;
                    t.bbox.x2 = alpha * det.bbox.x2 + (1.0 - alpha) * t.bbox.x2;
                    t.bbox.y2 = alpha * det.bbox.y2 + (1.0 - alpha) * t.bbox.y2;
                    t.age += 1;
                    t.missed = 0;

                    // Propagate metadata: keep best identity, latest emotion/posture
                    if let (Some(name), Some(conf)) = (&det.identity, det.identity_confidence) {
                        if t.identity_confidence.map_or(true, |c| conf > c) {
                            t.identity = Some(name.clone());
                            t.identity_confidence = Some(conf);
                        }
                    }
                    if det.emotion.is_some() {
                        t.emotion = det.emotion.clone();
                        t.emotion_confidence = det.emotion_confidence;
                    }
                    if det.posture.is_some() {
                        t.posture = det.posture.clone();
                        t.posture_confidence = det.posture_confidence;
                    }
                    // Propagate 3D fields (latest snapshot)
                    if det.distance.is_some() {
                        t.distance = det.distance;
                        t.position_x = det.position_x;
                        t.position_y = det.position_y;
                        t.position_z = det.position_z;
                        t.ws_confidence = det.ws_confidence;
                    }
                    if det.recon_error.is_some() {
                        t.recon_error = det.recon_error;
                    }
                    if det.plausibility.is_some() {
                        t.plausibility = det.plausibility.clone();
                    }

                    matched_ids.push(id);
                }
            } else {
                // New track
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.insert(id, PersonTrack {
                    id,
                    bbox: det.bbox,
                    age: 1,
                    missed: 0,
                    identity: det.identity.clone(),
                    identity_confidence: det.identity_confidence,
                    emotion: det.emotion.clone(),
                    emotion_confidence: det.emotion_confidence,
                    posture: det.posture.clone(),
                    posture_confidence: det.posture_confidence,
                    distance: det.distance,
                    position_x: det.position_x,
                    position_y: det.position_y,
                    position_z: det.position_z,
                    ws_confidence: det.ws_confidence,
                    recon_error: det.recon_error,
                    plausibility: det.plausibility.clone(),
                });
            }
        }

        self.tracks.retain(|_, t| t.missed < self.timeout_frames);
    }

    pub fn active(&self) -> impl Iterator<Item = &PersonTrack> {
        self.tracks.values().filter(|t| t.missed == 0)
    }

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

    // ── PersonTracker ─────────────────────────────────────

    /// Helper: bare detection (no enrichment).
    fn det(x1: f32, y1: f32, x2: f32, y2: f32) -> Vec<Detection> {
        vec![Detection {
            bbox: BBox { x1, y1, x2, y2 },
            identity: None,
            identity_confidence: None,
            emotion: None,
            emotion_confidence: None,
            posture: None,
            posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }]
    }

    #[test]
    fn tracker_creates_new_track_on_first_detection() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].id, 0);
        assert_eq!(active[0].age, 1);
    }

    #[test]
    fn tracker_propagates_identity() {
        let mut t = PersonTracker::new();
        let enriched = vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: Some("Нейтральная".into()),
            emotion_confidence: Some(0.65),
            posture: Some("сидит".into()),
            posture_confidence: Some(0.95),
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }];
        t.tick(&enriched);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
        assert_eq!(active[0].emotion.as_deref(), Some("Нейтральная"));
        assert_eq!(active[0].posture.as_deref(), Some("сидит"));
    }

    #[test]
    fn tracker_keeps_best_identity() {
        let mut t = PersonTracker::new();
        // First detection: low confidence
        t.tick(&vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("unknown".into()),
            identity_confidence: Some(0.30),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }]);
        // Second detection: higher confidence, same bbox
        t.tick(&vec![Detection {
            bbox: BBox { x1: 0.31, y1: 0.31, x2: 0.71, y2: 0.71 },
            identity: Some("Вадим".into()),
            identity_confidence: Some(0.72),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }]);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].identity.as_deref(), Some("Вадим"));
        assert!((active[0].identity_confidence.unwrap() - 0.72).abs() < 0.01);
    }

    #[test]
    fn tracker_updates_emotion_on_match() {
        let mut t = PersonTracker::new();
        t.tick(&vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.5, y2: 0.5 },
            emotion: Some("Злость".into()),
            emotion_confidence: Some(0.6),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
                distance: None, position_x: None, position_y: None, position_z: None,
                ws_confidence: None, recon_error: None, plausibility: None,
        }]);
        // Same person, new emotion
        t.tick(&vec![Detection {
            bbox: BBox { x1: 0.32, y1: 0.31, x2: 0.52, y2: 0.51 },
            emotion: Some("Радость".into()),
            emotion_confidence: Some(0.8),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
                distance: None, position_x: None, position_y: None, position_z: None,
                ws_confidence: None, recon_error: None, plausibility: None,
        }]);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].emotion.as_deref(), Some("Радость"));
    }

    #[test]
    fn tracker_matches_by_iou() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&det(0.32, 0.31, 0.72, 0.71));
        assert_eq!(t.track_count(), 1);
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].id, 0);
        assert_eq!(active[0].age, 2);
    }

    #[test]
    fn tracker_creates_separate_tracks_for_non_overlapping() {
        let mut t = PersonTracker::new();
        t.tick(&vec![
            Detection { bbox: BBox { x1: 0.1, y1: 0.1, x2: 0.3, y2: 0.5 }, identity: None, identity_confidence: None, emotion: None, emotion_confidence: None, posture: None, posture_confidence: None, distance: None, position_x: None, position_y: None, position_z: None, ws_confidence: None, recon_error: None, plausibility: None },
            Detection { bbox: BBox { x1: 0.7, y1: 0.1, x2: 0.9, y2: 0.5 }, identity: None, identity_confidence: None, emotion: None, emotion_confidence: None, posture: None, posture_confidence: None, distance: None, position_x: None, position_y: None, position_z: None, ws_confidence: None, recon_error: None, plausibility: None },
        ]);
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn tracker_does_not_match_below_threshold() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.0, 0.0, 0.2, 0.2));
        t.tick(&det(0.7, 0.7, 0.9, 0.9));
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn tracker_increments_missed_when_not_seen() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&[]);
        let all: Vec<_> = t.all().values().collect();
        assert_eq!(all[0].missed, 1);
        assert_eq!(all[0].age, 1);
    }

    #[test]
    fn tracker_expires_stale_tracks() {
        let mut t = PersonTracker::new();
        t.timeout_frames = 3;
        t.tick(&det(0.3, 0.3, 0.7, 0.7));
        t.tick(&[]);
        t.tick(&[]);
        t.tick(&[]);
        assert_eq!(t.track_count(), 0);
    }

    #[test]
    fn tracker_ema_smoothing() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.3, 0.3, 0.5, 0.5));
        // alpha=0.3: x1=0.3*0.35+0.7*0.3=0.315, x2=0.3*0.55+0.7*0.5=0.515
        t.tick(&det(0.35, 0.3, 0.55, 0.5));
        let active: Vec<_> = t.active().collect();
        let b = active[0].bbox;
        assert!((b.x1 - 0.315).abs() < 0.01);
        assert!((b.x2 - 0.515).abs() < 0.01);
        assert!((b.y1 - 0.3).abs() < 0.01);
    }

    #[test]
    fn tracker_active_filters_missed() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.1, 0.1, 0.3, 0.3));
        t.tick(&det(0.7, 0.7, 0.9, 0.9));
        let active: Vec<_> = t.active().collect();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, 1);
    }

    #[test]
    fn tracker_id_increments() {
        let mut t = PersonTracker::new();
        t.tick(&det(0.0, 0.0, 0.2, 0.2));
        t.tick(&[]);
        t.tick(&det(0.5, 0.5, 0.7, 0.7));
        assert_eq!(t.track_count(), 2);
    }

    #[test]
    fn tracker_empty_ticks() {
        let mut t = PersonTracker::new();
        for _ in 0..100 {
            t.tick(&[]);
        }
        assert_eq!(t.track_count(), 0);
    }
}
