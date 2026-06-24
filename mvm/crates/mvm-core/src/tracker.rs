//! Multi-person tracker with IoU-based identity persistence.
//!
//! Tracks people across frames, propagating identity (best confidence),
//! emotion and posture (latest snapshot), and 3D workspace data.

use std::collections::HashMap;
use crate::types::{Detection, PersonTrack};

/// Multi-person tracker — the temporal fusion layer of MVM.
pub struct PersonTracker {
    next_id: u32,
    tracks: HashMap<u32, PersonTrack>,
    pub timeout_frames: u64,
    /// Frames of no detection before emotion confidence starts decaying.
    pub decay_start_frames: u64,
    /// Per-frame multiplier applied to emotion_confidence during decay.
    pub decay_factor: f32,
}

impl PersonTracker {
    pub fn new() -> Self {
        Self {
            next_id: 0,
            tracks: HashMap::new(),
            timeout_frames: 1800, // 30s at 60fps
            decay_start_frames: 5,
            decay_factor: 0.8,
        }
    }

    /// Process one frame's detections. Matches new detections to existing
    /// tracks via IoU, propagates metadata, expires stale tracks.
    pub fn tick(&mut self, detections: &[Detection]) {
        for t in self.tracks.values_mut() {
            t.missed += 1;
            // Decay emotion confidence after grace period; clear when negligible.
            if t.missed > self.decay_start_frames {
                if let Some(conf) = t.emotion_confidence.as_mut() {
                    *conf *= self.decay_factor;
                    if *conf < 0.05 {
                        t.emotion = None;
                        t.emotion_confidence = None;
                    }
                }
            }
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
    use crate::types::{BBox, Detection};

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
        t.tick(&vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            identity: Some("unknown".into()),
            identity_confidence: Some(0.30),
            emotion: None, emotion_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }]);
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

    fn det_with_emotion(emotion: &str, conf: f32) -> Vec<Detection> {
        vec![Detection {
            bbox: BBox { x1: 0.3, y1: 0.3, x2: 0.7, y2: 0.7 },
            emotion: Some(emotion.into()),
            emotion_confidence: Some(conf),
            identity: None, identity_confidence: None,
            posture: None, posture_confidence: None,
            distance: None, position_x: None, position_y: None, position_z: None,
            ws_confidence: None, recon_error: None, plausibility: None,
        }]
    }

    #[test]
    fn emotion_confidence_decays_after_missed_frames() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 3;
        t.decay_factor = 0.8;
        t.tick(&det_with_emotion("Радость", 0.9));
        // Grace period — no decay yet
        for _ in 0..3 {
            t.tick(&[]);
        }
        let tracks: Vec<_> = t.all().values().collect();
        assert!((tracks[0].emotion_confidence.unwrap() - 0.9).abs() < 0.01);
        // One tick past grace period — decay kicks in
        t.tick(&[]);
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion_confidence.unwrap() < 0.9);
    }

    #[test]
    fn emotion_clears_when_confidence_drops_below_threshold() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 0;
        t.decay_factor = 0.5;
        t.tick(&det_with_emotion("Грусть", 0.8));
        // 0.8 * 0.5^N — reaches < 0.05 after ~4 decays
        for _ in 0..10 {
            t.tick(&[]);
        }
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion.is_none());
        assert!(tracks[0].emotion_confidence.is_none());
    }

    #[test]
    fn emotion_restores_on_rematch_after_decay() {
        let mut t = PersonTracker::new();
        t.decay_start_frames = 0;
        t.decay_factor = 0.5;
        t.tick(&det_with_emotion("Грусть", 0.8));
        // Decay to clear
        for _ in 0..10 {
            t.tick(&[]);
        }
        let tracks: Vec<_> = t.all().values().collect();
        assert!(tracks[0].emotion.is_none());
        // Person reappears with new emotion — should restore
        t.tick(&det_with_emotion("Радость", 0.75));
        let active: Vec<_> = t.active().collect();
        assert_eq!(active[0].emotion.as_deref(), Some("Радость"));
        assert!((active[0].emotion_confidence.unwrap() - 0.75).abs() < 0.01);
    }
}
