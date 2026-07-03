//! Activity FSM + event journal (phase 11.C, pure arithmetic).
//!
//! `infer_activities` re-derives the raw activity list every tick, so a
//! per-frame consumer sees it flicker: a wrist grazing the keyboard for one
//! frame reads as "typing", an object-cadence gap in YOLO reads as the typing
//! having stopped. This module is the temporal debounce over that raw stream —
//! the same dwell + hysteresis family as `MotionClassifier` and
//! `PostureStabilizer`, keyed per activity instance:
//!
//! ```text
//!                present ≥ ENTER_DWELL подряд
//! Candidate ───────────────────────────────────▶ Active   (started_at)
//!    │  absent → reap (не эмитится)                 │  ▲
//!    │                                       absent │  │ present
//!    │                                              ▼  │
//!    │                                           Cooling  (ещё активна — grace)
//!    │            absent ≥ EXIT_DWELL подряд           │
//!    └──────────────────────────────────────────────  ▼
//!                                                   Ended  (event → journal; reap)
//! ```
//!
//! The journal owns bounded state only (instance cap + completed-event cap) and
//! never calls models — a thin arithmetic wrapper over `&[Activity]`, identical
//! on the mesh and sync paths.

use crate::world::Activity;

/// Consecutive present-ticks before a candidate becomes Active (~0.2s @ 15fps;
/// the ID_LOCK_FRAMES / EMOTION_CADENCE family). Kills the 1-frame "typing".
pub const ACTIVITY_ENTER_DWELL: u32 = 3;
/// Consecutive absent-ticks before an Active/Cooling instance ends. Chosen >
/// `object_cadence` (3) so the inter-cadence YOLO flicker never ends an
/// activity — sticky-to-persist.
pub const ACTIVITY_EXIT_DWELL: u32 = 5;
/// Confidence EMA: `conf = λ·prev + (1−λ)·obs` — the emotion temporal-PoE λ family.
pub const ACTIVITY_CONF_LAMBDA: f32 = 0.7;
/// Hard cap on simultaneously tracked instances (bounded memory, the
/// MOTION_RING_CAP discipline). New candidates are rejected at the cap;
/// existing instances always keep updating.
pub const ACTIVITY_INSTANCE_CAP: usize = 64;
/// Hard cap on undrained completed events; oldest are dropped beyond it.
pub const ACTIVITY_EVENT_CAP: usize = 256;

/// Identity of an activity instance over time: the label plus the *sorted*
/// participant set. Exact-set match — a participant churn is honestly a
/// different instance (phase 11.C2 refines stitching).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActivityKey {
    pub label: String,
    /// Sorted ascending — id-space: reserved 0–9, ephemeral ≥10, objects ≥10000.
    pub participants: Vec<u32>,
}

impl ActivityKey {
    pub fn from_activity(a: &Activity) -> Self {
        let mut participants = a.participants.clone();
        participants.sort_unstable();
        Self { label: a.label.clone(), participants }
    }
}

/// Lifecycle phase of one activity instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivityPhase {
    /// Seen, not yet confirmed — never emitted.
    Candidate,
    /// Confirmed by the enter dwell — emitted.
    Active,
    /// Recently absent but within the exit grace — still emitted (hysteresis
    /// bridging occlusion / detector cadence gaps).
    Cooling,
    /// Exit dwell exceeded — event recorded, instance reaped this tick.
    Ended,
}

/// One keyed activity being tracked through the FSM.
#[derive(Debug, Clone)]
pub struct ActivityInstance {
    pub key: ActivityKey,
    pub phase: ActivityPhase,
    enter_streak: u32,
    absent_streak: u32,
    /// Journal time when the instance was first seen (the dwell is confirmation
    /// lag, not a later start).
    pub started_at: f64,
    pub first_seen_frame: u64,
    /// Journal time when the key was last present.
    pub last_seen: f64,
    /// EMA-smoothed confidence.
    pub confidence: f32,
    pub peak_confidence: f32,
}

/// A completed activity — the temporal-journal record (start/stop/duration).
#[derive(Debug, Clone)]
pub struct ActivityEvent {
    pub label: String,
    pub participants: Vec<u32>,
    pub started_at: f64,
    /// Last time the activity was actually observed (not + the exit grace).
    pub ended_at: f64,
    pub duration_s: f64,
    pub peak_confidence: f32,
}

/// The temporal debounce + event journal over raw per-tick activities.
#[derive(Debug, Default)]
pub struct ActivityJournal {
    /// Sorted by key — deterministic iteration/emission order.
    instances: Vec<ActivityInstance>,
    completed: Vec<ActivityEvent>,
}

impl ActivityJournal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Advance the FSM one tick. `raw` is this tick's `infer_activities`
    /// output; `t_secs` a monotonic clock in seconds; `frame_id` the frame the
    /// raw list was derived from. Duplicate keys in `raw` collapse to one
    /// observation (max confidence).
    pub fn update(&mut self, raw: &[Activity], t_secs: f64, frame_id: u64) {
        // Present keys → strongest observed confidence this tick.
        let mut present: Vec<(ActivityKey, f32)> = Vec::with_capacity(raw.len());
        for a in raw {
            let key = ActivityKey::from_activity(a);
            match present.binary_search_by(|(k, _)| k.cmp(&key)) {
                Ok(i) => present[i].1 = present[i].1.max(a.confidence),
                Err(i) => present.insert(i, (key, a.confidence)),
            }
        }

        // Advance existing instances.
        for inst in &mut self.instances {
            let obs = present
                .binary_search_by(|(k, _)| k.cmp(&inst.key))
                .ok()
                .map(|i| present.remove(i).1);
            match obs {
                Some(conf) => {
                    inst.absent_streak = 0;
                    inst.last_seen = t_secs;
                    inst.confidence = ACTIVITY_CONF_LAMBDA * inst.confidence
                        + (1.0 - ACTIVITY_CONF_LAMBDA) * conf;
                    inst.peak_confidence = inst.peak_confidence.max(inst.confidence);
                    match inst.phase {
                        ActivityPhase::Candidate => {
                            inst.enter_streak += 1;
                            if inst.enter_streak >= ACTIVITY_ENTER_DWELL {
                                inst.phase = ActivityPhase::Active;
                            }
                        }
                        ActivityPhase::Cooling => inst.phase = ActivityPhase::Active,
                        ActivityPhase::Active | ActivityPhase::Ended => {}
                    }
                }
                None => match inst.phase {
                    // An unconfirmed candidate dies on its first gap — the
                    // 1-frame spike never reaches the journal.
                    ActivityPhase::Candidate => inst.phase = ActivityPhase::Ended,
                    ActivityPhase::Active | ActivityPhase::Cooling => {
                        inst.phase = ActivityPhase::Cooling;
                        inst.absent_streak += 1;
                        if inst.absent_streak >= ACTIVITY_EXIT_DWELL {
                            inst.phase = ActivityPhase::Ended;
                            self.completed.push(ActivityEvent {
                                label: inst.key.label.clone(),
                                participants: inst.key.participants.clone(),
                                started_at: inst.started_at,
                                ended_at: inst.last_seen,
                                duration_s: inst.last_seen - inst.started_at,
                                peak_confidence: inst.peak_confidence,
                            });
                        }
                    }
                    ActivityPhase::Ended => {}
                },
            }
        }
        // Reap: candidates that lapsed and instances whose event was recorded.
        self.instances.retain(|i| i.phase != ActivityPhase::Ended);

        // Keys still unmatched are newborn candidates (bounded).
        for (key, conf) in present {
            if self.instances.len() >= ACTIVITY_INSTANCE_CAP {
                break;
            }
            let inst = ActivityInstance {
                key,
                phase: ActivityPhase::Candidate,
                enter_streak: 1,
                absent_streak: 0,
                started_at: t_secs,
                first_seen_frame: frame_id,
                last_seen: t_secs,
                confidence: conf,
                peak_confidence: conf,
            };
            let at = self
                .instances
                .binary_search_by(|i| i.key.cmp(&inst.key))
                .unwrap_err();
            self.instances.insert(at, inst);
        }

        // Bounded event backlog even if the consumer never drains.
        if self.completed.len() > ACTIVITY_EVENT_CAP {
            let drop = self.completed.len() - ACTIVITY_EVENT_CAP;
            self.completed.drain(..drop);
        }
    }

    /// The stabilized activities to emit: Active plus Cooling (grace-bridged).
    pub fn active(&self) -> impl Iterator<Item = &ActivityInstance> {
        self.instances
            .iter()
            .filter(|i| matches!(i.phase, ActivityPhase::Active | ActivityPhase::Cooling))
    }

    /// Completed events since the last drain (start/stop/duration journal).
    pub fn drain_events(&mut self) -> Vec<ActivityEvent> {
        std::mem::take(&mut self.completed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn act(label: &str, participants: Vec<u32>, conf: f32) -> Activity {
        Activity { label: label.into(), confidence: conf, participants }
    }

    fn typing() -> Activity {
        act("typing", vec![0, 10001], 0.8)
    }

    /// Drive `n` ticks with the given raw list, 0.1s apart, starting at `t0`/`f0`.
    fn drive(j: &mut ActivityJournal, raw: &[Activity], n: u32, t0: f64, f0: u64) -> (f64, u64) {
        let mut t = t0;
        let mut f = f0;
        for _ in 0..n {
            j.update(raw, t, f);
            t += 0.1;
            f += 1;
        }
        (t, f)
    }

    #[test]
    fn one_frame_spike_never_emits() {
        let mut j = ActivityJournal::new();
        j.update(&[typing()], 0.0, 0);
        assert_eq!(j.active().count(), 0, "candidate is not emitted");
        j.update(&[], 0.1, 1);
        j.update(&[typing()], 0.2, 2); // must start over, not resume streak 2
        j.update(&[], 0.3, 3);
        assert_eq!(j.active().count(), 0);
        assert!(j.drain_events().is_empty(), "spikes leave no journal trace");
    }

    #[test]
    fn enter_dwell_confirms_after_three_consecutive() {
        let mut j = ActivityJournal::new();
        drive(&mut j, &[typing()], 2, 0.0, 0);
        assert_eq!(j.active().count(), 0, "2 consecutive < dwell");
        j.update(&[typing()], 0.2, 2);
        let inst: Vec<_> = j.active().collect();
        assert_eq!(inst.len(), 1, "3rd consecutive tick activates");
        assert_eq!(inst[0].phase, ActivityPhase::Active);
        assert_eq!(inst[0].started_at, 0.0, "started_at = first sighting, not activation");
        assert_eq!(inst[0].first_seen_frame, 0);
    }

    #[test]
    fn short_dropout_is_bridged_by_cooling() {
        let mut j = ActivityJournal::new();
        let (t, f) = drive(&mut j, &[typing()], 3, 0.0, 0);
        // 2 absent ticks (< EXIT_DWELL) — still emitted, phase Cooling.
        let (t, f) = drive(&mut j, &[], 2, t, f);
        let inst: Vec<_> = j.active().collect();
        assert_eq!(inst.len(), 1, "cooling bridges the gap");
        assert_eq!(inst[0].phase, ActivityPhase::Cooling);
        // Comes back → Active again, same instance, no event.
        j.update(&[typing()], t, f);
        let inst: Vec<_> = j.active().collect();
        assert_eq!(inst[0].phase, ActivityPhase::Active);
        assert_eq!(inst[0].started_at, 0.0, "same instance survived");
        assert!(j.drain_events().is_empty(), "no end event for a bridged gap");
    }

    #[test]
    fn exit_dwell_ends_with_honest_duration() {
        let mut j = ActivityJournal::new();
        // Present for 10 ticks: t = 0.0 .. 0.9 (last_seen 0.9).
        let (t, f) = drive(&mut j, &[typing()], 10, 0.0, 0);
        // Absent for EXIT_DWELL ticks → ends.
        drive(&mut j, &[], ACTIVITY_EXIT_DWELL, t, f);
        assert_eq!(j.active().count(), 0, "ended instance no longer emitted");
        let ev = j.drain_events();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].label, "typing");
        assert_eq!(ev[0].participants, vec![0, 10001]);
        assert_eq!(ev[0].started_at, 0.0);
        assert!((ev[0].ended_at - 0.9).abs() < 1e-9, "ended_at = last sighting, not + grace");
        assert!((ev[0].duration_s - 0.9).abs() < 1e-9);
        assert!(j.drain_events().is_empty(), "drain empties the backlog");
    }

    #[test]
    fn confidence_is_ema_and_peak_tracked() {
        let mut j = ActivityJournal::new();
        j.update(&[act("typing", vec![0, 10001], 1.0)], 0.0, 0);
        j.update(&[act("typing", vec![0, 10001], 0.0)], 0.1, 1);
        let inst: Vec<_> = j.instances.iter().collect();
        // init 1.0, then 0.7·1.0 + 0.3·0.0 = 0.7.
        assert!((inst[0].confidence - 0.7).abs() < 1e-6);
        assert!((inst[0].peak_confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn duplicate_keys_and_participant_order_collapse() {
        let mut j = ActivityJournal::new();
        // Same instance spelled two ways (participant order + duplicate).
        let raw = [act("typing", vec![10001, 0], 0.5), act("typing", vec![0, 10001], 0.9)];
        drive(&mut j, &raw, 3, 0.0, 0);
        assert_eq!(j.active().count(), 1, "one instance, not two/three");
    }

    #[test]
    fn distinct_participant_sets_are_distinct_instances() {
        let mut j = ActivityJournal::new();
        let raw = [act("typing", vec![0, 10001], 0.8), act("typing", vec![1, 10001], 0.8)];
        drive(&mut j, &raw, 3, 0.0, 0);
        assert_eq!(j.active().count(), 2, "different people typing = different instances");
    }

    #[test]
    fn emission_order_is_deterministic() {
        let mut a = ActivityJournal::new();
        let mut b = ActivityJournal::new();
        let raw_ab = [act("typing", vec![0, 10001], 0.8), act("reading", vec![1, 10002], 0.8)];
        let raw_ba = [act("reading", vec![1, 10002], 0.8), act("typing", vec![0, 10001], 0.8)];
        drive(&mut a, &raw_ab, 3, 0.0, 0);
        drive(&mut b, &raw_ba, 3, 0.0, 0);
        let ka: Vec<_> = a.active().map(|i| i.key.clone()).collect();
        let kb: Vec<_> = b.active().map(|i| i.key.clone()).collect();
        assert_eq!(ka, kb, "raw order must not leak into emission order");
    }

    #[test]
    fn instance_cap_bounds_memory() {
        let mut j = ActivityJournal::new();
        let flood: Vec<Activity> =
            (0..2 * ACTIVITY_INSTANCE_CAP as u32).map(|i| act("typing", vec![i, 10001], 0.5)).collect();
        drive(&mut j, &flood, 3, 0.0, 0);
        assert!(j.instances.len() <= ACTIVITY_INSTANCE_CAP, "len {}", j.instances.len());
    }

    #[test]
    fn event_cap_bounds_undrained_backlog() {
        let mut j = ActivityJournal::new();
        let mut t = 0.0;
        let mut f = 0u64;
        for i in 0..ACTIVITY_EVENT_CAP as u32 + 40 {
            // Each iteration: confirm one keyed activity, then end it.
            let raw = [act("typing", vec![i, 10001], 0.5)];
            for _ in 0..ACTIVITY_ENTER_DWELL {
                j.update(&raw, t, f);
                t += 0.1;
                f += 1;
            }
            for _ in 0..ACTIVITY_EXIT_DWELL {
                j.update(&[], t, f);
                t += 0.1;
                f += 1;
            }
        }
        assert!(j.completed.len() <= ACTIVITY_EVENT_CAP, "len {}", j.completed.len());
    }
}
