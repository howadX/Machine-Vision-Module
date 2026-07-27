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
/// Exit dwell while the instance is younger than [`ACTIVITY_D_MIN_S`] — harder
/// to exit young (θ_lo(d), poor man's HSMM): a just-confirmed activity is not
/// chopped by a gap a mid-life one would survive anyway.
pub const ACTIVITY_EXIT_DWELL_YOUNG: u32 = 8;
/// Exit dwell once the instance outlives [`ACTIVITY_D_TYP_S`] — easier to exit
/// past the typical duration. Still > `object_cadence` (3): the YOLO
/// inter-cadence flicker must never end an activity.
pub const ACTIVITY_EXIT_DWELL_OLD: u32 = 4;
//
// ── Phase 15.6 P7: FSM exit-dwell sweep — DEFERRED (not shipped) ──────────────
// LJ4/S1 found the exit dwell is the ONE knob that fixes episode fragmentation
// under detector STRESS: raising base/young 5/8 → 12/15 lifts stress precision
// med 0.133 → 0.667, heavy 0.029 → 0.545 (no tracker/stabilizer/face/flip knob
// touches it). Verified in 15.6: 12/15 keeps the golden CI green (FLOOR P/R still
// 1.000, idle G6 clean) — no regression. It is NOT shipped, deliberately:
//   • the golden CI is FLOOR-only, so the *benefit* cannot be adjudicated by the
//     shipping gate — it exists only against LJ4's synthetic stress model;
//   • under normal (FLOOR) load 5/8 already scores P/R 1.000, so the sweep buys
//     nothing there while imposing an always-on cost: ~0.5 s later event
//     finalization, plus merging episodes separated by < ~0.8 s.
// It is a one-line flip (5→12, 8→15) once the owner live-validates that real
// multi-person/occlusion fragmentation warrants the latency. Keep at 5/8.
/// Below this observed duration an instance counts as "young" for θ_lo(d).
pub const ACTIVITY_D_MIN_S: f64 = 2.0;
/// Above this observed duration an instance counts as "past typical" for θ_lo(d).
/// Data-informed placeholder — to be refined from event-journal histograms.
pub const ACTIVITY_D_TYP_S: f64 = 600.0;
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
        Self {
            label: a.label.clone(),
            participants,
        }
    }

    /// Every participant is a durable id — gallery-reserved (0–9) or object
    /// (≥10000). Ephemeral person ids (10..10000) are reused across track churn,
    /// so an event keyed on one is not safely attributable over time.
    pub fn stable_participants(&self) -> bool {
        self.participants.iter().all(|&id| id < 10 || id >= 10000)
    }
}

/// Churn class of an id (11.C6): `None` for reserved gallery ids (0–9 — sticky,
/// never a churn); `Some(0)` for ephemeral persons (10–9999, reused across
/// walking-fragmentation); `Some(1)` for objects (≥10000, reused across
/// object-track fragmentation). Two instances stitch only across a differing id
/// of the **same churnable class** — so an anon person's id churn and a
/// keyboard's id churn each continue their episode, but two *different* gallery
/// people never merge.
fn churn_class(id: u32) -> Option<u8> {
    if id < 10 {
        None
    } else if id < 10_000 {
        Some(0)
    } else {
        Some(1)
    }
}

/// Whether two sorted participant sets differ in exactly one id on each side and
/// those two ids share the same churnable class (the rest — the anchor — is
/// identical). The stitch predicate (11.C6): person-churn (anon id 12→15 on the
/// same keyboard) and object-churn (keyboard id 10000→10007 under the same
/// person) are both "one churnable participant moved, anchor held".
fn one_churn_apart(a: &[u32], b: &[u32]) -> bool {
    let (mut i, mut j) = (0usize, 0usize);
    let (mut a_only, mut b_only) = (Vec::new(), Vec::new());
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            std::cmp::Ordering::Less => {
                a_only.push(a[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                b_only.push(b[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    a_only.extend_from_slice(&a[i..]);
    b_only.extend_from_slice(&b[j..]);
    a_only.len() == 1
        && b_only.len() == 1
        && churn_class(a_only[0]).is_some()
        && churn_class(a_only[0]) == churn_class(b_only[0])
}

/// Bounded-duration cap per label (poor man's HSMM surrogate for a learned
/// p(d)): a key that never goes absent still emits journal events no longer
/// than this. Data-informed placeholders — to be refined from event-journal
/// histograms once they exist.
pub fn max_dwell_s(label: &str) -> f64 {
    match label {
        // Transient by nature: a drink in the hand / an object in use for an
        // hour straight is two-or-more episodes, not one. (Vocab v2: the old
        // `using_phone` folded into the `using_object(...)` umbrella — the cap
        // rides the whole family by prefix.)
        "holding_drink" => 900.0,
        l if l.starts_with("using_object(") => 1800.0,
        _ => 3600.0,
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

impl ActivityPhase {
    /// Stable lowercase name for the output contract (phase 11.C5).
    pub fn as_str(self) -> &'static str {
        match self {
            ActivityPhase::Candidate => "candidate",
            ActivityPhase::Active => "active",
            ActivityPhase::Cooling => "cooling",
            ActivityPhase::Ended => "ended",
        }
    }
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
    /// All participants durable (reserved/object ids) — see
    /// [`ActivityKey::stable_participants`].
    pub stable_participants: bool,
}

/// The journal record for an instance ending now (exit dwell or duration cap).
fn event_from(inst: &ActivityInstance) -> ActivityEvent {
    ActivityEvent {
        label: inst.key.label.clone(),
        participants: inst.key.participants.clone(),
        started_at: inst.started_at,
        ended_at: inst.last_seen,
        duration_s: inst.last_seen - inst.started_at,
        peak_confidence: inst.peak_confidence,
        stable_participants: inst.key.stable_participants(),
    }
}

/// The temporal debounce + event journal over raw per-tick activities.
#[derive(Debug, Default)]
pub struct ActivityJournal {
    /// Sorted by key — deterministic iteration/emission order.
    instances: Vec<ActivityInstance>,
    completed: Vec<ActivityEvent>,
    /// Stitch a single participant's id-churn into one continuous instance when
    /// it happens on a held anchor within the cooling window (phase 11.C6):
    /// an anon person's id (12→15) on the same object, OR — the live-observed
    /// case — an object's id (keyboard 10000→10007) under the same person, which
    /// object-track fragmentation churns every few frames and would otherwise
    /// spam the journal with dozens of ½-second typing fragments. Off by default
    /// — the honest behaviour is churn = end + restart (11.C2); stitching is only
    /// safe once masks make the "same entity" claim (person 10.7 / object 10.5),
    /// so it is opt-in.
    stitch_churn: bool,
}

impl ActivityJournal {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable single-participant churn stitching — person- and object-id churn
    /// (phase 11.C6, gated behind masks 10.5/10.7).
    pub fn with_stitching(mut self, on: bool) -> Self {
        self.stitch_churn = on;
        self
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
                    // Bounded-duration cap (11.C2): a never-absent key still
                    // journals in ≤ max_dwell episodes; it re-enters as a fresh
                    // candidate on the next tick.
                    if inst.phase == ActivityPhase::Active
                        && inst.last_seen - inst.started_at >= max_dwell_s(&inst.key.label)
                    {
                        inst.phase = ActivityPhase::Ended;
                        self.completed.push(event_from(inst));
                    }
                }
                None => match inst.phase {
                    // An unconfirmed candidate dies on its first gap — the
                    // 1-frame spike never reaches the journal.
                    ActivityPhase::Candidate => inst.phase = ActivityPhase::Ended,
                    ActivityPhase::Active | ActivityPhase::Cooling => {
                        inst.phase = ActivityPhase::Cooling;
                        inst.absent_streak += 1;
                        // θ_lo(d) (11.C2): the exit dwell depends on the observed
                        // duration — semi-Markov awareness as scalar arithmetic.
                        let d = inst.last_seen - inst.started_at;
                        let exit_dwell = if d < ACTIVITY_D_MIN_S {
                            ACTIVITY_EXIT_DWELL_YOUNG
                        } else if d > ACTIVITY_D_TYP_S {
                            ACTIVITY_EXIT_DWELL_OLD
                        } else {
                            ACTIVITY_EXIT_DWELL
                        };
                        if inst.absent_streak >= exit_dwell {
                            inst.phase = ActivityPhase::Ended;
                            self.completed.push(event_from(inst));
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
            // 11.C6: if this newborn is one churned participant away from a
            // *cooling* instance (an anon person's id, or an object's id, moved
            // while the anchor held), treat it as that instance continuing —
            // adopt its start so the final event spans the whole episode, and
            // drop the source silently (no split event). Gated by `stitch_churn`.
            let stitched = if self.stitch_churn && key.participants.len() >= 2 {
                self.take_stitch_source(&key)
            } else {
                None
            };
            let inst = match stitched {
                Some(src) => ActivityInstance {
                    key,
                    phase: ActivityPhase::Active, // continuation of a confirmed activity
                    enter_streak: ACTIVITY_ENTER_DWELL,
                    absent_streak: 0,
                    started_at: src.started_at,
                    first_seen_frame: src.first_seen_frame,
                    last_seen: t_secs,
                    confidence: conf,
                    peak_confidence: src.peak_confidence.max(conf),
                },
                None => ActivityInstance {
                    key,
                    phase: ActivityPhase::Candidate,
                    enter_streak: 1,
                    absent_streak: 0,
                    started_at: t_secs,
                    first_seen_frame: frame_id,
                    last_seen: t_secs,
                    confidence: conf,
                    peak_confidence: conf,
                },
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

    /// Find and remove a cooling instance that this newborn key is the churned
    /// continuation of (11.C6): same label, currently absent (cooling), and
    /// exactly one churnable participant apart (anon person or object id moved,
    /// anchor held). Returns its timing metadata for adoption. Removing it
    /// silently means no split event.
    fn take_stitch_source(&mut self, newborn: &ActivityKey) -> Option<ActivityInstance> {
        let idx = self.instances.iter().position(|i| {
            i.phase == ActivityPhase::Cooling
                && i.key.label == newborn.label
                && i.key.participants.len() == newborn.participants.len()
                && one_churn_apart(&i.key.participants, &newborn.participants)
        })?;
        Some(self.instances.remove(idx))
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
        Activity {
            label: label.into(),
            confidence: conf,
            participants,
        }
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
        assert_eq!(
            inst[0].started_at, 0.0,
            "started_at = first sighting, not activation"
        );
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
        assert!(
            j.drain_events().is_empty(),
            "no end event for a bridged gap"
        );
    }

    #[test]
    fn exit_dwell_ends_with_honest_duration() {
        let mut j = ActivityJournal::new();
        // Present for 30 ticks: t = 0.0 .. 2.9 (past D_MIN → base exit dwell).
        let (t, f) = drive(&mut j, &[typing()], 30, 0.0, 0);
        // Absent for EXIT_DWELL ticks → ends.
        drive(&mut j, &[], ACTIVITY_EXIT_DWELL, t, f);
        assert_eq!(j.active().count(), 0, "ended instance no longer emitted");
        let ev = j.drain_events();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].label, "typing");
        assert_eq!(ev[0].participants, vec![0, 10001]);
        assert_eq!(ev[0].started_at, 0.0);
        assert!(
            (ev[0].ended_at - 2.9).abs() < 1e-6,
            "ended_at = last sighting, not + grace"
        );
        assert!((ev[0].duration_s - 2.9).abs() < 1e-6);
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
        let raw = [
            act("typing", vec![10001, 0], 0.5),
            act("typing", vec![0, 10001], 0.9),
        ];
        drive(&mut j, &raw, 3, 0.0, 0);
        assert_eq!(j.active().count(), 1, "one instance, not two/three");
    }

    #[test]
    fn distinct_participant_sets_are_distinct_instances() {
        let mut j = ActivityJournal::new();
        let raw = [
            act("typing", vec![0, 10001], 0.8),
            act("typing", vec![1, 10001], 0.8),
        ];
        drive(&mut j, &raw, 3, 0.0, 0);
        assert_eq!(
            j.active().count(),
            2,
            "different people typing = different instances"
        );
    }

    #[test]
    fn emission_order_is_deterministic() {
        let mut a = ActivityJournal::new();
        let mut b = ActivityJournal::new();
        let raw_ab = [
            act("typing", vec![0, 10001], 0.8),
            act("reading", vec![1, 10002], 0.8),
        ];
        let raw_ba = [
            act("reading", vec![1, 10002], 0.8),
            act("typing", vec![0, 10001], 0.8),
        ];
        drive(&mut a, &raw_ab, 3, 0.0, 0);
        drive(&mut b, &raw_ba, 3, 0.0, 0);
        let ka: Vec<_> = a.active().map(|i| i.key.clone()).collect();
        let kb: Vec<_> = b.active().map(|i| i.key.clone()).collect();
        assert_eq!(ka, kb, "raw order must not leak into emission order");
    }

    // ── 11.C2: keying/churn + poor-man's HSMM ──

    #[test]
    fn participant_churn_ends_and_restarts() {
        // Track churn (walking fragmentation): person 10 → 11 mid-typing. The
        // exact-set key honestly ends the old instance and births a new one —
        // stitching two people into one event would be a lie.
        let mut j = ActivityJournal::new();
        let (t, f) = drive(&mut j, &[act("typing", vec![10, 10001], 0.8)], 6, 0.0, 0);
        let (t, f) = drive(&mut j, &[act("typing", vec![11, 10001], 0.8)], 8, t, f);
        let active: Vec<_> = j.active().collect();
        assert_eq!(active.len(), 1, "new instance active, got {active:?}");
        assert_eq!(active[0].key.participants, vec![11, 10001]);
        let ev = j.drain_events();
        assert_eq!(ev.len(), 1, "old instance ended, got {ev:?}");
        assert_eq!(ev[0].participants, vec![10, 10001]);
        let _ = (t, f);
    }

    #[test]
    fn stitching_off_by_default_ends_and_restarts() {
        // Sanity: without opt-in, churn is still end+restart (11.C2 preserved).
        let mut j = ActivityJournal::new();
        let (t, f) = drive(&mut j, &[act("typing", vec![12, 10001], 0.8)], 6, 0.0, 0);
        drive(&mut j, &[act("typing", vec![15, 10001], 0.8)], 12, t, f);
        assert!(
            !j.drain_events().is_empty(),
            "default: old episode ends on churn"
        );
    }

    #[test]
    fn stitching_bridges_ephemeral_churn_on_same_object() {
        // 11.C6: with stitching on, an anon id 12→15 on the SAME keyboard is one
        // continuous episode — no split event mid-churn, and the final event
        // spans from the id=12 era.
        let mut j = ActivityJournal::new().with_stitching(true);
        let (t, f) = drive(&mut j, &[act("typing", vec![12, 10001], 0.8)], 30, 0.0, 0);
        // id churns to 15 within the cooling window (immediate next frame).
        let (t, f) = drive(&mut j, &[act("typing", vec![15, 10001], 0.8)], 8, t, f);
        // No event emitted during the churn — it's one continuous episode.
        assert!(
            j.drain_events().is_empty(),
            "churn must not split the episode"
        );
        let inst: Vec<_> = j.active().collect();
        assert_eq!(inst.len(), 1, "one continuous instance, got {inst:?}");
        assert_eq!(inst[0].key.participants, vec![15, 10001]);
        assert_eq!(inst[0].started_at, 0.0, "adopted the id=12 start");
        // End it → single event spanning the whole thing.
        drive(&mut j, &[], ACTIVITY_EXIT_DWELL, t, f);
        let ev = j.drain_events();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].started_at, 0.0, "event spans from the original start");
        assert!(ev[0].duration_s > 3.0, "duration {}", ev[0].duration_s);
    }

    #[test]
    fn stitching_does_not_bridge_across_different_objects() {
        // A churn where BOTH participants differ (id 12→15 AND 10001→10002) is
        // not a single-participant churn → honest end+restart.
        let mut j = ActivityJournal::new().with_stitching(true);
        let (t, f) = drive(&mut j, &[act("typing", vec![12, 10001], 0.8)], 6, 0.0, 0);
        drive(&mut j, &[act("typing", vec![15, 10002], 0.8)], 8, t, f);
        assert!(
            !j.drain_events().is_empty(),
            "two-participant churn → end+restart"
        );
    }

    #[test]
    fn stitching_bridges_object_churn_under_stable_person() {
        // The live-observed case (mesh 2026-07-03): a reserved person (id 0)
        // typing while the keyboard's object-id churns 10000→10007 — one
        // continuous typing episode, not two ½-second fragments.
        let mut j = ActivityJournal::new().with_stitching(true);
        let (t, f) = drive(&mut j, &[act("typing", vec![0, 10000], 0.8)], 30, 0.0, 0);
        let (t, f) = drive(&mut j, &[act("typing", vec![0, 10007], 0.8)], 8, t, f);
        assert!(
            j.drain_events().is_empty(),
            "object churn must not split the episode"
        );
        let inst: Vec<_> = j.active().collect();
        assert_eq!(inst.len(), 1, "one continuous instance, got {inst:?}");
        assert_eq!(inst[0].key.participants, vec![0, 10007]);
        assert_eq!(inst[0].started_at, 0.0, "adopted the id=10000 start");
        drive(&mut j, &[], ACTIVITY_EXIT_DWELL, t, f);
        let ev = j.drain_events();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev[0].started_at, 0.0);
        assert!(ev[0].duration_s > 3.0, "duration {}", ev[0].duration_s);
    }

    #[test]
    fn stitching_never_merges_two_gallery_people() {
        // Two DIFFERENT reserved people on the same keyboard are genuinely
        // distinct — reserved ids are sticky, not a churn class.
        let mut j = ActivityJournal::new().with_stitching(true);
        let (t, f) = drive(&mut j, &[act("typing", vec![0, 10001], 0.8)], 6, 0.0, 0);
        drive(&mut j, &[act("typing", vec![5, 10001], 0.8)], 8, t, f);
        assert!(
            !j.drain_events().is_empty(),
            "person 0 ≠ person 5 → honest end+restart"
        );
    }

    #[test]
    fn stable_participants_reflects_id_spaces() {
        // reserved (0–9) + object (≥10000) → stable; ephemeral (10..10000) → not.
        assert!(
            ActivityKey {
                label: "typing".into(),
                participants: vec![0, 10001]
            }
            .stable_participants()
        );
        assert!(
            !ActivityKey {
                label: "typing".into(),
                participants: vec![42, 10001]
            }
            .stable_participants()
        );
        let mut j = ActivityJournal::new();
        let (t, f) = drive(&mut j, &[act("typing", vec![0, 10001], 0.8)], 3, 0.0, 0);
        drive(&mut j, &[], ACTIVITY_EXIT_DWELL + 3, t, f);
        let ev = j.drain_events();
        assert_eq!(ev.len(), 1);
        assert!(ev[0].stable_participants, "reserved+object event is stable");
    }

    #[test]
    fn max_dwell_caps_duration_and_restarts() {
        // A key that never goes absent must still emit bounded events — the
        // bounded-duration cap (poor man's HSMM surrogate for p(d)).
        let mut j = ActivityJournal::new();
        let raw = [act("typing", vec![0, 10001], 0.8)];
        let mut t = 0.0;
        for f in 0..5000u64 {
            j.update(&raw, t, f);
            t += 1.0; // 1s ticks → crosses any sane cap well within the loop
        }
        let ev = j.drain_events();
        assert!(!ev.is_empty(), "cap must have force-ended at least once");
        for e in &ev {
            assert!(
                e.duration_s <= max_dwell_s("typing") + 1.0,
                "duration {} > cap",
                e.duration_s
            );
        }
        assert_eq!(
            j.active().count(),
            1,
            "activity restarts after the forced end"
        );
    }

    #[test]
    fn young_activity_survives_longer_gap() {
        // θ_lo(d): while d < d_min the exit dwell is longer — a just-confirmed
        // activity is not chopped by a gap that would end a mid-life one.
        let mut j = ActivityJournal::new();
        let (t, f) = drive(&mut j, &[typing()], 3, 0.0, 0); // duration 0.2s < D_MIN
        let (t, f) = drive(&mut j, &[], ACTIVITY_EXIT_DWELL + 1, t, f); // > base dwell
        assert_eq!(j.active().count(), 1, "young instance still cooling");
        j.update(&[typing()], t, f);
        let inst: Vec<_> = j.active().collect();
        assert_eq!(inst[0].started_at, 0.0, "same instance survived the gap");
        assert!(j.drain_events().is_empty());
    }

    #[test]
    fn old_activity_exits_faster() {
        // θ_lo(d): past d_typ the exit dwell shortens.
        let mut j = ActivityJournal::new();
        // Confirm, then age the instance beyond D_TYP with sparse 100s ticks.
        let (_, f) = drive(&mut j, &[typing()], 3, 0.0, 0);
        let mut t = 100.0;
        let mut f = f;
        while t < ACTIVITY_D_TYP_S + 200.0 {
            j.update(&[typing()], t, f);
            t += 100.0;
            f += 1;
        }
        drive(&mut j, &[], ACTIVITY_EXIT_DWELL_OLD, t, f);
        assert_eq!(
            j.active().count(),
            0,
            "old instance ends after the short dwell"
        );
        assert_eq!(j.drain_events().len(), 1);
    }

    #[test]
    fn instance_cap_bounds_memory() {
        let mut j = ActivityJournal::new();
        let flood: Vec<Activity> = (0..2 * ACTIVITY_INSTANCE_CAP as u32)
            .map(|i| act("typing", vec![i, 10001], 0.5))
            .collect();
        drive(&mut j, &flood, 3, 0.0, 0);
        assert!(
            j.instances.len() <= ACTIVITY_INSTANCE_CAP,
            "len {}",
            j.instances.len()
        );
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
        assert!(
            j.completed.len() <= ACTIVITY_EVENT_CAP,
            "len {}",
            j.completed.len()
        );
    }
}
