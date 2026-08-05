//! CLI-owned bounded execution control for advisory active re-observation.
//!
//! This module deliberately knows about heavy experts while `mvm-mesh` remains
//! a generic executor. Permits are one-shot and frame-bounded; they never
//! rewrite persistent cadence or canonical Fusion state.

use super::{AttentionPolicyBus, DepthObservationBus, PolicyOutcomeBus, SceneObservationBus};
use mvm_cortex::{
    AttentionPolicy, ObservationRequest, ObservationTarget, PolicyOutcomeFrame, ReasonCode,
    RequestOutcome, Resolution,
};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ControlState {
    permit: Option<ObservationRequest>,
    dispatched: Vec<u64>,
}

#[derive(Clone, Default)]
pub(crate) struct ExpertExecutionControl(Arc<Mutex<ControlState>>);

impl ExpertExecutionControl {
    pub(crate) fn grant(&self, request: ObservationRequest) {
        self.0.lock().unwrap().permit = Some(request);
    }

    /// Consume exactly one permit for this camera tick. Expired permits are
    /// discarded, so a late expert run cannot become an implicit confirmation.
    pub(crate) fn take_for(&self, frame_id: u64) -> Option<ObservationRequest> {
        let mut state = self.0.lock().unwrap();
        match state.permit.as_ref() {
            Some(request) if frame_id <= request.expires_at_frame => {
                let request = state.permit.take();
                if let Some(request) = &request {
                    state.dispatched.push(request.request_id);
                }
                request
            }
            Some(_) => {
                state.permit.take();
                None
            }
            None => None,
        }
    }

    fn take_dispatched(&self) -> Vec<u64> {
        std::mem::take(&mut self.0.lock().unwrap().dispatched)
    }
}

pub(crate) type SceneExecutionControl = ExpertExecutionControl;
pub(crate) type DepthExecutionControl = ExpertExecutionControl;

pub(crate) struct PolicySchedulerWiring {
    pub(crate) policies: AttentionPolicyBus,
    pub(crate) scenes: SceneObservationBus,
    pub(crate) depths: DepthObservationBus,
    pub(crate) outcomes: PolicyOutcomeBus,
    pub(crate) scene_control: SceneExecutionControl,
    pub(crate) depth_control: DepthExecutionControl,
    pub(crate) scene_available: bool,
    pub(crate) depth_available: bool,
}

pub(crate) struct PolicySchedulerAgent {
    policies: AttentionPolicyBus,
    scenes: SceneObservationBus,
    depths: DepthObservationBus,
    outcomes: PolicyOutcomeBus,
    scheduler: PolicyScheduler,
    last_policy_frame: Option<u64>,
    last_scene_frame: Option<u64>,
    last_depth_frame: Option<u64>,
    published_outcomes: usize,
}

impl PolicySchedulerAgent {
    pub(crate) fn new(wiring: PolicySchedulerWiring) -> Self {
        let mut scheduler = PolicyScheduler::with_controls(
            wiring.scene_control,
            wiring.depth_control,
            SchedulerBudget::default(),
        );
        scheduler.scene_available = wiring.scene_available;
        scheduler.depth_available = wiring.depth_available;
        Self {
            policies: wiring.policies,
            scenes: wiring.scenes,
            depths: wiring.depths,
            outcomes: wiring.outcomes,
            scheduler,
            last_policy_frame: None,
            last_scene_frame: None,
            last_depth_frame: None,
            published_outcomes: 0,
        }
    }
}

impl Agent for PolicySchedulerAgent {
    fn name(&self) -> String {
        "cortex-policy-scheduler".into()
    }
    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut ticks = 0;
        while ctx.running() {
            let now = ctx.frames.latest().map(|frame| frame.id).unwrap_or(0);
            if let Some(policy) = self.policies.latest()
                && self.last_policy_frame != Some(policy.source_scene_frame_id)
            {
                self.last_policy_frame = Some(policy.source_scene_frame_id);
                self.scheduler.submit(policy, now);
            }
            for request_id in self.scheduler.scene_control.take_dispatched() {
                self.scheduler.dispatched(request_id);
            }
            for request_id in self.scheduler.depth_control.take_dispatched() {
                self.scheduler.dispatched(request_id);
            }
            if let Some(scene) = self.scenes.latest()
                && self.last_scene_frame != Some(scene.stamp.source_frame_id)
            {
                self.last_scene_frame = Some(scene.stamp.source_frame_id);
                let pending: Vec<_> = self
                    .scheduler
                    .pending
                    .iter()
                    .filter(|(_, request)| {
                        request.target == ObservationTarget::Scene
                            && self.scheduler.dispatched.contains(&request.request_id)
                            && scene.stamp.source_frame_id > request.source_scene_frame_id
                    })
                    .map(|(id, _)| *id)
                    .collect();
                for request_id in pending {
                    self.scheduler.resolve(
                        request_id,
                        scene.stamp.source_frame_id,
                        Resolution::Inconclusive,
                    );
                }
            }
            if let Some(depth) = self.depths.latest()
                && self.last_depth_frame != Some(depth.stamp.source_frame_id)
            {
                self.last_depth_frame = Some(depth.stamp.source_frame_id);
                let pending: Vec<_> = self
                    .scheduler
                    .pending
                    .iter()
                    .filter(|(id, request)| {
                        request.target == ObservationTarget::Depth
                            && self.scheduler.dispatched.contains(id)
                            && depth.stamp.source_frame_id > request.source_scene_frame_id
                    })
                    .map(|(id, _)| *id)
                    .collect();
                for request_id in pending {
                    self.scheduler.resolve(
                        request_id,
                        depth.stamp.source_frame_id,
                        Resolution::Inconclusive,
                    );
                }
            }
            self.scheduler.expire(now);
            if self.published_outcomes != self.scheduler.outcome_revision {
                self.published_outcomes = self.scheduler.outcome_revision;
                self.outcomes.publish(PolicyOutcomeFrame {
                    source_scene_frame_id: self.scheduler.last_policy_frame.unwrap_or(0),
                    outcomes: self.scheduler.outcomes.clone(),
                });
            }
            ticks += 1;
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        AgentHealth {
            name: self.name(),
            ticks,
            errors: 0,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SchedulerBudget {
    max_pending: usize,
    cooldown_frames: u64,
    global_min_interval_frames: u64,
    max_outcomes: usize,
}

impl Default for SchedulerBudget {
    fn default() -> Self {
        Self {
            max_pending: 2,
            cooldown_frames: 2,
            global_min_interval_frames: 1,
            max_outcomes: 128,
        }
    }
}

pub(crate) struct PolicyScheduler {
    scene_control: SceneExecutionControl,
    depth_control: DepthExecutionControl,
    budget: SchedulerBudget,
    pending: BTreeMap<u64, ObservationRequest>,
    dispatched: BTreeSet<u64>,
    cooldowns: Vec<(ObservationTarget, ReasonCode, u64)>,
    outcomes: Vec<(u64, RequestOutcome)>,
    last_policy_frame: Option<u64>,
    last_grant_frame: Option<u64>,
    outcome_revision: usize,
    scene_available: bool,
    depth_available: bool,
}

impl PolicyScheduler {
    pub(crate) fn with_controls(
        scene_control: SceneExecutionControl,
        depth_control: DepthExecutionControl,
        budget: SchedulerBudget,
    ) -> Self {
        Self {
            scene_control,
            depth_control,
            budget,
            pending: BTreeMap::new(),
            dispatched: BTreeSet::new(),
            cooldowns: Vec::new(),
            outcomes: Vec::new(),
            last_policy_frame: None,
            last_grant_frame: None,
            outcome_revision: 0,
            scene_available: true,
            depth_available: true,
        }
    }

    pub(crate) fn submit(&mut self, policy: AttentionPolicy, now_frame_id: u64) {
        self.last_policy_frame = Some(policy.source_scene_frame_id);
        self.cooldowns
            .retain(|(_, _, until)| *until >= now_frame_id);
        let mut requests = policy.requests;
        requests.sort_by(|left, right| {
            right
                .priority
                .cmp(&left.priority)
                .then_with(|| left.request_id.cmp(&right.request_id))
        });
        for request in requests {
            if self.pending.contains_key(&request.request_id) {
                continue;
            }
            if request.expires_at_frame < now_frame_id {
                self.push_outcome(request.request_id, RequestOutcome::Expired);
                continue;
            }
            if request.source_scene_frame_id != policy.source_scene_frame_id
                || request.max_extra_runs != 1
                || request.evidence.is_empty()
                || request
                    .evidence
                    .iter()
                    .any(|evidence| evidence.scene_frame_id > request.source_scene_frame_id)
            {
                self.push_outcome(
                    request.request_id,
                    RequestOutcome::Rejected {
                        reason: "invalid_request_contract".into(),
                    },
                );
                continue;
            }
            if matches!(request.target, ObservationTarget::ExternalAgent { .. }) {
                self.push_outcome(request.request_id, RequestOutcome::Unavailable);
                continue;
            }
            let expert_unavailable = match request.target {
                ObservationTarget::Scene => !self.scene_available,
                ObservationTarget::Depth => !self.depth_available,
                ObservationTarget::ExternalAgent { .. } => true,
            };
            if expert_unavailable {
                self.push_outcome(request.request_id, RequestOutcome::Unavailable);
                continue;
            }
            let semantic_duplicate = self.pending.values().any(|pending| {
                pending.target == request.target && pending.reason == request.reason
            }) || self.cooldowns.iter().any(|(target, reason, until)| {
                *target == request.target && *reason == request.reason && *until >= now_frame_id
            });
            if semantic_duplicate {
                self.push_outcome(
                    request.request_id,
                    RequestOutcome::Deferred {
                        retry_at_frame: now_frame_id.saturating_add(self.budget.cooldown_frames),
                        reason: "dedupe_cooldown".into(),
                    },
                );
                continue;
            }
            let target_busy = self
                .pending
                .values()
                .any(|pending| pending.target == request.target);
            if self.pending.len() >= self.budget.max_pending || target_busy {
                self.push_outcome(
                    request.request_id,
                    RequestOutcome::Deferred {
                        retry_at_frame: now_frame_id.saturating_add(1),
                        reason: "expert_budget".into(),
                    },
                );
                continue;
            }
            if self.last_grant_frame.is_some_and(|last| {
                now_frame_id.saturating_sub(last) < self.budget.global_min_interval_frames
            }) {
                self.push_outcome(
                    request.request_id,
                    RequestOutcome::Deferred {
                        retry_at_frame: now_frame_id
                            .saturating_add(self.budget.global_min_interval_frames),
                        reason: "global_inter_run_interval".into(),
                    },
                );
                continue;
            }
            match request.target {
                ObservationTarget::Scene => self.scene_control.grant(request.clone()),
                ObservationTarget::Depth => self.depth_control.grant(request.clone()),
                ObservationTarget::ExternalAgent { .. } => unreachable!("handled above"),
            }
            self.cooldowns.push((
                request.target.clone(),
                request.reason.clone(),
                now_frame_id.saturating_add(self.budget.cooldown_frames),
            ));
            self.last_grant_frame = Some(now_frame_id);
            self.push_outcome(
                request.request_id,
                RequestOutcome::Accepted {
                    dispatch_deadline_frame: request.expires_at_frame,
                },
            );
            self.pending.insert(request.request_id, request);
        }
    }

    pub(crate) fn dispatched(&mut self, request_id: u64) {
        if self.pending.contains_key(&request_id) && self.dispatched.insert(request_id) {
            self.push_outcome(request_id, RequestOutcome::Dispatched);
        }
    }

    pub(crate) fn resolve(&mut self, request_id: u64, source_frame_id: u64, outcome: Resolution) {
        if self.dispatched.remove(&request_id) && self.pending.remove(&request_id).is_some() {
            self.push_outcome(
                request_id,
                RequestOutcome::Resolved {
                    source_frame_id,
                    outcome,
                },
            );
        }
    }

    pub(crate) fn expire(&mut self, now_frame_id: u64) {
        let expired: Vec<_> = self
            .pending
            .iter()
            .filter(|(_, request)| request.expires_at_frame < now_frame_id)
            .map(|(id, _)| *id)
            .collect();
        for id in expired {
            self.pending.remove(&id);
            self.dispatched.remove(&id);
            self.push_outcome(id, RequestOutcome::Expired);
        }
    }

    fn push_outcome(&mut self, request_id: u64, outcome: RequestOutcome) {
        let capacity = self.budget.max_outcomes.max(1);
        if self.outcomes.len() == capacity {
            self.outcomes.remove(0);
        }
        self.outcomes.push((request_id, outcome));
        self.outcome_revision = self.outcome_revision.saturating_add(1);
    }

    #[cfg(test)]
    fn outcomes(&self) -> &[(u64, RequestOutcome)] {
        &self.outcomes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_cortex::{EvidenceRef, ReasonCode};
    use serde::Deserialize;

    fn request(id: u64, expires_at_frame: u64) -> ObservationRequest {
        ObservationRequest {
            request_id: id,
            source_scene_frame_id: 7,
            target: ObservationTarget::Scene,
            reason: ReasonCode::UnexpectedAppearance,
            evidence: vec![EvidenceRef {
                scene_frame_id: 7,
                entity: None,
                event_seq: Some(id),
            }],
            priority: 1,
            max_extra_runs: 1,
            expires_at_frame,
        }
    }

    fn scheduler(control: SceneExecutionControl) -> PolicyScheduler {
        PolicyScheduler::with_controls(
            control,
            DepthExecutionControl::default(),
            SchedulerBudget::default(),
        )
    }

    #[test]
    fn one_shot_permit_resolves_without_mutating_facts() {
        let control = SceneExecutionControl::default();
        let mut scheduler = scheduler(control.clone());
        scheduler.submit(
            AttentionPolicy {
                source_scene_frame_id: 7,
                requests: vec![request(9, 10)],
            },
            8,
        );
        assert_eq!(control.take_for(8).unwrap().request_id, 9);
        assert!(control.take_for(8).is_none());
        scheduler.dispatched(9);
        scheduler.resolve(9, 8, Resolution::Inconclusive);
        assert!(matches!(
            scheduler.outcomes().last(),
            Some((
                9,
                RequestOutcome::Resolved {
                    source_frame_id: 8,
                    outcome: Resolution::Inconclusive
                }
            ))
        ));
    }

    #[test]
    fn accepted_is_not_resolved_until_the_permit_is_dispatched() {
        let control = SceneExecutionControl::default();
        let mut scheduler = scheduler(control.clone());
        scheduler.submit(
            AttentionPolicy {
                source_scene_frame_id: 7,
                requests: vec![request(10, 12)],
            },
            8,
        );

        scheduler.resolve(10, 8, Resolution::Supported);
        assert!(scheduler.pending.contains_key(&10));
        assert!(
            !scheduler
                .outcomes()
                .iter()
                .any(|(_, outcome)| matches!(outcome, RequestOutcome::Resolved { .. }))
        );

        assert_eq!(control.take_for(8).unwrap().request_id, 10);
        scheduler.dispatched(10);
        scheduler.resolve(10, 8, Resolution::Supported);
        assert!(!scheduler.pending.contains_key(&10));
    }

    #[test]
    fn replaying_the_same_request_id_does_not_duplicate_outcomes() {
        let control = SceneExecutionControl::default();
        let mut scheduler = scheduler(control);
        let policy = AttentionPolicy {
            source_scene_frame_id: 7,
            requests: vec![request(11, 12)],
        };

        scheduler.submit(policy.clone(), 8);
        let outcome_count = scheduler.outcomes().len();
        scheduler.submit(policy, 8);

        assert_eq!(scheduler.outcomes().len(), outcome_count);
        assert_eq!(scheduler.pending.len(), 1);
    }

    #[test]
    fn expired_or_budget_limited_requests_are_explicit() {
        let control = SceneExecutionControl::default();
        let mut scheduler = scheduler(control);
        scheduler.submit(
            AttentionPolicy {
                source_scene_frame_id: 7,
                requests: vec![request(1, 7), request(2, 10), request(3, 10)],
            },
            8,
        );
        assert!(matches!(scheduler.outcomes()[0].1, RequestOutcome::Expired));
        assert!(matches!(
            scheduler.outcomes()[2].1,
            RequestOutcome::Deferred { .. }
        ));
    }

    #[test]
    fn semantic_dedupe_cooldown_and_depth_budget_are_bounded() {
        let scene_control = SceneExecutionControl::default();
        let depth_control = DepthExecutionControl::default();
        let mut scheduler = PolicyScheduler::with_controls(
            scene_control.clone(),
            depth_control.clone(),
            SchedulerBudget {
                global_min_interval_frames: 0,
                ..SchedulerBudget::default()
            },
        );
        let mut duplicate = request(20, 20);
        duplicate.request_id = 21;
        let mut depth = request(22, 20);
        depth.target = ObservationTarget::Depth;

        scheduler.submit(
            AttentionPolicy {
                source_scene_frame_id: 7,
                requests: vec![request(20, 20), duplicate, depth],
            },
            8,
        );

        assert_eq!(scene_control.take_for(8).unwrap().request_id, 20);
        assert_eq!(depth_control.take_for(8).unwrap().request_id, 22);
        assert_eq!(scheduler.pending.len(), 2);
        assert!(scheduler.outcomes().iter().any(|(id, outcome)| {
            *id == 21 && matches!(outcome, RequestOutcome::Deferred { .. })
        }));
    }

    #[test]
    fn external_request_is_explicitly_unavailable_without_side_effect() {
        let control = SceneExecutionControl::default();
        let mut scheduler = scheduler(control);
        let mut external = request(30, 20);
        external.target = ObservationTarget::ExternalAgent {
            question: "confirm camera motion".into(),
        };

        scheduler.submit(
            AttentionPolicy {
                source_scene_frame_id: 7,
                requests: vec![external],
            },
            8,
        );

        assert!(matches!(
            scheduler.outcomes().last(),
            Some((30, RequestOutcome::Unavailable))
        ));
    }

    #[test]
    fn missing_depth_expert_is_unavailable_instead_of_silent_expiry() {
        let control = SceneExecutionControl::default();
        let mut scheduler = scheduler(control);
        scheduler.depth_available = false;
        let mut depth = request(31, 20);
        depth.target = ObservationTarget::Depth;

        scheduler.submit(
            AttentionPolicy {
                source_scene_frame_id: 7,
                requests: vec![depth],
            },
            8,
        );

        assert!(matches!(
            scheduler.outcomes().last(),
            Some((31, RequestOutcome::Unavailable))
        ));
        assert!(scheduler.pending.is_empty());
    }

    #[derive(Deserialize)]
    struct PolicyFixture {
        cases: Vec<PolicyCase>,
    }

    #[derive(Deserialize)]
    struct PolicyCase {
        action: String,
        expected: String,
    }

    #[test]
    fn active_reobservation_lifecycle_matches_synthetic_replay_contract() {
        let fixture: PolicyFixture =
            serde_json::from_str(include_str!("../../../../benchmark/scenarios/019.json")).unwrap();

        for (index, case) in fixture.cases.iter().enumerate() {
            let scene_control = SceneExecutionControl::default();
            let depth_control = DepthExecutionControl::default();
            let mut scheduler = PolicyScheduler::with_controls(
                scene_control.clone(),
                depth_control.clone(),
                SchedulerBudget {
                    global_min_interval_frames: 0,
                    ..SchedulerBudget::default()
                },
            );
            let id = u64::try_from(index).unwrap().saturating_add(100);
            let mut primary = request(id, 12);
            match case.action.as_str() {
                "duplicate" => {
                    let mut duplicate = primary.clone();
                    duplicate.request_id = id + 1;
                    scheduler.submit(
                        AttentionPolicy {
                            source_scene_frame_id: 7,
                            requests: vec![primary, duplicate],
                        },
                        8,
                    );
                }
                "expired" => {
                    primary.expires_at_frame = 7;
                    scheduler.submit(
                        AttentionPolicy {
                            source_scene_frame_id: 7,
                            requests: vec![primary],
                        },
                        8,
                    );
                }
                "depth" => {
                    primary.target = ObservationTarget::Depth;
                    scheduler.submit(
                        AttentionPolicy {
                            source_scene_frame_id: 7,
                            requests: vec![primary],
                        },
                        8,
                    );
                    let dispatched = depth_control.take_for(8).unwrap().request_id;
                    scheduler.dispatched(dispatched);
                }
                "resolve" => {
                    scheduler.submit(
                        AttentionPolicy {
                            source_scene_frame_id: 7,
                            requests: vec![primary],
                        },
                        8,
                    );
                    let dispatched = scene_control.take_for(8).unwrap().request_id;
                    scheduler.dispatched(dispatched);
                    scheduler.resolve(dispatched, 8, Resolution::Inconclusive);
                }
                "external" => {
                    primary.target = ObservationTarget::ExternalAgent {
                        question: "confirm observation".into(),
                    };
                    scheduler.submit(
                        AttentionPolicy {
                            source_scene_frame_id: 7,
                            requests: vec![primary],
                        },
                        8,
                    );
                }
                "overload" => {
                    let mut competing = primary.clone();
                    competing.request_id = id + 1;
                    competing.reason = ReasonCode::HighUncertaintySceneSource;
                    scheduler.submit(
                        AttentionPolicy {
                            source_scene_frame_id: 7,
                            requests: vec![primary, competing],
                        },
                        8,
                    );
                }
                action => panic!("unknown fixture action: {action}"),
            }

            let matched = scheduler.outcomes().iter().any(|(_, outcome)| {
                matches!(
                    (case.expected.as_str(), outcome),
                    ("deferred", RequestOutcome::Deferred { .. })
                        | ("expired", RequestOutcome::Expired)
                        | ("dispatched", RequestOutcome::Dispatched)
                        | ("resolved", RequestOutcome::Resolved { .. })
                        | ("unavailable", RequestOutcome::Unavailable)
                )
            });
            assert!(
                matched,
                "fixture action {} did not produce {}",
                case.action, case.expected
            );
        }
    }
}
