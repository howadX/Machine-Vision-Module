//! CLI-owned bounded execution control for the 18.3b Scene re-observation slice.
//!
//! This module deliberately knows about `Scene` while `mvm-mesh` remains a
//! generic executor. A permit is one-shot, frame-bounded and cannot rewrite
//! persistent cadence or canonical Fusion state.

use super::{AttentionPolicyBus, SceneObservationBus};
use mvm_cortex::{
    AttentionPolicy, ObservationRequest, ObservationTarget, RequestOutcome, Resolution,
};
use mvm_mesh::{Agent, AgentHealth, MeshContext};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct ControlState {
    permit: Option<ObservationRequest>,
    dispatched: Vec<u64>,
}

#[derive(Clone, Default)]
pub(crate) struct SceneExecutionControl(Arc<Mutex<ControlState>>);

impl SceneExecutionControl {
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

pub(crate) struct PolicySchedulerAgent {
    policies: AttentionPolicyBus,
    scenes: SceneObservationBus,
    scheduler: PolicyScheduler,
    last_policy_frame: Option<u64>,
    last_scene_frame: Option<u64>,
}

impl PolicySchedulerAgent {
    pub(crate) fn new(
        policies: AttentionPolicyBus,
        scenes: SceneObservationBus,
        control: SceneExecutionControl,
    ) -> Self {
        Self {
            policies,
            scenes,
            scheduler: PolicyScheduler::new(control),
            last_policy_frame: None,
            last_scene_frame: None,
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
            for request_id in self.scheduler.control.take_dispatched() {
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
                        scene.stamp.source_frame_id > request.source_scene_frame_id
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
            self.scheduler.expire(now);
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

pub(crate) struct PolicyScheduler {
    control: SceneExecutionControl,
    pending: BTreeMap<u64, ObservationRequest>,
    outcomes: Vec<(u64, RequestOutcome)>,
}

impl PolicyScheduler {
    pub(crate) fn new(control: SceneExecutionControl) -> Self {
        Self {
            control,
            pending: BTreeMap::new(),
            outcomes: Vec::new(),
        }
    }

    pub(crate) fn submit(&mut self, policy: AttentionPolicy, now_frame_id: u64) {
        for request in policy.requests {
            if self.pending.contains_key(&request.request_id) {
                continue;
            }
            if request.expires_at_frame < now_frame_id {
                self.outcomes
                    .push((request.request_id, RequestOutcome::Expired));
            } else if request.target != ObservationTarget::Scene
                || request.max_extra_runs != 1
                || !self.pending.is_empty()
            {
                self.outcomes.push((
                    request.request_id,
                    RequestOutcome::Deferred {
                        retry_at_frame: now_frame_id.saturating_add(1),
                        reason: "scene_budget".into(),
                    },
                ));
            } else {
                self.control.grant(request.clone());
                self.outcomes.push((
                    request.request_id,
                    RequestOutcome::Accepted {
                        dispatch_deadline_frame: request.expires_at_frame,
                    },
                ));
                self.pending.insert(request.request_id, request);
            }
        }
    }

    pub(crate) fn dispatched(&mut self, request_id: u64) {
        self.outcomes.push((request_id, RequestOutcome::Dispatched));
    }

    pub(crate) fn resolve(&mut self, request_id: u64, source_frame_id: u64, outcome: Resolution) {
        if self.pending.remove(&request_id).is_some() {
            self.outcomes.push((
                request_id,
                RequestOutcome::Resolved {
                    source_frame_id,
                    outcome,
                },
            ));
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
            self.outcomes.push((id, RequestOutcome::Expired));
        }
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

    #[test]
    fn one_shot_permit_resolves_without_mutating_facts() {
        let control = SceneExecutionControl::default();
        let mut scheduler = PolicyScheduler::new(control.clone());
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
    fn expired_or_budget_limited_requests_are_explicit() {
        let control = SceneExecutionControl::default();
        let mut scheduler = PolicyScheduler::new(control);
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
}
