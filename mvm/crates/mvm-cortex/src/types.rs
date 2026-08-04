use mvm_core::{
    Config,
    canonical::{EntityKey, FusedSceneFrame, SceneEvent},
};
use serde::{Deserialize, Serialize};

/// Runtime configuration for the cognitive layer. It defaults off so adding the
/// skeleton cannot change existing `mvm run` behaviour.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct CortexConfig {
    pub enabled: bool,
}

/// Honesty status for an interpretation. Unlike `SceneFact`, this describes a
/// cognitive claim, not an observed world fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClaimStatus {
    Asserted,
    Withheld,
    Unknown,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReasonCode {
    InsufficientEvidence,
    MissingSource,
    StaleSource,
    Skeleton,
    UnexpectedAppearance,
    HighUncertaintySceneSource,
    Other(String),
}

/// A bounded, advisory request for one additional observation. It never
/// changes canonical facts and expires without implying confirmation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionPolicy {
    pub source_scene_frame_id: u64,
    pub requests: Vec<ObservationRequest>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationRequest {
    pub request_id: u64,
    pub source_scene_frame_id: u64,
    pub target: ObservationTarget,
    pub reason: ReasonCode,
    pub evidence: Vec<EvidenceRef>,
    pub priority: u8,
    pub max_extra_runs: u8,
    pub expires_at_frame: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObservationTarget {
    Scene,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Resolution {
    Supported,
    Contradicted,
    Inconclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RequestOutcome {
    Proposed,
    Accepted {
        dispatch_deadline_frame: u64,
    },
    Deferred {
        retry_at_frame: u64,
        reason: String,
    },
    Dispatched,
    Resolved {
        source_frame_id: u64,
        outcome: Resolution,
    },
    Expired,
}

/// Reference to canonical evidence supporting a cognitive claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    pub scene_frame_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<EntityKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event_seq: Option<u64>,
}

/// Evidence-linked interpretation. Missing claims never carry a value or a
/// fabricated confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveClaim<T> {
    pub status: ClaimStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reasons: Vec<ReasonCode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

impl<T> CognitiveClaim<T> {
    pub fn asserted(value: T, confidence: f32, evidence: Vec<EvidenceRef>) -> Result<Self, String> {
        let claim = Self {
            status: ClaimStatus::Asserted,
            value: Some(value),
            confidence: Some(confidence),
            reasons: Vec::new(),
            evidence,
        };
        claim.validate()?;
        Ok(claim)
    }

    pub fn withheld(reason: ReasonCode, evidence: Vec<EvidenceRef>) -> Self {
        Self {
            status: ClaimStatus::Withheld,
            value: None,
            confidence: None,
            reasons: vec![reason],
            evidence,
        }
    }

    pub fn unknown(reason: ReasonCode) -> Self {
        Self {
            status: ClaimStatus::Unknown,
            value: None,
            confidence: None,
            reasons: vec![reason],
            evidence: Vec::new(),
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        match self.status {
            ClaimStatus::Asserted => {
                if self.value.is_none() || self.confidence.is_none() || self.evidence.is_empty() {
                    return Err(
                        "asserted cognitive claim requires value, confidence, and evidence".into(),
                    );
                }
                let confidence = self.confidence.expect("checked above");
                if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
                    return Err("cognitive claim confidence must be finite and within [0,1]".into());
                }
            }
            ClaimStatus::Withheld | ClaimStatus::Unknown => {
                if self.value.is_some() || self.confidence.is_some() {
                    return Err(
                        "withheld/unknown cognitive claim must not carry value or confidence"
                            .into(),
                    );
                }
            }
        }
        Ok(())
    }
}

/// Immutable input for one Cortex tick. `scene_events` is a consumer-owned,
/// non-destructive delta from Fusion's event cursor.
pub struct CortexInput<'a> {
    pub previous_scene: Option<&'a FusedSceneFrame>,
    pub current_scene: &'a FusedSceneFrame,
    pub scene_events: &'a [SceneEvent],
    pub config: &'a Config,
    pub now_unix_ms: i64,
}

/// A serializable cognitive view. Phase 18.2 publishes no cognitive claims yet;
/// the empty `claims` vector is intentional and schema-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CortexFrame {
    pub timestamp_unix_ms: i64,
    pub scene_frame_id: u64,
    #[serde(default)]
    pub claims: Vec<CognitiveClaim<String>>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub scene_event_refs: Vec<EvidenceRef>,
}
