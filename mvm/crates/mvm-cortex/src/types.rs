use mvm_core::{
    BBox, Config,
    canonical::{EntityKey, FusedSceneFrame, SceneEvent, SceneEventKind},
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
    CoastedSource,
    LowConfidence,
    Occluded,
    Uncalibrated,
    EventGap,
    OrdinalOnly,
    Skeleton,
    UnexpectedAppearance,
    HighUncertaintySceneSource,
    RememberedSource,
    WeakSource,
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
    Depth,
    ExternalAgent { question: String },
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
    Rejected {
        reason: String,
    },
    Dispatched,
    Resolved {
        source_frame_id: u64,
        outcome: Resolution,
    },
    Expired,
    Unavailable,
}

/// Latest-wins operational status of bounded re-observation requests. This is
/// cognitive runtime state, not a canonical scene fact or durable history.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct PolicyOutcomeFrame {
    pub source_scene_frame_id: u64,
    pub outcomes: Vec<(u64, RequestOutcome)>,
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
    /// Consumer-local gap in the bounded canonical event tail. A gap is not a
    /// scene fact, so it travels with this Cortex read instead of mutating the
    /// immutable `FusedSceneFrame`.
    pub scene_event_gap: bool,
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
    /// One-second cognitive projection derived from the current immutable scene.
    /// It is never an observed fact and may contain withheld component claims.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forecast: Option<SceneForecast>,
    /// Bounded forecast set produced by the same predictor. `forecast` remains
    /// the one-second compatibility projection; this field makes every
    /// published horizon explicit instead of letting a report pretend that a
    /// requested horizon was computed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub forecasts: Vec<SceneForecast>,
    /// Differences between the newest due forecast and the current observation.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub prediction_errors: Vec<PredictionError>,
    /// Bounded runtime-only episodic context. It is a cognitive view, never a
    /// second durable event log or an observed scene fact.
    #[serde(default)]
    pub memory: MemoryContext,
    /// Observer-relative spatial limits and calibration honesty for this tick.
    #[serde(default)]
    pub self_model: VisualSelfModel,
    /// Deterministic inventory of claims the observer cannot safely assert.
    #[serde(default)]
    pub uncertainty: UncertaintyReport,
    /// Evidence-backed action possibilities, never physical commands or scene facts.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub affordances: Vec<AffordanceResult>,
    /// Bounded read-only salience ranking over this cognitive tick's inputs.
    #[serde(default)]
    pub attention: AttentionMap,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AffordanceKind {
    Reachable,
    Held,
    LikelyUsed,
    Attended,
    Blocked,
    Approaching,
    Leaving,
    Sittable,
    Interactable,
}

/// Cognitive interpretation of one potential interaction. `target`/`actor`
/// identify the canonical entities when the available evidence permits it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AffordanceResult {
    pub kind: AffordanceKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<EntityKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub actor: Option<EntityKey>,
    pub claim: CognitiveClaim<bool>,
}

impl AffordanceResult {
    pub fn validate(&self) -> Result<(), String> {
        self.claim.validate()
    }
}

/// The factual subject currently ranked by Cortex. A target is a reference to
/// the immutable scene, not a new scene fact or an execution instruction.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionTarget {
    Scene,
    Entity(EntityKey),
    Zone(u64),
}

/// Interpretable normalized salience inputs. Every component is finite and in
/// `[0, 1]`; a zero means that this tick has no evidence for that signal.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct AttentionBreakdown {
    pub novelty: f32,
    pub prediction_error: f32,
    pub uncertainty: f32,
    pub event_recent: f32,
    pub activity: f32,
    pub anomaly: f32,
    pub identity: f32,
    pub affordance: f32,
    pub zone_violation: f32,
}

/// One evidence-backed ranking result. It is a priority hint only: it never
/// promotes a hypothesis to an observed fact and has no actuator authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AttentionScore {
    pub target: AttentionTarget,
    pub score: f32,
    pub breakdown: AttentionBreakdown,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

impl AttentionScore {
    pub fn validate(&self) -> Result<(), String> {
        if !self.score.is_finite() || !(0.0..=1.0).contains(&self.score) {
            return Err("attention score must be finite and within [0,1]".into());
        }
        let components = [
            self.breakdown.novelty,
            self.breakdown.prediction_error,
            self.breakdown.uncertainty,
            self.breakdown.event_recent,
            self.breakdown.activity,
            self.breakdown.anomaly,
            self.breakdown.identity,
            self.breakdown.affordance,
            self.breakdown.zone_violation,
        ];
        if components
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        {
            return Err("attention breakdown must be finite and within [0,1]".into());
        }
        if self.score > 0.0 && self.evidence.is_empty() {
            return Err("non-zero attention score requires evidence".into());
        }
        Ok(())
    }
}

/// Bounded read-only focus map for one immutable fused scene revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct AttentionMap {
    pub source_scene_frame_id: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub top_k: Vec<AttentionScore>,
}

impl AttentionMap {
    pub fn validate(&self) -> Result<(), String> {
        let mut previous: Option<&AttentionScore> = None;
        for score in &self.top_k {
            score.validate()?;
            if let Some(previous) = previous
                && (previous.score < score.score
                    || (previous.score == score.score && previous.target > score.target))
            {
                return Err("attention scores must be deterministically sorted".into());
            }
            previous = Some(score);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CoordinateFrame {
    #[default]
    CameraRelative,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AxisDirection {
    Right,
    Down,
    Forward,
}

/// Coordinate convention of the current MVM spatial substrate.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpatialSelf {
    pub coordinate_frame: CoordinateFrame,
    pub x_axis: AxisDirection,
    pub y_axis: AxisDirection,
    pub z_axis: AxisDirection,
}

impl Default for SpatialSelf {
    fn default() -> Self {
        Self {
            coordinate_frame: CoordinateFrame::CameraRelative,
            x_axis: AxisDirection::Right,
            y_axis: AxisDirection::Down,
            z_axis: AxisDirection::Forward,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CameraPose {
    pub reference_frame: CoordinateFrame,
    pub position_m: [f32; 3],
    pub orientation_rpy_degrees: [f32; 3],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisibilityState {
    /// All currently registered members have usable image evidence. This does
    /// not claim that the zone's unmodelled spatial extent is fully visible.
    ObservedMembers,
    PartiallyObserved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ZoneVisibility {
    pub zone_id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub visibility: CognitiveClaim<VisibilityState>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationBasis {
    ConfiguredPrior,
    UnverifiedPrior,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationValue {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub value: Option<f32>,
    pub basis: CalibrationBasis,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationReport {
    pub fx_ratio: CalibrationValue,
    pub torso_length_m: CalibrationValue,
    pub camera_height_m: CalibrationValue,
    pub world_extrinsics: CalibrationBasis,
}

impl Default for CalibrationReport {
    fn default() -> Self {
        Self {
            fx_ratio: CalibrationValue {
                value: None,
                basis: CalibrationBasis::Unavailable,
            },
            torso_length_m: CalibrationValue {
                value: None,
                basis: CalibrationBasis::Unavailable,
            },
            camera_height_m: CalibrationValue {
                value: None,
                basis: CalibrationBasis::Unavailable,
            },
            world_extrinsics: CalibrationBasis::Unavailable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VisualSelfModel {
    pub source_scene_frame_id: u64,
    pub spatial: SpatialSelf,
    pub camera_relative_pose: CognitiveClaim<CameraPose>,
    pub world_pose: CognitiveClaim<CameraPose>,
    pub zones: Vec<ZoneVisibility>,
    pub calibration: CalibrationReport,
}

impl Default for VisualSelfModel {
    fn default() -> Self {
        Self {
            source_scene_frame_id: 0,
            spatial: SpatialSelf::default(),
            camera_relative_pose: CognitiveClaim::unknown(ReasonCode::MissingSource),
            world_pose: CognitiveClaim::unknown(ReasonCode::MissingSource),
            zones: Vec::new(),
            calibration: CalibrationReport::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UncertaintyCode {
    Posture,
    EmotionDistribution,
    Identity,
    Existence,
    SourceFreshness,
    MissingExpert,
    EventDelivery,
    EntityMetricPosition,
    MetricCalibration,
    ZoneVisibility,
    CameraWorldPose,
    SceneFactInventory,
}

/// One evidence-linked item in the answer to "what do you not know?".
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertainClaim {
    pub code: UncertaintyCode,
    pub status: ClaimStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<EntityKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub zone_id: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<mvm_core::canonical::FactSource>,
    pub reasons: Vec<ReasonCode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub evidence: Vec<EvidenceRef>,
}

impl UncertainClaim {
    pub fn validate(&self) -> Result<(), String> {
        if self.status == ClaimStatus::Asserted {
            return Err("an uncertain claim cannot be asserted".into());
        }
        if self.reasons.is_empty() {
            return Err("an uncertain claim requires an explicit reason".into());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct UncertaintyReport {
    pub source_scene_frame_id: u64,
    pub claims: Vec<UncertainClaim>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ForecastPosition3D {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityForecast {
    pub key: EntityKey,
    pub existence: CognitiveClaim<mvm_core::canonical::Existence>,
    pub bbox: CognitiveClaim<BBox>,
    pub position_3d: CognitiveClaim<ForecastPosition3D>,
    pub posture: CognitiveClaim<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActivityForecast {
    pub label: String,
    pub participants: Vec<EntityKey>,
    pub persistence: CognitiveClaim<bool>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RelationForecast {
    pub a: EntityKey,
    pub b: EntityKey,
    pub kind: mvm_core::canonical::SceneRelationKind,
    pub persistence: CognitiveClaim<bool>,
}

/// Bounded forecast produced from one canonical scene revision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneForecast {
    pub source_scene_frame_id: u64,
    pub source_timestamp_unix_ms: i64,
    pub target_timestamp_unix_ms: i64,
    pub horizon_ms: i64,
    pub entities: Vec<EntityForecast>,
    pub activities: Vec<ActivityForecast>,
    pub relations: Vec<RelationForecast>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PredictionErrorKind {
    BboxDrift,
    Position3dDrift,
    EntityAppeared,
    EntityDisappeared,
    ActivityChanged,
    RelationBroken,
    PostureChanged,
}

/// Evidence-linked mismatch between a due cognitive forecast and observation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PredictionError {
    pub kind: PredictionErrorKind,
    pub forecast_source_scene_frame_id: u64,
    /// Exact horizon of the forecast that became due. This keeps a two-second
    /// mismatch distinguishable from a one-second mismatch on the same tick.
    #[serde(default)]
    pub forecast_horizon_ms: i64,
    pub observed_scene_frame_id: u64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<EntityKey>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub magnitude: Option<f32>,
    pub confidence: f32,
    pub evidence: Vec<EvidenceRef>,
}

/// Configuration for bounded in-process event memory. This configuration is
/// deliberately independent from persistent MVM configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub capacity: usize,
    pub recent_limit: usize,
    pub relevant_limit: usize,
    pub time_window_ms: i64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 128,
            recent_limit: 5,
            relevant_limit: 5,
            time_window_ms: 30_000,
        }
    }
}

/// An evidence-preserving runtime projection of one canonical scene event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryEvent {
    pub event_seq: u64,
    pub source_frame_id: u64,
    pub source_timestamp_unix_ms: i64,
    pub kind: SceneEventKind,
    pub entities: Vec<EntityKey>,
    pub confidence: f32,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub object_classes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contact_duration_ms: Option<u64>,
}

/// Compact bounded context emitted for a single cognitive tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MemoryContext {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub recent: Vec<MemoryEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub relevant: Vec<MemoryEvent>,
    pub summary: String,
}
