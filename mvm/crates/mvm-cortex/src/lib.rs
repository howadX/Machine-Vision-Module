//! `mvm-cortex` — cognitive interpretation over immutable fused scene facts.
//!
//! Fusion remains the sole authority for observed scene facts. Cortex adds
//! evidence-linked interpretation, bounded runtime memory, forecast,
//! prediction error, observer self-model and explicit uncertainty; it neither
//! invokes experts nor mutates
//! [`mvm_core::canonical::FusedSceneFrame`].

mod affordance;
mod attention;
mod cortex;
mod events;
mod memory;
mod predict;
mod report;
mod self_model;
mod types;
mod uncertainty;

pub use cortex::VisualCortex;
pub use types::{
    ActivityForecast, AffordanceKind, AffordanceResult, AttentionPolicy, AxisDirection,
    CalibrationBasis, CalibrationReport, CalibrationValue, CameraPose, ClaimStatus, CognitiveClaim,
    CoordinateFrame, CortexConfig, CortexFrame, CortexInput, EntityForecast, EvidenceRef,
    ForecastPosition3D, MemoryConfig, MemoryContext, MemoryEvent, ObservationRequest,
    ObservationTarget, PredictionError, PredictionErrorKind, ReasonCode, RelationForecast,
    RequestOutcome, Resolution, SceneForecast, SpatialSelf, UncertainClaim, UncertaintyCode,
    UncertaintyReport, VisibilityState, VisualSelfModel, ZoneVisibility,
};
