//! `mvm-cortex` — cognitive interpretation over immutable fused scene facts.
//!
//! Fusion remains the sole authority for observed scene facts. Cortex adds
//! evidence-linked interpretation, bounded runtime memory, forecast and
//! prediction error; it neither invokes experts nor mutates
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
    ActivityForecast, AttentionPolicy, ClaimStatus, CognitiveClaim, CortexConfig, CortexFrame,
    CortexInput, EntityForecast, EvidenceRef, ForecastPosition3D, MemoryConfig, MemoryContext,
    MemoryEvent, ObservationRequest, ObservationTarget, PredictionError, PredictionErrorKind,
    ReasonCode, RelationForecast, RequestOutcome, Resolution, SceneForecast,
};
