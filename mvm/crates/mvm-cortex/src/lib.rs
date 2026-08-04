//! `mvm-cortex` — cognitive interpretation over immutable fused scene facts.
//!
//! Phase 18.2 deliberately provides only the stable skeleton. Fusion remains
//! the sole authority for observed scene facts; this crate neither invokes
//! experts nor mutates [`mvm_core::canonical::FusedSceneFrame`].

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
    AttentionPolicy, ClaimStatus, CognitiveClaim, CortexConfig, CortexFrame, CortexInput,
    EvidenceRef, ObservationRequest, ObservationTarget, ReasonCode, RequestOutcome, Resolution,
};
