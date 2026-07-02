//! Module trait — the contract every MVM module must fulfill.
//!
//! Each module (detect, identity, emotion, posture, fusion) implements this
//! trait. The pipeline knows only the trait, not the concrete implementation:
//! swapping SFace for ArcFace means changing one crate, not the pipeline.

use crate::config::Config;

/// Metadata about a module.
#[derive(Debug, Clone)]
pub struct ModuleMetadata {
    pub name: &'static str,
    pub version: &'static str,
    /// Expected accuracy on standard benchmark (0.0–1.0), if known.
    pub accuracy: Option<f32>,
}

/// The contract every MVM module must fulfill.
///
/// Type parameters:
/// - `Input`: what the module receives (e.g. image frame, crop, landmarks).
/// - `Output`: what the module produces (e.g. `Detection`, `Vec<f32>`).
///
/// # Design
///
/// - `init()` is called once at startup. It receives the global `Config`.
/// - `process()` is called per frame. It must be fast and allocation-aware.
/// - `metadata()` returns name/version/accuracy for introspection.
pub trait Module {
    type Input;
    type Output;

    fn init(config: &Config) -> Result<Self, anyhow::Error>
    where
        Self: Sized;

    fn process(&mut self, input: &Self::Input) -> Result<Self::Output, anyhow::Error>;

    fn metadata(&self) -> ModuleMetadata;
}
