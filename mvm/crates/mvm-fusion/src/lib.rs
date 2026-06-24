//! mvm-fusion — logical fusion layer (Phase 3 stub).

use mvm_core::{Config, Detection, Module, ModuleMetadata};

pub struct FusionModule;

impl Module for FusionModule {
    type Input = Vec<Detection>;
    type Output = Vec<Detection>;

    fn init(_config: &Config) -> anyhow::Result<Self> { Ok(Self) }

    fn process(&mut self, input: &Self::Input) -> anyhow::Result<Self::Output> {
        Ok(input.clone())
    }

    fn metadata(&self) -> ModuleMetadata {
        ModuleMetadata { name: "mvm-fusion (stub)", version: env!("CARGO_PKG_VERSION"), accuracy: None }
    }
}
