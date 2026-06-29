//! mvm-mesh — the Agent Mesh threading layer (phase 8.C).
//!
//! A `std::thread`-per-agent runtime over a single-slot latest-frame channel and a
//! shared `WorldState` blackboard (from `mvm-core`). Decision 8.0.1 (locked by the
//! spikes in `spikes/mesh-spike/`): **std::thread, not tokio**; latest-value frame
//! channel (not broadcast); fusion via `build_scene`, not MBT. The mesh exists to
//! remove head-of-line blocking once heavy detectors (depth ~230ms, hands) landed.
//!
//! This is the skeleton: `Agent` trait, `Frame`/`LatestFrame`, stub producer/consumer,
//! and the `AgentMesh` orchestrator. The real `HumanAgent` (wrapping the body-primary
//! chain), `CaptureAgent` (wrapping `NativeCamera`), object/depth/audio agents, and a
//! `StateFileWriter` come next — each is just another `Agent`.

pub mod agent;
pub mod frame;
pub mod mesh;
pub mod stubs;

pub use agent::{Agent, AgentError, AgentHealth, MeshContext};
pub use frame::{Frame, LatestFrame};
pub use mesh::AgentMesh;
pub use stubs::{FrameCounterAgent, StubCaptureAgent};
