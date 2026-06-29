//! Stub agents for wiring/validating the mesh without the real detector chain
//! (phase 8.C). `StubCaptureAgent` is a synthetic producer; `FrameCounterAgent` a
//! minimal consumer. The real `HumanAgent` (wrapping the body-primary chain) and
//! `CaptureAgent` (wrapping `mvm-pipeline::NativeCamera`) come in a later step.

use crate::agent::{Agent, AgentHealth, MeshContext};
use crate::frame::Frame;
use image::DynamicImage;
use std::time::{Duration, Instant};

/// Synthetic capture: publishes `max_frames` blank frames at `fps`, then signals
/// the mesh to stop. Models the camera producer for tests/`mesh --stub`.
pub struct StubCaptureAgent {
    fps: f32,
    max_frames: u64,
}

impl StubCaptureAgent {
    pub fn new(fps: f32, max_frames: u64) -> Self {
        Self { fps, max_frames }
    }
}

impl Agent for StubCaptureAgent {
    fn name(&self) -> String {
        "stub-capture".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let period = Duration::from_secs_f32(1.0 / self.fps.max(1.0));
        let mut ticks = 0u64;
        for id in 0..self.max_frames {
            if !ctx.running() {
                break;
            }
            ctx.frames.publish(Frame::new(id, DynamicImage::new_rgb8(16, 16), Instant::now()));
            ticks += 1;
            std::thread::sleep(period);
        }
        // Frame budget exhausted → tell the consumers to wind down.
        ctx.request_stop();
        AgentHealth { name: self.name(), ticks, errors: 0 }
    }
}

/// Minimal consumer: tracks the freshest frame id into `WorldState.frame_id` and
/// counts distinct frames it actually processed (proves latest-wins + world write).
#[derive(Default)]
pub struct FrameCounterAgent {
    poll: Option<Duration>,
}

impl FrameCounterAgent {
    pub fn new() -> Self {
        Self { poll: None }
    }
    /// Override poll interval (default 1ms).
    pub fn with_poll(poll: Duration) -> Self {
        Self { poll: Some(poll) }
    }
}

impl Agent for FrameCounterAgent {
    fn name(&self) -> String {
        "frame-counter".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let poll = self.poll.unwrap_or(Duration::from_millis(1));
        let mut ticks = 0u64;
        let mut last_id: Option<u64> = None;
        while ctx.running() {
            if let Some(f) = ctx.frames.latest() {
                if last_id != Some(f.id) {
                    last_id = Some(f.id);
                    ticks += 1;
                    if let Ok(mut w) = ctx.world.write() {
                        w.frame_id = f.id;
                        w.timestamp = f.captured;
                    }
                }
            }
            std::thread::sleep(poll);
        }
        AgentHealth { name: self.name(), ticks, errors: 0 }
    }
}
