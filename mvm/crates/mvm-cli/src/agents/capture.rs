//! CaptureAgent (phase 8.C.5) — the mesh producer. Wraps a frame source
//! (`NativeCamera` in production) and publishes each frame to the latest-frame
//! slot at the camera's own tempo. When its optional frame budget is exhausted
//! (or the source dies), it signals the mesh to stop — it is the driver.

use mvm_mesh::{Agent, AgentHealth, Frame, MeshContext};
use std::time::Instant;

/// A source of frames the `CaptureAgent` pulls from. `NativeCamera` implements
/// it directly; tests use a synthetic source (so the agent is testable without
/// hardware).
pub(crate) trait FrameSource: Send {
    fn read(&mut self) -> anyhow::Result<image::DynamicImage>;
}

impl FrameSource for mvm_pipeline::native::NativeCamera {
    fn read(&mut self) -> anyhow::Result<image::DynamicImage> {
        mvm_pipeline::native::NativeCamera::read(self)
    }
}

/// The production camera arrives through the platform seam (phase 14.0.3) as a
/// boxed [`CameraSource`](mvm_pipeline::source::CameraSource).
impl FrameSource for Box<dyn mvm_pipeline::source::CameraSource> {
    fn read(&mut self) -> anyhow::Result<image::DynamicImage> {
        self.as_mut().read()
    }
}

pub(crate) struct CaptureAgent {
    source: Box<dyn FrameSource>,
    /// Stop after this many frames (for `mesh --max-frames` / tests). `None` =
    /// run until the mesh is stopped or the source errors.
    max_frames: Option<u64>,
}

impl CaptureAgent {
    pub(crate) fn new(source: Box<dyn FrameSource>, max_frames: Option<u64>) -> Self {
        Self { source, max_frames }
    }
}

impl Agent for CaptureAgent {
    fn name(&self) -> String {
        "capture".into()
    }

    fn run(&mut self, ctx: &MeshContext) -> AgentHealth {
        let mut frame_id = 0u64;
        let mut ticks = 0u64;
        let mut errors = 0u64;
        while ctx.running() {
            if let Some(max) = self.max_frames {
                if frame_id >= max {
                    break;
                }
            }
            match self.source.read() {
                Ok(img) => {
                    ctx.frames
                        .publish(Frame::new(frame_id, img, Instant::now()));
                    frame_id += 1;
                    ticks += 1;
                }
                Err(e) => {
                    eprintln!("  capture: source error: {e}");
                    errors += 1;
                    break;
                }
            }
        }
        // Capture drives the mesh: when it stops producing, wind everyone down.
        ctx.request_stop();
        AgentHealth {
            name: self.name(),
            ticks,
            errors,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mvm_core::WorldState;
    use mvm_mesh::LatestFrame;
    use std::sync::atomic::AtomicBool;
    use std::sync::{Arc, RwLock};

    /// Synthetic source: hands out `n` blank frames, then errors (EOF). Lets us
    /// drive `CaptureAgent::run` with no camera.
    struct StubSource {
        remaining: u64,
    }
    impl FrameSource for StubSource {
        fn read(&mut self) -> anyhow::Result<image::DynamicImage> {
            if self.remaining == 0 {
                anyhow::bail!("eof");
            }
            self.remaining -= 1;
            Ok(image::DynamicImage::new_rgb8(8, 8))
        }
    }

    fn ctx() -> MeshContext {
        MeshContext {
            frames: LatestFrame::new(),
            world: Arc::new(RwLock::new(WorldState::new(Instant::now()))),
            stop: Arc::new(AtomicBool::new(false)),
        }
    }

    #[test]
    fn name_is_capture() {
        let a = CaptureAgent::new(Box::new(StubSource { remaining: 0 }), None);
        assert_eq!(a.name(), "capture");
    }

    #[test]
    fn publishes_up_to_max_frames_then_stops() {
        let c = ctx();
        let mut a = CaptureAgent::new(Box::new(StubSource { remaining: 100 }), Some(5));
        let health = a.run(&c);
        assert_eq!(health.ticks, 5, "should publish exactly the frame budget");
        assert_eq!(health.errors, 0);
        assert_eq!(
            c.frames.latest().unwrap().id,
            4,
            "latest frame id = budget-1"
        );
        assert!(!c.running(), "capture requests stop when done");
    }

    #[test]
    fn source_error_stops_and_counts() {
        let c = ctx();
        // Budget larger than the source supplies → source errors mid-run.
        let mut a = CaptureAgent::new(Box::new(StubSource { remaining: 3 }), Some(10));
        let health = a.run(&c);
        assert_eq!(health.ticks, 3, "three good frames before EOF");
        assert_eq!(health.errors, 1, "the EOF read is counted as one error");
        assert!(!c.running());
    }
}
