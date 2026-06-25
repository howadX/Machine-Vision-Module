//! Native v4l2 camera capture (phase 7.2) — pure-Rust, no Python bridge.
//!
//! Promoted from `spikes/cam-spike` (gap G3), which proved on this host that
//! the pure-Rust `v4l` crate reads `/dev/video0` under a live PipeWire session
//! via mmap streaming — no system dev libs, no PipeWire portal. See
//! `spikes/cam-spike/FINDINGS.md`.
//!
//! Design choices fixed by the spike:
//! - **Negotiate MJPG** (not the default YUYV-1080) → ~25 fps capture ceiling
//!   vs 3.4 fps for raw YUYV. Frames arrive as JPEG, decoded via the `image`
//!   crate (already a project dep).
//! - **ROTATE_180** after decode (camera is ceiling-mounted upside-down — the
//!   same rotation the Python bridge does with `cv2.ROTATE_180`).
//!
//! This module is **additive**: it does NOT replace the Python bridge. Wiring it
//! as the live frame source is a separate, approval-gated step (Python-light).

use anyhow::{bail, Context, Result};
use image::DynamicImage;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

/// A pure-Rust v4l2 MJPG camera. Owns the device for the process lifetime.
pub struct NativeCamera {
    // MmapStream borrows the Device. Rather than a self-referential struct, the
    // Device is leaked to 'static — fine for a single process-lifetime camera;
    // the stream below releases its v4l buffers on drop.
    stream: MmapStream<'static>,
    rotate_180: bool,
    width: u32,
    height: u32,
}

impl NativeCamera {
    /// Open `path` and negotiate `MJPG width×height` mmap streaming.
    ///
    /// `rotate_180` mirrors the bridge's `cv2.ROTATE_180` for the upside-down
    /// ceiling mount.
    pub fn open(path: &str, width: u32, height: u32, rotate_180: bool) -> Result<Self> {
        let dev = Device::with_path(path)
            .with_context(|| format!("open v4l2 device {path}"))?;

        let want = v4l::Format::new(width, height, FourCC::new(b"MJPG"));
        let applied = Capture::set_format(&dev, &want).context("set MJPG format")?;
        if &applied.fourcc.repr != b"MJPG" {
            bail!(
                "driver refused MJPG (got {})",
                std::str::from_utf8(&applied.fourcc.repr).unwrap_or("????")
            );
        }

        // Leak the Device to 'static so the borrowing MmapStream can be owned.
        let dev: &'static Device = Box::leak(Box::new(dev));
        let stream = MmapStream::with_buffers(dev, Type::VideoCapture, 4)
            .context("start mmap stream (the PipeWire/QBUF risk — cleared by spike)")?;

        Ok(Self {
            stream,
            rotate_180,
            width: applied.width,
            height: applied.height,
        })
    }

    /// Capture and decode the next frame as an RGB image (rotated if configured).
    pub fn read(&mut self) -> Result<DynamicImage> {
        let (buf, meta) = self.stream.next().context("dequeue v4l2 frame")?;
        // `buf.len()` is the mmap buffer capacity; the JPEG occupies bytesused.
        let used = (meta.bytesused as usize).min(buf.len());
        decode_mjpeg(&buf[..used], self.rotate_180)
    }

    pub fn dimensions(&self) -> (u32, u32) {
        (self.width, self.height)
    }
}

/// Decode an MJPG/JPEG frame and optionally rotate 180° — pure, unit-testable.
pub fn decode_mjpeg(jpeg: &[u8], rotate_180: bool) -> Result<DynamicImage> {
    let img = image::load_from_memory_with_format(jpeg, image::ImageFormat::Jpeg)
        .context("decode MJPG frame")?;
    Ok(if rotate_180 { img.rotate180() } else { img })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, GenericImageView, RgbImage};

    /// Encode a tiny gradient image to JPEG bytes for decode round-trip tests.
    fn jpeg_fixture(w: u32, h: u32) -> Vec<u8> {
        let mut img = RgbImage::new(w, h);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([(x * 8) as u8, (y * 8) as u8, 128]);
        }
        let mut bytes: Vec<u8> = Vec::new();
        DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Jpeg)
            .unwrap();
        bytes
    }

    #[test]
    fn decode_preserves_dimensions() {
        let jpeg = jpeg_fixture(16, 12);
        let img = decode_mjpeg(&jpeg, false).unwrap();
        assert_eq!(img.dimensions(), (16, 12));
    }

    #[test]
    fn rotate_180_keeps_dimensions_and_swaps_corners() {
        let jpeg = jpeg_fixture(16, 12);
        let up = decode_mjpeg(&jpeg, false).unwrap();
        let down = decode_mjpeg(&jpeg, true).unwrap();
        assert_eq!(down.dimensions(), (16, 12), "rotation preserves dims");
        // Top-left of rotated ≈ bottom-right of original (allow JPEG slack).
        let (w, h) = up.dimensions();
        let orig_br = up.get_pixel(w - 1, h - 1);
        let rot_tl = down.get_pixel(0, 0);
        let close = |a: u8, b: u8| (a as i32 - b as i32).abs() <= 24;
        assert!(
            close(orig_br[0], rot_tl[0]) && close(orig_br[1], rot_tl[1]),
            "rotated top-left should match original bottom-right (got {:?} vs {:?})",
            rot_tl, orig_br
        );
    }

    #[test]
    fn decode_rejects_garbage() {
        assert!(decode_mjpeg(&[0, 1, 2, 3, 4], false).is_err());
    }

    /// Live hardware check (needs /dev/video0). Run explicitly:
    ///   cargo test -p mvm-pipeline -- --ignored native_camera_captures_live
    #[test]
    #[ignore]
    fn native_camera_captures_live() {
        let mut cam = NativeCamera::open("/dev/video0", 1280, 720, true).unwrap();
        assert_eq!(cam.dimensions(), (1280, 720));
        let frame = cam.read().unwrap();
        assert_eq!(frame.dimensions(), (1280, 720), "decoded frame is 720p");
    }
}
