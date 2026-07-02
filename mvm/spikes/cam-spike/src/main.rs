//! Phase 7.2 camera spike (G3).
//!
//! Decisive risk-removal experiment: can pure-Rust v4l2 mmap-streaming read
//! `/dev/video0` while PipeWire holds the camera? cv2 needed `CAP_FFMPEG`
//! because OpenCV's V4L2 mmap path threw `VIDIOC_QBUF: Bad file descriptor`.
//! This host's device is STREAMING-only (no READWRITE), so mmap is the only
//! native option — this spike tests exactly that.
//!
//! Exit 0 + steady fps printed  => 7.2 is real with the `v4l` crate.
//! Panic / QBUF error            => native v4l2 blocked under PipeWire here.

use std::time::Instant;
use v4l::buffer::Type;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;
use v4l::FourCC;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "/dev/video0".into());
    let n_frames: usize = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    println!("cam-spike: opening {path} (v4l2 mmap streaming)…");
    let dev = match Device::with_path(&path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL open: {e}");
            std::process::exit(2);
        }
    };

    // Enumerate formats + frame sizes — tells 7.2 which (fourcc,res) to negotiate
    // for higher fps than the default YUYV-1080.
    println!("  supported formats (fourcc → sizes):");
    if let Ok(fmts) = Capture::enum_formats(&dev) {
        for f in fmts {
            let cc = std::str::from_utf8(&f.fourcc.repr).unwrap_or("????").to_string();
            let mut sizes = String::new();
            if let Ok(fs) = Capture::enum_framesizes(&dev, f.fourcc) {
                for s in fs.iter().take(6) {
                    if let v4l::framesize::FrameSizeEnum::Discrete(d) = &s.size {
                        sizes.push_str(&format!(" {}x{}", d.width, d.height));
                    }
                }
            }
            println!("    {cc} ({}){}", f.description, sizes);
        }
    }

    // Optional format override: args[3]=fourcc (e.g. MJPG), args[4]=WxH.
    // Lets the spike measure MJPG/low-res fps headroom vs default YUYV-1080.
    if let Some(fourcc_s) = std::env::args().nth(3) {
        let (w, h) = std::env::args()
            .nth(4)
            .and_then(|wh| {
                let mut it = wh.split('x');
                Some((it.next()?.parse().ok()?, it.next()?.parse().ok()?))
            })
            .unwrap_or((1280u32, 720u32));
        let mut cc = [b' '; 4];
        for (i, b) in fourcc_s.bytes().take(4).enumerate() {
            cc[i] = b;
        }
        let want = v4l::Format::new(w, h, FourCC::new(&cc));
        match Capture::set_format(&dev, &want) {
            Ok(applied) => println!(
                "  override → requested {fourcc_s} {w}x{h}, driver gave {}x{} {}",
                applied.width,
                applied.height,
                std::str::from_utf8(&applied.fourcc.repr).unwrap_or("????")
            ),
            Err(e) => println!("  override failed: {e}"),
        }
    }

    // Report negotiated format (don't force — accept whatever PipeWire/driver gives).
    match Capture::format(&dev) {
        Ok(fmt) => {
            let cc: FourCC = fmt.fourcc;
            println!(
                "  format: {}x{}  fourcc={}  field={:?}",
                fmt.width,
                fmt.height,
                std::str::from_utf8(&cc.repr).unwrap_or("????"),
                fmt.field_order
            );
        }
        Err(e) => println!("  (format query failed: {e})"),
    }

    // The moment of truth: request mmap buffers + QBUF/DQBUF loop.
    let mut stream = match MmapStream::with_buffers(&dev, Type::VideoCapture, 4) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL MmapStream::with_buffers (this is the PipeWire/QBUF risk): {e}");
            std::process::exit(3);
        }
    };

    let mut got = 0usize;
    let mut first_t: Option<Instant> = None;
    let mut last_bytes = 0usize;
    let start = Instant::now();

    for i in 0..n_frames {
        match stream.next() {
            Ok((buf, meta)) => {
                if first_t.is_none() {
                    first_t = Some(Instant::now()); // start timing after first (warmup) frame
                }
                got += 1;
                last_bytes = buf.len();
                if i < 3 || i == n_frames - 1 {
                    println!(
                        "  frame {i:>3}: {} bytes  seq={}  ts={}.{:06}",
                        buf.len(),
                        meta.sequence,
                        meta.timestamp.sec,
                        meta.timestamp.usec
                    );
                }
            }
            Err(e) => {
                eprintln!("FAIL stream.next() at frame {i}: {e}");
                std::process::exit(4);
            }
        }
    }

    let total = start.elapsed().as_secs_f64();
    let steady = match first_t {
        Some(t) if got > 1 => (got as f64 - 1.0) / t.elapsed().as_secs_f64(),
        _ => 0.0,
    };

    println!("\n=== VERDICT ===");
    println!("  GREEN: pure-Rust v4l2 mmap read {got}/{n_frames} frames under live PipeWire");
    println!("  last frame: {last_bytes} bytes");
    println!("  steady fps: {steady:.1}  (raw capture only, no inference)");
    println!("  wall: {total:.1}s");
    println!("  => 7.2 is feasible with the `v4l` crate on this host.");
}
