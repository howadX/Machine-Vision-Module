//! Spike B — "latest frame wins" channel semantics (phase 8.0.1 Agent Mesh prep).
//!
//! The mesh wants: a fast producer (camera ~30fps) feeds slow consumers
//! (agents), and each consumer always processes the FRESHEST frame, silently
//! dropping intermediate ones — no backlog, no "catch up from oldest".
//!
//! tokio::broadcast does NOT give this: on lag it yields RecvError::Lagged and
//! resumes from the OLDEST buffered frame (catch-up), not the latest. The right
//! primitive is a single-slot latest-value cell. This spike proves the semantic
//! with std only (Mutex<Option<Arc<Frame>>> — an ArcSwap would be the lock-free
//! production version; semantics identical). Zero extra deps, zero tokio.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
struct Frame {
    id: u64,
}

fn main() {
    // Single-slot "latest frame" cell. Producer overwrites; consumer reads latest.
    let slot: Arc<Mutex<Option<Arc<Frame>>>> = Arc::new(Mutex::new(None));
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    // Producer: 30fps (~33ms), 60 frames total (~2s).
    let prod_slot = slot.clone();
    let producer = std::thread::spawn(move || {
        for id in 0..60u64 {
            *prod_slot.lock().unwrap() = Some(Arc::new(Frame { id }));
            std::thread::sleep(Duration::from_millis(33));
        }
    });

    // Consumer: ~5fps (200ms work per frame). Should see ids ~0,6,12,... with
    // gaps (intermediate frames dropped), and NEVER fall behind into a backlog.
    let cons_slot = slot.clone();
    let cons_stop = stop.clone();
    let consumer = std::thread::spawn(move || {
        let mut seen: Vec<u64> = Vec::new();
        let mut last: Option<u64> = None;
        while !cons_stop.load(std::sync::atomic::Ordering::Relaxed) {
            let frame = cons_slot.lock().unwrap().clone();
            if let Some(f) = frame {
                if last != Some(f.id) {
                    seen.push(f.id);
                    last = Some(f.id);
                }
            }
            std::thread::sleep(Duration::from_millis(200)); // slow agent
        }
        seen
    });

    let t0 = Instant::now();
    producer.join().unwrap();
    // let consumer finish its in-flight read, then stop
    std::thread::sleep(Duration::from_millis(250));
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let seen = consumer.join().unwrap();
    let elapsed = t0.elapsed().as_secs_f64();

    println!("=== Spike B: latest-frame channel (std single-slot) ===");
    println!("producer: 60 frames @ 30fps (~2.0s) | consumer: ~5fps (200ms/frame)\n");
    println!("frames consumer actually processed (ids): {seen:?}");
    println!("count: {} over {:.2}s", seen.len(), elapsed);

    // Verify "latest wins": gaps between consumed ids should be ~5-7 (dropped
    // intermediates), and the consumer must reach near the final id (no backlog
    // lag that leaves it stuck on old frames).
    let gaps: Vec<u64> = seen.windows(2).map(|w| w[1] - w[0]).collect();
    let avg_gap = if gaps.is_empty() { 0.0 } else { gaps.iter().sum::<u64>() as f64 / gaps.len() as f64 };
    let last_seen = seen.last().copied().unwrap_or(0);
    println!("avg gap between consumed frames: {avg_gap:.1} (≈ producer/consumer rate ratio = 6)");
    println!("last id consumed: {last_seen} (producer's last = 59)\n");

    if avg_gap >= 4.0 && last_seen >= 53 {
        println!("✓ PASS: consumer drops intermediate frames AND stays current (no backlog).");
        println!("  → single-slot latest-value cell is the right primitive (ArcSwap in prod).");
        println!("  → tokio::broadcast NOT needed/wanted (its lag-catchup is the wrong semantic).");
    } else {
        println!("✗ unexpected: avg_gap={avg_gap:.1}, last_seen={last_seen} — investigate.");
    }
}
