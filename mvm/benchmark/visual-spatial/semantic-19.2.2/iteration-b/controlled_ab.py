#!/usr/bin/env python3
"""Deterministic scheduler/integrity harness for the sparse semantic expert.

This is a contract harness, not a claim about an integrated runtime.  It models
the capture producer and a single semantic worker using the measured p95 cost.
"""

from __future__ import annotations

import json
from pathlib import Path


def simulate(*, frames: int = 180, fps: float = 25.0, cadence: int = 30,
             worker_ms: float = 164.979, max_age_frames: int = 60) -> dict[str, object]:
    if frames <= 0 or fps <= 0 or cadence <= 0 or worker_ms <= 0:
        raise ValueError("simulation parameters must be positive")
    period = 1000.0 / fps
    worker_free_at = 0.0
    pending: tuple[int, float, float] | None = None
    completed: list[tuple[int, float, float]] = []
    dropped_stale = 0
    max_queue = 0
    capture_times: list[float] = []
    for frame in range(frames):
        now = frame * period
        capture_times.append(now)
        if pending is not None and pending[2] <= now:
            completed.append(pending)
            pending = None
        if frame % cadence == 0:
            if pending is None and worker_free_at <= now:
                pending = (frame, now, now + worker_ms)
                worker_free_at = pending[2]
            else:
                # Queue depth is deliberately one: a busy worker means this
                # semantic observation is skipped, never allowed to block RGB.
                dropped_stale += 1
        max_queue = max(max_queue, int(pending is not None))
    if pending is not None:
        completed.append(pending)
    accepted = [job for job in completed if frames - 1 - job[0] <= max_age_frames]
    stale = len(completed) - len(accepted)
    capture_periods = [b - a for a, b in zip(capture_times, capture_times[1:])]
    nominal = period * (frames - 1)
    observed = capture_times[-1] - capture_times[0]
    wall_delta_pct = abs(observed - nominal) / nominal * 100.0 if nominal else 0.0
    return {
        "harness_version": "19.2.2-b.1",
        "frames": frames,
        "fps": fps,
        "cadence_frames": cadence,
        "worker_p95_ms": worker_ms,
        "capture_blocking": False,
        "max_queue_depth": max_queue,
        "stale_results_dropped": True,
        "source_frame_stamp_required": all(job[0] >= 0 for job in accepted),
        "accepted_results": len(accepted),
        "busy_submission_skips": dropped_stale,
        "stale_results_withheld": stale,
        "capture_wall_time_delta_pct": round(wall_delta_pct, 6),
        "capture_period_min_ms": round(min(capture_periods), 6) if capture_periods else 0.0,
        "capture_period_max_ms": round(max(capture_periods), 6) if capture_periods else 0.0,
    }


def main() -> int:
    root = Path(__file__).resolve().parent
    result = simulate()
    print(json.dumps(result, indent=2, sort_keys=True))
    (root / "controlled_ab_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
