#!/usr/bin/env python3
"""Deterministic A/B/C adapter for the frozen depth-assisted tracker replay."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from pathlib import Path


class ContractError(ValueError):
    pass


def iou(a, b):
    x1, y1, x2, y2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union > 0 else 0.0


def center(b):
    return ((b[0]+b[2])*0.5, (b[1]+b[3])*0.5)


def distance(a, b):
    return math.sqrt(sum((x-y)**2 for x, y in zip(a, b)))


def validate(data):
    if data.get("schema") != "mvm-depth-tracker-replay/1.0" or data.get("status") not in {"synthetic_replay", "captured_replay"}:
        raise ContractError("unsupported replay schema/status")
    ids = set()
    for scenario in data.get("scenarios", []):
        sid = scenario.get("id")
        if not isinstance(sid, str) or sid in ids or scenario.get("domain") not in {"person", "object"}:
            raise ContractError("invalid or duplicate scenario")
        ids.add(sid)
        for frame in scenario.get("frames", []):
            if not isinstance(frame, list):
                raise ContractError(f"{sid}: frame must be a list")
            for det in frame:
                bbox = det.get("bbox")
                if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    raise ContractError(f"{sid}: invalid bbox")
                xyz = det.get("xyz")
                if xyz is not None and (not isinstance(xyz, list) or len(xyz) != 3 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in xyz)):
                    raise ContractError(f"{sid}: invalid xyz")
    return ids


def best_assignment(scores, n_tracks):
    if not scores or n_tracks == 0:
        return [None] * len(scores)
    best_score, best = -math.inf, [None] * len(scores)
    choices = list(range(n_tracks)) + [None]
    for assignment in itertools.product(choices, repeat=len(scores)):
        used = [x for x in assignment if x is not None]
        if len(set(used)) != len(used):
            continue
        total = sum(scores[i][j] for i, j in enumerate(assignment) if j is not None)
        if any(j is not None and not math.isfinite(scores[i][j]) for i, j in enumerate(assignment)):
            continue
        if total > best_score:
            best_score, best = total, list(assignment)
    return best


def pair_score(det, track, variant):
    overlap = iou(det["bbox"], track["bbox"])
    dc, tc = center(det["bbox"]), center(track["bbox"])
    centre_dist = distance(dc, tc)
    if overlap <= 0.30 and centre_dist >= 0.15:
        return -math.inf
    score = overlap - 0.10 * centre_dist
    xyz, prior = det.get("xyz"), track.get("xyz")
    if variant in {"B", "C"} and xyz is not None and prior is not None and abs(xyz[2] - prior[2]) > 0.75:
        return -math.inf
    if variant == "C" and xyz is not None and prior is not None:
        predicted = prior
        if track.get("xyz_velocity") is not None:
            predicted = [prior[i] + track["xyz_velocity"][i] * (track["missed"] + 1) for i in range(3)]
        metric_dist = distance(xyz, predicted)
        if metric_dist > 1.25:
            return -math.inf
        score += 0.20 * (1.0 - metric_dist / 1.25)
    return score


def run_scenario(scenario, variant):
    tracks, next_id = [], 0
    assignments, false_births = [], 0
    for frame_index, detections in enumerate(scenario["frames"]):
        for track in tracks:
            track["missed"] += 1
        live = [track for track in tracks if track["missed"] <= 3]
        scores = [[pair_score(det, track, variant) for track in live] for det in detections]
        selected = best_assignment(scores, len(live))
        for det_index, det in enumerate(detections):
            track_index = selected[det_index]
            if track_index is None:
                track = {"id": next_id, "bbox": det["bbox"], "xyz": det.get("xyz"), "xyz_velocity": None, "missed": 0, "truths": set()}
                next_id += 1
                tracks.append(track)
                if det.get("truth") is None:
                    false_births += 1
            else:
                track = live[track_index]
                old_xyz = track.get("xyz")
                new_xyz = det.get("xyz")
                if old_xyz is not None and new_xyz is not None:
                    track["xyz_velocity"] = [new_xyz[i] - old_xyz[i] for i in range(3)]
                if new_xyz is not None:
                    track["xyz"] = new_xyz
                track["bbox"] = det["bbox"]
                track["missed"] = 0
            truth = det.get("truth")
            if truth is not None:
                track["truths"].add(truth)
                assignments.append((frame_index, truth, track["id"]))
    by_truth = {}
    for frame, truth, track_id in assignments:
        by_truth.setdefault(truth, []).append((frame, track_id))
    switches = fragments = correct = 0
    for observed in by_truth.values():
        counts = {}
        for _, tid in observed:
            counts[tid] = counts.get(tid, 0) + 1
        correct += max(counts.values())
        for (frame_a, id_a), (frame_b, id_b) in zip(observed, observed[1:]):
            if id_a != id_b:
                if frame_b == frame_a + 1:
                    switches += 1
                else:
                    fragments += 1
    false_merges = sum(len(track["truths"]) > 1 for track in tracks)
    total = len(assignments)
    return {"observations": total, "idtp": correct, "id_switches": switches, "fragmentations": fragments, "false_births": false_births, "false_merges": false_merges}


def aggregate(results):
    if not results:
        return None
    value = {key: sum(item[key] for item in results) for key in results[0]}
    observations = value["observations"]
    association = value["idtp"] / observations if observations else 1.0
    detection_precision = observations / (observations + value["false_births"]) if observations else 1.0
    value["idf1_proxy"] = association
    value["assa_proxy"] = association
    value["hota_proxy"] = math.sqrt(association * detection_precision)
    return value


def production_metrics(observations):
    """Score sampled production track ids against reviewed one-person truth.

    This is deliberately separate from the A/B/C adapter. With one physical
    truth identity, the global ID assignment maps the most frequent selected
    production id to that truth. Extra/stale ids are false positives.
    """
    truth_frames = sum(item.get("truth") is not None for item in observations)
    predictions = sum(len(item.get("predicted_track_ids", [])) for item in observations)
    selected = [
        item.get("selected_track_id")
        for item in observations
        if item.get("truth") is not None and item.get("selected_track_id") is not None
    ]
    counts = {}
    for track_id in selected:
        counts[track_id] = counts.get(track_id, 0) + 1
    idtp = max(counts.values()) if counts else 0
    idfn = truth_frames - idtp
    idfp = predictions - idtp
    false_negatives = truth_frames - len(selected)
    false_positives = sum(
        len(item.get("predicted_track_ids", []))
        - (1 if item.get("truth") is not None and item.get("selected_track_id") is not None else 0)
        for item in observations
    )
    duplicate_predictions = sum(
        max(0, len(item.get("predicted_track_ids", [])) - 1)
        for item in observations
        if item.get("truth") is not None
    )
    stale_predictions_on_absent = sum(
        len(item.get("predicted_track_ids", []))
        for item in observations
        if item.get("truth") is None
    )
    switches = 0
    previous = None
    for item in observations:
        if item.get("truth") is None:
            continue
        current = item.get("selected_track_id")
        if current is not None and previous is not None and current != previous:
            switches += 1
        if current is not None:
            previous = current
    fragments = 0
    seen = False
    missed_after_seen = False
    for item in observations:
        if item.get("truth") is None:
            continue
        detected = item.get("selected_track_id") is not None
        if detected and seen and missed_after_seen:
            fragments += 1
            missed_after_seen = False
        elif not detected and seen:
            missed_after_seen = True
        seen = seen or detected
    denominator = 2 * idtp + idfp + idfn
    return {
        "sample_count": len(observations),
        "truth_positive_samples": truth_frames,
        "prediction_count": predictions,
        "matched_samples": len(selected),
        "false_negatives": false_negatives,
        "false_positives": false_positives,
        "duplicate_predictions": duplicate_predictions,
        "stale_predictions_on_absent": stale_predictions_on_absent,
        "unique_selected_track_ids": sorted(counts),
        "id_switches": switches,
        "fragmentations": fragments,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
        "idf1": 2 * idtp / denominator if denominator else 1.0,
        "detection_recall": len(selected) / truth_frames if truth_frames else 1.0,
    }


def evaluate(data, loops):
    output = {"schema": data["schema"], "status": data["status"], "variants": {}}
    if data.get("production_observations") is not None:
        output["production_tracker"] = production_metrics(data["production_observations"])
    for variant in ("A", "B", "C"):
        start = time.perf_counter_ns()
        results = None
        for _ in range(loops):
            results = [run_scenario(scenario, variant) for scenario in data["scenarios"]]
        elapsed = time.perf_counter_ns() - start
        metrics = aggregate(results)
        metrics["by_domain"] = {
            domain: aggregate([
                result for result, scenario in zip(results, data["scenarios"])
                if scenario["domain"] == domain
            ])
            for domain in ("person", "object")
            if any(scenario["domain"] == domain for scenario in data["scenarios"])
        }
        metrics["mean_runtime_us_per_scenario"] = elapsed / loops / len(results) / 1000.0
        metrics["scenario_count"] = len(results)
        output["variants"][variant] = metrics
    baseline = output["variants"]["A"]["mean_runtime_us_per_scenario"]
    for metrics in output["variants"].values():
        metrics["runtime_delta_percent"] = (metrics["mean_runtime_us_per_scenario"] / baseline - 1.0) * 100.0
    base = output["variants"]["A"]
    accepted = []
    for variant in ("B", "C"):
        metrics = output["variants"][variant]
        improves_target = any(metrics[key] < base[key] for key in ("id_switches", "fragmentations", "false_births"))
        no_merge_growth = metrics["false_merges"] <= base["false_merges"]
        runtime_ok = metrics["runtime_delta_percent"] <= 5.0
        if data["status"] == "captured_replay" and improves_target and no_merge_growth and runtime_ok:
            accepted.append(variant)
    if accepted:
        reason = "accepted minimal depth cue set"
    elif data["status"] != "captured_replay":
        reason = "reject depth cues: captured replay required"
    else:
        reason = "reject depth cues: no target-metric improvement within merge/runtime gates"
    output["verdict"] = {
        "selected": min(accepted, key=lambda name: (output["variants"][name]["runtime_delta_percent"], name)) if accepted else "A",
        "production_change": bool(accepted),
        "reason": reason,
    }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", type=Path, default=Path(__file__).with_name("replay.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--loops", type=int, default=200)
    args = parser.parse_args()
    if args.loops <= 0:
        raise SystemExit("--loops must be positive")
    data = json.loads(args.replay.read_text(encoding="utf-8"))
    validate(data)
    result = evaluate(data, args.loops)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
