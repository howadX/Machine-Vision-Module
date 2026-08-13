#!/usr/bin/env python3
"""Join reviewed RGB annotations with frozen MVM Kinect session evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--scene-json", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (args.session / "labels.jsonl").read_text(encoding="utf-8").splitlines()]
    reviewed = annotations["frames"]
    if len(rows) != len(reviewed):
        raise SystemExit(f"annotation/label count mismatch: {len(reviewed)} != {len(rows)}")

    frames = []
    production = []
    for row, annotation in zip(rows, reviewed):
        seq = row["session"]["seq_in_session"]
        if seq != annotation["seq"]:
            raise SystemExit(f"sequence mismatch at {seq}")
        humans = [entity for entity in row["entities"] if entity.get("group") == "human"]
        selected = annotation["selected_track_id"]
        selected_entity = next((entity for entity in humans if entity["track_id"] == selected), None)
        if selected is not None and selected_entity is None:
            raise SystemExit(f"selected track {selected} absent at sequence {seq}")
        xyz = None
        if selected_entity is not None:
            position = selected_entity.get("attributes", {}).get("position_3d")
            if position is not None:
                xyz = [position[axis] for axis in ("x", "y", "z")]
        detections = []
        if annotation["truth"] is not None:
            detections.append({"truth": annotation["truth"], "bbox": annotation["bbox"], "xyz": xyz})
        frames.append(detections)
        depth = row["frame"]["depth"]
        rgb_path = args.session / row["frame"]["frame_path"]
        depth_path = args.session / row["frame"]["depth_path"]
        if not rgb_path.is_file() or not depth_path.is_file():
            raise SystemExit(f"missing RGB-D artifact at sequence {seq}")
        production.append({
            "seq": seq,
            "frame_id": row["frame_id"],
            "timestamp": row["frame"]["timestamp"],
            "visibility": annotation["visibility"],
            "truth": annotation["truth"],
            "truth_bbox": annotation["bbox"],
            "predicted_track_ids": [entity["track_id"] for entity in humans],
            "predicted_human_entities": humans,
            "selected_track_id": selected,
            "selected_metric_xyz_m": xyz,
            "depth_metadata": depth,
            "frame_path": row["frame"]["frame_path"],
            "depth_path": row["frame"]["depth_path"],
            "frame_size_bytes": rgb_path.stat().st_size,
            "frame_sha256": digest(rgb_path),
            "depth_size_bytes": depth_path.stat().st_size,
            "depth_sha256": digest(depth_path),
        })

    output = {
        "schema": "mvm-depth-tracker-replay/1.0",
        "status": "captured_replay",
        "capture": {
            "session_id": annotations["session_id"],
            "sensor": "kinect_v1:A00364911518102A",
            "calibration_revision": rows[0]["frame"]["depth"]["calibration"]["revision_sha256"],
            "labels_sha256": digest(args.session / "labels.jsonl"),
            "session_toml_sha256": digest(args.session / "session.toml"),
            "annotations_sha256": digest(args.annotations),
            "scene_json_sha256": digest(args.scene_json),
            "run_log_sha256": digest(args.run_log),
            "limitations": [
                "single physical person; no captured multi-person crossing",
                "manual truth bboxes are approximate visible extents",
                "xyz is selected MVM direct metric output, not independent surveyed ground truth",
                "YOLO object and segmentation models unavailable; no captured object/mask slice",
            ],
        },
        "production_observations": production,
        "scenarios": [{"id": "captured_one_person_motion", "domain": "person", "frames": frames}],
    }
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
