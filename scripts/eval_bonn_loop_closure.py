#!/usr/bin/env python3
"""Compare Bonn map scoring with mocap, session registration, and keyframe loops."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2

from bonn_estimated_poses import export_visit, merge_trial, pose_rmse
from eval_bonn_map_changes import INTRINSICS, Sequence, evaluate_map, select_frames


def compare_poses(pair, sequences, work, tools, separation, level_from_depth):
    visits, scoring_samples = [], []
    for visit in pair["visits"]:
        sequence = sequences[visit["sequence"]]
        holdout = {item["frame"] for item in visit["labels"]}
        scoring = select_frames(visit["window"], holdout)
        tracking = select_frames(visit.get("tracking_window", visit["window"]), holdout)
        samples = list(sequence.observations(sorted(set(scoring) | set(tracking)), work / sequence.root.name))
        visits.append((sequence, samples))
        scoring_samples.append([item for item in samples if int(item[1].stem) in scoring])
    runs = [export_visit(seq, frames, INTRINSICS, work / f"visit_{i}", tools / "tum_rgbd_save_map",
                         level_from_depth=level_from_depth)
            for i, (seq, frames) in enumerate(visits)]
    result = {"mocap": evaluate_map(pair, sequences, work, 0.07, 0.10, 2, diagnostics=True)}
    for variant, options in (
        ("session_alignment", {"refine": False}),
        ("default_loops", {}),
        ("short_baseline_loops", {"separation": separation}),
    ):
        poses, alignment = merge_trial(
            visits, runs, work / variant, tools / "basalt_map_merge", **options,
        )
        score = evaluate_map(pair, sequences, work, 0.07, 0.10, 2, diagnostics=True, poses=poses)
        score["scoring_pose_errors"] = [pose_rmse(sample, visit_poses)
                                         for sample, visit_poses in zip(scoring_samples, poses, strict=True)]
        score["alignment"] = alignment
        result[variant] = score
    if level_from_depth:
        result["depth_floor_support"] = [
            json.loads((run[0].parent / "depth_floor.json").read_text()) for run in runs
        ]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, action="append", required=True)
    parser.add_argument("--map-tools", type=Path, required=True)
    parser.add_argument("--level-from-depth", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = {}
    with TemporaryDirectory(prefix="bonn-loop-") as directory:
        work = Path(directory)
        for path in args.annotations:
            pairs = json.loads(path.read_text())["maps"]
            names = {v["sequence"] for pair in pairs for v in pair["visits"]}
            sequences = {name: Sequence(args.data / f"rgbd_bonn_{name}") for name in names}
            for pair in pairs:
                result[pair["name"]] = compare_poses(pair, sequences, work / pair["name"],
                                                     args.map_tools, 0.15, args.level_from_depth)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
