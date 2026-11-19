#!/usr/bin/env python3
"""Compare Bonn map scoring with mocap, session registration, and keyframe loops."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from bonn_estimated_poses import export_visit, merge_trial, pose_rmse
from change_loop_gate import select_by_local_change
from eval_bonn_map_changes import INTRINSICS, Sequence, evaluate_map, select_frames
from loop_depth_gate import audit_loop, score_validation_depth, se3, select_depth_variant, validation_pairs


def compare_poses(pair, sequences, work, tools, separation, level_from_depth, leave_one_out=False):
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
    poses_by_variant = {}

    def score_variant(variant, **options):
        poses, alignment = merge_trial(
            visits, runs, work / variant, tools / "basalt_map_merge", **options,
        )
        poses_by_variant[variant] = poses
        score = evaluate_map(pair, sequences, work, 0.07, 0.10, 2, diagnostics=True, poses=poses)
        score["scoring_pose_errors"] = [pose_rmse(sample, visit_poses)
                                         for sample, visit_poses in zip(scoring_samples, poses, strict=True)]
        score["alignment"] = alignment
        return score

    result["session_alignment"] = score_variant("session_alignment", refine=False)
    result["default_loops"] = score_variant("default_loops")
    result["short_baseline_loops"] = score_variant("short_baseline_loops", separation=separation)
    if leave_one_out:
        refinement = result["short_baseline_loops"]["alignment"]["session_alignment"]["keyframe_refinement"]
        rejected = []
        endpoints = [set() for _ in visits]
        result["depth_gate"] = []
        for group in refinement["pairs"]:
            base, query = group["base"], group["query"]
            if not group["selected"]:
                continue
            for measurement in group["measurements"]:
                a, b = measurement["base_image"], measurement["query_image"]
                key = [base, query, a, b]
                base_frame = runs[base][1][a - 1][0]
                query_frame = runs[query][1][b - 1][0]
                endpoints[base].add(base_frame)
                endpoints[query].add(query_frame)
                evidence = audit_loop(visits[base][0].depth(base_frame) * 0.001,
                                      visits[query][0].depth(query_frame) * 0.001,
                                      INTRINSICS, INTRINSICS,
                                      se3(measurement["query_camera_from_base_camera"]))
                result["depth_gate"].append({"edge": key, **evidence})
                if evidence["decision"] == "reject":
                    rejected.append(key)
        result["depth_gated"] = score_variant("depth_gated", separation=separation, exclude=rejected)
        result["leave_one_out"] = []
        for group in refinement["pairs"]:
            if group["selected"] < 3:
                continue
            for a, b in group["image_pairs"]:
                key = [group["base"], group["query"], a, b]
                trial = score_variant(f"without_{'_'.join(map(str, key))}", separation=separation, exclude=[key])
                result["leave_one_out"].append({"excluded": key, "result": trial})
        reference = poses_by_variant["session_alignment"]
        available = []
        for i, (_, samples) in enumerate(visits):
            excluded = endpoints[i] | {int(path.stem) for _, path in scoring_samples[i]}
            available.append({int(path.stem): reference[i][int(path.stem)] for _, path in samples
                              if int(path.stem) not in excluded})
        pairs = validation_pairs(*available)
        depth_rows = [{frame: (visits[i][0].depth(frame) * 0.001, np.eye(4))
                       for frame in {pair[i] for pair in pairs}} for i in range(2)]
        options = [name for name in poses_by_variant if name != "default_loops"]
        scores = {name: score_validation_depth(pairs, poses_by_variant[name], depth_rows,
                                               (INTRINSICS, INTRINSICS)) for name in options}
        result["validation_depth"] = {
            "frame_pairs": pairs, "mean_agreement": scores, "preferred": select_depth_variant(scores, len(pairs)),
        }
        local_options = ("session_alignment", "short_baseline_loops")
        depth_paths = [{int(path.stem): path for _, path in samples} for _, samples in visits]
        result["local_change_selection"] = select_by_local_change(
            pairs, {name: poses_by_variant[name] for name in local_options}, depth_paths, INTRINSICS,
            {name: scores[name] for name in local_options},
        )
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
    parser.add_argument("--leave-one-out", action="store_true")
    parser.add_argument("--pair", help="run one named annotation pair")
    parser.add_argument("--level-from-depth", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    result = {}
    with TemporaryDirectory(prefix="bonn-loop-") as directory:
        work = Path(directory)
        for path in args.annotations:
            pairs = [pair for pair in json.loads(path.read_text())["maps"]
                     if args.pair is None or pair["name"] == args.pair]
            names = {v["sequence"] for pair in pairs for v in pair["visits"]}
            sequences = {name: Sequence(args.data / f"rgbd_bonn_{name}") for name in names}
            for pair in pairs:
                result[pair["name"]] = compare_poses(pair, sequences, work / pair["name"],
                                                     args.map_tools, 0.15, args.level_from_depth, args.leave_one_out)
    if not result:
        parser.error("no annotation pair matched")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
