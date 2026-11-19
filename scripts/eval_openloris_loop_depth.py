#!/usr/bin/env python3
"""Validate a label-free loop depth gate on a pair of OpenLORIS visits."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.spatial.transform import Rotation

from bonn_estimated_poses import run_binary
from evaluate_map_localization import associate, fit_to_ground_truth, load_euroc_ground_truth, load_tum_seconds
from loop_depth_gate import audit_loop, score_validation_depth, se3, select_depth_variant, validation_pairs
from openloris_map_changes import load_depth
from rgbd_map_cloud import colour_intrinsics, keyframe_poses, nearest, read_stamped, yaml_matrices


class Visit:
    def __init__(self, root, name):
        self.name = name
        self.map = root / "maps" / name / "equalized" / "map"
        depth = root / "rgbd" / name
        self.depth_root = depth
        self.intrinsics = colour_intrinsics(depth / "sensors.yaml")
        self.times, self.depth_paths = read_stamped(depth / "aligned_depth.txt")
        self.transforms = yaml_matrices(depth / "trans_matrix.yaml")
        with (self.map / "images.csv").open() as stream:
            self.images = {int(row["image_id"]): (float(row["timestamp_ns"]) * 1e-9, int(row["cam_id"]))
                           for row in csv.DictReader(stream) if row["timestamp_ns"]}
        self.truth = load_euroc_ground_truth(root / "data" / name / "mav0/state_groundtruth_estimate0/data.csv")
        self.trajectory = self.map.parent / "slam" / "trajectory_online_kf.tum"

    def depth_at(self, image):
        timestamp, camera = self.images[image]
        index = nearest(self.times, timestamp)
        if abs(self.times[index] - timestamp) > 0.020:
            return None
        left = "t265_fisheye1_optical_frame"
        left_from_colour = (
            np.linalg.inv(self.transforms["base_link", left])
            @ self.transforms["base_link", "d400_color_optical_frame"]
        )
        camera_from_colour = left_from_colour
        if camera == 1:
            camera_from_colour = (
                np.linalg.inv(self.transforms[left, "t265_fisheye2_optical_frame"]) @ left_from_colour
            )
        elif camera != 0:
            raise ValueError(f"unsupported OpenLORIS camera {camera}")
        return load_depth(self.depth_root / self.depth_paths[index], None), camera_from_colour


def score(path, visits, alignment):
    summary = json.loads((path / "map.json").read_text())
    rotation, translation = alignment
    reports = {}
    for index, visit in enumerate(visits):
        session = summary["sessions"][index]
        calibration = session.get("t_imu_cam", [summary["cameras"][0]["t_imu_cam"]])
        cam_from_imu = np.linalg.inv(se3(calibration[0]))
        estimate = {int(round(t * 1e9)): camera @ cam_from_imu for t, camera in keyframe_poses(path, session=index)}
        positions, angles = [], []
        for est_time, gt_time in associate(estimate, visit.truth, 10_000_000):
            est = estimate[est_time]
            gt_position, gt_rotation = visit.truth[gt_time]
            positions.append(np.linalg.norm(rotation @ est[:3, 3] + translation - gt_position))
            error = gt_rotation.T @ rotation @ est[:3, :3]
            angles.append(np.degrees(Rotation.from_matrix(error).magnitude()))
        reports[visit.name] = {
            "associated_keyframes": len(positions),
            "translation_rmse_m": float(np.sqrt(np.mean(np.square(positions)))) if positions else None,
            "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(angles)))) if angles else None,
        }
    return {"pose_errors": reports, "alignment": summary["session_alignment"]}


def run(root, names, binary, work):
    visits = [Visit(root, name) for name in names]
    common = [
        "--base", visits[0].map, "--add", visits[1].map, "--optimize-sessions",
        "--threads", 2, "--min-inliers", 10,
    ]
    full, gated, baseline = work / "full", work / "gated", work / "baseline"
    run_binary(binary, *common, "--out", full, "--refine-keyframes")
    run_binary(binary, *common, "--out", baseline)
    groups = json.loads((full / "map.json").read_text())["session_alignment"]["keyframe_refinement"]["pairs"]
    evidence, rejected, endpoints = [], [], [set(), set()]
    for group in groups:
        if not group["selected"]:
            continue
        a, b = group["base"], group["query"]
        for row in group["measurements"]:
            key = [a, b, row["base_image"], row["query_image"]]
            endpoints[a].add(visits[a].images[key[2]][0])
            endpoints[b].add(visits[b].images[key[3]][0])
            base = visits[a].depth_at(key[2])
            query = visits[b].depth_at(key[3])
            audit = {"decision": "unavailable"}
            if base is not None and query is not None:
                transform = np.linalg.inv(query[1]) @ se3(row["query_camera_from_base_camera"]) @ base[1]
                audit = audit_loop(base[0], query[0], visits[a].intrinsics, visits[b].intrinsics, transform)
            evidence.append({"edge": key, **audit})
            if audit["decision"] == "reject":
                rejected.extend(["--exclude-keyframe-loop", ":".join(map(str, key))])
    run_binary(binary, *common, "--out", gated, "--refine-keyframes", *rejected)
    options = {"session_alignment": baseline, "all_loops": full, "depth_gated": gated}
    poses = {name: [dict(keyframe_poses(path, session=i)) for i in range(2)]
             for name, path in options.items()}
    available = []
    image_ids = [{timestamp: image for image, (timestamp, cam) in v.images.items() if cam == 0}
                 for v in visits]
    for i, baseline_poses in enumerate(poses["session_alignment"]):
        compatible = {}
        for timestamp, pose in baseline_poses.items():
            if timestamp not in image_ids[i] or timestamp in endpoints[i]:
                continue
            index = nearest(visits[i].times, timestamp)
            if abs(visits[i].times[index] - timestamp) <= 0.020:
                compatible[timestamp] = pose
        available.append(compatible)
    pairs = validation_pairs(*available)
    depth_rows = [{stamp: visits[i].depth_at(image_ids[i][stamp]) for stamp in {p[i] for p in pairs}}
                  for i in range(2)]
    validation = {name: score_validation_depth(pairs, estimates, depth_rows,
                                                [v.intrinsics for v in visits])
                  for name, estimates in poses.items()}
    alignment = fit_to_ground_truth(load_tum_seconds(visits[0].trajectory), visits[0].truth, 10_000_000)
    return {"visits": names, "depth_evidence": evidence,
            "validation_depth": {"frame_pairs": pairs, "mean_agreement": validation,
                                 "preferred": select_depth_variant(validation, len(pairs))},
            **{name: score(path, visits, alignment) for name, path in options.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="OpenLORIS root with maps, data and rgbd subdirectories")
    parser.add_argument("--merge", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with TemporaryDirectory(prefix="openloris-depth-") as work:
        result = run(args.data, ["office1-1", "office1-6"], args.merge, Path(work))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
