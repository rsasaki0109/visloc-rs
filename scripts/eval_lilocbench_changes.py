#!/usr/bin/env python3
"""Evaluate fixed loop selection on labeled LILocBench RGB-D revisits."""

import argparse
from functools import lru_cache
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import yaml

from change_loop_gate import select_by_local_change
from bonn_estimated_poses import export_visit, merge_trial, pose_rmse
from eval_bonn_map_changes import evaluate_map, select_frames
from loop_depth_gate import score_validation_depth, select_depth_variant, validation_pairs


def yaml_matrix(row):
    t, q = row["translation"], row["rotation"]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat([q[k] for k in "xyzw"]).as_matrix()
    pose[:3, 3] = [t[k] for k in "xyz"]
    return pose


def calibration_matrix(path):
    document = yaml.safe_load(path.read_text())
    if (document["width"], document["height"]) != (640, 480):
        raise ValueError("expected the calibrated 640 × 480 LILocBench stream")
    return np.asarray(document["K"]).reshape(3, 3), np.asarray(document["distortion_coefficients"])


class Visit:
    def __init__(self, root, case, sequence):
        self.root = root / "dataset" / sequence
        manifest = json.loads((root / f"{case}_{sequence}.json").read_text())
        self.rgb_paths = manifest["rgb"]
        self.depth_paths = manifest["depth"]
        self.times = np.array([float(Path(name).stem) for name in self.rgb_paths])
        depth_times = np.array([float(Path(name).stem) for name in self.depth_paths])
        if not (len(self.times) == len(depth_times) and np.all(np.diff(self.times) > 0)
                and np.all(np.abs(depth_times - self.times) < 0.035)):
            raise ValueError(f"{case}/{sequence}: invalid RGB-D synchronization")
        truth = np.loadtxt(root / f"{sequence}.tum")
        self.truth_time = truth[:, 0]
        self.positions = truth[:, 1:4]
        self.rotations = Slerp(self.truth_time, Rotation.from_quat(truth[:, 4:8]))
        extrinsics = yaml.safe_load((self.root / "transformations.yaml").read_text())
        transforms = {(entry["parent"], entry["child"]): yaml_matrix(entry) for entry in extrinsics}
        cam = "camera_front"
        base_from_link = transforms["base_link", f"{cam}_link"]
        self.base_from_color = (
            base_from_link @ transforms[f"{cam}_link", f"{cam}_color_frame"]
            @ transforms[f"{cam}_color_frame", f"{cam}_color_optical_frame"]
        )
        depth_k, depth_dist = calibration_matrix(self.root / cam / "depth" / "intrinsics.yaml")
        self.k, dist = calibration_matrix(self.root / cam / "color" / "intrinsics.yaml")
        self.intrinsics = (self.k[0, 0], self.k[1, 1], self.k[0, 2], self.k[1, 2])
        self.maps = cv2.initUndistortRectifyMap(self.k, dist, None, self.k, (640, 480), cv2.CV_16SC2)
        base_from_depth = (
            base_from_link @ transforms[f"{cam}_link", f"{cam}_depth_frame"]
            @ transforms[f"{cam}_depth_frame", f"{cam}_depth_optical_frame"]
        )
        self.color_from_depth = np.linalg.inv(self.base_from_color) @ base_from_depth
        wheel = np.loadtxt(self.root / "dingo_velocity_controller" / "odom.txt", usecols=range(8))
        _, keep = np.unique(wheel[:, 0], return_index=True)
        self.odom = wheel[keep]
        self.odom_rotations = Slerp(self.odom[:, 0], Rotation.from_quat(self.odom[:, 4:8]))
        # Derive rays from the depth camera calibration before z-buffering them into RGB.
        xy = np.indices((480, 640))[::-1].reshape(2, -1).T.astype(np.float64)
        undistorted = cv2.undistortPoints(xy, depth_k, depth_dist).reshape(-1, 2)
        self.rays = np.column_stack((undistorted, np.ones(len(xy))))

    def pose(self, timestamp):
        base = np.eye(4)
        base[:3, :3] = self.rotations(timestamp).as_matrix()
        base[:3, 3] = [np.interp(timestamp, self.truth_time, p) for p in self.positions.T]
        return base @ self.base_from_color

    def odometry(self, frames):
        estimates = {}
        for frame in frames:
            timestamp = self.times[frame]
            base = np.eye(4)
            base[:3, :3] = self.odom_rotations(timestamp).as_matrix()
            base[:3, 3] = [np.interp(timestamp, self.odom[:, 0], position)
                           for position in self.odom[:, 1:4].T]
            estimates[frame] = base @ self.base_from_color
        return estimates

    def rgb_index(self, frame):
        return frame

    def rectify(self, path):
        with Image.open(self.root / path) as image:
            pixels = np.asarray(image.convert("RGB"))
        return cv2.remap(pixels, *self.maps, interpolation=cv2.INTER_LINEAR)

    @lru_cache(maxsize=16)
    def depth(self, frame):
        with Image.open(self.root / self.depth_paths[frame]) as image:
            depth = np.asarray(image, dtype=np.float64).ravel() * 0.001
        valid = (depth > 0.10) & (depth < 12)
        points = self.rays[valid] * depth[valid, None]
        transform = self.color_from_depth
        points = points @ transform[:3, :3].T + transform[:3, 3]
        points = points[points[:, 2] > 0.10]
        z = points[:, 2]
        fx, fy, cx, cy = self.intrinsics
        u = np.rint(fx * points[:, 0] / z + cx).astype(int)
        v = np.rint(fy * points[:, 1] / z + cy).astype(int)
        inside = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        aligned = np.full(640 * 480, np.inf)
        np.minimum.at(aligned, v[inside] * 640 + u[inside], z[inside])
        aligned[np.isinf(aligned)] = 0
        return np.rint(aligned.reshape(480, 640) * 1000).astype(np.uint16)

    def observations(self, frames, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            timestamp = self.times[frame]
            if not self.truth_time[0] <= timestamp <= self.truth_time[-1]:
                raise ValueError(f"frame {frame} outside published pose interval")
            path = directory / f"{frame}.png"
            if not path.is_file():
                Image.fromarray(self.depth(frame)).save(path)
            yield self.pose(timestamp), path


def compare(pair, root, tools, work):
    sequences = {config["sequence"]: Visit(root, pair["name"], config["sequence"])
                 for config in pair["visits"]}
    visits, scoring, runs = [], [], []
    intrinsics = next(iter(sequences.values())).intrinsics
    for i, config in enumerate(pair["visits"]):
        sequence = sequences[config["sequence"]]
        if not np.allclose(sequence.intrinsics, intrinsics):
            raise ValueError("front camera calibration varies between visits")
        held_out = {item["frame"] for item in config["labels"]}
        eligible = select_frames(config["window"], held_out, 1)
        frames = select_frames(config["window"], held_out, config.get("every", 5))
        samples = list(sequence.observations(eligible, work / sequence.root.name))
        visits.append((sequence, samples))
        scoring.append([entry for entry in samples if int(entry[1].stem) in frames])
        runs.append(export_visit(sequence, samples, intrinsics, work / f"visit_{i}",
                                 tools / "tum_rgbd_save_map", odometry=sequence.odometry(eligible)))

    kwargs = {"diagnostics": True, "intrinsics": intrinsics}
    result = {
        "sample_intervals": [
            {"sequence": config["sequence"], "first_rgb_timestamp_s": float(sequence.times[config["window"][0]]),
             "last_rgb_timestamp_s": float(sequence.times[config["window"][1]])}
            for config, (sequence, _) in zip(pair["visits"], visits, strict=True)
        ],
        "reference": evaluate_map(pair, sequences, work, 0.07, 0.10, 2, **kwargs),
    }
    by_variant = {}
    for name, refine in (("session_alignment", False), ("all_loops", True)):
        poses, alignment = merge_trial(visits, runs, work / name, tools / "basalt_map_merge", refine=refine)
        by_variant[name] = poses
        result[name] = evaluate_map(pair, sequences, work, 0.07, 0.10, 2, poses=poses, **kwargs)
        result[name]["alignment"] = alignment
        result[name]["scoring_pose_errors"] = [pose_rmse(rows, estimated)
                                                for rows, estimated in zip(scoring, poses, strict=True)]

    endpoints = [set() for _ in visits]
    for group in result["all_loops"]["alignment"]["session_alignment"]["keyframe_refinement"]["pairs"]:
        for a, b in group.get("image_pairs", []):
            endpoints[group["base"]].add(runs[group["base"]][1][a - 1][0])
            endpoints[group["query"]].add(runs[group["query"]][1][b - 1][0])
    available = []
    for i, poses in enumerate(by_variant["session_alignment"]):
        withheld = endpoints[i] | {int(path.stem) for _, path in scoring[i]}
        available.append({frame: pose for frame, pose in poses.items() if frame not in withheld})
    pairs = validation_pairs(*available)
    depths = [{frame: (visits[i][0].depth(frame) * 0.001, np.eye(4))
               for frame in {pair[i] for pair in pairs}} for i in range(2)]
    agreements = {name: score_validation_depth(pairs, poses, depths, (intrinsics, intrinsics))
                  for name, poses in by_variant.items()}
    result["selection"] = {"frame_pairs": pairs, "mean_agreement": agreements,
                           "preferred": select_depth_variant(agreements, len(pairs))}
    paths = [{int(path.stem): path for _, path in samples} for _, samples in visits]
    result["change_selection"] = select_by_local_change(pairs, by_variant, paths, intrinsics, agreements)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--map-tools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    pairs = json.loads(args.annotations.read_text())["maps"]
    with TemporaryDirectory(prefix="liloc-") as directory:
        result = {pair["name"]: compare(pair, args.data, args.map_tools, Path(directory) / pair["name"])
                  for pair in pairs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
