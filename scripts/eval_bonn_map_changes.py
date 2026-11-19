#!/usr/bin/env python3
"""Evaluate RGB-D surface fusion with manually labeled Bonn RGB-D frames."""

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from bonn_estimated_poses import loop_closed_poses
from openloris_map_changes import classify, protected_clearance, surface_voxels, visibility
from rgbd_map_cloud import nearest, read_stamped


INTRINSICS = (542.822841, 542.576870, 315.593520, 237.756098)
FX, FY, CX, CY = INTRINSICS
K = np.array([[FX, 0, CX], [0, FY, CY], [0, 0, 1]])
DISTORTION = np.array([0.039903, -0.099343, -0.000730, -0.000144, 0.0])
ROS_TO_MOCAP = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
MARKER_FROM_CAMERA = np.array([
    [1.0157, 0.1828, -0.2389],
    [0.0009, -0.8431, -0.6413],
    [-0.3009, 0.6147, -0.8085],
])
U, _, VT = np.linalg.svd(MARKER_FROM_CAMERA)
MARKER_FROM_CAMERA = U @ VT
MARKER_CAMERA_OFFSET = np.array([0.0113, -0.0098, 0.0111])


class Sequence:
    def __init__(self, root):
        self.root = root
        self.times, self.depth_paths = read_stamped(root / "depth.txt")
        self.rgb_times, self.rgb_paths = read_stamped(root / "rgb.txt")
        trajectory = np.loadtxt(root / "groundtruth.txt")
        order = np.argsort(trajectory[:, 0])
        self.pose_times, keep = np.unique(trajectory[order, 0], return_index=True)
        self.pose_rows = trajectory[order[keep]]
        self.rotations = Slerp(self.pose_times, Rotation.from_quat(self.pose_rows[:, 4:8]))
        self.maps = cv2.initUndistortRectifyMap(K, DISTORTION, None, K, (640, 480), cv2.CV_16SC2)

    def pose(self, time):
        matrix = np.eye(4)
        marker = ROS_TO_MOCAP @ self.rotations(time).as_matrix() @ ROS_TO_MOCAP
        position = np.array([np.interp(time, self.pose_times, self.pose_rows[:, i]) for i in (1, 2, 3)])
        matrix[:3, :3] = marker @ MARKER_FROM_CAMERA
        matrix[:3, 3] = ROS_TO_MOCAP @ position + marker @ MARKER_CAMERA_OFFSET
        return matrix

    def rectify(self, path):
        with Image.open(self.root / path) as source:
            return cv2.remap(np.asarray(source), *self.maps, interpolation=cv2.INTER_NEAREST)

    def rgb_index(self, frame):
        i = nearest(self.rgb_times, self.times[frame])
        if abs(self.rgb_times[i] - self.times[frame]) > 0.02:
            raise ValueError(f"{self.root.name}, frame {frame}: no RGB within 20 ms")
        return i

    def depth(self, frame):
        # A TUM PNG encodes 5 units/mm. The production reader expects millimetres.
        return np.rint(self.rectify(self.depth_paths[frame]) / 5).astype(np.uint16)

    def observations(self, frames, directory):
        directory.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            if not (self.root / self.depth_paths[frame]).is_file():
                continue  # Some published depth manifests include a missing PNG.
            time = self.times[frame]
            if self.pose_times[0] <= time <= self.pose_times[-1]:
                if abs(self.pose_times[nearest(self.pose_times, time)] - time) > 0.02:
                    continue
                path = directory / f"{frame}.png"
                if not path.exists():
                    Image.fromarray(self.depth(frame)).save(path)
                yield self.pose(time), path


def points_at(pose, depth, x, y, intrinsics=INTRINSICS):
    fx, fy, cx, cy = intrinsics
    z = depth[y, x] * 0.001
    local = np.column_stack(((x - cx) * z / fx, (y - cy) * z / fy, z))
    return local @ pose[:3, :3].T + pose[:3, 3], z


def polygon_mask(polygons):
    canvas = Image.new("1", (640, 480))
    draw = ImageDraw.Draw(canvas)
    for polygon in polygons:
        draw.polygon([tuple(point) for point in polygon], fill=1)
    return np.asarray(canvas)


def keys(points, voxel):
    return np.floor(points / voxel).astype(np.int32)


def precision_recall(truth, prediction):
    positives = np.count_nonzero(truth)
    negatives = len(truth) - positives
    tp = int(np.count_nonzero(truth & prediction))
    fp = int(np.count_nonzero(~truth & prediction))
    fn = int(positives) - tp
    tn = int(negatives) - fp
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
    }


def annotation_proposals(item):
    polygons = item["moving"] + item.get("parked", [])
    for object_id, polygon in zip(item["object_ids"], polygons, strict=True):
        x, y = np.asarray(polygon).T
        yield [object_id, 0, 1, int(x.min()), int(y.min()), int(x.max()), int(y.max())]


def motion_box_masks(rows, x, y):
    mask = np.zeros(x.shape, dtype=bool)
    for _, _, _, left, top, right, bottom in rows:
        mask |= (left <= x) & (x <= right) & (top <= y) & (y <= bottom)
    return mask


def write_proposals(path, annotations, frame_stamps, stamps, interpolate):
    tracks = defaultdict(list)
    rows = defaultdict(list)
    for item in annotations:
        stamp = frame_stamps[item["frame"]]
        for row in annotation_proposals(item):
            rows[stamp].append(row)
            tracks[row[0]].append((stamp, row))
    if interpolate:
        for track in tracks.values():
            for (start, left), (end, right) in zip(track, track[1:]):
                if end - start > 1_350_000_000:
                    continue
                if any(stamp not in (start, end) and start < stamp < end for stamp in frame_stamps.values()):
                    continue
                for stamp in stamps[np.searchsorted(stamps, start, side="right"):np.searchsorted(stamps, end)]:
                    t = (stamp - start) / (end - start)
                    box = [round(a + (b - a) * t) for a, b in zip(left[3:], right[3:], strict=True)]
                    rows[stamp].append([*left[:3], *box])
    with path.open("w") as stream:
        writer = csv.writer(stream)
        for stamp, boxes in sorted(rows.items()):
            writer.writerows([stamp, *row] for row in boxes)


def export_motion_boxes(sequence, annotations, basalt, work, interpolate=False, hold_ms=None, diagnostics=False):
    cam = work / "mav0" / "cam0"
    (cam / "data").mkdir(parents=True, exist_ok=True)
    imu = work / "mav0" / "imu0"
    imu.mkdir(exist_ok=True)
    # The rounded timestamp is shared by the RGB manifest and all box CSVs.
    stamps = [int(round((t - sequence.rgb_times[0]) * 1e9)) + 1_000_000_000 for t in sequence.rgb_times]
    # The manifest loader requires two rows. Keep parser padding outside every
    # camera interval; visual-only mode also discards the initialization sample.
    padding = (stamps[-1] + 1, stamps[-1] + 2)
    (imu / "data.csv").write_text("".join(f"{stamp},0,0,0,0,0,0\n" for stamp in padding))
    with (cam / "data.csv").open("w") as stream:
        writer = csv.writer(stream)
        for stamp, path in zip(stamps, sequence.rgb_paths, strict=True):
            filename = f"{stamp}.png"
            destination = cam / "data" / filename
            if not destination.is_file():
                Image.fromarray(sequence.rectify(path)).convert("L").save(destination)
            writer.writerow([stamp, filename])
    frame_stamps = {}
    for item in annotations:
        frame = item["frame"]
        frame_stamps[frame] = stamps[sequence.rgb_index(frame)]
    if len(set(frame_stamps.values())) != len(frame_stamps):
        raise ValueError("multiple annotated depths resolve to the same RGB frame")
    proposal_rows = work / "proposals.csv"
    detected_rows = work / "confirmed.csv"
    write_proposals(proposal_rows, annotations, frame_stamps, stamps, interpolate)
    trace_path = work / "decisions.json"
    flags = ["--object-decisions", str(trace_path)] if diagnostics else []
    if hold_ms is not None:
        flags.extend(["--object-continuity-hold-ms", str(hold_ms)])
    # The DS camera with xi=alpha=0 is pinhole. Unused IMU fields retain the
    # repository fixture defaults because the reader requires their schema.
    repo = Path(__file__).resolve().parents[1]
    calibration = json.loads(
        (repo / "pipelines/basalt/tests/fixtures/euroc_ds_calib_minimal.json").read_text()
    )["value0"]
    calibration.update(
        intrinsics=[{"camera_type": "ds", "intrinsics": dict(zip(
            ("fx", "fy", "cx", "cy", "xi", "alpha"), (*INTRINSICS, 0.0, 0.0), strict=True,
        ))}],
        T_imu_cam=[dict(px=0.0, py=0.0, pz=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0)],
        resolution=[[640, 480]], cam_time_offset_ns=0,
    )
    calibration_path = work / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    run_dir = work / "replay"
    completed = subprocess.run([
        str(basalt.resolve()), "--euroc-dir", str(work), "--calibration", str(calibration_path),
        "--config", str(repo / "configs/basalt/euroc_config_mono_vo.json"),
        "--out-dir", str(run_dir), "--mono-vo", "--threads", "1", "--no-trace", "--no-marg-data",
        "--object-boxes", str(proposal_rows), "--export-moving-boxes", str(detected_rows), *flags,
    ], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"Basalt replay failed: {completed.stderr}")
    by_frame = {frame: [] for frame in frame_stamps}
    by_stamp = {stamp: frame for frame, stamp in frame_stamps.items()}
    with detected_rows.open() as stream:
        for row in csv.reader(stream):
            if not row or row[0] == "timestamp_ns":
                continue
            frame = by_stamp.get(int(row[0]))
            if frame is not None:
                by_frame[frame].append([float(v) for v in row[1:]])
    summary = dict(line.split("=", 1) for line in (run_dir / "summary.txt").read_text().splitlines() if "=" in line)
    if summary.get("camera_mode") != "mono_visual_only" or summary.get("imu_samples_delivered") != "0":
        raise ValueError("expected a visual-only replay without any delivered IMU samples")
    decisions = defaultdict(list)
    if diagnostics:
        for row in json.loads(trace_path.read_text()):
            frame = by_stamp.get(row["timestamp_ns"])
            if frame is not None:
                decisions[frame].append(row)
    return by_frame, decisions, {key: int(summary[key]) for key in (
        "frames_processed", "imu_samples_delivered", "object_box_frames", "object_boxes", "moving_object_boxes",
        "undecided_object_boxes", "moving_object_dropped_observations",
    )}


def evaluate_motion(sequence, annotations, basalt, work, interpolate=False, hold_ms=None, diagnostics=False):
    confirmed, decisions, replay_stats = export_motion_boxes(
        sequence, annotations, basalt, work, interpolate, hold_ms, diagnostics,
    )
    missed = Counter()
    y, x = np.mgrid[:480, :640]
    # Conservative silhouettes are scored outside a 10 px band at hand-drawn
    # edges. The image border is omitted because of lens rectification.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    image_interior = (10 <= x) & (x < 630) & (10 <= y) & (y < 470)
    variants = ("no_exclusion", "all_proposals", "basalt_motion_confirmed")
    truths, predictions, frame_scores = [], {name: [] for name in variants}, []
    for item in annotations:
        truth = polygon_mask(item["moving"])
        mask = truth.astype(np.uint8)
        boundary = cv2.dilate(mask, kernel) != cv2.erode(mask, kernel)
        ignored = polygon_mask(item.get("ignore", []))
        valid = image_interior & ~boundary & ~ignored
        output = {
            "no_exclusion": np.zeros(truth.shape, dtype=bool),
            "all_proposals": motion_box_masks(annotation_proposals(item), x, y),
            "basalt_motion_confirmed": motion_box_masks(confirmed[item["frame"]], x, y),
        }
        truths.append(truth[valid])
        for name, prediction in output.items():
            predictions[name].append(prediction[valid])
        confirmed_ids = sorted(int(row[0]) for row in confirmed[item["frame"]])
        if diagnostics:
            by_id = {row["object_id"]: row for row in decisions[item["frame"]]}
            for object_id in item["object_ids"][:len(item["moving"])]:
                if object_id not in confirmed_ids:
                    missed[by_id.get(object_id, {}).get("evidence", "not_evaluated")] += 1
        frame_scores.append({
            "frame": item["frame"], "proposed_object_ids": item["object_ids"],
            "confirmed_object_ids": confirmed_ids,
            **({"decisions": decisions[item["frame"]]} if diagnostics else {}),
            **{name: precision_recall(truth[valid], output[name][valid]) for name in variants},
        })
    combined = np.concatenate(truths)
    return {
        "replay": replay_stats, "labeled_frames": frame_scores,
        **({"missed_moving_proposals": dict(missed)} if diagnostics else {}),
        **{name: precision_recall(combined, np.concatenate(predictions[name])) for name in variants},
    }


def labeled_voxels(sequence, annotations, voxel, intrinsics=INTRINSICS):
    labeled = {label: set() for label in ("stable", "box", "floor")}
    for item in annotations:
        frame = item["frame"]
        depth = sequence.depth(frame)
        pose = sequence.pose(sequence.times[frame])
        for label in labeled:
            y, x = np.nonzero(polygon_mask(item.get(label, [])))
            points, z = points_at(pose, depth, x, y, intrinsics)
            valid = (z >= 0.3) & (z <= 4.0)
            labeled[label].update(map(tuple, np.unique(keys(points[valid], voxel), axis=0)))
    return labeled


def floor_behind_box(floor, visits, sequences, voxel, intrinsics=INTRINSICS):
    fx, fy, cx, cy = intrinsics
    floor_keys = sorted(floor)
    points = (np.asarray(floor_keys).reshape(-1, 3) + 0.5) * voxel
    covered = np.zeros(len(points), dtype=bool)
    for visit in visits:
        sequence = sequences[visit["sequence"]]
        for item in visit["labels"]:
            if not item.get("box"):
                continue
            frame = item["frame"]
            pose = sequence.pose(sequence.times[frame])
            local = (points - pose[:3, 3]) @ pose[:3, :3]
            indices = np.flatnonzero((local[:, 2] >= 0.3) & (local[:, 2] <= 4.0))
            x, y, z = local[indices].T
            u = np.rint(fx * x / z + cx).astype(int)
            v = np.rint(fy * y / z + cy).astype(int)
            valid = (0 <= u) & (u < 640) & (0 <= v) & (v < 480)
            indices, u, v, z = indices[valid], u[valid], v[valid], z[valid]
            box_pixels = polygon_mask(item["box"])[v, u]
            depth = sequence.depth(frame)[v, u] * 0.001
            covered[indices[box_pixels & (depth >= 0.3) & (depth + 0.10 < z)]] = True
    return {key for key, hide in zip(floor_keys, covered, strict=True) if hide}


def select_frames(window, holdout, every=5):
    return [frame for frame in range(window[0], window[1] + 1, every) if frame not in holdout]


def explain_map_misses(truth, found, changed, hits, free, evidence, min_views, expected_direction):
    occupied_visit = 0 if expected_direction else 1
    clear_visit = 1 - occupied_visit
    occupied = (hits[occupied_visit] >= min_views) & (hits[occupied_visit] >= 2 * free[occupied_visit])
    occupied_detail, clear_detail = evidence[occupied_visit], evidence[clear_visit]
    remaining = truth & ~changed
    reasons = {}
    for name, condition in (
        ("missing_surface_candidate", ~found),
        ("outside_occupied_view", occupied_detail["projected"] < min_views),
        ("invalid_occupied_depth", occupied_detail["valid_center"] < min_views),
        ("insufficient_occupied_votes", ~occupied),
        ("outside_clear_view", clear_detail["projected"] < min_views),
        ("invalid_clear_depth", clear_detail["valid_center"] < min_views),
        ("patch_rejected_clearance", (free[clear_visit] < min_views) & (clear_detail["center_free"] >= min_views)),
        ("occluded_in_clear_visit", (free[clear_visit] < min_views) & (clear_detail["occluded"] >= min_views)),
        ("insufficient_clear_votes", free[clear_visit] < min_views),
        ("conflicting_clear_votes", np.ones_like(truth)),
    ):
        reasons[name] = int(np.count_nonzero(remaining & condition))
        remaining &= ~condition
    return reasons


def evaluate_map(pair, sequences, work, voxel, margin, min_views, *, stride=8, diagnostics=False, poses=None,
                 intrinsics=INTRINSICS, compare_removals=False):
    samples, labels, occupancy, missing = [], [], [], []
    for visit in pair["visits"]:
        sequence = sequences[visit["sequence"]]
        annotation_frames = {item["frame"] for item in visit["labels"]}
        frames = select_frames(visit["window"], annotation_frames, visit.get("every", 5))
        if not annotation_frames or not all(visit["window"][0] <= f <= visit["window"][1]
                                            for f in annotation_frames):
            raise ValueError(f"{pair['name']}: label frames must be within each visit's window")
        missing.append([frame for frame in frames if not (sequence.root / sequence.depth_paths[frame]).is_file()])
        sample = list(sequence.observations(frames, work / sequence.root.name))
        if poses is not None:
            visit_poses = poses[len(samples)]
            sample = [(visit_poses[int(path.stem)], path) for _, path in sample]
        if len(sample) < min_views:
            raise ValueError(f"{pair['name']}: insufficient synchronized frames")
        samples.append(sample)
        occupancy.append(surface_voxels(sample, intrinsics, stride, voxel, None))
        labels.append(labeled_voxels(sequence, visit["labels"], voxel, intrinsics))

    box = labels[0]["box"] | labels[1]["box"]
    floor = labels[0]["floor"] | labels[1]["floor"]
    stable = labels[0]["stable"] | labels[1]["stable"] | floor
    occluded = floor_behind_box(floor, pair["visits"], sequences, voxel, intrinsics)
    ambiguous = box & stable
    box -= ambiguous
    stable -= ambiguous
    occluded -= ambiguous
    if not box or not stable:
        raise ValueError(f"{pair['name']}: missing labeled box or stable voxels")
    gt_keys = sorted(box | stable)
    gt_points = (np.asarray(gt_keys) + 0.5) * voxel
    truth = np.array([key in box for key in gt_keys])
    gt_occluded = np.array([key in occluded for key in gt_keys])
    seen = np.array([[key in occupied for key in gt_keys] for occupied in occupancy])
    evidence = [{}, {}] if diagnostics else [None, None]
    votes = [visibility(gt_points, sample, intrinsics, margin, evidence=detail)
             for sample, detail in zip(samples, evidence, strict=True)]
    hits, free = (np.stack(v) for v in zip(*votes))
    found = seen.any(axis=0)
    # In the production CLI, an annotation outside the generated candidates would be missing.
    classified = classify(hits, free, min_views)
    classified[~found] = 0
    changed = classified == 2
    occupancy_difference = seen[0] ^ seen[1]
    cleared_in_second = (free[1] >= min_views) & (free[1] >= 2 * hits[1])
    expected_direction = pair["direction"] == "removal"
    correct_direction = (cleared_in_second == expected_direction) & changed
    removals = {}
    if compare_removals and expected_direction:
        removable = seen[0] & changed & cleared_in_second
        removals["baseline_removal"] = precision_recall(truth, removable)
        for policy, baseline_m in (("patch", 0.0), ("separated", 0.25)):
            confirmed = removable.copy()
            confirmed[removable] = protected_clearance(
                gt_points[removable], samples[1], intrinsics, min_baseline_m=baseline_m,
            )
            removals[f"{policy}_removal"] = precision_recall(truth, confirmed)
    return {
        **removals,
        "name": pair["name"],
        "direction": pair["direction"],
        "held_out_frames": [len(visit["labels"]) for visit in pair["visits"]],
        "sampled_depth_files_missing": missing,
        "synchronized_frames": [len(sample) for sample in samples],
        "ambiguous_excluded": len(ambiguous),
        "labeled_changes_outside_candidates": int(np.count_nonzero(truth & ~found)),
        "unknown": int(np.count_nonzero(classified == 0)),
        "intermittent": int(np.count_nonzero(classified == 3)),
        "occluded_static_voxels": int(np.count_nonzero(gt_occluded)),
        "occupancy_difference": precision_recall(truth, occupancy_difference),
        "occupancy_difference_occluded_false_positives": int(np.count_nonzero(gt_occluded & occupancy_difference)),
        "visibility": precision_recall(truth, changed),
        **({"missed_changes": explain_map_misses(truth, found, changed, hits, free, evidence, min_views,
                                                expected_direction)} if diagnostics else {}),
        "visibility_stable_correct": int(np.count_nonzero(~truth & (classified == 1))),
        "visibility_occluded_false_positives": int(np.count_nonzero(gt_occluded & changed)),
        "visibility_true_positives_with_correct_direction": int(np.count_nonzero(truth & correct_direction)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="directory containing extracted rgbd_bonn_* sequences")
    parser.add_argument("--basalt", type=Path, required=True, help="built basalt_euroc_vio_demo example")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--voxel", type=float, default=0.07)
    parser.add_argument("--depth-margin", type=float, default=0.10)
    parser.add_argument("--min-views", type=int, default=2)
    parser.add_argument("--map-repeats", type=int, default=1, help="independent Basalt map merge trials")
    parser.add_argument("--map-tools", type=Path, help="directory with tum_rgbd_save_map and basalt_map_merge")
    parser.add_argument("--map-ablation", action="store_true", help="diagnose changes across candidate strides and depth margins")
    parser.add_argument("--motion-ablation", action="store_true", help="diagnose and compare motion hold modes")
    args = parser.parse_args()
    if not np.isfinite([args.voxel, args.depth_margin]).all() or min(args.voxel, args.depth_margin, args.min_views) <= 0:
        parser.error("thresholds must be finite and positive")
    if args.map_repeats < 1:
        parser.error("--map-repeats must be positive")
    annotations = json.loads(args.annotations.read_text())
    names = {annotations["motion"]["sequence"]}
    names.update(visit["sequence"] for pair in annotations["maps"] for visit in pair["visits"])
    sequences = {name: Sequence(args.data / f"rgbd_bonn_{name}") for name in names}
    with TemporaryDirectory(prefix="visloc-bonn-") as directory:
        work = Path(directory)
        motion = evaluate_motion(sequences[annotations["motion"]["sequence"]], annotations["motion"]["labels"],
                                 args.basalt, work / "motion", diagnostics=args.motion_ablation)
        if args.motion_ablation:
            motion["interpolated"] = evaluate_motion(
                sequences[annotations["motion"]["sequence"]], annotations["motion"]["labels"],
                args.basalt, work / "motion", interpolate=True, diagnostics=True,
            )
            motion["interpolated_continuity_hold"] = evaluate_motion(
                sequences[annotations["motion"]["sequence"]], annotations["motion"]["labels"],
                args.basalt, work / "motion", interpolate=True, hold_ms=2000, diagnostics=True,
            )
        maps = [evaluate_map(pair, sequences, work, args.voxel, args.depth_margin, args.min_views,
                             diagnostics=args.map_ablation) for pair in annotations["maps"]]
        if args.map_ablation:
            for pair, metrics in zip(annotations["maps"], maps, strict=True):
                metrics["ablation"] = {
                    "stride_4": evaluate_map(pair, sequences, work, args.voxel, args.depth_margin, args.min_views,
                                             stride=4, diagnostics=True),
                    "margin_7cm": evaluate_map(pair, sequences, work, args.voxel, 0.07, args.min_views,
                                               diagnostics=True),
                    "margin_13cm": evaluate_map(pair, sequences, work, args.voxel, 0.13, args.min_views,
                                                diagnostics=True),
                }
        if args.map_tools:
            for pair, metrics in zip(annotations["maps"], maps, strict=True):
                visits = []
                for visit in pair["visits"]:
                    sequence = sequences[visit["sequence"]]
                    held_out = {item["frame"] for item in visit["labels"]}
                    scoring_frames = select_frames(visit["window"], held_out)
                    support_frames = select_frames(visit.get("tracking_window", visit["window"]), held_out)
                    frames = sorted(set(scoring_frames) | set(support_frames))
                    visits.append((sequence, list(sequence.observations(frames, work / sequence.root.name))))
                metrics["estimated_pose_trials"] = []
                trials = loop_closed_poses(
                    visits, INTRINSICS, work / pair["name"],
                    args.map_tools / "tum_rgbd_save_map", args.map_tools / "basalt_map_merge",
                    args.map_repeats,
                )
                for poses, metadata in trials:
                    estimated = evaluate_map(
                        pair, sequences, work, args.voxel, args.depth_margin, args.min_views,
                        diagnostics=args.map_ablation, poses=poses,
                    )
                    estimated["alignment"] = metadata
                    metrics["estimated_pose_trials"].append(estimated)
    result = {"voxel_m": args.voxel, "depth_margin_m": args.depth_margin, "min_views": args.min_views,
              "motion": motion, "maps": maps}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
