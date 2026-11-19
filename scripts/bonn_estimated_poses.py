"""Metric RGB-D odometry and Basalt multi-session refinement for the Bonn benchmark."""

import json
import subprocess

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rgbd_map_cloud import keyframe_poses


def rgbd_pose(previous, current, depth, intrinsics, matcher):
    before, before_descriptors = previous
    now, now_descriptors = current
    matches = matcher.knnMatch(before_descriptors, now_descriptors, k=2)
    kept = [m for pair in matches if len(pair) == 2 for m, n in [pair] if m.distance < 0.75 * n.distance]
    if len(kept) < 12:
        raise ValueError("insufficient RGB-D matches")
    xy = np.array([before[m.queryIdx].pt for m in kept])
    uv = np.rint(xy).astype(int).clip([0, 0], [depth.shape[1] - 1, depth.shape[0] - 1])
    z = depth[uv[:, 1], uv[:, 0]] * 0.001
    valid = (z >= 0.3) & (z <= 4.0)
    fx, fy, cx, cy = intrinsics
    k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    y = np.array([now[m.trainIdx].pt for m in kept])[valid]
    z = z[valid]
    if len(z) < 12:
        raise ValueError("insufficient depth-backed RGB-D matches")
    x = np.column_stack(((xy[valid, 0] - cx) * z / fx, (xy[valid, 1] - cy) * z / fy, z))
    ok, rot, trans, inliers = cv2.solvePnPRansac(
        x, y, k, None, iterationsCount=200, reprojectionError=2.5, flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or len(inliers) < 12:
        raise ValueError("insufficient RGB-D PnP inliers")
    rotation, translation = cv2.solvePnPRefineLM(x[inliers[:, 0]], y[inliers[:, 0]], k, None, rot, trans)
    current_from_previous = np.eye(4)
    current_from_previous[:3, :3] = cv2.Rodrigues(rotation)[0]
    current_from_previous[:3, 3] = translation.ravel()
    return np.linalg.inv(current_from_previous), len(inliers)


def run_binary(binary, *arguments):
    completed = subprocess.run([str(binary.resolve()), *map(str, arguments)], capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"{binary.name}: {completed.stdout}\n{completed.stderr}")


def export_visit(sequence, observations, intrinsics, work, map_builder):
    work.mkdir(parents=True)
    cv2.setRNGSeed(0)
    detector = cv2.SIFT_create(nfeatures=1800)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    camera = np.eye(4)
    previous = previous_depth = None
    timestamps, trajectory, manifest, inliers = [], [], [], []
    epoch = sequence.times[int(observations[0][1].stem)] - 1.0
    for _, path in observations:
        frame = int(path.stem)
        stamp = sequence.times[frame] - epoch
        rgb = sequence.rectify(sequence.rgb_paths[sequence.rgb_index(frame)])
        grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        depth = sequence.depth(frame)
        current = detector.detectAndCompute(grey, None)
        if current[1] is None:
            raise ValueError(f"{sequence.root.name} frame {frame}: no RGB features")
        if previous is not None:
            try:
                step, support = rgbd_pose(previous, current, previous_depth, intrinsics, matcher)
            except ValueError as error:
                raise ValueError(f"{sequence.root.name} frame {frame}: {error}") from error
            camera = camera @ step
            inliers.append(support)
        previous, previous_depth = current, depth
        q = Rotation.from_matrix(camera[:3, :3]).as_quat()
        trajectory.append([stamp, *camera[:3, 3], *q])
        timestamps.append((frame, stamp))
        Image.fromarray(grey).save(work / f"rgb_{frame}.png")
        Image.fromarray(depth).save(work / f"depth_{frame}.png")
        manifest.append((stamp, frame))
    (work / "rgb.txt").write_text("".join(f"{stamp:.9f} rgb_{frame}.png\n" for stamp, frame in manifest))
    (work / "depth.txt").write_text("".join(f"{stamp:.9f} depth_{frame}.png\n" for stamp, frame in manifest))
    trajectory_file = work / "vo.tum"
    # Bracket the manifest timestamps to account for formatting and the export helper.
    first = [trajectory[0][0] - 0.001, *trajectory[0][1:]]
    last = [trajectory[-1][0] + 0.001, *trajectory[-1][1:]]
    np.savetxt(trajectory_file, [first, *trajectory, last], fmt="%.12f")
    map_path = work / "map"
    camera_args = [value for name, number in zip(("fx", "fy", "cx", "cy"), intrinsics, strict=True)
                   for value in (f"--{name}", number)]
    run_binary(
        map_builder, "--seq-dir", work, "--trajectory", trajectory_file, "--out", map_path,
        "--keyframe-m", "0", "--keyframe-deg", "0", "--depth-scale", "0.001",
        "--k1", "0", "--k2", "0", "--p1", "0", "--p2", "0", "--k3", "0", *camera_args,
    )
    return map_path, timestamps, inliers


def loop_closed_poses(visits, intrinsics, work, map_builder, map_merge, repeats):
    cv2.setNumThreads(1)
    runs = [export_visit(sequence, observations, intrinsics, work / str(i), map_builder)
            for i, (sequence, observations) in enumerate(visits)]
    for trial in range(repeats):
        yield merge_trial(visits, runs, work / f"merged_{trial}", map_merge)


def merge_trial(visits, runs, merged, map_merge):
    run_binary(
        map_merge, "--base", runs[0][0], "--add", runs[1][0], "--out", merged,
        "--optimize-sessions", "--refine-keyframes", "--threads", "1", "--min-inliers", "10",
    )
    estimated = [keyframe_poses(merged, session=i) for i in range(len(runs))]
    for matched, (_, times, _) in zip(estimated, runs, strict=True):
        if len(matched) != len(times) or any(
            abs(frame_time - estimate_time) > 0.001
            for (_, frame_time), (estimate_time, _) in zip(times, matched, strict=True)
        ):
            raise ValueError(f"merged map pose coverage differs from inputs ({len(matched)}/{len(times)})")
    # One reference pose defines the global gauge. Both visits then use the
    # same transform, so cross-session registration errors remain in the score.
    anchor = visits[0][1][0][0] @ np.linalg.inv(estimated[0][0][1])
    aligned, errors = [], []
    for (_, observations), matched, (_, times, _) in zip(visits, estimated, runs, strict=True):
        aligned.append({frame: anchor @ pose for (frame, _), (_, pose) in zip(times, matched, strict=True)})
        delta = np.stack([np.linalg.inv(gt) @ pred
                          for (gt, _), pred in zip(observations, aligned[-1].values(), strict=True)])
        angles = np.degrees(Rotation.from_matrix(delta[:, :3, :3]).magnitude())
        errors.append({
            "translation_rmse_m": float(np.sqrt(np.mean(np.sum(delta[:, :3, 3] ** 2, axis=1)))),
            "rotation_rmse_deg": float(np.sqrt(np.mean(angles ** 2))),
        })
    summary = json.loads((merged / "map.json").read_text())
    return aligned, {
        "pose_errors": errors, "minimum_odometry_inliers": [min(run[2]) for run in runs],
        "session_alignment": summary["session_alignment"],
    }
