"""Measure person-depth exclusion using visual fixes and wheel propagation."""

import json

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from openloris_map_changes import visibility


def wheel_fallback(visit, visual, frames):
    wheel = visit.odometry(frames)
    propagated = {}
    correction = None
    for frame in frames:
        if frame in visual:
            correction = visual[frame] @ np.linalg.inv(wheel[frame])
        if correction is not None:
            propagated[frame] = correction @ wheel[frame]
    return propagated


def pose_errors(visit, estimates):
    translations, rotations = [], []
    for frame, pose in estimates.items():
        delta = np.linalg.inv(visit.pose(visit.times[frame])) @ pose
        translations.append(np.linalg.norm(delta[:3, 3]))
        rotations.append(np.degrees(Rotation.from_matrix(delta[:3, :3]).magnitude()))
    return {
        "localized": len(estimates),
        "translation_rmse_m": float(np.sqrt(np.mean(np.square(translations)))) if translations else None,
        "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotations)))) if rotations else None,
    }


def score_motion(label_path, visit, background, pose_variants):
    labels = json.loads(label_path.read_text())
    if labels["sequence"] != visit.root.name:
        raise ValueError("expected motion labels for the third visit")
    results = {}
    fx, fy, cx, cy = visit.intrinsics
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    for label in labels["frames"]:
        frame = label["frame"]
        polygon = np.asarray(label["polygon"], dtype=np.int32)
        depth = visit.depth(frame) * 0.001
        silhouette = cv2.fillPoly(np.zeros(depth.shape, dtype=np.uint8), [polygon], 1)
        positive = cv2.erode(silhouette, kernel).astype(bool)
        negative = ~cv2.dilate(silhouette, kernel).astype(bool)
        x, y, w, h = cv2.boundingRect(polygon)
        region = np.zeros(depth.shape, dtype=bool)
        region[max(0, y - 24):y + h + 24, max(0, x - 24):x + w + 24] = True
        valid = region & (positive | negative) & (depth >= 0.3) & (depth <= 4)
        v, u = np.nonzero(valid)
        z = depth[v, u]
        points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
        truth = positive[v, u]
        outcomes = {}
        for name, poses in pose_variants.items():
            pred = np.zeros(len(points), dtype=bool)
            if frame in poses:
                pose = poses[frame]
                world = points @ pose[:3, :3].T + pose[:3, 3]
                hits, free = visibility(world, background, visit.intrinsics, 0.10)
                pred = (free >= 2) & (free >= 2 * hits)
            outcomes[name] = {
                "has_pose": frame in poses, "tp": int(np.count_nonzero(pred & truth)),
                "fp": int(np.count_nonzero(pred & ~truth)), "fn": int(np.count_nonzero(~pred & truth)),
            }
        results[frame] = {"target": label["target"], "outcomes": outcomes, "scored_depth_pixels": len(points)}
    summary = {}
    for name, poses in pose_variants.items():
        outcomes = [r["outcomes"][name] for r in results.values()]
        tp, fp, fn = (sum(outcome[key] for outcome in outcomes) for key in ("tp", "fp", "fn"))
        summary[name] = {
            "labeled_frames_with_pose": sum(row["has_pose"] for row in outcomes),
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "f1": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None,
            **pose_errors(visit, poses),
        }
    return {"summary": summary, "frames": results}
