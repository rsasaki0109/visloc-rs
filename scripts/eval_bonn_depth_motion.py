#!/usr/bin/env python3
"""Score a causal depth fallback for undecided Basalt motion boxes."""

import argparse
import json
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from bonn_estimated_poses import rgbd_pose
from eval_bonn_map_changes import (
    INTRINSICS, Sequence, annotation_proposals, motion_box_masks, points_at,
    polygon_mask, precision_recall,
)
from openloris_map_changes import visibility

DEPTH_STRIDE = 4
REFERENCE_INTERVAL = 0.22
REFERENCE_AGE = 5.0
CLEAR_MARGIN_M = 0.15
PREDICTION_HORIZON = 0.35


def extrapolate(history, time):
    if len(history) < 2:
        return None
    first_t, first = history[-2][:2]
    last_t, last = history[-1][:2]
    span = last_t - first_t
    if span <= 0 or span > 0.35 or time - last_t > PREDICTION_HORIZON:
        return None
    ratio = (time - last_t) / span
    delta = Rotation.from_matrix(first[:3, :3].T @ last[:3, :3]).as_rotvec()
    estimate = last.copy()
    estimate[:3, :3] @= Rotation.from_rotvec(delta * ratio).as_matrix()
    estimate[:3, 3] += (last[:3, 3] - first[:3, 3]) * ratio
    return estimate


def background_poses(sequence, recorded, proposals):
    detector = cv2.SIFT_create(nfeatures=1200)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    y, x = np.mgrid[:480, :640]
    poses, status, history = {}, {}, []
    for frame in recorded:
        time = sequence.times[frame]
        try:
            rgb_id = sequence.rgb_index(frame)
        except ValueError:
            status[frame] = "rgb_unavailable"
            continue
        gray = cv2.cvtColor(sequence.rectify(sequence.rgb_paths[rgb_id]), cv2.COLOR_RGB2GRAY)
        available = ~motion_box_masks(proposals.get(frame, []), x, y)
        current = detector.detectAndCompute(gray, available.astype(np.uint8) * 255)
        measured = None
        if current[1] is not None:
            if not history:
                measured = np.eye(4)
            else:
                recent = history[-6:]
                older = history[:-6]
                anchors = older[::max(1, len(older) // 6)][:7]
                for previous_t, previous_pose, previous_features, previous_depth in reversed(anchors + recent):
                    try:
                        relative, support = rgbd_pose(previous_features, current, previous_depth, INTRINSICS, matcher)
                    except (ValueError, cv2.error):
                        continue
                    dt = time - previous_t
                    angle = Rotation.from_matrix(relative[:3, :3]).magnitude()
                    required_support = 25 if dt <= 0.5 else 40
                    distance = np.linalg.norm(relative[:3, 3])
                    if support >= required_support and distance <= 1.8 * dt + 0.02 and angle <= 2.5 * dt + 0.04:
                        measured = previous_pose @ relative
                        break
        if measured is not None:
            depth = sequence.depth(frame)
            # Reference points inside a current proposal cannot localize future frames.
            depth[~available] = 0
            history.append((time, measured, current, depth))
            poses[frame] = measured
            status[frame] = "measured"
            continue
        prediction = extrapolate(history, time)
        if prediction is not None:
            poses[frame] = prediction
        status[frame] = "predicted" if prediction is not None else "unavailable"
    return poses, status


def background_rows(sequence, work, required, proposals):
    work.mkdir(parents=True)
    reference_frames, time = [], -np.inf
    for i, stamp in enumerate(sequence.times):
        if stamp - time >= REFERENCE_INTERVAL and (sequence.root / sequence.depth_paths[i]).is_file():
            reference_frames.append(i)
            time = stamp
    frames = sorted(set(reference_frames) | required)
    rgbd = dict((int(path.stem), (pose, path))
                for pose, path in sequence.observations(frames, work))
    # Apply supplied rectangles as masks only when they exist at that reference time.
    masks = work / "proposal_masks"
    masks.mkdir()
    y, x = np.mgrid[:480, :640]
    for frame in reference_frames:
        if frame in proposals and frame in rgbd:
            mask = motion_box_masks(proposals[frame], x, y)
            Image.fromarray(mask.astype(np.uint8)).save(masks / rgbd[frame][1].name)
    return rgbd, [i for i in reference_frames if i in rgbd], masks


def depth_pixels(sequence, frame, pose, references, mask_dir, candidate, x, y):
    prediction = np.zeros(x.shape, dtype=bool)
    iy, ix = np.nonzero(candidate)
    points, z = points_at(pose, sequence.depth(frame), x[iy, ix], y[iy, ix])
    valid = (z >= 0.3) & (z <= 4.0)
    hits, free = visibility(points[valid], references, INTRINSICS, CLEAR_MARGIN_M, mask_dir)
    supported = (free >= 2) & (free >= 2 * hits)
    prediction[iy[valid][supported], ix[valid][supported]] = True
    return prediction


def evaluate(sequence, labels, baseline, work):
    labels = {item["frame"]: item for item in labels}
    frames = {row["frame"]: row for row in baseline["labeled_frames"]}
    if frames.keys() != labels.keys() or any(row["proposed_object_ids"] != labels[i]["object_ids"]
                                             for i, row in frames.items()):
        raise ValueError("the Basalt report does not match the supplied labels")
    proposals = {i: list(annotation_proposals(item)) for i, item in labels.items()}
    rgbd, ref_frames, mask_dir = background_rows(sequence, work, set(labels), proposals)
    estimated, status = background_poses(sequence, sorted(rgbd), proposals)
    pose_sources = {
        "estimated": estimated,
        "mocap": {i: pose for i, (pose, _) in rgbd.items()},
    }
    y, x = np.mgrid[:480:DEPTH_STRIDE, :640:DEPTH_STRIDE]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    valid_image = (10 <= x) & (x < 630) & (10 <= y) & (y < 470)
    truth, outputs, details = [], {"baseline": [], **{name: [] for name in pose_sources}}, []
    for frame, item in labels.items():
        row = frames[frame]
        mask = polygon_mask(item["moving"]).astype(np.uint8)
        valid = valid_image & (cv2.dilate(mask, kernel) == cv2.erode(mask, kernel))[y, x]
        valid &= ~polygon_mask(item.get("ignore", []))[y, x]
        truth.append((mask[y, x] != 0)[valid])
        base = motion_box_masks([r for r in proposals[frame] if r[0] in row["confirmed_object_ids"]], x, y)
        fallback_ids = {d["object_id"] for d in row["decisions"]
                        if not d["moving"] and d["evidence"] in ("missing_background", "missing_object_and_background")}
        candidate = motion_box_masks([r for r in proposals[frame] if r[0] in fallback_ids], x, y) & ~base
        prior = [i for i in ref_frames if 0 < sequence.times[frame] - sequence.times[i] <= REFERENCE_AGE]
        outputs["baseline"].append(base[valid])
        detail = {"frame": frame, "fallback_proposals": sorted(fallback_ids), "estimated_pose": status.get(frame),
                  "results": {}}
        for name, poses in pose_sources.items():
            # Use measured poses for the reference map. Extrapolation serves only at the query.
            references = [(poses[i], rgbd[i][1]) for i in prior
                          if i in poses and (name == "mocap" or status[i] == "measured")]
            fallback = np.zeros_like(base)
            if candidate.any() and frame in poses and len(references) >= 2:
                fallback = depth_pixels(sequence, frame, poses[frame], references, mask_dir, candidate, x, y)
            outputs[name].append((base | fallback)[valid])
            positives = truth[-1]
            detail["results"][name] = {
                "reference_frames": len(references),
                "recovered_moving_pixels": int(np.count_nonzero(positives & fallback[valid])),
                "removed_static_pixels": int(np.count_nonzero(~positives & fallback[valid])),
            }
        details.append(detail)
    return {"sequence": sequence.root.name, "stride_px": DEPTH_STRIDE, "clear_margin_m": CLEAR_MARGIN_M,
            "estimated_pose_coverage": dict(Counter(status.values())), "frames": details,
            "scores": {name: precision_recall(np.concatenate(truth), np.concatenate(prediction))
                       for name, prediction in outputs.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    annotations = json.loads(args.annotations.read_text())["motion"]
    baseline = json.loads(args.baseline.read_text())["motion"]
    seq = Sequence(args.data / f"rgbd_bonn_{annotations['sequence']}")
    with TemporaryDirectory(prefix="bonn-depth-") as work:
        result = evaluate(seq, annotations["labels"], baseline, Path(work) / "depth")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
