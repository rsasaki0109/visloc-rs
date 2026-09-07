#!/usr/bin/env python3
"""Independently audit text PINHOLE models; print JSON, never modify inputs.

Recompute Euclidean pixel errors from serialized poses and landmarks. Validate
both directions of every track reference and distinguish poses from support.
Only PINHOLE is accepted: unsupported distortion must not be silently ignored.
"""

import argparse
import json
import math
from pathlib import Path


def rows(path):
    with path.open() as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                yield line.split()


def finite(values):
    result = tuple(map(float, values))
    if not all(map(math.isfinite, result)):
        raise ValueError("nonfinite model value")
    return result


def rotation(q):
    w, x, y, z = finite(q)
    norm = math.sqrt(w*w + x*x + y*y + z*z)
    if abs(norm - 1) > 1e-6:
        raise ValueError("non-unit quaternion")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return ((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)),
            (2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)),
            (2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)))


def audit(model):
    cameras = {}
    for row in rows(model / "cameras.txt"):
        camera_id = int(row[0])
        if camera_id in cameras or len(row) != 8 or row[1] != "PINHOLE":
            raise ValueError("duplicate or unsupported camera (PINHOLE required)")
        cameras[camera_id] = finite(row[4:])
    images = {}
    names = set()
    with (model / "images.txt").open() as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = line.split()
            if len(row) != 10:
                raise ValueError("malformed image header")
            image_id, camera_id, name = int(row[0]), int(row[8]), row[9]
            if image_id in images or name in names or camera_id not in cameras:
                raise ValueError("duplicate image or unknown camera")
            names.add(name)
            keypoint_line = next(stream, None)
            if keypoint_line is None:
                raise ValueError("missing keypoint row")
            fields = keypoint_line.split()
            if len(fields) % 3:
                raise ValueError("malformed keypoint row")
            keypoints = [(finite(fields[i:i+2]), int(fields[i+2]))
                         for i in range(0, len(fields), 3)]
            images[image_id] = (rotation(row[1:5]), finite(row[5:8]),
                                cameras[camera_id], keypoints)
    seen = set()
    point_ids = set()
    errors_sum = 0.0
    maximum = 0.0
    stored_difference = 0.0
    behind = 0
    same_image_tracks = 0
    same_image_excess = 0
    support = dict.fromkeys(images, 0)
    for row in rows(model / "points3D.txt"):
        if len(row) < 12 or (len(row) - 8) % 2:
            raise ValueError("malformed point or track shorter than two")
        point_id = int(row[0])
        if point_id in point_ids or point_id < 0:
            raise ValueError("duplicate or invalid point id")
        point_ids.add(point_id)
        xyz = finite(row[1:4])
        stored = finite(row[7:8])[0]
        point_errors = []
        track_images = set()
        for offset in range(8, len(row), 2):
            image_id, index = map(int, row[offset:offset+2])
            key = (image_id, index)
            if key in seen or index < 0:
                raise ValueError("duplicate or invalid observation")
            seen.add(key)
            track_images.add(image_id)
            rot, trans, camera, keypoints = images[image_id]
            xy, reverse_id = keypoints[index]
            if reverse_id != point_id:
                raise ValueError("point-to-image reference mismatch")
            coords = [sum(a*b for a, b in zip(axis, xyz)) + shift
                      for axis, shift in zip(rot, trans)]
            if coords[2] <= 0:
                behind += 1
            if coords[2] == 0:
                raise ValueError("zero-depth projection")
            fx, fy, cx, cy = camera
            error = math.hypot(fx*coords[0]/coords[2]+cx-xy[0],
                               fy*coords[1]/coords[2]+cy-xy[1])
            if not math.isfinite(error):
                raise ValueError("nonfinite reprojection")
            point_errors.append(error)
            support[image_id] += 1
        point_sum = math.fsum(point_errors)
        excess = len(point_errors) - len(track_images)
        same_image_tracks += int(excess > 0)
        same_image_excess += excess
        errors_sum += point_sum
        maximum = max(maximum, max(point_errors))
        stored_difference = max(stored_difference,
                                abs(stored - point_sum/len(point_errors)))
    for image_id, (_, _, _, keypoints) in images.items():
        for index, (_, point_id) in enumerate(keypoints):
            if point_id != -1 and (image_id, index) not in seen:
                raise ValueError("image-to-point reference mismatch")
    if not seen:
        raise ValueError("no supported observations")
    return {"model": str(model), "image_poses": len(images),
            "supported_images": sum(count > 0 for count in support.values()),
            "unsupported_image_ids": sorted(i for i, n in support.items() if not n),
            "landmarks": len(point_ids), "observations": len(seen),
            "mean_reprojection_px": errors_sum/len(seen),
            "max_reprojection_px": maximum,
            "max_stored_mean_difference_px": stored_difference,
            "nonpositive_depth_observations": behind,
            "tracks_with_multiple_keypoints_in_one_image": same_image_tracks,
            "excess_same_image_track_observations": same_image_excess,
            "bidirectional_references_valid": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="+")
    args = parser.parse_args()
    print(json.dumps([audit(path) for path in args.model], indent=2, allow_nan=False))
