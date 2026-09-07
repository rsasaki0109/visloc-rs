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


def audit_rig(images, identities, support, manifest):
    """Infer the body pose independently from each serialized sensor pose."""
    sensors, assignments, frame_sensors = {}, {}, set()
    for row in rows(manifest):
        if row[0] == "S" and len(row) == 16:
            sensor = int(row[1])
            if sensor in sensors:
                raise ValueError("duplicate rig sensor")
            sensors[sensor] = (int(row[2]), finite(row[5:9]),
                               rotation(row[9:13]), finite(row[13:16]))
        elif row[0] == "F" and len(row) == 4:
            frame, name, sensor = int(row[1]), row[2], int(row[3])
            if name in assignments or (frame, sensor) in frame_sensors:
                raise ValueError("duplicate rig image or frame/sensor")
            assignments[name] = (frame, sensor)
            frame_sensors.add((frame, sensor))
        else:
            raise ValueError("malformed rig manifest row")
    if not sensors or not assignments:
        raise ValueError("empty rig manifest")
    if any(sensor not in sensors for _, sensor in assignments.values()):
        raise ValueError("unknown rig sensor")
    frame_poses, frame_support = {}, {}
    max_center, max_angle = 0.0, 0.0
    for image_id, (name, camera_id) in identities.items():
        if name not in assignments:
            raise ValueError("image absent from rig manifest")
        frame, sensor = assignments[name]
        expected_id, expected_k, extrinsic_r, extrinsic_t = sensors[sensor]
        image_r, image_t, intrinsics, _ = images[image_id]
        if camera_id != expected_id or any(
                abs(a-b) > 1e-8 for a, b in zip(intrinsics, expected_k)):
            raise ValueError("rig camera calibration mismatch")
        # T_rig<-world = inverse(T_sensor<-rig) * T_sensor<-world.
        body_r = tuple(tuple(sum(extrinsic_r[k][i]*image_r[k][j]
                                 for k in range(3)) for j in range(3))
                       for i in range(3))
        body_t = tuple(sum(extrinsic_r[k][i]*(image_t[k]-extrinsic_t[k])
                           for k in range(3)) for i in range(3))
        center = tuple(-sum(body_r[k][i]*body_t[k] for k in range(3))
                       for i in range(3))
        if frame in frame_poses:
            reference_r, reference_center = frame_poses[frame]
            max_center = max(max_center, math.dist(reference_center, center))
            cosine = (sum(reference_r[i][j]*body_r[i][j]
                          for i in range(3) for j in range(3)) - 1) / 2
            max_angle = max(max_angle, math.degrees(math.acos(max(-1, min(1, cosine)))))
        else:
            frame_poses[frame] = (body_r, center)
        frame_support[frame] = frame_support.get(frame, 0) + support[image_id]
    if max_center > 1e-4 or max_angle > 1e-3:
        raise ValueError("fixed sensor extrinsics violated by serialized poses")
    parent = {frame: frame for frame in frame_support}

    def find(frame):
        while parent[frame] != frame:
            parent[frame] = parent[parent[frame]]
            frame = parent[frame]
        return frame

    first_frame_for_point = {}
    for image_id, (name, _) in identities.items():
        frame, _ = assignments[name]
        for _, point_id in images[image_id][3]:
            if point_id == -1:
                continue
            first = first_frame_for_point.setdefault(point_id, frame)
            parent[find(frame)] = find(first)
    groups = {}
    for frame in parent:
        groups.setdefault(find(frame), []).append(frame)
    components = sorted((sorted(group) for group in groups.values()),
                        key=lambda group: (-len(group), group[0]))
    return {"rig_manifest": str(manifest), "rig_frames": len(frame_support),
            "supported_rig_frames": sum(n > 0 for n in frame_support.values()),
            "unsupported_rig_frame_ids": sorted(i for i, n in frame_support.items() if not n),
            "max_inferred_rig_center_disagreement_m": max_center,
            "max_inferred_rig_rotation_disagreement_deg": max_angle,
            "fixed_sensor_extrinsics_valid": True,
            "track_connected_components": len(components),
            "track_connected_component_sizes": [len(group) for group in components],
            "track_connected_component_frame_ranges": [[group[0], group[-1]]
                                                        for group in components]}


def audit(model, rig_manifest=None):
    cameras = {}
    for row in rows(model / "cameras.txt"):
        camera_id = int(row[0])
        if camera_id in cameras or len(row) != 8 or row[1] != "PINHOLE":
            raise ValueError("duplicate or unsupported camera (PINHOLE required)")
        cameras[camera_id] = finite(row[4:])
    images = {}
    identities = {}
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
            identities[image_id] = (name, camera_id)
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
    point_means_sum = 0.0
    maximum_point_mean = 0.0
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
        point_mean = point_sum / len(point_errors)
        point_means_sum += point_mean
        maximum_point_mean = max(maximum_point_mean, point_mean)
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
    report = {"model": str(model), "image_poses": len(images),
            "supported_images": sum(count > 0 for count in support.values()),
            "unsupported_image_ids": sorted(i for i, n in support.items() if not n),
            "landmarks": len(point_ids), "observations": len(seen),
            "mean_reprojection_px": errors_sum/len(seen),
            "observation_weighted_mean_reprojection_px": errors_sum/len(seen),
            "point_weighted_mean_reprojection_px": point_means_sum/len(point_ids),
            "max_track_mean_reprojection_px": maximum_point_mean,
            "max_reprojection_px": maximum,
            "max_stored_mean_difference_px": stored_difference,
            "nonpositive_depth_observations": behind,
            "tracks_with_multiple_keypoints_in_one_image": same_image_tracks,
            "excess_same_image_track_observations": same_image_excess,
            "bidirectional_references_valid": True}
    if rig_manifest is not None:
        report["rig"] = audit_rig(images, identities, support, rig_manifest)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, nargs="+")
    parser.add_argument("--rig-manifest", type=Path,
                        help="Optional S/F manifest with matching image names; validate fixed rig geometry")
    args = parser.parse_args()
    print(json.dumps([audit(path, args.rig_manifest) for path in args.model],
                     indent=2, allow_nan=False))
