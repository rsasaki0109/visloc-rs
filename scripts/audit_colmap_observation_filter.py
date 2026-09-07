#!/usr/bin/env python3
"""Read-only identity/removal audit of two COLMAP text image files.

Complement audit_colmap_pinhole_model.py: this checks a deletion-only transition,
not projection quality or points3D.txt consistency. Image rows are streamed;
point-ID mappings consume O(points) memory. Point IDs may be renumbered.
"""

import argparse
import json
from itertools import zip_longest
from pathlib import Path


def image_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            pose = line.split()
            if len(pose) != 10:
                raise ValueError("expected ten image header fields")
            points_line = next(stream, None)
            if points_line is None or points_line.startswith("#"):
                raise ValueError("missing POINTS2D row")
            points = points_line.split()
            if len(points) % 3:
                raise ValueError("invalid POINTS2D triple count")
            yield pose, points


def audit(before, after):
    old_to_new = {}
    new_to_old = {}
    old_points = set()
    image_ids = set()
    result = dict(images=0, changed_pose_images=0, observations_before=0,
                  observations_after=0, removed_observations=0,
                  supported_images_before=0, supported_images_after=0)
    for old_row, new_row in zip_longest(image_rows(before), image_rows(after)):
        if old_row is None or new_row is None:
            raise ValueError("image count changed")
        old_header, old_keys = old_row
        new_header, new_keys = new_row
        if old_header[0] != new_header[0] or old_header[8:] != new_header[8:]:
            raise ValueError("image identity/order/calibration assignment changed")
        image_id = int(old_header[0])
        if image_id in image_ids:
            raise ValueError("duplicate image ID")
        image_ids.add(image_id)
        if len(old_keys) != len(new_keys):
            raise ValueError("keypoint count changed")
        old_supported = new_supported = False
        for offset in range(0, len(old_keys), 3):
            if old_keys[offset:offset + 2] != new_keys[offset:offset + 2]:
                raise ValueError("keypoint coordinates/order changed")
            old_id = int(old_keys[offset + 2])
            new_id = int(new_keys[offset + 2])
            if old_id < -1 or new_id < -1:
                raise ValueError("invalid negative point ID")
            if old_id != -1:
                old_points.add(old_id)
                old_supported = True
                result["observations_before"] += 1
            if new_id == -1:
                result["removed_observations"] += old_id != -1
                continue
            if old_id == -1:
                raise ValueError("added an observation")
            if old_to_new.setdefault(old_id, new_id) != new_id:
                raise ValueError("split a track")
            if new_to_old.setdefault(new_id, old_id) != old_id:
                raise ValueError("merged tracks")
            new_supported = True
            result["observations_after"] += 1
        if old_supported and not new_supported:
            raise ValueError(f"lost support for image {image_id}")
        result["supported_images_before"] += old_supported
        result["supported_images_after"] += new_supported
        result["images"] += 1
        result["changed_pose_images"] += old_header[1:8] != new_header[1:8]
    result["points_before"] = len(old_points)
    result["points_after"] = len(new_to_old)
    result["removed_points"] = len(old_points) - len(old_to_new)
    result["deletion_only_identity_and_image_support_valid"] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path, help="baseline images.txt")
    parser.add_argument("after", type=Path, help="filtered images.txt")
    args = parser.parse_args()
    print(json.dumps(audit(args.before, args.after), indent=2))


if __name__ == "__main__":
    main()
