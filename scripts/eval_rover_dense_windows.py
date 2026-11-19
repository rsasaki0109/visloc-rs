#!/usr/bin/env python3
"""Check whether denser RGB sampling creates a bridge between ROVER submaps."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import yaml

from eval_rover_local_bridges import generate_window
from eval_rover_seasonal_geometry import assemble_tracks, reconstruct
from eval_rover_seasonal_overlap import Capture, EfficientLoFTR, epipolar_inliers, mutual_sift_matches


def generate_transfer(capture, offsets, k, distortion, matcher, out):
    images = [capture.image(t)[1] for t in offsets]
    sift = [capture.features(t) for t in offsets]

    def matches():
        for i, image in enumerate(images):
            for j in range(i + 1, min(i + 4, len(images))):
                left, right = mutual_sift_matches(sift[i], sift[j])
                if j - i <= 2 and epipolar_inliers(left, right, k, distortion).sum() < 30:
                    left, right = matcher.match(image, images[j])
                yield i, j, left, right

    with TemporaryDirectory(prefix="rover-transfer-") as directory:
        work = Path(directory)
        points, pairs, _ = assemble_tracks(images, matches(), k, distortion, len(images))
        models = reconstruct(images, points, pairs, k, distortion, work)
        out.mkdir(parents=True)
        for identifier, model in models.items():
            path = out / str(identifier)
            path.mkdir()
            model.write(path)


def evaluate(cached, windows, baseline):
    import pycolmap

    original = {str(model["id"]): model["registered_offsets_s"] for model in baseline["models"]}
    output = {}
    for name, config in windows.items():
        times = config["seconds"]
        lookup = [(season, stamp) for season, group in enumerate(times) for stamp in group]
        results = []
        for path in sorted(p for p in (cached / name).iterdir() if p.is_dir()):
            model = pycolmap.Reconstruction(path)
            registered = [[], []]
            for identifier in sorted(model.reg_image_ids()):
                season, stamp = lookup[identifier - 1]
                registered[season].append(stamp)
            support = [len(set(registered[season]) & set(original[node][season]))
                       for season, node in config["targets"]]
            results.append({"registered_offsets_s": registered, "shared_frames": support,
                            "enough_shared_frames_for_sim3_test": min(support) >= 4})
        output[name] = {"samples_per_season": list(map(len, times)), "target_submaps": config["targets"],
                        "local_models": results}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("summer", "winter", "calibration", "site-results", "windows", "local-models", "out"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--efficient-loftr", type=Path, help="build the local models")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)
    windows = json.loads(args.windows.read_text())
    baseline = json.loads(args.site_results.read_text())
    if args.efficient_loftr:
        captures = [Capture(root, "realsense_D435i") for root in (args.summer, args.winter)]
        camera = yaml.safe_load(args.calibration.read_text())["Cam_Intrinsics"]
        fx, fy, cx, cy = camera["intrinsics"]
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        distortion = np.array(camera["distortion_coeffs"])
        matcher = EfficientLoFTR(args.efficient_loftr, 384)
        for name, config in windows.items():
            cv2.setRNGSeed(0)
            destination = args.local_models / name
            times = config["seconds"]
            if times[1]:
                generate_window(captures, times, k, distortion, matcher, destination)
            else:
                generate_transfer(captures[0], times[0], k, distortion, matcher, destination)
    scores = evaluate(args.local_models, windows, baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(scores, indent=2) + "\n")


if __name__ == "__main__":
    main()
