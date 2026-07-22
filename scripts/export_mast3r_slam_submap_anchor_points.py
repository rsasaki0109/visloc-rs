#!/usr/bin/env python3
"""Run two independent MASt3R-SLAM submaps and export loop-anchor pointmaps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


PINNED_REVISION = "6717231a2daf55d501a5824bbec43314d4fb77d9"
PATCH_NAME = "mast3r_slam_windows_6717231_submap_export.patch"
K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-slam-root", type=Path, required=True)
    parser.add_argument("--mav0", type=Path, required=True)
    parser.add_argument("--descriptor-dump", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--old-anchor", type=int, default=38)
    parser.add_argument("--new-anchor", type=int, default=462)
    parser.add_argument("--radius", type=int, default=16)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--extract-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()


def image_rows(mav0: Path) -> list[str]:
    rows = []
    with (mav0 / "cam0" / "data.csv").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split(",", 1)[1])
    return rows


def descriptor_keypoints(dump: Path, anchor: int) -> np.ndarray:
    with (dump / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        row = next(
            row for row in csv.DictReader(handle) if int(row["arrival_index"]) == anchor
        )
    return np.load(dump / row["keypoints_file"]).astype(np.float64) * 8.0


def prepare_side(
    mav0: Path, filenames: list[str], anchor: int, radius: int, side_dir: Path
) -> list[int]:
    arrivals = list(range(anchor - radius, anchor + radius + 1))
    if arrivals[0] < 0 or arrivals[-1] >= len(filenames):
        raise ValueError(f"submap [{arrivals[0]}, {arrivals[-1]}] is out of range")
    images = side_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    for local_index, arrival in enumerate(arrivals):
        raw = cv2.imread(
            str(mav0 / "cam0" / "data" / filenames[arrival]), cv2.IMREAD_GRAYSCALE
        )
        if raw is None:
            raise FileNotFoundError(f"image arrival {arrival}")
        # DPVO descriptor coordinates are on this same K-preserving undistortion.
        undistorted = cv2.undistort(raw, K, DIST)
        output = images / f"{local_index:06}.png"
        if not cv2.imwrite(str(output), undistorted):
            raise OSError(f"failed to write {output}")
    return arrivals


def write_calibration(path: Path) -> None:
    path.write_text(
        "width: 752\n"
        "height: 480\n"
        "calibration: [458.654, 457.296, 367.215, 248.375]\n",
        encoding="utf-8",
    )


def run_side(args: argparse.Namespace, side: str, anchor: int) -> tuple[list[int], Path]:
    side_dir = args.out_dir / side
    arrivals = prepare_side(args.mav0, image_rows(args.mav0), anchor, args.radius, side_dir)
    calibration = side_dir / "calibration.yaml"
    state = side_dir / "optimized_state.npz"
    write_calibration(calibration)
    command = [
        args.python,
        "main.py",
        "--dataset",
        str((side_dir / "images").resolve()),
        "--config",
        str(args.config.resolve()),
        "--calib",
        str(calibration.resolve()),
        "--no-viz",
        "--no-save-results",
        "--export-state",
        str(state.resolve()),
        "--export-frame-id",
        str(args.radius),
    ]
    with (side_dir / "run.log").open("w", encoding="utf-8") as log:
        subprocess.run(
            command,
            cwd=args.mast3r_slam_root,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return arrivals, state


def extract_anchor_points(
    state_path: Path, keypoints: np.ndarray, output: Path, local_anchor: int
) -> int:
    state = np.load(state_path)
    pointmap_ids = state["pointmap_frame_ids"]
    matches = np.flatnonzero(pointmap_ids == local_anchor)
    if len(matches) != 1:
        raise ValueError(f"anchor {local_anchor} missing or duplicated in {state_path}")
    slot = int(matches[0])
    height, width = (int(value) for value in state["image_shapes"][slot])
    points = state["pointmaps"][slot].reshape(height, width, 3)
    confidence = state["confidences"][slot].reshape(height, width)
    intrinsics = state["intrinsics"]

    # Inputs were already undistorted into the original K gauge. K_frame maps
    # those rays into the cropped 512-wide model image used by the pointmap.
    u = intrinsics[0, 0] * (keypoints[:, 0] - K[0, 2]) / K[0, 0] + intrinsics[0, 2]
    v = intrinsics[1, 1] * (keypoints[:, 1] - K[1, 2]) / K[1, 1] + intrinsics[1, 2]

    kept = 0
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# KEYPOINT_INDEX X Y Z CONFIDENCE\n")
        for index, (px, py) in enumerate(zip(u, v)):
            if not (0.0 <= px < width and 0.0 <= py < height):
                continue
            x = int(np.clip(round(px), 0, width - 1))
            y = int(np.clip(round(py), 0, height - 1))
            point = points[y, x]
            conf = float(confidence[y, x])
            if not np.isfinite(point).all() or point[2] <= 0.0 or not np.isfinite(conf):
                continue
            handle.write(
                f"{index} {float(point[0]):.9g} {float(point[1]):.9g} "
                f"{float(point[2]):.9g} {conf:.9g}\n"
            )
            kept += 1
    return kept


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    revision = git_revision(args.mast3r_slam_root)
    if revision != PINNED_REVISION:
        raise RuntimeError(f"MASt3R-SLAM revision {revision}, expected {PINNED_REVISION}")
    main_source = (args.mast3r_slam_root / "main.py").read_text(encoding="utf-8")
    evaluate_source = (args.mast3r_slam_root / "mast3r_slam" / "evaluate.py").read_text(
        encoding="utf-8"
    )
    if "--export-state" not in main_source or "def save_state_npz(" not in evaluate_source:
        raise RuntimeError(f"apply scripts/patches/{PATCH_NAME} to MASt3R-SLAM first")
    patch_path = Path(__file__).resolve().parent / "patches" / PATCH_NAME

    manifest = {
        "schema_version": 1,
        "official_revision": revision,
        "independent_process_per_side": True,
        "radius": args.radius,
        "config": str(args.config.resolve()),
        "config_sha256": sha256(args.config),
        "source_patch": str(patch_path),
        "source_patch_sha256": sha256(patch_path),
        "sides": {},
    }
    for side, anchor in (("old", args.old_anchor), ("new", args.new_anchor)):
        side_dir = args.out_dir / side
        if args.extract_only:
            arrivals = list(range(anchor - args.radius, anchor + args.radius + 1))
            state = side_dir / "optimized_state.npz"
        else:
            arrivals, state = run_side(args, side, anchor)
        output = args.out_dir / f"{side}_anchor_points.txt"
        kept = extract_anchor_points(
            state, descriptor_keypoints(args.descriptor_dump, anchor), output, args.radius
        )
        manifest["sides"][side] = {
            "anchor_arrival": anchor,
            "local_anchor_index": args.radius,
            "arrivals": arrivals,
            "state": str(state.resolve()),
            "state_sha256": sha256(state),
            "anchor_points": str(output.resolve()),
            "anchor_points_sha256": sha256(output),
            "points_kept": kept,
        }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
