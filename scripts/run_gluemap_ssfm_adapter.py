#!/usr/bin/env python3
"""Run official GLUEMAP and normalize its final COLMAP model.

This adapter receives rectified images, calibration, and image timestamps only;
it never accepts pose ground truth. GLUEMAP's ``gt_intrinsics_path`` expects a
COLMAP reconstruction, so a calibration-only model with identity dummy poses is
created solely to associate the known rectified camera with every image name.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--images-path", type=Path, required=True)
    parser.add_argument("--calibration-path", type=Path, required=True)
    parser.add_argument("--timestamps-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--gluemap-command", default="gluemap-demo")
    return parser.parse_args()


def parse_rectified_p0(path: Path) -> tuple[float, float, float, float]:
    p0_line = next(
        (line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("P0:")),
        None,
    )
    if p0_line is None:
        raise ValueError(f"{path}: missing P0 calibration")
    values = [float(value) for value in p0_line.split(":", 1)[1].split()]
    if len(values) != 12 or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{path}: P0 must contain 12 finite values")
    if values[1] != 0.0 or values[4] != 0.0 or values[8:12] != [0.0, 0.0, 1.0, 0.0]:
        raise ValueError(f"{path}: P0 is not a rectified pinhole projection")
    fx, fy, cx, cy = values[0], values[5], values[2], values[6]
    if min(fx, fy) <= 0.0:
        raise ValueError(f"{path}: focal length must be positive")
    return fx, fy, cx, cy


def image_files(path: Path) -> list[Path]:
    return sorted(
        candidate.resolve()
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in IMAGE_SUFFIXES
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    # Prepared held-out inputs are PNG. Reading the IHDR avoids adding an
    # adapter-only Pillow/OpenCV dependency to the already-heavy environment.
    header = path.read_bytes()[:24]
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"unsupported prepared image format: {path}")
    return int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big")


def write_calibration_model(
    output: Path,
    images: list[Path],
    calibration: tuple[float, float, float, float],
) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    width, height = png_dimensions(images[0])
    if any(png_dimensions(image) != (width, height) for image in images):
        raise ValueError("prepared images do not share one rectified resolution")
    fx, fy, cx, cy = calibration
    (output / "cameras.txt").write_text(
        f"# calibration-only; no pose ground truth\n1 PINHOLE {width} {height} {fx:.17g} {fy:.17g} {cx:.17g} {cy:.17g}\n",
        encoding="utf-8",
    )
    with (output / "images.txt").open("w", encoding="utf-8") as stream:
        stream.write("# identity dummy poses for intrinsics lookup only; no pose ground truth\n")
        for image_id, image in enumerate(images, 1):
            stream.write(f"{image_id} 1 0 0 0 0 0 0 1 {image.as_posix()}\n\n")
    (output / "points3D.txt").write_text("# calibration-only\n", encoding="utf-8")


def gluemap_command(args: argparse.Namespace, calibration_model: Path, work: Path) -> list[str]:
    return [
        args.gluemap_command,
        "--config",
        str((args.source_dir / "configs" / "example.yaml").resolve()),
        "--images_path",
        str(args.images_path.resolve()),
        "--write_path",
        str(work.resolve()),
        "--chosen_model",
        "pi3",
        "--intrinsics_mode",
        "SHARED",
        "--use_gt_intrinsics",
        "--gt_intrinsics_path",
        str(calibration_model.resolve()),
        "--is_sequential",
        "--sample_frequency",
        "1",
        "--no-skip_doppelgangers",
        "--no-coarse_only",
    ]


def quat_wxyz_to_rotation(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm <= 0.0 or not math.isfinite(norm):
        raise ValueError("invalid COLMAP quaternion")
    qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def rotation_to_quat_xyzw(rotation: list[list[float]]) -> tuple[float, float, float, float]:
    trace = rotation[0][0] + rotation[1][1] + rotation[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (rotation[2][1] - rotation[1][2]) / scale
        qy = (rotation[0][2] - rotation[2][0]) / scale
        qz = (rotation[1][0] - rotation[0][1]) / scale
    elif rotation[0][0] > rotation[1][1] and rotation[0][0] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[0][0] - rotation[1][1] - rotation[2][2]) * 2.0
        qw = (rotation[2][1] - rotation[1][2]) / scale
        qx = 0.25 * scale
        qy = (rotation[0][1] + rotation[1][0]) / scale
        qz = (rotation[0][2] + rotation[2][0]) / scale
    elif rotation[1][1] > rotation[2][2]:
        scale = math.sqrt(1.0 + rotation[1][1] - rotation[0][0] - rotation[2][2]) * 2.0
        qw = (rotation[0][2] - rotation[2][0]) / scale
        qx = (rotation[0][1] + rotation[1][0]) / scale
        qy = 0.25 * scale
        qz = (rotation[1][2] + rotation[2][1]) / scale
    else:
        scale = math.sqrt(1.0 + rotation[2][2] - rotation[0][0] - rotation[1][1]) * 2.0
        qw = (rotation[1][0] - rotation[0][1]) / scale
        qx = (rotation[0][2] + rotation[2][0]) / scale
        qy = (rotation[1][2] + rotation[2][1]) / scale
        qz = 0.25 * scale
    return qx, qy, qz, qw


def pose_rows(images_txt: Path) -> list[list[str]]:
    rows = []
    for line in images_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 10:
            continue
        try:
            int(parts[0])
            [float(value) for value in parts[1:8]]
            int(parts[8])
        except ValueError:
            continue
        rows.append(parts)
    return rows


def write_tum(images_txt: Path, timestamps_path: Path, output: Path) -> int:
    timestamps = {}
    for line in timestamps_path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) >= 2:
            timestamps[int(fields[0])] = int(fields[1])
    rows = []
    for parts in pose_rows(images_txt):
        name = os.path.basename(parts[9])
        match = re.search(r"\d+", name)
        if match is None or int(match.group()) not in timestamps:
            continue
        frame = int(match.group())
        qw, qx, qy, qz, tx, ty, tz = map(float, parts[1:8])
        rotation_cw = quat_wxyz_to_rotation(qw, qx, qy, qz)
        rotation_wc = [[rotation_cw[column][row] for column in range(3)] for row in range(3)]
        translation = [tx, ty, tz]
        center = [
            -sum(rotation_wc[row][column] * translation[column] for column in range(3))
            for row in range(3)
        ]
        ox, oy, oz, ow = rotation_to_quat_xyzw(rotation_wc)
        rows.append((timestamps[frame] * 1.0e-9, *center, ox, oy, oz, ow))
    rows.sort()
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(" ".join(f"{value:.9f}" for value in row) + "\n")
    return len(rows)


def main() -> int:
    args = parse_args()
    if args.expected_frames < 1:
        raise ValueError("expected frames must be positive")
    for path in (args.source_dir, args.images_path, args.calibration_path, args.timestamps_path):
        if not path.exists():
            raise FileNotFoundError(path)
    config = args.source_dir / "configs" / "example.yaml"
    if not config.is_file():
        raise FileNotFoundError(config)
    images = image_files(args.images_path)
    if len(images) != args.expected_frames:
        raise ValueError(f"found {len(images)} images, expected {args.expected_frames}")
    args.output_path.mkdir(parents=True, exist_ok=True)
    calibration_model = args.output_path / "calibration_only_colmap"
    write_calibration_model(calibration_model, images, parse_rectified_p0(args.calibration_path))
    work = args.output_path / "official_output"
    command = gluemap_command(args, calibration_model, work)
    completed = subprocess.run(command, cwd=args.source_dir, check=False)
    if completed.returncode != 0:
        return completed.returncode

    model_path = work / "gluemap_aba"
    if not model_path.is_dir():
        raise FileNotFoundError(f"official GLUEMAP final model missing: {model_path}")
    try:
        import pycolmap
    except ImportError as error:
        raise RuntimeError("GLUEMAP environment has no pycolmap") from error
    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(str(model_path))
    text_model = args.output_path / "model_text"
    text_model.mkdir()
    reconstruction.write_text(str(text_model))
    registered = write_tum(
        text_model / "images.txt", args.timestamps_path, args.output_path / "trajectory.tum"
    )
    if registered <= 0 or registered > args.expected_frames:
        raise ValueError(f"invalid registered image count: {registered}")
    mean_reprojection = None
    compute_error = getattr(reconstruction, "compute_mean_reprojection_error", None)
    if callable(compute_error):
        candidate = float(compute_error())
        if math.isfinite(candidate):
            mean_reprojection = candidate
    points3d = (
        int(reconstruction.num_points3D())
        if callable(getattr(reconstruction, "num_points3D", None))
        else len(reconstruction.points3D)
    )
    result = {
        "schema_version": 1,
        "engine": "gluemap",
        "official_command": command,
        "ground_truth_read": False,
        "registered_images": registered,
        "points3d": points3d,
        "mean_reprojection_px": mean_reprojection,
    }
    (args.output_path / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
