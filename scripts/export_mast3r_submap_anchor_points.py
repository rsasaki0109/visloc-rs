#!/usr/bin/env python3
"""Export independently pair-conditioned MASt3R anchor point maps for V1."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mast3r-root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mav0", type=Path, required=True)
    parser.add_argument("--descriptor-dump", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--old-anchor", type=int, default=38)
    parser.add_argument("--new-anchor", type=int, default=462)
    parser.add_argument("--partner-offset", type=int, default=-8)
    return parser.parse_args()


def image_rows(mav0: Path) -> list[str]:
    rows = []
    with (mav0 / "cam0" / "data.csv").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(line.split(",", 1)[1])
    return rows


def descriptor_keypoints(dump: Path, anchor: int) -> np.ndarray:
    rows = list(csv.DictReader((dump / "manifest.csv").open(newline="", encoding="utf-8")))
    row = next(row for row in rows if int(row["arrival_index"]) == anchor)
    return np.load(dump / row["keypoints_file"]).astype(np.float64) * 8.0


def undistort(mav0: Path, filenames: list[str], arrival: int, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = cv2.imread(str(mav0 / "cam0" / "data" / filenames[arrival]), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise FileNotFoundError(f"image arrival {arrival}")
    image = cv2.undistort(raw, K, DIST)
    path = out_dir / f"{arrival:06}.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write {path}")
    return path


def export_side(model, anchor_path: Path, partner_path: Path, keypoints: np.ndarray, output: Path) -> None:
    from dust3r.inference import inference
    from dust3r.utils.image import load_images

    images = load_images([str(anchor_path), str(partner_path)], size=512, verbose=False)
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        prediction = inference([tuple(images)], model, "cuda", batch_size=1, verbose=False)
    points = prediction["pred1"]["pts3d"][0].float().cpu().numpy()
    confidence = prediction["pred1"]["conf"][0].float().cpu().numpy()
    height, width = points.shape[:2]

    original_width, original_height = 752, 480
    scale = 512.0 / max(original_width, original_height)
    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)
    crop_x = (resized_width - width) / 2.0
    crop_y = (resized_height - height) / 2.0
    transformed = keypoints * scale - np.array([crop_x, crop_y])

    with output.open("w", encoding="utf-8") as handle:
        handle.write("# KEYPOINT_INDEX X Y Z CONFIDENCE\n")
        kept = 0
        for index, (u, v) in enumerate(transformed):
            x = int(np.clip(round(u), 0, width - 1))
            y = int(np.clip(round(v), 0, height - 1))
            point = points[y, x]
            conf = float(confidence[y, x])
            if not np.isfinite(point).all() or point[2] <= 0.0 or not np.isfinite(conf):
                continue
            handle.write(
                f"{index} {float(point[0]):.9g} {float(point[1]):.9g} "
                f"{float(point[2]):.9g} {conf:.9g}\n"
            )
            kept += 1
    print(
        f"output={output} shape={width}x{height} points={kept} "
        f"confidence_median={float(np.median(confidence)):.6g} "
        f"peak_vram_mib={torch.cuda.max_memory_allocated() / 1048576:.1f}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    sys.path.insert(0, str(args.mast3r_root))
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from mast3r.model import AsymmetricMASt3R

    args.out_dir.mkdir(parents=True, exist_ok=True)
    filenames = image_rows(args.mav0)
    model = AsymmetricMASt3R.from_pretrained(str(args.model)).eval().cuda()
    for side, anchor in (("old", args.old_anchor), ("new", args.new_anchor)):
        partner = anchor + args.partner_offset
        if not 0 <= partner < len(filenames):
            raise ValueError(f"{side} partner arrival out of range: {partner}")
        image_dir = args.out_dir / side / "undistorted"
        anchor_path = undistort(args.mav0, filenames, anchor, image_dir)
        partner_path = undistort(args.mav0, filenames, partner, image_dir)
        export_side(
            model,
            anchor_path,
            partner_path,
            descriptor_keypoints(args.descriptor_dump, anchor),
            args.out_dir / f"{side}_anchor_points.txt",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
