#!/usr/bin/env python3
"""Infer two independent VGGT windows and export anchor-keypoint 3D lifts."""

from __future__ import annotations

import argparse
import csv
import sys
from contextlib import nullcontext
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors import safe_open


K = np.array([[458.654, 0.0, 367.215], [0.0, 457.296, 248.375], [0.0, 0.0, 1.0]])
DIST = np.array([-0.28340811, 0.07395907, 0.00019359, 1.76187114e-05])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vggt-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--mav0", type=Path, required=True)
    parser.add_argument("--descriptor-dump", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--old-anchor", type=int, default=38)
    parser.add_argument("--new-anchor", type=int, default=462)
    parser.add_argument("--offsets", default="-16,-8,0,8,16")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--geometry", choices=("depth", "point", "camera"), default="depth")
    return parser.parse_args()


def load_model(vggt_root: Path, weights: Path, geometry: str):
    sys.path.insert(0, str(vggt_root))
    from vggt.models.vggt import VGGT

    model = VGGT(
        enable_point=geometry == "point",
        enable_depth=geometry == "depth",
        enable_track=False,
    )
    wanted = {}
    with safe_open(weights, framework="pt", device="cpu") as archive:
        for key in archive.keys():
            disabled_heads = ["track_head."]
            if geometry != "point":
                disabled_heads.append("point_head.")
            if geometry != "depth":
                disabled_heads.append("depth_head.")
            if not key.startswith(tuple(disabled_heads)):
                wanted[key] = archive.get_tensor(key)
    missing, unexpected = model.load_state_dict(wanted, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"VGGT state mismatch missing={missing} unexpected={unexpected}")
    # Keep weights in fp32. VGGT's heads explicitly disable autocast; forcing
    # them to fp16 on pre-Ampere GPUs can overflow even when the backbone fits.
    return model.eval().cuda()


def image_rows(mav0: Path) -> list[str]:
    rows = []
    with (mav0 / "cam0" / "data.csv").open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split(",", 1)[1])
    return rows


def descriptor_keypoints(dump: Path, anchor: int) -> np.ndarray:
    rows = list(csv.DictReader((dump / "manifest.csv").open(newline="", encoding="utf-8")))
    row = next(row for row in rows if int(row["arrival_index"]) == anchor)
    return np.load(dump / row["keypoints_file"]).astype(np.float64) * 8.0


def prepare_images(mav0: Path, filenames: list[str], arrivals: list[int], out: Path) -> list[Path]:
    image_dir = out / "undistorted"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for arrival in arrivals:
        raw = cv2.imread(str(mav0 / "cam0" / "data" / filenames[arrival]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise FileNotFoundError(f"image arrival {arrival}")
        undistorted = cv2.undistort(raw, K, DIST)
        path = image_dir / f"{arrival:06}.png"
        if not cv2.imwrite(str(path), undistorted):
            raise OSError(f"failed to write {path}")
        paths.append(path)
    return paths


def infer_side(
    model,
    image_paths: list[Path],
    anchor_slot: int,
    keypoints: np.ndarray,
    output: Path,
    precision: str,
    geometry: str,
    arrivals: list[int],
) -> None:
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    images = load_and_preprocess_images([str(path) for path in image_paths], mode="crop").cuda()
    height, width = images.shape[-2:]
    torch.cuda.reset_peak_memory_stats()
    precision_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision == "fp16"
        else nullcontext()
    )
    with torch.no_grad(), precision_context:
        tokens, patch_start = model.aggregator(images[None])
        pose_enc = model.camera_head(tokens)[-1]
        if geometry == "depth":
            geometry_map, confidence = model.depth_head(
                tokens, images=images[None], patch_start_idx=patch_start
            )
        elif geometry == "point":
            geometry_map, confidence = model.point_head(
                tokens, images=images[None], patch_start_idx=patch_start
            )
    print(
        f"tokens_finite={torch.isfinite(tokens[-1]).sum().item()}/{tokens[-1].numel()} "
        f"pose_finite={torch.isfinite(pose_enc).sum().item()}/{pose_enc.numel()} precision={precision}",
        flush=True,
    )
    extrinsics, intrinsics = pose_encoding_to_extri_intri(pose_enc.float(), (height, width))
    if geometry == "camera":
        extrinsics_np = extrinsics[0].float().cpu().numpy()
        anchor_extrinsic = extrinsics_np[anchor_slot]
        with output.open("w", encoding="utf-8") as handle:
            handle.write("# ARRIVAL X Y Z CONFIDENCE\n")
            for arrival, extrinsic in zip(arrivals, extrinsics_np):
                center = -(extrinsic[:, :3].T @ extrinsic[:, 3])
                center = anchor_extrinsic[:, :3] @ center + anchor_extrinsic[:, 3]
                handle.write(
                    f"{arrival} {float(center[0]):.9g} {float(center[1]):.9g} "
                    f"{float(center[2]):.9g} 1\n"
                )
        print(
            f"output={output} views={len(image_paths)} geometry=camera "
            f"peak_vram_mib={torch.cuda.max_memory_allocated() / 1048576:.1f}",
            flush=True,
        )
        return
    geometry_map = geometry_map[0, anchor_slot].float().cpu().numpy()
    confidence = confidence[0, anchor_slot].float().cpu().numpy()
    intrinsic = intrinsics[0, anchor_slot].float().cpu().numpy()
    extrinsic = extrinsics[0, anchor_slot].float().cpu().numpy()

    scaled = keypoints.copy()
    scaled[:, 0] *= width / 752.0
    scaled[:, 1] *= height / 480.0
    finite_geometry = geometry_map[np.isfinite(geometry_map)]
    finite_confidence = confidence[np.isfinite(confidence)]
    print(
        f"geometry={geometry} finite={finite_geometry.size}/{geometry_map.size} "
        f"range={finite_geometry.min() if finite_geometry.size else float('nan'):.6g},"
        f"{finite_geometry.max() if finite_geometry.size else float('nan'):.6g} "
        f"confidence_finite={finite_confidence.size}/{confidence.size}",
        flush=True,
    )
    with output.open("w", encoding="utf-8") as handle:
        handle.write("# KEYPOINT_INDEX X Y Z CONFIDENCE\n")
        kept = 0
        for index, (u, v) in enumerate(scaled):
            x = int(np.clip(round(u), 0, width - 1))
            y = int(np.clip(round(v), 0, height - 1))
            conf = float(confidence[y, x])
            if geometry == "depth":
                z = float(geometry_map[y, x, 0])
                if not np.isfinite(z) or z <= 0.0:
                    continue
                px = (u - intrinsic[0, 2]) / intrinsic[0, 0] * z
                py = (v - intrinsic[1, 2]) / intrinsic[1, 1] * z
            else:
                world = geometry_map[y, x]
                camera = extrinsic[:, :3] @ world + extrinsic[:, 3]
                px, py, z = map(float, camera)
                if not np.isfinite(camera).all() or z <= 0.0:
                    continue
            if not np.isfinite(conf):
                continue
            handle.write(f"{index} {px:.9g} {py:.9g} {z:.9g} {conf:.9g}\n")
            kept += 1
    print(
        f"output={output} views={len(image_paths)} shape={width}x{height} points={kept} "
        f"peak_vram_mib={torch.cuda.max_memory_allocated() / 1048576:.1f}",
        flush=True,
    )
    del images, tokens, pose_enc, geometry_map, confidence, intrinsics
    torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [int(value) for value in args.offsets.split(",")]
    filenames = image_rows(args.mav0)
    model = load_model(args.vggt_root, args.weights, args.geometry)
    for side, anchor in (("old", args.old_anchor), ("new", args.new_anchor)):
        arrivals = [anchor + offset for offset in offsets]
        if any(arrival < 0 or arrival >= len(filenames) for arrival in arrivals):
            raise ValueError(f"{side} arrivals out of range: {arrivals}")
        paths = prepare_images(args.mav0, filenames, arrivals, args.out_dir / side)
        infer_side(
            model,
            paths,
            offsets.index(0),
            descriptor_keypoints(args.descriptor_dump, anchor),
            args.out_dir
            / (f"{side}_camera_centers.txt" if args.geometry == "camera" else f"{side}_anchor_points.txt"),
            args.precision,
            args.geometry,
            arrivals,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
