#!/usr/bin/env python3
"""Fetch a stride-subsampled subset of a KITTI odometry sequence.

Streams only the requested PNG entries out of the public
`data_odometry_gray.zip` archive on the KITTI S3 mirror via HTTP byte
ranges (no full archive download). Also fetches `calib.txt` and
`times.txt` so the existing `read_kitti_image_sequence_dir` API can
consume the local subset.

Asset-generation tool, not part of CI. Runs once per machine.

Usage:
    python3 scripts/fetch_kitti_seq00_images.py \
        --sequence 00 \
        --stride 4 \
        --max-frames 600 \
        --out-dir ~/datasets/kitti_seq00_subset
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import remotezip

KITTI_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_gray.zip"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=str, default="00",
                        help="KITTI odometry sequence id, e.g. 00..21. "
                             "Ground-truth poses are only available for 00..10.")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--also-fetch-poses", action="store_true",
                        help="Also fetch poses/<sequence>.txt from data_odometry_poses.zip "
                             "(small download, lets you compare VO vs GT later).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames that already exist locally.")
    parser.add_argument("--cameras", type=str, default="image_0",
                        help="Comma-separated KITTI camera dirs to pull "
                             "(e.g. 'image_0,image_1' for the rectified gray "
                             "stereo pair used by the stereo VO demo).")
    args = parser.parse_args()

    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    if not cameras:
        print("# --cameras must include at least one entry", file=sys.stderr)
        return 2
    sequence = f"{int(args.sequence):02d}"
    sequence_prefix = f"dataset/sequences/{sequence}"

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"# Streaming KITTI {sequence} subset (stride={args.stride}, "
          f"max_frames={args.max_frames}, cameras={cameras}) → {out_dir}")
    print(f"# remote: {KITTI_URL}")

    t0 = time.time()
    rz = remotezip.RemoteZip(KITTI_URL)
    print(f"  central directory loaded in {time.time() - t0:.1f}s")

    # Pull calib.txt and times.txt (small).
    for small in ("calib.txt", "times.txt"):
        name = f"{sequence_prefix}/{small}"
        with rz.open(name) as src, open(out_dir / small, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"  fetched {small}")

    # Pull stride-subsampled frames for each requested camera. Frame
    # selection is shared across cameras so left/right pairs stay aligned;
    # otherwise the stereo VO demo would mis-match images across time.
    reference_camera = cameras[0]
    image_names_ref = sorted(
        n for n in rz.namelist()
        if n.startswith(f"{sequence_prefix}/{reference_camera}/") and n.endswith(".png")
    )
    selected_indices: list[int] = []
    for index, _ in enumerate(image_names_ref):
        if index < args.start_frame:
            continue
        if (index - args.start_frame) % args.stride != 0:
            continue
        selected_indices.append(index)
        if len(selected_indices) >= args.max_frames:
            break
    print(f"  selected {len(selected_indices)} of {len(image_names_ref)} frames")

    selected: list[tuple[str, Path]] = []
    for cam in cameras:
        cam_dir = out_dir / cam
        cam_dir.mkdir(parents=True, exist_ok=True)
        cam_names = sorted(
            n for n in rz.namelist()
            if n.startswith(f"{sequence_prefix}/{cam}/") and n.endswith(".png")
        )
        if len(cam_names) != len(image_names_ref):
            print(f"# warning: camera {cam} has {len(cam_names)} frames "
                  f"but {reference_camera} has {len(image_names_ref)}; "
                  f"using min().", file=sys.stderr)
        for idx in selected_indices:
            if idx >= len(cam_names):
                continue
            selected.append((cam_names[idx], cam_dir / Path(cam_names[idx]).name))

    if args.skip_existing:
        before = len(selected)
        selected = [(name, dst) for (name, dst) in selected if not dst.exists()]
        print(f"  skipping {before - len(selected)} frames already on disk; "
              f"{len(selected)} remain to fetch")

    fetch_lock = threading.Lock()
    counter = {"done": 0, "bytes": 0}
    fetch_start = time.time()

    def fetch_one(name_and_path: tuple[str, Path]) -> int:
        name, local_path = name_and_path
        local_rz = remotezip.RemoteZip(KITTI_URL)
        try:
            with local_rz.open(name) as src, open(local_path, "wb") as dst:
                data = src.read()
                dst.write(data)
                size = len(data)
        finally:
            local_rz.close()
        with fetch_lock:
            counter["done"] += 1
            counter["bytes"] += size
            if counter["done"] % 50 == 0 or counter["done"] == len(selected):
                elapsed = time.time() - fetch_start
                mb = counter["bytes"] / (1024 * 1024)
                rate = mb / max(elapsed, 1e-6)
                print(f"  fetched {counter['done']}/{len(selected)} frames "
                      f"({mb:.1f} MiB at {rate:.2f} MiB/s)")
        return size

    if selected:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for _ in as_completed([pool.submit(fetch_one, np) for np in selected]):
                pass

    if args.also_fetch_poses:
        print(f"# Fetching ground-truth poses (poses/{sequence}.txt) ...")
        poses_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
        rz_poses = remotezip.RemoteZip(poses_url)
        pose_path = out_dir / f"poses_{sequence}.txt"
        with rz_poses.open(f"dataset/poses/{sequence}.txt") as src, \
             open(pose_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"  fetched {pose_path}")

    print(f"# Done. Output dir: {out_dir}")
    if "image_0" in cameras and "image_1" in cameras:
        print(f"# Run the stereo VO demo with:")
        print(
            f"  cargo run --release --features image-io \\\n"
            f"      --example online_slam_stereo_vo_kitti_demo -- \\\n"
            f"      --image-left  {out_dir / 'image_0'} \\\n"
            f"      --image-right {out_dir / 'image_1'} \\\n"
            f"      --calib       {out_dir / 'calib.txt'} \\\n"
            f"      --max-frames {args.max_frames} --frame-stride 1 \\\n"
            f"      --out-dir target/kitti_stereo_vo_demo"
        )
    else:
        print(f"# Run the (monocular) demo with:")
        print(
            f"  cargo run --release --features image-io \\\n"
            f"      --example online_slam_image_vo_loop_demo -- \\\n"
            f"      --image-dir {out_dir / 'image_0'} \\\n"
            f"      --calib    {out_dir / 'calib.txt'} \\\n"
            f"      --max-frames {args.max_frames} --frame-stride 1 \\\n"
            f"      --out-dir target/kitti_image_vo_loop_demo"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
