#!/usr/bin/env python3
"""Fetch a stride-subsampled subset of KITTI odometry seq 00 image_0.

Streams only the requested PNG entries out of the public
`data_odometry_gray.zip` archive on the KITTI S3 mirror via HTTP byte
ranges (no full archive download). Also fetches `calib.txt` and
`times.txt` so the existing `read_kitti_image_sequence_dir` API can
consume the local subset.

Asset-generation tool, not part of CI. Runs once per machine.

Usage:
    python3 scripts/fetch_kitti_seq00_images.py \
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
SEQ00_PREFIX = "dataset/sequences/00"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--also-fetch-poses", action="store_true",
                        help="Also fetch poses/00.txt from data_odometry_poses.zip "
                             "(small download, lets you compare VO vs GT later).")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel download workers.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip frames that already exist locally.")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    image_dir = out_dir / "image_0"
    image_dir.mkdir(parents=True, exist_ok=True)

    print(f"# Streaming KITTI 00 subset (stride={args.stride}, "
          f"max_frames={args.max_frames}) → {out_dir}")
    print(f"# remote: {KITTI_URL}")

    t0 = time.time()
    rz = remotezip.RemoteZip(KITTI_URL)
    print(f"  central directory loaded in {time.time() - t0:.1f}s")

    # Pull calib.txt and times.txt (small).
    for small in ("calib.txt", "times.txt"):
        name = f"{SEQ00_PREFIX}/{small}"
        with rz.open(name) as src, open(out_dir / small, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"  fetched {small}")

    # Pull stride-subsampled image_0 frames.
    image_names = sorted(
        n for n in rz.namelist()
        if n.startswith(f"{SEQ00_PREFIX}/image_0/") and n.endswith(".png")
    )
    selected: list[str] = []
    for index, name in enumerate(image_names):
        # KITTI image filenames are zero-padded frame indices, but iterate by
        # entry order which matches numeric order after sort().
        if index < args.start_frame:
            continue
        if (index - args.start_frame) % args.stride != 0:
            continue
        selected.append(name)
        if len(selected) >= args.max_frames:
            break
    print(f"  selected {len(selected)} of {len(image_names)} frames")

    if args.skip_existing:
        before = len(selected)
        selected = [n for n in selected if not (image_dir / Path(n).name).exists()]
        print(f"  skipping {before - len(selected)} frames already on disk; "
              f"{len(selected)} remain to fetch")

    fetch_lock = threading.Lock()
    counter = {"done": 0, "bytes": 0}
    fetch_start = time.time()

    def fetch_one(name: str) -> int:
        local_rz = remotezip.RemoteZip(KITTI_URL)
        try:
            local_path = image_dir / Path(name).name
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
            for _ in as_completed([pool.submit(fetch_one, n) for n in selected]):
                pass

    if args.also_fetch_poses:
        print("# Fetching ground-truth poses (poses/00.txt) ...")
        poses_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_poses.zip"
        rz_poses = remotezip.RemoteZip(poses_url)
        with rz_poses.open("dataset/poses/00.txt") as src, \
             open(out_dir / "poses_00.txt", "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"  fetched {out_dir / 'poses_00.txt'}")

    print(f"# Done. Output dir: {out_dir}")
    print(f"# Run the demo with:")
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
