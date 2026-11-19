#!/usr/bin/env python3
"""Fetch the labeled KITTI Tracking frames for the LiDAR clearance replay."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

from fetch_lilocbench_windows import RemoteZip, fetch_member, write_atomically


BASE = "https://s3.eu-central-1.amazonaws.com/avg-kitti"
VELODYNE_BYTES = 36829852418


def fetch(root, config):
    for name in ("calib", "oxts", "label_2"):
        path = root / f"{name}.zip"
        if not path.is_file():
            with urlopen(f"{BASE}/data_tracking_{name}.zip", timeout=30) as response:
                write_atomically(path, response.read())
    sequence = config["sequence"]
    if len(sequence) != 4 or not sequence.isascii() or not sequence.isdecimal():
        raise ValueError("expected a four-digit KITTI training sequence")
    frames = sorted({frame for key in ("map_frames", "label_frames", "replay_frames",
                                        "insertion_frames", "insertion_label_frames")
                     for frame in config.get(key, [])})
    names = [f"training/velodyne/{sequence}/{frame:06d}.bin" for frame in frames]
    url = f"{BASE}/data_tracking_velodyne.zip"
    with RemoteZip(url, VELODYNE_BYTES) as remote, ZipFile(remote) as archive:
        members = [archive.getinfo(name) for name in names]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda member: fetch_member(url, VELODYNE_BYTES, root, member), members))
    print(f"{sequence}: verified {len(names)} LiDAR scans")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fetch(args.out, json.loads(args.annotations.read_text()))


if __name__ == "__main__":
    main()
