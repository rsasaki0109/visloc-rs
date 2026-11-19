#!/usr/bin/env python3
"""Fetch the annotated LILocBench RGB-D windows using HTTP ranges."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import struct
import time
from urllib.request import Request, urlopen
import zipfile
import zlib

import numpy as np

from rgbd_map_cloud import nearest

ROOT_URL = "https://www.ipb.uni-bonn.de/html/projects/localization_benchmark/data"
ARCHIVE_BYTES = {"mapping": 18271426492, "lt_changes_0": 10008762359,
                 "lt_changes_dynamics_0": 12940694854, "lt_changes_dynamics_1": 15112070989}
CAMERA = "camera_front"
CALIBRATION = (
    "transformations.yaml", f"{CAMERA}/color/intrinsics.yaml", f"{CAMERA}/depth/intrinsics.yaml",
    "dingo_velocity_controller/odom.txt",
)


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        super().__init__()
        self.url = url
        self.size = size
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = (0, self.position, self.size)[whence] + offset
        if position < 0:
            raise OSError("negative archive offset")
        self.position = position
        return position

    def read(self, count=-1):
        end = self.size if count < 0 else min(self.size, self.position + count)
        if end <= self.position:
            return b""
        span = f"{self.position}-{end - 1}"
        request = Request(f"{self.url}?range={span}", headers={"Range": f"bytes={span}"})
        for attempt in range(3):
            try:
                with urlopen(request, timeout=25) as response:
                    if response.status != 206 or response.headers.get("Content-Range") != f"bytes {span}/{self.size}":
                        raise OSError(f"unexpected response for archive range {span}")
                    data = response.read(end - self.position)
                if len(data) != end - self.position:
                    raise OSError(f"short response for archive range {span}")
                self.position = end
                return data
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(2 ** attempt)


def write_atomically(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(data)
    temporary.replace(path)


def fetch_member(url, size, destination, member):
    path = destination / member.filename
    if path.is_file() and path.stat().st_size == member.file_size:
        if zlib.crc32(path.read_bytes()) == member.CRC:
            return
    with RemoteZip(url, size) as remote:
        remote.seek(member.header_offset)
        data = remote.read(30 + len(member.filename.encode()) + 128 + member.compress_size)
        if data[:4] != b"PK\x03\x04":
            raise OSError(f"{member.filename}: invalid ZIP header")
        name_bytes, extra_bytes = struct.unpack_from("<HH", data, 26)
        header_bytes = 30 + name_bytes + extra_bytes
        compressed = data[header_bytes:header_bytes + member.compress_size]
        if len(compressed) < member.compress_size:
            remote.seek(member.header_offset + header_bytes)
            compressed = remote.read(member.compress_size)
    if member.compress_type == zipfile.ZIP_DEFLATED:
        content = zlib.decompress(compressed, -15)
    elif member.compress_type == zipfile.ZIP_STORED:
        content = compressed
    else:
        raise ValueError(f"{member.filename}: unsupported ZIP compression")
    if len(content) != member.file_size or zlib.crc32(content) != member.CRC:
        raise OSError(f"{member.filename}: ZIP checksum mismatch")
    write_atomically(path, content)


def timestamped_members(members, kind):
    prefix = f"{CAMERA}/{kind}/images/"
    paths = sorted(name for name in members if name.startswith(prefix)
                   and "/" not in name[len(prefix):] and name.endswith(".png"))
    return np.array([float(Path(path).stem) for path in paths]), paths


def fetch_sequence(sequence, pairs, output, workers):
    size = ARCHIVE_BYTES[sequence]
    url = f"{ROOT_URL}/data_files_zipped/{sequence}.zip"
    needs_truth = any(visit["sequence"] == sequence and visit.get("anchor", "reference") == "reference"
                      for pair in pairs for visit in pair["visits"])
    if needs_truth:
        truth = output / f"{sequence}.tum"
        if not truth.is_file():
            with urlopen(f"{ROOT_URL}/gt_poses_public/{sequence}/gt_poses.txt", timeout=25) as response:
                write_atomically(truth, response.read())
        reference_start = float(truth.read_text().split(maxsplit=1)[0])
    with RemoteZip(url, size) as remote, zipfile.ZipFile(remote) as archive:
        members = {entry.filename: entry for entry in archive.infolist()}
    rgb_times, rgb_names = timestamped_members(members, "color")
    depth_times, depth_names = timestamped_members(members, "depth")
    files = set(CALIBRATION)
    manifests = {}
    for pair in pairs:
        if not pair["name"].replace("_", "").isalnum():
            raise ValueError("expected an alphanumeric case name with optional underscores")
        for visit in pair["visits"]:
            if visit["sequence"] != sequence:
                continue
            anchor = visit.get("anchor", "reference")
            if anchor not in ("reference", "first_rgb"):
                raise ValueError(f"unknown capture anchor: {anchor}")
            start = rgb_times[0] if anchor == "first_rgb" else reference_start
            first, last = visit["capture_seconds"]
            relative = np.arange(first, last + 0.001, 0.25)
            rgb_ids = [nearest(rgb_times, start + value) for value in relative]
            depth_ids = [nearest(depth_times, rgb_times[index]) for index in rgb_ids]
            rgb = [rgb_names[index] for index in rgb_ids]
            depth = [depth_names[index] for index in depth_ids]
            files.update(rgb)
            files.update(depth)
            manifests[output / f"{pair['name']}_{sequence}.json"] = {
                "rgb": rgb, "depth": depth, "requested_relative_seconds": relative.tolist(),
            }
    destination = output / "dataset" / sequence
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda name: fetch_member(url, size, destination, members[name]), sorted(files)))
    for path, manifest in manifests.items():
        write_atomically(path, (json.dumps(manifest) + "\n").encode())
    print(f"{sequence}: verified {len(files)} archive members")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    pairs = json.loads(args.annotations.read_text())["maps"]
    for sequence in sorted({visit["sequence"] for pair in pairs for visit in pair["visits"]}):
        fetch_sequence(sequence, pairs, args.out, args.workers)


if __name__ == "__main__":
    main()
