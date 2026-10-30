#!/usr/bin/env python3
"""Fetch the parts of a KITTI raw drive that VIO and the object evaluation need.

    python3 scripts/fetch_kitti_raw_drive.py --drive 2011_09_26_drive_0091 --out data/kitti_raw

- From the drive's `_sync` zip: image_00 and image_01 (rectified grey, 10 Hz)
  and the 10 Hz OXTS. Only these members are fetched, with HTTP range
  requests (`fetch_kitti_raw_oxts.py`); the colour cameras and the velodyne
  stay on the server.
- From the `_extract` zip: the 100 Hz OXTS, which is the IMU.
- The date's calibration and the drive's tracklets, in full (they are small).

The output keeps KITTI's raw layout:
`<out>/<date>/{calib_*.txt, <drive>_sync/{image_00,image_01,oxts,tracklet_labels.xml}, <drive>_extract/oxts}`.
Existing files are kept.
"""

from __future__ import annotations

import argparse
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_kitti_raw_oxts import (  # noqa: E402
    annotate_next_local_offsets,
    content_length,
    download_members,
    find_central_directory,
    read_central_directory,
)

BASE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"


def fetch_members(url: str, keep, out: Path, timeout: int) -> int:
    """Download the members of the remote zip `url` whose path parts `keep`
    accepts, under `out`; returns how many were written."""
    size = content_length(url, timeout)
    cd_offset, cd_size, _ = find_central_directory(url, size, timeout)
    entries = read_central_directory(url, cd_offset, cd_size, timeout)
    annotate_next_local_offsets(entries, size)
    wanted = []
    for entry in entries:
        name = str(entry["name"])
        if name.endswith("/") or not keep(name.strip("/").split("/")):
            continue
        if not (out / name).exists():
            wanted.append((entry, name))
    for _, name, data in download_members(url, wanted, timeout):
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return len(wanted)


def fetch_zip(url: str, out: Path, timeout: int) -> int:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    names = [n for n in archive.namelist() if not n.endswith("/") and not (out / n).exists()]
    for name in names:
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(archive.read(name))
    return len(names)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--drive", required=True, help="for example 2011_09_26_drive_0091")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    date = args.drive[:10]
    url = f"{BASE_URL}/{args.drive}/{args.drive}"

    n = fetch_zip(f"{BASE_URL}/{date}_calib.zip", args.out, args.timeout)
    print(f"calibration: {n} files", file=sys.stderr)
    n = fetch_zip(f"{url}_tracklets.zip", args.out, args.timeout)
    print(f"tracklets: {n} files", file=sys.stderr)
    n = fetch_members(f"{url}_extract.zip", lambda p: len(p) > 2 and p[2] == "oxts", args.out, args.timeout)
    print(f"extract oxts (100 Hz): {n} files", file=sys.stderr)
    n = fetch_members(
        f"{url}_sync.zip", lambda p: len(p) > 2 and p[2] in ("image_00", "image_01", "oxts"), args.out, args.timeout
    )
    print(f"sync image_00, image_01, oxts: {n} files", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
