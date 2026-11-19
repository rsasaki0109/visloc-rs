#!/usr/bin/env python3
"""Fetch 3-second ROVER garden RGB samples without downloading both full archives."""

import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
from threading import local
from urllib.request import urlopen
from zipfile import ZipFile
from zlib import crc32

import numpy as np

from fetch_lilocbench_windows import RemoteZip, write_atomically


DATASET = "iis-esslingen/ROVER"
REVISION = "69c76ca729afe15ce2c75b71236627298cf6445e"
SEQUENCES = ("garden_small_summer_2023-08-18", "garden_small_winter_2024-01-13")
HUB = "https://huggingface.co"


class SplitZip(io.RawIOBase):
    def __init__(self, parts):
        self.parts = [RemoteZip(f"{HUB}/datasets/{DATASET}/resolve/{REVISION}/{part['path']}",
                                part["size"]) for part in parts]
        self.ends = np.cumsum([0] + [part.size for part in self.parts]).tolist()
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        target = (0, self.position, self.ends[-1])[whence] + offset
        if target < 0:
            raise OSError("negative split archive offset")
        self.position = target
        return target

    def read(self, count=-1):
        end = self.ends[-1] if count < 0 else min(self.ends[-1], self.position + count)
        chunks = []
        while self.position < end:
            index = bisect_right(self.ends, self.position) - 1
            part = self.parts[index]
            part.seek(self.position - self.ends[index])
            size = min(end, self.ends[index + 1]) - self.position
            chunks.append(part.read(size))
            self.position += size
        return b"".join(chunks)


def sample_sequence(sequence, listing, root, workers, extra_offsets):
    parts = sorted((entry for entry in listing if entry["path"].startswith(f"{sequence}.zip.part-")),
                   key=lambda entry: entry["path"])
    if not parts:
        raise ValueError(f"missing split archive for {sequence}")
    folder = root / sequence
    date = sequence.rsplit("_", 1)[-1]
    stream = f"{date}/realsense_D435i"
    with ZipFile(SplitZip(parts)) as archive:
        for name in (f"{stream}/rgb.txt", f"{date}/groundtruth.txt"):
            write_atomically(folder / Path(name).name, archive.read(name))
    rows = [row.split() for row in (folder / "rgb.txt").read_text().splitlines()]
    times = np.asarray([float(row[0]) for row in rows])
    state = local()

    def fetch(seconds):
        index = int(np.argmin(np.abs(times - times[0] - seconds)))
        member = f"{stream}/{rows[index][1]}"
        if not hasattr(state, "archive"):
            state.archive = ZipFile(SplitZip(parts))
        target = folder / f"{seconds:03d}-rgb.png"
        expected = state.archive.getinfo(member)
        if not target.is_file() or crc32(target.read_bytes()) != expected.CRC:
            write_atomically(target, state.archive.read(member))
        return str(seconds), {"rgb_time": float(times[index])}

    offsets = sorted(set(range(0, int(times[-1] - times[0]) + 1, 3)) | set(extra_offsets))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        samples = dict(pool.map(fetch, offsets))
    path = folder / "selected.json"
    existing = json.loads(path.read_text()) if path.is_file() else {}
    for key, value in samples.items():
        existing.setdefault(key, {}).update(value)
    write_atomically(path, (json.dumps(existing, indent=2) + "\n").encode())
    print(f"{sequence}: {len(samples)} RGB samples", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--facade-window", type=Path, help="also fetch the facade evaluation offsets")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    url = f"{HUB}/api/datasets/{DATASET}/tree/{REVISION}?limit=1000"
    with urlopen(url, timeout=25) as response:
        listing = json.load(response)
    windows = json.loads(args.facade_window.read_text()) if args.facade_window else ([], [])
    for sequence, extra in zip(SEQUENCES, windows, strict=True):
        sample_sequence(sequence, listing, args.out, args.workers, extra)


if __name__ == "__main__":
    main()
