#!/usr/bin/env python3
"""Fetch one TUM-VI room sequence (512 px, EuRoC/ASL export) from the TUM server.

    python scripts/tumvi_fetch.py room1 DEST_DIR

The server ships each sequence as an uncompressed tar (dataset-room1_512_16.tar,
about 1.7 GB). It is read as a stream and only what basalt_euroc_vio_demo and the
evaluators read is kept: cam0, cam1, imu0 and mocap0 (motion-capture ground
truth of the IMU frame, in the EuRoC ground-truth layout), as
DEST_DIR/<sequence>/mav0. Only the room sequences have ground truth for their
whole length. A member already on disk with the right size is skipped, and a
failed transfer is restarted (after Retry-After on 429/503).
"""
import argparse
import http.client
import os
import sys
import tarfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from http_zip import retry_delay  # noqa: E402

URL = "https://cdn3.vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-{sequence}_512_16.tar"
SEQUENCES = tuple(f"room{i}" for i in range(1, 7))
KEEP = ("cam0/", "cam1/", "imu0/", "mocap0/")
ATTEMPTS = 6


def mav0_path(name):
    parts = name.split("/")
    return "/".join(parts[parts.index("mav0"):]) if "mav0" in parts else None


def kept(path):
    return path is not None and path[len("mav0/"):].startswith(KEEP)


def extract_stream(stream, dest, log):
    """Extract the kept regular files of the tar read from `stream` under `dest`."""
    dest = os.path.realpath(dest)
    count = skipped = 0
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            path = mav0_path(member.name)
            if not member.isfile() or not kept(path):
                continue
            target = os.path.realpath(os.path.join(dest, path))
            if not target.startswith(dest + os.sep):
                raise ValueError(f"unsafe path {member.name}")
            if os.path.exists(target) and os.path.getsize(target) == member.size:
                skipped += 1
                continue
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with tar.extractfile(member) as src, open(target + ".part", "wb") as out:
                while block := src.read(1 << 20):
                    out.write(block)
            os.replace(target + ".part", target)
            count += 1
            if count % 2000 == 0:
                log(f"{count} files")
    log(f"extracted {count} files, {skipped} already on disk")


def fetch(sequence, dest_root, url=None, log=print):
    if sequence not in SEQUENCES:
        raise SystemExit(f"unknown TUM-VI sequence {sequence!r}; one of {', '.join(SEQUENCES)}")
    url = url or URL.format(sequence=sequence)
    dest = Path(dest_root) / sequence
    dest.mkdir(parents=True, exist_ok=True)
    for attempt in range(ATTEMPTS):
        try:
            t0 = time.time()
            with urllib.request.urlopen(url, timeout=120) as r:
                extract_stream(r, dest, log)
            log(f"{sequence} in {time.time() - t0:.0f} s")
            return dest
        except (OSError, http.client.HTTPException, tarfile.TarError) as e:
            if attempt == ATTEMPTS - 1:
                raise
            delay = retry_delay(attempt, e)
            log(f"retry {attempt} {url}: {e!r}, waiting {delay} s")
            time.sleep(delay)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sequence", help="room1 ... room6")
    p.add_argument("dest", type=Path, help="the sequence lands in DEST/<sequence>/mav0")
    args = p.parse_args()
    print(fetch(args.sequence, args.dest, log=lambda m: print(m, flush=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
