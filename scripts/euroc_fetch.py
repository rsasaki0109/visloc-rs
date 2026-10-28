#!/usr/bin/env python3
"""Fetch one EuRoC MAV sequence from the ETH Research Collection, without its family zip.

    python scripts/euroc_fetch.py MH_01_easy DEST_DIR [--workers 4]

The official archives bundle a family (machine_hall 12.7 GB, vicon_room1 and
vicon_room2 6 GB each; ids in benchmarks/basalt/euroc_dataset_manifest.py) with
one zip per sequence inside. The family zip is read over HTTP range requests
(scripts/http_zip.py). If the sequence's zip is stored uncompressed in it, its
members are read by range too; otherwise it is streamed to DEST_DIR, extracted
and deleted. A family zip that holds the sequence's files directly also works.

Only what basalt_euroc_vio_demo and the evaluators read is kept: cam0, cam1,
imu0, state_groundtruth_estimate0 and the yamls, as DEST_DIR/<sequence>/mav0.
A member already on disk with the right size is skipped, so a rerun resumes.
"""
import argparse
import importlib.util
import io
import os
import shutil
import struct
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from http_zip import HttpFile  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "euroc_dataset_manifest", ROOT / "benchmarks" / "basalt" / "euroc_dataset_manifest.py"
)
_manifest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_manifest)

BITSTREAM_URL = "https://www.research-collection.ethz.ch/server/api/core/bitstreams/{id}/content"
FAMILIES = {"MH_": "machine_hall", "V1_": "vicon_room1", "V2_": "vicon_room2"}
KEEP = ("cam0/", "cam1/", "imu0/", "state_groundtruth_estimate0/")


def family_url(sequence):
    family = next((f for prefix, f in FAMILIES.items() if sequence.startswith(prefix)), None)
    if sequence not in _manifest.SEQUENCES or family is None:
        raise SystemExit(f"unknown EuRoC sequence {sequence!r}; one of {', '.join(_manifest.SEQUENCES)}")
    bitstream = next(b for b in _manifest.OFFICIAL_BITSTREAMS if b["family"] == family)
    return BITSTREAM_URL.format(id=bitstream["id"])


def mav0_path(name):
    """`name` from its `mav0/` component on (`MH_01_easy/mav0/cam0/...` -> `mav0/cam0/...`)."""
    parts = name.split("/")
    return "/".join(parts[parts.index("mav0"):]) if "mav0" in parts else None


def kept(path):
    if path is None or path.endswith("/"):
        return False
    rest = path[len("mav0/"):]
    return rest.startswith(KEEP) or ("/" not in rest and rest.endswith(".yaml"))


class Window(io.RawIOBase):
    """Bytes [start, start + size) of a seekable raw file, as a file of its own."""

    def __init__(self, raw, start, size):
        self.raw, self.start, self.size, self.pos = raw, start, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: off, io.SEEK_CUR: self.pos + off, io.SEEK_END: self.size + off}[whence]
        return self.pos

    def readinto(self, b):
        n = max(0, min(len(b), self.size - self.pos))
        if n == 0:
            return 0
        self.raw.seek(self.start + self.pos)
        got = self.raw.readinto(memoryview(b)[:n])
        self.pos += got
        return got


def open_family(url):
    raw = HttpFile(url)
    return raw, zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))


def stored_member_window(raw, info):
    """The bytes of a stored (uncompressed) member, by its local header."""
    raw.seek(info.header_offset)
    header = raw.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError(f"no local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", header[26:30])
    return Window(raw, info.header_offset + 30 + name_len + extra_len, info.compress_size)


def extract(open_archive, dest, workers, log):
    """Extract the kept members of the zip that `open_archive()` opens (one per worker)."""
    infos = [i for i in open_archive().infolist() if kept(mav0_path(i.filename))]
    if not infos:
        raise SystemExit("no mav0 camera/IMU members found in the archive")
    infos.sort(key=lambda i: i.header_offset)
    dest = os.path.realpath(dest)
    for info in infos:
        if not os.path.realpath(os.path.join(dest, mav0_path(info.filename))).startswith(dest + os.sep):
            raise ValueError(f"unsafe path {info.filename}")
    total = sum(i.compress_size for i in infos)
    parts, part, size = [], [], 0
    for info in infos:  # contiguous parts of about equal size
        part.append(info.filename)
        size += info.compress_size
        if size >= total * (len(parts) + 1) / workers:
            parts.append(part)
            part = []
    parts.append(part)

    def run(names):
        archive = open_archive()  # one connection per part
        for k, name in enumerate(names):
            info = archive.getinfo(name)
            target = os.path.join(dest, mav0_path(name))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and os.path.getsize(target) == info.file_size:
                continue
            with archive.open(info) as src, open(target + ".part", "wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            os.replace(target + ".part", target)
            if k % 2000 == 0:
                log(f"{k}/{len(names)} {name}")

    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(run, p) for p in parts if p]:
            future.result()
    log(f"extracted {len(infos)} members, {total / 1e9:.2f} GB in {time.time() - t0:.0f} s")


def fetch(sequence, dest_root, workers=4, log=print):
    url = family_url(sequence)
    dest = Path(dest_root) / sequence
    dest.mkdir(parents=True, exist_ok=True)
    _, family = open_family(url)
    names = family.namelist()
    nested = [n for n in names if n.rsplit("/", 1)[-1] == f"{sequence}.zip"]
    if not nested:  # the family zip holds the sequence's files directly
        if not any(f"{sequence}/mav0/" in n for n in names):
            raise SystemExit(f"{sequence} is not in {url}")

        def open_direct():
            archive = open_family(url)[1]
            # keep only this sequence's members visible to extract()
            archive.filelist = [i for i in archive.filelist if f"{sequence}/mav0/" in i.filename]
            return archive

        extract(open_direct, dest, workers, log)
        return dest
    inner = family.getinfo(nested[0])
    log(f"{sequence}: {inner.filename}, {inner.compress_size / 1e9:.2f} GB, "
        f"{'stored' if inner.compress_type == zipfile.ZIP_STORED else 'compressed'} in the family zip")
    if inner.compress_type == zipfile.ZIP_STORED:

        def open_inner():
            raw, family = open_family(url)
            window = stored_member_window(raw, family.getinfo(inner.filename))
            return zipfile.ZipFile(io.BufferedReader(window, buffer_size=1 << 20))

        extract(open_inner, dest, workers, log)
        return dest
    local = dest / f"{sequence}.zip"
    t0 = time.time()
    with family.open(inner) as src, open(local, "wb") as out:
        shutil.copyfileobj(src, out, 8 << 20)
    log(f"streamed {local.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s")
    extract(lambda: zipfile.ZipFile(local), dest, 1, log)
    local.unlink()
    return dest


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sequence", help="e.g. MH_01_easy")
    p.add_argument("dest", type=Path, help="the sequence lands in DEST/<sequence>/mav0")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    out = fetch(args.sequence, args.dest, args.workers, lambda m: print(m, flush=True))
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
