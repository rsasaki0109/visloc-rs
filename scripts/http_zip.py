#!/usr/bin/env python3
"""Extract a remote zip over HTTP range requests, without storing the zip.

    python scripts/http_zip.py URL DEST_DIR [--workers 4]

The members are read in the order they lie in the archive, split into --workers
contiguous parts that download in parallel; zipfile checks each member's CRC.
A member already on disk with the right size is skipped, so a rerun resumes.
A LaMAria test sequence is up to 29 GB, more than a CI runner can hold twice.
"""
import argparse
import http.client
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

CHUNK = 32 << 20


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, read CHUNK bytes at a time."""

    def __init__(self, url):
        self.url = url
        for attempt in range(8):  # the server sometimes stalls for minutes (TLS handshake timeouts)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
                    self.size = int(r.headers["Content-Length"])
                break
            except (OSError, http.client.HTTPException) as e:
                if attempt == 7:
                    raise
                print(f"retry {attempt} HEAD {url}: {e!r}", file=sys.stderr, flush=True)
                time.sleep(2 ** attempt)
        self.pos = 0
        self.buf_start = 0
        self.buf = b""

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=io.SEEK_SET):
        self.pos = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence] + off
        return self.pos

    def _fetch(self, start, n):
        end = min(self.size, start + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})
        for attempt in range(8):
            try:
                with urllib.request.urlopen(request, timeout=120) as r:
                    data = r.read()
                if r.status == 206 and len(data) == end - start + 1:
                    return data
                error = f"status {r.status}, {len(data)} of {end - start + 1} bytes"
            except (OSError, http.client.HTTPException) as e:  # IncompleteRead: the connection dropped mid-body
                error = repr(e)
            print(f"retry {attempt} at {start}: {error}", file=sys.stderr, flush=True)
            time.sleep(2 ** attempt)
        raise IOError(f"range {start}-{end} of {self.url} failed")

    def read(self, n=-1):
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        off = self.pos - self.buf_start
        if not (0 <= off and off + n <= len(self.buf)):
            if n > CHUNK:
                data = self._fetch(self.pos, n)
                self.pos += n
                return data
            self.buf_start, self.buf, off = self.pos, self._fetch(self.pos, CHUNK), 0
        data = self.buf[off:off + n]
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 20))


def _extract_part(url, dest, names, log):
    archive = open_zip(url)  # one connection and read buffer per part
    for k, name in enumerate(names):
        info = archive.getinfo(name)
        target = os.path.join(dest, name)
        if info.is_dir():
            os.makedirs(target, exist_ok=True)
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.exists(target) and os.path.getsize(target) == info.file_size:
            continue
        with archive.open(info) as src, open(target + ".part", "wb") as out:
            while block := src.read(1 << 20):
                out.write(block)
        os.replace(target + ".part", target)
        if k % 5000 == 0:
            log(f"{k}/{len(names)} {name}")


def extract(url, dest, workers=4, log=print):
    """Extract every member of the zip at `url` under `dest`."""
    dest = os.path.realpath(dest)
    infos = sorted(open_zip(url).infolist(), key=lambda i: i.header_offset)
    for info in infos:
        if not os.path.realpath(os.path.join(dest, info.filename)).startswith(dest + os.sep):
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
    t0 = time.time()
    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(_extract_part, url, dest, p, log) for p in parts if p]:
            f.result()
    log(f"extracted {len(infos)} members, {total / 1e9:.1f} GB in {time.time() - t0:.0f} s")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("url")
    p.add_argument("dest")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    extract(a.url, a.dest, a.workers, log=lambda m: print(m, flush=True))


if __name__ == "__main__":
    main()
