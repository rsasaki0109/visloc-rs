#!/usr/bin/env python3
"""Fetch the GPS of a LaMAria raw recording without its images: a sparse copy of the VRS file.

    python scripts/lamaria_vrs_fetch.py https://cvg-data.inf.ethz.ch/lamaria/raw_data/test/sequence_5_5.vrs \
        work/sequence_5_5/sequence_5_5.vrs

Aria recordings keep the index of every record in the file's head, before the first user record. The script reads
the head (4-30 MB), then fetches only the records a VRS reader needs to open the file and read the GPS: every
configuration and state record, the first and the last data record of every stream (the reader looks for the last
one from the end of the file when it opens it), and all data records of --streams (GPS, UTC time and the time sync
by default). They are fetched with multi-range HTTP requests and written at their offsets into a sparse file of the
full size, which projectaria_tools then opens like the whole recording (lamaria_gps_extract.py). A test recording,
up to 31 GB, comes down to 6-35 MB.

SRC can also be a local VRS file: the same records are copied, to check that the sparse copy reads like the whole.
The index is zstd-compressed; the `zstd` command must be on PATH. See docs/lamaria_gps.md.
"""
import argparse
import os
import re
import struct
import subprocess
import sys
import urllib.request

FILE_HEADER = "<IIQIIqqqQQQII"
RECORD_HEADER = "<IIiIdHBBI"
DISK_RECORD = "<dIBiH"
STATE, CONFIGURATION, DATA = 1, 2, 3
GPS_STREAMS = "281-1,285-1,286-1"


def first_user_offset(header):
    """The file header's offset of the first user record: everything before it is the head."""
    if header[:8] != b"VisionRe":
        raise ValueError("not a VRS file")
    return struct.unpack(FILE_HEADER, header[:struct.calcsize(FILE_HEADER)])[7]


def parse_index(head):
    """The records the head's index lists, in file order: (offset, size, record type, (type id, instance id))."""
    first_user = first_user_offset(head)
    index_off = struct.unpack(FILE_HEADER, head[:struct.calcsize(FILE_HEADER)])[5]
    rec_size = struct.unpack(RECORD_HEADER, head[index_off:index_off + 32])[0]
    if len(head) < index_off + rec_size:
        raise ValueError(f"the head ends at {len(head)}, the index at {index_off + rec_size}")
    body = head[index_off + 32:index_off + rec_size]
    n_streams = struct.unpack("<I", body[:4])[0]
    pos = 4 + 6 * n_streams
    n_records = struct.unpack("<I", body[pos:pos + 4])[0]
    # zstd writes the whole index, then exits with 1 on the bytes after its frame ("unsupported format"): check the
    # length instead.
    raw = subprocess.run(["zstd", "-d", "-c"], input=body[pos + 4:], capture_output=True).stdout
    size = struct.calcsize(DISK_RECORD)
    if len(raw) < size * n_records:
        raise ValueError(f"index decompressed to {len(raw)} bytes, {size * n_records} expected")
    records, off = [], first_user
    for i in range(n_records):
        _ts, rec, rtype, tid, iid = struct.unpack(DISK_RECORD, raw[size * i:size * (i + 1)])
        records.append((off, rec, rtype, (tid, iid)))
        off += rec
    return first_user, records


def keep(records, streams):
    """(offset, size) of the records to fetch, sorted, and the count kept per stream."""
    out, kept, seen, last = set(), {}, set(), {}
    for off, size, rtype, sid in records:
        first = rtype == DATA and sid not in seen
        if rtype == DATA:
            seen.add(sid)
            last[sid] = (off, size)
        if rtype in (STATE, CONFIGURATION) or sid in streams or first:
            out.add((off, size))
            kept[sid] = kept.get(sid, 0) + 1
    for sid, r in last.items():
        if r not in out:
            out.add(r)
            kept[sid] += 1
    return sorted(out), kept


def merge(ranges, gap):
    """[(offset, size)] -> [[start, end_inclusive]], joining ranges at most `gap` bytes apart."""
    merged = []
    for o, s in sorted(ranges):
        if merged and o - (merged[-1][1] + 1) <= gap:
            merged[-1][1] = max(merged[-1][1], o + s - 1)
        else:
            merged.append([o, o + s - 1])
    return merged


def http_get(url, ranges):
    """[(start, end_inclusive)] -> ({start: bytes}, file size); refuses anything but a 206 answer."""
    spec = ",".join(f"{s}-{e}" for s, e in ranges)
    req = urllib.request.Request(url, headers={"Range": f"bytes={spec}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        if r.status != 206:
            raise RuntimeError(f"answer {r.status}, not a partial content")
        ctype = r.headers.get("Content-Type", "")
        body = r.read()
        if not ctype.startswith("multipart/byteranges"):
            m = re.match(r"bytes (\d+)-(\d+)/(\d+)", r.headers["Content-Range"])
            return {int(m.group(1)): body}, int(m.group(3))
    boundary = re.search(r"boundary=([^;]+)", ctype).group(1).strip('"').encode()
    out, total, pos = {}, None, 0
    while True:
        i = body.find(b"--" + boundary, pos)
        if i < 0 or body[i + 2 + len(boundary):i + 4 + len(boundary)] == b"--":
            break
        h_end = body.find(b"\r\n\r\n", i)
        m = re.search(rb"Content-Range: bytes (\d+)-(\d+)/(\d+)", body[i:h_end], re.I)
        s, e, total = int(m.group(1)), int(m.group(2)), int(m.group(3))
        out[s] = body[h_end + 4:h_end + 4 + e - s + 1]
        pos = h_end + 4 + e - s + 1
    return out, total


def file_get(path, ranges):
    with open(path, "rb") as f:
        out = {}
        for s, e in ranges:
            f.seek(s)
            out[s] = f.read(e - s + 1)
        return out, os.path.getsize(path)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("src", help="the recording's URL, or a local VRS file")
    p.add_argument("out", help="the sparse copy to write (the head goes to OUT.head, kept to resume)")
    p.add_argument("--streams", default=GPS_STREAMS, help="streams whose data records are all fetched")
    p.add_argument("--merge-gap", type=int, default=4096, help="bytes between two ranges fetched as one")
    p.add_argument("--per-request", type=int, default=150, help="ranges per HTTP request")
    a = p.parse_args()

    remote = re.match(r"https?://", a.src) is not None
    get = (lambda r: http_get(a.src, r)) if remote else (lambda r: file_get(a.src, r))
    hdr, total = get([(0, 79)])
    first_user = first_user_offset(hdr[0])
    print(f"{a.src}: {total} bytes, head (header and index) {first_user} bytes")
    head_path = a.out + ".head"
    if not os.path.exists(head_path) or os.path.getsize(head_path) != first_user:
        parts, _ = get([(0, first_user - 1)])
        with open(head_path, "wb") as f:
            f.write(parts[0])
    with open(head_path, "rb") as f:
        head = f.read()
    _, records = parse_index(head)
    streams = {tuple(map(int, s.split("-"))) for s in a.streams.split(",")}
    want, kept = keep(records, streams)
    print(f"{len(records)} records; kept per stream: "
          + ", ".join(f"{t}-{i}: {n}" for (t, i), n in sorted(kept.items())))
    merged = merge(want, a.merge_gap)
    print(f"{len(want)} records in {len(merged)} ranges, {sum(e - s + 1 for s, e in merged)} bytes")
    with open(a.out, "wb") as f:
        f.truncate(total)
        f.seek(0)
        f.write(head)
        got = 0
        for k in range(0, len(merged), a.per_request):
            chunk = merged[k:k + a.per_request]
            parts, _ = get([tuple(r) for r in chunk])
            for s, e in chunk:
                # The server may join neighbouring ranges: find the part that covers this one.
                src = max((q for q in parts if q <= s), default=None)
                if src is None or src + len(parts[src]) < e + 1:
                    sys.exit(f"range {s}-{e} missing from the answer")
                f.seek(s)
                f.write(parts[src][s - src:e - src + 1])
                got += e - s + 1
    print(f"wrote {a.out}: {got} bytes of records")


if __name__ == "__main__":
    main()
