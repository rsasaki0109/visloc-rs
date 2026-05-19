#!/usr/bin/env python3
"""Fetch only KITTI raw OXTS/timestamp files from a large sync zip.

The KITTI raw sync archives are often tens of GB because they contain every
camera image.  For visual-inertial VO experiments we only need:

  * oxts/data/*.txt
  * oxts/timestamps.txt
  * image_00/timestamps.txt (or another camera timestamp stream)

This script reads the remote zip central directory with HTTP Range requests,
then downloads just those small members.  It intentionally uses only the
Python standard library so it can run on a fresh machine.
"""

from __future__ import annotations

import argparse
import binascii
import os
from pathlib import Path
import re
import struct
import sys
import urllib.error
import urllib.request
import zlib


RAW_SYNC_URLS = {
    "00": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip",
    "01": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0042/2011_10_03_drive_0042_sync.zip",
    "02": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_10_03_drive_0034/2011_10_03_drive_0034_sync.zip",
    "03": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_26_drive_0067/2011_09_26_drive_0067_sync.zip",
    "04": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip",
    "05": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip",
    "06": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0020/2011_09_30_drive_0020_sync.zip",
    "07": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip",
    "08": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0028/2011_09_30_drive_0028_sync.zip",
    "09": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0033/2011_09_30_drive_0033_sync.zip",
    "10": "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/2011_09_30_drive_0034/2011_09_30_drive_0034_sync.zip",
}

RAW_ODOMETRY_START_FRAMES = {
    "00": 0,
    "01": 0,
    "02": 0,
    "03": 0,
    "04": 0,
    "05": 0,
    "06": 0,
    "07": 0,
    "08": 1100,
    "09": 0,
    "10": 0,
}


EOCD_SIG = b"PK\x05\x06"
ZIP64_LOCATOR_SIG = b"PK\x06\x07"
ZIP64_EOCD_SIG = b"PK\x06\x06"
CENTRAL_FILE_SIG = b"PK\x01\x02"
LOCAL_FILE_SIG = b"PK\x03\x04"


class FetchError(RuntimeError):
    pass


def request_bytes(url: str, start: int, end: int, timeout: int) -> bytes:
    if start < 0 or end < start:
        raise ValueError(f"bad byte range {start}-{end}")
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = response.read()
        if response.status not in (200, 206):
            raise FetchError(f"GET range {start}-{end} returned HTTP {response.status}")
    expected = end - start + 1
    if len(data) != expected:
        raise FetchError(f"GET range {start}-{end} returned {len(data)} bytes, expected {expected}")
    return data


def content_length(url: str, timeout: int) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        ranges = response.headers.get("Accept-Ranges", "")
        if not length:
            raise FetchError("remote server did not send Content-Length")
        if "bytes" not in ranges.lower():
            raise FetchError("remote server does not advertise byte-range support")
        return int(length)


def find_central_directory(url: str, zip_size: int, timeout: int) -> tuple[int, int, int]:
    tail_size = min(zip_size, 1024 * 1024)
    tail_start = zip_size - tail_size
    tail = request_bytes(url, tail_start, zip_size - 1, timeout)
    eocd_index = tail.rfind(EOCD_SIG)
    if eocd_index < 0:
        raise FetchError("could not find zip EOCD record in remote tail")

    eocd_offset = tail_start + eocd_index
    if eocd_index + 22 > len(tail):
        raise FetchError("truncated zip EOCD record")
    fields = struct.unpack_from("<4s4H2LH", tail, eocd_index)
    entries_total = fields[4]
    cd_size = fields[5]
    cd_offset = fields[6]

    needs_zip64 = entries_total == 0xFFFF or cd_size == 0xFFFFFFFF or cd_offset == 0xFFFFFFFF
    if not needs_zip64:
        return cd_offset, cd_size, entries_total

    locator_index = tail.rfind(ZIP64_LOCATOR_SIG, 0, eocd_index)
    if locator_index < 0:
        raise FetchError("zip64 EOCD locator was not found")
    locator = struct.unpack_from("<4sLQL", tail, locator_index)
    zip64_eocd_offset = locator[2]
    record = request_bytes(url, zip64_eocd_offset, zip64_eocd_offset + 55, timeout)
    if not record.startswith(ZIP64_EOCD_SIG):
        raise FetchError("zip64 EOCD record has an invalid signature")
    fields64 = struct.unpack_from("<4sQ2H2L4Q", record, 0)
    entries_total_64 = fields64[7]
    cd_size_64 = fields64[8]
    cd_offset_64 = fields64[9]
    if cd_offset_64 >= eocd_offset:
        raise FetchError("zip64 central directory offset points past EOCD")
    return cd_offset_64, cd_size_64, entries_total_64


def parse_zip64_extra(extra: bytes, compressed: int, uncompressed: int, offset: int) -> tuple[int, int, int]:
    pos = 0
    while pos + 4 <= len(extra):
        header_id, data_size = struct.unpack_from("<HH", extra, pos)
        pos += 4
        data = extra[pos : pos + data_size]
        pos += data_size
        if header_id != 0x0001:
            continue
        data_pos = 0
        if uncompressed == 0xFFFFFFFF:
            uncompressed = struct.unpack_from("<Q", data, data_pos)[0]
            data_pos += 8
        if compressed == 0xFFFFFFFF:
            compressed = struct.unpack_from("<Q", data, data_pos)[0]
            data_pos += 8
        if offset == 0xFFFFFFFF:
            offset = struct.unpack_from("<Q", data, data_pos)[0]
        break
    return compressed, uncompressed, offset


def read_central_directory(url: str, cd_offset: int, cd_size: int, timeout: int) -> list[dict[str, object]]:
    if cd_size > 128 * 1024 * 1024:
        raise FetchError(f"central directory is unexpectedly large: {cd_size} bytes")
    data = request_bytes(url, cd_offset, cd_offset + cd_size - 1, timeout)
    entries: list[dict[str, object]] = []
    pos = 0
    while pos + 46 <= len(data):
        if data[pos : pos + 4] != CENTRAL_FILE_SIG:
            break
        header = struct.unpack_from("<4s6H3L5H2L", data, pos)
        flags = header[3]
        method = header[4]
        crc32 = header[7]
        compressed = header[8]
        uncompressed = header[9]
        name_len = header[10]
        extra_len = header[11]
        comment_len = header[12]
        local_offset = header[16]
        name_start = pos + 46
        extra_start = name_start + name_len
        comment_start = extra_start + extra_len
        end = comment_start + comment_len
        if end > len(data):
            raise FetchError("truncated central directory entry")
        name_bytes = data[name_start:extra_start]
        if flags & 0x800:
            name = name_bytes.decode("utf-8")
        else:
            name = name_bytes.decode("cp437")
        compressed, uncompressed, local_offset = parse_zip64_extra(
            data[extra_start:comment_start],
            compressed,
            uncompressed,
            local_offset,
        )
        entries.append(
            {
                "name": name,
                "method": method,
                "crc32": crc32,
                "compressed": compressed,
                "uncompressed": uncompressed,
                "local_offset": local_offset,
            }
        )
        pos = end
    return entries


def annotate_next_local_offsets(entries: list[dict[str, object]], zip_size: int) -> None:
    ordered = sorted(entries, key=lambda entry: int(entry["local_offset"]))
    for index, entry in enumerate(ordered):
        if index + 1 < len(ordered):
            entry["next_local_offset"] = int(ordered[index + 1]["local_offset"])
        else:
            entry["next_local_offset"] = zip_size


def decode_member_payload(entry: dict[str, object], payload: bytes) -> bytes:
    method = int(entry["method"])
    uncompressed_size = int(entry["uncompressed"])
    if method == 0:
        data = payload
    elif method == 8:
        data = zlib.decompress(payload, -zlib.MAX_WBITS)
    else:
        raise FetchError(f"{entry['name']}: unsupported zip compression method {method}")
    if len(data) != uncompressed_size:
        raise FetchError(f"{entry['name']}: got {len(data)} bytes, expected {uncompressed_size}")
    crc = binascii.crc32(data) & 0xFFFFFFFF
    if crc != int(entry["crc32"]):
        raise FetchError(f"{entry['name']}: CRC mismatch")
    return data


def read_member_from_local_chunk(entry: dict[str, object], local_chunk: bytes) -> bytes:
    if len(local_chunk) < 30:
        raise FetchError(f"{entry['name']}: truncated local header")
    header = local_chunk[:30]
    if not header.startswith(LOCAL_FILE_SIG):
        raise FetchError(f"{entry['name']}: local header has an invalid signature")
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    name_len = fields[9]
    extra_len = fields[10]
    data_offset = 30 + name_len + extra_len
    compressed_size = int(entry["compressed"])
    payload = local_chunk[data_offset : data_offset + compressed_size]
    if len(payload) != compressed_size:
        raise FetchError(
            f"{entry['name']}: compressed payload is truncated ({len(payload)} bytes, expected {compressed_size})"
        )
    return decode_member_payload(entry, payload)


def download_member(url: str, entry: dict[str, object], timeout: int) -> bytes:
    local_offset = int(entry["local_offset"])
    compressed_size = int(entry["compressed"])
    header = request_bytes(url, local_offset, local_offset + 29, timeout)
    if not header.startswith(LOCAL_FILE_SIG):
        raise FetchError(f"{entry['name']}: local header has an invalid signature")
    fields = struct.unpack_from("<4s5H3L2H", header, 0)
    name_len = fields[9]
    extra_len = fields[10]
    data_offset = local_offset + 30 + name_len + extra_len
    if compressed_size == 0:
        payload = b""
    else:
        payload = request_bytes(url, data_offset, data_offset + compressed_size - 1, timeout)
    return decode_member_payload(entry, payload)


def download_members(
    url: str,
    items: list[tuple[dict[str, object], str]],
    timeout: int,
) -> list[tuple[dict[str, object], str, bytes]]:
    """Download selected members, coalescing adjacent local-file ranges."""
    if not items:
        return []
    ordered = sorted(items, key=lambda item: int(item[0]["local_offset"]))
    output: list[tuple[dict[str, object], str, bytes]] = []
    group: list[tuple[dict[str, object], str]] = []
    group_start = 0
    group_end = 0
    max_group_bytes = 32 * 1024 * 1024

    def flush() -> None:
        nonlocal group, group_start, group_end
        if not group:
            return
        if len(group) == 1:
            entry, rel = group[0]
            output.append((entry, rel, download_member(url, entry, timeout)))
        else:
            chunk = request_bytes(url, group_start, group_end - 1, timeout)
            for entry, rel in group:
                start = int(entry["local_offset"]) - group_start
                end = int(entry["next_local_offset"]) - group_start
                output.append((entry, rel, read_member_from_local_chunk(entry, chunk[start:end])))
        group = []
        group_start = 0
        group_end = 0

    for entry, rel in ordered:
        start = int(entry["local_offset"])
        end = int(entry.get("next_local_offset", start))
        if end <= start:
            flush()
            output.append((entry, rel, download_member(url, entry, timeout)))
            continue
        if not group:
            group = [(entry, rel)]
            group_start = start
            group_end = end
            continue
        if start >= group_end and end - group_start <= max_group_bytes:
            group.append((entry, rel))
            group_end = end
        else:
            flush()
            group = [(entry, rel)]
            group_start = start
            group_end = end
    flush()
    return output


def relative_member_name(name: str, camera: str) -> str | None:
    parts = name.strip("/").split("/")
    for index, part in enumerate(parts):
        if part == "oxts":
            return "/".join(parts[index:])
        if part == camera and index + 1 < len(parts) and parts[index + 1] == "timestamps.txt":
            return f"{camera}/timestamps.txt"
    return None


def frame_index_from_oxts_data(rel_name: str) -> int | None:
    match = re.fullmatch(r"oxts/data/(\d+)\.txt", rel_name)
    if not match:
        return None
    return int(match.group(1))


def selected_indices(start_frame: int, frame_stride: int, frames: int | None) -> set[int] | None:
    if frames is None:
        return None
    return {start_frame + i * frame_stride for i in range(frames)}


def filter_timestamp_lines(text: str, start_frame: int, frame_stride: int, frames: int | None) -> str:
    lines = text.splitlines()
    if frames is None:
        return "\n".join(lines) + ("\n" if lines else "")
    picked = []
    for i in range(frames):
        source_index = start_frame + i * frame_stride
        if source_index >= len(lines):
            raise FetchError(
                f"timestamp stream has {len(lines)} lines, cannot select frame index {source_index}"
            )
        picked.append(lines[source_index])
    return "\n".join(picked) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--url", help="KITTI raw *_sync.zip URL")
    source.add_argument(
        "--odometry-seq",
        choices=sorted(RAW_SYNC_URLS.keys(), key=int),
        help="KITTI odometry sequence id mapped to its raw sync zip",
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--camera", default="image_00", help="timestamp stream to extract, default image_00")
    parser.add_argument(
        "--start-frame",
        type=int,
        help=(
            "raw frame index to start from; defaults to the KITTI odometry-to-raw "
            "mapping start for --odometry-seq, or 0 with --url"
        ),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--frames", type=int, help="extract a subset matching the first N benchmark frames")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true", help="print matching members without extracting")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.start_frame is None:
        args.start_frame = RAW_ODOMETRY_START_FRAMES.get(args.odometry_seq, 0)
    if args.start_frame < 0:
        raise SystemExit("--start-frame must be non-negative")
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    if args.frames is not None and args.frames <= 0:
        raise SystemExit("--frames must be positive")

    url = args.url or RAW_SYNC_URLS[args.odometry_seq]
    wanted_indices = selected_indices(args.start_frame, args.frame_stride, args.frames)

    zip_size = content_length(url, args.timeout)
    cd_offset, cd_size, cd_entries = find_central_directory(url, zip_size, args.timeout)
    entries = read_central_directory(url, cd_offset, cd_size, args.timeout)
    annotate_next_local_offsets(entries, zip_size)
    print(
        f"remote zip: {zip_size} bytes, central_directory={cd_size} bytes, entries={len(entries)}",
        file=sys.stderr,
    )
    if cd_entries != len(entries):
        print(f"warning: EOCD advertised {cd_entries} entries, parsed {len(entries)}", file=sys.stderr)

    selected: list[tuple[dict[str, object], str]] = []
    timestamp_entries: list[tuple[dict[str, object], str]] = []
    for entry in entries:
        name = str(entry["name"])
        if name.endswith("/"):
            continue
        rel = relative_member_name(name, args.camera)
        if rel is None:
            continue
        data_index = frame_index_from_oxts_data(rel)
        if data_index is not None and wanted_indices is not None and data_index not in wanted_indices:
            continue
        if rel in ("oxts/timestamps.txt", f"{args.camera}/timestamps.txt"):
            timestamp_entries.append((entry, rel))
        elif rel.startswith("oxts/"):
            selected.append((entry, rel))

    all_selected = selected + timestamp_entries
    if args.list_only:
        for _, rel in sorted(all_selected, key=lambda item: item[1]):
            print(rel)
        return 0
    if not all_selected:
        raise SystemExit("no matching OXTS/timestamp members found")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    to_download = []
    for entry, rel in sorted(all_selected, key=lambda item: item[1]):
        out_path = args.out_dir / rel
        if out_path.exists() and not args.overwrite:
            print(f"skip existing {out_path}", file=sys.stderr)
            continue
        to_download.append((entry, rel))

    for entry, rel, data in download_members(url, to_download, args.timeout):
        out_path = args.out_dir / rel
        if rel in ("oxts/timestamps.txt", f"{args.camera}/timestamps.txt"):
            text = data.decode("utf-8")
            data = filter_timestamp_lines(text, args.start_frame, args.frame_stride, args.frames).encode("utf-8")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
        tmp_path.write_bytes(data)
        os.replace(tmp_path, out_path)
        written += 1
        print(f"wrote {out_path}")

    print(f"done: wrote {written} files under {args.out_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FetchError, urllib.error.URLError, TimeoutError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
