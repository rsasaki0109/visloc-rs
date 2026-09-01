#!/usr/bin/env python3
"""Bounded synthetic I/O stress for large-scale SfM manifests.

This deliberately makes no geometry claim.  It streams an O(NK) local pair
schedule into atomic, hash-checked shards without ever materializing the full
pair list.  The output is root-independent so interrupted/resumed and clean
runs can be compared byte for byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time
from typing import Iterator


SCHEMA = "visloc_scale_io_stress_v1"
INTERRUPTED = 75
MAX_NEIGHBORS = 64
MAX_PAIRS_PER_SHARD = 65_536


class StressError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def expected_pair_count(image_count: int, neighbors: int) -> int:
    width = min(neighbors, max(0, image_count - 1))
    return width * image_count - width * (width + 1) // 2


def iter_pairs(image_count: int, neighbors: int) -> Iterator[tuple[int, int]]:
    for first in range(image_count):
        for second in range(first + 1, min(image_count, first + neighbors + 1)):
            yield first, second


def iter_shard_payloads(
    image_count: int, neighbors: int, pairs_per_shard: int
) -> Iterator[tuple[int, int, bytes]]:
    shard_id = 0
    lines: list[str] = []
    for first, second in iter_pairs(image_count, neighbors):
        lines.append(f"{first} {second}\n")
        if len(lines) == pairs_per_shard:
            yield shard_id, len(lines), "".join(lines).encode()
            shard_id += 1
            lines.clear()
    if lines:
        yield shard_id, len(lines), "".join(lines).encode()


def image_manifest_bytes(image_count: int) -> bytes:
    lines = [f"{SCHEMA}\n", f"images {image_count}\n"]
    lines.extend(f"image {index} synthetic/{index:09d}.png\n" for index in range(image_count))
    return "".join(lines).encode()


def peak_rss_kib() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform != "darwin" else value // 1024


def load_existing_index(path: Path, config: dict[str, int | str]) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise StressError(f"invalid index {path}: {error}") from error
    if not isinstance(value, dict):
        raise StressError("existing index must be a JSON object")
    if value.get("schema") != SCHEMA or value.get("config") != config:
        raise StressError("existing index schema/config does not match this run")
    if not isinstance(value.get("complete"), bool) or not isinstance(value.get("shards"), list):
        raise StressError("existing index has invalid complete/shards fields")
    seen: set[int] = set()
    for entry in value["shards"]:
        if not isinstance(entry, dict):
            raise StressError("existing index shard entry must be an object")
        required = {"bytes": int, "id": int, "path": str, "rows": int, "sha256": str}
        if any(not isinstance(entry.get(key), kind) for key, kind in required.items()):
            raise StressError("existing index has a malformed shard entry")
        shard_id = entry["id"]
        if shard_id < 0 or shard_id in seen:
            raise StressError("existing index has a duplicate/invalid shard id")
        seen.add(shard_id)
    return value


def run_stress(args: argparse.Namespace) -> tuple[int, dict]:
    started = time.monotonic()
    root = args.artifact_root.resolve()
    index_path = root / "index.json"
    image_path = root / "images.manifest"
    shard_dir = root / "candidate-shards"
    pair_count = expected_pair_count(args.images, args.neighbors)
    shard_count = (pair_count + args.pairs_per_shard - 1) // args.pairs_per_shard
    config: dict[str, int | str] = {
        "image_count": args.images,
        "neighbors": args.neighbors,
        "pair_policy": "forward-local-window-v1",
        "pairs_per_shard": args.pairs_per_shard,
    }
    config_sha = sha256_bytes(canonical_json(config))
    existing = load_existing_index(index_path, config)
    if existing is not None and not (args.resume or args.verify_only):
        raise StressError("artifact root already has an index; pass --resume")
    if existing is not None and existing["complete"] and args.inject_stop_after_shards is not None:
        raise StressError("cannot inject an interruption into a complete run")
    if args.inject_stop_after_shards is not None and args.inject_stop_after_shards > shard_count:
        raise StressError("injected stop exceeds the expected shard count")

    expected_images = image_manifest_bytes(args.images)
    expected_images_sha = sha256_bytes(expected_images)
    if existing is not None:
        expected_metadata = {
            "config_sha256": config_sha,
            "expected_pair_count": pair_count,
            "expected_shard_count": shard_count,
            "image_manifest_sha256": expected_images_sha,
        }
        if any(existing.get(key) != value for key, value in expected_metadata.items()):
            raise StressError("existing index deterministic metadata is corrupt")
    if image_path.exists():
        if sha256_file(image_path) != expected_images_sha:
            raise StressError("image manifest hash/content mismatch")
    elif args.verify_only:
        raise StressError("image manifest is missing")
    else:
        atomic_write(image_path, expected_images)

    indexed_shards = {
        int(entry["id"]): entry for entry in (existing or {}).get("shards", [])
    }
    if any(shard_id >= shard_count for shard_id in indexed_shards):
        raise StressError("existing index contains an out-of-range shard")
    existing_final_paths = set(shard_dir.glob("candidate-*.txt")) if shard_dir.exists() else set()
    if existing is None and existing_final_paths and not args.resume:
        raise StressError("unindexed shards exist; pass --resume to validate and adopt them")
    entries: list[dict[str, int | str]] = []
    aggregate = hashlib.sha256()
    reused = 0
    written = 0

    for shard_id, rows, payload in iter_shard_payloads(
        args.images, args.neighbors, args.pairs_per_shard
    ):
        relative = f"candidate-shards/candidate-{shard_id:06d}.txt"
        path = root / relative
        expected_sha = sha256_bytes(payload)
        old = indexed_shards.get(shard_id)
        if old is not None and (
            old.get("path") != relative
            or old.get("rows") != rows
            or old.get("sha256") != expected_sha
            or old.get("bytes") != len(payload)
        ):
            raise StressError(f"index metadata mismatch for shard {shard_id}")
        if path.exists():
            if path.stat().st_size != len(payload) or sha256_file(path) != expected_sha:
                raise StressError(f"shard {shard_id} is corrupt")
            reused += 1
        elif args.verify_only:
            raise StressError(f"shard {shard_id} is missing")
        else:
            atomic_write(path, payload)
            written += 1
        aggregate.update(bytes.fromhex(expected_sha))
        entries.append(
            {
                "bytes": len(payload),
                "id": shard_id,
                "path": relative,
                "rows": rows,
                "sha256": expected_sha,
            }
        )
        existing_final_paths.discard(path)

        processed = shard_id + 1
        if args.inject_stop_after_shards == processed:
            partial = {
                "complete": False,
                "config": config,
                "config_sha256": config_sha,
                "expected_pair_count": pair_count,
                "expected_shard_count": shard_count,
                "image_manifest_sha256": expected_images_sha,
                "schema": SCHEMA,
                "shards": entries,
            }
            atomic_write(index_path, canonical_json(partial))
            return INTERRUPTED, {
                "complete": False,
                "processed_shards": processed,
                "reused_shards": reused,
                "written_shards": written,
            }

    if existing_final_paths:
        raise StressError("unexpected extra candidate shard exists")
    if len(entries) != shard_count or sum(int(x["rows"]) for x in entries) != pair_count:
        raise StressError("internal shard/pair count mismatch")
    final_index = {
        "aggregate_shard_sha256": aggregate.hexdigest(),
        "complete": True,
        "config": config,
        "config_sha256": config_sha,
        "expected_pair_count": pair_count,
        "expected_shard_count": shard_count,
        "image_manifest_sha256": expected_images_sha,
        "schema": SCHEMA,
        "shards": entries,
    }
    final_bytes = canonical_json(final_index)
    final_sha = sha256_bytes(final_bytes)
    if args.verify_only:
        if existing is None or not existing.get("complete"):
            raise StressError("complete index is missing")
        if sha256_file(index_path) != final_sha:
            raise StressError("complete index is non-canonical or corrupt")
    else:
        atomic_write(index_path, final_bytes)

    rss_kib = peak_rss_kib()
    if args.max_rss_mib is not None and rss_kib > args.max_rss_mib * 1024:
        raise StressError(
            f"peak RSS {rss_kib / 1024:.1f} MiB exceeds {args.max_rss_mib} MiB"
        )
    return 0, {
        "aggregate_shard_sha256": aggregate.hexdigest(),
        "complete": True,
        "elapsed_seconds": time.monotonic() - started,
        "image_count": args.images,
        "index_sha256": final_sha,
        "neighbors": args.neighbors,
        "pair_count": pair_count,
        "peak_rss_kib": rss_kib,
        "reused_shards": reused,
        "schema": SCHEMA,
        "shard_count": shard_count,
        "written_shards": written,
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--artifact-root", required=True, type=Path)
    value.add_argument("--images", required=True, type=int)
    value.add_argument("--neighbors", type=int, default=32)
    value.add_argument("--pairs-per-shard", type=int, default=4096)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--verify-only", action="store_true")
    value.add_argument("--inject-stop-after-shards", type=int)
    value.add_argument("--max-rss-mib", type=int)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if (
        args.images < 2
        or not 1 <= args.neighbors <= MAX_NEIGHBORS
        or not 1 <= args.pairs_per_shard <= MAX_PAIRS_PER_SHARD
    ):
        print(
            f"error: images >= 2, neighbors in 1..{MAX_NEIGHBORS}, "
            f"pairs-per-shard in 1..{MAX_PAIRS_PER_SHARD} required",
            file=sys.stderr,
        )
        return 2
    if args.inject_stop_after_shards is not None and args.inject_stop_after_shards < 1:
        print("error: --inject-stop-after-shards must be >= 1", file=sys.stderr)
        return 2
    if args.verify_only and (args.resume or args.inject_stop_after_shards is not None):
        print("error: --verify-only cannot be combined with mutation options", file=sys.stderr)
        return 2
    try:
        status, summary = run_stress(args)
    except StressError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
