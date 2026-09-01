#!/usr/bin/env python3
"""Prepare and run a resumable ETH3D ``electro`` unordered-SfM benchmark.

The runner keeps the expensive local matching stage in deterministic,
image-name-bound candidate shards.  Each shard is written atomically and is
considered complete only when its SHA-256 validates.  Matching workers emit
the existing binary verified-pair snapshot format; the small Rust merge helper
combines disjoint snapshots before one mapper invocation.  Ground truth and
reference poses are intentionally absent from every candidate/mapping command;
they can be used by a separate post-mapping scoring step.

The default mode is ``--verify-only``.  ``--prepare`` creates candidate shards,
``--match`` runs or resumes workers and merges them, ``--map`` runs the mapper,
and ``--run`` performs all three stages.  The module is dependency-free so
manifest checks can run on a fresh machine before Rust, NumPy, or COLMAP are
installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_MAGIC = "visloc_candidate_manifest_v1"
CANDIDATE_INDEX_SCHEMA = "visloc_electro_candidate_shards_v1"
MATCH_INDEX_SCHEMA = "visloc_electro_match_shards_v1"
FEATURE_MANIFEST_SCHEMA = "visloc_electro_feature_manifest_v1"
PERSISTENT_MATCH_WORKER_PLAN_MAGIC = "visloc_match_worker_plan_v1"
SHA256_RE = set("0123456789abcdef")
GNU_TIME = Path("/usr/bin/time")
PERSISTENT_MATCH_ENV = {
    "RAYON_NUM_THREADS": "4",
    "MALLOC_ARENA_MAX": "1",
}
MEMORY_BOUNDED_MERGE_ENV = {
    "MALLOC_ARENA_MAX": "1",
}


class ValidationError(RuntimeError):
    """An electro input or resumable artifact failed validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValidationError(f"{label} must be a 64-character SHA-256")
    lowered = value.lower()
    if any(char not in SHA256_RE for char in lowered):
        raise ValidationError(f"{label} must be a hexadecimal SHA-256")
    return lowered


def _safe_relative(value: str, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or not value or any(part in ("", ".", "..") for part in path.parts):
        raise ValidationError(f"{label} must be a simple relative path: {value!r}")
    return path


def _persistent_safe_relative(value: str, label: str) -> Path:
    """Validate a plan/log path using the same POSIX contract as Rust."""

    if "\\" in value:
        raise ValidationError(f"{label} must use POSIX-style separators: {value!r}")
    return _safe_relative(value, label)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}"
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            temporary.unlink()
        except OSError:
            pass
        raise ValidationError(f"cannot atomically install {path}: {exc}") from exc


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode())


def _noncomment_lines(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read {path}: {exc}") from exc
    return [line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]


def parse_candidate_manifest(path: Path) -> tuple[list[str], list[tuple[int, int]]]:
    """Parse the Rust demo's image-name-bound candidate manifest."""

    names, pairs, _ = parse_candidate_manifest_with_metadata(path)
    return names, pairs


def parse_candidate_manifest_with_metadata(
    path: Path,
) -> tuple[list[str], list[tuple[int, int]], dict[str, str]]:
    """Parse a candidate manifest and its optional deterministic policy block."""

    lines = _noncomment_lines(path)
    cursor = 0

    def next_line(label: str) -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValidationError(f"candidate manifest {path} is truncated while reading {label}")
        value = lines[cursor]
        cursor += 1
        return value

    if next_line("header") != CANDIDATE_MAGIC:
        raise ValidationError(f"candidate manifest {path} has unsupported header")
    image_header = next_line("image count").split()
    if len(image_header) != 2 or image_header[0] != "images":
        raise ValidationError(f"candidate manifest {path} image count must be images N")
    try:
        image_count = int(image_header[1])
    except ValueError as exc:
        raise ValidationError(f"candidate manifest {path} image count is not numeric") from exc
    if image_count < 2:
        raise ValidationError(f"candidate manifest {path} must contain at least two images")
    names: list[str] = []
    for expected in range(image_count):
        fields = next_line("image entry").split(maxsplit=2)
        if len(fields) != 3 or fields[0] != "image":
            raise ValidationError(f"candidate manifest {path} image entry must be image INDEX NAME")
        try:
            index = int(fields[1])
        except ValueError as exc:
            raise ValidationError(f"candidate manifest {path} image index is not numeric") from exc
        if index != expected or not fields[2] or any(char.isspace() for char in fields[2]):
            raise ValidationError(f"candidate manifest {path} image entry {expected} is invalid")
        names.append(fields[2])
    metadata: dict[str, str] = {}
    while cursor < len(lines) and lines[cursor].split(maxsplit=1)[0] == "metadata":
        fields = lines[cursor].split()
        if len(fields) != 3 or not fields[1] or not fields[2]:
            raise ValidationError(
                f"candidate manifest {path} metadata must be metadata KEY VALUE"
            )
        if fields[1] in metadata:
            raise ValidationError(f"candidate manifest {path} repeats metadata key {fields[1]!r}")
        metadata[fields[1]] = fields[2]
        cursor += 1
    pair_header = next_line("pair count").split()
    if len(pair_header) != 2 or pair_header[0] != "pairs":
        raise ValidationError(f"candidate manifest {path} pair count must be pairs N")
    try:
        pair_count = int(pair_header[1])
    except ValueError as exc:
        raise ValidationError(f"candidate manifest {path} pair count is not numeric") from exc
    if pair_count < 0:
        raise ValidationError(f"candidate manifest {path} has a negative pair count")
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pair_number in range(pair_count):
        fields = next_line("pair entry").split()
        if len(fields) != 3 or fields[0] != "pair":
            raise ValidationError(f"candidate manifest {path} pair {pair_number} must be pair I J")
        try:
            first, second = int(fields[1]), int(fields[2])
        except ValueError as exc:
            raise ValidationError(f"candidate manifest {path} pair {pair_number} is not numeric") from exc
        if not (0 <= first < second < image_count):
            raise ValidationError(
                f"candidate manifest {path} pair {pair_number} must satisfy 0 <= I < J < {image_count}"
            )
        pair = (first, second)
        if pair in seen:
            raise ValidationError(f"candidate manifest {path} repeats pair {pair}")
        seen.add(pair)
        pairs.append(pair)
    if cursor != len(lines):
        raise ValidationError(f"candidate manifest {path} has unexpected trailing data")
    return names, pairs, metadata


def write_candidate_manifest(
    path: Path,
    image_names: list[str],
    pairs: Iterable[tuple[int, int]],
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Atomically write one canonical candidate manifest."""

    pairs = list(pairs)
    if len(image_names) < 2:
        raise ValidationError("candidate manifest requires at least two images")
    lines = [CANDIDATE_MAGIC, f"images {len(image_names)}"]
    for index, name in enumerate(image_names):
        if not name or any(char.isspace() for char in name):
            raise ValidationError(f"candidate image name {name!r} cannot contain whitespace")
        lines.append(f"image {index} {name}")
    for key, value in sorted((metadata or {}).items()):
        if not key or not value or any(char.isspace() for char in key) or any(char.isspace() for char in value):
            raise ValidationError(f"candidate metadata must use whitespace-free KEY VALUE: {key!r}={value!r}")
        lines.append(f"metadata {key} {value}")
    lines.append(f"pairs {len(pairs)}")
    seen: set[tuple[int, int]] = set()
    for pair in pairs:
        if len(pair) != 2:
            raise ValidationError(f"candidate pair must contain two indices: {pair!r}")
        first, second = pair
        if not (0 <= first < second < len(image_names)):
            raise ValidationError(f"candidate pair {pair!r} is outside canonical image order")
        if pair in seen:
            raise ValidationError(f"candidate pair {pair!r} is duplicated")
        seen.add(pair)
        lines.append(f"pair {first} {second}")
    _atomic_bytes(path, ("\n".join(lines) + "\n").encode())


def _relative_artifact(path: Path, root: Path, label: str) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValidationError(f"{label} {path} must be under {root}") from exc


def split_candidate_manifest(
    source: Path,
    output_dir: Path,
    pairs_per_shard: int,
    *,
    resume: bool = True,
    retrieval_topk: int | None = None,
    local_stem_window: int | None = None,
    candidate_budget: int | None = None,
    local_grouping: str | None = None,
    pair_source: str | None = None,
    temporal_pyramid_max_offset: int | None = None,
) -> dict[str, Any]:
    """Create deterministic candidate shards and an image-bound index.

    Existing shards are reused only after their full hash and parsed contents
    match the expected contiguous range.  A malformed existing shard is
    replaced with an atomic write, never accepted by size alone.
    """

    if pairs_per_shard <= 0:
        raise ValidationError("pairs_per_shard must be positive")
    source = source.resolve()
    names, pairs, metadata = parse_candidate_manifest_with_metadata(source)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_hash = sha256_file(source)
    shards: list[dict[str, Any]] = []
    for shard_number, start in enumerate(range(0, len(pairs), pairs_per_shard)):
        end = min(start + pairs_per_shard, len(pairs))
        path = output_dir / f"candidate-{shard_number:06d}.txt"
        expected_pairs = pairs[start:end]
        reusable = False
        if resume and path.is_file():
            try:
                existing_names, existing_pairs, existing_metadata = parse_candidate_manifest_with_metadata(path)
                reusable = (
                    existing_names == names
                    and existing_pairs == expected_pairs
                    and existing_metadata == metadata
                )
                if reusable:
                    # A successful parse is not enough: the index records and
                    # later resume checks rely on this exact digest.
                    sha256_file(path)
            except ValidationError:
                reusable = False
        if not reusable:
            write_candidate_manifest(path, names, expected_pairs, metadata=metadata)
        digest = sha256_file(path)
        shards.append(
            {
                "id": shard_number,
                "path": path.name,
                "start": start,
                "end": end,
                "pair_count": end - start,
                "sha256": digest,
                "status": "complete",
            }
        )
    candidate_policy: dict[str, Any] = {
        "retrieval_topk": retrieval_topk,
        "local_stem_window": local_stem_window,
        "candidate_budget": candidate_budget,
    }
    if pair_source is not None:
        candidate_policy["pair_source"] = pair_source
    if temporal_pyramid_max_offset is not None:
        candidate_policy["temporal_pyramid_max_offset"] = temporal_pyramid_max_offset
    if local_grouping is not None:
        candidate_policy["local_grouping"] = local_grouping
    index = {
        "schema": CANDIDATE_INDEX_SCHEMA,
        "source_manifest": str(source),
        "source_manifest_sha256": source_hash,
        "image_names": names,
        "pair_count": len(pairs),
        "pairs_per_shard": pairs_per_shard,
        "candidate_policy": candidate_policy,
        "candidate_manifest_metadata": metadata,
        "shards": shards,
    }
    atomic_json(output_dir / "index.json", index)
    return index


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValidationError(f"cannot parse {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"{label} {path} must contain a JSON object")
    return value


def validate_candidate_shards(index_path: Path) -> dict[str, Any]:
    """Validate every candidate shard and its contiguous global coverage."""

    index_path = index_path.resolve()
    index = _load_json(index_path, "candidate index")
    if index.get("schema") != CANDIDATE_INDEX_SCHEMA:
        raise ValidationError(f"candidate index {index_path} has unsupported schema")
    source_value = index.get("source_manifest")
    if not isinstance(source_value, str) or not source_value:
        raise ValidationError("candidate index source_manifest is missing")
    source = Path(source_value).expanduser()
    expected_source_hash = _sha256(
        index.get("source_manifest_sha256"), "candidate source manifest hash"
    )
    actual_source_hash = sha256_file(source)
    if actual_source_hash != expected_source_hash:
        raise ValidationError(
            f"candidate source manifest hash mismatch: expected {expected_source_hash}, got {actual_source_hash}"
        )
    names = index.get("image_names")
    if not isinstance(names, list) or len(names) < 2 or any(not isinstance(name, str) for name in names):
        raise ValidationError("candidate index image_names must be a non-empty string list")
    pair_count = index.get("pair_count")
    if not isinstance(pair_count, int) or pair_count < 0:
        raise ValidationError("candidate index pair_count must be a non-negative integer")
    policy = index.get("candidate_policy")
    if policy is not None:
        if not isinstance(policy, dict):
            raise ValidationError("candidate index candidate_policy must be an object")
        for key in (
            "retrieval_topk",
            "local_stem_window",
            "candidate_budget",
            "temporal_pyramid_max_offset",
        ):
            value = policy.get(key)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValidationError(f"candidate index candidate_policy {key} must be positive")
        pair_source = policy.get("pair_source")
        if pair_source is not None and pair_source not in {
            "vlad",
            "vlad-mutual",
            "vlad-union",
            "temporal-pyramid",
            "vocab-tree",
            "transitive",
        }:
            raise ValidationError(
                f"candidate index candidate_policy pair_source is unsupported: {pair_source!r}"
            )
        grouping = policy.get("local_grouping")
        if grouping is not None and grouping not in {
            "numeric-stem-v1",
            "rig-prefix-timestamp-v1",
        }:
            raise ValidationError(f"candidate index candidate_policy local_grouping is unsupported: {grouping!r}")
    metadata = index.get("candidate_manifest_metadata", {})
    if not isinstance(metadata, dict) or any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ValidationError("candidate index candidate_manifest_metadata must map strings to strings")
    shards = index.get("shards")
    if not isinstance(shards, list):
        raise ValidationError("candidate index shards must be a list")
    cursor = 0
    seen: set[tuple[int, int]] = set()
    for expected_id, shard in enumerate(shards):
        if not isinstance(shard, dict):
            raise ValidationError(f"candidate shard {expected_id} must be an object")
        if shard.get("id") != expected_id:
            raise ValidationError(f"candidate shards are not ordered at {expected_id}")
        relative = shard.get("path")
        if not isinstance(relative, str):
            raise ValidationError(f"candidate shard {expected_id} path is missing")
        path = index_path.parent / _safe_relative(relative, f"candidate shard {expected_id} path")
        expected_hash = _sha256(shard.get("sha256"), f"candidate shard {expected_id} hash")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValidationError(
                f"candidate shard {expected_id} hash mismatch: expected {expected_hash}, got {actual_hash}"
            )
        shard_names, shard_pairs, shard_metadata = parse_candidate_manifest_with_metadata(path)
        if shard_names != names:
            raise ValidationError(f"candidate shard {expected_id} image order differs from index")
        if shard_metadata != metadata:
            raise ValidationError(f"candidate shard {expected_id} metadata differs from index")
        start, end = shard.get("start"), shard.get("end")
        if not isinstance(start, int) or not isinstance(end, int) or start != cursor or end < start:
            raise ValidationError(f"candidate shard {expected_id} has a non-contiguous range")
        if shard.get("pair_count") != end - start or len(shard_pairs) != end - start:
            raise ValidationError(f"candidate shard {expected_id} pair count/range disagrees")
        for pair in shard_pairs:
            if pair in seen:
                raise ValidationError(f"candidate shards repeat pair {pair}")
            seen.add(pair)
        cursor = end
    if cursor != pair_count or len(seen) != pair_count:
        raise ValidationError(
            f"candidate shards cover {cursor} pairs but index declares {pair_count}"
        )
    return {
        "index": index,
        "index_sha256": sha256_file(index_path),
        "image_names": names,
        "pair_count": pair_count,
        "candidate_manifest_metadata": metadata,
        "shards": shards,
    }


def feature_manifest(
    features_dir: Path,
    *,
    feature_suffix: str = "_features.txt",
    source_dir: Path | None = None,
) -> dict[str, Any]:
    """Build a compact hash manifest for a flat external-feature directory."""

    files = sorted(
        path for path in features_dir.iterdir() if path.is_file() and path.name.endswith(feature_suffix)
    )
    if not files:
        raise ValidationError(f"no {feature_suffix!r} files found in {features_dir}")
    images = []
    for path in files:
        try:
            with path.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip() and not line.lstrip().startswith("#"))
        except (OSError, UnicodeDecodeError) as exc:
            raise ValidationError(f"cannot inspect feature file {path}: {exc}") from exc
        entry: dict[str, Any] = {
            "name": path.name[: -len(feature_suffix)] + ".png",
            "feature": path.name,
            "rows": rows,
            "sha256": sha256_file(path),
        }
        if source_dir is not None:
            source = source_dir / entry["name"]
            if not source.is_file():
                raise ValidationError(f"source image for {entry['name']} is missing: {source}")
            entry["source_sha256"] = sha256_file(source)
        images.append(entry)
    return {
        "schema": FEATURE_MANIFEST_SCHEMA,
        "feature_suffix": feature_suffix,
        "image_count": len(images),
        "image_names": [entry["name"] for entry in images],
        "images": images,
    }


def write_feature_manifest(path: Path, manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != FEATURE_MANIFEST_SCHEMA:
        raise ValidationError("feature manifest has unsupported schema")
    atomic_json(path, manifest)


def validate_feature_manifest(
    path: Path, features_dir: Path, *, source_dir: Path | None = None
) -> dict[str, Any]:
    manifest = _load_json(path, "feature manifest")
    if manifest.get("schema") != FEATURE_MANIFEST_SCHEMA:
        raise ValidationError(f"feature manifest {path} has unsupported schema")
    suffix = manifest.get("feature_suffix")
    images = manifest.get("images")
    if not isinstance(suffix, str) or not isinstance(images, list) or not images:
        raise ValidationError(f"feature manifest {path} has invalid image entries")
    if manifest.get("image_count") != len(images):
        raise ValidationError(f"feature manifest {path} image_count disagrees")
    seen: set[str] = set()
    total_rows = 0
    for entry in images:
        if not isinstance(entry, dict) or not isinstance(entry.get("feature"), str):
            raise ValidationError(f"feature manifest {path} has a malformed image entry")
        filename = entry["feature"]
        if filename in seen:
            raise ValidationError(f"feature manifest {path} repeats {filename}")
        seen.add(filename)
        feature_path = features_dir / _safe_relative(filename, "feature manifest path")
        expected = _sha256(entry.get("sha256"), f"feature {filename} hash")
        actual = sha256_file(feature_path)
        if actual != expected:
            raise ValidationError(f"feature {filename} hash mismatch: expected {expected}, got {actual}")
        try:
            with feature_path.open(encoding="utf-8") as stream:
                rows = sum(1 for line in stream if line.strip() and not line.lstrip().startswith("#"))
        except (OSError, UnicodeDecodeError) as exc:
            raise ValidationError(f"cannot inspect feature file {feature_path}: {exc}") from exc
        if rows != entry.get("rows"):
            raise ValidationError(f"feature {filename} row count mismatch: expected {entry.get('rows')}, got {rows}")
        if source_dir is not None:
            source_name = entry.get("name")
            if not isinstance(source_name, str):
                raise ValidationError(f"feature manifest {path} image name is missing")
            source_path = source_dir / _safe_relative(source_name, "feature source image path")
            if not source_path.is_file():
                raise ValidationError(f"source image is missing: {source_path}")
            expected_source = _sha256(entry.get("source_sha256"), f"source image {source_name} hash")
            actual_source = sha256_file(source_path)
            if actual_source != expected_source:
                raise ValidationError(
                    f"source image {source_name} hash mismatch: expected {expected_source}, got {actual_source}"
                )
        total_rows += rows
    return {
        "image_count": len(images),
        "image_names": [entry.get("name") for entry in images],
        "total_rows": total_rows,
        "sha256": sha256_file(path),
    }


def parse_gnu_time(path: Path) -> dict[str, Any]:
    """Parse the stable resource fields emitted by ``/usr/bin/time -v``."""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValidationError(f"cannot read GNU time report {path}: {exc}") from exc
    fields: dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.strip().split(":", 1)
        fields[key] = value.strip()

    def number(key: str, *, integer: bool = False) -> float | int:
        value = fields.get(key)
        if value is None:
            raise ValidationError(f"GNU time report {path} is missing {key!r}")
        try:
            return int(value) if integer else float(value)
        except ValueError as exc:
            raise ValidationError(
                f"GNU time report {path} has invalid {key!r}: {value!r}"
            ) from exc

    return {
        "user_s": number("User time (seconds)"),
        "system_s": number("System time (seconds)"),
        "peak_rss_kib": number("Maximum resident set size (kbytes)", integer=True),
        "major_page_faults": number("Major (requiring I/O) page faults", integer=True),
        "minor_page_faults": number("Minor (reclaiming a frame) page faults", integer=True),
        "filesystem_inputs": number("File system inputs", integer=True),
        "filesystem_outputs": number("File system outputs", integer=True),
        "exit_status": number("Exit status", integer=True),
        "report": str(path.resolve()),
        "report_sha256": sha256_file(path),
    }


def measured_phase(path: Path, elapsed_s: float) -> dict[str, Any]:
    measurement = parse_gnu_time(path)
    measurement["elapsed_s"] = elapsed_s
    return measurement


def carry_forward_phase_ledger(
    current: dict[str, Any], previous: dict[str, Any] | None
) -> dict[str, Any]:
    """Keep completed phase measurements across prepare/match/map invocations."""

    if not isinstance(previous, dict):
        return current
    result = dict(current)
    for key in (
        "candidate_generation",
        "candidate_sharding",
        "persistent_worker",
        "merge",
        "mapping",
    ):
        if result.get(key) is None and previous.get(key) is not None:
            result[key] = previous[key]
    return result


def tree_bytes(path: Path) -> int:
    """Return allocated artifact bytes without following symlinks."""

    total = 0
    if not path.exists():
        return total
    for entry in path.rglob("*"):
        if entry.is_file() and not entry.is_symlink():
            try:
                total += entry.stat().st_size
            except OSError as exc:
                raise ValidationError(f"cannot stat artifact {entry}: {exc}") from exc
    return total


def _run_command(
    command: list[str],
    log_path: Path,
    *,
    cwd: Path = REPO_ROOT,
    timing_path: Path | None = None,
    env_overrides: dict[str, str] | None = None,
) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    executed = command
    if timing_path is not None:
        if not GNU_TIME.is_file():
            raise ValidationError(f"GNU time executable is missing: {GNU_TIME}")
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        executed = [str(GNU_TIME), "-v", "-o", str(timing_path), "--", *command]
    started = time.monotonic()
    print(f"$ {shlex.join(executed)}", file=sys.stderr)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                executed,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "LC_ALL": "C", **(env_overrides or {})},
            )
    except OSError as exc:
        raise ValidationError(f"cannot execute {executed[0]}: {exc}") from exc
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise ValidationError(f"command failed ({result.returncode}): {shlex.join(command)}\n" + "\n".join(tail))
    return time.monotonic() - started


def build_candidate_command(
    binary: Path,
    *,
    features_dir: Path,
    calibration_dir: Path,
    candidate_manifest: Path,
    feature_suffix: str = "_features.txt",
    image_suffix: str = ".png",
    images_dir: Path | None = None,
    retrieval_topk: int = 32,
    local_stem_window: int = 3,
    candidate_budget: int | None = None,
    rig_local_grouping: bool = False,
    pair_source: str = "vlad-union",
    temporal_pyramid_max_offset: int = 32,
    stream_candidate_features: bool = False,
    retrieval_backend: str = "exact",
    ann_tables: int = 8,
    ann_bits: int = 0,
    ann_probes: int = 6,
) -> list[str]:
    """Build the GT-free command that generates one candidate manifest."""

    if pair_source not in {"vlad-union", "temporal-pyramid"}:
        raise ValidationError(
            f"electro candidate runner supports vlad-union or temporal-pyramid, got {pair_source!r}"
        )
    if temporal_pyramid_max_offset <= 0:
        raise ValidationError("temporal_pyramid_max_offset must be positive")
    if retrieval_backend not in {"exact", "lsh"}:
        raise ValidationError(f"unsupported retrieval_backend {retrieval_backend!r}")
    if retrieval_backend == "lsh" and not stream_candidate_features:
        raise ValidationError("LSH retrieval requires stream_candidate_features")
    if (
        ann_tables <= 0
        or not 0 <= ann_bits <= 63
        or not 0 <= ann_probes <= 63
        or (ann_bits != 0 and ann_probes > ann_bits)
    ):
        raise ValidationError(
            "ANN settings require ann_tables >= 1, ann_bits auto (0) or 1..=63, and ann_probes <= the effective bit count"
        )

    command = [
        str(binary),
        "--feature-extractor",
        "files",
        "--features-dir",
        str(features_dir),
        "--feature-suffix",
        feature_suffix,
        "--image-suffix",
        image_suffix,
        "--input-colmap-calibration",
        str(calibration_dir),
        "--pair-source",
        pair_source,
        "--retrieval-topk",
        str(retrieval_topk),
        "--export-candidate-manifest",
        str(candidate_manifest),
        # The demo parser keeps --out-colmap mandatory for all modes, even
        # though candidate export exits before a model is written.
        "--out-colmap",
        str(candidate_manifest.parent / f"unused-model-{candidate_manifest.stem}"),
    ]
    if images_dir is not None:
        command.extend(["--images-dir", str(images_dir)])
    if pair_source == "vlad-union":
        command.extend(["--local-stem-window", str(local_stem_window)])
    if pair_source == "vlad-union" and rig_local_grouping:
        command.append("--rig-local-grouping")
    if pair_source == "temporal-pyramid":
        command.extend(["--temporal-pyramid-max-offset", str(temporal_pyramid_max_offset)])
    if candidate_budget is not None:
        command.extend(["--candidate-budget", str(candidate_budget)])
    if stream_candidate_features:
        command.append("--stream-candidate-features")
    if retrieval_backend == "lsh":
        command.extend(
            [
                "--retrieval-backend",
                "lsh",
                "--ann-tables",
                str(ann_tables),
                "--ann-bits",
                str(ann_bits),
                "--ann-probes",
                str(ann_probes),
            ]
        )
    return command


def build_match_command(
    binary: Path,
    *,
    features_dir: Path,
    calibration_dir: Path,
    candidate_shard: Path,
    snapshot: Path,
    feature_suffix: str = "_features.txt",
    image_suffix: str = ".png",
    images_dir: Path | None = None,
    min_matches: int = 30,
    match_ratio: float = 0.8,
    max_mapper_matches_per_pair: int | None = None,
) -> list[str]:
    """Build one bounded, export-only matcher/verifier worker command."""

    command = [
        str(binary),
        "--feature-extractor",
        "files",
        "--features-dir",
        str(features_dir),
        "--feature-suffix",
        feature_suffix,
        "--image-suffix",
        image_suffix,
        "--input-colmap-calibration",
        str(calibration_dir),
        "--candidate-manifest",
        str(candidate_shard),
        "--min-matches",
        str(min_matches),
        "--match-ratio",
        str(match_ratio),
        "--verification-mode",
        "full",
        "--export-verified-pairs-snapshot",
        str(snapshot),
        "--export-verified-pairs-only",
        "--out-colmap",
        str(snapshot.parent / f"unused-model-{snapshot.stem}"),
    ]
    if images_dir is not None:
        command.extend(["--images-dir", str(images_dir)])
    # Validate the optional cap for callers that pass one through a shared
    # configuration, but deliberately do not put it on worker commands:
    # snapshots are lossless and the final mapper owns the explicit resource
    # guard.
    if max_mapper_matches_per_pair is not None:
        if max_mapper_matches_per_pair <= 0:
            raise ValidationError("max_mapper_matches_per_pair must be positive")
    return command


def build_persistent_match_command(
    binary: Path,
    *,
    features_dir: Path,
    calibration_dir: Path,
    plan: Path,
    feature_suffix: str = "_features.txt",
    image_suffix: str = ".png",
    images_dir: Path | None = None,
    min_matches: int = 30,
    match_ratio: float = 0.8,
) -> list[str]:
    """Build the single-process frozen NN/full persistent worker command."""

    command = [
        str(binary),
        "--feature-extractor",
        "files",
        "--features-dir",
        str(features_dir),
        "--feature-suffix",
        feature_suffix,
        "--image-suffix",
        image_suffix,
        "--input-colmap-calibration",
        str(calibration_dir),
        "--verification-mode",
        "full",
        "--matcher",
        "nn",
        "--mapper",
        "incremental",
        "--min-matches",
        str(min_matches),
        "--match-ratio",
        str(match_ratio),
        "--persistent-match-worker-plan",
        str(plan),
        "--out-colmap",
        str(plan.parent / "matches" / "unused-model-persistent"),
    ]
    if images_dir is not None:
        command.extend(["--images-dir", str(images_dir)])
    return command


def build_mapping_command(
    binary: Path,
    *,
    features_dir: Path,
    calibration_dir: Path,
    merged_snapshot: Path,
    output_model: Path,
    feature_suffix: str = "_features.txt",
    image_suffix: str = ".png",
    images_dir: Path | None = None,
    min_pnp_inliers: int = 12,
    max_mapper_matches_per_pair: int | None = None,
    snapshot_keypoints_only: bool = False,
    periodic_ba_min_registered_images: int | None = None,
    seed_trials: int | None = None,
    ba_linear_solver: str | None = None,
    ba_max_iterations: int | None = None,
    post_refinement_registration: bool = False,
    final_iterative_refinement: bool = False,
    global_ba_max_refinements: int | None = None,
    final_ba: bool = True,
) -> list[str]:
    command = [
        str(binary),
        "--feature-extractor",
        "files",
        "--features-dir",
        str(features_dir),
        "--feature-suffix",
        feature_suffix,
        "--image-suffix",
        image_suffix,
        "--input-colmap-calibration",
        str(calibration_dir),
        "--import-verified-pairs-snapshot",
        str(merged_snapshot),
        "--verification-mode",
        "full",
        "--mapper",
        "incremental",
        "--min-pnp-inliers",
        str(min_pnp_inliers),
        "--out-colmap",
        str(output_model),
    ]
    if images_dir is not None:
        command.extend(["--images-dir", str(images_dir)])
    if max_mapper_matches_per_pair is not None:
        if max_mapper_matches_per_pair <= 0:
            raise ValidationError("max_mapper_matches_per_pair must be positive")
        command.extend(["--max-mapper-matches-per-pair", str(max_mapper_matches_per_pair)])
    if snapshot_keypoints_only:
        command.append("--snapshot-keypoints-only")
    positive_options = (
        ("--periodic-ba-min-registered-images", periodic_ba_min_registered_images),
        ("--seed-trials", seed_trials),
        ("--ba-max-iterations", ba_max_iterations),
    )
    for option, value in positive_options:
        if value is not None:
            if value <= 0:
                raise ValidationError(f"{option} must be positive")
            command.extend([option, str(value)])
    if ba_linear_solver is not None:
        if ba_linear_solver not in {"dense", "sparse", "auto"}:
            raise ValidationError("ba_linear_solver must be dense, sparse, or auto")
        command.extend(["--ba-linear-solver", ba_linear_solver])
    if post_refinement_registration:
        command.append("--post-refinement-registration")
    if final_iterative_refinement:
        command.append("--final-iterative-refinement")
    if global_ba_max_refinements is not None:
        if global_ba_max_refinements < 0:
            raise ValidationError("--global-ba-max-refinements must be non-negative")
        command.extend(["--global-ba-max-refinements", str(global_ba_max_refinements)])
    if not final_ba:
        command.append("--no-final-ba")
    return command


def prepare_match_index(candidate_index_path: Path, output_dir: Path) -> dict[str, Any]:
    candidate = validate_candidate_shards(candidate_index_path)
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for shard in candidate["shards"]:
        shards.append(
            {
                "id": shard["id"],
                "candidate_path": shard["path"],
                "candidate_sha256": shard["sha256"],
                "snapshot_path": f"verified-{shard['id']:06d}.vps",
                "status": "pending",
            }
        )
    index = {
        "schema": MATCH_INDEX_SCHEMA,
        "candidate_index_sha256": candidate["index_sha256"],
        "image_names": candidate["image_names"],
        "pair_count": candidate["pair_count"],
        "shards": shards,
    }
    atomic_json(output_dir / "index.json", index)
    return index


def parse_persistent_match_worker_plan(path: Path) -> dict[str, Any]:
    """Parse the dependency-free plan consumed by the Rust persistent worker."""

    lines = _noncomment_lines(path)
    cursor = 0

    def next_line(label: str) -> str:
        nonlocal cursor
        if cursor >= len(lines):
            raise ValidationError(
                f"persistent match worker plan {path} is truncated while reading {label}"
            )
        line = lines[cursor]
        cursor += 1
        return line

    def count_line(label: str) -> int:
        fields = next_line(label).split()
        if len(fields) != 2 or fields[0] != label:
            raise ValidationError(f"persistent match worker plan requires `{label} N`")
        try:
            value = int(fields[1])
        except ValueError as exc:
            raise ValidationError(f"persistent match worker {label} count is not numeric") from exc
        if value < 0:
            raise ValidationError(f"persistent match worker {label} count must be non-negative")
        return value

    if next_line("header") != PERSISTENT_MATCH_WORKER_PLAN_MAGIC:
        raise ValidationError(
            f"persistent match worker plan {path} has unsupported header"
        )
    image_count = count_line("images")
    if image_count < 2:
        raise ValidationError("persistent match worker plan needs at least two images")
    image_names = []
    for expected_index in range(image_count):
        fields = next_line("image entry").split()
        if len(fields) != 3 or fields[0] != "image":
            raise ValidationError("persistent match worker image entry must be image INDEX NAME")
        try:
            index = int(fields[1])
        except ValueError as exc:
            raise ValidationError("persistent match worker image index is not numeric") from exc
        if index != expected_index or not fields[2]:
            raise ValidationError("persistent match worker image entries must be ordered")
        image_names.append(fields[2])

    def hash_line(label: str) -> str:
        fields = next_line(label).split()
        if len(fields) != 2 or fields[0] != label:
            raise ValidationError(f"persistent match worker plan requires `{label} SHA256`")
        return _sha256(fields[1], label)

    candidate_index_sha256 = hash_line("candidate_index_sha256")
    feature_manifest_sha256 = hash_line("feature_manifest_sha256")
    pair_count = count_line("pairs")
    if pair_count <= 0:
        raise ValidationError("persistent match worker plan must contain at least one pair")
    shard_count = count_line("shards")
    if shard_count <= 0:
        raise ValidationError("persistent match worker plan must contain at least one shard")
    shards = []
    all_paths: set[Path] = set()
    previous_id: int | None = None
    for expected_id in range(shard_count):
        fields = next_line("shard entry").split()
        if len(fields) != 5 or fields[0] != "shard":
            raise ValidationError(
                "persistent match worker shard entry must be shard ID CANDIDATE SNAPSHOT CANDIDATE_SHA256"
            )
        try:
            shard_id = int(fields[1])
        except ValueError as exc:
            raise ValidationError("persistent match worker shard id is not numeric") from exc
        if shard_id < 0:
            raise ValidationError("persistent match worker shard id must be non-negative")
        if previous_id is not None and shard_id <= previous_id:
            raise ValidationError("persistent match worker shard IDs must be strictly increasing")
        previous_id = shard_id
        candidate_path = _persistent_safe_relative(
            fields[2], f"persistent shard {shard_id} candidate path"
        )
        snapshot_path = _persistent_safe_relative(
            fields[3], f"persistent shard {shard_id} snapshot path"
        )
        if candidate_path == snapshot_path:
            raise ValidationError(
                f"persistent match worker shard {shard_id} reuses one path for candidate and snapshot"
            )
        if candidate_path in all_paths or snapshot_path in all_paths:
            raise ValidationError(
                f"persistent match worker repeats candidate or snapshot path at shard {shard_id}"
            )
        all_paths.add(candidate_path)
        all_paths.add(snapshot_path)
        shards.append(
            {
                "id": shard_id,
                "candidate_path": candidate_path.as_posix(),
                "snapshot_path": snapshot_path.as_posix(),
                "candidate_sha256": _sha256(fields[4], f"persistent shard {shard_id} candidate hash"),
            }
        )
    if cursor != len(lines):
        raise ValidationError(f"persistent match worker plan {path} has unexpected trailing data")
    return {
        "schema": PERSISTENT_MATCH_WORKER_PLAN_MAGIC,
        "image_names": image_names,
        "pair_count": pair_count,
        "candidate_index_sha256": candidate_index_sha256,
        "feature_manifest_sha256": feature_manifest_sha256,
        "shards": shards,
    }


def _artifact_relative(path: Path, artifact_root: Path, label: str) -> str:
    try:
        relative = path.resolve().relative_to(artifact_root.resolve())
    except ValueError as exc:
        raise ValidationError(f"{label} is outside the artifact root: {path}") from exc
    return _safe_relative(relative.as_posix(), label).as_posix()


def validate_persistent_match_worker_plan(
    plan_path: Path,
    candidate_index_path: Path,
    match_index_path: Path,
    feature_manifest_path: Path,
    *,
    allow_completed: bool = False,
) -> dict[str, Any]:
    """Validate plan bindings against the candidate/match/feature artifacts."""

    plan = parse_persistent_match_worker_plan(plan_path)
    candidate = validate_candidate_shards(candidate_index_path)
    match = validate_match_index(match_index_path, candidate_index_path)
    artifact_root = match_index_path.resolve().parent.parent
    if plan["image_names"] != candidate["image_names"]:
        raise ValidationError("persistent worker plan image order differs from candidate index")
    if plan["candidate_index_sha256"] != candidate["index_sha256"]:
        raise ValidationError("persistent worker plan candidate index hash differs")
    if plan["feature_manifest_sha256"] != sha256_file(feature_manifest_path):
        raise ValidationError("persistent worker plan feature manifest hash differs")
    candidate_prefix = _artifact_relative(
        candidate_index_path.resolve().parent, artifact_root, "candidate index directory"
    )
    match_prefix = _artifact_relative(
        match_index_path.resolve().parent, artifact_root, "match index directory"
    )
    pending = [entry for entry in match["index"]["shards"] if entry["status"] != "complete"]
    if allow_completed:
        match_by_id = {entry["id"]: entry for entry in match["index"]["shards"]}
        expected_entries = []
        for plan_shard in plan["shards"]:
            entry = match_by_id.get(plan_shard["id"])
            if entry is None:
                raise ValidationError(
                    f"persistent worker plan refers to unknown match shard {plan_shard['id']}"
                )
            expected_entries.append(entry)
    else:
        expected_entries = pending
        if len(plan["shards"]) != len(pending):
            raise ValidationError("persistent worker plan does not contain exactly the pending shards")
    expected_pairs = 0
    for plan_shard, match_entry in zip(plan["shards"], expected_entries):
        candidate_entry = candidate["shards"][match_entry["id"]]
        expected_candidate = f"{candidate_prefix}/{candidate_entry['path']}"
        expected_snapshot = f"{match_prefix}/{match_entry['snapshot_path']}"
        if plan_shard["id"] != match_entry["id"]:
            raise ValidationError("persistent worker plan shard order differs from match index")
        if plan_shard["candidate_path"] != expected_candidate:
            raise ValidationError(f"persistent worker shard {plan_shard['id']} candidate path differs")
        if plan_shard["snapshot_path"] != expected_snapshot:
            raise ValidationError(f"persistent worker shard {plan_shard['id']} snapshot path differs")
        if plan_shard["candidate_sha256"] != candidate_entry["sha256"]:
            raise ValidationError(f"persistent worker shard {plan_shard['id']} candidate hash differs")
        expected_pairs += int(candidate_entry["pair_count"])
    if plan["pair_count"] != expected_pairs:
        raise ValidationError(
            f"persistent worker plan pair count {plan['pair_count']} differs from pending {expected_pairs}"
        )
    return {
        "plan_path": str(plan_path.resolve()),
        "plan": plan,
        "plan_sha256": sha256_file(plan_path),
        "candidate": candidate,
        "match": match,
        "pending": pending,
    }


def write_persistent_match_worker_plan(
    candidate_index_path: Path,
    match_index_path: Path,
    feature_manifest_path: Path,
    *,
    output: Path | None = None,
) -> Path | None:
    """Write a plan for pending match shards, or return None when complete."""

    candidate = validate_candidate_shards(candidate_index_path)
    match = validate_match_index(match_index_path, candidate_index_path)
    pending = [entry for entry in match["index"]["shards"] if entry["status"] != "complete"]
    if not pending:
        return None
    artifact_root = match_index_path.resolve().parent.parent
    candidate_prefix = _artifact_relative(
        candidate_index_path.resolve().parent, artifact_root, "candidate index directory"
    )
    match_prefix = _artifact_relative(
        match_index_path.resolve().parent, artifact_root, "match index directory"
    )
    feature_manifest_sha256 = sha256_file(feature_manifest_path)
    lines = [
        PERSISTENT_MATCH_WORKER_PLAN_MAGIC,
        f"images {len(candidate['image_names'])}",
        *[f"image {index} {name}" for index, name in enumerate(candidate["image_names"])],
        f"candidate_index_sha256 {candidate['index_sha256']}",
        f"feature_manifest_sha256 {feature_manifest_sha256}",
        f"pairs {sum(int(candidate['shards'][entry['id']]['pair_count']) for entry in pending)}",
        f"shards {len(pending)}",
    ]
    for entry in pending:
        shard = candidate["shards"][entry["id"]]
        candidate_path = f"{candidate_prefix}/{shard['path']}"
        snapshot_path = f"{match_prefix}/{entry['snapshot_path']}"
        lines.append(
            f"shard {entry['id']} {candidate_path} {snapshot_path} {shard['sha256']}"
        )
    plan_path = (output or artifact_root / "match-worker.plan").resolve()
    _atomic_bytes(plan_path, ("\n".join(lines) + "\n").encode("utf-8"))
    validate_persistent_match_worker_plan(
        plan_path, candidate_index_path, match_index_path, feature_manifest_path
    )
    return plan_path


def validate_match_index(
    index_path: Path,
    candidate_index_path: Path,
    *,
    require_complete: bool = False,
) -> dict[str, Any]:
    candidate = validate_candidate_shards(candidate_index_path)
    index_path = index_path.resolve()
    index = _load_json(index_path, "match index")
    if index.get("schema") != MATCH_INDEX_SCHEMA:
        raise ValidationError(f"match index {index_path} has unsupported schema")
    if index.get("candidate_index_sha256") != candidate["index_sha256"]:
        raise ValidationError("match index candidate index hash differs; regenerate the match plan")
    if index.get("image_names") != candidate["image_names"] or index.get("pair_count") != candidate["pair_count"]:
        raise ValidationError("match index image/pair envelope differs from candidate index")
    shards = index.get("shards")
    if not isinstance(shards, list) or len(shards) != len(candidate["shards"]):
        raise ValidationError("match index shard list does not match candidate index")
    for expected_id, (entry, candidate_entry) in enumerate(zip(shards, candidate["shards"])):
        if not isinstance(entry, dict) or entry.get("id") != expected_id:
            raise ValidationError(f"match index shard {expected_id} is malformed")
        if entry.get("candidate_path") != candidate_entry["path"] or entry.get("candidate_sha256") != candidate_entry["sha256"]:
            raise ValidationError(f"match index shard {expected_id} candidate binding differs")
        status = entry.get("status")
        if status not in {"pending", "running", "complete", "failed"}:
            raise ValidationError(f"match index shard {expected_id} has invalid status {status!r}")
        if require_complete and status != "complete":
            raise ValidationError(
                f"match shard {expected_id} is not complete (status {status!r})"
            )
        if status == "complete":
            snapshot_path = index_path.parent / _safe_relative(entry.get("snapshot_path", ""), f"match shard {expected_id} snapshot path")
            expected_hash = _sha256(entry.get("snapshot_sha256"), f"match shard {expected_id} snapshot hash")
            actual_hash = sha256_file(snapshot_path)
            if actual_hash != expected_hash:
                raise ValidationError(f"match shard {expected_id} snapshot hash mismatch")
    return {"index": index, "index_sha256": sha256_file(index_path), "candidate": candidate}


def run_match_shards(
    candidate_index_path: Path,
    match_index_path: Path,
    *,
    binary: Path,
    features_dir: Path,
    calibration_dir: Path,
    images_dir: Path | None = None,
    feature_suffix: str = "_features.txt",
    image_suffix: str = ".png",
    min_matches: int = 30,
    match_ratio: float = 0.8,
    resume: bool = True,
    persistent_matcher: bool = False,
    feature_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Run pending match shards, updating the index only after hash checks."""

    if persistent_matcher:
        if feature_manifest_path is None:
            raise ValidationError(
                "persistent matcher requires the validated feature manifest path"
            )
        return run_persistent_matcher(
            candidate_index_path,
            match_index_path,
            binary=binary,
            features_dir=features_dir,
            calibration_dir=calibration_dir,
            feature_manifest_path=feature_manifest_path,
            images_dir=images_dir,
            feature_suffix=feature_suffix,
            image_suffix=image_suffix,
            min_matches=min_matches,
            match_ratio=match_ratio,
            resume=resume,
        )

    plan = validate_match_index(match_index_path, candidate_index_path)
    index = plan["index"]
    root = match_index_path.resolve().parent
    for entry in index["shards"]:
        shard_id = entry["id"]
        snapshot_path = root / _safe_relative(entry["snapshot_path"], f"match shard {shard_id} snapshot path")
        if entry["status"] == "complete":
            if resume:
                continue
            raise ValidationError(f"match shard {shard_id} is already complete; pass --resume")
        candidate_path = candidate_index_path.resolve().parent / _safe_relative(entry["candidate_path"], f"candidate shard {shard_id} path")
        entry["status"] = "running"
        atomic_json(match_index_path, index)
        command = build_match_command(
            binary,
            features_dir=features_dir,
            calibration_dir=calibration_dir,
            candidate_shard=candidate_path,
            snapshot=snapshot_path,
            feature_suffix=feature_suffix,
            image_suffix=image_suffix,
            images_dir=images_dir,
            min_matches=min_matches,
            match_ratio=match_ratio,
        )
        try:
            timing_path = root / "timing" / f"match-{shard_id:06d}.time.txt"
            elapsed = _run_command(
                command,
                root / f"match-{shard_id:06d}.log",
                timing_path=timing_path,
            )
            digest = sha256_file(snapshot_path)
        except (ValidationError, OSError):
            entry["status"] = "failed"
            atomic_json(match_index_path, index)
            raise
        entry.update(
            {
                "status": "complete",
                "snapshot_sha256": digest,
                "elapsed_s": elapsed,
                "measurement": measured_phase(timing_path, elapsed),
            }
        )
        atomic_json(match_index_path, index)
    validated = validate_match_index(match_index_path, candidate_index_path)
    complete = [entry for entry in validated["index"]["shards"] if entry["status"] == "complete"]
    if len(complete) != len(validated["index"]["shards"]):
        raise ValidationError("match stage ended with incomplete shards")
    return validated


def _parse_persistent_key_value_record(line: str, prefix: str) -> dict[str, str]:
    if not line.startswith(prefix + " "):
        raise ValidationError(f"persistent worker log line does not start with {prefix!r}")
    record: dict[str, str] = {}
    for token in line[len(prefix) + 1 :].split():
        key, separator, value = token.partition("=")
        if not separator or not key or not value or key in record:
            raise ValidationError(f"malformed persistent worker record: {line!r}")
        record[key] = value
    return record


def parse_persistent_match_plan_header(path: Path) -> dict[str, str]:
    """Read the worker's flushed plan-binding line from its stdout log."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read persistent worker log {path}: {exc}") from exc
    for line in lines:
        if line.startswith("persistent-match-plan "):
            record = _parse_persistent_key_value_record(line, "persistent-match-plan")
            if set(record) != {"candidate_index_sha256", "feature_manifest_sha256"}:
                raise ValidationError(
                    "persistent worker plan-binding line has unexpected fields"
                )
            for key in ("candidate_index_sha256", "feature_manifest_sha256"):
                _sha256(record.get(key), f"persistent worker {key}")
            return record
    raise ValidationError(f"persistent worker log {path} has no plan-binding line")


def parse_persistent_match_completions(path: Path) -> list[dict[str, Any]]:
    """Parse flushed per-shard completion records, including a killed prefix."""

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        raise ValidationError(f"cannot read persistent worker log {path}: {exc}") from exc
    required = {
        "shard_id",
        "candidate_path",
        "snapshot_path",
        "candidate_sha256",
        "candidate_pairs",
        "pairs",
        "accepted",
        "ordered_edge_fnv1a64",
        "unordered_edge_fnv1a64",
        "elapsed_s",
    }
    completions: list[dict[str, Any]] = []
    seen: set[int] = set()
    for line in lines:
        if not line.startswith("persistent-match-complete "):
            continue
        record = _parse_persistent_key_value_record(line, "persistent-match-complete")
        if set(record) != required:
            raise ValidationError(
                f"persistent worker completion has unexpected fields: {sorted(record)}"
            )
        try:
            shard_id = int(record["shard_id"])
            candidate_pairs = int(record["candidate_pairs"])
            pairs = int(record["pairs"])
            accepted = int(record["accepted"])
            elapsed_s = float(record["elapsed_s"])
        except ValueError as exc:
            raise ValidationError(f"persistent worker completion has invalid numeric field: {line!r}") from exc
        if shard_id < 0 or candidate_pairs < 0 or pairs < 0 or accepted < 0:
            raise ValidationError(f"persistent worker completion has a negative count: {line!r}")
        if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
            raise ValidationError(f"persistent worker completion has invalid elapsed_s: {line!r}")
        if shard_id in seen:
            raise ValidationError(f"persistent worker repeats completion for shard {shard_id}")
        seen.add(shard_id)
        candidate_path = _persistent_safe_relative(
            record["candidate_path"], f"persistent shard {shard_id} candidate path"
        )
        snapshot_path = _persistent_safe_relative(
            record["snapshot_path"], f"persistent shard {shard_id} snapshot path"
        )
        if candidate_path == snapshot_path:
            raise ValidationError(
                f"persistent worker shard {shard_id} reuses one path for candidate and snapshot"
            )

        def edge_hash(value: str, label: str) -> str:
            if len(value) != 16 or any(char not in "0123456789abcdefABCDEF" for char in value):
                raise ValidationError(f"persistent worker {label} must be a 16-digit hexadecimal hash")
            return value.lower()

        completions.append(
            {
                "shard_id": shard_id,
                "candidate_path": candidate_path.as_posix(),
                "snapshot_path": snapshot_path.as_posix(),
                "candidate_sha256": _sha256(
                    record["candidate_sha256"], f"persistent shard {shard_id} candidate hash"
                ),
                "candidate_pairs": candidate_pairs,
                "pairs": pairs,
                "accepted": accepted,
                "ordered_edge_fnv1a64": edge_hash(
                    record["ordered_edge_fnv1a64"], "ordered edge hash"
                ),
                "unordered_edge_fnv1a64": edge_hash(
                    record["unordered_edge_fnv1a64"], "unordered edge hash"
                ),
                "elapsed_s": elapsed_s,
            }
        )
    return completions


def _apply_persistent_match_completions(
    index: dict[str, Any],
    plan_validation: dict[str, Any],
    completions: list[dict[str, Any]],
    *,
    worker_measurement: dict[str, Any] | None = None,
    worker_elapsed_s: float | None = None,
    require_all: bool,
) -> None:
    """Install only hash-valid completed shards into a match index."""

    plan = plan_validation["plan"]
    pending_entries = {
        entry["id"]: entry
        for entry in index["shards"]
        if entry["status"] != "complete"
    }
    plan_entries = {entry["id"]: entry for entry in plan["shards"]}
    seen: set[int] = set()
    # Completion paths are relative to the plan's artifact root.  Derive that
    # root from the validated plan path rather than trusting a log to name an
    # arbitrary output location.
    plan_path = plan_validation.get("plan_path")
    if not isinstance(plan_path, str) or not plan_path:
        raise ValidationError("persistent worker validation omitted plan path")
    plan_root = Path(plan_path).resolve().parent
    for completion in completions:
        shard_id = completion["shard_id"]
        if shard_id in seen:
            raise ValidationError(f"persistent worker repeats completion for shard {shard_id}")
        seen.add(shard_id)
        entry = pending_entries.get(shard_id)
        expected = plan_entries.get(shard_id)
        if entry is None or expected is None:
            raise ValidationError(f"persistent worker completed unexpected shard {shard_id}")
        if completion["candidate_path"] != expected["candidate_path"]:
            raise ValidationError(f"persistent worker shard {shard_id} candidate path differs")
        if completion["snapshot_path"] != expected["snapshot_path"]:
            raise ValidationError(f"persistent worker shard {shard_id} snapshot path differs")
        if completion["candidate_sha256"] != expected["candidate_sha256"]:
            raise ValidationError(f"persistent worker shard {shard_id} candidate hash differs")
        candidate_entry = plan_validation["candidate"]["shards"][shard_id]
        if completion["candidate_pairs"] != candidate_entry["pair_count"]:
            raise ValidationError(f"persistent worker shard {shard_id} candidate pair count differs")
        if completion["pairs"] > completion["candidate_pairs"]:
            raise ValidationError(f"persistent worker shard {shard_id} verified pair count exceeds candidates")
        snapshot_path = plan_root / completion["snapshot_path"]
        actual_hash = sha256_file(snapshot_path)
        entry.update(
            {
                "status": "complete",
                "snapshot_sha256": actual_hash,
                "elapsed_s": completion["elapsed_s"],
                "ordered_edge_fnv1a64": completion["ordered_edge_fnv1a64"],
                "unordered_edge_fnv1a64": completion["unordered_edge_fnv1a64"],
                "accepted_correspondences": completion["accepted"],
            }
        )
        if worker_measurement is not None:
            measurement = dict(worker_measurement)
            measurement["elapsed_s"] = completion["elapsed_s"]
            if worker_elapsed_s is not None:
                measurement["worker_elapsed_s"] = worker_elapsed_s
            entry["measurement"] = measurement
    missing = sorted(set(pending_entries) - seen)
    if require_all and missing:
        raise ValidationError(f"persistent worker omitted completion records for shards {missing}")


def _recover_persistent_worker_log(
    index: dict[str, Any],
    match_index_path: Path,
    candidate_index_path: Path,
    feature_manifest_path: Path,
) -> None:
    """Recover flushed snapshots if the runner died before its index update."""

    worker_plan = index.get("worker_plan")
    if not isinstance(worker_plan, str) or not worker_plan:
        return
    plan_path = Path(worker_plan).expanduser().resolve()
    artifact_root = match_index_path.resolve().parent.parent
    try:
        plan_path.relative_to(artifact_root)
    except ValueError as exc:
        raise ValidationError("persistent worker plan is outside the artifact root") from exc
    log_path = match_index_path.resolve().parent / "persistent-match.log"
    if not plan_path.is_file() or not log_path.is_file():
        return
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ValidationError(f"cannot read persistent worker log {log_path}: {exc}") from exc
    if "persistent-match-" not in log_text:
        return
    plan_validation = validate_persistent_match_worker_plan(
        plan_path,
        candidate_index_path,
        match_index_path,
        feature_manifest_path,
        allow_completed=True,
    )
    plan_validation["plan_path"] = str(plan_path)
    header = parse_persistent_match_plan_header(log_path)
    plan = plan_validation["plan"]
    if header["candidate_index_sha256"] != plan["candidate_index_sha256"]:
        raise ValidationError("persistent worker recovery candidate index binding differs")
    if header["feature_manifest_sha256"] != plan["feature_manifest_sha256"]:
        raise ValidationError("persistent worker recovery feature manifest binding differs")
    completions = parse_persistent_match_completions(log_path)
    plan_ids = {entry["id"] for entry in plan["shards"]}
    if any(completion["shard_id"] not in plan_ids for completion in completions):
        raise ValidationError("persistent worker recovery contains an unknown shard")
    pending_ids = {
        entry["id"] for entry in index["shards"] if entry["status"] != "complete"
    }
    # A runner that was interrupted after atomically updating the index may
    # leave the same completion line in its log.  The index hash validation
    # above already authenticates those completed snapshots; only apply the
    # still-pending prefix to the current index.
    completions = [
        completion for completion in completions if completion["shard_id"] in pending_ids
    ]
    if not completions:
        return
    before = {
        entry["id"]: entry.get("status") for entry in index["shards"]
    }
    _apply_persistent_match_completions(
        index, plan_validation, completions, require_all=False
    )
    if any(before[entry["id"]] != entry.get("status") for entry in index["shards"]):
        atomic_json(match_index_path, index)


def run_persistent_matcher(
    candidate_index_path: Path,
    match_index_path: Path,
    *,
    binary: Path,
    features_dir: Path,
    calibration_dir: Path,
    feature_manifest_path: Path,
    images_dir: Path | None = None,
    feature_suffix: str = "_features.txt",
    image_suffix: str = ".png",
    min_matches: int = 30,
    match_ratio: float = 0.8,
    resume: bool = True,
) -> dict[str, Any]:
    """Run all pending shards through one Rust feature-bank process."""

    validated = validate_match_index(match_index_path, candidate_index_path)
    index = validated["index"]
    if not resume and any(entry["status"] == "complete" for entry in index["shards"]):
        raise ValidationError("a persistent match shard is already complete; pass --resume")
    _recover_persistent_worker_log(
        index, match_index_path, candidate_index_path, feature_manifest_path
    )
    plan_path = write_persistent_match_worker_plan(
        candidate_index_path, match_index_path, feature_manifest_path
    )
    if plan_path is None:
        existing = index.get("persistent_worker")
        if isinstance(existing, dict):
            return {**validated, "persistent_worker": existing}
        return validated
    plan_validation = validate_persistent_match_worker_plan(
        plan_path, candidate_index_path, match_index_path, feature_manifest_path
    )
    plan_validation["plan_path"] = str(plan_path.resolve())
    pending_ids = {entry["id"] for entry in plan_validation["pending"]}
    for entry in index["shards"]:
        if entry["id"] in pending_ids:
            entry["status"] = "running"
    index["worker_mode"] = "persistent-v1"
    index["worker_plan"] = str(plan_path.resolve())
    atomic_json(match_index_path, index)
    root = match_index_path.resolve().parent
    command = build_persistent_match_command(
        binary,
        features_dir=features_dir,
        calibration_dir=calibration_dir,
        plan=plan_path,
        feature_suffix=feature_suffix,
        image_suffix=image_suffix,
        images_dir=images_dir,
        min_matches=min_matches,
        match_ratio=match_ratio,
    )
    timing_path = root.parent / "timing" / "persistent-match.time.txt"
    log_path = root / "persistent-match.log"
    try:
        worker_elapsed = _run_command(
            command,
            log_path,
            timing_path=timing_path,
            env_overrides=PERSISTENT_MATCH_ENV,
        )
    except (ValidationError, OSError):
        # The Rust worker flushes one completion line after every atomic
        # snapshot.  Recover that flushed prefix before marking the remainder
        # failed, so a SIGKILL never recomputes a valid completed shard.
        try:
            partial = parse_persistent_match_completions(log_path)
            _apply_persistent_match_completions(index, plan_validation, partial, require_all=False)
        except ValidationError:
            partial = []
        for entry in index["shards"]:
            if entry["id"] in pending_ids and entry["status"] != "complete":
                entry["status"] = "failed"
        atomic_json(match_index_path, index)
        raise
    header = parse_persistent_match_plan_header(log_path)
    if header["candidate_index_sha256"] != plan_validation["plan"]["candidate_index_sha256"]:
        raise ValidationError("persistent worker candidate index binding differs from plan")
    if header["feature_manifest_sha256"] != plan_validation["plan"]["feature_manifest_sha256"]:
        raise ValidationError("persistent worker feature manifest binding differs from plan")
    completions = parse_persistent_match_completions(log_path)
    worker_measurement = measured_phase(timing_path, worker_elapsed)
    _apply_persistent_match_completions(
        index,
        plan_validation,
        completions,
        worker_measurement=worker_measurement,
        worker_elapsed_s=worker_elapsed,
        require_all=True,
    )
    persistent_worker = {
        "mode": "persistent-v1",
        "plan": str(plan_path.resolve()),
        "plan_sha256": sha256_file(plan_path),
        "elapsed_s": worker_elapsed,
        "shard_elapsed_sum_s": sum(entry["elapsed_s"] for entry in completions),
        "measurement": worker_measurement,
        "environment": dict(PERSISTENT_MATCH_ENV),
        "shards": len(completions),
    }
    index["persistent_worker"] = persistent_worker
    atomic_json(match_index_path, index)
    validated = validate_match_index(match_index_path, candidate_index_path)
    return {**validated, "persistent_worker": persistent_worker}


def build_merge_command(merge_binary: Path, output: Path, snapshots: list[Path]) -> list[str]:
    if not snapshots:
        raise ValidationError("cannot merge an empty snapshot list")
    command = [str(merge_binary), "--output", str(output)]
    for snapshot in snapshots:
        command.extend(["--snapshot", str(snapshot)])
    return command


def merge_match_shards(
    match_index_path: Path,
    *,
    candidate_index_path: Path | None = None,
    merge_binary: Path,
    output: Path,
) -> dict[str, Any]:
    """Merge only a hash-valid, complete match index."""

    if candidate_index_path is None:
        candidate_index_path = (
            match_index_path.resolve().parent.parent / "candidates" / "index.json"
        )
    index = validate_match_index(
        match_index_path, candidate_index_path, require_complete=True
    )["index"]
    root = match_index_path.resolve().parent
    snapshots = []
    for entry in index.get("shards", []):
        if not isinstance(entry, dict) or entry.get("status") != "complete":
            raise ValidationError("cannot merge while a match shard is incomplete")
        path = root / _safe_relative(entry.get("snapshot_path", ""), "match snapshot path")
        expected = _sha256(entry.get("snapshot_sha256"), "match snapshot hash")
        if sha256_file(path) != expected:
            raise ValidationError(f"match snapshot hash mismatch: {path}")
        snapshots.append(path)
    timing_path = root / "timing" / "merge.time.txt"
    elapsed = _run_command(
        build_merge_command(merge_binary, output, snapshots),
        root / "merge.log",
        timing_path=timing_path,
        env_overrides=MEMORY_BOUNDED_MERGE_ENV,
    )
    return {
        "output": str(output),
        "sha256": sha256_file(output),
        "shards": len(snapshots),
        "elapsed_s": elapsed,
        "measurement": measured_phase(timing_path, elapsed),
    }


def _binary(path: str | None, default: Path) -> Path:
    value = Path(path).expanduser() if path else default
    if not value.is_file():
        raise ValidationError(f"executable is missing: {value}; build it first or pass an override")
    return value.resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="validate existing manifests/artifacts (default)")
    mode.add_argument("--prepare", action="store_true", help="generate and shard the candidate manifest")
    mode.add_argument("--match", action="store_true", help="run/resume match shards and merge snapshots")
    mode.add_argument("--map", action="store_true", help="map an existing merged snapshot")
    mode.add_argument("--run", action="store_true", help="prepare, match, merge, and map")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--calibration-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True, help="external run root")
    parser.add_argument("--candidate-manifest", type=Path, help="existing generated candidate manifest")
    parser.add_argument("--feature-manifest", type=Path)
    parser.add_argument(
        "--pairs-per-shard",
        type=int,
        help="candidate pairs per shard (default: 32 with --persistent-matcher, otherwise 256)",
    )
    parser.add_argument("--retrieval-topk", type=int, default=32)
    parser.add_argument(
        "--pair-source",
        choices=("vlad-union", "temporal-pyramid"),
        default="vlad-union",
        help="candidate policy (temporal-pyramid is rig-aware and uses VLAD only as budget fill)",
    )
    parser.add_argument("--local-stem-window", type=int, default=3)
    parser.add_argument(
        "--rig-local-grouping",
        action="store_true",
        help="use camera-prefix/timestamp local edges plus same-timestamp rig edges",
    )
    parser.add_argument("--temporal-pyramid-max-offset", type=int, default=32)
    parser.add_argument("--candidate-budget", type=int)
    parser.add_argument(
        "--stream-candidate-features",
        action="store_true",
        help="stream feature files while building global descriptors",
    )
    parser.add_argument(
        "--retrieval-backend",
        choices=("exact", "lsh"),
        default="exact",
        help="global-descriptor retrieval backend (LSH requires streaming)",
    )
    parser.add_argument("--ann-tables", type=int, default=8)
    parser.add_argument("--ann-bits", type=int, default=0, help="LSH bits (0: scale automatically with image count)")
    parser.add_argument("--ann-probes", type=int, default=6)
    parser.add_argument("--feature-suffix", default="_features.txt")
    parser.add_argument("--image-suffix", default=".png")
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--match-ratio", type=float, default=0.8)
    parser.add_argument(
        "--persistent-matcher",
        action="store_true",
        help="run all pending match shards in one plan-driven Rust worker (opt-in)",
    )
    parser.add_argument("--min-pnp-inliers", type=int, default=12)
    parser.add_argument("--max-mapper-matches-per-pair", type=int)
    parser.add_argument(
        "--snapshot-keypoints-only",
        action="store_true",
        help="drop descriptor payloads during file-backed snapshot replay (opt-in)",
    )
    parser.add_argument("--periodic-ba-min-registered-images", type=int)
    parser.add_argument("--seed-trials", type=int)
    parser.add_argument("--ba-linear-solver", choices=("dense", "sparse", "auto"))
    parser.add_argument("--ba-max-iterations", type=int)
    parser.add_argument("--post-refinement-registration", action="store_true")
    parser.add_argument("--final-iterative-refinement", action="store_true")
    parser.add_argument("--global-ba-max-refinements", type=int)
    parser.add_argument(
        "--no-final-ba",
        action="store_true",
        help="skip the mapper's final bundle adjustment (for a phase-isolated baseline)",
    )
    parser.add_argument("--binary")
    parser.add_argument("--merge-binary")
    parser.add_argument("--resume", action="store_true", default=False)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    mode = "verify" if args.verify_only or not any((args.prepare, args.match, args.map, args.run)) else next(name for name, value in (("prepare", args.prepare), ("match", args.match), ("map", args.map), ("run", args.run)) if value)
    pairs_per_shard = args.pairs_per_shard
    if pairs_per_shard is None:
        pairs_per_shard = 32 if args.persistent_matcher else 256
    try:
        if args.persistent_matcher and mode not in {"match", "run"}:
            raise ValidationError("--persistent-matcher is only valid with --match or --run")
        artifact_root = args.artifact_root.expanduser().resolve()
        if mode == "verify":
            if not artifact_root.is_dir():
                raise ValidationError(f"artifact root is missing: {artifact_root}")
        else:
            artifact_root.mkdir(parents=True, exist_ok=True)
        features_dir = args.features_dir.expanduser().resolve()
        calibration_dir = args.calibration_dir.expanduser().resolve()
        if not features_dir.is_dir():
            raise ValidationError(f"features directory is missing: {features_dir}")
        if not calibration_dir.is_dir():
            raise ValidationError(f"calibration directory is missing: {calibration_dir}")
        feature_validation_started = time.monotonic()
        feature_manifest_path = (args.feature_manifest or artifact_root / "features.json").resolve()
        if feature_manifest_path.is_file():
            feature_summary = validate_feature_manifest(
                feature_manifest_path, features_dir, source_dir=args.images_dir
            )
        else:
            if mode == "verify":
                raise ValidationError(f"feature manifest is missing: {feature_manifest_path}")
            feature_summary = feature_manifest(features_dir, feature_suffix=args.feature_suffix, source_dir=args.images_dir)
            write_feature_manifest(feature_manifest_path, feature_summary)
            feature_summary = validate_feature_manifest(
                feature_manifest_path, features_dir, source_dir=args.images_dir
            )
        feature_validation_elapsed = time.monotonic() - feature_validation_started
        candidate_source = args.candidate_manifest
        if candidate_source is None:
            candidate_source = artifact_root / "candidates.txt"
        candidate_source = candidate_source.expanduser().resolve()
        binary = None
        merge_binary = None
        if mode in {"prepare", "match", "map", "run"}:
            binary = _binary(args.binary, REPO_ROOT / "target" / "release" / "examples" / "unordered_sfm_demo")
        if mode in {"match", "run"}:
            merge_binary = _binary(args.merge_binary, REPO_ROOT / "target" / "release" / "examples" / "merge_verified_pair_snapshots")
        candidate_dir = artifact_root / "candidates"
        candidate_index_path = candidate_dir / "index.json"
        candidate_measurement = None
        candidate_sharding_elapsed = None
        merge_result = None
        mapping_measurement = None
        persistent_worker_measurement = None
        if mode in {"prepare", "run"}:
            if not candidate_source.is_file():
                candidate_source.parent.mkdir(parents=True, exist_ok=True)
                candidate_timing_path = artifact_root / "timing" / "candidate-generation.time.txt"
                candidate_elapsed = _run_command(
                    build_candidate_command(
                        binary,
                        features_dir=features_dir,
                        calibration_dir=calibration_dir,
                        candidate_manifest=candidate_source,
                        feature_suffix=args.feature_suffix,
                        image_suffix=args.image_suffix,
                        images_dir=args.images_dir,
                        retrieval_topk=args.retrieval_topk,
                        local_stem_window=args.local_stem_window,
                        candidate_budget=args.candidate_budget,
                        rig_local_grouping=args.rig_local_grouping,
                        pair_source=args.pair_source,
                        temporal_pyramid_max_offset=args.temporal_pyramid_max_offset,
                        stream_candidate_features=args.stream_candidate_features,
                        retrieval_backend=args.retrieval_backend,
                        ann_tables=args.ann_tables,
                        ann_bits=args.ann_bits,
                        ann_probes=args.ann_probes,
                    ),
                    artifact_root / "candidate-generation.log",
                    timing_path=candidate_timing_path,
                )
                candidate_measurement = measured_phase(
                    candidate_timing_path, candidate_elapsed
                )
            sharding_started = time.monotonic()
            split_candidate_manifest(
                candidate_source,
                candidate_dir,
                pairs_per_shard,
                resume=not args.no_resume,
                retrieval_topk=args.retrieval_topk,
                local_stem_window=(
                    args.local_stem_window if args.pair_source == "vlad-union" else None
                ),
                candidate_budget=args.candidate_budget,
                pair_source=args.pair_source,
                temporal_pyramid_max_offset=(
                    args.temporal_pyramid_max_offset
                    if args.pair_source == "temporal-pyramid"
                    else None
                ),
                local_grouping=(
                    "rig-prefix-timestamp-v1"
                    if args.rig_local_grouping or args.pair_source == "temporal-pyramid"
                    else None
                ),
            )
            candidate_sharding_elapsed = time.monotonic() - sharding_started
        if mode in {"verify", "match", "map"} and not candidate_index_path.is_file():
            raise ValidationError(f"candidate index is missing: {candidate_index_path}; run --prepare first")
        candidate_summary = validate_candidate_shards(candidate_index_path)
        if args.persistent_matcher and any(
            int(shard["pair_count"]) > 32 for shard in candidate_summary["shards"]
        ):
            raise ValidationError(
                "--persistent-matcher requires candidate shards with at most 32 pairs; "
                "prepare them with --pairs-per-shard 32"
            )
        feature_names = feature_summary.get("image_names")
        if feature_names != candidate_summary["image_names"]:
            raise ValidationError(
                "candidate image order differs from the validated feature manifest"
            )
        match_dir = artifact_root / "matches"
        match_index_path = match_dir / "index.json"
        if mode == "verify" and match_index_path.is_file():
            validate_match_index(
                match_index_path, candidate_index_path, require_complete=True
            )
        if mode in {"match", "run"}:
            if not match_index_path.is_file() or args.no_resume:
                prepare_match_index(candidate_index_path, match_dir)
            match_result = run_match_shards(
                candidate_index_path,
                match_index_path,
                binary=binary,
                features_dir=features_dir,
                calibration_dir=calibration_dir,
                images_dir=args.images_dir,
                feature_suffix=args.feature_suffix,
                image_suffix=args.image_suffix,
                min_matches=args.min_matches,
                match_ratio=args.match_ratio,
                resume=not args.no_resume,
                persistent_matcher=args.persistent_matcher,
                feature_manifest_path=feature_manifest_path,
            )
            persistent_worker_measurement = match_result.get("persistent_worker")
            merged_snapshot = artifact_root / "mapping" / "verified-merged.vps"
            merge_result = merge_match_shards(
                match_index_path,
                candidate_index_path=candidate_index_path,
                merge_binary=merge_binary,
                output=merged_snapshot,
            )
        else:
            merged_snapshot = artifact_root / "mapping" / "verified-merged.vps"
        if mode in {"map", "run"}:
            if not merged_snapshot.is_file():
                raise ValidationError(f"merged snapshot is missing: {merged_snapshot}; run --match first")
            output_model = artifact_root / "mapping" / "colmap"
            mapping_timing_path = artifact_root / "timing" / "mapping.time.txt"
            mapping_elapsed = _run_command(
                build_mapping_command(
                    binary,
                    features_dir=features_dir,
                    calibration_dir=calibration_dir,
                    merged_snapshot=merged_snapshot,
                    output_model=output_model,
                    feature_suffix=args.feature_suffix,
                    image_suffix=args.image_suffix,
                    images_dir=args.images_dir,
                    min_pnp_inliers=args.min_pnp_inliers,
                    max_mapper_matches_per_pair=args.max_mapper_matches_per_pair,
                    snapshot_keypoints_only=args.snapshot_keypoints_only,
                    periodic_ba_min_registered_images=args.periodic_ba_min_registered_images,
                    seed_trials=args.seed_trials,
                    ba_linear_solver=args.ba_linear_solver,
                    ba_max_iterations=args.ba_max_iterations,
                    post_refinement_registration=args.post_refinement_registration,
                    final_iterative_refinement=args.final_iterative_refinement,
                    global_ba_max_refinements=args.global_ba_max_refinements,
                    final_ba=not args.no_final_ba,
                ),
                artifact_root / "mapping.log",
                timing_path=mapping_timing_path,
            )
            mapping_measurement = measured_phase(mapping_timing_path, mapping_elapsed)
        match_measurements = []
        if match_index_path.is_file():
            match_index = _load_json(match_index_path, "match index")
            match_measurements = [
                entry["measurement"]
                for entry in match_index.get("shards", [])
                if isinstance(entry, dict) and isinstance(entry.get("measurement"), dict)
            ]
        phase_ledger = {
            "feature_manifest_validation": {
                "elapsed_s": feature_validation_elapsed,
                "feature_extraction_included": False,
            },
            "candidate_generation": candidate_measurement,
            "candidate_sharding": (
                {"elapsed_s": candidate_sharding_elapsed}
                if candidate_sharding_elapsed is not None
                else None
            ),
            "persistent_worker": persistent_worker_measurement,
            "match_shards": match_measurements,
            "matching_elapsed_sum_s": sum(
                float(entry.get("elapsed_s", 0.0)) for entry in match_measurements
            ),
            "matching_peak_rss_kib": max(
                (int(entry.get("peak_rss_kib", 0)) for entry in match_measurements),
                default=0,
            ),
            "merge": merge_result["measurement"] if merge_result else None,
            "mapping": mapping_measurement,
        }
        previous_summary_path = artifact_root / "summary.json"
        previous_phase_ledger = None
        if previous_summary_path.is_file():
            previous_summary = _load_json(previous_summary_path, "previous run summary")
            if previous_summary.get("schema") == "visloc_electro_run_summary_v1":
                previous_phase_ledger = previous_summary.get("phase_ledger")
        phase_ledger = carry_forward_phase_ledger(
            phase_ledger, previous_phase_ledger
        )
        summary = {
            "schema": "visloc_electro_run_summary_v1",
            "mode": mode,
            "features": feature_summary,
            "candidate_index": str(candidate_index_path),
            "candidate_index_sha256": sha256_file(candidate_index_path),
            "candidate_pairs": candidate_summary["pair_count"],
            "match_index": str(match_index_path) if match_index_path.is_file() else None,
            "merged_snapshot": str(merged_snapshot) if merged_snapshot.is_file() else None,
            "merged_snapshot_sha256": sha256_file(merged_snapshot) if merged_snapshot.is_file() else None,
            "phase_ledger": phase_ledger,
            "artifact_bytes": tree_bytes(artifact_root),
            "ground_truth_used_for_selection_or_mapping": False,
        }
        atomic_json(artifact_root / "summary.json", summary)
        print(json.dumps(summary, sort_keys=True, indent=2))
        return 0
    except ValidationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
