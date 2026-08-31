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
SHA256_RE = set("0123456789abcdef")


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


def _run_command(command: list[str], log_path: Path, *, cwd: Path = REPO_ROOT) -> float:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"$ {shlex.join(command)}", file=sys.stderr)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=False)
    except OSError as exc:
        raise ValidationError(f"cannot execute {command[0]}: {exc}") from exc
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
) -> list[str]:
    """Build the GT-free command that generates one candidate manifest."""

    if pair_source not in {"vlad-union", "temporal-pyramid"}:
        raise ValidationError(
            f"electro candidate runner supports vlad-union or temporal-pyramid, got {pair_source!r}"
        )
    if temporal_pyramid_max_offset <= 0:
        raise ValidationError("temporal_pyramid_max_offset must be positive")

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


def validate_match_index(index_path: Path, candidate_index_path: Path) -> dict[str, Any]:
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
) -> dict[str, Any]:
    """Run pending match shards, updating the index only after hash checks."""

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
            elapsed = _run_command(command, root / f"match-{shard_id:06d}.log")
            digest = sha256_file(snapshot_path)
        except (ValidationError, OSError):
            entry["status"] = "failed"
            atomic_json(match_index_path, index)
            raise
        entry.update({"status": "complete", "snapshot_sha256": digest, "elapsed_s": elapsed})
        atomic_json(match_index_path, index)
    validated = validate_match_index(match_index_path, candidate_index_path)
    complete = [entry for entry in validated["index"]["shards"] if entry["status"] == "complete"]
    if len(complete) != len(validated["index"]["shards"]):
        raise ValidationError("match stage ended with incomplete shards")
    return validated


def build_merge_command(merge_binary: Path, output: Path, snapshots: list[Path]) -> list[str]:
    if not snapshots:
        raise ValidationError("cannot merge an empty snapshot list")
    command = [str(merge_binary), "--output", str(output)]
    for snapshot in snapshots:
        command.extend(["--snapshot", str(snapshot)])
    return command


def merge_match_shards(match_index_path: Path, *, merge_binary: Path, output: Path) -> dict[str, Any]:
    index = _load_json(match_index_path, "match index")
    if index.get("schema") != MATCH_INDEX_SCHEMA:
        raise ValidationError("match index has unsupported schema")
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
    elapsed = _run_command(build_merge_command(merge_binary, output, snapshots), root / "merge.log")
    return {"output": str(output), "sha256": sha256_file(output), "shards": len(snapshots), "elapsed_s": elapsed}


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
    parser.add_argument("--pairs-per-shard", type=int, default=256)
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
    parser.add_argument("--feature-suffix", default="_features.txt")
    parser.add_argument("--image-suffix", default=".png")
    parser.add_argument("--min-matches", type=int, default=30)
    parser.add_argument("--match-ratio", type=float, default=0.8)
    parser.add_argument("--min-pnp-inliers", type=int, default=12)
    parser.add_argument("--max-mapper-matches-per-pair", type=int)
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
    try:
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
        if mode in {"prepare", "run"}:
            if not candidate_source.is_file():
                candidate_source.parent.mkdir(parents=True, exist_ok=True)
                _run_command(
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
                    ),
                    artifact_root / "candidate-generation.log",
                )
            split_candidate_manifest(
                candidate_source,
                candidate_dir,
                args.pairs_per_shard,
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
        if mode in {"verify", "match", "map"} and not candidate_index_path.is_file():
            raise ValidationError(f"candidate index is missing: {candidate_index_path}; run --prepare first")
        candidate_summary = validate_candidate_shards(candidate_index_path)
        feature_names = feature_summary.get("image_names")
        if feature_names != candidate_summary["image_names"]:
            raise ValidationError(
                "candidate image order differs from the validated feature manifest"
            )
        match_dir = artifact_root / "matches"
        match_index_path = match_dir / "index.json"
        if mode in {"match", "run"}:
            if not match_index_path.is_file() or args.no_resume:
                prepare_match_index(candidate_index_path, match_dir)
            run_match_shards(
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
            )
            merged_snapshot = artifact_root / "mapping" / "verified-merged.vps"
            merge_match_shards(match_index_path, merge_binary=merge_binary, output=merged_snapshot)
        else:
            merged_snapshot = artifact_root / "mapping" / "verified-merged.vps"
        if mode in {"map", "run"}:
            if not merged_snapshot.is_file():
                raise ValidationError(f"merged snapshot is missing: {merged_snapshot}; run --match first")
            output_model = artifact_root / "mapping" / "colmap"
            _run_command(
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
                    final_ba=not args.no_final_ba,
                ),
                artifact_root / "mapping.log",
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
