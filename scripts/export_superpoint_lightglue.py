#!/usr/bin/env python3
"""Export SuperPoint features and LightGlue matches for visloc-rs.

This helper is optional and intentionally not part of normal cargo tests. It
expects the Python `lightglue`, `torch`, and image-loading dependencies to be
installed by the caller.

Output formats consumed by `visloc_io::external_deep`:

    features: X Y SCORE D0 D1 ...
    matches:  QUERY_IDX TRAIN_IDX CONFIDENCE [DISTANCE]

Pair mode writes:

    image0_features.txt
    image1_features.txt
    matches.txt

Sequence mode writes files consumed by
`examples/stereo_vo_external_deep_files.rs`:

    frame_000000_left_features.txt
    frame_000000_right_features.txt
    frame_000000_stereo_matches.txt
    frame_000001_temporal_matches.txt

Sequence and mono exports can be split into explicit source-index ranges with
`--start-index`/`--end-index` (end exclusive).  Those names retain source
indices so disjoint workers can share an output directory.  Every output is
written through a same-directory temporary and atomic rename; `--skip-existing`
performs structural validation, and `--manifest` records per-file SHA-256
digests.  `--validate-only` validates existing outputs without importing the
optional LightGlue stack.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, TextIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image0", type=Path, default=None)
    parser.add_argument("--image1", type=Path, default=None)
    parser.add_argument("--left-dir", type=Path, default=None)
    parser.add_argument("--right-dir", type=Path, default=None)
    parser.add_argument("--mono-dir", type=Path, default=None,
                        help="single-camera sequence mode: features only, no stereo/temporal matches")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help=(
            "sequence/mono source index to include first (inclusive); unlike "
            "--start-frame, output names retain the source index"
        ),
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="sequence/mono source index at which to stop (exclusive)",
    )
    parser.add_argument("--extension", default=".png")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="atomically write a structural SHA-256 manifest after export/validation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate existing sequence/mono outputs and optionally write --manifest",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In sequence/mono mode, skip only structurally valid output files. "
            "For stereo sequence resumes, skipped left features are recomputed "
            "in memory so the next missing temporal match remains correct."
        ),
    )
    args = parser.parse_args()

    pair_mode = args.image0 is not None or args.image1 is not None
    sequence_mode = args.left_dir is not None or args.right_dir is not None
    mono_mode = args.mono_dir is not None
    selected = sum([pair_mode, sequence_mode, mono_mode])
    if selected != 1:
        parser.error(
            "use exactly one of: --image0/--image1 pair mode, --left-dir/--right-dir stereo "
            "sequence mode, or --mono-dir single-camera sequence mode"
        )
    if pair_mode and (args.image0 is None or args.image1 is None):
        parser.error("pair mode requires both --image0 and --image1")
    if sequence_mode and (args.left_dir is None or args.right_dir is None):
        parser.error("stereo sequence mode requires both --left-dir and --right-dir")
    if args.frame_stride <= 0:
        parser.error("--frame-stride must be positive")
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if args.start_index is not None and args.start_index < 0:
        parser.error("--start-index must be non-negative")
    if args.end_index is not None and args.end_index < 0:
        parser.error("--end-index must be non-negative")
    if (
        args.start_index is not None
        and args.end_index is not None
        and args.end_index <= args.start_index
    ):
        parser.error("--end-index must be greater than --start-index")
    if (args.start_index is not None or args.end_index is not None) and pair_mode:
        parser.error("--start-index/--end-index are only valid in sequence or mono mode")
    if (
        (args.start_index is not None or args.end_index is not None)
        and args.start_frame != 0
    ):
        parser.error("--start-frame cannot be combined with --start-index/--end-index")
    if args.manifest is not None and pair_mode:
        parser.error("--manifest is only valid in sequence or mono mode")
    if args.validate_only and pair_mode:
        parser.error("--validate-only is only valid in sequence or mono mode")
    if args.validate_only and args.manifest is None:
        parser.error("--validate-only requires --manifest")
    return args


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def squeeze_batch(value: Any) -> Any:
    if hasattr(value, "dim") and value.dim() > 0 and value.shape[0] == 1:
        return value[0]
    return value


def feature_field(features: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in features:
            return squeeze_batch(features[name])
    raise KeyError(f"feature output missing any of: {', '.join(names)}")


def atomic_text_write(path: Path, writer: Callable[[TextIO], None]) -> None:
    """Write a text file through a same-directory temporary and atomic rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def write_features(path: Path, features: dict[str, Any]) -> None:
    keypoints = feature_field(features, "keypoints")
    descriptors = feature_field(features, "descriptors")
    scores = feature_field(features, "keypoint_scores", "scores")

    # LightGlue versions have used both [N, D] and [D, N] descriptors.
    if descriptors.dim() != 2:
        raise ValueError(f"expected 2-D descriptors, got shape {tuple(descriptors.shape)}")
    if descriptors.shape[0] != keypoints.shape[0] and descriptors.shape[1] == keypoints.shape[0]:
        descriptors = descriptors.transpose(0, 1)
    if descriptors.shape[0] != keypoints.shape[0]:
        raise ValueError(
            "descriptor/keypoint count mismatch: "
            f"{tuple(descriptors.shape)} vs {tuple(keypoints.shape)}"
        )

    keypoints = keypoints.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()
    descriptors = descriptors.detach().cpu().tolist()

    def write(output: TextIO) -> None:
        output.write("# X Y SCORE D0 D1 ...\n")
        for xy, score, descriptor in zip(keypoints, scores, descriptors):
            values = [xy[0], xy[1], score, *descriptor]
            output.write(" ".join(f"{float(value):.9g}" for value in values))
            output.write("\n")

    atomic_text_write(path, write)


def write_matches(path: Path, match_output: dict[str, Any]) -> None:
    matches = feature_field(match_output, "matches")
    scores = feature_field(match_output, "scores", "matching_scores")

    matches = matches.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()

    def write(output: TextIO) -> None:
        output.write("# QUERY_IDX TRAIN_IDX CONFIDENCE DISTANCE\n")
        for (query_index, train_index), score in zip(matches, scores):
            confidence = float(score)
            distance = 1.0 - confidence
            output.write(
                f"{int(query_index)} {int(train_index)} "
                f"{confidence:.9g} {distance:.9g}\n"
            )

    atomic_text_write(path, write)


def image_files(directory: Path, extension: str) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.suffix == extension)
    if not files:
        raise FileNotFoundError(f"no {extension} files in {directory}")
    return files


def _indexed_files(files: list[Path], args: argparse.Namespace) -> list[tuple[int, Path]]:
    if args.start_index is not None or args.end_index is not None:
        start = args.start_index if args.start_index is not None else 0
        end = args.end_index if args.end_index is not None else len(files)
        if start >= len(files) or end > len(files):
            raise ValueError(
                f"requested source range [{start}, {end}) outside {len(files)} files"
            )
        selected = [(index, files[index]) for index in range(start, end, args.frame_stride)]
        if args.frames is not None:
            selected = selected[: args.frames]
        return selected

    selected = files[args.start_frame :: args.frame_stride]
    if args.frames is not None:
        selected = selected[: args.frames]
    # Keep the historical local output numbering unless the explicit source
    # range is requested.
    return list(enumerate(selected))


def selected_frame_pairs(
    args: argparse.Namespace,
) -> list[tuple[int, Path, Path]]:
    left_files = image_files(args.left_dir, args.extension)
    right_files = image_files(args.right_dir, args.extension)
    if len(left_files) != len(right_files):
        raise ValueError(
            f"left/right frame count mismatch: {len(left_files)} vs {len(right_files)}"
        )

    pairs = [
        (index, left_path, right_files[index])
        for index, left_path in _indexed_files(left_files, args)
    ]
    if len(pairs) < 2:
        raise ValueError(f"sequence mode needs at least 2 selected frames, got {len(pairs)}")
    return pairs


def frame_mono_features_name(frame_index: int) -> str:
    return f"frame_{frame_index:06}_features.txt"


def frame_left_features_name(frame_index: int) -> str:
    return f"frame_{frame_index:06}_left_features.txt"


def frame_right_features_name(frame_index: int) -> str:
    return f"frame_{frame_index:06}_right_features.txt"


def frame_stereo_matches_name(frame_index: int) -> str:
    return f"frame_{frame_index:06}_stereo_matches.txt"


def frame_temporal_matches_name(frame_index: int) -> str:
    return f"frame_{frame_index:06}_temporal_matches.txt"


def _feature_file_metadata(path: Path) -> tuple[int, int] | None:
    """Return (row count, descriptor dimension) for a complete feature file."""

    rows = 0
    descriptor_dimension: int | None = None
    saw_header = False
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    saw_header = True
                    continue
                fields = stripped.split()
                if len(fields) < 4:
                    return None
                try:
                    values = [float(value) for value in fields]
                except ValueError:
                    return None
                if not all(math.isfinite(value) for value in values):
                    return None
                dimension = len(fields) - 3
                if descriptor_dimension is None:
                    descriptor_dimension = dimension
                elif dimension != descriptor_dimension:
                    return None
                rows += 1
    except (OSError, UnicodeError):
        return None
    if not saw_header or rows == 0 or descriptor_dimension is None:
        return None
    return rows, descriptor_dimension


def _match_file_rows(
    path: Path,
    query_rows: int | None = None,
    train_rows: int | None = None,
) -> int | None:
    rows = 0
    saw_header = False
    try:
        with path.open("r", encoding="utf-8") as input_file:
            for line in input_file:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith("#"):
                    saw_header = True
                    continue
                fields = stripped.split()
                if len(fields) < 3:
                    return None
                try:
                    query_index = int(fields[0])
                    train_index = int(fields[1])
                    numeric = [float(value) for value in fields[2:]]
                except ValueError:
                    return None
                if query_index < 0 or train_index < 0:
                    return None
                if query_rows is not None and query_index >= query_rows:
                    return None
                if train_rows is not None and train_index >= train_rows:
                    return None
                if not numeric or not all(math.isfinite(value) for value in numeric):
                    return None
                rows += 1
    except (OSError, UnicodeError):
        return None
    if not saw_header:
        return None
    return rows


def validate_sequence_frame_outputs(
    out_dir: Path,
    frame_index: int,
    previous_frame_index: int | None = None,
) -> dict[str, tuple[int, int] | int] | None:
    left = _feature_file_metadata(out_dir / frame_left_features_name(frame_index))
    right = _feature_file_metadata(out_dir / frame_right_features_name(frame_index))
    if left is None or right is None or left[1] != right[1]:
        return None
    stereo_rows = _match_file_rows(
        out_dir / frame_stereo_matches_name(frame_index), left[0], right[0]
    )
    if stereo_rows is None:
        return None
    if previous_frame_index is None:
        return {
            "left_features": left,
            "right_features": right,
            "stereo_matches": stereo_rows,
        }
    previous = _feature_file_metadata(
        out_dir / frame_left_features_name(previous_frame_index)
    )
    if previous is not None and previous[1] != left[1]:
        return None
    temporal_rows = _match_file_rows(
        out_dir / frame_temporal_matches_name(frame_index),
        previous[0] if previous is not None else None,
        left[0],
    )
    if temporal_rows is None:
        return None
    return {
        "left_features": left,
        "right_features": right,
        "stereo_matches": stereo_rows,
        "temporal_matches": temporal_rows,
    }


def sequence_frame_outputs_exist(
    out_dir: Path,
    frame_index: int,
    previous_frame_index: int | None = None,
) -> bool:
    return (
        validate_sequence_frame_outputs(out_dir, frame_index, previous_frame_index)
        is not None
    )


def _validate_mono_output(out_dir: Path, frame_index: int) -> tuple[int, int] | None:
    return _feature_file_metadata(out_dir / frame_mono_features_name(frame_index))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_file(
    path: Path,
    rows: int | None = None,
    descriptor_dimension: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }
    if rows is not None:
        result["rows"] = rows
    if descriptor_dimension is not None:
        result["descriptor_dimension"] = descriptor_dimension
    return result


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    output = dict(manifest)
    output["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()

    def write(stream: TextIO) -> None:
        json.dump(output, stream, indent=2, sort_keys=True)
        stream.write("\n")

    atomic_text_write(path, write)


def build_sequence_manifest(
    out_dir: Path,
    entries: list[tuple[int, Path, Path]],
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    step = entries[1][0] - entries[0][0] if len(entries) > 1 else 1
    if step <= 0:
        raise ValueError("sequence manifest entries must be strictly increasing")
    for position, (frame_index, left_path, right_path) in enumerate(entries):
        # A range manifest still records its first boundary temporal match;
        # the predecessor feature file may belong to another worker and is
        # therefore optional for index-bound validation here.
        previous_index = (
            entries[position - 1][0]
            if position > 0
            else (frame_index - step if frame_index >= step else None)
        )
        metadata = validate_sequence_frame_outputs(out_dir, frame_index, previous_index)
        if metadata is None:
            raise ValueError(f"invalid or incomplete outputs for frame {frame_index:06}")
        files: dict[str, Any] = {
            "left_features": _manifest_file(
                out_dir / frame_left_features_name(frame_index),
                metadata["left_features"][0],
                metadata["left_features"][1],
            ),
            "right_features": _manifest_file(
                out_dir / frame_right_features_name(frame_index),
                metadata["right_features"][0],
                metadata["right_features"][1],
            ),
            "stereo_matches": _manifest_file(
                out_dir / frame_stereo_matches_name(frame_index),
                metadata["stereo_matches"],
            ),
        }
        if previous_index is not None:
            files["temporal_matches"] = _manifest_file(
                out_dir / frame_temporal_matches_name(frame_index),
                metadata["temporal_matches"],
            )
        frames.append(
            {
                "index": frame_index,
                "left_image": left_path.name,
                "right_image": right_path.name,
                "files": files,
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "stereo_sequence",
        "frame_count": len(frames),
        "frames": frames,
    }
    if frames:
        manifest["range"] = {
            "start_index": frames[0]["index"],
            "end_index_exclusive": frames[-1]["index"] + 1,
        }
    return manifest


def build_mono_manifest(
    out_dir: Path,
    entries: list[tuple[int, Path]],
) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for frame_index, image_path in entries:
        metadata = _validate_mono_output(out_dir, frame_index)
        if metadata is None:
            raise ValueError(f"invalid or incomplete output for frame {frame_index:06}")
        frames.append(
            {
                "index": frame_index,
                "image": image_path.name,
                "file": _manifest_file(
                    out_dir / frame_mono_features_name(frame_index),
                    metadata[0],
                    metadata[1],
                ),
            }
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "mono_sequence",
        "frame_count": len(frames),
        "frames": frames,
    }
    if frames:
        manifest["range"] = {
            "start_index": frames[0]["index"],
            "end_index_exclusive": frames[-1]["index"] + 1,
        }
    return manifest


def _pair_outputs_exist(out_dir: Path) -> bool:
    left = _feature_file_metadata(out_dir / "image0_features.txt")
    right = _feature_file_metadata(out_dir / "image1_features.txt")
    return (
        left is not None
        and right is not None
        and _match_file_rows(out_dir / "matches.txt", left[0], right[0]) is not None
    )


def export_pair(args: argparse.Namespace, extractor: Any, matcher: Any, load_image: Any, rbd: Any) -> None:
    if args.skip_existing and _pair_outputs_exist(args.out_dir):
        print("pair outputs already exist; skipping", flush=True)
        return
    image0 = load_image(args.image0).to(args.resolved_device)
    image1 = load_image(args.image1).to(args.resolved_device)

    features0 = extractor.extract(image0)
    features1 = extractor.extract(image1)
    match_output = matcher({"image0": features0, "image1": features1})

    write_features(args.out_dir / "image0_features.txt", rbd(features0))
    write_features(args.out_dir / "image1_features.txt", rbd(features1))
    write_matches(args.out_dir / "matches.txt", rbd(match_output))


def export_sequence(
    args: argparse.Namespace,
    extractor: Any,
    matcher: Any,
    load_image: Any,
    rbd: Any,
) -> None:
    pairs = selected_frame_pairs(args)
    previous_left_features: dict[str, Any] | None = None
    previous_left_path: Path | None = None
    previous_frame_index: int | None = None

    # An explicit source-index range is intended for disjoint workers.  Seed
    # the first range with its predecessor so its boundary temporal match is
    # generated exactly once, while retaining the old local numbering path.
    if args.start_index is not None or args.end_index is not None:
        first_frame_index = pairs[0][0]
        predecessor_index = first_frame_index - args.frame_stride
        if predecessor_index >= 0:
            left_files = image_files(args.left_dir, args.extension)
            previous_left_path = left_files[predecessor_index]
            previous_frame_index = predecessor_index

    for position, (frame_index, left_path, right_path) in enumerate(pairs):
        frame_previous_index = previous_frame_index
        if position > 0:
            frame_previous_index = pairs[position - 1][0]
        if args.skip_existing and sequence_frame_outputs_exist(
            args.out_dir, frame_index, frame_previous_index
        ):
            previous_left_features = None
            previous_left_path = left_path
            previous_frame_index = frame_index
            if frame_index % 25 == 0:
                print(
                    f"frame {frame_index:06}: left={left_path.name} right={right_path.name} skipped",
                    flush=True,
                )
            continue

        left_image = load_image(left_path).to(args.resolved_device)
        right_image = load_image(right_path).to(args.resolved_device)
        left_features = extractor.extract(left_image)
        right_features = extractor.extract(right_image)

        stereo_matches = matcher({"image0": left_features, "image1": right_features})
        write_features(args.out_dir / frame_left_features_name(frame_index), rbd(left_features))
        write_features(args.out_dir / frame_right_features_name(frame_index), rbd(right_features))
        write_matches(args.out_dir / frame_stereo_matches_name(frame_index), rbd(stereo_matches))

        if previous_left_features is None and previous_left_path is not None:
            previous_left_image = load_image(previous_left_path).to(args.resolved_device)
            previous_left_features = extractor.extract(previous_left_image)

        if previous_left_features is not None:
            temporal_matches = matcher(
                {"image0": previous_left_features, "image1": left_features}
            )
            write_matches(
                args.out_dir / frame_temporal_matches_name(frame_index),
                rbd(temporal_matches),
            )

        previous_left_features = left_features
        previous_left_path = left_path
        previous_frame_index = frame_index
        print(
            f"frame {frame_index:06}: left={left_path.name} right={right_path.name}",
            flush=True,
        )


def selected_mono_frames(args: argparse.Namespace) -> list[tuple[int, Path]]:
    files = image_files(args.mono_dir, args.extension)
    selected = _indexed_files(files, args)
    if len(selected) < 1:
        raise ValueError(f"mono mode needs at least 1 selected frame, got {len(selected)}")
    return selected


def export_mono(
    args: argparse.Namespace,
    extractor: Any,
    load_image: Any,
    rbd: Any,
) -> None:
    frames = selected_mono_frames(args)
    for frame_index, frame_path in frames:
        output = args.out_dir / frame_mono_features_name(frame_index)
        if args.skip_existing and _validate_mono_output(args.out_dir, frame_index) is not None:
            if frame_index % 25 == 0:
                print(f"frame {frame_index:06}: {frame_path.name} skipped", flush=True)
            continue
        image = load_image(frame_path).to(args.resolved_device)
        features = extractor.extract(image)
        write_features(output, rbd(features))
        if frame_index % 25 == 0:
            print(f"frame {frame_index:06}: {frame_path.name}", flush=True)


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.validate_only:
        if args.mono_dir is not None:
            _write_manifest(
                args.manifest,
                build_mono_manifest(args.out_dir, selected_mono_frames(args)),
            )
        else:
            _write_manifest(
                args.manifest,
                build_sequence_manifest(args.out_dir, selected_frame_pairs(args)),
            )
        print(f"validated {args.out_dir}; wrote {args.manifest}")
        return 0

    try:
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd
    except ImportError as error:
        print(
            "missing optional LightGlue Python stack; install lightglue/torch first",
            file=sys.stderr,
        )
        print(f"import error: {error}", file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    args.resolved_device = device

    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)

    if args.mono_dir is not None:
        # Matcher unused in mono mode; skip its (non-trivial) cold start.
        export_mono(args, extractor, load_image, rbd)
    else:
        matcher = LightGlue(features="superpoint").eval().to(device)
        if args.image0 is not None:
            export_pair(args, extractor, matcher, load_image, rbd)
        else:
            export_sequence(args, extractor, matcher, load_image, rbd)

    if args.manifest is not None:
        if args.mono_dir is not None:
            manifest = build_mono_manifest(args.out_dir, selected_mono_frames(args))
        else:
            manifest = build_sequence_manifest(args.out_dir, selected_frame_pairs(args))
        _write_manifest(args.manifest, manifest)

    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
