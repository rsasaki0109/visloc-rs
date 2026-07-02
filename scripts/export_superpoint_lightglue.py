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
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any


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
    parser.add_argument("--extension", default=".png")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "In sequence/mono mode, skip output files that already exist. "
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

    with path.open("w", encoding="utf-8") as f:
        f.write("# X Y SCORE D0 D1 ...\n")
        for xy, score, descriptor in zip(keypoints, scores, descriptors):
            values = [xy[0], xy[1], score, *descriptor]
            f.write(" ".join(f"{float(value):.9g}" for value in values))
            f.write("\n")


def write_matches(path: Path, match_output: dict[str, Any]) -> None:
    matches = feature_field(match_output, "matches")
    scores = feature_field(match_output, "scores", "matching_scores")

    matches = matches.detach().cpu().tolist()
    scores = scores.detach().cpu().tolist()

    with path.open("w", encoding="utf-8") as f:
        f.write("# QUERY_IDX TRAIN_IDX CONFIDENCE DISTANCE\n")
        for (query_index, train_index), score in zip(matches, scores):
            confidence = float(score)
            distance = 1.0 - confidence
            f.write(
                f"{int(query_index)} {int(train_index)} "
                f"{confidence:.9g} {distance:.9g}\n"
            )


def image_files(directory: Path, extension: str) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.suffix == extension)
    if not files:
        raise FileNotFoundError(f"no {extension} files in {directory}")
    return files


def selected_frame_pairs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    left_files = image_files(args.left_dir, args.extension)
    right_files = image_files(args.right_dir, args.extension)
    if len(left_files) != len(right_files):
        raise ValueError(
            f"left/right frame count mismatch: {len(left_files)} vs {len(right_files)}"
        )

    pairs = list(zip(left_files, right_files))
    pairs = pairs[args.start_frame::args.frame_stride]
    if args.frames is not None:
        pairs = pairs[:args.frames]
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


def sequence_frame_outputs_exist(out_dir: Path, frame_index: int) -> bool:
    required = [
        out_dir / frame_left_features_name(frame_index),
        out_dir / frame_right_features_name(frame_index),
        out_dir / frame_stereo_matches_name(frame_index),
    ]
    if frame_index > 0:
        required.append(out_dir / frame_temporal_matches_name(frame_index))
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def export_pair(args: argparse.Namespace, extractor: Any, matcher: Any, load_image: Any, rbd: Any) -> None:
    if args.skip_existing and all(
        (args.out_dir / name).exists() and (args.out_dir / name).stat().st_size > 0
        for name in ("image0_features.txt", "image1_features.txt", "matches.txt")
    ):
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

    for frame_index, (left_path, right_path) in enumerate(pairs):
        if args.skip_existing and sequence_frame_outputs_exist(args.out_dir, frame_index):
            previous_left_features = None
            previous_left_path = left_path
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
        print(
            f"frame {frame_index:06}: left={left_path.name} right={right_path.name}",
            flush=True,
        )


def selected_mono_frames(args: argparse.Namespace) -> list[Path]:
    files = image_files(args.mono_dir, args.extension)
    files = files[args.start_frame::args.frame_stride]
    if args.frames is not None:
        files = files[: args.frames]
    if len(files) < 1:
        raise ValueError(f"mono mode needs at least 1 selected frame, got {len(files)}")
    return files


def export_mono(
    args: argparse.Namespace,
    extractor: Any,
    load_image: Any,
    rbd: Any,
) -> None:
    frames = selected_mono_frames(args)
    for frame_index, frame_path in enumerate(frames):
        output = args.out_dir / frame_mono_features_name(frame_index)
        if args.skip_existing and output.exists() and output.stat().st_size > 0:
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
