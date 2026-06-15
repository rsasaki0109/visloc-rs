#!/usr/bin/env python3
"""Export a TUM RGB-D sequence as *virtual-stereo* deep features for visloc-rs.

The visloc-rs stereo VO backend (`examples/stereo_vo_external_deep_files.rs`)
consumes precomputed SuperPoint/LightGlue feature + match files and only ever
uses the *right* image to triangulate metric 3D points; everything downstream
(PnP / Kabsch / online BA / loop closure) runs on the resulting `point_cam`.

TUM RGB-D gives a single RGB image plus a registered depth map (16-bit PNG,
`depth_m = pixel / 5000`). We turn each depth-valid SuperPoint keypoint into a
*synthetic right* keypoint shifted by the stereo disparity it would have at a
chosen virtual baseline:

    disparity = baseline * fx / depth_m        (pixels)
    u_right   = u_left - disparity,  v_right = v_left

with the SAME descriptor, and emit a 1:1 stereo match. The Rust triangulator
inverts this exactly (`depth = baseline * fx / disparity`), so as long as the
SAME `--baseline` is passed to the Rust binary the recovered depth — and hence
the metric scale — is the true TUM depth, independent of the (arbitrary)
baseline. This reuses the entire stereo VO/BA/loop backend with zero Rust
changes.

Outputs (consumed by `stereo_vo_external_deep_files --features-dir`):

    frame_000000_left_features.txt     # X Y SCORE D0 D1 ...  (all keypoints)
    frame_000000_right_features.txt    # synthetic right, depth-valid subset
    frame_000000_stereo_matches.txt    # QUERY_IDX TRAIN_IDX CONF DIST (1:1)
    frame_000001_temporal_matches.txt  # previous-left -> current-left (LightGlue)
    ...
    frame_timestamps.txt               # "<frame_idx> <timestamp_ns>" for kitti_poses_to_tum.py

The original `groundtruth.txt` is already TUM format and is associated against
the per-frame timestamps by evo_ape at evaluation time, so it is not rewritten
here.

Freiburg1 intrinsics are the defaults (fx 517.3 fy 516.5 cx 318.6 cy 255.3);
pass --fx/--fy/--cx/--cy for fr2 (520.9/521.0/325.1/249.7) or fr3
(535.4/539.2/320.1/247.6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seq-dir", type=Path, required=True,
                        help="TUM sequence dir containing rgb.txt, depth.txt, rgb/, depth/")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    # Freiburg1 intrinsics by default.
    parser.add_argument("--fx", type=float, default=517.3)
    parser.add_argument("--fy", type=float, default=516.5)
    parser.add_argument("--cx", type=float, default=318.6)
    parser.add_argument("--cy", type=float, default=255.3)
    parser.add_argument("--depth-scale", type=float, default=5000.0,
                        help="depth_meters = pixel / depth_scale (TUM = 5000)")
    parser.add_argument("--baseline", type=float, default=0.1,
                        help="virtual stereo baseline (m); pass the SAME value to the Rust binary")
    parser.add_argument("--min-depth", type=float, default=0.3)
    parser.add_argument("--max-depth", type=float, default=8.0)
    parser.add_argument("--assoc-max-diff", type=float, default=0.02,
                        help="max rgb<->depth timestamp difference (s)")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def read_tum_index(path: Path) -> list[tuple[float, str]]:
    """Read a TUM 'timestamp filename' index file (rgb.txt / depth.txt)."""
    entries: list[tuple[float, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            entries.append((float(parts[0]), parts[1]))
    return entries


def associate(rgb: list[tuple[float, str]], depth: list[tuple[float, str]],
              max_diff: float) -> list[tuple[float, str, str]]:
    """Nearest-timestamp rgb<->depth association (TUM associate.py semantics)."""
    depth_ts = np.array([d[0] for d in depth])
    pairs: list[tuple[float, str, str]] = []
    for ts, rgb_name in rgb:
        j = int(np.argmin(np.abs(depth_ts - ts)))
        if abs(depth_ts[j] - ts) <= max_diff:
            pairs.append((ts, rgb_name, depth[j][1]))
    return pairs


def squeeze_batch(value: Any) -> Any:
    if hasattr(value, "dim") and value.dim() > 0 and value.shape[0] == 1:
        return value[0]
    return value


def feature_field(features: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in features:
            return squeeze_batch(features[name])
    raise KeyError(f"feature output missing any of: {', '.join(names)}")


def features_to_arrays(features: dict[str, Any]):
    keypoints = feature_field(features, "keypoints")
    descriptors = feature_field(features, "descriptors")
    scores = feature_field(features, "keypoint_scores", "scores")
    if descriptors.dim() != 2:
        raise ValueError(f"expected 2-D descriptors, got {tuple(descriptors.shape)}")
    if descriptors.shape[0] != keypoints.shape[0] and descriptors.shape[1] == keypoints.shape[0]:
        descriptors = descriptors.transpose(0, 1)
    kp = keypoints.detach().cpu().numpy()
    sc = scores.detach().cpu().numpy().reshape(-1)
    de = descriptors.detach().cpu().numpy()
    return kp, sc, de


def write_feature_rows(path: Path, kp: np.ndarray, sc: np.ndarray, de: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# X Y SCORE D0 D1 ...\n")
        for (x, y), score, descriptor in zip(kp, sc, de):
            values = [float(x), float(y), float(score), *(float(v) for v in descriptor)]
            f.write(" ".join(f"{value:.9g}" for value in values))
            f.write("\n")


def write_matches(path: Path, matches: np.ndarray, scores: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# QUERY_IDX TRAIN_IDX CONFIDENCE DISTANCE\n")
        for (q, t), score in zip(matches, scores):
            conf = float(score)
            f.write(f"{int(q)} {int(t)} {conf:.9g} {1.0 - conf:.9g}\n")


def main() -> int:
    args = parse_args()
    try:
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import load_image, rbd
    except ImportError as error:
        print("missing optional LightGlue/torch stack; install lightglue first", file=sys.stderr)
        print(f"import error: {error}", file=sys.stderr)
        return 2
    from PIL import Image

    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rgb = read_tum_index(args.seq_dir / "rgb.txt")
    depth = read_tum_index(args.seq_dir / "depth.txt")
    pairs = associate(rgb, depth, args.assoc_max_diff)
    pairs = pairs[args.start_frame::args.frame_stride]
    if args.frames is not None:
        pairs = pairs[: args.frames]
    if len(pairs) < 2:
        print(f"need >=2 associated frames, got {len(pairs)}", file=sys.stderr)
        return 1

    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    fx, fy, cx, cy = args.fx, args.fy, args.cx, args.cy
    ts_lines: list[str] = []
    prev_left_features: dict[str, Any] | None = None
    kept_stereo_total = 0

    with torch.no_grad():
        for frame_index, (ts, rgb_name, depth_name) in enumerate(pairs):
            image = load_image(args.seq_dir / rgb_name).to(device)
            left_features = extractor.extract(image)
            kp, sc, de = features_to_arrays(rbd(left_features))

            depth_png = np.asarray(Image.open(args.seq_dir / depth_name)).astype(np.float64)
            h, w = depth_png.shape[:2]

            # Synthetic-right keypoints for depth-valid left keypoints.
            right_kp: list[tuple[float, float]] = []
            right_sc: list[float] = []
            right_de: list[np.ndarray] = []
            stereo_q: list[int] = []
            stereo_t: list[int] = []
            for i, ((x, y), score, descriptor) in enumerate(zip(kp, sc, de)):
                u = int(round(float(x)))
                v = int(round(float(y)))
                if u < 0 or u >= w or v < 0 or v >= h:
                    continue
                d = depth_png[v, u] / args.depth_scale
                if d < args.min_depth or d > args.max_depth:
                    continue
                disparity = args.baseline * fx / d
                right_kp.append((float(x) - disparity, float(y)))
                right_sc.append(float(score))
                right_de.append(descriptor)
                stereo_q.append(i)
                stereo_t.append(len(right_kp) - 1)
            kept_stereo_total += len(right_kp)

            write_feature_rows(
                args.out_dir / f"frame_{frame_index:06}_left_features.txt", kp, sc, de
            )
            write_feature_rows(
                args.out_dir / f"frame_{frame_index:06}_right_features.txt",
                np.array(right_kp, dtype=np.float64).reshape(-1, 2),
                np.array(right_sc, dtype=np.float64),
                np.array(right_de, dtype=np.float64).reshape(len(right_de), -1)
                if right_de else np.zeros((0, de.shape[1])),
            )
            stereo_matches = np.array(list(zip(stereo_q, stereo_t)), dtype=np.int64).reshape(-1, 2)
            write_matches(
                args.out_dir / f"frame_{frame_index:06}_stereo_matches.txt",
                stereo_matches,
                np.ones(len(stereo_q), dtype=np.float64),
            )

            if prev_left_features is not None:
                temporal = rbd(matcher({"image0": prev_left_features, "image1": left_features}))
                tm = feature_field(temporal, "matches").detach().cpu().numpy().reshape(-1, 2)
                tsc = feature_field(temporal, "scores", "matching_scores").detach().cpu().numpy().reshape(-1)
                write_matches(
                    args.out_dir / f"frame_{frame_index:06}_temporal_matches.txt", tm, tsc
                )

            prev_left_features = left_features
            ts_lines.append(f"{frame_index} {int(round(ts * 1e9))}")
            if frame_index % 25 == 0:
                print(
                    f"frame {frame_index:06}: rgb={rgb_name} kp={len(kp)} stereo={len(right_kp)}",
                    flush=True,
                )

    (args.out_dir / "frame_timestamps.txt").write_text("\n".join(ts_lines) + "\n")
    print(
        f"wrote {len(pairs)} frames to {args.out_dir} "
        f"(avg {kept_stereo_total / len(pairs):.0f} stereo pts/frame, baseline={args.baseline} m)"
    )
    print("NOTE: pass the SAME --baseline and intrinsics to stereo_vo_external_deep_files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
