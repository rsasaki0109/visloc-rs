#!/usr/bin/env python3
"""Generate the README's measured ETH3D 10-scene scale visual.

Every trajectory in the PNG/GIF is read from a supplied COLMAP ``images.txt``
model.  Scenes are centred, PCA-projected, and independently normalized only
for the small-multiple layout; no synthetic camera or geometry is introduced.
The evidence JSON supplies the displayed registration, accuracy, and memory
measurements.

Example::

    python3 scripts/generate_eth3d_scale_readme_visuals.py \
      --evidence-json benchmarks/electro/m5-eth3d-scale-validation.json \
      --output-dir docs/assets \
      --model terrains=/path/to/terrains/images.txt \
      --model electro=/path/to/electro/images.txt
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


SCENE_ORDER = [
    "terrains",
    "delivery_area",
    "forest",
    "playground",
    "electro",
    "lakeside",
    "sand_box",
    "storage_room",
    "storage_room_2",
    "tunnel",
]


def parse_model(value: str) -> tuple[str, Path]:
    scene, separator, path = value.partition("=")
    if not separator or not scene or not path:
        raise argparse.ArgumentTypeError("--model must be SCENE=/path/to/images.txt")
    return scene, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=parse_model, action="append", default=[])
    parser.add_argument("--prefix", default="eth3d_10008_scale_validation")
    parser.add_argument("--dpi", type=int, default=130)
    parser.add_argument("--fps", type=int, default=3)
    return parser.parse_args()


def rotation(qw: float, qx: float, qy: float, qz: float):
    import numpy as np

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("zero or non-finite COLMAP quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        [
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ],
        dtype=float,
    )


def read_centres(path: Path):
    import numpy as np

    centres = []
    expect_points = False
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if expect_points:
            if line.startswith("#"):
                continue
            expect_points = False
            continue
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"malformed COLMAP pose row {path}:{line_number}")
        try:
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = np.asarray([float(value) for value in fields[5:8]])
        except ValueError as exc:
            raise ValueError(f"non-numeric COLMAP pose row {path}:{line_number}") from exc
        centre = -rotation(qw, qx, qy, qz).T @ translation
        if not np.all(np.isfinite(centre)):
            raise ValueError(f"non-finite camera centre {path}:{line_number}")
        centres.append(centre)
        expect_points = True
    if len(centres) < 2:
        raise ValueError(f"model needs at least two registered cameras: {path}")
    return np.asarray(centres)


def project(centres):
    """PCA-project and normalize a real trajectory for a small-multiple panel."""

    import numpy as np

    centred = centres - centres.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    projected = centred @ vt[:2].T
    extent = np.ptp(projected, axis=0)
    scale = float(max(extent.max(), 1.0e-12))
    return projected / scale


def load_evidence(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    scenes = data.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("evidence JSON must contain a scenes list")
    by_name = {scene["scene"]: scene for scene in scenes}
    missing = [scene for scene in SCENE_ORDER if scene not in by_name]
    if missing:
        raise ValueError(f"evidence JSON is missing scenes: {missing}")
    return data


def main() -> int:
    args = parse_args()
    if args.dpi <= 0 or args.fps <= 0:
        print("dpi and fps must be positive", file=sys.stderr)
        return 2
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import numpy as np
    except ImportError as exc:
        print(f"missing plotting dependency: {exc}", file=sys.stderr)
        return 2

    try:
        evidence = load_evidence(args.evidence_json)
        model_paths = dict(args.model)
        if set(model_paths) != set(SCENE_ORDER):
            missing = sorted(set(SCENE_ORDER) - set(model_paths))
            extra = sorted(set(model_paths) - set(SCENE_ORDER))
            raise ValueError(f"exactly ten --model entries required; missing={missing}, extra={extra}")
        trajectories = {scene: project(read_centres(model_paths[scene])) for scene in SCENE_ORDER}
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    scene_evidence = {scene["scene"]: scene for scene in evidence["scenes"]}
    total_images = sum(int(scene_evidence[name]["images"]) for name in SCENE_ORDER)
    total_registered = sum(int(scene_evidence[name]["registered"]) for name in SCENE_ORDER)
    max_rss = max(float(scene_evidence[name]["peak_rss_kib"]) for name in SCENE_ORDER)
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, len(SCENE_ORDER)))

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold"})
    fig, axes = plt.subplots(2, 5, figsize=(12.8, 7.2), facecolor="#07111f")
    fig.subplots_adjust(left=0.035, right=0.985, top=0.79, bottom=0.08, wspace=0.12, hspace=0.34)
    title = fig.text(
        0.5,
        0.955,
        "10,008 real ETH3D images · 10 independent reconstructions",
        color="white",
        fontsize=20,
        fontweight="bold",
        ha="center",
    )
    subtitle = fig.text(
        0.5,
        0.885,
        f"{total_registered:,}/{total_images:,} cameras registered  ·  max mapper RSS {max_rss / 1048576:.2f} GiB  ·  GT score-only",
        color="#8de7ff",
        fontsize=12,
        ha="center",
    )
    artists = []
    for index, (axis, scene) in enumerate(zip(axes.flat, SCENE_ORDER)):
        axis.set_facecolor("#0b1b2e")
        axis.set_aspect("equal", adjustable="box")
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_color("#21405e")
        points = trajectories[scene]
        line, = axis.plot([], [], color=colors[index], linewidth=1.1, alpha=0.68)
        dots = axis.scatter([], [], s=4, color=colors[index], alpha=0.9, edgecolors="none")
        axis.set_xlim(points[:, 0].min() - 0.08, points[:, 0].max() + 0.08)
        axis.set_ylim(points[:, 1].min() - 0.08, points[:, 1].max() + 0.08)
        metric = scene_evidence[scene]
        rmse = float(metric["rmse_m"]) * 100.0
        axis.set_title(
            f"{scene.replace('_', ' ')}\n{metric['registered']}/{metric['images']} · {rmse:.2f} cm RMSE",
            color="white",
            fontsize=9,
            pad=7,
        )
        artists.append((line, dots, points))

    footer = fig.text(
        0.5,
        0.025,
        "Each panel is the measured model's camera centres · independently PCA-projected for display · unrelated scenes are never joined",
        color="#91a8bf",
        fontsize=8.5,
        ha="center",
    )

    def update(frame: int):
        for index, (line, dots, points) in enumerate(artists):
            if index > frame:
                visible = points[:0]
            elif index < frame:
                visible = points
            else:
                fraction = min(1.0, (frame - index + 1.0))
                visible = points[: max(2, int(len(points) * fraction))]
            line.set_data(visible[:, 0], visible[:, 1])
            dots.set_offsets(visible)
        return [title, subtitle, footer, *(artist for pair in artists for artist in pair[:2])]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.prefix}.png"
    gif_path = args.output_dir / f"{args.prefix}.gif"
    update(len(SCENE_ORDER) - 1)
    fig.savefig(png_path, dpi=args.dpi, facecolor=fig.get_facecolor())
    animation = FuncAnimation(fig, update, frames=len(SCENE_ORDER), interval=1000 / args.fps, blit=False)
    animation.save(gif_path, writer=PillowWriter(fps=args.fps), dpi=max(72, args.dpi // 2))
    plt.close(fig)
    print(f"wrote {png_path}")
    print(f"wrote {gif_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
