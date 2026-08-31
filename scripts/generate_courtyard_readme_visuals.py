#!/usr/bin/env python3
"""Generate the measured courtyard SfM visuals used by the README.

The inputs are COLMAP text models.  Camera centres from both reconstructions
are independently Sim(3)-aligned to the same supplied calibration model, so
the plotted trajectories share a frame and scale.  No image rendering or
synthetic geometry is used: the sparse cloud and residuals come directly from
the supplied model files.

Usage::

    python3 scripts/generate_courtyard_readme_visuals.py \
        --visloc-model /path/to/visloc_model \
        --colmap-model /path/to/colmap_model \
        --reference-model /path/to/calibration \
        --output-dir docs/assets

Optional plotting dependencies are numpy, matplotlib, and Pillow.  This is an
asset-generation helper, not part of the core build or test path.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path


STEM_RE = re.compile(r"(\d+)$")
RIG_DIR_RE = re.compile(r"images_rig_(cam\d+)_undistorted$")


def stem_key(stem: str) -> tuple[int, object]:
    match = STEM_RE.search(stem)
    return (0, int(match.group(1))) if match else (1, stem)


def canonical_image_name(raw_name: str) -> str:
    """Normalize ETH3D rig directories to the benchmark's flat names."""
    path = Path(raw_name)
    match = RIG_DIR_RE.search(path.parent.as_posix())
    if match:
        return f"{match.group(1)}_{path.stem}"
    return path.stem


def plot_label(name: str, index: int) -> str:
    """Keep dense rig plots legible while retaining short legacy labels."""
    camera = name.split("_", 1)[0]
    if re.fullmatch(r"cam\d+", camera):
        return f"{camera}·{index}"
    return name[-4:]


def read_images(path: Path) -> dict[str, tuple[object, object]]:
    """Read ``stem -> (camera centre, image id)`` from COLMAP images.txt."""
    import numpy as np

    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, tuple[object, object]] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line or line.startswith("#"):
            index += 1
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"invalid COLMAP image row {path}:{index + 1}")
        try:
            image_id = int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = np.asarray([float(value) for value in fields[5:8]], dtype=float)
        except ValueError as exc:
            raise ValueError(f"invalid COLMAP pose row {path}:{index + 1}") from exc
        quaternion = np.asarray([qw, qx, qy, qz], dtype=float)
        norm = float(np.linalg.norm(quaternion))
        if not math.isfinite(norm) or norm == 0.0 or not np.all(np.isfinite(translation)):
            raise ValueError(f"non-finite COLMAP pose row {path}:{index + 1}")
        qw, qx, qy, qz = quaternion / norm
        rotation = np.asarray(
            [
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
            ],
            dtype=float,
        )
        # Keep camera identity so synchronized rig frames with identical
        # timestamp stems remain distinct across calibration/model layouts.
        name = canonical_image_name(fields[9])
        if name in result:
            raise ValueError(f"duplicate image stem {name!r} in {path}")
        result[name] = (-rotation.T @ translation, image_id)

        # COLMAP stores a second, potentially empty, POINTS2D line for every
        # pose.  Skip comments/blanks defensively before consuming it.
        index += 1
        while index < len(lines) and (not lines[index].strip() or lines[index].lstrip().startswith("#")):
            index += 1
        if index < len(lines):
            index += 1
    return result


def read_points(path: Path):
    """Return finite point coordinates and the mean stored point error."""
    import numpy as np

    points = []
    errors = []
    observations = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"invalid COLMAP point row {path}:{line_number}")
        try:
            point = [float(value) for value in fields[1:4]]
            error = float(fields[7])
        except ValueError as exc:
            raise ValueError(f"invalid COLMAP point row {path}:{line_number}") from exc
        if all(math.isfinite(value) for value in point):
            points.append(point)
        if math.isfinite(error):
            errors.append(error)
        # TRACK[] is a sequence of IMAGE_ID POINT2D_IDX pairs.
        observations += max(0, (len(fields) - 8) // 2)
    return np.asarray(points, dtype=float), float(np.mean(errors)) if errors else float("nan"), observations


def umeyama(source, destination):
    """Return aligned source, scale, rotation, and translation."""
    import numpy as np

    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.sum(source_centered * source_centered) / len(source))
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    translation = destination_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation, translation


def load_model(model_dir: Path, reference_names: list[str]):
    import numpy as np

    images = read_images(model_dir / "images.txt")
    points, mean_error, observations = read_points(model_dir / "points3D.txt")
    missing = sorted(set(reference_names) - set(images), key=stem_key)
    if missing:
        raise ValueError(f"{model_dir} is missing reference image stems: {missing}")
    centres = np.asarray([images[name][0] for name in reference_names], dtype=float)
    return centres, points, mean_error, observations, len(images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visloc-model", type=Path, required=True, help="visloc COLMAP text model directory")
    parser.add_argument("--colmap-model", type=Path, required=True, help="official COLMAP text model directory")
    parser.add_argument("--reference-model", type=Path, required=True, help="shared calibration model directory")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="courtyard_sfm_comparison")
    parser.add_argument("--scene-title", default="Courtyard")
    parser.add_argument(
        "--summary-label",
        default="Official high-resolution SIFT / 703 exhaustive pairs",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="align and plot the image stems shared by both reconstructions",
    )
    parser.add_argument("--visloc-mapper-seconds", type=float)
    parser.add_argument("--colmap-mapper-seconds", type=float)
    parser.add_argument("--visloc-peak-rss-kib", type=float)
    parser.add_argument("--colmap-peak-rss-kib", type=float)
    parser.add_argument("--max-points", type=int, default=4500)
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=140)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.max_points <= 0 or args.frames < 2 or args.fps <= 0 or args.dpi <= 0:
        print("max-points, frames, fps, and dpi must be positive", file=sys.stderr)
        return 2
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        import numpy as np
    except ImportError as exc:
        print(f"missing plotting dependency: {exc}; install numpy matplotlib pillow", file=sys.stderr)
        return 2

    try:
        reference = read_images(args.reference_model / "images.txt")
        reference_names = sorted(reference, key=stem_key)
        names = reference_names
        if args.allow_partial:
            visloc_names = set(read_images(args.visloc_model / "images.txt"))
            colmap_names = set(read_images(args.colmap_model / "images.txt"))
            names = [
                name
                for name in reference_names
                if name in visloc_names and name in colmap_names
            ]
        if len(names) < 3:
            raise ValueError("reference model must contain at least three images")
        reference_centres = np.asarray([reference[name][0] for name in names], dtype=float)
        visloc, visloc_points, visloc_point_error, visloc_observations, visloc_image_count = load_model(
            args.visloc_model, names
        )
        colmap, colmap_points, colmap_point_error, colmap_observations, colmap_image_count = load_model(
            args.colmap_model, names
        )
        if not np.all(np.isfinite(reference_centres)):
            raise ValueError("reference centres contain non-finite values")
        aligned_visloc, visloc_scale, _, _ = umeyama(visloc, reference_centres)
        aligned_colmap, colmap_scale, _, _ = umeyama(colmap, reference_centres)
        visloc_errors = np.linalg.norm(aligned_visloc - reference_centres, axis=1)
        colmap_errors = np.linalg.norm(aligned_colmap - reference_centres, axis=1)
        aligned_points = None
        if len(visloc_points):
            aligned_points = (visloc_scale * (umeyama(visloc, reference_centres)[2] @ visloc_points.T)).T
            # Apply the translation returned above without recomputing the fit.
            _, _, rotation, translation = umeyama(visloc, reference_centres)
            aligned_points = (visloc_scale * (rotation @ visloc_points.T)).T + translation
        if aligned_points is not None and len(aligned_points) > args.max_points:
            # Keep the central scene cloud and avoid a handful of bad rays
            # expanding the plot.  The selection is deterministic.
            centre = reference_centres.mean(axis=0)
            distances = np.linalg.norm(aligned_points - centre, axis=1)
            keep = np.argsort(distances, kind="stable")[: args.max_points]
            aligned_points = aligned_points[keep]
        print(
            f"visloc: {visloc_image_count}/{len(reference_names)} cameras, {len(visloc_points)} points, "
            f"{visloc_observations} observations, centre RMSE {np.sqrt(np.mean(visloc_errors**2))*100:.4f} cm"
        )
        print(
            f"colmap: {colmap_image_count}/{len(reference_names)} cameras, {len(colmap_points)} points, "
            f"{colmap_observations} observations, centre RMSE {np.sqrt(np.mean(colmap_errors**2))*100:.4f} cm"
        )
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 1

    def pca_projection(points):
        mean = reference_centres.mean(axis=0)
        _, _, basis = np.linalg.svd(reference_centres - mean, full_matrices=False)
        return np.column_stack(((points - mean) @ basis[0], (points - mean) @ basis[1]))

    reference_2d = pca_projection(reference_centres)
    visloc_2d = pca_projection(aligned_visloc)
    colmap_2d = pca_projection(aligned_colmap)
    points_2d = pca_projection(aligned_points) if aligned_points is not None and len(aligned_points) else None
    limits = np.concatenate([reference_2d, visloc_2d, colmap_2d], axis=0)
    if points_2d is not None and len(points_2d):
        limits = np.concatenate([limits, points_2d], axis=0)
    low = limits.min(axis=0)
    high = limits.max(axis=0)
    padding = max(0.05 * float(np.max(high - low)), 0.25)
    xlim = (float(low[0] - padding), float(high[0] + padding))
    ylim = (float(low[1] - padding), float(high[1] + padding))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.prefix}.png"
    gif_path = args.output_dir / f"{args.prefix}.gif"
    visloc_rmse = float(np.sqrt(np.mean(visloc_errors**2)) * 100.0)
    colmap_rmse = float(np.sqrt(np.mean(colmap_errors**2)) * 100.0)
    improvement = 100.0 * (1.0 - visloc_rmse / colmap_rmse)
    ratio = colmap_rmse / visloc_rmse

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9})
    fig = plt.figure(figsize=(12.0, 7.0), dpi=args.dpi, facecolor="white")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.65, 1.0), height_ratios=(1.0, 0.72), hspace=0.32, wspace=0.24)
    trajectory = fig.add_subplot(grid[:, 0])
    residuals = fig.add_subplot(grid[0, 1])
    summary = fig.add_subplot(grid[1, 1])

    if points_2d is not None and len(points_2d):
        trajectory.scatter(points_2d[:, 0], points_2d[:, 1], s=1.0, c="#b9c1c9", alpha=0.32, linewidths=0, label="visloc sparse points")
    trajectory.plot(reference_2d[:, 0], reference_2d[:, 1], color="#252a30", lw=1.3, ls=":", label="calibration reference")
    trajectory.plot(colmap_2d[:, 0], colmap_2d[:, 1], color="#e07a24", lw=1.5, ls="--", marker="o", ms=3.0, label="COLMAP CPU")
    trajectory.plot(visloc_2d[:, 0], visloc_2d[:, 1], color="#087f8c", lw=2.0, marker="o", ms=3.2, label="visloc-rs")
    trajectory.scatter(reference_2d[:, 0], reference_2d[:, 1], marker="*", s=42, facecolors="white", edgecolors="#252a30", linewidths=0.8, zorder=5)
    label_stride = max(6, len(names) // 12)
    for index, name in enumerate(names):
        if index in (0, len(names) - 1) or index % label_stride == 0:
            trajectory.annotate(plot_label(name, index), (reference_2d[index, 0], reference_2d[index, 1]), xytext=(3, 3), textcoords="offset points", fontsize=7, color="#343a40")
    trajectory.set_xlim(*xlim)
    trajectory.set_ylim(*ylim)
    trajectory.set_aspect("equal", adjustable="box")
    trajectory.set_xlabel("PCA axis 1 [m]")
    trajectory.set_ylabel("PCA axis 2 [m]")
    trajectory.set_title(f"{args.scene_title} camera centres and sparse cloud")
    trajectory.grid(True, ls=":", alpha=0.35)
    trajectory.legend(loc="best", fontsize=8, framealpha=0.94)

    positions = np.arange(len(names))
    width = 0.38
    residuals.bar(positions - width / 2, colmap_errors * 100.0, width, color="#e07a24", alpha=0.9, label="COLMAP")
    residuals.bar(positions + width / 2, visloc_errors * 100.0, width, color="#087f8c", alpha=0.9, label="visloc-rs")
    residuals.set_xticks(positions[:: max(1, len(positions) // 6)])
    residuals.set_xticklabels([plot_label(names[i], int(i)) for i in positions[:: max(1, len(positions) // 6)]], rotation=45, ha="right", fontsize=7)
    residuals.set_ylabel("centre error [cm]")
    residuals.set_title("Per-camera residual to calibration")
    residuals.grid(True, axis="y", ls=":", alpha=0.35)
    residuals.legend(fontsize=8)

    summary.axis("off")
    summary.text(0.0, 0.98, args.summary_label, transform=summary.transAxes, va="top", fontweight="bold", color="#252a30")
    summary.text(0.0, 0.79, f"visloc-rs   {visloc_rmse:.4f} cm centre RMSE\nCOLMAP      {colmap_rmse:.4f} cm centre RMSE", transform=summary.transAxes, va="top", family="monospace", color="#087f8c")
    if improvement >= 0.0:
        quality_text = f"{improvement:.1f}% lower RMSE  ·  {ratio:.2f}× lower"
    else:
        quality_text = f"quality gap: {visloc_rmse / colmap_rmse:.2f}× COLMAP RMSE"
    camera_text = (
        f"{visloc_image_count}/{len(reference_names)} visloc cameras  ·  "
        f"{colmap_image_count}/{len(reference_names)} COLMAP cameras"
    )
    summary.text(0.0, 0.49, f"{quality_text}\n{camera_text}", transform=summary.transAxes, va="top", fontweight="bold", color="#252a30")
    performance_lines = []
    if args.visloc_mapper_seconds and args.colmap_mapper_seconds:
        performance_lines.append(
            f"mapper: {args.visloc_mapper_seconds:.0f}s vs {args.colmap_mapper_seconds:.0f}s "
            f"({args.colmap_mapper_seconds / args.visloc_mapper_seconds:.1f}× faster)"
        )
    if args.visloc_peak_rss_kib and args.colmap_peak_rss_kib:
        performance_lines.append(
            f"peak RSS: {args.visloc_peak_rss_kib / 1048576:.2f} vs "
            f"{args.colmap_peak_rss_kib / 1048576:.2f} GiB"
        )
    if performance_lines:
        summary.text(0.0, 0.22, "\n".join(performance_lines), transform=summary.transAxes, va="top", fontsize=8, fontweight="bold", color="#087f8c")
        note_y = 0.02
    else:
        note_y = 0.20
    summary.text(0.0, note_y, "Centre RMSE is Sim(3)-aligned to the supplied calibration proxy.\nTracks/points and reprojection reports use different pipelines.", transform=summary.transAxes, va="top", fontsize=7.5, color="#4e5964")

    fig.suptitle(f"Measured {args.scene_title} SfM comparison", fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # The GIF progressively reveals the two estimated paths over a fixed
    # calibration frame.  It is deliberately 2-D/PCA rather than a dramatic
    # fabricated fly-through: every plotted point is from the input model.
    animation_figure = plt.figure(figsize=(8.4, 4.7), dpi=92, facecolor="white")
    animation_grid = animation_figure.add_gridspec(1, 2, width_ratios=(1.65, 0.9), wspace=0.22)
    animation_ax = animation_figure.add_subplot(animation_grid[0, 0])
    animation_error_ax = animation_figure.add_subplot(animation_grid[0, 1])
    if points_2d is not None and len(points_2d):
        animation_ax.scatter(points_2d[:, 0], points_2d[:, 1], s=0.8, c="#b9c1c9", alpha=0.27, linewidths=0)
    animation_ax.plot(reference_2d[:, 0], reference_2d[:, 1], color="#252a30", lw=1.1, ls=":", label="reference")
    animation_ax.scatter(reference_2d[:, 0], reference_2d[:, 1], marker="*", s=30, facecolors="white", edgecolors="#252a30", linewidths=0.7, label="reference centres")
    estimated_line, = animation_ax.plot([], [], color="#087f8c", lw=2.0, marker="o", ms=3.0, label="visloc-rs")
    colmap_line, = animation_ax.plot([], [], color="#e07a24", lw=1.5, ls="--", marker="o", ms=2.7, label="COLMAP")
    current_marker, = animation_ax.plot([], [], marker="o", ms=8, mec="#087f8c", mfc="none", mew=1.6, linestyle="none")
    animation_ax.set_xlim(*xlim)
    animation_ax.set_ylim(*ylim)
    animation_ax.set_aspect("equal", adjustable="box")
    animation_ax.set_xlabel("PCA axis 1 [m]")
    animation_ax.set_ylabel("PCA axis 2 [m]")
    animation_ax.grid(True, ls=":", alpha=0.35)
    animation_ax.legend(loc="best", fontsize=7, framealpha=0.94)
    animation_ax.set_title("Progressive camera-centre reveal")

    animation_error_ax.set_title("Centre residual [cm]")
    animation_error_ax.set_ylabel("camera index")
    animation_error_ax.set_xlim(0.0, max(float(np.max(colmap_errors)), float(np.max(visloc_errors))) * 100.0 * 1.12)
    animation_error_ax.set_ylim(-0.5, len(names) - 0.5)
    animation_error_ax.invert_yaxis()
    animation_error_ax.grid(True, axis="x", ls=":", alpha=0.35)
    animation_error_ax.barh(positions, colmap_errors * 100.0, color="#e07a24", alpha=0.82, height=0.62, label="COLMAP")
    animation_error_ax.barh(positions, visloc_errors * 100.0, color="#087f8c", alpha=0.82, height=0.34, label="visloc-rs")
    animation_error_ax.set_yticks(positions[:: max(1, len(positions) // 6)])
    animation_error_ax.set_yticklabels([plot_label(names[i], int(i)) for i in positions[:: max(1, len(positions) // 6)]], fontsize=7)
    animation_error_ax.legend(fontsize=7, loc="lower right")

    def update(frame: int):
        count = min(len(names), 1 + int(round(frame * (len(names) - 1) / max(1, args.frames - 1))))
        estimated_line.set_data(visloc_2d[:count, 0], visloc_2d[:count, 1])
        colmap_line.set_data(colmap_2d[:count, 0], colmap_2d[:count, 1])
        current_marker.set_data([visloc_2d[count - 1, 0]], [visloc_2d[count - 1, 1]])
        animation_figure.suptitle(f"{args.scene_title} SfM · {count}/{len(names)} cameras revealed", fontsize=12, fontweight="bold")
        return estimated_line, colmap_line, current_marker

    animation = FuncAnimation(animation_figure, update, frames=args.frames, interval=1000 / args.fps, blit=False, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=args.fps), dpi=92)
    plt.close(animation_figure)
    print(f"wrote {png_path} ({png_path.stat().st_size} bytes)")
    print(f"wrote {gif_path} ({gif_path.stat().st_size} bytes, {args.frames} frames)")
    if improvement >= 0.0:
        print(f"visloc improvement: {improvement:.2f}% lower centre RMSE ({ratio:.3f}x lower)")
    else:
        print(f"visloc quality gap: {visloc_rmse / colmap_rmse:.3f}x COLMAP centre RMSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
