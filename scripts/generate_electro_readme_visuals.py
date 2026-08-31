#!/usr/bin/env python3
"""Generate the measured ETH3D Electro visuals used by the README.

The script consumes three COLMAP text models: the official Electro reference
poses, a visloc-rs model, and a COLMAP control model.  Camera centres from the
two reconstructions are independently Sim(3)-aligned to the reference.  The
PNG and GIF therefore show real camera centres and sparse points from the
supplied models; no synthetic trajectory or geometry is introduced.

Timing, memory, registration, and score figures are accepted as explicit
arguments because they come from the benchmark runner and post-mapping score,
not from the COLMAP text model itself.  Omitting a score metric falls back to
the geometry computed by this script.  The generated asset is intentionally a
small, deterministic README visual rather than a benchmark report.

Example::

    python3 scripts/generate_electro_readme_visuals.py \
        --visloc-model /path/to/visloc/model \
        --colmap-model /path/to/colmap/model \
        --reference-model /path/to/rig_calibration_undistorted \
        --output-dir docs/assets \
        --visloc-wall-seconds 336.895 \
        --visloc-core-seconds 230.3015 \
        --visloc-peak-rss-kib 1459194 \
        --visloc-registered 1200 \
        --visloc-rmse-m 0.0350113265 \
        --visloc-median-m 0.0172787756 \
        --visloc-p95-m 0.0770968702 \
        --colmap-wall-seconds 4929.56 \
        --colmap-peak-rss-kib 1255996 \
        --colmap-registered 1200 \
        --colmap-rmse-m 0.0467931597 \
        --colmap-median-m 0.0316 \
        --colmap-p95-m 0.0968

Optional plotting dependencies are numpy, matplotlib, and Pillow.  This is an
asset-generation helper and is not part of the core build or test path.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path


ELECTRO_RIG_RE = re.compile(
    r"(?:^|/)images_rig_cam(?P<camera>[0-9]+)_undistorted/"
    r"(?P<timestamp>[0-9]+)\.[^/]+$"
)
ELECTRO_FLAT_RE = re.compile(
    r"(?:^|/)cam(?P<camera>[0-9]+)_(?P<timestamp>[0-9]+)\.[^/]+$"
)


@dataclass
class ModelData:
    """Camera centres and sparse points loaded from one COLMAP text model."""

    centres: dict[tuple[int, int], object]
    points: object
    observations: int


@dataclass
class RunMetrics:
    """Measured metrics displayed in the comparison cards."""

    label: str
    wall_seconds: float | None
    core_seconds: float | None
    peak_rss_kib: float | None
    registered: int
    rmse_m: float
    median_m: float
    p95_m: float


def image_key(raw_name: str) -> tuple[int, int]:
    """Map flat and official rig image names to ``(camera, timestamp)``."""

    normalized = raw_name.replace("\\", "/")
    match = ELECTRO_RIG_RE.search(normalized) or ELECTRO_FLAT_RE.search(normalized)
    if match is None:
        raise ValueError(
            "unsupported Electro image name "
            f"{raw_name!r}; expected camN_TIMESTAMP.ext or "
            "images_rig_camN_undistorted/TIMESTAMP.ext"
        )
    return int(match.group("camera")), int(match.group("timestamp"))


def image_sort_key(key: tuple[int, int]) -> tuple[int, int]:
    """Order rig cameras temporally, then keep camera order deterministic."""

    camera, timestamp = key
    return timestamp, camera


def display_name(key: tuple[int, int]) -> str:
    camera, timestamp = key
    return f"cam{camera}·{timestamp}"


def rotation_from_quaternion(qw: float, qx: float, qy: float, qz: float):
    import numpy as np

    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError("COLMAP pose has a zero or non-finite quaternion")
    qw, qx, qy, qz = (value / norm for value in (qw, qx, qy, qz))
    return np.asarray(
        [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def read_images(path: Path) -> dict[tuple[int, int], object]:
    """Read ``(camera, timestamp) -> camera centre`` from ``images.txt``."""

    import numpy as np

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read COLMAP images model {path}: {exc}") from exc

    centres: dict[tuple[int, int], object] = {}
    expect_points = False
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if expect_points:
            # Every valid COLMAP text model has one POINTS2D row after each
            # pose.  It can be empty, but it still occupies a line.
            expect_points = False
            continue
        fields = line.split()
        if len(fields) < 10:
            raise ValueError(f"invalid COLMAP image row {path}:{line_number}")
        try:
            int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            translation = np.asarray(
                [float(value) for value in fields[5:8]], dtype=float
            )
            int(fields[8])
        except ValueError as exc:
            raise ValueError(f"invalid COLMAP pose row {path}:{line_number}") from exc
        if not np.all(np.isfinite(translation)):
            raise ValueError(f"non-finite COLMAP translation row {path}:{line_number}")
        key = image_key(fields[9])
        if key in centres:
            raise ValueError(f"duplicate Electro image key {key} in {path}")
        rotation = rotation_from_quaternion(qw, qx, qy, qz)
        centres[key] = -rotation.T @ translation
        expect_points = True

    if not centres:
        raise ValueError(f"COLMAP model contains no registered images: {path}")
    return centres


def read_points(path: Path):
    """Read finite sparse XYZ points and count their track observations."""

    import numpy as np

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read COLMAP points model {path}: {exc}") from exc

    points = []
    observations = 0
    for line_number, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 8:
            raise ValueError(f"invalid COLMAP point row {path}:{line_number}")
        try:
            point = [float(value) for value in fields[1:4]]
        except ValueError as exc:
            raise ValueError(f"invalid COLMAP point row {path}:{line_number}") from exc
        if all(math.isfinite(value) for value in point):
            points.append(point)
        # TRACK[] is a sequence of IMAGE_ID POINT2D_IDX pairs.
        observations += max(0, (len(fields) - 8) // 2)
    return np.asarray(points, dtype=float).reshape((-1, 3)), observations


def load_model(model_dir: Path) -> ModelData:
    """Load one COLMAP text model with explicit file validation."""

    images_path = model_dir / "images.txt"
    points_path = model_dir / "points3D.txt"
    points, observations = read_points(points_path)
    return ModelData(
        centres=read_images(images_path),
        points=points,
        observations=observations,
    )


def umeyama(source, destination):
    """Return ``(scale, rotation, translation, aligned_source)``."""

    import numpy as np

    if source.ndim != 2 or destination.ndim != 2 or source.shape != destination.shape:
        raise ValueError("Sim(3) inputs must have matching N×3 shapes")
    if len(source) < 3:
        raise ValueError("at least three common camera centres are required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(destination)):
        raise ValueError("Sim(3) inputs contain non-finite camera centres")
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean
    covariance = destination_centered.T @ source_centered / len(source)
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    variance = float(np.sum(source_centered * source_centered) / len(source))
    if not math.isfinite(variance) or variance <= 1.0e-15:
        raise ValueError("common camera centres are degenerate for Sim(3)")
    scale = float(np.trace(np.diag(singular_values) @ correction) / variance)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Sim(3) alignment produced an invalid scale")
    translation = destination_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return scale, rotation, translation, aligned


def apply_transform(points, scale, rotation, translation):
    return (scale * (rotation @ points.T)).T + translation


def deterministic_point_sample(points, limit: int):
    """Keep a stable, spatially broad prefix of a COLMAP point file."""

    import numpy as np

    if len(points) <= limit:
        return points
    indices = np.rint(np.linspace(0, len(points) - 1, limit)).astype(np.int64)
    return points[np.unique(indices)]


def select_points_for_view(points_2d, xlim, ylim, limit: int):
    """Keep actual sparse points inside the camera-centre view window."""

    import numpy as np

    if not len(points_2d):
        return points_2d
    visible = (
        np.all(np.isfinite(points_2d), axis=1)
        & (points_2d[:, 0] >= xlim[0])
        & (points_2d[:, 0] <= xlim[1])
        & (points_2d[:, 1] >= ylim[0])
        & (points_2d[:, 1] <= ylim[1])
    )
    return deterministic_point_sample(points_2d[visible], limit)


def parse_nonnegative_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {raw!r}") from exc
    if not math.isfinite(value) or value < 0.0:
        raise argparse.ArgumentTypeError(f"expected a finite non-negative number, got {raw!r}")
    return value


def parse_nonnegative_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {raw!r}") from exc
    if value < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {raw!r}")
    return value


def optional_metric(parser, option: str, dest: str, help_text: str) -> None:
    parser.add_argument(option, dest=dest, type=parse_nonnegative_float, help=help_text)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visloc-model", type=Path, required=True, help="visloc COLMAP text model directory")
    parser.add_argument("--colmap-model", type=Path, required=True, help="COLMAP control text model directory")
    parser.add_argument("--reference-model", type=Path, required=True, help="official Electro reference model directory")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"), help="asset output directory (default: docs/assets)")
    parser.add_argument("--prefix", default="electro_1200_sfm_comparison", help="output filename prefix")
    parser.add_argument("--scene-title", default="ETH3D Electro · 1,200-image unordered SfM")
    parser.add_argument("--run-label", default="Frozen 12,000 candidate-pair control")
    parser.add_argument("--allow-partial", action="store_true", help="align and plot only image keys shared by both reconstructions")
    parser.add_argument("--max-points", type=parse_nonnegative_int, default=5000, help="maximum plotted sparse points per model")
    parser.add_argument("--frames", type=parse_nonnegative_int, default=24, help="GIF frame count")
    parser.add_argument("--fps", type=parse_nonnegative_int, default=8, help="GIF frames per second")
    parser.add_argument("--dpi", type=parse_nonnegative_int, default=130, help="PNG render DPI")

    parser.add_argument("--visloc-label", default="visloc-rs · current champion")
    parser.add_argument("--colmap-label", default="COLMAP · CPU control")
    parser.add_argument("--visloc-registered", type=parse_nonnegative_int)
    parser.add_argument("--colmap-registered", type=parse_nonnegative_int)
    optional_metric(parser, "--visloc-wall-seconds", "visloc_wall_seconds", "measured visloc mapper wall seconds")
    optional_metric(parser, "--visloc-core-seconds", "visloc_core_seconds", "measured visloc mapper core seconds")
    optional_metric(parser, "--visloc-peak-rss-kib", "visloc_peak_rss_kib", "measured visloc peak RSS in KiB")
    optional_metric(parser, "--visloc-rmse-m", "visloc_rmse_m", "measured visloc Sim(3) centre RMSE in metres")
    optional_metric(parser, "--visloc-median-m", "visloc_median_m", "measured visloc centre-error median in metres")
    optional_metric(parser, "--visloc-p95-m", "visloc_p95_m", "measured visloc centre-error p95 in metres")
    optional_metric(parser, "--colmap-wall-seconds", "colmap_wall_seconds", "measured COLMAP mapper wall seconds")
    optional_metric(parser, "--colmap-core-seconds", "colmap_core_seconds", "measured COLMAP mapper core seconds")
    optional_metric(parser, "--colmap-peak-rss-kib", "colmap_peak_rss_kib", "measured COLMAP peak RSS in KiB")
    optional_metric(parser, "--colmap-rmse-m", "colmap_rmse_m", "measured COLMAP Sim(3) centre RMSE in metres")
    optional_metric(parser, "--colmap-median-m", "colmap_median_m", "measured COLMAP centre-error median in metres")
    optional_metric(parser, "--colmap-p95-m", "colmap_p95_m", "measured COLMAP centre-error p95 in metres")
    return parser


def metric_or(value: float | None, fallback: float) -> float:
    return fallback if value is None else value


def compute_alignment(model: ModelData, keys, reference_centres):
    import numpy as np

    source_centres = np.asarray([model.centres[key] for key in keys], dtype=float)
    scale, rotation, translation, aligned_centres = umeyama(
        source_centres, reference_centres
    )
    errors = np.linalg.norm(aligned_centres - reference_centres, axis=1)
    aligned_points = apply_transform(model.points, scale, rotation, translation)
    return aligned_centres, aligned_points, errors, scale


def format_metric_m(value: float) -> str:
    return f"{value * 100.0:.2f} cm"


def format_seconds(value: float | None) -> str:
    return "—" if value is None else f"{value:,.2f} s"


def format_rss(value: float | None) -> str:
    return "—" if value is None else f"{value / 1048576.0:.2f} GiB"


def make_metrics(args, label: str, model: ModelData, errors, prefix: str) -> RunMetrics:
    import numpy as np

    computed_rmse = float(np.sqrt(np.mean(errors**2)))
    computed_median = float(np.median(errors))
    computed_p95 = float(np.percentile(errors, 95.0))
    registered_arg = getattr(args, f"{prefix}_registered")
    registered = len(model.centres) if registered_arg is None else registered_arg
    if registered > len(args.reference_keys):
        raise ValueError(
            f"{prefix} registered count {registered} exceeds reference image count "
            f"{len(args.reference_keys)}"
        )
    return RunMetrics(
        label=label,
        wall_seconds=getattr(args, f"{prefix}_wall_seconds"),
        core_seconds=getattr(args, f"{prefix}_core_seconds"),
        peak_rss_kib=getattr(args, f"{prefix}_peak_rss_kib"),
        registered=registered,
        rmse_m=metric_or(getattr(args, f"{prefix}_rmse_m"), computed_rmse),
        median_m=metric_or(getattr(args, f"{prefix}_median_m"), computed_median),
        p95_m=metric_or(getattr(args, f"{prefix}_p95_m"), computed_p95),
    )


def configure_axes(ax, panel_color: str, ink: str, grid_color: str) -> None:
    ax.set_facecolor(panel_color)
    for spine in ax.spines.values():
        spine.set_color("#d7dee5")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=ink, labelsize=8)
    ax.grid(True, color=grid_color, linestyle=":", linewidth=0.8, alpha=0.8)


def project_pca(points, origin, basis):
    return (points - origin) @ basis.T


def plot_limits(*arrays):
    import numpy as np

    finite = [array[np.all(np.isfinite(array), axis=1)] for array in arrays if len(array)]
    values = np.concatenate(finite, axis=0)
    # The limits intentionally follow camera centres, not the sparse cloud.
    # A few triangulation outliers otherwise make the real camera path a
    # postage stamp.  Point clouds are filtered to this camera-focused window
    # before plotting, while every displayed point remains model-derived.
    low = np.quantile(values, 0.005, axis=0)
    high = np.quantile(values, 0.995, axis=0)
    span = np.maximum(high - low, 1.0e-6)
    padding = np.maximum(span * 0.08, 0.25)
    return (float(low[0] - padding[0]), float(high[0] + padding[0])), (
        float(low[1] - padding[1]), float(high[1] + padding[1])
    )


def add_summary_card(ax, y_top: float, metrics: RunMetrics, total: int, color: str, ink: str, muted: str) -> None:
    from matplotlib.patches import FancyBboxPatch

    # The summary axes occupy a compact part of the figure.  Keep this to
    # three deliberately separated baselines so the card remains legible at
    # README display size as well as in the source PNG.
    card_height = 0.250
    rect = FancyBboxPatch(
        (0.02, y_top - card_height),
        0.96,
        card_height - 0.012,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        transform=ax.transAxes,
        facecolor="#ffffff",
        edgecolor="#dce3e9",
        linewidth=0.8,
    )
    ax.add_patch(rect)
    ax.text(
        0.055,
        y_top - 0.030,
        metrics.label,
        transform=ax.transAxes,
        color=color,
        fontsize=9,
        fontweight="bold",
        va="top",
    )
    ax.text(
        0.055,
        y_top - 0.119,
        f"reg {metrics.registered:,}/{total:,}  ·  RMSE {format_metric_m(metrics.rmse_m)}",
        transform=ax.transAxes,
        color=ink,
        fontsize=7.15,
        va="top",
        family="DejaVu Sans Mono",
    )
    ax.text(
        0.60,
        y_top - 0.119,
        f"median {format_metric_m(metrics.median_m)}  ·  p95 {format_metric_m(metrics.p95_m)}",
        transform=ax.transAxes,
        color=ink,
        fontsize=7.15,
        va="top",
        family="DejaVu Sans Mono",
    )
    timing = f"wall {format_seconds(metrics.wall_seconds)}"
    if metrics.core_seconds is not None:
        timing += f"  ·  core {format_seconds(metrics.core_seconds)}"
    timing += f"  ·  RSS {format_rss(metrics.peak_rss_kib)}"
    ax.text(
        0.055,
        y_top - 0.207,
        timing,
        transform=ax.transAxes,
        color=muted,
        fontsize=7.15,
        va="top",
        family="DejaVu Sans Mono",
    )
def build_png(
    png_path: Path,
    args,
    keys,
    reference_2d,
    visloc_2d,
    colmap_2d,
    visloc_points_2d,
    colmap_points_2d,
    visloc_errors,
    colmap_errors,
    visloc_metrics: RunMetrics,
    colmap_metrics: RunMetrics,
    xlim,
    ylim,
    total: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    bg = "#f5f8fb"
    panel = "#ffffff"
    ink = "#17212b"
    muted = "#667684"
    grid = "#d7e0e8"
    teal = "#008c95"
    orange = "#e7842a"
    reference_color = "#2d3740"
    visloc_cloud = "#62b8bd"
    colmap_cloud = "#e6a267"

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 11,
            "axes.labelsize": 8.5,
            "figure.facecolor": bg,
        }
    )
    fig = plt.figure(figsize=(13.0, 7.2), dpi=args.dpi, facecolor=bg)
    grid_spec = fig.add_gridspec(
        2,
        2,
        width_ratios=(1.7, 1.0),
        height_ratios=(1.0, 0.92),
        left=0.040,
        right=0.970,
        bottom=0.070,
        top=0.815,
        wspace=0.24,
        hspace=0.32,
    )
    trajectory = fig.add_subplot(grid_spec[:, 0])
    distribution = fig.add_subplot(grid_spec[0, 1])
    summary = fig.add_subplot(grid_spec[1, 1])
    for ax in (trajectory, distribution, summary):
        ax.set_facecolor(panel)
    configure_axes(trajectory, panel, ink, grid)
    configure_axes(distribution, panel, ink, grid)
    summary.axis("off")

    if len(visloc_points_2d):
        trajectory.scatter(
            visloc_points_2d[:, 0],
            visloc_points_2d[:, 1],
            s=1.4,
            color=visloc_cloud,
            alpha=0.17,
            linewidths=0,
            label="visloc sparse points",
            rasterized=True,
        )
    if len(colmap_points_2d):
        trajectory.scatter(
            colmap_points_2d[:, 0],
            colmap_points_2d[:, 1],
            s=1.2,
            color=colmap_cloud,
            alpha=0.12,
            linewidths=0,
            label="COLMAP sparse points",
            rasterized=True,
        )
    trajectory.plot(
        reference_2d[:, 0],
        reference_2d[:, 1],
        color=reference_color,
        linewidth=1.1,
        linestyle=":",
        label="official reference",
        zorder=3,
    )
    trajectory.plot(
        colmap_2d[:, 0],
        colmap_2d[:, 1],
        color=orange,
        linewidth=1.45,
        linestyle="--",
        label=colmap_metrics.label,
        zorder=4,
    )
    trajectory.plot(
        visloc_2d[:, 0],
        visloc_2d[:, 1],
        color=teal,
        linewidth=2.0,
        label=visloc_metrics.label,
        zorder=5,
    )
    trajectory.scatter(
        reference_2d[:, 0],
        reference_2d[:, 1],
        s=18,
        marker="x",
        color=reference_color,
        linewidths=0.55,
        alpha=0.72,
        zorder=6,
    )
    label_indices = np.unique(np.rint(np.linspace(0, len(keys) - 1, 7)).astype(int))
    for index in label_indices:
        trajectory.annotate(
            f"cam{keys[index][0]} · {index}",
            (reference_2d[index, 0], reference_2d[index, 1]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.7,
            color=muted,
        )
    trajectory.set_xlim(*xlim)
    trajectory.set_ylim(*ylim)
    trajectory.set_aspect("equal", adjustable="box")
    trajectory.set_xlabel("PCA axis 1 [m]")
    trajectory.set_ylabel("PCA axis 2 [m]")
    trajectory.set_title("Aligned camera centres + sparse structure", loc="left", color=ink, pad=8, fontweight="bold")
    trajectory.legend(loc="upper right", fontsize=7.2, framealpha=0.94, facecolor="#ffffff")

    vis_error_cm = np.sort(visloc_errors * 100.0)
    col_error_cm = np.sort(colmap_errors * 100.0)
    ecdf_y = np.linspace(100.0 / len(keys), 100.0, len(keys))
    distribution.plot(col_error_cm, ecdf_y, color=orange, linewidth=1.8, label=colmap_metrics.label)
    distribution.plot(vis_error_cm, ecdf_y, color=teal, linewidth=2.1, label=visloc_metrics.label)
    distribution.fill_between(vis_error_cm, ecdf_y, color=teal, alpha=0.08)
    distribution.set_xlim(0.0, max(float(col_error_cm[-1]), float(vis_error_cm[-1])) * 1.06)
    distribution.set_ylim(0.0, 100.0)
    distribution.set_xlabel("camera-centre error [cm]")
    distribution.set_ylabel("cameras within error [%]")
    distribution.set_title("Residual distribution to official reference", loc="left", color=ink, pad=8, fontweight="bold")
    distribution.legend(fontsize=7.1, loc="lower right", framealpha=0.94, facecolor="#ffffff")
    distribution.text(0.03, 0.045, "lower curve = tighter camera-centre agreement", transform=distribution.transAxes, color=muted, fontsize=6.7, va="bottom")

    summary.text(0.03, 0.98, "MEASURED RUN", transform=summary.transAxes, color=muted, fontsize=7.4, fontweight="bold", va="top")
    summary.text(0.03, 0.91, args.run_label, transform=summary.transAxes, color=ink, fontsize=10.2, fontweight="bold", va="top")
    add_summary_card(summary, 0.82, visloc_metrics, total, teal, ink, muted)
    add_summary_card(summary, 0.530, colmap_metrics, total, orange, ink, muted)

    speed_ratio = None
    if visloc_metrics.wall_seconds and colmap_metrics.wall_seconds:
        speed_ratio = colmap_metrics.wall_seconds / visloc_metrics.wall_seconds
    rmse_delta = 100.0 * (1.0 - visloc_metrics.rmse_m / colmap_metrics.rmse_m) if colmap_metrics.rmse_m else None
    callout = []
    if speed_ratio is not None:
        callout.append(f"{speed_ratio:.1f}× faster mapper wall")
    if rmse_delta is not None:
        callout.append(f"{rmse_delta:.1f}% lower centre RMSE")
    if callout:
        summary.text(0.03, 0.240, callout[0], transform=summary.transAxes, color=teal, fontsize=9.4, fontweight="bold", va="top")
    if len(callout) > 1:
        summary.text(0.03, 0.190, callout[1], transform=summary.transAxes, color=teal, fontsize=9.4, fontweight="bold", va="top")
    summary.text(
        0.03,
        0.105,
        "Centres: independent Sim(3) to official reference.\n"
        "Timing / RSS / scores: supplied metrics; points: raw model.",
        transform=summary.transAxes,
        color=muted,
        fontsize=6.8,
        va="top",
    )

    fig.suptitle(args.scene_title, x=0.040, y=0.965, ha="left", color=ink, fontsize=17, fontweight="bold")
    fig.text(0.040, 0.925, "Current visloc-rs champion  ·  COLMAP control  ·  official reference  ·  real sparse structure", color=muted, fontsize=8.4)
    fig.savefig(png_path, dpi=args.dpi, facecolor=bg)
    plt.close(fig)


def build_gif(
    gif_path: Path,
    args,
    reference_2d,
    visloc_2d,
    colmap_2d,
    visloc_points_2d,
    colmap_points_2d,
    visloc_errors,
    colmap_errors,
    visloc_metrics: RunMetrics,
    colmap_metrics: RunMetrics,
    xlim,
    ylim,
    total: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FuncAnimation, PillowWriter

    bg = "#f5f8fb"
    panel = "#ffffff"
    ink = "#17212b"
    muted = "#667684"
    grid = "#d7e0e8"
    teal = "#008c95"
    orange = "#e7842a"
    reference_color = "#2d3740"
    visloc_cloud = "#62b8bd"
    colmap_cloud = "#e6a267"
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": bg})

    fig = plt.figure(figsize=(9.6, 5.4), dpi=90, facecolor=bg)
    spec = fig.add_gridspec(1, 2, width_ratios=(1.68, 1.0), left=0.065, right=0.97, bottom=0.10, top=0.83, wspace=0.24)
    trajectory = fig.add_subplot(spec[0, 0])
    summary = fig.add_subplot(spec[0, 1])
    configure_axes(trajectory, panel, ink, grid)
    summary.axis("off")
    if len(visloc_points_2d):
        trajectory.scatter(visloc_points_2d[:, 0], visloc_points_2d[:, 1], s=1.0, color=visloc_cloud, alpha=0.14, linewidths=0, rasterized=True)
    if len(colmap_points_2d):
        trajectory.scatter(colmap_points_2d[:, 0], colmap_points_2d[:, 1], s=0.9, color=colmap_cloud, alpha=0.10, linewidths=0, rasterized=True)
    trajectory.plot(reference_2d[:, 0], reference_2d[:, 1], color=reference_color, linewidth=1.0, linestyle=":", label="reference")
    trajectory.scatter(reference_2d[:, 0], reference_2d[:, 1], s=11, marker="x", color=reference_color, linewidths=0.45, alpha=0.68)
    visloc_line, = trajectory.plot([], [], color=teal, linewidth=2.0, label="visloc-rs")
    colmap_line, = trajectory.plot([], [], color=orange, linewidth=1.45, linestyle="--", label="COLMAP")
    visloc_marker, = trajectory.plot([], [], marker="o", markersize=7, markerfacecolor="white", markeredgecolor=teal, markeredgewidth=1.5, linestyle="none")
    colmap_marker, = trajectory.plot([], [], marker="o", markersize=6, markerfacecolor="white", markeredgecolor=orange, markeredgewidth=1.3, linestyle="none")
    trajectory.set_xlim(*xlim)
    trajectory.set_ylim(*ylim)
    trajectory.set_aspect("equal", adjustable="box")
    trajectory.set_xlabel("PCA axis 1 [m]")
    trajectory.set_ylabel("PCA axis 2 [m]")
    trajectory.legend(loc="upper right", fontsize=7, framealpha=0.94, facecolor="#ffffff")

    summary.text(0.03, 0.98, "MEASURED COMPARISON", transform=summary.transAxes, color=muted, fontsize=7.4, fontweight="bold", va="top")
    summary.text(0.03, 0.90, args.run_label, transform=summary.transAxes, color=ink, fontsize=9.3, fontweight="bold", va="top", wrap=True)
    summary.text(0.03, 0.79, "visloc-rs · current champion", transform=summary.transAxes, color=teal, fontsize=8.6, fontweight="bold", va="top")
    visloc_text = summary.text(0.03, 0.73, "", transform=summary.transAxes, color=ink, fontsize=8.0, family="DejaVu Sans Mono", va="top")
    summary.text(0.03, 0.56, "COLMAP · CPU control", transform=summary.transAxes, color=orange, fontsize=8.6, fontweight="bold", va="top")
    colmap_text = summary.text(0.03, 0.50, "", transform=summary.transAxes, color=ink, fontsize=8.0, family="DejaVu Sans Mono", va="top")
    speed_ratio = None
    if visloc_metrics.wall_seconds and colmap_metrics.wall_seconds:
        speed_ratio = colmap_metrics.wall_seconds / visloc_metrics.wall_seconds
    rmse_delta = 100.0 * (1.0 - visloc_metrics.rmse_m / colmap_metrics.rmse_m) if colmap_metrics.rmse_m else None
    callout = []
    if speed_ratio is not None:
        callout.append(f"{speed_ratio:.1f}× faster wall")
    if rmse_delta is not None:
        callout.append(f"{rmse_delta:.1f}% lower RMSE")
    summary.text(0.03, 0.29, "   ·   ".join(callout), transform=summary.transAxes, color=teal, fontsize=9.2, fontweight="bold", va="top")
    summary.text(0.03, 0.14, "Actual centres + sparse points\nSim(3)-aligned to official reference", transform=summary.transAxes, color=muted, fontsize=7.0, va="top")

    def update(frame: int):
        count = min(total, 1 + int(round(frame * (total - 1) / max(1, args.frames - 1))))
        visloc_line.set_data(visloc_2d[:count, 0], visloc_2d[:count, 1])
        colmap_line.set_data(colmap_2d[:count, 0], colmap_2d[:count, 1])
        visloc_marker.set_data([visloc_2d[count - 1, 0]], [visloc_2d[count - 1, 1]])
        colmap_marker.set_data([colmap_2d[count - 1, 0]], [colmap_2d[count - 1, 1]])
        visloc_text.set_text(
            f"{count:,}/{total:,} cameras\nRMSE {format_metric_m(visloc_metrics.rmse_m)}\n"
            f"wall {format_seconds(visloc_metrics.wall_seconds)}  RSS {format_rss(visloc_metrics.peak_rss_kib)}"
        )
        colmap_text.set_text(
            f"{count:,}/{total:,} cameras\nRMSE {format_metric_m(colmap_metrics.rmse_m)}\n"
            f"wall {format_seconds(colmap_metrics.wall_seconds)}  RSS {format_rss(colmap_metrics.peak_rss_kib)}"
        )
        fig.suptitle(f"{args.scene_title}  ·  {count:,}/{total:,} camera centres revealed", x=0.065, y=0.92, ha="left", color=ink, fontsize=12, fontweight="bold")
        return visloc_line, colmap_line, visloc_marker, colmap_marker, visloc_text, colmap_text

    animation = FuncAnimation(fig, update, frames=args.frames, interval=1000.0 / args.fps, blit=False, repeat=True)
    animation.save(gif_path, writer=PillowWriter(fps=args.fps), dpi=90)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_points <= 0 or args.frames < 2 or args.fps <= 0 or args.dpi <= 0:
        print("max-points must be positive, frames must be at least 2, fps and dpi must be positive", file=sys.stderr)
        return 2
    try:
        import matplotlib

        matplotlib.use("Agg")
        import numpy as np

        reference_model = load_model(args.reference_model)
        visloc_model = load_model(args.visloc_model)
        colmap_model = load_model(args.colmap_model)
        reference_keys = sorted(reference_model.centres, key=image_sort_key)
        visloc_keys = set(visloc_model.centres)
        colmap_keys = set(colmap_model.centres)
        missing_visloc = [key for key in reference_keys if key not in visloc_keys]
        missing_colmap = [key for key in reference_keys if key not in colmap_keys]
        if not args.allow_partial and (missing_visloc or missing_colmap):
            details = []
            if missing_visloc:
                details.append(f"visloc missing {len(missing_visloc)} (first {display_name(missing_visloc[0])})")
            if missing_colmap:
                details.append(f"COLMAP missing {len(missing_colmap)} (first {display_name(missing_colmap[0])})")
            raise ValueError("models do not cover the full reference: " + "; ".join(details))
        keys = [key for key in reference_keys if key in visloc_keys and key in colmap_keys]
        if len(keys) < 3:
            raise ValueError(f"only {len(keys)} common Electro image keys; at least three are required")
        reference_centres = np.asarray([reference_model.centres[key] for key in keys], dtype=float)
        if not np.all(np.isfinite(reference_centres)):
            raise ValueError("reference centres contain non-finite values")

        visloc_centres, visloc_points, visloc_errors, visloc_scale = compute_alignment(visloc_model, keys, reference_centres)
        colmap_centres, colmap_points, colmap_errors, colmap_scale = compute_alignment(colmap_model, keys, reference_centres)
        args.reference_keys = reference_keys
        visloc_metrics = make_metrics(args, args.visloc_label, visloc_model, visloc_errors, "visloc")
        colmap_metrics = make_metrics(args, args.colmap_label, colmap_model, colmap_errors, "colmap")

        reference_mean = reference_centres.mean(axis=0)
        _, _, pca_basis = np.linalg.svd(reference_centres - reference_mean, full_matrices=False)
        pca_basis = pca_basis[:2]
        reference_2d = project_pca(reference_centres, reference_mean, pca_basis)
        visloc_2d = project_pca(visloc_centres, reference_mean, pca_basis)
        colmap_2d = project_pca(colmap_centres, reference_mean, pca_basis)
        xlim, ylim = plot_limits(reference_2d, visloc_2d, colmap_2d)
        # Keep the displayed cloud tied to the same camera-focused limits.
        # This avoids a handful of far-out triangulation points compressing
        # the 1,200-camera trajectory while retaining real nearby structure.
        visloc_points_2d = select_points_for_view(
            project_pca(visloc_points, reference_mean, pca_basis),
            xlim,
            ylim,
            args.max_points,
        )
        colmap_points_2d = select_points_for_view(
            project_pca(colmap_points, reference_mean, pca_basis),
            xlim,
            ylim,
            args.max_points,
        )

        args.output_dir.mkdir(parents=True, exist_ok=True)
        png_path = args.output_dir / f"{args.prefix}.png"
        gif_path = args.output_dir / f"{args.prefix}.gif"
        build_png(
            png_path,
            args,
            keys,
            reference_2d,
            visloc_2d,
            colmap_2d,
            visloc_points_2d,
            colmap_points_2d,
            visloc_errors,
            colmap_errors,
            visloc_metrics,
            colmap_metrics,
            xlim,
            ylim,
            len(reference_keys),
        )
        build_gif(
            gif_path,
            args,
            reference_2d,
            visloc_2d,
            colmap_2d,
            visloc_points_2d,
            colmap_points_2d,
            visloc_errors,
            colmap_errors,
            visloc_metrics,
            colmap_metrics,
            xlim,
            ylim,
            len(reference_keys),
        )
        print(
            f"reference: {len(reference_model.centres):,} cameras, {len(reference_model.points):,} points"
        )
        print(
            f"visloc: {len(visloc_model.centres):,} cameras, {len(visloc_model.points):,} points, "
            f"{visloc_model.observations:,} observations, geometry RMSE {format_metric_m(float(np.sqrt(np.mean(visloc_errors**2))))}"
        )
        print(
            f"COLMAP: {len(colmap_model.centres):,} cameras, {len(colmap_model.points):,} points, "
            f"{colmap_model.observations:,} observations, geometry RMSE {format_metric_m(float(np.sqrt(np.mean(colmap_errors**2))))}"
        )
        print(f"Sim(3) scales: visloc {visloc_scale:.9f}, COLMAP {colmap_scale:.9f}")
        print(f"wrote {png_path} ({png_path.stat().st_size:,} bytes)")
        print(f"wrote {gif_path} ({gif_path.stat().st_size:,} bytes, {args.frames} frames)")
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"input/render error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
