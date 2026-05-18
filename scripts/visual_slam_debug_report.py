#!/usr/bin/env python3
"""Build a compact visual-SLAM debug report from demo output files.

The tool is intentionally dependency-free. Point it at an output directory from
`online_slam_stereo_vo_kitti_demo` or `run_kitti_deep_vo_smoke.sh`; it will
combine frontend pair diagnostics, per-pair GT-relative errors, and KITTI
segment errors into JSON, Markdown, HTML, and worst-pair CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="VO/SLAM output directory")
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="Optional baseline run directory to diff against run_dir",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Report output directory (default: <run_dir>/slam_debug)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=12,
        help="Number of worst rows to include per table",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_run_dir(path: Path) -> Path:
    path = path.expanduser().resolve()
    if (path / "frontend_pair_diagnostics.csv").exists() or (path / "summary.txt").exists():
        return path
    candidates = [
        child
        for child in path.iterdir()
        if child.is_dir()
        and ((child / "frontend_pair_diagnostics.csv").exists() or (child / "summary.txt").exists())
    ] if path.exists() else []
    if len(candidates) == 1:
        return candidates[0]
    return path


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def f64(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def i64(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def parse_summary_txt(path: Path) -> dict[str, float | int | str]:
    metrics: dict[str, float | int | str] = {}
    if not path.exists():
        return metrics
    for token in path.read_text().replace("\n", " ").split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        parsed_f = f64(value)
        if parsed_f is None:
            metrics[key] = value
        elif parsed_f.is_integer():
            metrics[key] = int(parsed_f)
        else:
            metrics[key] = parsed_f
    return metrics


def load_kitti_summary(run_dir: Path, name: str) -> dict[str, Any]:
    return read_json(run_dir / name / "kitti_odometry_summary.json")


def describe(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "max": max(values),
    }


def top_by(rows: list[dict[str, Any]], key: str, n: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: row.get(key) or -math.inf, reverse=True)[:n]


def load_pair_rows(run_dir: Path) -> list[dict[str, Any]]:
    frontend = read_csv(run_dir / "frontend_pair_diagnostics.csv")
    relative = {
        (row.get("from_id"), row.get("to_id")): row
        for row in read_csv(run_dir / "relative_pose_errors.csv")
    }
    rows: list[dict[str, Any]] = []
    for row in frontend:
        key = (row.get("from_id"), row.get("to_id"))
        rel = relative.get(key, {})
        pnp_corr = i64(row.get("pnp_correspondences")) or 0
        inliers = i64(row.get("inliers")) or 0
        rescue_tags = "|".join(
            label
            for key, label in [
                ("motion_scale_rescued", "scale"),
                ("translation_direction_rescued", "dir"),
                ("stereo_vertical_aligned", "vertical_align"),
                ("rotation_spike_rescued", "rot"),
                ("rotation_vector_rescued", "rotvec"),
            ]
            if row.get(key) == "true"
        )
        rows.append(
            {
                "from_id": i64(row.get("from_id")),
                "to_id": i64(row.get("to_id")),
                "source": row.get("source", ""),
                "temporal_matches": i64(row.get("temporal_matches")),
                "pnp_correspondences": pnp_corr,
                "stereo_pair_correspondences": i64(row.get("stereo_pair_correspondences")),
                "inliers": inliers,
                "inlier_ratio": inliers / pnp_corr if pnp_corr > 0 else None,
                "raw_translation_m": f64(row.get("raw_translation_m")),
                "raw_rotation_deg": f64(row.get("raw_rotation_deg")),
                "translation_m": f64(row.get("translation_m")),
                "rotation_deg": f64(row.get("rotation_deg")),
                "motion_scale_rescued": row.get("motion_scale_rescued", ""),
                "translation_direction_rescued": row.get("translation_direction_rescued", ""),
                "stereo_vertical_aligned": row.get("stereo_vertical_aligned", ""),
                "rotation_spike_rescued": row.get("rotation_spike_rescued", ""),
                "rotation_vector_rescued": row.get("rotation_vector_rescued", ""),
                "rescue_tags": rescue_tags,
                "pnp_mean_reprojection_error_px": f64(row.get("pnp_mean_reprojection_error_px")),
                "kabsch_mean_residual_m": f64(row.get("kabsch_mean_residual_m")),
                "estimated_translation_m": f64(rel.get("estimated_translation_m")),
                "reference_translation_m": f64(rel.get("reference_translation_m")),
                "translation_magnitude_error_m": f64(rel.get("translation_magnitude_error_m")),
                "translation_vector_error_m": f64(rel.get("translation_vector_error_m")),
                "estimated_tx_m": f64(rel.get("estimated_tx_m")),
                "estimated_ty_m": f64(rel.get("estimated_ty_m")),
                "estimated_tz_m": f64(rel.get("estimated_tz_m")),
                "reference_tx_m": f64(rel.get("reference_tx_m")),
                "reference_ty_m": f64(rel.get("reference_ty_m")),
                "reference_tz_m": f64(rel.get("reference_tz_m")),
                "translation_error_x_m": f64(rel.get("translation_error_x_m")),
                "translation_error_y_m": f64(rel.get("translation_error_y_m")),
                "translation_error_z_m": f64(rel.get("translation_error_z_m")),
                "abs_translation_error_x_m": abs(f64(rel.get("translation_error_x_m")) or 0.0)
                if f64(rel.get("translation_error_x_m")) is not None
                else None,
                "abs_translation_error_y_m": abs(f64(rel.get("translation_error_y_m")) or 0.0)
                if f64(rel.get("translation_error_y_m")) is not None
                else None,
                "abs_translation_error_z_m": abs(f64(rel.get("translation_error_z_m")) or 0.0)
                if f64(rel.get("translation_error_z_m")) is not None
                else None,
                "rotation_error_deg": f64(rel.get("rotation_error_deg")),
            }
        )
    return rows


def load_segment_rows(run_dir: Path) -> list[dict[str, Any]]:
    candidates = [
        run_dir / "kitti_eval_public_lengths" / "kitti_odometry_segments.csv",
        run_dir / "kitti_eval_100m" / "kitti_odometry_segments.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        rows.append(
            {
                "first_frame_id": i64(row.get("first_frame_id")),
                "last_frame_id": i64(row.get("last_frame_id")),
                "length_m": f64(row.get("length_m")),
                "translational_error_percent": f64(row.get("translational_error_percent")),
                "rotational_error_deg_per_m": f64(row.get("rotational_error_deg_per_m")),
            }
        )
    return rows


def annotate_segments_with_component_errors(
    segment_rows: list[dict[str, Any]],
    pair_rows: list[dict[str, Any]],
) -> None:
    usable_pairs = [
        row
        for row in pair_rows
        if row.get("from_id") is not None
        and row.get("to_id") is not None
        and row.get("translation_error_x_m") is not None
        and row.get("translation_error_y_m") is not None
        and row.get("translation_error_z_m") is not None
    ]
    for segment in segment_rows:
        first = segment.get("first_frame_id")
        last = segment.get("last_frame_id")
        if first is None or last is None:
            continue
        pairs = [
            row
            for row in usable_pairs
            if first <= row["from_id"] and row["to_id"] <= last
        ]
        if not pairs:
            continue
        err_x = sum(row["translation_error_x_m"] for row in pairs)
        err_y = sum(row["translation_error_y_m"] for row in pairs)
        err_z = sum(row["translation_error_z_m"] for row in pairs)
        component_errors = {"x": abs(err_x), "y": abs(err_y), "z": abs(err_z)}
        segment["component_pair_count"] = len(pairs)
        segment["segment_translation_error_x_m"] = err_x
        segment["segment_translation_error_y_m"] = err_y
        segment["segment_translation_error_z_m"] = err_z
        segment["segment_translation_vector_error_m"] = math.sqrt(
            err_x * err_x + err_y * err_y + err_z * err_z
        )
        segment["dominant_translation_error_axis"] = max(
            component_errors,
            key=component_errors.get,
        )


def source_stats(pair_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in pair_rows:
        grouped[row["source"]].append(row)
    stats: dict[str, dict[str, Any]] = {}
    for source, rows in grouped.items():
        stats[source] = {
            "count": len(rows),
            "inliers": describe([r["inliers"] for r in rows if r["inliers"] is not None]),
            "inlier_ratio": describe(
                [r["inlier_ratio"] for r in rows if r["inlier_ratio"] is not None]
            ),
            "translation_magnitude_error_m": describe(
                [
                    r["translation_magnitude_error_m"]
                    for r in rows
                    if r["translation_magnitude_error_m"] is not None
                ]
            ),
            "rotation_error_deg": describe(
                [r["rotation_error_deg"] for r in rows if r["rotation_error_deg"] is not None]
            ),
        }
    return stats


def problem_tags(row: dict[str, Any]) -> str:
    tags: list[str] = []
    if row.get("source") in {"kabsch_fallback", "pnp_fallback"}:
        tags.append("fallback")
    if row.get("motion_scale_rescued") == "true":
        tags.append("scale_rescue")
    if row.get("translation_direction_rescued") == "true":
        tags.append("dir_rescue")
    if row.get("rotation_spike_rescued") == "true":
        tags.append("rot_rescue")
    if row.get("rotation_vector_rescued") == "true":
        tags.append("rotvec_rescue")
    inlier_ratio = row.get("inlier_ratio")
    if inlier_ratio is not None and inlier_ratio < 0.45:
        tags.append("weak_pnp")
    t_err = row.get("translation_magnitude_error_m")
    if t_err is not None and t_err > 0.5:
        tags.append("scale")
    y_err = row.get("abs_translation_error_y_m")
    if y_err is not None and y_err > 0.1:
        tags.append("vertical")
    r_err = row.get("rotation_error_deg")
    if r_err is not None and r_err > 1.0:
        tags.append("rotation")
    reproj = row.get("pnp_mean_reprojection_error_px")
    if reproj is not None and reproj > 3.0:
        tags.append("reprojection")
    return "|".join(tags)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for label, _ in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(key)) for _, key in columns) + " |")
    return "\n".join(lines)


def html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for label, _ in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{html.escape(fmt(row.get(key)))}</td>" for _, key in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def write_worst_pairs(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "from_id",
        "to_id",
        "source",
        "tags",
        "inliers",
        "inlier_ratio",
        "raw_translation_m",
        "raw_rotation_deg",
        "translation_m",
        "rotation_deg",
        "motion_scale_rescued",
        "translation_direction_rescued",
        "stereo_vertical_aligned",
        "rotation_spike_rescued",
        "rotation_vector_rescued",
        "translation_magnitude_error_m",
        "translation_vector_error_m",
        "translation_error_x_m",
        "translation_error_y_m",
        "translation_error_z_m",
        "rotation_error_deg",
        "pnp_mean_reprojection_error_px",
        "kabsch_mean_residual_m",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            out = {key: row.get(key) for key in columns}
            out["tags"] = problem_tags(row)
            writer.writerow(out)


def build_report(run_dir: Path, top: int) -> dict[str, Any]:
    run_dir = resolve_run_dir(run_dir)
    pair_rows = load_pair_rows(run_dir)
    segment_rows = load_segment_rows(run_dir)
    annotate_segments_with_component_errors(segment_rows, pair_rows)
    summary = parse_summary_txt(run_dir / "summary.txt")
    worst_t_pairs = top_by(pair_rows, "translation_magnitude_error_m", top)
    worst_y_pairs = top_by(pair_rows, "abs_translation_error_y_m", top)
    worst_r_pairs = top_by(pair_rows, "rotation_error_deg", top)
    weakest_pairs = sorted(
        [r for r in pair_rows if r.get("inlier_ratio") is not None],
        key=lambda row: row["inlier_ratio"],
    )[:top]
    worst_segments = top_by(segment_rows, "translational_error_percent", top)
    fallback_counts = Counter(row["source"] for row in pair_rows)

    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "kitti_public_lengths": load_kitti_summary(run_dir, "kitti_eval_public_lengths"),
        "kitti_100m": load_kitti_summary(run_dir, "kitti_eval_100m"),
        "pair_count": len(pair_rows),
        "fallback_counts": dict(fallback_counts),
        "source_stats": source_stats(pair_rows),
        "pair_metrics": {
            "translation_magnitude_error_m": describe(
                [
                    r["translation_magnitude_error_m"]
                    for r in pair_rows
                    if r["translation_magnitude_error_m"] is not None
                ]
            ),
            "translation_vector_error_m": describe(
                [
                    r["translation_vector_error_m"]
                    for r in pair_rows
                    if r["translation_vector_error_m"] is not None
                ]
            ),
            "translation_error_x_m": describe(
                [r["translation_error_x_m"] for r in pair_rows if r["translation_error_x_m"] is not None]
            ),
            "translation_error_y_m": describe(
                [r["translation_error_y_m"] for r in pair_rows if r["translation_error_y_m"] is not None]
            ),
            "translation_error_z_m": describe(
                [r["translation_error_z_m"] for r in pair_rows if r["translation_error_z_m"] is not None]
            ),
            "abs_translation_error_x_m": describe(
                [
                    r["abs_translation_error_x_m"]
                    for r in pair_rows
                    if r["abs_translation_error_x_m"] is not None
                ]
            ),
            "abs_translation_error_y_m": describe(
                [
                    r["abs_translation_error_y_m"]
                    for r in pair_rows
                    if r["abs_translation_error_y_m"] is not None
                ]
            ),
            "abs_translation_error_z_m": describe(
                [
                    r["abs_translation_error_z_m"]
                    for r in pair_rows
                    if r["abs_translation_error_z_m"] is not None
                ]
            ),
            "rotation_error_deg": describe(
                [r["rotation_error_deg"] for r in pair_rows if r["rotation_error_deg"] is not None]
            ),
            "inlier_ratio": describe(
                [r["inlier_ratio"] for r in pair_rows if r["inlier_ratio"] is not None]
            ),
        },
        "worst_translation_pairs": worst_t_pairs,
        "worst_vertical_pairs": worst_y_pairs,
        "worst_rotation_pairs": worst_r_pairs,
        "weakest_inlier_pairs": weakest_pairs,
        "kitti_segments": segment_rows,
        "worst_kitti_segments": worst_segments,
    }


def unique_worst_pairs(report: dict[str, Any]) -> list[dict[str, Any]]:
    worst_pairs = []
    seen = set()
    for row in (
        report["worst_translation_pairs"]
        + report["worst_vertical_pairs"]
        + report["worst_rotation_pairs"]
        + report["weakest_inlier_pairs"]
    ):
        key = (row.get("from_id"), row.get("to_id"), row.get("source"))
        if key not in seen:
            seen.add(key)
            worst_pairs.append(row)
    return worst_pairs


def write_single_report(report: dict[str, Any], out_dir: Path) -> list[Path]:
    summary = report["summary"]
    fallback_counts = report["fallback_counts"]
    json_path = out_dir / "slam_debug_summary.json"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    worst_pairs_path = out_dir / "slam_debug_worst_pairs.csv"
    write_worst_pairs(worst_pairs_path, unique_worst_pairs(report))

    pair_columns = [
        ("from", "from_id"),
        ("to", "to_id"),
        ("source", "source"),
        ("inliers", "inliers"),
        ("inlier ratio", "inlier_ratio"),
        ("raw t m", "raw_translation_m"),
        ("t err m", "translation_magnitude_error_m"),
        ("ty err m", "translation_error_y_m"),
        ("rot deg", "rotation_deg"),
        ("rot err deg", "rotation_error_deg"),
        ("rescues", "rescue_tags"),
        ("pnp reproj px", "pnp_mean_reprojection_error_px"),
    ]
    segment_columns = [
        ("first", "first_frame_id"),
        ("last", "last_frame_id"),
        ("len m", "length_m"),
        ("t rel %", "translational_error_percent"),
        ("seg err x m", "segment_translation_error_x_m"),
        ("seg err y m", "segment_translation_error_y_m"),
        ("seg err z m", "segment_translation_error_z_m"),
        ("axis", "dominant_translation_error_axis"),
        ("r deg/m", "rotational_error_deg_per_m"),
    ]

    md = [
        "# Visual SLAM Debug Report",
        "",
        f"run_dir: `{report['run_dir']}`",
        "",
        "## Headline",
        "",
        f"- pairs: {report['pair_count']}",
        f"- sources: {dict(fallback_counts)}",
        f"- VO ATE mean/RMSE/max: {summary.get('vo_ate_mean_m', '')} / "
        f"{summary.get('vo_ate_rmse_m', '')} / {summary.get('vo_ate_max_m', '')} m",
        f"- KITTI public t_rel/r_rel: "
        f"{report['kitti_public_lengths'].get('mean_translational_error_percent', '')} / "
        f"{report['kitti_public_lengths'].get('mean_rotational_error_deg_per_m', '')}",
        f"- relative mean/max translation-magnitude error: "
        f"{summary.get('relative_pose_mean_t_mag_err_m', '')} / "
        f"{summary.get('relative_pose_max_t_mag_err_m', '')} m",
        f"- relative mean/max |ty| error: "
        f"{summary.get('relative_pose_mean_abs_ty_err_m', '')} / "
        f"{summary.get('relative_pose_max_abs_ty_err_m', '')} m",
        f"- relative mean/max rotation error: "
        f"{summary.get('relative_pose_mean_rot_err_deg', '')} / "
        f"{summary.get('relative_pose_max_rot_err_deg', '')} deg",
        "",
        "## Worst Translation Pairs",
        "",
        markdown_table(report["worst_translation_pairs"], pair_columns),
        "",
        "## Worst Vertical Translation Pairs",
        "",
        markdown_table(report["worst_vertical_pairs"], pair_columns),
        "",
        "## Worst Rotation Pairs",
        "",
        markdown_table(report["worst_rotation_pairs"], pair_columns),
        "",
        "## Weakest Inlier-Ratio Pairs",
        "",
        markdown_table(report["weakest_inlier_pairs"], pair_columns),
        "",
        "## Worst KITTI Segments",
        "",
        markdown_table(report["worst_kitti_segments"], segment_columns),
        "",
    ]
    md_path = out_dir / "slam_debug_report.md"
    md_path.write_text("\n".join(md))

    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Visual SLAM Debug Report</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #202124; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
th:nth-child(3), td:nth-child(3) {{ text-align: left; }}
th {{ background: #f6f8fa; }}
code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
</style>
<h1>Visual SLAM Debug Report</h1>
<p>run_dir: <code>{html.escape(str(report['run_dir']))}</code></p>
<h2>Headline</h2>
<ul>
  <li>pairs: {report['pair_count']}</li>
  <li>sources: {html.escape(str(dict(fallback_counts)))}</li>
  <li>VO ATE mean/RMSE/max: {fmt(summary.get('vo_ate_mean_m'))} /
      {fmt(summary.get('vo_ate_rmse_m'))} / {fmt(summary.get('vo_ate_max_m'))} m</li>
  <li>KITTI public t_rel/r_rel: {fmt(report['kitti_public_lengths'].get('mean_translational_error_percent'))} /
      {fmt(report['kitti_public_lengths'].get('mean_rotational_error_deg_per_m'))}</li>
</ul>
<h2>Worst Translation Pairs</h2>
{html_table(report["worst_translation_pairs"], pair_columns)}
<h2>Worst Vertical Translation Pairs</h2>
{html_table(report["worst_vertical_pairs"], pair_columns)}
<h2>Worst Rotation Pairs</h2>
{html_table(report["worst_rotation_pairs"], pair_columns)}
<h2>Weakest Inlier-Ratio Pairs</h2>
{html_table(report["weakest_inlier_pairs"], pair_columns)}
<h2>Worst KITTI Segments</h2>
{html_table(report["worst_kitti_segments"], segment_columns)}
"""
    html_path = out_dir / "slam_debug_report.html"
    html_path.write_text(html_doc)

    return [json_path, md_path, html_path, worst_pairs_path]


def nested_get(data: dict[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def diff_metric(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    path: str,
    lower_is_better: bool = True,
) -> dict[str, Any]:
    base = nested_get(baseline, path)
    cand = nested_get(candidate, path)
    delta = cand - base if isinstance(base, (int, float)) and isinstance(cand, (int, float)) else None
    percent = (
        (delta / base * 100.0)
        if isinstance(delta, (int, float)) and isinstance(base, (int, float)) and base != 0
        else None
    )
    if delta is None:
        verdict = ""
    elif abs(delta) < 1.0e-12:
        verdict = "same"
    elif (delta < 0 and lower_is_better) or (delta > 0 and not lower_is_better):
        verdict = "better"
    else:
        verdict = "worse"
    return {
        "metric": path,
        "baseline": base,
        "candidate": cand,
        "delta": delta,
        "delta_percent": percent,
        "verdict": verdict,
    }


def pair_key(row: dict[str, Any]) -> tuple[Any, Any]:
    return (row.get("from_id"), row.get("to_id"))


def segment_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (row.get("first_frame_id"), row.get("last_frame_id"), row.get("length_m"))


def worst_pair_diff(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    table: str,
    metric: str,
) -> list[dict[str, Any]]:
    base_by_pair = {pair_key(row): row for row in baseline.get(table, [])}
    cand_by_pair = {pair_key(row): row for row in candidate.get(table, [])}
    pairs = set(base_by_pair) | set(cand_by_pair)
    rows: list[dict[str, Any]] = []
    for key in pairs:
        base_row = base_by_pair.get(key, {})
        cand_row = cand_by_pair.get(key, {})
        base_value = base_row.get(metric)
        cand_value = cand_row.get(metric)
        delta = (
            cand_value - base_value
            if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float))
            else None
        )
        rows.append(
            {
                "from_id": key[0],
                "to_id": key[1],
                "baseline_source": base_row.get("source"),
                "candidate_source": cand_row.get("source"),
                f"baseline_{metric}": base_value,
                f"candidate_{metric}": cand_value,
                "delta": delta,
                "status": "shared"
                if base_row and cand_row
                else ("new_candidate_worst" if cand_row else "removed_from_worst"),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(row["delta"]) if isinstance(row["delta"], (int, float)) else -1.0,
        reverse=True,
    )


def worst_segment_diff(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base_by_segment = {segment_key(row): row for row in baseline.get("kitti_segments", [])}
    cand_by_segment = {segment_key(row): row for row in candidate.get("kitti_segments", [])}
    worst_segments = {
        segment_key(row)
        for row in baseline.get("worst_kitti_segments", []) + candidate.get("worst_kitti_segments", [])
    }
    segments = worst_segments or (set(base_by_segment) | set(cand_by_segment))
    rows: list[dict[str, Any]] = []
    for key in segments:
        base_row = base_by_segment.get(key, {})
        cand_row = cand_by_segment.get(key, {})
        base_t = base_row.get("translational_error_percent")
        cand_t = cand_row.get("translational_error_percent")
        base_r = base_row.get("rotational_error_deg_per_m")
        cand_r = cand_row.get("rotational_error_deg_per_m")
        rows.append(
            {
                "first_frame_id": key[0],
                "last_frame_id": key[1],
                "length_m": key[2],
                "baseline_t_rel_percent": base_t,
                "candidate_t_rel_percent": cand_t,
                "delta_t_rel_percent": cand_t - base_t
                if isinstance(base_t, (int, float)) and isinstance(cand_t, (int, float))
                else None,
                "baseline_r_rel_deg_per_m": base_r,
                "candidate_r_rel_deg_per_m": cand_r,
                "delta_r_rel_deg_per_m": cand_r - base_r
                if isinstance(base_r, (int, float)) and isinstance(cand_r, (int, float))
                else None,
                "status": "shared"
                if base_row and cand_row
                else ("new_candidate_worst" if cand_row else "removed_from_worst"),
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(row["delta_t_rel_percent"])
        if isinstance(row["delta_t_rel_percent"], (int, float))
        else -1.0,
        reverse=True,
    )


def compare_reports(baseline: dict[str, Any], candidate: dict[str, Any], top: int) -> dict[str, Any]:
    metric_paths = [
        "summary.vo_ate_mean_m",
        "summary.vo_ate_rmse_m",
        "summary.vo_ate_max_m",
        "kitti_public_lengths.mean_translational_error_percent",
        "kitti_public_lengths.mean_rotational_error_deg_per_m",
        "kitti_public_lengths.max_translational_error_percent",
        "kitti_100m.mean_translational_error_percent",
        "kitti_100m.max_translational_error_percent",
        "summary.relative_pose_mean_t_mag_err_m",
        "summary.relative_pose_max_t_mag_err_m",
        "summary.relative_pose_mean_abs_ty_err_m",
        "summary.relative_pose_max_abs_ty_err_m",
        "summary.relative_pose_mean_rot_err_deg",
        "summary.relative_pose_max_rot_err_deg",
        "pair_metrics.inlier_ratio.mean",
        "pair_metrics.translation_magnitude_error_m.mean",
        "pair_metrics.translation_magnitude_error_m.max",
        "pair_metrics.translation_vector_error_m.mean",
        "pair_metrics.abs_translation_error_y_m.mean",
        "pair_metrics.abs_translation_error_y_m.max",
        "pair_metrics.rotation_error_deg.mean",
        "pair_metrics.rotation_error_deg.max",
    ]
    metrics = [
        diff_metric(
            baseline,
            candidate,
            path,
            lower_is_better=not path.endswith("inlier_ratio.mean"),
        )
        for path in metric_paths
    ]
    sources = sorted(set(baseline["fallback_counts"]) | set(candidate["fallback_counts"]))
    source_deltas = [
        {
            "source": source,
            "baseline": baseline["fallback_counts"].get(source, 0),
            "candidate": candidate["fallback_counts"].get(source, 0),
            "delta": candidate["fallback_counts"].get(source, 0)
            - baseline["fallback_counts"].get(source, 0),
        }
        for source in sources
    ]
    return {
        "baseline_run_dir": baseline["run_dir"],
        "candidate_run_dir": candidate["run_dir"],
        "metrics": metrics,
        "source_deltas": source_deltas,
        "worst_translation_pair_deltas": worst_pair_diff(
            baseline, candidate, "worst_translation_pairs", "translation_magnitude_error_m"
        )[:top],
        "worst_vertical_pair_deltas": worst_pair_diff(
            baseline, candidate, "worst_vertical_pairs", "abs_translation_error_y_m"
        )[:top],
        "worst_rotation_pair_deltas": worst_pair_diff(
            baseline, candidate, "worst_rotation_pairs", "rotation_error_deg"
        )[:top],
        "worst_kitti_segment_deltas": worst_segment_diff(baseline, candidate)[:top],
    }


def write_compare_csv(path: Path, comparison: dict[str, Any]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "baseline", "candidate", "delta", "delta_percent", "verdict"],
        )
        writer.writeheader()
        writer.writerows(comparison["metrics"])


def write_compare_report(comparison: dict[str, Any], out_dir: Path) -> list[Path]:
    json_path = out_dir / "slam_debug_compare.json"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    csv_path = out_dir / "slam_debug_compare_metrics.csv"
    write_compare_csv(csv_path, comparison)

    metric_columns = [
        ("metric", "metric"),
        ("baseline", "baseline"),
        ("candidate", "candidate"),
        ("delta", "delta"),
        ("delta %", "delta_percent"),
        ("verdict", "verdict"),
    ]
    source_columns = [
        ("source", "source"),
        ("baseline", "baseline"),
        ("candidate", "candidate"),
        ("delta", "delta"),
    ]
    pair_delta_columns = [
        ("from", "from_id"),
        ("to", "to_id"),
        ("base src", "baseline_source"),
        ("cand src", "candidate_source"),
        ("delta", "delta"),
        ("status", "status"),
    ]
    segment_delta_columns = [
        ("first", "first_frame_id"),
        ("last", "last_frame_id"),
        ("len m", "length_m"),
        ("base t %", "baseline_t_rel_percent"),
        ("cand t %", "candidate_t_rel_percent"),
        ("delta t %", "delta_t_rel_percent"),
        ("status", "status"),
    ]
    md = [
        "# Visual SLAM Debug Comparison",
        "",
        f"baseline: `{comparison['baseline_run_dir']}`",
        f"candidate: `{comparison['candidate_run_dir']}`",
        "",
        "## Metric Deltas",
        "",
        markdown_table(comparison["metrics"], metric_columns),
        "",
        "## Source Deltas",
        "",
        markdown_table(comparison["source_deltas"], source_columns),
        "",
        "## Worst Translation Pair Deltas",
        "",
        markdown_table(comparison["worst_translation_pair_deltas"], pair_delta_columns),
        "",
        "## Worst Vertical Pair Deltas",
        "",
        markdown_table(comparison["worst_vertical_pair_deltas"], pair_delta_columns),
        "",
        "## Worst Rotation Pair Deltas",
        "",
        markdown_table(comparison["worst_rotation_pair_deltas"], pair_delta_columns),
        "",
        "## Worst KITTI Segment Deltas",
        "",
        markdown_table(comparison["worst_kitti_segment_deltas"], segment_delta_columns),
        "",
    ]
    md_path = out_dir / "slam_debug_compare.md"
    md_path.write_text("\n".join(md))

    html_doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Visual SLAM Debug Comparison</title>
<style>
body {{ font: 14px/1.45 system-ui, sans-serif; margin: 24px; color: #202124; }}
table {{ border-collapse: collapse; margin: 12px 0 24px; width: 100%; }}
th, td {{ border: 1px solid #d0d7de; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f6f8fa; }}
code {{ background: #f6f8fa; padding: 2px 4px; border-radius: 4px; }}
</style>
<h1>Visual SLAM Debug Comparison</h1>
<p>baseline: <code>{html.escape(comparison['baseline_run_dir'])}</code></p>
<p>candidate: <code>{html.escape(comparison['candidate_run_dir'])}</code></p>
<h2>Metric Deltas</h2>
{html_table(comparison["metrics"], metric_columns)}
<h2>Source Deltas</h2>
{html_table(comparison["source_deltas"], source_columns)}
<h2>Worst Translation Pair Deltas</h2>
{html_table(comparison["worst_translation_pair_deltas"], pair_delta_columns)}
<h2>Worst Vertical Pair Deltas</h2>
{html_table(comparison["worst_vertical_pair_deltas"], pair_delta_columns)}
<h2>Worst Rotation Pair Deltas</h2>
{html_table(comparison["worst_rotation_pair_deltas"], pair_delta_columns)}
<h2>Worst KITTI Segment Deltas</h2>
{html_table(comparison["worst_kitti_segment_deltas"], segment_delta_columns)}
"""
    html_path = out_dir / "slam_debug_compare.html"
    html_path.write_text(html_doc)
    return [json_path, csv_path, md_path, html_path]


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    out_dir = (args.out_dir or (run_dir / "slam_debug")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report = build_report(run_dir, args.top)
    paths = write_single_report(report, out_dir)
    if args.compare is not None:
        baseline_dir = args.compare.expanduser().resolve()
        baseline_report = build_report(baseline_dir, args.top)
        comparison = compare_reports(baseline_report, report, args.top)
        paths.extend(write_compare_report(comparison, out_dir))

    for path in paths:
        print(f"# Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
