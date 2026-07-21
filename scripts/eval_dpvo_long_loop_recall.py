#!/usr/bin/env python3
"""Stage-1 measurement harness for DPVO long-range loop retrieval recall.

Context: `docs/visual_slam_sequential_sfm_plan.md` section "A3 -- Sound
long-range loop closure", stage 1 ("retrieval recall"); `docs/dpvo_droid_port_plan.md`
M11/M12 sections describe the retrieval pipeline (`pipelines/slam/src/dpvo_long_loop.rs`)
and its known gap: the tightest GT revisit on MH_01, processed-frame pair
`i=42, j=456` (GT distance `0.160 m`, per the M11 "GT revisit pre-check"), is
never surfaced as a retrieval candidate on the acceptance runs measured so far.

This script scores a demo run's `long_loop_candidates.csv` (written by
`examples/euroc_dpvo_vo_demo.rs` when `--long-loop` is on -- see that file's
module doc, "M11 open item 2 instrumentation") against EuRoC ground truth,
independent of the Rust engine.

Facts verified directly from source (not assumed) before writing this script:

* **Arrival -> source image index.** `examples/euroc_dpvo_vo_demo.rs`'s main
  loop builds its processed-frame list as
  ``dataset.cam0_images.iter().step_by(args.stride.max(1)).take(frame_cap)``
  (`frame_cap = usize::MAX` when `--max-frames 0`, else `--max-frames`).
  `crates/io/src/euroc.rs::read_euroc_image_manifest` pushes `cam0/data.csv`
  rows in FILE order with no re-sort. So processed arrival index `a` (0-based)
  is exactly source image row `a * stride` in `cam0/data.csv`, and its
  timestamp is that row's `#timestamp [ns]` field. This script reproduces that
  mapping directly (`--stride`, `--max-frames`).
* **Index gating defaults**, from `pipelines/slam/src/dpvo_long_loop.rs`'s
  `DpvoLongLoopConfig::default()`: `min_temporal_gap = 150` (the "long-range,
  not proximity" knob -- a candidate `f` is only considered for query
  `current_arrival` when `current_arrival - f.arrival_index >= min_temporal_gap`,
  see `DpvoLongLoopIndex::query_candidates`), `top_k = 3`, `query_frequency = 40`
  (re-check throttle in `DpvoLongLoopIndex::due`). This script's `--min-gap`
  default (`150`) mirrors `min_temporal_gap` so genuine-revisit labelling only
  counts pairs retrieval was structurally ALLOWED to return (per this task's
  own instruction) -- a pair with a smaller gap could never appear in the CSV
  regardless of how good retrieval is, so it must not be scored as a miss.
  The three archived runs this script was run against were empirically
  cross-checked against these defaults: every row's own `gap` column is
  `>= 150`, and every query's max `rank` is `2` (`top_k = 3`), consistent with
  the CLI never overriding `--ll-min-temporal-gap` / `--ll-top-k`.

GT pose interpolation: EuRoC ground truth
(`mav0/state_groundtruth_estimate0/data.csv`) is ~200 Hz; cam0 is 20 Hz, so
the nearest GT sample to any image timestamp is normally within a couple of
milliseconds. This script uses nearest-neighbour GT (position AND
orientation) whenever the nearest sample is within `--gt-tol-ns` (default
5 ms) of the image timestamp; otherwise it falls back to LINEAR
interpolation for position between the two bracketing samples and
nearest-neighbour (no slerp) for orientation -- slerp was not implemented
since the >5ms fallback path is not expected to be exercised on EuRoC's dense
200 Hz ground truth, and stating "nearest orientation" honestly is better
than a silently wrong interpolation.

Quaternion convention: EuRoC's `q_RS` is documented as "rotation from the
sensor(body) frame S to the reference/world frame R", i.e. `R_WB`. The
`quaternion_matrix` helper here is the same formula already used by
`scripts/evaluate_euroc_trajectory.py::quaternion_matrix` (verified to match
by inspection), just re-derived locally to keep this script standalone.
`cam0/sensor.yaml`'s `T_BS` is "transformation from sensor(camera) frame S to
body frame B" (per its own comment, "Sensor extrinsics wrt. the body-frame"),
so its rotation block is `R_BC` (camera-to-body). The camera's own optical
axis (`+z` in the camera frame) in world coordinates is therefore
`R_WB @ R_BC @ [0, 0, 1]`, per this task's own instruction.

PyYAML is not assumed to be installed; `T_BS` is parsed out of the small,
fixed-shape `sensor.yaml` block by hand (find the `T_BS:` -> `data: [...]`
span and parse the 16 floats), not with a YAML parser.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# `pipelines/slam/src/dpvo_long_loop.rs`'s `DpvoLongLoopConfig::default()`.
DEFAULT_MIN_TEMPORAL_GAP = 150
DEFAULT_TOP_K = 3
DEFAULT_QUERY_FREQUENCY = 40

CANDIDATE_REQUIRED_COLUMNS = {
    "query_arrival",
    "rank",
    "candidate_arrival",
    "gap",
    "similarity",
    "accepted",
}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def quaternion_wxyz_to_matrix(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Rotation matrix for a Hamilton quaternion given as (w, x, y, z).

    Same formula as `scripts/evaluate_euroc_trajectory.py::quaternion_matrix`
    (there parameterised as `(x, y, z, w)`), re-derived here to keep this
    script standalone. For EuRoC's `q_RS` this returns `R_WB`.
    """
    q = np.asarray([qw, qx, qy, qz], dtype=float)
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("invalid zero/non-finite quaternion")
    w, x, y, z = q / norm
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def parse_cam0_t_bs(sensor_yaml_path: Path) -> np.ndarray:
    """Parse the 4x4 `T_BS` matrix out of a cam0 `sensor.yaml` by hand.

    Deliberately does not require PyYAML: finds the `T_BS:` block, then its
    `data: [...]` list, and parses the 16 floats directly (handles the
    values spanning multiple lines, as EuRoC's own `sensor.yaml` files do).
    """
    text = sensor_yaml_path.read_text(encoding="utf-8")
    t_bs_idx = text.find("T_BS:")
    if t_bs_idx < 0:
        raise ValueError(f"no 'T_BS:' block found in {sensor_yaml_path}")
    sub = text[t_bs_idx:]
    data_idx = sub.find("data:")
    if data_idx < 0:
        raise ValueError(f"no 'data:' key inside 'T_BS:' block in {sensor_yaml_path}")
    start = sub.find("[", data_idx)
    end = sub.find("]", start)
    if start < 0 or end < 0:
        raise ValueError(f"malformed 'data: [...]' list in {sensor_yaml_path}")
    raw_values = sub[start + 1 : end].replace("\n", " ")
    values = [float(token) for token in raw_values.split(",") if token.strip()]
    if len(values) != 16:
        raise ValueError(
            f"expected 16 floats in T_BS data, got {len(values)} in {sensor_yaml_path}"
        )
    return np.asarray(values, dtype=float).reshape(4, 4)


# ---------------------------------------------------------------------------
# EuRoC loading
# ---------------------------------------------------------------------------


@dataclass
class GroundTruth:
    timestamps_ns: list[int]
    positions: np.ndarray  # (N, 3)
    quats_wxyz: np.ndarray  # (N, 4)


def load_ground_truth(path: Path) -> GroundTruth:
    timestamps: list[int] = []
    positions: list[tuple[float, float, float]] = []
    quats: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            if len(row) < 8:
                raise ValueError("EuRoC ground-truth row has fewer than 8 fields")
            ts = int(row[0].strip().split(".", 1)[0])
            px, py, pz = (float(v) for v in row[1:4])
            qw, qx, qy, qz = (float(v) for v in row[4:8])
            timestamps.append(ts)
            positions.append((px, py, pz))
            quats.append((qw, qx, qy, qz))
    if not timestamps:
        raise ValueError(f"ground-truth CSV is empty: {path}")
    order = np.argsort(timestamps)
    timestamps_sorted = [timestamps[i] for i in order]
    positions_arr = np.asarray(positions, dtype=float)[order]
    quats_arr = np.asarray(quats, dtype=float)[order]
    return GroundTruth(timestamps_sorted, positions_arr, quats_arr)


def load_cam0_timestamps(path: Path) -> list[int]:
    """Read `cam0/data.csv` timestamps in FILE order (matches `read_euroc_image_manifest`)."""
    timestamps: list[int] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for row in csv.reader(stream):
            if not row or row[0].lstrip().startswith("#"):
                continue
            timestamps.append(int(row[0].strip().split(".", 1)[0]))
    if not timestamps:
        raise ValueError(f"cam0 data CSV is empty: {path}")
    return timestamps


@dataclass
class CandidateRow:
    query_arrival: int
    rank: int
    candidate_arrival: int
    gap: int
    similarity: float
    accepted: bool


def load_candidates(path: Path) -> list[CandidateRow]:
    """Load `long_loop_candidates.csv`.

    Tolerant of the header both WITH and WITHOUT the optional
    `rotation_disagreement_deg` trailing column (both variants were found
    among the archived acceptance runs this script was pointed at); only the
    required columns are used.
    """
    rows: list[CandidateRow] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or [])
        missing = CANDIDATE_REQUIRED_COLUMNS - fields
        if missing:
            raise ValueError(
                "candidates CSV is missing columns: {}".format(", ".join(sorted(missing)))
            )
        for row in reader:
            rows.append(
                CandidateRow(
                    query_arrival=int(row["query_arrival"]),
                    rank=int(row["rank"]),
                    candidate_arrival=int(row["candidate_arrival"]),
                    gap=int(row["gap"]),
                    similarity=float(row["similarity"]),
                    accepted=row["accepted"].strip().lower() in ("true", "1"),
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Arrival -> GT pose mapping
# ---------------------------------------------------------------------------


def num_arrivals(image_count: int, stride: int, max_frames: int) -> int:
    """Mirror `.step_by(stride).take(frame_cap)`'s resulting length."""
    stride = max(stride, 1)
    available = (image_count + stride - 1) // stride
    if max_frames and max_frames > 0:
        return min(max_frames, available)
    return available


def interpolate_gt_pose(
    timestamp_ns: int,
    gt: GroundTruth,
    tol_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(position, quat_wxyz)` at `timestamp_ns`.

    Nearest-neighbour (position AND orientation) if the nearest GT sample is
    within `tol_ns`; otherwise linear interpolation for position between the
    two bracketing samples and nearest-neighbour for orientation (no slerp
    -- see module docstring).
    """
    stamps = gt.timestamps_ns
    n = len(stamps)
    idx = bisect.bisect_left(stamps, timestamp_ns)
    if idx <= 0:
        return gt.positions[0], gt.quats_wxyz[0]
    if idx >= n:
        return gt.positions[-1], gt.quats_wxyz[-1]
    lo, hi = idx - 1, idx
    d_lo = abs(timestamp_ns - stamps[lo])
    d_hi = abs(stamps[hi] - timestamp_ns)
    nearest = lo if d_lo <= d_hi else hi
    nearest_delta = min(d_lo, d_hi)
    if nearest_delta <= tol_ns:
        return gt.positions[nearest], gt.quats_wxyz[nearest]
    span = stamps[hi] - stamps[lo]
    alpha = 0.0 if span == 0 else (timestamp_ns - stamps[lo]) / span
    position = gt.positions[lo] * (1.0 - alpha) + gt.positions[hi] * alpha
    quat = gt.quats_wxyz[nearest]
    return position, quat


def build_arrival_poses(
    n_arrivals: int,
    stride: int,
    cam0_timestamps: list[int],
    gt: GroundTruth,
    r_bc: np.ndarray,
    tol_ns: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return `(positions[N,3], camera_axes_world[N,3])`, one row per arrival."""
    positions = np.zeros((n_arrivals, 3), dtype=float)
    axes = np.zeros((n_arrivals, 3), dtype=float)
    axis_cam = np.asarray([0.0, 0.0, 1.0])
    for arrival in range(n_arrivals):
        image_index = arrival * stride
        if image_index >= len(cam0_timestamps):
            raise ValueError(
                f"arrival {arrival} maps to cam0 image index {image_index}, "
                f"but only {len(cam0_timestamps)} cam0 images are available"
            )
        ts = cam0_timestamps[image_index]
        position, quat_wxyz = interpolate_gt_pose(ts, gt, tol_ns)
        r_wb = quaternion_wxyz_to_matrix(*quat_wxyz)
        positions[arrival] = position
        axes[arrival] = r_wb @ (r_bc @ axis_cam)
    return positions, axes


# ---------------------------------------------------------------------------
# Labelling + recall
# ---------------------------------------------------------------------------


def label_revisit_pairs(
    positions: np.ndarray,
    axes: np.ndarray,
    min_gap: int,
    radius_m: float,
    max_angle_deg: float,
) -> np.ndarray:
    """Boolean `(N, N)` matrix; `mask[i, j]` true iff `(i, j)` is a genuine revisit.

    `i` is the older ("candidate") arrival, `j` the newer ("query") arrival:
    `j - i >= min_gap`, GT position distance `< radius_m`, and camera
    optical-axis angular difference `< max_angle_deg`.
    """
    n = positions.shape[0]
    idx = np.arange(n)
    gap = idx[None, :] - idx[:, None]  # gap[i, j] = j - i
    diff = positions[:, None, :] - positions[None, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    cos_angle = np.clip(np.sum(axes[:, None, :] * axes[None, :, :], axis=-1), -1.0, 1.0)
    angle_deg = np.degrees(np.arccos(cos_angle))
    mask = (gap >= min_gap) & (dist < radius_m) & (angle_deg < max_angle_deg)
    return mask


@dataclass
class RecallAtK:
    k: int
    hits: int
    denominator: int

    @property
    def recall(self) -> float | None:
        if self.denominator == 0:
            return None
        return self.hits / self.denominator


@dataclass
class SimilarityStats:
    count: int
    mean: float | None
    median: float | None


def similarity_stats(values: list[float]) -> SimilarityStats:
    if not values:
        return SimilarityStats(count=0, mean=None, median=None)
    return SimilarityStats(
        count=len(values),
        mean=float(statistics.mean(values)),
        median=float(statistics.median(values)),
    )


@dataclass
class RunEvaluation:
    radius_m: float
    max_angle_deg: float
    min_gap: int
    near_window: int
    labelled_pairs_total: int
    recall_at_k: list[RecallAtK]
    denom_queries: list[int]
    missed_queries: list[int]
    opportunity_coverage_missed_fraction: float | None
    similarity_all: SimilarityStats
    similarity_labelled_hit: SimilarityStats


def evaluate_run(
    rows: list[CandidateRow],
    mask: np.ndarray,
    min_gap: int,
    radius_m: float,
    max_angle_deg: float,
    near_window: int,
    k_values: list[int],
) -> RunEvaluation:
    candidates_by_query: dict[int, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        candidates_by_query[row.query_arrival].append(row)
    for query_rows in candidates_by_query.values():
        query_rows.sort(key=lambda r: r.rank)

    n = mask.shape[0]
    partner_map: dict[int, np.ndarray] = {}
    for j in range(n):
        partners = np.nonzero(mask[:, j])[0]
        if partners.size > 0:
            partner_map[j] = partners

    issued_queries = set(candidates_by_query.keys())
    denom_queries = sorted(set(partner_map) & issued_queries)
    missed_queries = sorted(set(partner_map) - issued_queries)

    recall_at_k: list[RecallAtK] = []
    for k in k_values:
        hits = 0
        for j in denom_queries:
            partners = partner_map[j]
            top_rows = [r for r in candidates_by_query[j] if r.rank < k]
            hit = any(
                np.any(np.abs(partners - r.candidate_arrival) <= near_window) for r in top_rows
            )
            if hit:
                hits += 1
        recall_at_k.append(RecallAtK(k=k, hits=hits, denominator=len(denom_queries)))

    opportunity_coverage_missed_fraction = (
        len(missed_queries) / len(partner_map) if partner_map else None
    )

    all_sims = [r.similarity for r in rows]
    hit_sims: list[float] = []
    for r in rows:
        partners = partner_map.get(r.query_arrival)
        if partners is not None and np.any(np.abs(partners - r.candidate_arrival) <= near_window):
            hit_sims.append(r.similarity)

    return RunEvaluation(
        radius_m=radius_m,
        max_angle_deg=max_angle_deg,
        min_gap=min_gap,
        near_window=near_window,
        labelled_pairs_total=int(mask.sum()),
        recall_at_k=recall_at_k,
        denom_queries=denom_queries,
        missed_queries=missed_queries,
        opportunity_coverage_missed_fraction=opportunity_coverage_missed_fraction,
        similarity_all=similarity_stats(all_sims),
        similarity_labelled_hit=similarity_stats(hit_sims),
    )


@dataclass
class DiagnosticRow:
    query_arrival: int
    rank: int
    candidate_arrival: int
    gap: int
    similarity: float
    accepted: bool
    near_target_candidate: bool


@dataclass
class DiagnosticReport:
    target_i: int
    target_j: int
    near_window: int
    query_arrivals_near_target: list[int]
    rows: list[DiagnosticRow]
    found_near_target: bool


def diagnose_tightest_pair(
    rows: list[CandidateRow],
    target_i: int,
    target_j: int,
    near_window: int,
) -> DiagnosticReport:
    candidates_by_query: dict[int, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        candidates_by_query[row.query_arrival].append(row)

    near_queries = sorted(j for j in candidates_by_query if abs(j - target_j) <= near_window)
    diag_rows: list[DiagnosticRow] = []
    found = False
    for j in near_queries:
        for r in sorted(candidates_by_query[j], key=lambda r: r.rank):
            near_target = abs(r.candidate_arrival - target_i) <= near_window
            found = found or near_target
            diag_rows.append(
                DiagnosticRow(
                    query_arrival=r.query_arrival,
                    rank=r.rank,
                    candidate_arrival=r.candidate_arrival,
                    gap=r.gap,
                    similarity=r.similarity,
                    accepted=r.accepted,
                    near_target_candidate=near_target,
                )
            )
    return DiagnosticReport(
        target_i=target_i,
        target_j=target_j,
        near_window=near_window,
        query_arrivals_near_target=near_queries,
        rows=diag_rows,
        found_near_target=found,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_report(
    candidates_csv: Path,
    gt_dir: Path,
    stride: int,
    max_frames: int,
    min_gap: int,
    radius_m: float,
    radius_secondary_m: float,
    max_angle_deg: float,
    near_window: int,
    gt_tol_ns: int,
    k_values: list[int],
    diagnostic_i: int,
    diagnostic_j: int,
) -> dict:
    rows = load_candidates(candidates_csv)

    gt_csv = gt_dir / "state_groundtruth_estimate0" / "data.csv"
    cam0_csv = gt_dir / "cam0" / "data.csv"
    cam0_sensor_yaml = gt_dir / "cam0" / "sensor.yaml"

    gt = load_ground_truth(gt_csv)
    cam0_timestamps = load_cam0_timestamps(cam0_csv)
    t_bs = parse_cam0_t_bs(cam0_sensor_yaml)
    r_bc = t_bs[:3, :3]

    n_arrivals = num_arrivals(len(cam0_timestamps), stride, max_frames)
    positions, axes = build_arrival_poses(n_arrivals, stride, cam0_timestamps, gt, r_bc, gt_tol_ns)

    observed_min_gap = min((r.gap for r in rows), default=None)
    observed_max_rank = max((r.rank for r in rows), default=None)
    observed_top_k = observed_max_rank + 1 if observed_max_rank is not None else None
    query_arrivals_sorted = sorted({r.query_arrival for r in rows})
    observed_query_frequency = None
    if len(query_arrivals_sorted) > 1:
        deltas = [
            b - a for a, b in zip(query_arrivals_sorted, query_arrivals_sorted[1:])
        ]
        observed_query_frequency = min(deltas)

    evaluations = []
    for radius in (radius_m, radius_secondary_m):
        mask = label_revisit_pairs(positions, axes, min_gap, radius, max_angle_deg)
        evaluations.append(
            evaluate_run(rows, mask, min_gap, radius, max_angle_deg, near_window, k_values)
        )

    diagnostic = diagnose_tightest_pair(rows, diagnostic_i, diagnostic_j, near_window)

    report = {
        "candidates_csv": str(candidates_csv),
        "gt_dir": str(gt_dir),
        "config": {
            "stride": stride,
            "max_frames": max_frames,
            "n_arrivals": n_arrivals,
            "min_gap": min_gap,
            "radius_m_primary": radius_m,
            "radius_m_secondary": radius_secondary_m,
            "max_angle_deg": max_angle_deg,
            "near_window": near_window,
            "gt_tol_ns": gt_tol_ns,
            "k_values": k_values,
            "index_gating_defaults_source": "pipelines/slam/src/dpvo_long_loop.rs::DpvoLongLoopConfig::default()",
        },
        "csv_sanity": {
            "row_count": len(rows),
            "distinct_query_arrivals": len(query_arrivals_sorted),
            "observed_min_gap": observed_min_gap,
            "observed_top_k": observed_top_k,
            "observed_min_query_spacing": observed_query_frequency,
            "expected_min_gap_default": DEFAULT_MIN_TEMPORAL_GAP,
            "expected_top_k_default": DEFAULT_TOP_K,
            "expected_query_frequency_default": DEFAULT_QUERY_FREQUENCY,
        },
        "evaluations": [
            {
                "radius_m": ev.radius_m,
                "max_angle_deg": ev.max_angle_deg,
                "min_gap": ev.min_gap,
                "near_window": ev.near_window,
                "labelled_pairs_total": ev.labelled_pairs_total,
                "recall_at_k": [
                    {"k": r.k, "hits": r.hits, "denominator": r.denominator, "recall": r.recall}
                    for r in ev.recall_at_k
                ],
                "opportunity_coverage_missed_fraction": ev.opportunity_coverage_missed_fraction,
                "opportunity_coverage_denominator": len(ev.denom_queries) + len(ev.missed_queries),
                "missed_query_arrivals": ev.missed_queries,
                "similarity_all": vars(ev.similarity_all),
                "similarity_labelled_hit": vars(ev.similarity_labelled_hit),
            }
            for ev in evaluations
        ],
        "diagnostic_tightest_pair": {
            "target_i": diagnostic.target_i,
            "target_j": diagnostic.target_j,
            "near_window": diagnostic.near_window,
            "query_arrivals_near_target": diagnostic.query_arrivals_near_target,
            "found_near_target": diagnostic.found_near_target,
            "rows": [vars(r) for r in diagnostic.rows],
        },
    }
    return report


def format_text_report(report: dict) -> str:
    lines: list[str] = []
    lines.append(f"candidates_csv: {report['candidates_csv']}")
    lines.append(f"gt_dir: {report['gt_dir']}")
    cfg = report["config"]
    lines.append(
        f"stride={cfg['stride']} max_frames={cfg['max_frames']} n_arrivals={cfg['n_arrivals']} "
        f"min_gap={cfg['min_gap']} max_angle_deg={cfg['max_angle_deg']} "
        f"near_window={cfg['near_window']} gt_tol_ns={cfg['gt_tol_ns']}"
    )
    sanity = report["csv_sanity"]
    lines.append(
        f"csv rows={sanity['row_count']} distinct_query_arrivals={sanity['distinct_query_arrivals']} "
        f"observed_min_gap={sanity['observed_min_gap']} (expect >= {sanity['expected_min_gap_default']}) "
        f"observed_top_k={sanity['observed_top_k']} (expect == {sanity['expected_top_k_default']}) "
        f"observed_min_query_spacing={sanity['observed_min_query_spacing']} "
        f"(expect >= {sanity['expected_query_frequency_default']})"
    )
    lines.append("")
    for ev in report["evaluations"]:
        lines.append(f"=== radius={ev['radius_m']} m, max_angle={ev['max_angle_deg']} deg ===")
        lines.append(f"labelled_pairs_total={ev['labelled_pairs_total']}")
        for row in ev["recall_at_k"]:
            recall = row["recall"]
            recall_str = f"{recall:.4f}" if recall is not None else "n/a"
            lines.append(
                f"  recall@{row['k']}: {row['hits']}/{row['denominator']} = {recall_str}"
            )
        occ = ev["opportunity_coverage_missed_fraction"]
        occ_str = f"{occ:.4f}" if occ is not None else "n/a"
        lines.append(
            f"  opportunity_coverage_missed_fraction: {occ_str} "
            f"(missed {len(ev['missed_query_arrivals'])} / {ev['opportunity_coverage_denominator']} "
            "labelled-revisit query arrivals never issued a query)"
        )
        sim_all = ev["similarity_all"]
        sim_hit = ev["similarity_labelled_hit"]
        lines.append(
            f"  similarity (all candidates): n={sim_all['count']} mean={sim_all['mean']} "
            f"median={sim_all['median']}"
        )
        lines.append(
            f"  similarity (labelled-hit candidates): n={sim_hit['count']} mean={sim_hit['mean']} "
            f"median={sim_hit['median']}"
        )
        lines.append("")

    diag = report["diagnostic_tightest_pair"]
    lines.append(
        f"=== diagnostic: tightest GT pair (i={diag['target_i']}, j={diag['target_j']}), "
        f"near_window={diag['near_window']} ==="
    )
    lines.append(f"query_arrivals_near_target={diag['query_arrivals_near_target']}")
    if not diag["rows"]:
        lines.append(
            "  no query arrivals were issued within near_window of the target j -- "
            "retrieval never even queried near this revisit (a cadence miss, not a ranking miss)."
        )
    else:
        for row in diag["rows"]:
            flag = " <== NEAR TARGET i" if row["near_target_candidate"] else ""
            lines.append(
                f"  query={row['query_arrival']} rank={row['rank']} "
                f"candidate={row['candidate_arrival']} gap={row['gap']} "
                f"similarity={row['similarity']:.6f} accepted={row['accepted']}{flag}"
            )
    lines.append(f"found_near_target: {diag['found_near_target']}")
    return "\n".join(lines)


def parse_k_values(raw: str) -> list[int]:
    values = [int(token.strip()) for token in raw.split(",") if token.strip()]
    if not values:
        raise argparse.ArgumentTypeError("--k-values must contain at least one integer")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidates-csv", type=Path, required=True, help="path to long_loop_candidates.csv"
    )
    parser.add_argument(
        "--gt-dir",
        type=Path,
        required=True,
        help="EuRoC mav0 directory (contains cam0/, state_groundtruth_estimate0/)",
    )
    parser.add_argument("--stride", type=int, default=2, help="must match the demo's own --stride")
    parser.add_argument(
        "--max-frames",
        type=int,
        required=True,
        help="must match the demo's own --max-frames (0 = unbounded)",
    )
    parser.add_argument(
        "--min-gap",
        type=int,
        default=DEFAULT_MIN_TEMPORAL_GAP,
        help=f"DpvoLongLoopConfig::min_temporal_gap used by the run (default {DEFAULT_MIN_TEMPORAL_GAP})",
    )
    parser.add_argument("--radius", type=float, default=1.0, help="primary GT position radius (m)")
    parser.add_argument(
        "--radius-secondary", type=float, default=0.5, help="secondary GT position radius (m), always also reported"
    )
    parser.add_argument("--max-angle-deg", type=float, default=30.0)
    parser.add_argument("--near-window", type=int, default=5)
    parser.add_argument("--gt-tol-ns", type=int, default=5_000_000)
    parser.add_argument("--k-values", type=parse_k_values, default=parse_k_values("1,3,5,10"))
    parser.add_argument("--diagnostic-i", type=int, default=42)
    parser.add_argument("--diagnostic-j", type=int, default=456)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    report = build_report(
        candidates_csv=args.candidates_csv,
        gt_dir=args.gt_dir,
        stride=args.stride,
        max_frames=args.max_frames,
        min_gap=args.min_gap,
        radius_m=args.radius,
        radius_secondary_m=args.radius_secondary,
        max_angle_deg=args.max_angle_deg,
        near_window=args.near_window,
        gt_tol_ns=args.gt_tol_ns,
        k_values=args.k_values,
        diagnostic_i=args.diagnostic_i,
        diagnostic_j=args.diagnostic_j,
    )

    print(format_text_report(report))

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
