"""Tests for `scripts/eval_dpvo_long_loop_recall.py`.

Follows the module-loading style of `tests/test_evaluate_euroc_trajectory.py`
(load the script by path rather than requiring `scripts/` on `sys.path`), but
uses plain pytest functions + `tmp_path` fixtures per this test's own task
brief.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_dpvo_long_loop_recall.py"
SPEC = importlib.util.spec_from_file_location("eval_dpvo_long_loop_recall", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE  # dataclasses' typing introspection needs this registered
SPEC.loader.exec_module(MODULE)


def rotation_about_x(angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


# ---------------------------------------------------------------------------
# quaternion_wxyz_to_matrix
# ---------------------------------------------------------------------------


def test_quaternion_identity_is_identity_matrix() -> None:
    r = MODULE.quaternion_wxyz_to_matrix(1.0, 0.0, 0.0, 0.0)
    np.testing.assert_allclose(r, np.eye(3), atol=1e-12)


def test_quaternion_unnormalized_still_gives_orthonormal_matrix() -> None:
    r = MODULE.quaternion_wxyz_to_matrix(2.0, 0.0, 0.0, 0.0)  # unnormalized identity
    np.testing.assert_allclose(r, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert abs(np.linalg.det(r) - 1.0) < 1e-9


def test_quaternion_zero_raises() -> None:
    with pytest.raises(ValueError):
        MODULE.quaternion_wxyz_to_matrix(0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# parse_cam0_t_bs
# ---------------------------------------------------------------------------


def test_parse_cam0_t_bs_reads_16_values(tmp_path: Path) -> None:
    sensor_yaml = tmp_path / "sensor.yaml"
    sensor_yaml.write_text(
        "sensor_type: camera\n"
        "T_BS:\n"
        "  cols: 4\n"
        "  rows: 4\n"
        "  data: [1.0, 0.0, 0.0, 0.1,\n"
        "         0.0, 1.0, 0.0, 0.2,\n"
        "         0.0, 0.0, 1.0, 0.3,\n"
        "         0.0, 0.0, 0.0, 1.0]\n"
        "rate_hz: 20\n",
        encoding="utf-8",
    )
    t_bs = MODULE.parse_cam0_t_bs(sensor_yaml)
    expected = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.1],
            [0.0, 1.0, 0.0, 0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(t_bs, expected)


def test_parse_cam0_t_bs_missing_block_raises(tmp_path: Path) -> None:
    sensor_yaml = tmp_path / "sensor.yaml"
    sensor_yaml.write_text("sensor_type: camera\n", encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.parse_cam0_t_bs(sensor_yaml)


# ---------------------------------------------------------------------------
# num_arrivals (arrival -> image index mapping arithmetic)
# ---------------------------------------------------------------------------


def test_num_arrivals_unbounded_uses_all_strided_images() -> None:
    # 10 images, stride 2 -> images 0,2,4,6,8 => 5 arrivals.
    assert MODULE.num_arrivals(image_count=10, stride=2, max_frames=0) == 5


def test_num_arrivals_capped_by_max_frames() -> None:
    assert MODULE.num_arrivals(image_count=1000, stride=2, max_frames=400) == 400


def test_num_arrivals_max_frames_larger_than_available() -> None:
    assert MODULE.num_arrivals(image_count=9, stride=2, max_frames=999) == 5  # ceil(9/2)


# ---------------------------------------------------------------------------
# interpolate_gt_pose
# ---------------------------------------------------------------------------


def test_interpolate_gt_pose_uses_nearest_within_tolerance() -> None:
    gt = MODULE.GroundTruth(
        timestamps_ns=[0, 5_000_000, 10_000_000],
        positions=np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        quats_wxyz=np.asarray([[1.0, 0, 0, 0], [1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
    )
    # 4,999,999 ns from the middle sample: within the 5ms tolerance -> nearest.
    position, quat = MODULE.interpolate_gt_pose(5_000_000 + 1, gt, tol_ns=5_000_000)
    np.testing.assert_allclose(position, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(quat, [1.0, 0, 0, 0])


def test_interpolate_gt_pose_falls_back_to_linear_position_beyond_tolerance() -> None:
    gt = MODULE.GroundTruth(
        timestamps_ns=[0, 100_000_000],  # 100ms apart -- beyond a 5ms tolerance
        positions=np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
        quats_wxyz=np.asarray([[1.0, 0, 0, 0], [0.0, 1.0, 0, 0]]),
    )
    # Exactly halfway: linear position interpolation -> 5.0; orientation is
    # nearest-neighbour (no slerp), and both samples are equidistant so the
    # earlier (lo) one wins the tie in this implementation.
    position, quat = MODULE.interpolate_gt_pose(50_000_000, gt, tol_ns=5_000_000)
    np.testing.assert_allclose(position, [5.0, 0.0, 0.0])
    assert tuple(quat) in {(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)}


def test_interpolate_gt_pose_clamps_before_first_sample() -> None:
    gt = MODULE.GroundTruth(
        timestamps_ns=[1_000_000_000, 2_000_000_000],
        positions=np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        quats_wxyz=np.asarray([[1.0, 0, 0, 0], [1.0, 0, 0, 0]]),
    )
    position, _ = MODULE.interpolate_gt_pose(0, gt, tol_ns=5_000_000)
    np.testing.assert_allclose(position, [1.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# label_revisit_pairs: min-gap filtering + angle gating + radius
# ---------------------------------------------------------------------------


def test_label_revisit_pairs_min_gap_filtering() -> None:
    # Three co-located, identically-oriented arrivals 0,1,2.
    positions = np.zeros((3, 3))
    axes = np.tile(np.asarray([0.0, 0.0, 1.0]), (3, 1))
    mask = MODULE.label_revisit_pairs(positions, axes, min_gap=2, radius_m=1.0, max_angle_deg=30.0)
    assert not mask[0, 1]  # gap 1 < min_gap 2
    assert mask[0, 2]  # gap 2 >= min_gap 2, same position
    assert not mask[1, 2]  # gap 1 < min_gap 2


def test_label_revisit_pairs_angle_gating_excludes_disagreeing_orientation() -> None:
    positions = np.zeros((2, 3))  # co-located
    axes = np.asarray(
        [
            [0.0, 0.0, 1.0],
            (rotation_about_x(math.radians(90)) @ np.asarray([0.0, 0.0, 1.0])),
        ]
    )
    mask = MODULE.label_revisit_pairs(positions, axes, min_gap=1, radius_m=1.0, max_angle_deg=30.0)
    assert not mask[0, 1]  # 90 deg disagreement > 30 deg gate


def test_label_revisit_pairs_angle_gating_accepts_small_disagreement() -> None:
    positions = np.zeros((2, 3))
    axes = np.asarray(
        [
            [0.0, 0.0, 1.0],
            (rotation_about_x(math.radians(10)) @ np.asarray([0.0, 0.0, 1.0])),
        ]
    )
    mask = MODULE.label_revisit_pairs(positions, axes, min_gap=1, radius_m=1.0, max_angle_deg=30.0)
    assert mask[0, 1]  # 10 deg disagreement < 30 deg gate


def test_label_revisit_pairs_radius_is_strict_less_than() -> None:
    positions = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0 - 1e-9]])
    axes = np.tile(np.asarray([0.0, 0.0, 1.0]), (3, 1))
    mask = MODULE.label_revisit_pairs(positions, axes, min_gap=1, radius_m=1.0, max_angle_deg=30.0)
    assert not mask[0, 1]  # distance exactly == radius -> excluded
    assert mask[0, 2]  # distance just under radius -> included


# ---------------------------------------------------------------------------
# evaluate_run: recall hit/miss, denominator restricted to issued queries,
# opportunity coverage, similarity stats
# ---------------------------------------------------------------------------


def make_mask(n: int, labelled_pairs: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    for i, j in labelled_pairs:
        mask[i, j] = True
    return mask


def test_evaluate_run_hit_and_miss_within_near_window() -> None:
    # Query 5's genuine partner is arrival 0; query 4's genuine partner is
    # also arrival 0.
    mask = make_mask(6, [(0, 4), (0, 5)])
    rows = [
        MODULE.CandidateRow(
            query_arrival=5, rank=0, candidate_arrival=2, gap=5, similarity=0.9, accepted=False
        ),  # |2-0|=2 <= near_window(2) -> hit
        MODULE.CandidateRow(
            query_arrival=4, rank=0, candidate_arrival=3, gap=4, similarity=0.5, accepted=False
        ),  # |3-0|=3 > near_window(2) -> miss
    ]
    ev = MODULE.evaluate_run(
        rows, mask, min_gap=3, radius_m=1.0, max_angle_deg=30.0, near_window=2, k_values=[1]
    )
    assert ev.recall_at_k[0].k == 1
    assert ev.recall_at_k[0].denominator == 2
    assert ev.recall_at_k[0].hits == 1
    assert ev.recall_at_k[0].recall == pytest.approx(0.5)


def test_evaluate_run_recall_at_k_looks_beyond_rank_zero() -> None:
    # Query 5's genuine partner is arrival 0; the exact match is rank 2, not
    # rank 0 -- recall@1 should miss, recall@3 should hit.
    mask = make_mask(6, [(0, 5)])
    rows = [
        MODULE.CandidateRow(query_arrival=5, rank=0, candidate_arrival=50, gap=5, similarity=0.9, accepted=False),
        MODULE.CandidateRow(query_arrival=5, rank=1, candidate_arrival=51, gap=5, similarity=0.8, accepted=False),
        MODULE.CandidateRow(query_arrival=5, rank=2, candidate_arrival=0, gap=5, similarity=0.7, accepted=False),
    ]
    ev = MODULE.evaluate_run(
        rows, mask, min_gap=1, radius_m=1.0, max_angle_deg=30.0, near_window=0, k_values=[1, 3]
    )
    by_k = {r.k: r for r in ev.recall_at_k}
    assert by_k[1].recall == pytest.approx(0.0)
    assert by_k[3].recall == pytest.approx(1.0)


def test_evaluate_run_denominator_excludes_non_issued_queries() -> None:
    # Arrival 5 has a labelled partner but never appears as a query_arrival
    # in the candidate log at all (retrieval cadence never queried it).
    mask = make_mask(8, [(0, 5), (0, 7)])
    rows = [
        MODULE.CandidateRow(query_arrival=7, rank=0, candidate_arrival=0, gap=7, similarity=0.9, accepted=False),
    ]
    ev = MODULE.evaluate_run(
        rows, mask, min_gap=1, radius_m=1.0, max_angle_deg=30.0, near_window=0, k_values=[1]
    )
    assert ev.recall_at_k[0].denominator == 1  # only query 7 was issued
    assert ev.recall_at_k[0].hits == 1
    assert ev.missed_queries == [5]


def test_evaluate_run_opportunity_coverage_missed_fraction() -> None:
    # Two labelled query arrivals (4 and 5); only 5 is ever issued as a query.
    mask = make_mask(6, [(0, 4), (0, 5)])
    rows = [
        MODULE.CandidateRow(query_arrival=5, rank=0, candidate_arrival=0, gap=5, similarity=0.9, accepted=False),
    ]
    ev = MODULE.evaluate_run(
        rows, mask, min_gap=1, radius_m=1.0, max_angle_deg=30.0, near_window=0, k_values=[1]
    )
    assert ev.missed_queries == [4]
    assert ev.opportunity_coverage_missed_fraction == pytest.approx(0.5)  # 1 of 2 labelled queries missed


def test_evaluate_run_opportunity_coverage_is_none_when_no_labelled_pairs() -> None:
    mask = make_mask(4, [])
    rows: list = []
    ev = MODULE.evaluate_run(
        rows, mask, min_gap=1, radius_m=1.0, max_angle_deg=30.0, near_window=0, k_values=[1]
    )
    assert ev.opportunity_coverage_missed_fraction is None
    assert ev.labelled_pairs_total == 0


def test_evaluate_run_similarity_stats_separate_hits_from_all() -> None:
    mask = make_mask(6, [(0, 5)])
    rows = [
        MODULE.CandidateRow(query_arrival=5, rank=0, candidate_arrival=0, gap=5, similarity=0.8, accepted=False),
        MODULE.CandidateRow(query_arrival=5, rank=1, candidate_arrival=99, gap=5, similarity=0.2, accepted=False),
    ]
    ev = MODULE.evaluate_run(
        rows, mask, min_gap=1, radius_m=1.0, max_angle_deg=30.0, near_window=0, k_values=[1]
    )
    assert ev.similarity_all.count == 2
    assert ev.similarity_all.mean == pytest.approx(0.5)
    assert ev.similarity_labelled_hit.count == 1
    assert ev.similarity_labelled_hit.mean == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# load_candidates: tolerant of both header variants seen in archived runs
# ---------------------------------------------------------------------------


def test_load_candidates_with_rotation_column(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(
        "query_arrival,rank,candidate_arrival,gap,similarity,accepted,rotation_disagreement_deg\n"
        "10,0,1,9,0.5,false,\n"
        "10,1,2,8,0.4,true,12.5\n",
        encoding="utf-8",
    )
    rows = MODULE.load_candidates(path)
    assert len(rows) == 2
    assert rows[1].accepted is True


def test_load_candidates_without_rotation_column(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(
        "query_arrival,rank,candidate_arrival,gap,similarity,accepted\n"
        "10,0,1,9,0.5,false\n",
        encoding="utf-8",
    )
    rows = MODULE.load_candidates(path)
    assert len(rows) == 1
    assert rows[0].accepted is False


def test_load_candidates_missing_required_column_raises(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text("query_arrival,rank,candidate_arrival\n10,0,1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        MODULE.load_candidates(path)


# ---------------------------------------------------------------------------
# diagnose_tightest_pair
# ---------------------------------------------------------------------------


def test_diagnose_tightest_pair_found_near_target() -> None:
    rows = [
        MODULE.CandidateRow(query_arrival=456, rank=0, candidate_arrival=44, gap=412, similarity=0.3, accepted=False),
    ]
    diag = MODULE.diagnose_tightest_pair(rows, target_i=42, target_j=456, near_window=5)
    assert diag.found_near_target is True
    assert diag.query_arrivals_near_target == [456]


def test_diagnose_tightest_pair_not_found_when_no_query_nearby() -> None:
    rows = [
        MODULE.CandidateRow(query_arrival=200, rank=0, candidate_arrival=44, gap=156, similarity=0.3, accepted=False),
    ]
    diag = MODULE.diagnose_tightest_pair(rows, target_i=42, target_j=456, near_window=5)
    assert diag.found_near_target is False
    assert diag.query_arrivals_near_target == []
    assert diag.rows == []


# ---------------------------------------------------------------------------
# End-to-end integration: real synthetic EuRoC-shaped CSVs in tmp_path
# ---------------------------------------------------------------------------


def _write_synthetic_dataset(mav0_dir: Path) -> None:
    (mav0_dir / "cam0").mkdir(parents=True)
    (mav0_dir / "state_groundtruth_estimate0").mkdir(parents=True)

    # 6 arrivals (stride=1), timestamps exactly matching GT samples so
    # nearest-neighbour interpolation is exact (delta=0).
    timestamps = [0, 10_000_000, 20_000_000, 30_000_000, 40_000_000, 50_000_000]
    cam0_lines = ["#timestamp [ns],filename"]
    for ts in timestamps:
        cam0_lines.append(f"{ts},{ts}.png")
    (mav0_dir / "cam0" / "data.csv").write_text("\n".join(cam0_lines) + "\n", encoding="utf-8")

    (mav0_dir / "cam0" / "sensor.yaml").write_text(
        "sensor_type: camera\n"
        "T_BS:\n"
        "  cols: 4\n"
        "  rows: 4\n"
        "  data: [1.0, 0.0, 0.0, 0.0,\n"
        "         0.0, 1.0, 0.0, 0.0,\n"
        "         0.0, 0.0, 1.0, 0.0,\n"
        "         0.0, 0.0, 0.0, 1.0]\n"
        "rate_hz: 20\n",
        encoding="utf-8",
    )

    # Positions: 1000 units apart per arrival (never accidentally close),
    # except arrival 5 is placed 0.1m from arrival 0 -- the one designed
    # revisit pair, gap=5. Identity orientation everywhere (angle gate never
    # excludes in this integration test -- angle gating has its own
    # dedicated unit tests above).
    positions = {i: (1000.0 * i, 0.0, 0.0) for i in range(6)}
    positions[5] = (0.1, 0.0, 0.0)
    gt_lines = [
        "#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []"
    ]
    for i, ts in enumerate(timestamps):
        px, py, pz = positions[i]
        gt_lines.append(f"{ts},{px},{py},{pz},1.0,0.0,0.0,0.0")
    (mav0_dir / "state_groundtruth_estimate0" / "data.csv").write_text(
        "\n".join(gt_lines) + "\n", encoding="utf-8"
    )


def test_build_report_end_to_end_recovers_designed_revisit(tmp_path: Path) -> None:
    mav0_dir = tmp_path / "mav0"
    _write_synthetic_dataset(mav0_dir)

    candidates_csv = tmp_path / "long_loop_candidates.csv"
    candidates_csv.write_text(
        "query_arrival,rank,candidate_arrival,gap,similarity,accepted,rotation_disagreement_deg\n"
        "5,0,0,5,0.77,false,\n",
        encoding="utf-8",
    )

    report = MODULE.build_report(
        candidates_csv=candidates_csv,
        gt_dir=mav0_dir,
        stride=1,
        max_frames=6,
        min_gap=3,
        radius_m=1.0,
        radius_secondary_m=0.5,
        max_angle_deg=30.0,
        near_window=1,
        gt_tol_ns=5_000_000,
        k_values=[1, 3],
        diagnostic_i=0,
        diagnostic_j=5,
    )

    assert report["config"]["n_arrivals"] == 6
    primary = report["evaluations"][0]
    assert primary["radius_m"] == 1.0
    assert primary["labelled_pairs_total"] == 1  # only (0, 5)
    recall_by_k = {r["k"]: r for r in primary["recall_at_k"]}
    assert recall_by_k[1]["denominator"] == 1
    assert recall_by_k[1]["hits"] == 1
    assert recall_by_k[1]["recall"] == pytest.approx(1.0)
    assert primary["opportunity_coverage_missed_fraction"] == pytest.approx(0.0)

    diag = report["diagnostic_tightest_pair"]
    assert diag["found_near_target"] is True
    assert diag["query_arrivals_near_target"] == [5]


def test_build_report_secondary_radius_can_exclude_the_same_pair(tmp_path: Path) -> None:
    mav0_dir = tmp_path / "mav0"
    _write_synthetic_dataset(mav0_dir)  # designed pair is 0.1m apart

    candidates_csv = tmp_path / "long_loop_candidates.csv"
    candidates_csv.write_text(
        "query_arrival,rank,candidate_arrival,gap,similarity,accepted\n"
        "5,0,0,5,0.77,false\n",
        encoding="utf-8",
    )

    report = MODULE.build_report(
        candidates_csv=candidates_csv,
        gt_dir=mav0_dir,
        stride=1,
        max_frames=6,
        min_gap=3,
        radius_m=1.0,
        radius_secondary_m=0.05,  # tighter than the designed 0.1m gap
        max_angle_deg=30.0,
        near_window=1,
        gt_tol_ns=5_000_000,
        k_values=[1],
        diagnostic_i=0,
        diagnostic_j=5,
    )
    primary, secondary = report["evaluations"]
    assert primary["labelled_pairs_total"] == 1
    assert secondary["labelled_pairs_total"] == 0
