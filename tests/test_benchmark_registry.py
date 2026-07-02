from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from benchmark_registry import (  # noqa: E402
    check_generated,
    capture,
    command_record,
    format_metric,
    render_claim_matrix_table,
    render_claim_snapshot,
    render_readme,
    render_runs,
    render_runs_table,
    render_secondary_metrics,
    sha256_path,
)
from summarize_euroc_active_observation_sweep import (  # noqa: E402
    load_latest_runs as load_active_observation_runs,
    render as render_active_observation_sweep,
)
from summarize_euroc_covisibility_ab import (  # noqa: E402
    load_latest_runs as load_covisibility_ab_runs,
    render as render_covisibility_ab,
)
from summarize_euroc_covisibility_mh05_mitigation import (  # noqa: E402
    load_latest_runs as load_covisibility_mh05_mitigation_runs,
    render as render_covisibility_mh05_mitigation,
)
from summarize_euroc_covisibility_mh05_boundary_support_gate import (  # noqa: E402
    load_latest_runs as load_covisibility_mh05_boundary_support_gate_runs,
    render as render_covisibility_mh05_boundary_support_gate,
)
from summarize_euroc_covisibility_runtime_sweep import (  # noqa: E402
    load_latest_runs as load_covisibility_runtime_runs,
    render as render_covisibility_runtime_sweep,
)
from summarize_euroc_covisibility_window_sweep import (  # noqa: E402
    load_latest_runs as load_covisibility_window_runs,
    render as render_covisibility_window_sweep,
)


def test_command_record_preserves_windows_backslashes():
    raw = (
        r".\target\release\examples\euroc_online_slam_vi_image_demo.exe "
        r"--euroc-dir C:\Users\rsasa\dataset\MH_03_medium "
        r"--out-dir target\tracking_survival_diag\MH_03_medium\run"
    )
    record = command_record(raw)
    assert record["argv"][0] == r".\target\release\examples\euroc_online_slam_vi_image_demo.exe"
    assert record["argv"][2] == r"C:\Users\rsasa\dataset\MH_03_medium"
    assert record["argv"][4] == r"target\tracking_survival_diag\MH_03_medium\run"


def write_active_sweep_manifest(
    registry: Path,
    *,
    run_id: str,
    variant: str,
    max_frames: int,
    sequence: str = "MH_03_medium",
    floor: int = 20,
    fallback: str = "none",
) -> None:
    manifest = {
        "run_id": run_id,
        "created_utc": "2026-06-19T00:00:00Z",
        "status": "success",
        "benchmark": {"id": "euroc-keyframe-tracked-landmark-drop"},
        "dataset": {"sequence": sequence},
        "config": {
            "params": {
                "max_frames": max_frames,
                "variant": variant,
                "demo_args": (
                    "--covisibility-local-ba "
                    f"--covisibility-local-ba-min-active-observations {floor} "
                    f"--covisibility-local-ba-fallback-min-boundary-observations {fallback}"
                ),
            }
        },
        "metrics": [
            {"name": "tracking_success_rate", "value": 0.8},
            {"name": "ate_rigid_rmse_m", "value": 1.5},
            {"name": "ate_similarity_rmse_m", "value": 0.15},
            {"name": "map_keyframes", "value": 66},
            {"name": "covisibility_local_ba_successes", "value": 35},
            {"name": "covisibility_local_ba_failures", "value": 29},
            {"name": "covisibility_local_ba_active_observation_gate_failures", "value": 4},
            {"name": "covisibility_local_ba_no_local_landmarks_failures", "value": 24},
            {"name": "covisibility_local_ba_solver_failures", "value": 0},
        ],
    }
    (registry / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_complete_active_sweep_manifests(registry: Path, *, max_frames: int) -> None:
    write_active_sweep_manifest(
        registry,
        run_id="active-fixed",
        variant="fixed",
        max_frames=max_frames,
    )
    write_active_sweep_manifest(
        registry,
        run_id="active-tracked-drop",
        variant="tracked_drop",
        max_frames=max_frames,
    )


def write_runtime_sweep_manifest(
    registry: Path,
    *,
    run_id: str,
    cap: int,
    max_frames: int,
    sequence: str = "MH_03_medium",
    min_active: int = 20,
    fallback: str | None = None,
    neighbor: int = 10,
    boundary: int = 10,
) -> None:
    manifest = {
        "run_id": run_id,
        "created_utc": "2026-06-19T00:00:00Z",
        "status": "success",
        "benchmark": {"id": "euroc-covisibility-local-ba"},
        "dataset": {"sequence": sequence},
        "config": {
            "params": {
                "variant": "enabled",
                "max_frames": max_frames,
                "covisibility_local_ba_max_landmarks": cap,
                "covisibility_local_ba_min_active_observations": min_active,
                "covisibility_local_ba_fallback_min_boundary_observations": fallback,
                "covisibility_local_ba_remove_outliers": False,
                "covisibility_local_ba_max_outlier_observation_ratio": None,
                "covisibility_local_ba_boundary_support_min_optimized_keyframes": None,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes": 0,
                "covisibility_local_ba_max_neighbor_keyframes": neighbor,
                "covisibility_local_ba_max_boundary_keyframes": boundary,
            }
        },
        "metrics": [
            {"name": "tracking_success_rate", "value": 0.8},
            {"name": "ate_rigid_rmse_m", "value": 1.5},
            {"name": "ate_similarity_rmse_m", "value": 0.15},
            {"name": "covisibility_local_ba_successes", "value": 8},
            {"name": "covisibility_local_ba_failures", "value": 0},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": 0},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": 12.5},
            {"name": "covisibility_local_ba_elapsed_ms_max", "value": 30.0},
        ],
    }
    (registry / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_complete_runtime_sweep_manifests(
    registry: Path,
    *,
    max_frames: int,
    caps: tuple[int, ...] = (100, 200, 400),
) -> None:
    for cap in caps:
        write_runtime_sweep_manifest(
            registry,
            run_id=f"runtime-{cap}",
            cap=cap,
            max_frames=max_frames,
        )


def write_window_sweep_manifest(
    registry: Path,
    *,
    run_id: str,
    neighbor: int,
    boundary: int,
    max_frames: int,
    sequence: str = "MH_03_medium",
    landmark_cap: int = 200,
    min_keyframes: int = 3,
    trigger_every: int = 1,
    min_active: int = 20,
    fallback: str | None = None,
    max_outlier: float | None = None,
    quality_gate_failures: int = 0,
    boundary_support_min_optimized: int | None = None,
    boundary_support_min_fixed: int = 0,
    boundary_support_failures: int = 0,
) -> None:
    manifest = {
        "run_id": run_id,
        "created_utc": "2026-06-19T00:00:00Z",
        "status": "success",
        "benchmark": {"id": "euroc-covisibility-local-ba"},
        "dataset": {"sequence": sequence},
        "config": {
            "params": {
                "variant": "enabled",
                "max_frames": max_frames,
                "covisibility_local_ba_max_landmarks": landmark_cap,
                "covisibility_local_ba_min_keyframes": min_keyframes,
                "covisibility_local_ba_trigger_every": trigger_every,
                "covisibility_local_ba_min_active_observations": min_active,
                "covisibility_local_ba_fallback_min_boundary_observations": fallback,
                "covisibility_local_ba_remove_outliers": False,
                "covisibility_local_ba_max_outlier_observation_ratio": max_outlier,
                "covisibility_local_ba_boundary_support_min_optimized_keyframes": boundary_support_min_optimized,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes": boundary_support_min_fixed,
                "covisibility_local_ba_max_neighbor_keyframes": neighbor,
                "covisibility_local_ba_max_boundary_keyframes": boundary,
            }
        },
        "metrics": [
            {"name": "tracking_success_rate", "value": 0.8},
            {"name": "ate_rigid_rmse_m", "value": 1.5},
            {"name": "ate_similarity_rmse_m", "value": 0.15},
            {"name": "covisibility_local_ba_successes", "value": 8},
            {"name": "covisibility_local_ba_failures", "value": 0},
            {"name": "covisibility_local_ba_quality_gate_failures", "value": quality_gate_failures},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": boundary_support_failures},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": 12.5},
            {"name": "covisibility_local_ba_elapsed_ms_max", "value": 30.0},
        ],
    }
    (registry / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_complete_window_sweep_manifests(
    registry: Path,
    *,
    max_frames: int,
    windows: tuple[tuple[int, int], ...] = ((5, 5), (10, 10), (15, 15)),
) -> None:
    for neighbor, boundary in windows:
        write_window_sweep_manifest(
            registry,
            run_id=f"window-{neighbor}-{boundary}",
            neighbor=neighbor,
            boundary=boundary,
            max_frames=max_frames,
        )


def write_covisibility_ab_manifest(
    registry: Path,
    *,
    run_id: str,
    variant: str,
    max_frames: int,
    sequence: str = "MH_03_medium",
    neighbor: int = 10,
    boundary: int = 10,
    min_keyframes: int = 3,
    trigger_every: int = 1,
    landmark_cap: int = 200,
    min_active: int = 20,
    fallback: str | None = None,
    max_outlier: float | None = None,
    quality_gate_failures: int = 0,
    boundary_support_min_optimized: int | None = None,
    boundary_support_min_fixed: int = 0,
    boundary_support_failures: int = 0,
) -> None:
    params = {
        "variant": variant,
        "max_frames": max_frames,
    }
    if variant == "enabled":
        params.update(
            {
                "covisibility_local_ba_max_neighbor_keyframes": neighbor,
                "covisibility_local_ba_max_boundary_keyframes": boundary,
                "covisibility_local_ba_min_keyframes": min_keyframes,
                "covisibility_local_ba_trigger_every": trigger_every,
                "covisibility_local_ba_max_landmarks": landmark_cap,
                "covisibility_local_ba_min_active_observations": min_active,
                "covisibility_local_ba_fallback_min_boundary_observations": fallback,
                "covisibility_local_ba_remove_outliers": False,
                "covisibility_local_ba_max_outlier_observation_ratio": max_outlier,
                "covisibility_local_ba_boundary_support_min_optimized_keyframes": boundary_support_min_optimized,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes": boundary_support_min_fixed,
            }
        )
    manifest = {
        "run_id": run_id,
        "created_utc": "2026-06-19T00:00:00Z",
        "status": "success",
        "benchmark": {"id": "euroc-covisibility-local-ba"},
        "dataset": {"sequence": sequence},
        "config": {"params": params},
        "metrics": [
            {"name": "tracking_success_rate", "value": 0.8},
            {"name": "ate_rigid_rmse_m", "value": 1.5},
            {"name": "ate_similarity_rmse_m", "value": 0.15},
            {"name": "covisibility_local_ba_successes", "value": 8},
            {"name": "covisibility_local_ba_failures", "value": 0},
            {"name": "covisibility_local_ba_quality_gate_failures", "value": quality_gate_failures},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": boundary_support_failures},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": 12.5},
        ],
    }
    (registry / f"{run_id}.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_complete_covisibility_ab_manifests(registry: Path, *, max_frames: int) -> None:
    write_covisibility_ab_manifest(
        registry,
        run_id="ab-disabled",
        variant="disabled",
        max_frames=max_frames,
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="ab-enabled",
        variant="enabled",
        max_frames=max_frames,
    )


def write_complete_covisibility_mh05_mitigation_manifests(
    registry: Path,
    *,
    max_frames: int,
) -> None:
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-disabled",
        variant="disabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min3-every1",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=3,
        trigger_every=1,
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min6-every3",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=6,
        trigger_every=3,
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min10-every5",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=10,
        trigger_every=5,
    )


def write_complete_covisibility_mh05_quality_gate_manifests(
    registry: Path,
    *,
    max_frames: int,
    max_outlier: float = 0.3,
) -> None:
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min3-every1-gated",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=3,
        trigger_every=1,
        max_outlier=max_outlier,
        quality_gate_failures=4,
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min6-every3-gated",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=6,
        trigger_every=3,
        max_outlier=max_outlier,
        quality_gate_failures=0,
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min10-every5-gated",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=10,
        trigger_every=5,
        max_outlier=max_outlier,
        quality_gate_failures=0,
    )


def write_complete_covisibility_mh05_boundary_support_gate_manifests(
    registry: Path,
    *,
    max_frames: int,
    max_outlier: float = 0.3,
) -> None:
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min3-every1-boundary7",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=3,
        trigger_every=1,
        max_outlier=max_outlier,
        quality_gate_failures=0,
        boundary_support_min_optimized=7,
        boundary_support_min_fixed=2,
        boundary_support_failures=4,
    )
    write_covisibility_ab_manifest(
        registry,
        run_id="mh05-min3-every1-boundary10",
        variant="enabled",
        max_frames=max_frames,
        sequence="MH_05_difficult",
        min_keyframes=3,
        trigger_every=1,
        max_outlier=max_outlier,
        quality_gate_failures=2,
        boundary_support_min_optimized=10,
        boundary_support_min_fixed=2,
        boundary_support_failures=2,
    )


class BenchmarkRegistryRenderTest(unittest.TestCase):
    def test_sha256_path_hashes_dataset_tree_independent_of_creation_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first"
            second = root / "second"
            first.mkdir()
            second.mkdir()
            (first / "b.txt").write_text("two", encoding="utf-8")
            (first / "a.txt").write_text("one", encoding="utf-8")
            (second / "a.txt").write_text("one", encoding="utf-8")
            (second / "b.txt").write_text("two", encoding="utf-8")

            self.assertEqual(sha256_path(first), sha256_path(second))

            (second / "b.txt").write_text("changed", encoding="utf-8")
            self.assertNotEqual(sha256_path(first), sha256_path(second))

    def test_capture_auto_records_dataset_tree_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "calib.txt").write_text("P0: 1 0 0\n", encoding="utf-8")
            image_dir = dataset / "image_0"
            image_dir.mkdir()
            (image_dir / "000000.png").write_bytes(b"not-a-real-png")
            out = root / "run.json"

            rc = capture(
                Namespace(
                    out=str(out),
                    run_id="run-with-dataset-checksum",
                    benchmark_id="bench",
                    benchmark_name="Bench",
                    script="script.sh",
                    protocol="test protocol",
                    docs=[],
                    dataset_name="Dataset",
                    dataset_sequence="00",
                    dataset_version="local subset",
                    dataset_path=str(dataset),
                    dataset_checksum=None,
                    dataset_checksum_method=None,
                    result_kind="visloc_run",
                    claim_scope="exploratory",
                    status="success",
                    failure_reason=None,
                    command="script.sh --arg",
                    feature=["image-io"],
                    profile="release",
                    target=None,
                    config=[],
                    config_file=[],
                    seed=None,
                    metric=["frames=2"],
                    primary_metric="frames",
                    artifact=[],
                    model=[],
                    hardware_note=None,
                    notes="test run",
                )
            )

            self.assertEqual(rc, 0)
            manifest = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(manifest["dataset"]["checksum"], sha256_path(dataset))
            self.assertEqual(manifest["dataset"]["checksum_method"], "sha256_tree_v1")

    def test_format_metric_compacts_float_values(self) -> None:
        self.assertEqual(
            format_metric({"name": "ate_rmse_se3_m", "value": 1.2064326148378097, "unit": "m"}),
            "ate_rmse_se3_m=1.20643 m",
        )
        self.assertEqual(
            format_metric({"name": "verified_loops", "value": 117, "unit": "count"}),
            "verified_loops=117 count",
        )

    def test_secondary_metrics_prioritize_claim_relevant_values(self) -> None:
        text = render_secondary_metrics(
            [
                {"name": "primary", "value": 1.0, "primary": True},
                {"name": "map_landmarks", "value": 1500, "unit": "count", "primary": False},
                {"name": "map_keyframes", "value": 15, "unit": "count", "primary": False},
                {"name": "verified_loops", "value": 117, "unit": "count", "primary": False},
                {"name": "ate_rmse_sim3_m", "value": 1.1298809831524472, "unit": "m", "primary": False},
                {"name": "frames", "value": 2000, "unit": "count", "primary": False},
            ]
        )

        self.assertIn("ate_rmse_sim3_m=1.12988 m", text)
        self.assertIn("verified_loops=117 count", text)
        self.assertIn("frames=2000 count", text)
        self.assertNotIn("map_landmarks", text)
        self.assertNotIn("map_keyframes", text)

    def test_render_runs_includes_scope_secondary_metrics_and_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {
                "schema_version": 1,
                "run_id": "run-1",
                "created_utc": "2026-06-19T00:00:00Z",
                "result_kind": "visloc_run",
                "claim_scope": "negative",
                "status": "success",
                "failure_reason": None,
                "benchmark": {"id": "kitti-multiseq", "name": "KITTI", "docs": []},
                "dataset": {"name": "KITTI odometry", "sequence": "00"},
                "git": {"commit": "abc", "dirty": True},
                "build": {"cargo_lock_sha256": "abc", "features": []},
                "command": {"raw": "run", "argv": ["run"], "cwd": str(REPO_ROOT)},
                "hardware": {"os": "test", "machine": "x86"},
                "config": {"seed": None, "params": {}, "config_files": []},
                "models": [],
                "metrics": [
                    {"name": "ate_rmse_se3_m", "value": 1.241865061013204, "unit": "m", "primary": True},
                    {"name": "ate_rmse_sim3_m", "value": 1.1710946668481592, "unit": "m", "primary": False},
                    {"name": "verified_loops", "value": 111, "unit": "count", "primary": False},
                ],
                "artifacts": [],
                "notes": "negative evidence",
            }
            (root / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
            out = root / "registered_runs.md"

            rc = render_runs(Namespace(registry_dir=str(root), out=str(out), with_heading=True))

            self.assertEqual(rc, 0)
            text = out.read_text(encoding="utf-8")
            self.assertIn("# Registered Benchmark Runs", text)
            self.assertIn("| run-1 | kitti-multiseq | KITTI odometry | 00 | visloc_run | negative | success |", text)
            self.assertIn("ate_rmse_sim3_m=1.17109 m", text)
            self.assertIn("verified_loops=111 count", text)
            self.assertIn("negative evidence", text)

    def test_render_readme_heading_is_only_for_generated_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = root / "claims.json"
            claims.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "target": "README.md#benchmarks",
                        "claims": [
                            {
                                "benchmark": "Benchmark A",
                                "result_markdown": "**1.0 m**",
                                "claim_kind": "documented_historical",
                                "source_docs": ["docs/a.md"],
                                "evidence_run_ids": [],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            snapshot = root / "snapshot.md"
            readme = root / "README.md"
            readme.write_text(
                "before\n<!-- benchmark-registry:start -->\nold\n<!-- benchmark-registry:end -->\nafter\n",
                encoding="utf-8",
            )

            rc = render_readme(
                Namespace(
                    claims=str(claims),
                    out=str(snapshot),
                    readme=str(readme),
                    with_heading=True,
                )
            )

            self.assertEqual(rc, 0)
            self.assertIn("# Benchmark Snapshot", snapshot.read_text(encoding="utf-8"))
            readme_text = readme.read_text(encoding="utf-8")
            self.assertNotIn("# Benchmark Snapshot", readme_text)
            self.assertIn("| Benchmark A | **1.0 m** |", readme_text)

    def test_render_claim_matrix_keeps_verdicts_and_evidence_visible(self) -> None:
        matrix = {
            "schema_version": 1,
            "comparisons": [
                {
                    "comparison_id": "euroc-mh03-orbslam3",
                    "benchmark": "EuRoC MH_03",
                    "dataset": "EuRoC MAV",
                    "sequence": "MH_03",
                    "sensor_mode": "stereo visual",
                    "metric": "ATE RMSE SE(3), m",
                    "protocol": "published baseline comparison",
                    "visloc_result": "0.057 m",
                    "reference_system": "ORB-SLAM3",
                    "reference_result": "0.024 m",
                    "verdict": "behind",
                    "claim_kind": "mixed",
                    "claim_scope": "headline",
                    "source_docs": ["docs/euroc.md"],
                    "evidence_run_ids": [],
                    "notes": "not a win",
                }
            ],
        }

        text = render_claim_matrix_table(matrix, with_heading=True)

        self.assertIn("# Benchmark Claim Matrix", text)
        self.assertIn("EuRoC MH_03<br>ORB-SLAM3", text)
        self.assertIn("behind<br>not a win", text)
        self.assertIn("docs: docs/euroc.md", text)

    def test_check_generated_detects_stale_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = root / "claims.json"
            claim_matrix = root / "claim_matrix.json"
            registry = root / "runs"
            registry.mkdir()
            claims_obj = {
                "schema_version": 1,
                "target": "README.md#benchmarks",
                "claims": [
                    {
                        "benchmark": "Benchmark A",
                        "result_markdown": "**1.0 m**",
                        "claim_kind": "documented_historical",
                        "source_docs": ["docs/a.md"],
                        "evidence_run_ids": [],
                    }
                ],
            }
            claims.write_text(json.dumps(claims_obj), encoding="utf-8")
            claim_matrix_obj = {
                "schema_version": 1,
                "target": "claim-matrix",
                "comparisons": [
                    {
                        "comparison_id": "cmp-1",
                        "benchmark": "Bench 00",
                        "dataset": "Dataset",
                        "sequence": "00",
                        "sensor_mode": "stereo",
                        "metric": "ATE",
                        "protocol": "same protocol",
                        "visloc_result": "1.0 m",
                        "reference_system": "Reference",
                        "reference_result": "2.0 m",
                        "verdict": "win",
                        "claim_kind": "mixed",
                        "claim_scope": "headline",
                        "source_docs": ["docs/a.md"],
                        "evidence_run_ids": ["run-1"],
                    }
                ],
            }
            claim_matrix.write_text(json.dumps(claim_matrix_obj), encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "run_id": "run-1",
                "created_utc": "2026-06-19T00:00:00Z",
                "result_kind": "visloc_run",
                "claim_scope": "supporting",
                "status": "success",
                "failure_reason": None,
                "benchmark": {"id": "bench", "name": "Bench", "docs": []},
                "dataset": {"name": "Dataset", "sequence": "00"},
                "git": {"commit": "abc", "dirty": True},
                "build": {"cargo_lock_sha256": "abc", "features": []},
                "command": {"raw": "run", "argv": ["run"], "cwd": str(REPO_ROOT)},
                "hardware": {"os": "test", "machine": "x86"},
                "config": {"seed": None, "params": {}, "config_files": []},
                "models": [],
                "metrics": [
                    {"name": "ate", "value": 1.0, "unit": "m", "primary": True},
                ],
                "artifacts": [],
                "notes": "supporting evidence",
            }
            (registry / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
            table = render_claim_snapshot(claims_obj, with_heading=False)
            readme = root / "README.md"
            readme.write_text(
                f"before\n<!-- benchmark-registry:start -->\n{table}<!-- benchmark-registry:end -->\nafter\n",
                encoding="utf-8",
            )
            snapshot = root / "snapshot.md"
            snapshot.write_text(render_claim_snapshot(claims_obj, with_heading=True), encoding="utf-8")
            registered = root / "registered.md"
            registered.write_text(render_runs_table(str(registry), with_heading=True), encoding="utf-8")
            claim_matrix_out = root / "claim_matrix.md"
            claim_matrix_out.write_text(render_claim_matrix_table(claim_matrix_obj, with_heading=True), encoding="utf-8")
            active_registry = root / "active_runs"
            active_registry.mkdir()
            write_complete_active_sweep_manifests(active_registry, max_frames=400)
            active_sweep = root / "active_sweep.md"
            active_args = Namespace(
                registry_dir=active_registry,
                out=active_sweep,
                max_frames=400,
                sequence=["MH_03_medium"],
                active_floor=[20],
                fallback="none",
            )
            active_sweep.write_text(
                render_active_observation_sweep(
                    active_args,
                    load_active_observation_runs(active_args),
                ),
                encoding="utf-8",
            )
            runtime_registry = root / "runtime_runs"
            runtime_registry.mkdir()
            write_complete_runtime_sweep_manifests(runtime_registry, max_frames=80)
            runtime_sweep = root / "runtime_sweep.md"
            runtime_args = Namespace(
                registry_dir=runtime_registry,
                out=runtime_sweep,
                max_frames=80,
                sequence=["MH_03_medium"],
                landmark_cap=[100, 200, 400],
                neighbor_keyframes=10,
                boundary_keyframes=10,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            runtime_sweep.write_text(
                render_covisibility_runtime_sweep(
                    runtime_args,
                    load_covisibility_runtime_runs(runtime_args),
                ),
                encoding="utf-8",
            )
            window_registry = root / "window_runs"
            window_registry.mkdir()
            write_complete_window_sweep_manifests(window_registry, max_frames=80)
            window_sweep = root / "window_sweep.md"
            window_args = Namespace(
                registry_dir=window_registry,
                out=window_sweep,
                max_frames=80,
                sequence=["MH_03_medium"],
                window_cap=[(5, 5), (10, 10), (15, 15)],
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            window_sweep.write_text(
                render_covisibility_window_sweep(
                    window_args,
                    load_covisibility_window_runs(window_args),
                ),
                encoding="utf-8",
            )
            window_validation_registry = root / "window_validation_runs"
            window_validation_registry.mkdir()
            write_complete_window_sweep_manifests(
                window_validation_registry,
                max_frames=400,
                windows=((5, 5), (10, 10)),
            )
            window_validation = root / "window_validation.md"
            window_validation_args = Namespace(
                registry_dir=window_validation_registry,
                out=window_validation,
                max_frames=400,
                sequence=["MH_03_medium"],
                window_cap=[(5, 5), (10, 10)],
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            window_validation.write_text(
                render_covisibility_window_sweep(
                    window_validation_args,
                    load_covisibility_window_runs(window_validation_args),
                ),
                encoding="utf-8",
            )
            ab_registry = root / "ab_runs"
            ab_registry.mkdir()
            write_complete_covisibility_ab_manifests(ab_registry, max_frames=400)
            ab_summary = root / "ab.md"
            ab_args = Namespace(
                registry_dir=ab_registry,
                out=ab_summary,
                max_frames=400,
                sequence=["MH_03_medium"],
                enabled_neighbor_keyframes=10,
                enabled_boundary_keyframes=10,
                enabled_min_keyframes=3,
                enabled_trigger_every=1,
                enabled_landmark_cap=200,
                enabled_min_active_observations=20,
                enabled_fallback="none",
                enabled_remove_outliers=False,
                enabled_max_outlier_observation_ratio="none",
                enabled_boundary_support_min_optimized_keyframes="none",
                enabled_boundary_support_min_fixed_keyframes=0,
            )
            ab_summary.write_text(
                render_covisibility_ab(
                    ab_args,
                    load_covisibility_ab_runs(ab_args),
                ),
                encoding="utf-8",
            )
            mitigation_registry = root / "mitigation_runs"
            mitigation_registry.mkdir()
            write_complete_covisibility_mh05_mitigation_manifests(
                mitigation_registry,
                max_frames=400,
            )
            write_complete_covisibility_mh05_quality_gate_manifests(
                mitigation_registry,
                max_frames=400,
            )
            write_complete_covisibility_mh05_boundary_support_gate_manifests(
                mitigation_registry,
                max_frames=400,
            )
            mitigation_summary = root / "mh05_mitigation.md"
            quality_gate_summary = root / "mh05_quality_gate.md"
            boundary_support_gate_summary = root / "mh05_boundary_support_gate.md"
            boundary_support_gate_sweep = root / "mh05_boundary_support_gate_sweep.md"
            boundary_support_gate_sweep = root / "mh05_boundary_support_gate_sweep.md"
            mitigation_args = Namespace(
                registry_dir=mitigation_registry,
                out=mitigation_summary,
                sequence="MH_05_difficult",
                max_frames=400,
                neighbor_keyframes=10,
                boundary_keyframes=10,
                landmark_cap=200,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
                config=[
                    ("enabled min3/every1", 3, 1),
                    ("enabled min6/every3", 6, 3),
                    ("enabled min10/every5", 10, 5),
                ],
            )
            quality_gate_args = Namespace(
                **{
                    **vars(mitigation_args),
                    "out": quality_gate_summary,
                    "max_outlier_observation_ratio": "0.3",
                }
            )
            boundary_support_gate_args = Namespace(
                **{
                    **vars(mitigation_args),
                    "out": boundary_support_gate_summary,
                    "max_outlier_observation_ratio": "0.3",
                    "boundary_support_min_optimized_keyframes": "10",
                    "boundary_support_min_fixed_keyframes": 2,
                    "config": [("enabled min3/every1 boundary10", 3, 1)],
                }
            )
            boundary_support_gate_sweep_args = Namespace(
                registry_dir=mitigation_registry,
                out=boundary_support_gate_sweep,
                sequence="MH_05_difficult",
                max_frames=400,
                neighbor_keyframes=10,
                boundary_keyframes=10,
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="0.3",
                gate=[
                    ("quality-gate only", "none", 0),
                    ("boundary7/2", "7", 2),
                    ("boundary10/2", "10", 2),
                ],
            )
            mitigation_summary.write_text(
                render_covisibility_mh05_mitigation(
                    mitigation_args,
                    load_covisibility_mh05_mitigation_runs(mitigation_args),
                ),
                encoding="utf-8",
            )
            quality_gate_summary.write_text(
                render_covisibility_mh05_mitigation(
                    quality_gate_args,
                    load_covisibility_mh05_mitigation_runs(quality_gate_args),
                ),
                encoding="utf-8",
            )
            boundary_support_gate_summary.write_text(
                render_covisibility_mh05_mitigation(
                    boundary_support_gate_args,
                    load_covisibility_mh05_mitigation_runs(boundary_support_gate_args),
                ),
                encoding="utf-8",
            )
            boundary_support_gate_sweep.write_text(
                render_covisibility_mh05_boundary_support_gate(
                    boundary_support_gate_sweep_args,
                    load_covisibility_mh05_boundary_support_gate_runs(
                        boundary_support_gate_sweep_args
                    ),
                ),
                encoding="utf-8",
            )

            args = Namespace(
                claims=str(claims),
                claim_matrix=str(claim_matrix),
                registry_dir=str(registry),
                readme=str(readme),
                benchmark_snapshot=str(snapshot),
                registered_runs=str(registered),
                claim_matrix_out=str(claim_matrix_out),
                active_observation_sweep=str(active_sweep),
                active_observation_registry_dir=str(active_registry),
                active_observation_max_frames=400,
                active_observation_sequence=["MH_03_medium"],
                active_observation_floor=[20],
                active_observation_fallback="none",
                covisibility_runtime_sweep=str(runtime_sweep),
                covisibility_runtime_registry_dir=str(runtime_registry),
                covisibility_runtime_max_frames=80,
                covisibility_runtime_sequence=["MH_03_medium"],
                covisibility_runtime_landmark_cap=[100, 200, 400],
                covisibility_runtime_neighbor_keyframes=10,
                covisibility_runtime_boundary_keyframes=10,
                covisibility_runtime_min_active_observations=20,
                covisibility_runtime_fallback="none",
                covisibility_runtime_remove_outliers=False,
                covisibility_runtime_max_outlier_observation_ratio="none",
                covisibility_runtime_boundary_support_min_optimized_keyframes="none",
                covisibility_runtime_boundary_support_min_fixed_keyframes=0,
                covisibility_window_sweep=str(window_sweep),
                covisibility_window_registry_dir=str(window_registry),
                covisibility_window_max_frames=80,
                covisibility_window_sequence=["MH_03_medium"],
                covisibility_window_cap=[(5, 5), (10, 10), (15, 15)],
                covisibility_window_landmark_cap=200,
                covisibility_window_min_keyframes=3,
                covisibility_window_trigger_every=1,
                covisibility_window_min_active_observations=20,
                covisibility_window_fallback="none",
                covisibility_window_remove_outliers=False,
                covisibility_window_max_outlier_observation_ratio="none",
                covisibility_window_boundary_support_min_optimized_keyframes="none",
                covisibility_window_boundary_support_min_fixed_keyframes=0,
                covisibility_window_validation=str(window_validation),
                covisibility_window_validation_registry_dir=str(window_validation_registry),
                covisibility_window_validation_max_frames=400,
                covisibility_window_validation_sequence=["MH_03_medium"],
                covisibility_window_validation_cap=[(5, 5), (10, 10)],
                covisibility_window_validation_landmark_cap=200,
                covisibility_window_validation_min_keyframes=3,
                covisibility_window_validation_trigger_every=1,
                covisibility_window_validation_min_active_observations=20,
                covisibility_window_validation_fallback="none",
                covisibility_window_validation_remove_outliers=False,
                covisibility_window_validation_max_outlier_observation_ratio="none",
                covisibility_window_validation_boundary_support_min_optimized_keyframes="none",
                covisibility_window_validation_boundary_support_min_fixed_keyframes=0,
                covisibility_ab=str(ab_summary),
                covisibility_ab_registry_dir=str(ab_registry),
                covisibility_ab_max_frames=400,
                covisibility_ab_sequence=["MH_03_medium"],
                covisibility_ab_enabled_neighbor_keyframes=10,
                covisibility_ab_enabled_boundary_keyframes=10,
                covisibility_ab_enabled_min_keyframes=3,
                covisibility_ab_enabled_trigger_every=1,
                covisibility_ab_enabled_landmark_cap=200,
                covisibility_ab_enabled_min_active_observations=20,
                covisibility_ab_enabled_fallback="none",
                covisibility_ab_enabled_remove_outliers=False,
                covisibility_ab_enabled_max_outlier_observation_ratio="none",
                covisibility_ab_enabled_boundary_support_min_optimized_keyframes="none",
                covisibility_ab_enabled_boundary_support_min_fixed_keyframes=0,
                covisibility_mh05_mitigation=str(mitigation_summary),
                covisibility_mh05_quality_gate=str(quality_gate_summary),
                covisibility_mh05_boundary_support_gate=str(boundary_support_gate_summary),
                covisibility_mh05_boundary_support_gate_sweep=str(boundary_support_gate_sweep),
                covisibility_mh05_mitigation_registry_dir=str(mitigation_registry),
                covisibility_mh05_mitigation_sequence="MH_05_difficult",
                covisibility_mh05_mitigation_max_frames=400,
                covisibility_mh05_mitigation_neighbor_keyframes=10,
                covisibility_mh05_mitigation_boundary_keyframes=10,
                covisibility_mh05_mitigation_landmark_cap=200,
                covisibility_mh05_mitigation_min_keyframes=3,
                covisibility_mh05_mitigation_trigger_every=1,
                covisibility_mh05_mitigation_min_active_observations=20,
                covisibility_mh05_mitigation_fallback="none",
                covisibility_mh05_mitigation_remove_outliers=False,
                covisibility_mh05_mitigation_max_outlier_observation_ratio="none",
                covisibility_mh05_quality_gate_max_outlier_observation_ratio="0.3",
                covisibility_mh05_boundary_support_gate_max_outlier_observation_ratio="0.3",
                covisibility_mh05_mitigation_boundary_support_min_optimized_keyframes="none",
                covisibility_mh05_mitigation_boundary_support_min_fixed_keyframes=0,
                covisibility_mh05_quality_gate_boundary_support_min_optimized_keyframes="none",
                covisibility_mh05_quality_gate_boundary_support_min_fixed_keyframes=0,
                covisibility_mh05_boundary_support_gate_min_optimized_keyframes="10",
                covisibility_mh05_boundary_support_gate_min_fixed_keyframes=2,
                covisibility_mh05_mitigation_config=[
                    ("enabled min3/every1", 3, 1),
                    ("enabled min6/every3", 6, 3),
                    ("enabled min10/every5", 10, 5),
                ],
                covisibility_mh05_boundary_support_gate_config=[
                    ("enabled min3/every1 boundary10", 3, 1),
                ],
                covisibility_mh05_boundary_support_gate_sweep_gate=[
                    ("quality-gate only", "none", 0),
                    ("boundary7/2", "7", 2),
                    ("boundary10/2", "10", 2),
                ],
            )
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check_generated(args), 0)

            snapshot.write_text("stale\n", encoding="utf-8")
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check_generated(args), 1)

    def test_check_generated_honors_custom_active_sweep_filters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            claims = root / "claims.json"
            claim_matrix = root / "claim_matrix.json"
            registry = root / "runs"
            registry.mkdir()
            claims_obj = {"schema_version": 1, "claims": []}
            claim_matrix_obj = {"schema_version": 1, "comparisons": []}
            claims.write_text(json.dumps(claims_obj), encoding="utf-8")
            claim_matrix.write_text(json.dumps(claim_matrix_obj), encoding="utf-8")
            readme = root / "README.md"
            readme.write_text(
                "<!-- benchmark-registry:start -->\n"
                f"{render_claim_snapshot(claims_obj, with_heading=False)}"
                "<!-- benchmark-registry:end -->\n",
                encoding="utf-8",
            )
            snapshot = root / "snapshot.md"
            snapshot.write_text(
                render_claim_snapshot(claims_obj, with_heading=True),
                encoding="utf-8",
            )
            registered = root / "registered.md"
            registered.write_text(
                render_runs_table(str(registry), with_heading=True),
                encoding="utf-8",
            )
            claim_matrix_out = root / "claim_matrix.md"
            claim_matrix_out.write_text(
                render_claim_matrix_table(claim_matrix_obj, with_heading=True),
                encoding="utf-8",
            )
            active_registry = root / "active_runs"
            active_registry.mkdir()
            write_complete_active_sweep_manifests(active_registry, max_frames=5)
            active_sweep = root / "active_sweep.md"
            custom_active_args = Namespace(
                registry_dir=active_registry,
                out=active_sweep,
                max_frames=5,
                sequence=["MH_03_medium"],
                active_floor=[20],
                fallback="none",
            )
            active_sweep.write_text(
                render_active_observation_sweep(
                    custom_active_args,
                    load_active_observation_runs(custom_active_args),
                ),
                encoding="utf-8",
            )
            runtime_registry = root / "runtime_runs"
            runtime_registry.mkdir()
            write_complete_runtime_sweep_manifests(runtime_registry, max_frames=5, caps=(100, 200))
            runtime_sweep = root / "runtime_sweep.md"
            custom_runtime_args = Namespace(
                registry_dir=runtime_registry,
                out=runtime_sweep,
                max_frames=5,
                sequence=["MH_03_medium"],
                landmark_cap=[100, 200],
                neighbor_keyframes=10,
                boundary_keyframes=10,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            runtime_sweep.write_text(
                render_covisibility_runtime_sweep(
                    custom_runtime_args,
                    load_covisibility_runtime_runs(custom_runtime_args),
                ),
                encoding="utf-8",
            )
            window_registry = root / "window_runs"
            window_registry.mkdir()
            write_complete_window_sweep_manifests(
                window_registry,
                max_frames=5,
                windows=((5, 5), (10, 10)),
            )
            window_sweep = root / "window_sweep.md"
            custom_window_args = Namespace(
                registry_dir=window_registry,
                out=window_sweep,
                max_frames=5,
                sequence=["MH_03_medium"],
                window_cap=[(5, 5), (10, 10)],
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            window_sweep.write_text(
                render_covisibility_window_sweep(
                    custom_window_args,
                    load_covisibility_window_runs(custom_window_args),
                ),
                encoding="utf-8",
            )
            window_validation_registry = root / "window_validation_runs"
            window_validation_registry.mkdir()
            write_complete_window_sweep_manifests(
                window_validation_registry,
                max_frames=6,
                windows=((5, 5), (10, 10)),
            )
            window_validation = root / "window_validation.md"
            custom_window_validation_args = Namespace(
                registry_dir=window_validation_registry,
                out=window_validation,
                max_frames=6,
                sequence=["MH_03_medium"],
                window_cap=[(5, 5), (10, 10)],
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            window_validation.write_text(
                render_covisibility_window_sweep(
                    custom_window_validation_args,
                    load_covisibility_window_runs(custom_window_validation_args),
                ),
                encoding="utf-8",
            )
            ab_registry = root / "ab_runs"
            ab_registry.mkdir()
            write_complete_covisibility_ab_manifests(ab_registry, max_frames=7)
            ab_summary = root / "ab.md"
            custom_ab_args = Namespace(
                registry_dir=ab_registry,
                out=ab_summary,
                max_frames=7,
                sequence=["MH_03_medium"],
                enabled_neighbor_keyframes=10,
                enabled_boundary_keyframes=10,
                enabled_min_keyframes=3,
                enabled_trigger_every=1,
                enabled_landmark_cap=200,
                enabled_min_active_observations=20,
                enabled_fallback="none",
                enabled_remove_outliers=False,
                enabled_max_outlier_observation_ratio="none",
                enabled_boundary_support_min_optimized_keyframes="none",
                enabled_boundary_support_min_fixed_keyframes=0,
            )
            ab_summary.write_text(
                render_covisibility_ab(
                    custom_ab_args,
                    load_covisibility_ab_runs(custom_ab_args),
                ),
                encoding="utf-8",
            )
            mitigation_registry = root / "mitigation_runs"
            mitigation_registry.mkdir()
            write_complete_covisibility_mh05_mitigation_manifests(
                mitigation_registry,
                max_frames=8,
            )
            write_complete_covisibility_mh05_quality_gate_manifests(
                mitigation_registry,
                max_frames=8,
            )
            write_complete_covisibility_mh05_boundary_support_gate_manifests(
                mitigation_registry,
                max_frames=8,
            )
            mitigation_summary = root / "mh05_mitigation.md"
            quality_gate_summary = root / "mh05_quality_gate.md"
            boundary_support_gate_summary = root / "mh05_boundary_support_gate.md"
            boundary_support_gate_sweep = root / "mh05_boundary_support_gate_sweep.md"
            custom_mitigation_args = Namespace(
                registry_dir=mitigation_registry,
                out=mitigation_summary,
                sequence="MH_05_difficult",
                max_frames=8,
                neighbor_keyframes=10,
                boundary_keyframes=10,
                landmark_cap=200,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
                config=[
                    ("enabled min3/every1", 3, 1),
                    ("enabled min6/every3", 6, 3),
                    ("enabled min10/every5", 10, 5),
                ],
            )
            custom_quality_gate_args = Namespace(
                **{
                    **vars(custom_mitigation_args),
                    "out": quality_gate_summary,
                    "max_outlier_observation_ratio": "0.3",
                }
            )
            custom_boundary_support_gate_args = Namespace(
                **{
                    **vars(custom_mitigation_args),
                    "out": boundary_support_gate_summary,
                    "max_outlier_observation_ratio": "0.3",
                    "boundary_support_min_optimized_keyframes": "10",
                    "boundary_support_min_fixed_keyframes": 2,
                    "config": [("enabled min3/every1 boundary10", 3, 1)],
                }
            )
            custom_boundary_support_gate_sweep_args = Namespace(
                registry_dir=mitigation_registry,
                out=boundary_support_gate_sweep,
                sequence="MH_05_difficult",
                max_frames=8,
                neighbor_keyframes=10,
                boundary_keyframes=10,
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="0.3",
                gate=[
                    ("quality-gate only", "none", 0),
                    ("boundary7/2", "7", 2),
                    ("boundary10/2", "10", 2),
                ],
            )
            mitigation_summary.write_text(
                render_covisibility_mh05_mitigation(
                    custom_mitigation_args,
                    load_covisibility_mh05_mitigation_runs(custom_mitigation_args),
                ),
                encoding="utf-8",
            )
            quality_gate_summary.write_text(
                render_covisibility_mh05_mitigation(
                    custom_quality_gate_args,
                    load_covisibility_mh05_mitigation_runs(custom_quality_gate_args),
                ),
                encoding="utf-8",
            )
            boundary_support_gate_summary.write_text(
                render_covisibility_mh05_mitigation(
                    custom_boundary_support_gate_args,
                    load_covisibility_mh05_mitigation_runs(custom_boundary_support_gate_args),
                ),
                encoding="utf-8",
            )
            boundary_support_gate_sweep.write_text(
                render_covisibility_mh05_boundary_support_gate(
                    custom_boundary_support_gate_sweep_args,
                    load_covisibility_mh05_boundary_support_gate_runs(
                        custom_boundary_support_gate_sweep_args
                    ),
                ),
                encoding="utf-8",
            )
            args = Namespace(
                claims=str(claims),
                claim_matrix=str(claim_matrix),
                registry_dir=str(registry),
                readme=str(readme),
                benchmark_snapshot=str(snapshot),
                registered_runs=str(registered),
                claim_matrix_out=str(claim_matrix_out),
                active_observation_sweep=str(active_sweep),
                active_observation_registry_dir=str(active_registry),
                active_observation_max_frames=5,
                active_observation_sequence=["MH_03_medium"],
                active_observation_floor=[20],
                active_observation_fallback="none",
                covisibility_runtime_sweep=str(runtime_sweep),
                covisibility_runtime_registry_dir=str(runtime_registry),
                covisibility_runtime_max_frames=5,
                covisibility_runtime_sequence=["MH_03_medium"],
                covisibility_runtime_landmark_cap=[100, 200],
                covisibility_runtime_neighbor_keyframes=10,
                covisibility_runtime_boundary_keyframes=10,
                covisibility_runtime_min_active_observations=20,
                covisibility_runtime_fallback="none",
                covisibility_runtime_remove_outliers=False,
                covisibility_runtime_max_outlier_observation_ratio="none",
                covisibility_runtime_boundary_support_min_optimized_keyframes="none",
                covisibility_runtime_boundary_support_min_fixed_keyframes=0,
                covisibility_window_sweep=str(window_sweep),
                covisibility_window_registry_dir=str(window_registry),
                covisibility_window_max_frames=5,
                covisibility_window_sequence=["MH_03_medium"],
                covisibility_window_cap=[(5, 5), (10, 10)],
                covisibility_window_landmark_cap=200,
                covisibility_window_min_keyframes=3,
                covisibility_window_trigger_every=1,
                covisibility_window_min_active_observations=20,
                covisibility_window_fallback="none",
                covisibility_window_remove_outliers=False,
                covisibility_window_max_outlier_observation_ratio="none",
                covisibility_window_boundary_support_min_optimized_keyframes="none",
                covisibility_window_boundary_support_min_fixed_keyframes=0,
                covisibility_window_validation=str(window_validation),
                covisibility_window_validation_registry_dir=str(window_validation_registry),
                covisibility_window_validation_max_frames=6,
                covisibility_window_validation_sequence=["MH_03_medium"],
                covisibility_window_validation_cap=[(5, 5), (10, 10)],
                covisibility_window_validation_landmark_cap=200,
                covisibility_window_validation_min_keyframes=3,
                covisibility_window_validation_trigger_every=1,
                covisibility_window_validation_min_active_observations=20,
                covisibility_window_validation_fallback="none",
                covisibility_window_validation_remove_outliers=False,
                covisibility_window_validation_max_outlier_observation_ratio="none",
                covisibility_window_validation_boundary_support_min_optimized_keyframes="none",
                covisibility_window_validation_boundary_support_min_fixed_keyframes=0,
                covisibility_ab=str(ab_summary),
                covisibility_ab_registry_dir=str(ab_registry),
                covisibility_ab_max_frames=7,
                covisibility_ab_sequence=["MH_03_medium"],
                covisibility_ab_enabled_neighbor_keyframes=10,
                covisibility_ab_enabled_boundary_keyframes=10,
                covisibility_ab_enabled_min_keyframes=3,
                covisibility_ab_enabled_trigger_every=1,
                covisibility_ab_enabled_landmark_cap=200,
                covisibility_ab_enabled_min_active_observations=20,
                covisibility_ab_enabled_fallback="none",
                covisibility_ab_enabled_remove_outliers=False,
                covisibility_ab_enabled_max_outlier_observation_ratio="none",
                covisibility_ab_enabled_boundary_support_min_optimized_keyframes="none",
                covisibility_ab_enabled_boundary_support_min_fixed_keyframes=0,
                covisibility_mh05_mitigation=str(mitigation_summary),
                covisibility_mh05_quality_gate=str(quality_gate_summary),
                covisibility_mh05_boundary_support_gate=str(boundary_support_gate_summary),
                covisibility_mh05_boundary_support_gate_sweep=str(boundary_support_gate_sweep),
                covisibility_mh05_mitigation_registry_dir=str(mitigation_registry),
                covisibility_mh05_mitigation_sequence="MH_05_difficult",
                covisibility_mh05_mitigation_max_frames=8,
                covisibility_mh05_mitigation_neighbor_keyframes=10,
                covisibility_mh05_mitigation_boundary_keyframes=10,
                covisibility_mh05_mitigation_landmark_cap=200,
                covisibility_mh05_mitigation_min_keyframes=3,
                covisibility_mh05_mitigation_trigger_every=1,
                covisibility_mh05_mitigation_min_active_observations=20,
                covisibility_mh05_mitigation_fallback="none",
                covisibility_mh05_mitigation_remove_outliers=False,
                covisibility_mh05_mitigation_max_outlier_observation_ratio="none",
                covisibility_mh05_quality_gate_max_outlier_observation_ratio="0.3",
                covisibility_mh05_boundary_support_gate_max_outlier_observation_ratio="0.3",
                covisibility_mh05_mitigation_boundary_support_min_optimized_keyframes="none",
                covisibility_mh05_mitigation_boundary_support_min_fixed_keyframes=0,
                covisibility_mh05_quality_gate_boundary_support_min_optimized_keyframes="none",
                covisibility_mh05_quality_gate_boundary_support_min_fixed_keyframes=0,
                covisibility_mh05_boundary_support_gate_min_optimized_keyframes="10",
                covisibility_mh05_boundary_support_gate_min_fixed_keyframes=2,
                covisibility_mh05_mitigation_config=[
                    ("enabled min3/every1", 3, 1),
                    ("enabled min6/every3", 6, 3),
                    ("enabled min10/every5", 10, 5),
                ],
                covisibility_mh05_boundary_support_gate_config=[
                    ("enabled min3/every1 boundary10", 3, 1),
                ],
                covisibility_mh05_boundary_support_gate_sweep_gate=[
                    ("quality-gate only", "none", 0),
                    ("boundary7/2", "7", 2),
                    ("boundary10/2", "10", 2),
                ],
            )

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(check_generated(args), 0)

            (active_registry / "active-tracked-drop.json").unlink()
            custom_active_args.registry_dir = active_registry
            active_sweep.write_text(
                render_active_observation_sweep(
                    custom_active_args,
                    load_active_observation_runs(custom_active_args),
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(check_generated(args), 1)
            self.assertIn("missing active-observation sweep registry run(s)", stderr.getvalue())
            self.assertIn("variant=tracked_drop", stderr.getvalue())

            write_complete_active_sweep_manifests(active_registry, max_frames=5)
            active_sweep.write_text(
                render_active_observation_sweep(
                    custom_active_args,
                    load_active_observation_runs(custom_active_args),
                ),
                encoding="utf-8",
            )
            (runtime_registry / "runtime-200.json").unlink()
            runtime_sweep.write_text(
                render_covisibility_runtime_sweep(
                    custom_runtime_args,
                    load_covisibility_runtime_runs(custom_runtime_args),
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(check_generated(args), 1)
            self.assertIn("missing covisibility-runtime sweep registry run(s)", stderr.getvalue())
            self.assertIn("landmark_cap=200", stderr.getvalue())

            write_complete_runtime_sweep_manifests(runtime_registry, max_frames=5, caps=(100, 200))
            runtime_sweep.write_text(
                render_covisibility_runtime_sweep(
                    custom_runtime_args,
                    load_covisibility_runtime_runs(custom_runtime_args),
                ),
                encoding="utf-8",
            )
            (window_registry / "window-10-10.json").unlink()
            window_sweep.write_text(
                render_covisibility_window_sweep(
                    custom_window_args,
                    load_covisibility_window_runs(custom_window_args),
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(check_generated(args), 1)
            self.assertIn("missing covisibility-window sweep registry run(s)", stderr.getvalue())
            self.assertIn("neighbor=10 boundary=10", stderr.getvalue())

            write_complete_window_sweep_manifests(
                window_registry,
                max_frames=5,
                windows=((5, 5), (10, 10)),
            )
            window_sweep.write_text(
                render_covisibility_window_sweep(
                    custom_window_args,
                    load_covisibility_window_runs(custom_window_args),
                ),
                encoding="utf-8",
            )
            (window_validation_registry / "window-10-10.json").unlink()
            window_validation.write_text(
                render_covisibility_window_sweep(
                    custom_window_validation_args,
                    load_covisibility_window_runs(custom_window_validation_args),
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(check_generated(args), 1)
            self.assertIn("missing covisibility-window validation registry run(s)", stderr.getvalue())
            self.assertIn("neighbor=10 boundary=10", stderr.getvalue())

            write_complete_window_sweep_manifests(
                window_validation_registry,
                max_frames=6,
                windows=((5, 5), (10, 10)),
            )
            window_validation.write_text(
                render_covisibility_window_sweep(
                    custom_window_validation_args,
                    load_covisibility_window_runs(custom_window_validation_args),
                ),
                encoding="utf-8",
            )
            (ab_registry / "ab-enabled.json").unlink()
            ab_summary.write_text(
                render_covisibility_ab(
                    custom_ab_args,
                    load_covisibility_ab_runs(custom_ab_args),
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(check_generated(args), 1)
            self.assertIn("missing covisibility A/B registry run(s)", stderr.getvalue())
            self.assertIn("variant=enabled", stderr.getvalue())

            write_complete_covisibility_ab_manifests(ab_registry, max_frames=7)
            ab_summary.write_text(
                render_covisibility_ab(
                    custom_ab_args,
                    load_covisibility_ab_runs(custom_ab_args),
                ),
                encoding="utf-8",
            )
            (mitigation_registry / "mh05-min10-every5.json").unlink()
            mitigation_summary.write_text(
                render_covisibility_mh05_mitigation(
                    custom_mitigation_args,
                    load_covisibility_mh05_mitigation_runs(custom_mitigation_args),
                ),
                encoding="utf-8",
            )
            stderr = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(stderr):
                self.assertEqual(check_generated(args), 1)
            self.assertIn("missing covisibility MH_05 mitigation registry run(s)", stderr.getvalue())
            self.assertIn("config=enabled min10/every5", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
