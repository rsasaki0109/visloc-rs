import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "summarize_visual_slam_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "summarize_visual_slam_evaluation", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def experiment_manifest(*, schema_version: int = 4, include_gate: bool = True) -> dict:
    protocol = {
        "common_arguments": [],
        "variant_arguments": {
            "no_loop": [],
            "appearance_loop": [
                "--pose-graph-refinement",
                "--pose-graph-refinement-appearance-loops",
                "--pose-graph-refinement-fixed-loop-edge-weight",
                "1.0",
                "--pose-graph-refinement-gnc",
                "--pose-graph-refinement-pcm",
                "--pose-graph-refinement-pcm-pairwise-only",
            ],
        },
    }
    if include_gate:
        protocol["resource_gate"] = {
            "minimum_available_physical_bytes": 4 * 1024**3,
            "minimum_commit_headroom_bytes": 4 * 1024**3,
            "maximum_external_cpu_cores": 0.5,
            "sample_interval_seconds": 5,
        }
    return {
        "schema_version": schema_version,
        "executable_sha256": "1" * 64,
        "superpoint_model_sha256": "2" * 64,
        "ort_dylib_sha256": "3" * 64,
        "protocol": protocol,
    }


def run_summary(label: str, *, ate: float, rpe: float, accepted_loops: int):
    return MODULE.RunSummary(
        label=label,
        directory=label,
        frames=100,
        tracking_coverage=0.9,
        longest_continuous_frames=90,
        rigid_ate_m=ate,
        similarity_ate_m=None,
        similarity_scale=None,
        final_keyframe_rigid_ate_m=ate,
        final_keyframe_similarity_ate_m=None,
        final_keyframe_similarity_scale=None,
        rpe_delta1_translation_rmse_m=rpe,
        rpe_delta1_rotation_rmse_deg=rpe,
        rpe_delta10_translation_rmse_m=rpe,
        rpe_delta10_rotation_rmse_deg=rpe,
        accepted_loops=accepted_loops,
        evaluated_loops=accepted_loops,
        correct_loops=accepted_loops,
        loop_precision=1.0 if accepted_loops else None,
        wall_clock_ms_per_frame=10.0,
        frame_processing_ms_p95=12.0,
        sampled_peak_working_set_mb=100.0,
    )


class MatrixResourceGateTests(unittest.TestCase):
    def write_experiment(self, root: Path, manifest: dict) -> None:
        (root / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_schema_four_resource_gate_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest(include_gate=False))
            with self.assertRaisesRegex(ValueError, "missing its resource gate"):
                MODULE.validate_matrix_protocol(root)

    def test_schema_two_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest(schema_version=3))
            with self.assertRaisesRegex(ValueError, "schema_version=4"):
                MODULE.validate_matrix_protocol(root)

    def test_candidate_fixed_weight_requires_one_finite_positive_value(self) -> None:
        invalid_argument_lists = (
            ["--pose-graph-refinement-fixed-loop-edge-weight"],
            ["--pose-graph-refinement-fixed-loop-edge-weight", "nan"],
            ["--pose-graph-refinement-fixed-loop-edge-weight", "0"],
            [
                "--pose-graph-refinement-fixed-loop-edge-weight",
                "1",
                "--pose-graph-refinement-fixed-loop-edge-weight",
                "0.1",
            ],
        )
        for arguments in invalid_argument_lists:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                manifest = experiment_manifest()
                manifest["protocol"]["variant_arguments"]["appearance_loop"] = [
                    "--pose-graph-refinement",
                    "--pose-graph-refinement-appearance-loops",
                    "--pose-graph-refinement-gnc",
                    "--pose-graph-refinement-pcm",
                    "--pose-graph-refinement-pcm-pairwise-only",
                    *arguments,
                ]
                self.write_experiment(root, manifest)
                with self.assertRaisesRegex(ValueError, "fixed loop-edge weight"):
                    MODULE.validate_matrix_protocol(root)

    def test_covariance_information_protocol_replaces_fixed_weight(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest = experiment_manifest()
            protocol = manifest["protocol"]
            protocol["candidate_loop_pose_information"] = True
            appearance = protocol["variant_arguments"]["appearance_loop"]
            weight_index = appearance.index(
                "--pose-graph-refinement-fixed-loop-edge-weight"
            )
            del appearance[weight_index : weight_index + 2]
            appearance.append("--pose-graph-refinement-loop-pose-information")
            self.write_experiment(root, manifest)

            validated = MODULE.validate_matrix_protocol(root)
            self.assertIsNone(validated[2])
            self.assertTrue(validated[3])

    def test_covariance_information_protocol_rejects_fixed_weight_mixture(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            manifest = experiment_manifest()
            manifest["protocol"]["candidate_loop_pose_information"] = True
            manifest["protocol"]["variant_arguments"]["appearance_loop"].append(
                "--pose-graph-refinement-loop-pose-information"
            )
            self.write_experiment(root, manifest)
            with self.assertRaisesRegex(ValueError, "no fixed loop-edge weight"):
                MODULE.validate_matrix_protocol(root)

    def test_external_cpu_limit_must_be_finite_and_positive(self) -> None:
        for value in (None, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                manifest = experiment_manifest()
                manifest["protocol"]["resource_gate"][
                    "maximum_external_cpu_cores"
                ] = value
                self.write_experiment(root, manifest)
                with self.assertRaisesRegex(ValueError, "external-CPU limit"):
                    MODULE.validate_matrix_protocol(root)

    def test_experiment_artifact_hashes_must_be_sha256(self) -> None:
        for value in (None, "", "xyz", "1" * 63):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as raw_root:
                root = Path(raw_root)
                manifest = experiment_manifest()
                manifest["executable_sha256"] = value
                self.write_experiment(root, manifest)
                with self.assertRaisesRegex(ValueError, "invalid executable_sha256"):
                    MODULE.validate_matrix_protocol(root)

    def test_completed_run_must_satisfy_recorded_resource_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest())
            run_dir = root / "MH_01_easy_no_loop_r01"
            run_dir.mkdir()
            (run_dir / "summary.txt").write_text("", encoding="utf-8")
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "name": run_dir.name,
                        "sequence": "MH_01_easy",
                        "variant": "no_loop",
                        "repetition": 1,
                        "exit_code": 0,
                        "validation_error": None,
                        "executable_sha256": "1" * 64,
                        "superpoint_model_sha256": "2" * 64,
                        "ort_dylib_sha256": "3" * 64,
                        "minimum_available_physical_bytes": 4 * 1024**3 - 1,
                        "minimum_commit_headroom_bytes": 4 * 1024**3,
                        "preflight_external_cpu_cores": 0.0,
                        "sampled_max_external_cpu_cores": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "physical-memory minimum"):
                MODULE.load_matrix(root)

    def test_completed_candidate_summary_weight_must_match_protocol(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest())
            run_dir = root / "MH_01_easy_appearance_loop_r01"
            run_dir.mkdir()
            (run_dir / "summary.txt").write_text(
                "pose_graph_refinement=true\n"
                "pose_graph_refinement_fixed_loop_edge_weight=Some(0.1)\n",
                encoding="utf-8",
            )
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "name": run_dir.name,
                        "sequence": "MH_01_easy",
                        "variant": "appearance_loop",
                        "repetition": 1,
                        "exit_code": 0,
                        "validation_error": None,
                        "executable_sha256": "1" * 64,
                        "superpoint_model_sha256": "2" * 64,
                        "ort_dylib_sha256": "3" * 64,
                        "minimum_available_physical_bytes": 4 * 1024**3,
                        "minimum_commit_headroom_bytes": 4 * 1024**3,
                        "preflight_external_cpu_cores": 0.0,
                        "sampled_max_external_cpu_cores": 0.0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match protocol"):
                MODULE.load_matrix(root)

    def test_completed_run_must_satisfy_external_cpu_limit(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest())
            run_dir = root / "MH_01_easy_no_loop_r01"
            run_dir.mkdir()
            (run_dir / "summary.txt").write_text(
                "pose_graph_refinement=false\n"
                "pose_graph_refinement_fixed_loop_edge_weight=None\n",
                encoding="utf-8",
            )
            (run_dir / "run_manifest.json").write_text(
                json.dumps(
                    {
                        "name": run_dir.name,
                        "sequence": "MH_01_easy",
                        "variant": "no_loop",
                        "repetition": 1,
                        "exit_code": 0,
                        "validation_error": None,
                        "executable_sha256": "1" * 64,
                        "superpoint_model_sha256": "2" * 64,
                        "ort_dylib_sha256": "3" * 64,
                        "minimum_available_physical_bytes": 4 * 1024**3,
                        "minimum_commit_headroom_bytes": 4 * 1024**3,
                        "preflight_external_cpu_cores": 0.1,
                        "sampled_max_external_cpu_cores": 0.6,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "external-CPU sampled maximum"):
                MODULE.load_matrix(root)

    def test_completed_run_artifact_hash_must_match_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest())
            run_dir = root / "MH_01_easy_no_loop_r01"
            run_dir.mkdir()
            (run_dir / "summary.txt").write_text("", encoding="utf-8")
            manifest = {
                "name": run_dir.name,
                "sequence": "MH_01_easy",
                "variant": "no_loop",
                "repetition": 1,
                "exit_code": 0,
                "validation_error": None,
                "executable_sha256": "0" * 64,
                "superpoint_model_sha256": "2" * 64,
                "ort_dylib_sha256": "3" * 64,
            }
            (run_dir / "run_manifest.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "executable_sha256 does not match"):
                MODULE.load_matrix(root)


class NumericParsingTests(unittest.TestCase):
    def test_nonfinite_metrics_are_treated_as_missing(self) -> None:
        for value in ("nan", "NaN", "inf", "+inf", "-inf", "1e999", "Some(inf)"):
            with self.subTest(value=value):
                self.assertIsNone(MODULE.optional_float(value))

        self.assertEqual(MODULE.optional_float("Some(1.25)"), 1.25)


class PromotionRpeGateTests(unittest.TestCase):
    def verdict(self, candidate_rpe: float):
        runs = []
        groups = {}
        for repetition in range(1, 4):
            baseline_label = f"baseline_{repetition}"
            candidate_label = f"candidate_{repetition}"
            runs.append(
                run_summary(
                    baseline_label, ate=1.0, rpe=1.0, accepted_loops=0
                )
            )
            runs.append(
                run_summary(
                    candidate_label,
                    ate=0.8,
                    rpe=candidate_rpe,
                    accepted_loops=1,
                )
            )
            groups[baseline_label] = MODULE.RunIdentity(
                "MH_01_easy", "no_loop", repetition
            )
            groups[candidate_label] = MODULE.RunIdentity(
                "MH_01_easy", "appearance_loop", repetition
            )
        verdicts = MODULE.promotion_verdicts(runs, groups)
        self.assertEqual(len(verdicts), 1)
        return verdicts[0]

    def test_more_than_one_percent_rpe_regression_rejects_candidate(self) -> None:
        verdict = self.verdict(1.02)
        self.assertEqual(verdict.status, "REJECT")
        self.assertIn(
            "delta-1 translation RPE regresses by more than 1%", verdict.reasons
        )
        self.assertIn(
            "delta-10 rotation RPE regresses by more than 1%", verdict.reasons
        )

    def test_one_percent_rpe_tolerance_is_inclusive(self) -> None:
        verdict = self.verdict(1.01)
        self.assertEqual(verdict.status, "PROMOTE")

    def test_nonfinite_or_negative_rpe_is_incomplete(self) -> None:
        for value in (float("nan"), float("inf"), -0.1):
            with self.subTest(value=value):
                verdict = self.verdict(value)
                self.assertEqual(verdict.status, "INCOMPLETE")
                self.assertIn("missing delta-1 translation RPE", verdict.reasons)


if __name__ == "__main__":
    unittest.main()
