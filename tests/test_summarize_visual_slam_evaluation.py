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


def experiment_manifest(*, schema_version: int = 9, include_gate: bool = True) -> dict:
    protocol = {
        "candidate_loop_pose_information": False,
        "candidate_loop_information_max_eigenvalue": 1.0,
        "candidate_loop_information_loop_edge_scale": 1.0,
        "candidate_fuse_loop_observations": False,
        "candidate_loop_welding_ba": False,
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
            "external_cpu_violation_samples": 3,
            "sample_interval_seconds": 5,
        }
    return {
        "schema_version": schema_version,
        "executable_sha256": "1" * 64,
        "superpoint_model_sha256": "2" * 64,
        "ort_dylib_sha256": "3" * 64,
        "protocol": protocol,
    }


def run_summary(
    label: str,
    *,
    ate: float,
    rpe: float,
    accepted_loops: int,
    accuracy_hashes: dict[str, str] | None = None,
):
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
        accuracy_artifact_sha256=accuracy_hashes,
    )


class MatrixResourceGateTests(unittest.TestCase):
    def write_experiment(self, root: Path, manifest: dict) -> None:
        (root / "experiment_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

    def test_schema_nine_resource_gate_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest(include_gate=False))
            with self.assertRaisesRegex(ValueError, "missing its resource gate"):
                MODULE.validate_matrix_protocol(root)

    def test_old_schema_matrix_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            self.write_experiment(root, experiment_manifest(schema_version=3))
            with self.assertRaisesRegex(ValueError, "schema_version=9"):
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
            appearance.extend(
                [
                    "--pose-graph-refinement-loop-pose-information",
                    "--pose-graph-refinement-loop-pose-information-max-eigenvalue",
                    "1.0",
                    "--pose-graph-refinement-loop-pose-information-loop-edge-scale",
                    "1.0",
                ]
            )
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
                        "maximum_consecutive_external_cpu_violations": 0,
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
                        "maximum_consecutive_external_cpu_violations": 0,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not match protocol"):
                MODULE.load_matrix(root)

    def test_completed_run_must_satisfy_sustained_external_cpu_limit(self) -> None:
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
                        "maximum_consecutive_external_cpu_violations": 3,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "sustained external-CPU load"):
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


class PoseInformationFailureCountTests(unittest.TestCase):
    VALID = (
        "invalid_config:0,missing_keyframe:0,missing_pose:0,"
        "insufficient_correspondences:3,rank_deficient:1,"
        "ill_conditioned:2,unsupported_solver:0"
    )

    def test_parses_complete_failure_histogram(self) -> None:
        counts = MODULE.parse_pose_information_failure_counts(self.VALID)
        self.assertIsNotNone(counts)
        self.assertEqual(counts["insufficient_correspondences"], 3)
        self.assertEqual(sum(counts.values()), 6)

    def test_rejects_malformed_duplicate_unknown_and_missing_entries(self) -> None:
        malformed = (
            "invalid_config=0",
            self.VALID + ",invalid_config:1",
            self.VALID.replace("invalid_config:0", "other:0"),
            self.VALID.replace("invalid_config:0,", ""),
            self.VALID.replace("rank_deficient:1", "rank_deficient:-1"),
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                MODULE.parse_pose_information_failure_counts(value)

    def test_rejects_histogram_total_mismatch(self) -> None:
        values = {
            "pose_graph_refinement_pose_information_failure_counts": self.VALID,
            "pose_graph_refinement_pose_information_rejected": "5",
            "pose_graph_refinement_sequential_pose_information_failure_counts": self.VALID,
            "pose_graph_refinement_sequential_pose_information_fallbacks": "6",
        }
        with self.assertRaisesRegex(ValueError, "sums to 6"):
            MODULE.validate_pose_information_failure_counts(values, Path("run"))

    def test_aggregate_sums_each_failure_reason(self) -> None:
        first = MODULE.parse_pose_information_failure_counts(self.VALID)
        second = dict(first)
        second["ill_conditioned"] = 5
        total = MODULE.sum_pose_information_failure_counts([first, second, None])
        self.assertEqual(total["insufficient_correspondences"], 6)
        self.assertEqual(total["ill_conditioned"], 7)


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


class PromotionExactNoOpGateTests(unittest.TestCase):
    def test_accuracy_artifact_hashes_cover_every_required_file(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            for name in MODULE.ACCURACY_ARTIFACTS:
                (directory / name).write_bytes(f"content:{name}".encode())
            first = MODULE.accuracy_artifact_hashes(directory)
            self.assertEqual(set(first), set(MODULE.ACCURACY_ARTIFACTS))
            (directory / MODULE.ACCURACY_ARTIFACTS[0]).write_bytes(b"changed")
            second = MODULE.accuracy_artifact_hashes(directory)
            self.assertNotEqual(first, second)
            (directory / MODULE.ACCURACY_ARTIFACTS[-1]).unlink()
            self.assertIsNone(MODULE.accuracy_artifact_hashes(directory))

    def verdict(
        self,
        *,
        candidate_hash_suffix: str = "same",
        per_repetition_metric: tuple[float, float, float] = (1.0, 1.0, 1.0),
        declared_repetitions: int | None = None,
    ):
        runs = []
        groups = {}
        baseline_hashes = {
            name: f"{name}:same" for name in MODULE.ACCURACY_ARTIFACTS
        }
        candidate_hashes = dict(baseline_hashes)
        if candidate_hash_suffix != "same":
            candidate_hashes[MODULE.ACCURACY_ARTIFACTS[0]] = candidate_hash_suffix
        for repetition, metric in enumerate(per_repetition_metric, start=1):
            baseline_label = f"baseline_{repetition}"
            candidate_label = f"candidate_{repetition}"
            runs.append(
                run_summary(
                    baseline_label,
                    ate=metric,
                    rpe=metric,
                    accepted_loops=0,
                    accuracy_hashes=dict(baseline_hashes),
                )
            )
            runs.append(
                run_summary(
                    candidate_label,
                    ate=metric,
                    rpe=metric,
                    accepted_loops=0,
                    accuracy_hashes=dict(candidate_hashes),
                )
            )
            groups[baseline_label] = MODULE.RunIdentity(
                "MH_03_medium", "no_loop", repetition
            )
            groups[candidate_label] = MODULE.RunIdentity(
                "MH_03_medium", "appearance_loop", repetition
            )
        expected = (
            {
                MODULE.RunIdentity("MH_03_medium", variant, repetition)
                for variant in ("no_loop", "appearance_loop")
                for repetition in range(1, declared_repetitions + 1)
            }
            if declared_repetitions is not None
            else None
        )
        return MODULE.promotion_verdicts(runs, groups, expected)[0]

    def test_three_byte_identical_pairs_are_a_safe_no_op(self) -> None:
        verdict = self.verdict()
        self.assertEqual(verdict.status, "SAFE_NO_OP")
        self.assertEqual(verdict.exact_no_op_pairs, 3)
        self.assertEqual(verdict.reasons, [])

    def test_exact_pairs_remain_safe_across_between_repetition_variation(self) -> None:
        verdict = self.verdict(per_repetition_metric=(1.0, 1.2, 0.8))
        self.assertEqual(verdict.status, "SAFE_NO_OP")

    def test_missing_manifest_declared_repetition_is_incomplete(self) -> None:
        verdict = self.verdict(declared_repetitions=4)
        self.assertEqual(verdict.status, "INCOMPLETE")
        self.assertIn("missing declared repetitions", verdict.reasons)

    def test_one_mismatched_accuracy_artifact_rejects_zero_loop_candidate(self) -> None:
        verdict = self.verdict(candidate_hash_suffix="different")
        self.assertEqual(verdict.status, "REJECT")
        self.assertEqual(verdict.exact_no_op_pairs, 0)
        self.assertIn("zero accepted loops", verdict.reasons)
        self.assertIn(
            "candidate worst final-keyframe rigid ATE does not beat baseline best",
            verdict.reasons,
        )


class PromotionPairedResourceGateTests(unittest.TestCase):
    def verdict(self, candidate_ratios: tuple[float, float, float]):
        runs = []
        groups = {}
        baseline_runtime = (100.0, 50.0, 200.0)
        for repetition, (baseline_ms, ratio) in enumerate(
            zip(baseline_runtime, candidate_ratios), start=1
        ):
            baseline_label = f"baseline_{repetition}"
            candidate_label = f"candidate_{repetition}"
            baseline_run = run_summary(
                baseline_label, ate=1.0, rpe=1.0, accepted_loops=0
            )
            candidate_run = run_summary(
                candidate_label, ate=0.8, rpe=1.0, accepted_loops=1
            )
            baseline_run.wall_clock_ms_per_frame = baseline_ms
            candidate_run.wall_clock_ms_per_frame = baseline_ms * ratio
            runs.extend((baseline_run, candidate_run))
            groups[baseline_label] = MODULE.RunIdentity(
                "MH_01_easy", "no_loop", repetition
            )
            groups[candidate_label] = MODULE.RunIdentity(
                "MH_01_easy", "appearance_loop", repetition
            )
        return MODULE.promotion_verdicts(runs, groups)[0]

    def test_counterbalanced_runs_compare_within_repetition(self) -> None:
        verdict = self.verdict((1.2, 1.2, 1.2))
        self.assertEqual(verdict.status, "PROMOTE")

    def test_worst_paired_runtime_overhead_above_limit_rejects(self) -> None:
        verdict = self.verdict((1.0, 1.251, 1.0))
        self.assertEqual(verdict.status, "REJECT")
        self.assertIn("paired runtime regresses by more than 25%", verdict.reasons)


class MatrixDecisionTests(unittest.TestCase):
    class Verdict:
        def __init__(self, status: str):
            self.status = status

    def status(self, *statuses: str):
        return MODULE.matrix_promotion_status(
            [self.Verdict(status) for status in statuses]
        )

    def test_promote_plus_safe_no_ops_promotes_matrix(self) -> None:
        self.assertEqual(
            self.status("PROMOTE", "SAFE_NO_OP", "SAFE_NO_OP"), "PROMOTE"
        )

    def test_incomplete_precedes_reject_until_matrix_is_complete(self) -> None:
        self.assertEqual(self.status("REJECT", "INCOMPLETE"), "INCOMPLETE")

    def test_all_safe_no_ops_do_not_claim_improvement(self) -> None:
        self.assertEqual(self.status("SAFE_NO_OP", "SAFE_NO_OP"), "SAFE_NO_OP")

    def test_expected_identities_expand_manifest_sequences_and_repetitions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            root = Path(raw_root)
            (root / "experiment_manifest.json").write_text(
                json.dumps(
                    {
                        "sequences": ["MH_01_easy", "MH_03_medium"],
                        "repetitions": 3,
                    }
                ),
                encoding="utf-8",
            )
            identities = MODULE.expected_matrix_identities(root)
            self.assertEqual(len(identities), 12)
            self.assertIn(
                MODULE.RunIdentity("MH_03_medium", "appearance_loop", 3), identities
            )


if __name__ == "__main__":
    unittest.main()
