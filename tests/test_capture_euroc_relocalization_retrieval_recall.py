from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_euroc_relocalization_retrieval_recall import (  # noqa: E402
    build_capture_cmd,
    build_diagnostic_cmd,
    build_eval_cmd,
    choose_primary_metric,
    diagnostic_metric_args,
    parse_number,
    parse_summary,
    retrieval_metric_args,
    status_from_result,
    summary_metric_args,
)


def args_namespace() -> Namespace:
    return Namespace(
        sequence="MH_03_medium",
        candidates=[Path("target/run/relocalization_appearance_candidates.csv")],
        poses=Path("target/run/frame_groundtruth.csv"),
        run_dir=Path("target/run"),
        dataset_path=None,
        dataset_version="ASL EuRoC MAV dataset",
        input_kind="relocalization_appearance_candidates",
        distance_threshold_m=1.0,
        min_temporal_gap=30,
        min_path_length_m=None,
        ks=[20, 1, 5],
        query_ids=None,
        all_pose_queries=False,
        registry_dir=Path("benchmarks/registry/runs/euroc"),
        benchmark_id="euroc-relocalization-appearance-store",
        benchmark_name="EuRoC relocalization appearance retrieval recall",
        protocol="pose-derived true-revisit recall@K over EuRoC relocalization appearance candidates; no recovery PnP score",
        command=None,
        profile="release",
        feature=["image-io"],
        result_kind="visloc_run",
        claim_scope="negative",
        status="success",
        failure_reason=None,
        config=[],
        extra_artifact=[],
        notes=None,
        primary_recall_k=None,
    )


def recall_result() -> dict:
    return {
        "frontends": [
            {
                "frontend": "all",
                "candidate_count": 2991,
                "query_count": 195,
                "queries_with_candidates": 195,
                "eligible_query_count": 107,
                "recall_at_k": {"1": 0.3458, "5": 0.3458, "20": 0.3551},
                "mean_precision_at_k": {"1": 0.3458, "5": 0.0692, "20": 0.0178},
                "mrr": 0.3464,
                "mean_first_relevant_rank": 1.3947,
                "top1_false_positive_rate": 0.6542,
            }
        ]
    }


def diagnostic_result() -> dict:
    return {
        "frontends": [
            {
                "frontend": "all",
                "attempt_count": 19,
                "success_count": 1,
                "gate_pass_count": 1,
                "top1_relevant_count": 19,
                "any_relevant_count": 19,
                "top1_relevant_rejected_count": 18,
            }
        ]
    }


class CaptureEurocRelocalizationRetrievalRecallTest(unittest.TestCase):
    def test_eval_command_scopes_to_candidate_queries_by_default(self) -> None:
        args = args_namespace()
        cmd = build_eval_cmd(args, Path("target/out/r.json"), Path("target/out/r.md"))

        self.assertIn("scripts/eval_loop_retrieval_recall.py", cmd)
        self.assertIn("--query-ids-from-candidates", cmd)
        self.assertEqual(cmd[cmd.index("--ks") + 1 : cmd.index("--out-json")], ["1", "5", "20"])
        self.assertIn("relocalization_appearance_candidates", cmd)

    def test_eval_command_can_use_all_pose_queries(self) -> None:
        args = args_namespace()
        args.all_pose_queries = True
        cmd = build_eval_cmd(args, Path("target/out/r.json"), Path("target/out/r.md"))

        self.assertNotIn("--query-ids-from-candidates", cmd)

    def test_diagnostic_command_uses_same_pose_gates(self) -> None:
        args = args_namespace()
        cmd = build_diagnostic_cmd(
            args,
            Path("target/out/d.json"),
            Path("target/out/d.md"),
            Path("target/out/d.csv"),
        )

        self.assertIn("scripts/diagnose_relocalization_candidates.py", cmd)
        self.assertIn("--distance-threshold-m", cmd)
        self.assertIn("1.0", cmd)
        self.assertIn("--min-temporal-gap", cmd)
        self.assertIn("30", cmd)

    def test_summary_parser_and_metric_args_handle_some_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / "summary.txt"
            summary.write_text(
                "\n".join(
                    [
                        "tracking_success_rate=0.185",
                        "relocalization_attempts=197",
                        "relocalization_successes=1",
                        "relocalization_descriptor_store_landmark_count_mean=Some(247.5)",
                        "ate_similarity_scale=None",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            parsed = parse_summary(summary)
            metrics = summary_metric_args(parsed)

        self.assertEqual(parse_number("Some(247.5)"), 247.5)
        self.assertIsNone(parse_number("None"))
        self.assertIn("tracking_success_rate=0.185:ratio", metrics)
        self.assertIn("relocalization_attempts=197:count", metrics)
        self.assertIn("relocalization_descriptor_store_landmark_count_mean=247.5:count", metrics)
        self.assertNotIn("ate_similarity_scale=None:ratio", metrics)

    def test_retrieval_metrics_use_plain_retrieval_names(self) -> None:
        args = retrieval_metric_args(recall_result(), "retrieval_recall_at_1")

        self.assertIn("--primary-metric", args)
        self.assertIn("retrieval_recall_at_1", args)
        self.assertIn("retrieval_candidate_count=2991:count", args)
        self.assertIn("retrieval_query_count=195:count", args)
        self.assertIn("retrieval_recall_at_20=0.3551:ratio", args)
        self.assertIn("retrieval_mean_precision_at_5=0.0692:ratio", args)

    def test_diagnostic_metrics_are_aggregated_for_registry(self) -> None:
        args = diagnostic_metric_args(diagnostic_result())

        self.assertIn("candidate_diag_attempt_count=19:count", args)
        self.assertIn("candidate_diag_success_count=1:count", args)
        self.assertIn("candidate_diag_top1_relevant_count=19:count", args)
        self.assertIn("candidate_diag_top1_relevant_rejected_count=18:count", args)
        self.assertIn("candidate_diag_top1_relevant_rate=1.0:ratio", args)

    def test_primary_metric_defaults_to_smallest_k(self) -> None:
        args = args_namespace()

        self.assertEqual(choose_primary_metric(args), "retrieval_recall_at_1")

        args.primary_recall_k = 20
        self.assertEqual(choose_primary_metric(args), "retrieval_recall_at_20")

    def test_status_from_result_distinguishes_dnf_gates(self) -> None:
        self.assertEqual(status_from_result(7, None, "success", None, [])[0], "failure")
        self.assertEqual(status_from_result(0, {"frontends": []}, "success", None, [])[0], "dnf")

        status, reason = status_from_result(0, recall_result(), "success", None, [(20, 0.5)])

        self.assertEqual(status, "dnf")
        self.assertIn("recall@20=0.3551 below 0.5", reason or "")

    def test_capture_command_records_artifacts_configs_and_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "run"
            run_dir.mkdir()
            (run_dir / "summary.txt").write_text(
                "tracking_success_rate=0.185\nrelocalization_successes=1\n",
                encoding="utf-8",
            )
            (run_dir / "slam_trajectory.csv").write_text("timestamp_ns,frame_idx\n", encoding="utf-8")
            (run_dir / "relocalization_attempts.csv").write_text(
                "frame_idx,timestamp_ns,reject_reason\n",
                encoding="utf-8",
            )
            candidate = run_dir / "relocalization_appearance_candidates.csv"
            candidate.write_text("matched_keyframe_id,query_frame_id,score\n", encoding="utf-8")
            poses = run_dir / "frame_groundtruth.csv"
            poses.write_text("frame_idx,gt_px,gt_py,gt_pz\n", encoding="utf-8")

            args = args_namespace()
            args.run_dir = run_dir
            args.candidates = [candidate]
            args.poses = poses
            summary = parse_summary(run_dir / "summary.txt")
            eval_cmd = build_eval_cmd(args, root / "r.json", root / "r.md")

            cmd = build_capture_cmd(
                args,
                run_id="run-1",
                eval_cmd=eval_cmd,
                result=recall_result(),
                summary=summary,
                out_json=root / "r.json",
                out_md=root / "r.md",
                diag_result=diagnostic_result(),
                diag_json=root / "d.json",
                diag_md=root / "d.md",
                diag_csv=root / "d.csv",
                status="success",
                failure_reason=None,
            )

        joined = " ".join(cmd).replace("\\", "/")
        self.assertIn("scripts/benchmark_registry.py", cmd)
        self.assertIn("euroc-relocalization-appearance-store", cmd)
        self.assertIn("candidate_csv_0=", joined)
        self.assertIn("summary=", joined)
        self.assertIn("trajectory=", joined)
        self.assertIn("relocalization_attempts=", joined)
        self.assertIn("candidate_diagnostics_json=", joined)
        self.assertIn("candidate_diagnostics_csv=", joined)
        self.assertIn("query_scope=candidate_queries", cmd)
        self.assertIn("tracking_success_rate=0.185:ratio", cmd)
        self.assertIn("retrieval_recall_at_1=0.3458:ratio", cmd)
        self.assertIn("candidate_diag_top1_relevant_rejected_count=18:count", cmd)

    def test_capture_command_records_extra_artifacts(self) -> None:
        args = args_namespace()
        args.extra_artifact = ["frame_appearance_descriptors=target/run/frame_appearance_descriptors.csv"]

        cmd = build_capture_cmd(
            args,
            run_id="run-1",
            eval_cmd=build_eval_cmd(args, Path("target/out/r.json"), Path("target/out/r.md")),
            result=recall_result(),
            summary={},
            out_json=Path("target/out/r.json"),
            out_md=Path("target/out/r.md"),
            status="success",
            failure_reason=None,
        )

        self.assertIn("frame_appearance_descriptors=target/run/frame_appearance_descriptors.csv", cmd)


if __name__ == "__main__":
    unittest.main()
