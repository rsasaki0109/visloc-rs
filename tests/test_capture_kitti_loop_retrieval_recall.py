from __future__ import annotations

import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_kitti_loop_retrieval_recall import (  # noqa: E402
    build_capture_cmd,
    build_eval_cmd,
    choose_primary_metric,
    discover_verification_diagnostics,
    metric_args,
    parse_gate,
    sanitize_metric_prefix,
    status_from_result,
)


def args_namespace() -> Namespace:
    return Namespace(
        sequence="02",
        candidates=[Path("target/seq02/loop_candidates.csv")],
        poses=Path("datasets/kitti_seq02/poses_02.txt"),
        dataset_path=None,
        dataset_version="KITTI odometry grayscale",
        input_kind="raw_appearance_candidates",
        distance_threshold_m=10.0,
        min_temporal_gap=50,
        min_path_length_m=5.0,
        ks=[20, 1, 5],
        query_ids=None,
        query_ids_from_candidates=False,
        registry_dir=Path("benchmarks/registry/runs/kitti"),
    )


def recall_result() -> dict:
    return {
        "frontends": [
            {
                "frontend": "vlad(k=64)",
                "candidate_count": 42,
                "eligible_query_count": 10,
                "recall_at_k": {"1": 0.1, "5": 0.4, "20": 0.8},
                "mean_precision_at_k": {"1": 0.1, "5": 0.08, "20": 0.04},
                "mrr": 0.25,
                "mean_first_relevant_rank": 6.5,
                "top1_false_positive_rate": 0.9,
            }
        ]
    }


class CaptureKittiLoopRetrievalRecallTest(unittest.TestCase):
    def test_eval_command_contains_core_gates_and_outputs(self) -> None:
        args = args_namespace()
        cmd = build_eval_cmd(
            args,
            Path("target/out/retrieval_recall.json"),
            Path("target/out/retrieval_recall.md"),
        )

        self.assertIn("scripts/eval_loop_retrieval_recall.py", cmd)
        self.assertIn("--candidates", cmd)
        self.assertIn("target/seq02/loop_candidates.csv", {part.replace("\\", "/") for part in cmd})
        self.assertIn("--min-path-length-m", cmd)
        self.assertIn("5.0", cmd)
        self.assertEqual(cmd[cmd.index("--ks") + 1 : cmd.index("--out-json")], ["1", "5", "20"])

    def test_eval_command_can_scope_queries_to_candidate_rows(self) -> None:
        args = args_namespace()
        args.query_ids_from_candidates = True
        cmd = build_eval_cmd(
            args,
            Path("target/out/retrieval_recall.json"),
            Path("target/out/retrieval_recall.md"),
        )

        self.assertIn("--query-ids-from-candidates", cmd)

    def test_metric_names_are_sanitized_and_primary_uses_largest_k(self) -> None:
        result = recall_result()

        self.assertEqual(sanitize_metric_prefix("vlad(k=64)"), "vlad_k_64")
        self.assertEqual(choose_primary_metric(result, [1, 5, 20]), "vlad_k_64_recall_at_20")

        metrics = metric_args(result, "vlad_k_64_recall_at_20")

        self.assertIn("--primary-metric", metrics)
        self.assertIn("vlad_k_64_recall_at_20", metrics)
        self.assertIn("vlad_k_64_recall_at_20=0.8:ratio", metrics)
        self.assertIn("vlad_k_64_candidate_count=42:count", metrics)
        self.assertIn("vlad_k_64_mean_first_relevant_rank=6.5:rank", metrics)

    def test_status_from_result_distinguishes_success_dnf_and_failure(self) -> None:
        self.assertEqual(status_from_result(7, None, [])[0], "failure")
        self.assertEqual(status_from_result(0, None, [])[0], "failure")
        self.assertEqual(status_from_result(0, {"frontends": []}, [])[0], "dnf")
        self.assertEqual(
            status_from_result(0, {"frontends": [{"eligible_query_count": 0}]}, [])[0],
            "dnf",
        )

        status, reason = status_from_result(0, recall_result(), [(20, 0.9)])
        self.assertEqual(status, "dnf")
        self.assertIn("recall@20=0.8 below 0.9", reason or "")

        self.assertEqual(status_from_result(0, recall_result(), [(20, 0.8)]), ("success", None))

    def test_capture_command_records_artifacts_and_metrics(self) -> None:
        args = args_namespace()
        eval_cmd = build_eval_cmd(args, Path("target/out/r.json"), Path("target/out/r.md"))

        cmd = build_capture_cmd(
            args,
            run_id="run-1",
            eval_cmd=eval_cmd,
            result=recall_result(),
            out_json=Path("target/out/r.json"),
            out_md=Path("target/out/r.md"),
            status="success",
            failure_reason=None,
        )

        joined = " ".join(cmd)
        self.assertIn("scripts/benchmark_registry.py", cmd)
        self.assertIn("--benchmark-id", cmd)
        self.assertIn("kitti-loop-retrieval-recall", cmd)
        self.assertIn("candidate_csv_0=target/seq02/loop_candidates.csv", joined.replace("\\", "/"))
        self.assertIn("recall_json=target/out/r.json", joined.replace("\\", "/"))
        self.assertIn("vlad_k_64_recall_at_20=0.8:ratio", cmd)

    def test_capture_command_records_candidate_query_scope(self) -> None:
        args = args_namespace()
        args.query_ids_from_candidates = True
        eval_cmd = build_eval_cmd(args, Path("target/out/r.json"), Path("target/out/r.md"))

        cmd = build_capture_cmd(
            args,
            run_id="run-1",
            eval_cmd=eval_cmd,
            result=recall_result(),
            out_json=Path("target/out/r.json"),
            out_md=Path("target/out/r.md"),
            status="success",
            failure_reason=None,
        )

        self.assertIn("query_ids_from_candidates=true", cmd)

    def test_capture_command_records_sibling_verification_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate = root / "loop_candidates.csv"
            diagnostic = root / "loop_candidate_verifications.csv"
            candidate.write_text("frontend,matched_keyframe_id,query_frame_id,score\n")
            diagnostic.write_text(
                "frontend,matched_keyframe_id,query_frame_id,score,attempted\n"
            )
            args = args_namespace()
            args.candidates = [candidate]
            eval_cmd = build_eval_cmd(args, root / "r.json", root / "r.md")

            self.assertEqual(discover_verification_diagnostics([candidate]), [diagnostic])

            cmd = build_capture_cmd(
                args,
                run_id="run-1",
                eval_cmd=eval_cmd,
                result=recall_result(),
                out_json=root / "r.json",
                out_md=root / "r.md",
                status="success",
                failure_reason=None,
            )

        joined = " ".join(cmd).replace("\\", "/")
        self.assertIn("verification_diagnostics_csv_0=", joined)
        self.assertIn("loop_candidate_verifications.csv", joined)

    def test_parse_gate(self) -> None:
        self.assertEqual(parse_gate("20=0.01"), (20, 0.01))
        with self.assertRaises(Exception):
            parse_gate("20")


if __name__ == "__main__":
    unittest.main()
