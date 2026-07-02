from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_euroc_descriptor_retrieval_recall import (  # noqa: E402
    BENCHMARK_ID,
    build_capture_cmd,
    build_export_cmd,
)


def args_namespace() -> Namespace:
    return Namespace(
        sequence="MH_03_medium",
        descriptors=Path("target/run/frame_appearance_descriptors.csv"),
        frame_groundtruth=Path("target/run/frame_groundtruth.csv"),
        keyframe_decisions=Path("target/run/keyframe_decisions.csv"),
        database_ids=None,
        query_ids="100,105",
        run_dir=Path("target/run"),
        dataset_path=None,
        dataset_version="ASL EuRoC MAV dataset",
        top_k=20,
        exclude_recent_frame_gap=30,
        min_similarity=0.2,
        frontend="mean_hog",
        no_normalize=False,
        distance_threshold_m=1.0,
        min_temporal_gap=30,
        min_path_length_m=None,
        ks=[20, 1, 5],
        all_pose_queries=False,
        registry_dir=Path("benchmarks/registry/runs/euroc"),
        command=None,
        profile="release",
        feature=["image-io"],
        result_kind="visloc_run",
        claim_scope="exploratory",
        status="success",
        failure_reason=None,
        config=[],
        notes=None,
        primary_recall_k=None,
        dnf_if_recall_at=[],
        no_capture_registry=False,
    )


class CaptureEurocDescriptorRetrievalRecallTest(unittest.TestCase):
    def test_export_command_generates_descriptor_candidates(self) -> None:
        args = args_namespace()
        cmd = build_export_cmd(args, Path("target/out/candidates.csv"))

        self.assertIn("scripts/export_retrieval_candidates_from_descriptors.py", cmd)
        self.assertEqual(cmd[cmd.index("--descriptors") + 1], str(args.descriptors))
        self.assertEqual(cmd[cmd.index("--keyframe-decisions") + 1], str(args.keyframe_decisions))
        self.assertEqual(cmd[cmd.index("--query-ids") + 1], "100,105")
        self.assertEqual(cmd[cmd.index("--top-k") + 1], "20")
        self.assertEqual(cmd[cmd.index("--frontend") + 1], "mean_hog")

    def test_export_command_accepts_explicit_database_ids(self) -> None:
        args = args_namespace()
        args.keyframe_decisions = None
        args.database_ids = "1,2,3"
        args.no_normalize = True

        cmd = build_export_cmd(args, Path("target/out/candidates.csv"))

        self.assertIn("--database-ids", cmd)
        self.assertNotIn("--keyframe-decisions", cmd)
        self.assertIn("--no-normalize", cmd)

    def test_capture_command_delegates_to_euroc_recall_capture(self) -> None:
        args = args_namespace()
        export_cmd = build_export_cmd(args, Path("target/out/candidates.csv"))
        cmd = build_capture_cmd(
            args,
            candidates_csv=Path("target/out/candidates.csv"),
            run_id="run-1",
            out_dir=Path("target/out"),
            export_cmd=export_cmd,
        )

        joined = " ".join(cmd).replace("\\", "/")
        self.assertIn("scripts/capture_euroc_relocalization_retrieval_recall.py", cmd)
        self.assertIn(BENCHMARK_ID, cmd)
        self.assertIn("--candidates", cmd)
        self.assertIn("target/out/candidates.csv", joined)
        self.assertIn("--input-kind", cmd)
        self.assertIn("descriptor_retrieval_candidates", cmd)
        self.assertIn("descriptor_csv=target/run/frame_appearance_descriptors.csv", joined)
        self.assertIn("keyframe_decisions=target/run/keyframe_decisions.csv", joined)
        self.assertIn("frame_appearance_descriptors=target/run/frame_appearance_descriptors.csv", joined)
        self.assertIn("top_k=20", cmd)
        self.assertIn("--query-ids", cmd)
        self.assertIn("100,105", cmd)

    def test_capture_command_forwards_optional_registry_controls(self) -> None:
        args = args_namespace()
        args.primary_recall_k = 20
        args.dnf_if_recall_at = ["20=0.5"]
        args.no_capture_registry = True
        args.notes = "offline descriptor diagnostic"

        cmd = build_capture_cmd(
            args,
            candidates_csv=Path("target/out/candidates.csv"),
            run_id="run-1",
            out_dir=Path("target/out"),
            export_cmd=build_export_cmd(args, Path("target/out/candidates.csv")),
        )

        self.assertEqual(cmd[cmd.index("--primary-recall-k") + 1], "20")
        self.assertEqual(cmd[cmd.index("--dnf-if-recall-at") + 1], "20=0.5")
        self.assertIn("--no-capture-registry", cmd)
        self.assertIn("offline descriptor diagnostic", cmd)


if __name__ == "__main__":
    unittest.main()
