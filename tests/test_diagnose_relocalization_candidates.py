from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_relocalization_candidates import evaluate  # noqa: E402


class DiagnoseRelocalizationCandidatesTest(unittest.TestCase):
    def test_summarizes_retrieval_truth_against_acceptance_gates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "frame_groundtruth.csv"
            poses.write_text(
                "\n".join(
                    [
                        "frame_idx,gt_px,gt_py,gt_pz",
                        "0,0,0,0",
                        "1,10,0,0",
                        "2,0.5,0,0",
                        "3,0.4,0,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = root / "relocalization_appearance_candidates.csv"
            candidates.write_text(
                "\n".join(
                    [
                        "frontend,query_frame_id,matched_keyframe_id,score,rank,recovery_attempted,recovery_succeeded,passed_acceptance_gates,used_appearance_store,used_broader_fallback",
                        "appearance,2,0,0.9,1,1,0,0,1,0",
                        "appearance,3,1,0.8,1,1,1,1,1,0",
                        "appearance,3,0,0.7,2,1,1,1,1,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    distance_threshold_m=1.0,
                    min_temporal_gap=2,
                    min_path_length_m=None,
                )
            )

        summary = result["frontends"][0]
        self.assertEqual(summary["attempt_count"], 2)
        self.assertEqual(summary["success_count"], 1)
        self.assertEqual(summary["gate_pass_count"], 1)
        self.assertEqual(summary["top1_relevant_count"], 1)
        self.assertEqual(summary["any_relevant_count"], 2)
        self.assertEqual(summary["top1_relevant_rejected_count"], 1)
        attempts = {row["query_frame_id"]: row for row in result["attempts"]}
        self.assertTrue(attempts[2]["top1_relevant"])
        self.assertFalse(attempts[2]["recovery_succeeded"])
        self.assertFalse(attempts[3]["top1_relevant"])
        self.assertTrue(attempts[3]["any_relevant"])
        self.assertEqual(attempts[3]["first_relevant_rank"], 2)

    def test_cli_writes_json_markdown_and_attempt_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.csv"
            poses.write_text("frame_id,x,y,z\n0,0,0,0\n2,0.5,0,0\n", encoding="utf-8")
            candidates = root / "candidates.csv"
            candidates.write_text(
                "query_frame_id,matched_keyframe_id,score,rank,recovery_succeeded,passed_acceptance_gates\n"
                "2,0,0.9,1,1,1\n",
                encoding="utf-8",
            )
            out_json = root / "diag.json"
            out_md = root / "diag.md"
            out_csv = root / "diag.csv"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "diagnose_relocalization_candidates.py"),
                    "--candidates",
                    str(candidates),
                    "--poses",
                    str(poses),
                    "--distance-threshold-m",
                    "1",
                    "--min-temporal-gap",
                    "2",
                    "--out-json",
                    str(out_json),
                    "--out-md",
                    str(out_md),
                    "--out-csv",
                    str(out_csv),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            payload = json.loads(out_json.read_text(encoding="utf-8"))
            markdown = out_md.read_text(encoding="utf-8")
            attempt_csv = out_csv.read_text(encoding="utf-8")

        self.assertIn("# Relocalization Candidate Diagnostics", proc.stdout)
        self.assertIn("top1 true", markdown)
        self.assertIn("top1_distance_m", attempt_csv)
        self.assertEqual(payload["frontends"][0]["top1_relevant_count"], 1)

    def test_missing_acceptance_columns_are_unknown_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.csv"
            poses.write_text("frame_id,x,y,z\n0,0,0,0\n2,0.5,0,0\n", encoding="utf-8")
            candidates = root / "candidates.csv"
            candidates.write_text(
                "query_frame_id,matched_keyframe_id,score,rank\n2,0,0.9,1\n",
                encoding="utf-8",
            )

            result = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    distance_threshold_m=1.0,
                    min_temporal_gap=2,
                    min_path_length_m=None,
                )
            )

        summary = result["frontends"][0]
        self.assertEqual(summary["top1_relevant_count"], 1)
        self.assertEqual(summary["recovery_status_known_count"], 0)
        self.assertEqual(summary["gate_status_known_count"], 0)
        self.assertIsNone(summary["success_rate"])
        self.assertIsNone(summary["gate_pass_rate"])
        self.assertIsNone(summary["top1_relevant_rejected_rate"])


if __name__ == "__main__":
    unittest.main()
