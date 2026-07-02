from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_retrieval_candidates_from_descriptors import (  # noqa: E402
    generate_candidates,
    read_descriptors,
    read_keyframe_ids,
)


class ExportRetrievalCandidatesFromDescriptorsTest(unittest.TestCase):
    def test_reads_descriptor_columns_and_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "descriptors.csv"
            path.write_text(
                "frame_idx,timestamp_ns,d0,d1\n10,100,3,4\n20,200,0,2\n",
                encoding="utf-8",
            )

            descriptors = read_descriptors(path)

        self.assertAlmostEqual(descriptors[10][0], 0.6)
        self.assertAlmostEqual(descriptors[10][1], 0.8)
        self.assertAlmostEqual(descriptors[20][0], 0.0)
        self.assertAlmostEqual(descriptors[20][1], 1.0)

    def test_reads_descriptor_cell_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "descriptors.csv"
            path.write_text(
                'frame_idx,descriptor\n10,"1 0 0"\n20,"0, 1, 0"\n',
                encoding="utf-8",
            )

            descriptors = read_descriptors(path)

        self.assertEqual(descriptors[10], [1.0, 0.0, 0.0])
        self.assertEqual(descriptors[20], [0.0, 1.0, 0.0])

    def test_reads_selected_keyframes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keyframe_decisions.csv"
            path.write_text(
                "frame_idx,selected,reason\n10,1,First\n11,0,Gap\n20,true,Threshold\n",
                encoding="utf-8",
            )

            self.assertEqual(read_keyframe_ids(path), {10, 20})

    def test_generates_ranked_older_candidates_with_gap(self) -> None:
        descriptors = {
            0: [1.0, 0.0],
            5: [0.0, 1.0],
            10: [0.9, 0.1],
            12: [1.0, 0.0],
        }

        candidates = generate_candidates(
            descriptors,
            database_ids={0, 5, 12},
            query_ids={10, 12},
            top_k=2,
            exclude_recent_frame_gap=6,
            min_similarity=None,
        )

        self.assertEqual(
            [(c.query_frame_id, c.matched_keyframe_id, c.rank) for c in candidates],
            [(10, 0, 1), (12, 0, 1), (12, 5, 2)],
        )
        self.assertAlmostEqual(candidates[0].score, 0.9)

    def test_min_similarity_filters_candidates(self) -> None:
        descriptors = {
            0: [1.0, 0.0],
            5: [0.0, 1.0],
            10: [0.6, 0.4],
        }

        candidates = generate_candidates(
            descriptors,
            database_ids={0, 5},
            query_ids={10},
            top_k=5,
            exclude_recent_frame_gap=0,
            min_similarity=0.5,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].matched_keyframe_id, 0)

    def test_cli_writes_eval_compatible_candidate_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            descriptors = root / "descriptors.csv"
            descriptors.write_text(
                "frame_idx,d0,d1\n0,1,0\n5,0,1\n10,1,0\n",
                encoding="utf-8",
            )
            keyframes = root / "keyframe_decisions.csv"
            keyframes.write_text(
                "frame_idx,selected\n0,1\n5,1\n10,0\n",
                encoding="utf-8",
            )
            out = root / "candidates.csv"

            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts" / "export_retrieval_candidates_from_descriptors.py"),
                    "--descriptors",
                    str(descriptors),
                    "--keyframe-decisions",
                    str(keyframes),
                    "--query-ids",
                    "10",
                    "--top-k",
                    "2",
                    "--exclude-recent-frame-gap",
                    "0",
                    "--frontend",
                    "mean_hog",
                    "--out",
                    str(out),
                ],
                cwd=REPO_ROOT,
                text=True,
                capture_output=True,
                check=True,
            )

            with out.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertIn("wrote 2 candidates", proc.stdout)
        self.assertEqual(rows[0]["frontend"], "mean_hog")
        self.assertEqual(rows[0]["query_frame_id"], "10")
        self.assertEqual(rows[0]["matched_keyframe_id"], "0")
        self.assertEqual(rows[0]["rank"], "1")


if __name__ == "__main__":
    unittest.main()
