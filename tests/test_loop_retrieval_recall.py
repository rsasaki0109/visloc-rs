from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from eval_loop_retrieval_recall import (  # noqa: E402
    evaluate,
    parse_gate,
    read_candidates,
    read_pose_centres,
)


def kitti_pose_line(x: float, y: float, z: float) -> str:
    return f"1 0 0 {x} 0 1 0 {y} 0 0 1 {z}\n"


class LoopRetrievalRecallTest(unittest.TestCase):
    def test_evaluates_recall_at_k_from_kitti_pose_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.txt"
            poses.write_text(
                "".join(
                    [
                        kitti_pose_line(0.0, 0.0, 0.0),
                        kitti_pose_line(1.0, 0.0, 0.0),
                        kitti_pose_line(2.0, 0.0, 0.0),
                        *[kitti_pose_line(40.0, 0.0, 0.0) for _ in range(47)],
                        kitti_pose_line(100.0, 0.0, 0.0),
                        *[kitti_pose_line(40.0, 0.0, 0.0) for _ in range(49)],
                        kitti_pose_line(0.5, 0.0, 0.0),
                        kitti_pose_line(100.0, 0.0, 0.0),
                    ]
                ),
                encoding="utf-8",
            )
            candidates = root / "candidates.csv"
            candidates.write_text(
                "\n".join(
                    [
                        "frontend,matched_keyframe_id,query_frame_id,score",
                        "vlad,50,100,2.0",
                        "vlad,0,100,1.0",
                        "vlad,50,101,3.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    input_kind="unit",
                    distance_threshold_m=2.0,
                    min_temporal_gap=50,
                    min_path_length_m=None,
                    ks=[1, 2],
                    query_ids="100,101",
                )
            )

        frontend = result["frontends"][0]
        self.assertEqual(frontend["frontend"], "vlad")
        self.assertEqual(frontend["eligible_query_count"], 2)
        self.assertAlmostEqual(frontend["recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(frontend["recall_at_k"]["2"], 1.0)
        self.assertAlmostEqual(frontend["mrr"], 0.75)
        self.assertAlmostEqual(frontend["top1_false_positive_rate"], 0.5)

    def test_temporal_gap_controls_query_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.csv"
            poses.write_text(
                "\n".join(
                    [
                        "frame_id,x,y,z",
                        "0,0,0,0",
                        "3,0.5,0,0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            candidates = root / "candidates.csv"
            candidates.write_text(
                "matched_keyframe_id,query_frame_id,score\n0,3,1.0\n",
                encoding="utf-8",
            )

            loose = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    input_kind="unit",
                    distance_threshold_m=2.0,
                    min_temporal_gap=3,
                    min_path_length_m=None,
                    ks=[1],
                    query_ids="3",
                )
            )
            strict = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    input_kind="unit",
                    distance_threshold_m=2.0,
                    min_temporal_gap=4,
                    min_path_length_m=None,
                    ks=[1],
                    query_ids="3",
                )
            )

        self.assertEqual(loose["frontends"][0]["eligible_query_count"], 1)
        self.assertEqual(strict["frontends"][0]["eligible_query_count"], 0)
        self.assertIsNone(strict["frontends"][0]["recall_at_k"]["1"])

    def test_missing_candidate_for_positive_query_counts_as_recall_miss(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.txt"
            poses.write_text(
                "".join(
                    [
                        kitti_pose_line(0.0, 0.0, 0.0),
                        kitti_pose_line(100.0, 0.0, 0.0),
                        kitti_pose_line(0.5, 0.0, 0.0),
                        kitti_pose_line(0.6, 0.0, 0.0),
                    ]
                ),
                encoding="utf-8",
            )
            candidates = root / "candidates.csv"
            candidates.write_text(
                "matched_keyframe_id,query_frame_id,score\n0,2,1.0\n",
                encoding="utf-8",
            )

            result = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    input_kind="unit",
                    distance_threshold_m=1.0,
                    min_temporal_gap=2,
                    min_path_length_m=None,
                    ks=[1],
                    query_ids=None,
                )
            )

        frontend = result["frontends"][0]
        self.assertEqual(frontend["query_count"], 4)
        self.assertEqual(frontend["queries_with_candidates"], 1)
        self.assertEqual(frontend["eligible_query_count"], 2)
        self.assertAlmostEqual(frontend["recall_at_k"]["1"], 0.5)
        self.assertAlmostEqual(frontend["mrr"], 0.5)

    def test_query_ids_can_be_scoped_to_candidate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.txt"
            poses.write_text(
                "".join(
                    [
                        kitti_pose_line(0.0, 0.0, 0.0),
                        kitti_pose_line(100.0, 0.0, 0.0),
                        kitti_pose_line(0.5, 0.0, 0.0),
                        kitti_pose_line(0.6, 0.0, 0.0),
                    ]
                ),
                encoding="utf-8",
            )
            candidates = root / "candidates.csv"
            candidates.write_text(
                "matched_keyframe_id,query_frame_id,score\n0,2,1.0\n",
                encoding="utf-8",
            )

            result = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    input_kind="unit",
                    distance_threshold_m=1.0,
                    min_temporal_gap=2,
                    min_path_length_m=None,
                    ks=[1],
                    query_ids=None,
                    query_ids_from_candidates=True,
                )
            )

        frontend = result["frontends"][0]
        self.assertEqual(result["query_scope"], "candidate_queries")
        self.assertEqual(result["query_scope_count"], 1)
        self.assertEqual(frontend["query_count"], 1)
        self.assertEqual(frontend["eligible_query_count"], 1)
        self.assertAlmostEqual(frontend["recall_at_k"]["1"], 1.0)

    def test_reads_candidate_column_aliases_and_pose_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.csv"
            poses.write_text("id,tx,ty,tz\n7,1,2,3\n", encoding="utf-8")
            candidates = root / "raw.csv"
            candidates.write_text("retriever,older,newer,similarity\nx,7,10,0.9\n", encoding="utf-8")

            pose_map = read_pose_centres(poses)
            rows = read_candidates(candidates)

        self.assertEqual(pose_map[7].x, 1.0)
        self.assertEqual(rows[0].frontend, "x")
        self.assertEqual(rows[0].matched_keyframe_id, 7)
        self.assertEqual(rows[0].query_frame_id, 10)
        self.assertEqual(rows[0].score, 0.9)

    def test_reads_euroc_slam_error_pose_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "slam_errors.csv"
            poses.write_text(
                "\n".join(
                    [
                        "timestamp_ns,frame_idx,gt_px,gt_py,gt_pz,est_px,est_py,est_pz,position_error_m,orientation_error_deg",
                        "1403636580763555584,42,1.5,2.5,3.5,10,20,30,0.1,1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            pose_map = read_pose_centres(poses)

        self.assertEqual(pose_map[42].x, 1.5)
        self.assertEqual(pose_map[42].y, 2.5)
        self.assertEqual(pose_map[42].z, 3.5)

    def test_parse_gate_rejects_invalid_values(self) -> None:
        self.assertEqual(parse_gate("5=0.8"), (5, 0.8))
        with self.assertRaises(Exception):
            parse_gate("5")

    def test_json_result_is_serializable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            poses = root / "poses.txt"
            poses.write_text(kitti_pose_line(0, 0, 0) + kitti_pose_line(0, 0, 0), encoding="utf-8")
            candidates = root / "candidates.csv"
            candidates.write_text(
                "matched_keyframe_id,query_frame_id,score\n0,1,1.0\n",
                encoding="utf-8",
            )
            result = evaluate(
                Namespace(
                    poses=poses,
                    candidates=[candidates],
                    input_kind="unit",
                    distance_threshold_m=1.0,
                    min_temporal_gap=1,
                    min_path_length_m=None,
                    ks=[1],
                    query_ids=None,
                )
            )

        json.dumps(result)


if __name__ == "__main__":
    unittest.main()
