from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from capture_kitti_multiseq_run import (  # noqa: E402
    build_capture_cmd,
    load_evaluation,
    metric_args,
    optional_artifacts,
    parse_verified_loops,
)


def args_namespace(root: Path) -> Namespace:
    return Namespace(
        sequence="02",
        evaluation_json=root / "evaluation.json",
        vo_log=root / "vo.log",
        vo_poses=root / "vo_poses.txt",
        poses=root / "poses_02.txt",
        dataset_path=root / "dataset",
        dataset_version="KITTI odometry grayscale",
        features_dir=root / "external_deep",
        out_dir=root,
        registry_dir=Path("benchmarks/registry/runs/kitti"),
        run_id=None,
        command="scripts/run_kitti_multiseq_benchmark.sh --sequence 02 --skip-export",
        claim_scope="exploratory",
        status="success",
        failure_reason=None,
        config=["loop_pnp_confidence_weights=1"],
        notes=None,
        dry_run=False,
    )


class CaptureKittiMultiseqRunTest(unittest.TestCase):
    def test_load_evaluation_requires_core_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evaluation.json"
            path.write_text(
                json.dumps(
                    {
                        "sequence": "02",
                        "frames": 4661,
                        "ate_rmse_se3_m": 5.8106,
                        "ate_rmse_sim3_m": 5.6977,
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(load_evaluation(path)["frames"], 4661)

            path.write_text(json.dumps({"sequence": "02"}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_evaluation(path)

    def test_parse_verified_loops_uses_vo_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "vo.log"
            path.write_text("finished full run verified_loops=2 other=1\n", encoding="utf-8")
            self.assertEqual(parse_verified_loops(path), 2)

            path.write_text("no loop summary\n", encoding="utf-8")
            self.assertIsNone(parse_verified_loops(path))

    def test_metric_args_record_ate_frames_and_loops(self) -> None:
        metrics = metric_args(
            {
                "sequence": "02",
                "frames": 4661,
                "ate_rmse_se3_m": 5.8106,
                "ate_rmse_sim3_m": 5.6977,
            },
            verified_loops=2,
        )

        self.assertIn("--primary-metric", metrics)
        self.assertIn("ate_rmse_se3_m", metrics)
        self.assertIn("ate_rmse_se3_m=5.8106:m", metrics)
        self.assertIn("ate_rmse_sim3_m=5.6977:m", metrics)
        self.assertIn("frames=4661:count", metrics)
        self.assertIn("verified_loops=2:count", metrics)

    def test_capture_command_records_artifacts_config_and_scope(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in [
                "evaluation.json",
                "vo.log",
                "vo_poses.txt",
                "poses_02.txt",
                "summary.txt",
                "loop_candidates.csv",
                "loop_candidate_verifications.csv",
            ]:
                (root / name).write_text("x\n", encoding="utf-8")
            (root / "external_deep").mkdir()
            args = args_namespace(root)
            evaluation = {
                "sequence": "02",
                "frames": 4661,
                "ate_rmse_se3_m": 5.8106,
                "ate_rmse_sim3_m": 5.6977,
            }

            artifacts = optional_artifacts(args.out_dir, args.features_dir)
            self.assertIn(("loop_candidates", root / "loop_candidates.csv"), artifacts)
            self.assertIn(("features_dir", root / "external_deep"), artifacts)

            cmd = build_capture_cmd(args, run_id="run-1", evaluation=evaluation, verified_loops=2)

        joined = " ".join(cmd).replace("\\", "/")
        self.assertIn("scripts/benchmark_registry.py", cmd)
        self.assertIn("--claim-scope", cmd)
        self.assertEqual(cmd[cmd.index("--claim-scope") + 1], "exploratory")
        self.assertIn("evaluation_json=", joined)
        self.assertIn("loop_candidates.csv", joined)
        self.assertIn("features_dir=", joined)
        self.assertIn("loop_pnp_confidence_weights=1", cmd)
        self.assertIn("ate_rmse_se3_m=5.8106:m", cmd)


if __name__ == "__main__":
    unittest.main()
