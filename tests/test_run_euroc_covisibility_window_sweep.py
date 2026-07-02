from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_euroc_covisibility_window_sweep import (  # noqa: E402
    covisibility_runner_cmd,
    summarizer_cmd,
)


def args() -> Namespace:
    return Namespace(
        euroc_root=Path("/datasets/euroc"),
        sequence=["MH_03_medium"],
        euroc_dir=[],
        window_cap=[(5, 5), (10, 10)],
        max_frames=80,
        profile="dev",
        out_root=Path("target/window"),
        registry_dir=Path("benchmarks/registry/runs/euroc"),
        summary_out=Path("docs/generated/window.md"),
        landmark_cap=200,
        min_active_observations=20,
        fallback_min_boundary_observations="none",
        min_keyframes=3,
        trigger_every=1,
        min_shared=15,
        min_boundary_observations=5,
        outlier_threshold_px="5.0",
        max_outlier_observation_ratio=None,
        boundary_support_min_optimized_keyframes=None,
        boundary_support_min_fixed_keyframes=0,
        base_demo_args="--gravity 0,0,-9.81 --stereo-bootstrap-strict",
        dry_run=True,
        no_capture_registry=False,
    )


class RunEurocCovisibilityWindowSweepTests(unittest.TestCase):
    def test_runner_cmd_varies_neighbor_and_boundary_caps(self) -> None:
        cmd = covisibility_runner_cmd(args(), (5, 10), skip_build=True)

        self.assertIn("scripts/run_euroc_covisibility_local_ba_ab.py", cmd)
        self.assertIn("--only", cmd)
        self.assertIn("enabled", cmd)
        self.assertIn("--skip-build", cmd)
        self.assertEqual(cmd[cmd.index("--max-neighbor-keyframes") + 1], "5")
        self.assertEqual(cmd[cmd.index("--max-boundary-keyframes") + 1], "10")
        self.assertEqual(cmd[cmd.index("--max-landmarks") + 1], "200")
        out_root = cmd[cmd.index("--out-root") + 1]
        self.assertEqual(Path(out_root), Path("target/window/n5_b10"))

    def test_summarizer_cmd_carries_window_filters(self) -> None:
        cmd = summarizer_cmd(args())

        self.assertIn("scripts/summarize_euroc_covisibility_window_sweep.py", cmd)
        self.assertEqual(cmd.count("--window-cap"), 2)
        self.assertIn("5:5", cmd)
        self.assertIn("10:10", cmd)
        self.assertIn("--landmark-cap", cmd)
        self.assertIn("200", cmd)
        self.assertIn("--min-keyframes", cmd)
        self.assertIn("3", cmd)
        self.assertIn("--trigger-every", cmd)
        self.assertIn("--fallback", cmd)
        self.assertIn("none", cmd)

    def test_quality_gate_ratio_is_forwarded(self) -> None:
        a = args()
        a.max_outlier_observation_ratio = 0.3

        run_cmd = covisibility_runner_cmd(a, (5, 10), skip_build=True)
        summary_cmd = summarizer_cmd(a)

        self.assertEqual(run_cmd[run_cmd.index("--max-outlier-observation-ratio") + 1], "0.3")
        self.assertEqual(
            summary_cmd[summary_cmd.index("--max-outlier-observation-ratio") + 1],
            "0.3",
        )

    def test_boundary_support_gate_is_forwarded(self) -> None:
        a = args()
        a.boundary_support_min_optimized_keyframes = 7
        a.boundary_support_min_fixed_keyframes = 2

        run_cmd = covisibility_runner_cmd(a, (5, 10), skip_build=True)
        summary_cmd = summarizer_cmd(a)

        self.assertEqual(run_cmd[run_cmd.index("--boundary-support-min-optimized-keyframes") + 1], "7")
        self.assertEqual(run_cmd[run_cmd.index("--boundary-support-min-fixed-keyframes") + 1], "2")
        self.assertEqual(
            summary_cmd[summary_cmd.index("--boundary-support-min-optimized-keyframes") + 1],
            "7",
        )
        self.assertEqual(
            summary_cmd[summary_cmd.index("--boundary-support-min-fixed-keyframes") + 1],
            "2",
        )


if __name__ == "__main__":
    unittest.main()
