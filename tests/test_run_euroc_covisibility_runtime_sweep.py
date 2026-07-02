from __future__ import annotations

import sys
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_euroc_covisibility_runtime_sweep import (  # noqa: E402
    covisibility_runner_cmd,
    summarizer_cmd,
)


def args() -> Namespace:
    return Namespace(
        euroc_root=Path("/datasets/euroc"),
        sequence=["MH_03_medium"],
        euroc_dir=[],
        landmark_cap=[100, 200],
        max_frames=80,
        profile="dev",
        out_root=Path("target/runtime"),
        registry_dir=Path("benchmarks/registry/runs/euroc"),
        summary_out=Path("docs/generated/runtime.md"),
        min_active_observations=20,
        fallback_min_boundary_observations="none",
        min_keyframes=3,
        trigger_every=1,
        max_neighbor_keyframes=10,
        min_shared=15,
        max_boundary_keyframes=10,
        min_boundary_observations=5,
        outlier_threshold_px="5.0",
        max_outlier_observation_ratio=None,
        boundary_support_min_optimized_keyframes=None,
        boundary_support_min_fixed_keyframes=0,
        base_demo_args="--gravity 0,0,-9.81 --stereo-bootstrap-strict",
        dry_run=True,
        no_capture_registry=False,
    )


class RunEurocCovisibilityRuntimeSweepTests(unittest.TestCase):
    def test_runner_cmd_varies_landmark_cap_and_uses_enabled_only(self) -> None:
        cmd = covisibility_runner_cmd(args(), 200, skip_build=True)

        self.assertIn("scripts/run_euroc_covisibility_local_ba_ab.py", cmd)
        self.assertIn("--only", cmd)
        self.assertIn("enabled", cmd)
        self.assertIn("--skip-build", cmd)
        self.assertIn("--max-landmarks", cmd)
        self.assertEqual(cmd[cmd.index("--max-landmarks") + 1], "200")
        self.assertEqual(cmd.count("--sequence"), 1)
        self.assertIn("MH_03_medium", cmd)
        out_root = cmd[cmd.index("--out-root") + 1]
        self.assertEqual(Path(out_root), Path("target/runtime/landmarks200"))

    def test_summarizer_cmd_carries_filters(self) -> None:
        cmd = summarizer_cmd(args())

        self.assertIn("scripts/summarize_euroc_covisibility_runtime_sweep.py", cmd)
        self.assertEqual(cmd.count("--landmark-cap"), 2)
        self.assertIn("100", cmd)
        self.assertIn("200", cmd)
        self.assertIn("--neighbor-keyframes", cmd)
        self.assertIn("10", cmd)
        self.assertIn("--boundary-keyframes", cmd)
        self.assertIn("--min-active-observations", cmd)
        self.assertIn("20", cmd)
        self.assertIn("--fallback", cmd)
        self.assertIn("none", cmd)

    def test_quality_gate_ratio_is_forwarded(self) -> None:
        a = args()
        a.max_outlier_observation_ratio = 0.3

        run_cmd = covisibility_runner_cmd(a, 200, skip_build=True)
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

        run_cmd = covisibility_runner_cmd(a, 200, skip_build=True)
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
