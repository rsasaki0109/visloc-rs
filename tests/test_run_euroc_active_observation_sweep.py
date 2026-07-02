from __future__ import annotations

import shlex
import sys
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_euroc_active_observation_sweep import (  # noqa: E402
    demo_args_for_floor,
    keyframe_runner_cmd,
    summarizer_cmd,
)


def args() -> Namespace:
    return Namespace(
        euroc_root=Path("/datasets/euroc"),
        sequence=["MH_01_easy", "MH_03_medium"],
        active_floor=[20, 50],
        max_frames=400,
        profile="release",
        out_root=Path("target/sweep"),
        registry_dir=Path("benchmarks/registry/runs/euroc"),
        summary_out=Path("docs/generated/sweep.md"),
        tracked_landmark_ratio=0.9,
        min_tracked_landmarks=20,
        ba_min_keyframes=3,
        ba_trigger_every=1,
        ba_max_landmarks="200",
        ba_outlier_threshold_px="5.0",
        fallback_min_boundary_observations="none",
        base_demo_args="--gravity 0,0,-9.81 --keyframe-min-translation 0.1",
        dry_run=True,
        no_capture_registry=False,
    )


class RunEurocActiveObservationSweepTests(unittest.TestCase):
    def test_demo_args_for_floor_adds_covisibility_knobs(self) -> None:
        parts = shlex.split(demo_args_for_floor(args(), 20))

        self.assertIn("--gravity", parts)
        self.assertIn("--covisibility-local-ba", parts)
        self.assertIn("--covisibility-local-ba-min-active-observations", parts)
        idx = parts.index("--covisibility-local-ba-min-active-observations")
        self.assertEqual(parts[idx + 1], "20")
        fallback_idx = parts.index("--covisibility-local-ba-fallback-min-boundary-observations")
        self.assertEqual(parts[fallback_idx + 1], "none")

    def test_keyframe_runner_cmd_includes_sequences_and_skip_build(self) -> None:
        cmd = keyframe_runner_cmd(args(), 50, skip_build=True)

        self.assertIn("scripts/run_euroc_keyframe_policy_ab.py", cmd)
        self.assertIn("--skip-build", cmd)
        self.assertEqual(cmd.count("--sequence"), 2)
        self.assertIn("MH_01_easy", cmd)
        self.assertIn("MH_03_medium", cmd)
        out_root = cmd[cmd.index("--out-root") + 1]
        self.assertEqual(Path(out_root), Path("target/sweep/active50"))

    def test_summarizer_cmd_carries_filters(self) -> None:
        cmd = summarizer_cmd(args())

        self.assertIn("scripts/summarize_euroc_active_observation_sweep.py", cmd)
        self.assertEqual(cmd.count("--active-floor"), 2)
        self.assertIn("20", cmd)
        self.assertIn("50", cmd)
        self.assertIn("--fallback", cmd)
        self.assertIn("none", cmd)


if __name__ == "__main__":
    unittest.main()
