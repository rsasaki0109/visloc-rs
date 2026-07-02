from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_covisibility_mh05_boundary_support_gate import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    parse_gate,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    variant: str,
    boundary_min_optimized: int | None = None,
    boundary_min_fixed: int = 0,
    tracking: float = 0.265,
    rigid: float = 0.1614,
    mean_ms: float = 250.0,
    boundary_failures: int = 0,
    quality_failures: int = 4,
) -> dict:
    params = {
        "variant": variant,
        "max_frames": 400,
    }
    if variant == "enabled":
        params.update(
            {
                "covisibility_local_ba_max_neighbor_keyframes": 10,
                "covisibility_local_ba_max_boundary_keyframes": 10,
                "covisibility_local_ba_min_keyframes": 3,
                "covisibility_local_ba_trigger_every": 1,
                "covisibility_local_ba_max_landmarks": 200,
                "covisibility_local_ba_min_active_observations": 20,
                "covisibility_local_ba_fallback_min_boundary_observations": None,
                "covisibility_local_ba_remove_outliers": False,
                "covisibility_local_ba_max_outlier_observation_ratio": 0.3,
                "covisibility_local_ba_boundary_support_min_optimized_keyframes": boundary_min_optimized,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes": boundary_min_fixed,
            }
        )
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "euroc-covisibility-local-ba"},
        "dataset": {"sequence": "MH_05_difficult"},
        "config": {"params": params},
        "metrics": [
            {"name": "tracking_success_rate", "value": tracking},
            {"name": "ate_rigid_rmse_m", "value": rigid},
            {"name": "ate_similarity_rmse_m", "value": 0.1003},
            {"name": "covisibility_local_ba_successes", "value": 12},
            {"name": "covisibility_local_ba_failures", "value": 13},
            {"name": "covisibility_local_ba_quality_gate_failures", "value": quality_failures},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": boundary_failures},
            {"name": "covisibility_local_ba_no_local_landmarks_failures", "value": 9},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": mean_ms},
        ],
    }


class EurocCovisibilityMh05BoundarySupportGateSummaryTests(unittest.TestCase):
    def test_parse_gate(self) -> None:
        self.assertEqual(parse_gate("quality:none:0"), ("quality", "none", 0))
        self.assertEqual(parse_gate("boundary10:10:2"), ("boundary10", "10", 2))
        with self.assertRaises(Exception):
            parse_gate("bad")
        with self.assertRaises(Exception):
            parse_gate("bad:none:2")

    def test_load_latest_runs_filters_gate_and_renders_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "disabled.json").write_text(
                json.dumps(
                    manifest(
                        run_id="disabled",
                        created_utc="2026-06-19T00:00:00Z",
                        variant="disabled",
                        tracking=0.565,
                        rigid=0.1139,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "quality.json").write_text(
                json.dumps(
                    manifest(
                        run_id="quality",
                        created_utc="2026-06-19T01:00:00Z",
                        variant="enabled",
                        tracking=0.265,
                        mean_ms=304.5,
                        boundary_failures=0,
                        quality_failures=4,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "boundary7.json").write_text(
                json.dumps(
                    manifest(
                        run_id="boundary7",
                        created_utc="2026-06-19T02:00:00Z",
                        variant="enabled",
                        boundary_min_optimized=7,
                        boundary_min_fixed=2,
                        tracking=0.215,
                        mean_ms=74.8,
                        boundary_failures=4,
                        quality_failures=0,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "boundary10-old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="boundary10-old",
                        created_utc="2026-06-19T03:00:00Z",
                        variant="enabled",
                        boundary_min_optimized=10,
                        boundary_min_fixed=2,
                        tracking=0.265,
                        mean_ms=253.2,
                        boundary_failures=2,
                        quality_failures=2,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "boundary10-new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="boundary10-new",
                        created_utc="2026-06-19T04:00:00Z",
                        variant="enabled",
                        boundary_min_optimized=10,
                        boundary_min_fixed=2,
                        tracking=0.265,
                        mean_ms=254.9,
                        boundary_failures=2,
                        quality_failures=2,
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                sequence="MH_05_difficult",
                max_frames=400,
                neighbor_keyframes=10,
                boundary_keyframes=10,
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="0.3",
                gate=[
                    ("quality-gate only", "none", 0),
                    ("boundary7/2", "7", 2),
                    ("boundary10/2", "10", 2),
                ],
            )
            runs = load_latest_runs(args)

            self.assertEqual(runs["boundary10/2"]["run_id"], "boundary10-new")
            self.assertEqual(missing_expected_runs(args, runs), [])

            table = render(args, runs)
            self.assertIn("| boundary7/2 | 7/2 | 0.215 | -0.050 |", table)
            self.assertIn("| boundary10/2 | 10/2 | 0.265 | 0.000 |", table)
            self.assertIn("| boundary7/2 | 7/2 | 0.215 | -0.050 | 0.1614 | 0.1003 | 12 | 13 | 0 | 4 | 9 | 74.800 | -229.700 | reject | boundary7 |", table)
            self.assertIn("| boundary10/2 | 10/2 | 0.265 | 0.000 | 0.1614 | 0.1003 | 12 | 13 | 2 | 2 | 9 | 254.900 | -49.600 | candidate | boundary10-new |", table)
            self.assertIn("boundary10-new", table)


if __name__ == "__main__":
    unittest.main()
