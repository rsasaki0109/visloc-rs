from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_covisibility_window_sweep import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    parse_window_cap,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    neighbor: int = 10,
    boundary: int = 10,
    sequence: str = "MH_03_medium",
    max_frames: int = 80,
    landmark_cap: int = 200,
    min_keyframes: int = 3,
    trigger_every: int = 1,
    min_active: int = 20,
    fallback: str | None = None,
    remove_outliers: bool = False,
    max_outlier: float | None = None,
    boundary_support_min_optimized: int | None = None,
    boundary_support_min_fixed: int = 0,
    boundary_support_failures: int = 0,
    rigid: float = 1.0,
) -> dict:
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "euroc-covisibility-local-ba"},
        "dataset": {"sequence": sequence},
        "config": {
            "params": {
                "variant": "enabled",
                "max_frames": max_frames,
                "covisibility_local_ba_max_landmarks": landmark_cap,
                "covisibility_local_ba_min_keyframes": min_keyframes,
                "covisibility_local_ba_trigger_every": trigger_every,
                "covisibility_local_ba_max_neighbor_keyframes": neighbor,
                "covisibility_local_ba_max_boundary_keyframes": boundary,
                "covisibility_local_ba_min_active_observations": min_active,
                "covisibility_local_ba_fallback_min_boundary_observations": fallback,
                "covisibility_local_ba_remove_outliers": remove_outliers,
                "covisibility_local_ba_max_outlier_observation_ratio": max_outlier,
                "covisibility_local_ba_boundary_support_min_optimized_keyframes": boundary_support_min_optimized,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes": boundary_support_min_fixed,
            }
        },
        "metrics": [
            {"name": "tracking_success_rate", "value": 0.9},
            {"name": "ate_rigid_rmse_m", "value": rigid},
            {"name": "ate_similarity_rmse_m", "value": 0.5},
            {"name": "covisibility_local_ba_successes", "value": 8},
            {"name": "covisibility_local_ba_failures", "value": 1},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": boundary_support_failures},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": 12.5},
            {"name": "covisibility_local_ba_elapsed_ms_max", "value": 30.0},
        ],
    }


class EurocCovisibilityWindowSweepSummaryTests(unittest.TestCase):
    def test_parse_window_cap_accepts_colon_or_comma(self) -> None:
        self.assertEqual(parse_window_cap("5:10"), (5, 10))
        self.assertEqual(parse_window_cap("5,10"), (5, 10))

    def test_load_latest_runs_filters_window_and_takes_newest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="old",
                        created_utc="2026-06-19T00:00:00Z",
                        rigid=2.0,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="new",
                        created_utc="2026-06-19T01:00:00Z",
                        rigid=1.5,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "window15.json").write_text(
                json.dumps(
                    manifest(
                        run_id="window15",
                        created_utc="2026-06-19T02:00:00Z",
                        neighbor=15,
                        boundary=15,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "gated-newer.json").write_text(
                json.dumps(
                    manifest(
                        run_id="gated-newer",
                        created_utc="2026-06-19T03:00:00Z",
                        max_outlier=0.3,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "remove-outliers-newer.json").write_text(
                json.dumps(
                    manifest(
                        run_id="remove-outliers-newer",
                        created_utc="2026-06-19T04:00:00Z",
                        remove_outliers=True,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "boundary-support-newer.json").write_text(
                json.dumps(
                    manifest(
                        run_id="boundary-support-newer",
                        created_utc="2026-06-19T05:00:00Z",
                        boundary_support_min_optimized=7,
                        boundary_support_min_fixed=2,
                        boundary_support_failures=4,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                max_frames=80,
                sequence=["MH_03_medium"],
                window_cap=[(5, 5), (10, 10)],
                landmark_cap=200,
                min_keyframes=3,
                trigger_every=1,
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
            )
            runs = load_latest_runs(args)

            self.assertEqual(set(runs.keys()), {("MH_03_medium", 10, 10)})
            self.assertEqual(runs[("MH_03_medium", 10, 10)]["run_id"], "new")
            self.assertEqual(missing_expected_runs(args, runs), [("MH_03_medium", 5, 5)])

            table = render(args, runs)
            self.assertIn("| MH_03_medium | 5 | 5 |  |  |  |  |  |  |  |  |  | missing |", table)
            self.assertIn("| MH_03_medium | 10 | 10 | 0.900 | 1.5000 |", table)
            self.assertIn("12.500", table)
            self.assertIn("new", table)


if __name__ == "__main__":
    unittest.main()
