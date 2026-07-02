from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_covisibility_mh05_mitigation import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    parse_config,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    variant: str,
    min_keyframes: int = 3,
    trigger_every: int = 1,
    sequence: str = "MH_05_difficult",
    remove_outliers: bool = False,
    max_outlier: float | None = None,
    boundary_support_min_optimized: int | None = None,
    boundary_support_min_fixed: int = 0,
    boundary_support_failures: int = 0,
    tracking: float = 0.5,
    rigid: float = 1.0,
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
                "covisibility_local_ba_min_keyframes": min_keyframes,
                "covisibility_local_ba_trigger_every": trigger_every,
                "covisibility_local_ba_max_landmarks": 200,
                "covisibility_local_ba_min_active_observations": 20,
                "covisibility_local_ba_fallback_min_boundary_observations": None,
                "covisibility_local_ba_remove_outliers": remove_outliers,
                "covisibility_local_ba_max_outlier_observation_ratio": max_outlier,
                "covisibility_local_ba_boundary_support_min_optimized_keyframes": boundary_support_min_optimized,
                "covisibility_local_ba_boundary_support_min_fixed_keyframes": boundary_support_min_fixed,
            }
        )
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "euroc-covisibility-local-ba"},
        "dataset": {"sequence": sequence},
        "config": {"params": params},
        "metrics": [
            {"name": "tracking_success_rate", "value": tracking},
            {"name": "ate_rigid_rmse_m", "value": rigid},
            {"name": "ate_similarity_rmse_m", "value": rigid + 0.1},
            {"name": "map_keyframes", "value": 42},
            {"name": "covisibility_local_ba_triggers", "value": 7},
            {"name": "covisibility_local_ba_successes", "value": 6},
            {"name": "covisibility_local_ba_failures", "value": 1},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": boundary_support_failures},
            {"name": "covisibility_local_ba_no_local_landmarks_failures", "value": 1},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": 12.5},
        ],
    }


class EurocCovisibilityMh05MitigationSummaryTests(unittest.TestCase):
    def test_parse_config(self) -> None:
        self.assertEqual(parse_config("late:10:5"), ("late", 10, 5))
        with self.assertRaises(Exception):
            parse_config("bad")

    def test_load_latest_runs_filters_cadence_and_renders_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "disabled.json").write_text(
                json.dumps(
                    manifest(
                        run_id="disabled",
                        created_utc="2026-06-19T00:00:00Z",
                        variant="disabled",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "early.json").write_text(
                json.dumps(
                    manifest(
                        run_id="early",
                        created_utc="2026-06-19T01:00:00Z",
                        variant="enabled",
                        min_keyframes=3,
                        trigger_every=1,
                        tracking=0.2,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "late.json").write_text(
                json.dumps(
                    manifest(
                        run_id="late",
                        created_utc="2026-06-19T02:00:00Z",
                        variant="enabled",
                        min_keyframes=10,
                        trigger_every=5,
                        tracking=0.6,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "early-gated.json").write_text(
                json.dumps(
                    manifest(
                        run_id="early-gated",
                        created_utc="2026-06-19T03:00:00Z",
                        variant="enabled",
                        min_keyframes=3,
                        trigger_every=1,
                        max_outlier=0.3,
                        tracking=0.9,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "early-remove-outliers.json").write_text(
                json.dumps(
                    manifest(
                        run_id="early-remove-outliers",
                        created_utc="2026-06-19T04:00:00Z",
                        variant="enabled",
                        min_keyframes=3,
                        trigger_every=1,
                        remove_outliers=True,
                        tracking=0.95,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "early-boundary-support.json").write_text(
                json.dumps(
                    manifest(
                        run_id="early-boundary-support",
                        created_utc="2026-06-19T05:00:00Z",
                        variant="enabled",
                        min_keyframes=3,
                        trigger_every=1,
                        boundary_support_min_optimized=7,
                        boundary_support_min_fixed=2,
                        boundary_support_failures=4,
                        tracking=0.99,
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
                min_active_observations=20,
                fallback="none",
                remove_outliers=False,
                max_outlier_observation_ratio="none",
                boundary_support_min_optimized_keyframes="none",
                boundary_support_min_fixed_keyframes=0,
                config=[
                    ("enabled min3/every1", 3, 1),
                    ("enabled min6/every3", 6, 3),
                    ("enabled min10/every5", 10, 5),
                ],
            )
            runs = load_latest_runs(args)

            self.assertEqual(runs["disabled"]["run_id"], "disabled")
            self.assertEqual(runs["enabled min3/every1"]["run_id"], "early")
            self.assertEqual(runs["enabled min10/every5"]["run_id"], "late")
            self.assertEqual(missing_expected_runs(args, runs), ["enabled min6/every3"])

            table = render(args, runs)
            self.assertIn("| enabled min6/every3 |  |  |  |  |  |  |  |  |  |  |  | missing |", table)
            self.assertIn("| enabled min10/every5 | 0.600 |", table)


if __name__ == "__main__":
    unittest.main()
