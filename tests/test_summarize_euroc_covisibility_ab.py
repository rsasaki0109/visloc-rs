from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_covisibility_ab import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    variant: str,
    sequence: str = "MH_03_medium",
    max_frames: int = 400,
    neighbor: int = 10,
    boundary: int = 10,
    min_keyframes: int = 3,
    trigger_every: int = 1,
    landmark_cap: int = 200,
    min_active: int = 20,
    fallback: str | None = None,
    remove_outliers: bool = False,
    max_outlier: float | None = None,
    boundary_support_min_optimized: int | None = None,
    boundary_support_min_fixed: int = 0,
    boundary_support_failures: int = 0,
    tracking: float = 0.9,
    rigid: float = 1.0,
) -> dict:
    params = {
        "variant": variant,
        "max_frames": max_frames,
    }
    if variant == "enabled":
        params.update(
            {
                "covisibility_local_ba_max_neighbor_keyframes": neighbor,
                "covisibility_local_ba_max_boundary_keyframes": boundary,
                "covisibility_local_ba_min_keyframes": min_keyframes,
                "covisibility_local_ba_trigger_every": trigger_every,
                "covisibility_local_ba_max_landmarks": landmark_cap,
                "covisibility_local_ba_min_active_observations": min_active,
                "covisibility_local_ba_fallback_min_boundary_observations": fallback,
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
            {"name": "covisibility_local_ba_successes", "value": 8},
            {"name": "covisibility_local_ba_failures", "value": 1},
            {"name": "covisibility_local_ba_boundary_support_failures", "value": boundary_support_failures},
            {"name": "covisibility_local_ba_elapsed_ms_mean", "value": 12.5},
        ],
    }


class EurocCovisibilityAbSummaryTests(unittest.TestCase):
    def test_load_latest_runs_filters_enabled_config_and_renders_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "disabled-old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="disabled-old",
                        created_utc="2026-06-19T00:00:00Z",
                        variant="disabled",
                        tracking=0.5,
                        rigid=2.0,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "disabled-new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="disabled-new",
                        created_utc="2026-06-19T01:00:00Z",
                        variant="disabled",
                        tracking=0.6,
                        rigid=1.5,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "enabled.json").write_text(
                json.dumps(
                    manifest(
                        run_id="enabled",
                        created_utc="2026-06-19T02:00:00Z",
                        variant="enabled",
                        tracking=0.9,
                        rigid=1.0,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "enabled-window5.json").write_text(
                json.dumps(
                    manifest(
                        run_id="enabled-window5",
                        created_utc="2026-06-19T03:00:00Z",
                        variant="enabled",
                        neighbor=5,
                        boundary=5,
                        tracking=0.1,
                        rigid=0.1,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "enabled-gated.json").write_text(
                json.dumps(
                    manifest(
                        run_id="enabled-gated",
                        created_utc="2026-06-19T04:00:00Z",
                        variant="enabled",
                        max_outlier=0.3,
                        tracking=1.0,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "enabled-remove-outliers.json").write_text(
                json.dumps(
                    manifest(
                        run_id="enabled-remove-outliers",
                        created_utc="2026-06-19T05:00:00Z",
                        variant="enabled",
                        remove_outliers=True,
                        tracking=1.0,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "enabled-boundary-support.json").write_text(
                json.dumps(
                    manifest(
                        run_id="enabled-boundary-support",
                        created_utc="2026-06-19T06:00:00Z",
                        variant="enabled",
                        boundary_support_min_optimized=7,
                        boundary_support_min_fixed=2,
                        boundary_support_failures=4,
                        tracking=1.0,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                max_frames=400,
                sequence=["MH_03_medium", "MH_05_difficult"],
                enabled_neighbor_keyframes=10,
                enabled_boundary_keyframes=10,
                enabled_min_keyframes=3,
                enabled_trigger_every=1,
                enabled_landmark_cap=200,
                enabled_min_active_observations=20,
                enabled_fallback="none",
                enabled_remove_outliers=False,
                enabled_max_outlier_observation_ratio="none",
                enabled_boundary_support_min_optimized_keyframes="none",
                enabled_boundary_support_min_fixed_keyframes=0,
            )
            runs = load_latest_runs(args)

            self.assertEqual(runs[("MH_03_medium", "disabled")]["run_id"], "disabled-new")
            self.assertEqual(runs[("MH_03_medium", "enabled")]["run_id"], "enabled")
            self.assertEqual(
                missing_expected_runs(args, runs),
                [("MH_05_difficult", "disabled"), ("MH_05_difficult", "enabled")],
            )

            table = render(args, runs)
            self.assertIn("| MH_03_medium | 0.600 | 0.900 | 0.300 | 1.5000 | 1.0000 | 0.5000 |", table)
            self.assertIn("win", table)
            self.assertIn("| MH_05_difficult |  |  |  |  |  |  |  |  |  |  |  |  |  | missing | missing |", table)


if __name__ == "__main__":
    unittest.main()
