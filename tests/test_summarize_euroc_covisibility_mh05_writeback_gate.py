from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_covisibility_mh05_writeback_gate import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    variant: str,
    sequence: str = "MH_05_difficult",
    max_behind_camera_ratio: float | None = None,
    min_fixed_to_optimized_ratio: float | None = None,
    tracking: float = 0.5,
    rigid: float = 1.0,
    behind_camera_gate_failures: int = 0,
    fixed_ratio_gate_failures: int = 0,
) -> dict:
    params: dict = {
        "variant": variant,
        "max_frames": 400,
    }
    if variant in {"writeback_gate_enabled_nogate", "writeback_gate_enabled_gate"}:
        params.update(
            {
                "covisibility_local_ba_max_neighbor_keyframes": 10,
                "covisibility_local_ba_max_boundary_keyframes": 10,
                "covisibility_local_ba_min_keyframes": 3,
                "covisibility_local_ba_trigger_every": 1,
                "covisibility_local_ba_max_landmarks": 200,
                "covisibility_local_ba_min_active_observations": 20,
                "covisibility_local_ba_max_behind_camera_ratio": max_behind_camera_ratio,
                "covisibility_local_ba_min_fixed_to_optimized_ratio": min_fixed_to_optimized_ratio,
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
            {
                "name": "covisibility_local_ba_behind_camera_gate_failures",
                "value": behind_camera_gate_failures,
            },
            {
                "name": "covisibility_local_ba_fixed_ratio_gate_failures",
                "value": fixed_ratio_gate_failures,
            },
        ],
    }


class EurocCovisibilityMh05WritebackGateSummaryTests(unittest.TestCase):
    def test_load_latest_runs_filters_variant_and_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "disabled.json").write_text(
                json.dumps(
                    manifest(
                        run_id="disabled",
                        created_utc="2026-07-03T00:00:00Z",
                        variant="writeback_gate_disabled",
                        tracking=0.565,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "nogate.json").write_text(
                json.dumps(
                    manifest(
                        run_id="nogate",
                        created_utc="2026-07-03T00:00:00Z",
                        variant="writeback_gate_enabled_nogate",
                        tracking=0.220,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "gate-wrong.json").write_text(
                json.dumps(
                    manifest(
                        run_id="gate-wrong",
                        created_utc="2026-07-03T00:00:00Z",
                        variant="writeback_gate_enabled_gate",
                        max_behind_camera_ratio=0.5,
                        min_fixed_to_optimized_ratio=0.5,
                        tracking=0.99,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "gate-early.json").write_text(
                json.dumps(
                    manifest(
                        run_id="gate-early",
                        created_utc="2026-07-03T00:00:00Z",
                        variant="writeback_gate_enabled_gate",
                        max_behind_camera_ratio=0.3,
                        min_fixed_to_optimized_ratio=0.34,
                        tracking=0.258,
                        fixed_ratio_gate_failures=5,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "gate-late.json").write_text(
                json.dumps(
                    manifest(
                        run_id="gate-late",
                        created_utc="2026-07-03T00:00:01Z",
                        variant="writeback_gate_enabled_gate",
                        max_behind_camera_ratio=0.3,
                        min_fixed_to_optimized_ratio=0.34,
                        tracking=0.3,
                        fixed_ratio_gate_failures=6,
                    )
                ),
                encoding="utf-8",
            )
            # Different sequence entirely; must not leak into MH_05 rows.
            (registry / "mh01-disabled.json").write_text(
                json.dumps(
                    manifest(
                        run_id="mh01-disabled",
                        created_utc="2026-07-03T00:00:00Z",
                        variant="writeback_gate_disabled",
                        sequence="MH_01_easy",
                        tracking=0.380,
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                sequence=["MH_05_difficult"],
                max_frames=400,
                max_behind_camera_ratio="0.3",
                min_fixed_to_optimized_ratio="0.34",
            )
            runs = load_latest_runs(args)

            self.assertEqual(
                runs["MH_05_difficult"]["writeback_gate_disabled"]["run_id"], "disabled"
            )
            self.assertEqual(
                runs["MH_05_difficult"]["writeback_gate_enabled_nogate"]["run_id"], "nogate"
            )
            # Gate variant must match the configured ratios, and pick the latest
            # among matching manifests (gate-wrong is excluded, gate-late wins
            # over gate-early by created_utc).
            self.assertEqual(
                runs["MH_05_difficult"]["writeback_gate_enabled_gate"]["run_id"], "gate-late"
            )
            self.assertNotIn("MH_01_easy", runs)
            self.assertEqual(missing_expected_runs(args, runs), [])

            table = render(args, runs)
            self.assertIn("| MH_05_difficult | disabled | 0.565 |", table)
            self.assertIn("| MH_05_difficult | enabled, no gate | 0.220 |", table)
            self.assertIn("| MH_05_difficult | enabled + gate | 0.300 |", table)
            self.assertIn("MH_05 still regresses", table)
            self.assertIn("Run-to-run nondeterminism", table)

    def test_missing_expected_runs_reports_gaps(self) -> None:
        args = Namespace(
            registry_dir=Path("."),
            sequence=["MH_05_difficult"],
            max_frames=400,
            max_behind_camera_ratio="0.3",
            min_fixed_to_optimized_ratio="0.34",
        )
        runs: dict = {"MH_05_difficult": {"writeback_gate_disabled": {}}}
        missing = missing_expected_runs(args, runs)
        self.assertEqual(
            missing,
            [
                ("MH_05_difficult", "writeback_gate_enabled_nogate"),
                ("MH_05_difficult", "writeback_gate_enabled_gate"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
