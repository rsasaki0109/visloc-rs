from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_active_observation_sweep import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    sequence: str = "MH_03_medium",
    variant: str = "tracked_drop",
    floor: int = 20,
    fallback: str = "none",
    rigid: float = 1.0,
) -> dict:
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "euroc-keyframe-tracked-landmark-drop"},
        "dataset": {"sequence": sequence},
        "config": {
            "params": {
                "max_frames": 400,
                "variant": variant,
                "demo_args": (
                    "--covisibility-local-ba "
                    f"--covisibility-local-ba-min-active-observations {floor} "
                    f"--covisibility-local-ba-fallback-min-boundary-observations {fallback}"
                ),
            }
        },
        "metrics": [
            {"name": "tracking_success_rate", "value": 0.8},
            {"name": "ate_rigid_rmse_m", "value": rigid},
            {"name": "ate_similarity_rmse_m", "value": 0.15},
            {"name": "map_keyframes", "value": 66},
            {"name": "covisibility_local_ba_successes", "value": 35},
            {"name": "covisibility_local_ba_failures", "value": 29},
            {"name": "covisibility_local_ba_active_observation_gate_failures", "value": 4},
            {"name": "covisibility_local_ba_no_local_landmarks_failures", "value": 24},
            {"name": "covisibility_local_ba_solver_failures", "value": 0},
        ],
    }


class EurocActiveObservationSweepSummaryTests(unittest.TestCase):
    def test_load_latest_runs_filters_floor_fallback_and_takes_newest(self) -> None:
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
            (registry / "fallback.json").write_text(
                json.dumps(
                    manifest(
                        run_id="fallback",
                        created_utc="2026-06-19T02:00:00Z",
                        fallback="1",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "floor50.json").write_text(
                json.dumps(
                    manifest(
                        run_id="floor50",
                        created_utc="2026-06-19T03:00:00Z",
                        floor=50,
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                max_frames=400,
                sequence=["MH_03_medium"],
                active_floor=[20],
                fallback="none",
            )
            runs = load_latest_runs(args)

            self.assertEqual(set(runs.keys()), {(20, "MH_03_medium", "tracked_drop")})
            self.assertEqual(runs[(20, "MH_03_medium", "tracked_drop")]["run_id"], "new")
            self.assertEqual(
                missing_expected_runs(args, runs),
                [(20, "MH_03_medium", "fixed")],
            )

            table = render(args, runs)
            self.assertIn("| 20 | MH_03_medium | fixed |  |  |  |  |  |  |  |  |  | missing |", table)
            self.assertIn("| 20 | MH_03_medium | tracked_drop | 0.800 | 1.5000 |", table)
            self.assertIn("new", table)
            self.assertNotIn(" | fallback |", table)


if __name__ == "__main__":
    unittest.main()
