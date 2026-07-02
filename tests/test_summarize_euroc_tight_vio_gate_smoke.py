from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_tight_vio_gate_smoke import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    sequence: str = "MH_03_medium",
    variant: str = "baseline",
    max_frames: int = 400,
    velocity_cap: float | None = None,
    cost_ratio_cap: float | None = None,
    tracking: float = 0.1,
    rigid: float = 0.2,
    rejects: int = 0,
    velocity_rejects: int = 0,
    mirrors: int = 1,
) -> dict:
    params = {
        "max_frames": max_frames,
        "variant": variant,
    }
    if velocity_cap is not None:
        params["local_vi_ba_reject_velocity_above_mps"] = velocity_cap
    if cost_ratio_cap is not None:
        params["local_vi_ba_reject_writeback_above"] = cost_ratio_cap
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "euroc-tight-vio-local-ba-gates"},
        "dataset": {"sequence": sequence},
        "config": {"params": params},
        "metrics": [
            {"name": "ate_rigid_rmse_m", "value": rigid},
            {"name": "ate_similarity_rmse_m", "value": rigid + 0.01},
            {"name": "ate_similarity_scale", "value": 1.05},
            {"name": "tracking_success_rate", "value": tracking},
            {"name": "map_keyframes", "value": 4},
            {"name": "local_vi_ba_quality_gate_rejections", "value": rejects},
            {"name": "local_vi_ba_velocity_gate_rejections", "value": velocity_rejects},
            {"name": "local_vi_ba_mirrors_into_imu_motion_model", "value": mirrors},
        ],
    }


class EurocTightVioGateSmokeSummaryTests(unittest.TestCase):
    def test_load_latest_runs_filters_and_renders_gate_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "baseline-old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="baseline-old",
                        created_utc="2026-06-19T00:00:00Z",
                        tracking=0.05,
                        rigid=0.3,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "baseline-new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="baseline-new",
                        created_utc="2026-06-19T01:00:00Z",
                        tracking=0.1,
                        rigid=0.2,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "gated10.json").write_text(
                json.dumps(
                    manifest(
                        run_id="gated10",
                        created_utc="2026-06-19T02:00:00Z",
                        variant="gated_10mps",
                        velocity_cap=10.0,
                        cost_ratio_cap=1.0,
                        tracking=0.2,
                        rigid=0.4,
                        rejects=3,
                        velocity_rejects=3,
                        mirrors=0,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "wrong-frame-count.json").write_text(
                json.dumps(
                    manifest(
                        run_id="wrong-frame-count",
                        created_utc="2026-06-19T03:00:00Z",
                        variant="velocity_tripwire_1mps",
                        max_frames=80,
                        velocity_cap=1.0,
                        tracking=0.9,
                        rigid=0.01,
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(
                registry_dir=registry,
                max_frames=400,
                sequence=["MH_03_medium"],
                variant=[
                    "baseline",
                    "gated_10mps",
                    "velocity_tripwire_1mps",
                ],
            )
            runs = load_latest_runs(args)

            self.assertEqual(runs[("MH_03_medium", "baseline")]["run_id"], "baseline-new")
            self.assertEqual(runs[("MH_03_medium", "gated_10mps")]["run_id"], "gated10")
            self.assertEqual(
                missing_expected_runs(args, runs),
                [("MH_03_medium", "velocity_tripwire_1mps")],
            )

            table = render(args, runs)
            self.assertIn(
                "| MH_03_medium | baseline | none |  | none | 0 | 0 |  | 1 | 0.100 | 0.000 | 4 | 0.2000 | 0.0000 | 0.2100 | 1.050000 | baseline | baseline-new |",
                table,
            )
            self.assertIn(
                "| MH_03_medium | gated_10mps | 10.0 |  | 1.00 | 3 | 3 |  | 0 | 0.200 | 0.100 | 4 | 0.4000 | 0.2000 | 0.4100 | 1.050000 | mixed | gated10 |",
                table,
            )
            self.assertIn(
                "| MH_03_medium | velocity_tripwire_1mps |  |  |  |  |  |  |  |  |  |  |  |  |  |  | missing | missing |",
                table,
            )
            self.assertNotIn("wrong-frame-count", table)


if __name__ == "__main__":
    unittest.main()
