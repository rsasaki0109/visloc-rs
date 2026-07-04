from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_euroc_covisibility_anchor_gate import (  # noqa: E402
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
    anchor_weight: float | None = None,
    tracking: float = 0.5,
    rigid: float = 1.0,
) -> dict:
    params: dict = {
        "variant": variant,
        "max_frames": 400,
    }
    if variant == "enabled":
        params["covisibility_local_ba_anchor_weight"] = anchor_weight
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
            {"name": "covisibility_local_ba_successes", "value": 6},
        ],
    }


class EurocCovisibilityAnchorGateSummaryTests(unittest.TestCase):
    def test_load_latest_runs_pins_the_anchor_gate_stamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            stamp = "20260703T170345Z"

            def write(name: str, run_id: str, **kwargs: object) -> None:
                (registry / name).write_text(
                    json.dumps(manifest(run_id=run_id, **kwargs)),
                    encoding="utf-8",
                )

            # The anchor-gate manifests we want to select.
            write(
                "mh05-disabled.json",
                f"euroc-covisibility-local-ba-MH_05_difficult-disabled-{stamp}",
                created_utc="2026-07-03T17:04:00Z",
                variant="disabled",
                tracking=0.565,
                rigid=0.1139,
            )
            write(
                "mh05-enabled.json",
                f"euroc-covisibility-local-ba-MH_05_difficult-enabled-{stamp}",
                created_utc="2026-07-03T17:06:00Z",
                variant="enabled",
                anchor_weight=10.0,
                tracking=0.420,
                rigid=0.0884,
            )
            # An older window-sweep run sharing the bare disabled/enabled
            # variant tokens on the same benchmark id -- must NOT be picked
            # up even though it is "latest" by naive created_utc scanning if
            # it were newer.
            write(
                "mh05-window-sweep-enabled.json",
                "euroc-covisibility-local-ba-MH_05_difficult-enabled-20260619T211409Z",
                created_utc="2026-07-04T00:00:00Z",
                variant="enabled",
                anchor_weight=None,
                tracking=0.220,
                rigid=0.9999,
            )
            # A writeback-gate manifest with a namespaced variant token and a
            # different stamp -- must NOT be picked up either.
            write(
                "mh05-writeback-nogate.json",
                "euroc-covisibility-local-ba-writeback_gate_enabled_nogate-"
                "MH_05_difficult-20260703T000000Z",
                created_utc="2026-07-03T00:00:00Z",
                variant="writeback_gate_enabled_nogate",
                tracking=0.220,
                rigid=0.1683,
            )
            # Different sequence entirely; must not leak into MH_05 rows.
            write(
                "mh01-disabled.json",
                f"euroc-covisibility-local-ba-MH_01_easy-disabled-{stamp}",
                created_utc="2026-07-03T17:04:00Z",
                sequence="MH_01_easy",
                variant="disabled",
                tracking=0.380,
                rigid=0.0642,
            )

            args = Namespace(
                registry_dir=registry,
                sequence=["MH_05_difficult"],
                max_frames=400,
            )
            runs = load_latest_runs(args)

            self.assertEqual(
                runs["MH_05_difficult"]["disabled"]["run_id"],
                f"euroc-covisibility-local-ba-MH_05_difficult-disabled-{stamp}",
            )
            self.assertEqual(
                runs["MH_05_difficult"]["enabled"]["run_id"],
                f"euroc-covisibility-local-ba-MH_05_difficult-enabled-{stamp}",
            )
            self.assertNotIn("MH_01_easy", runs)
            self.assertEqual(missing_expected_runs(args, runs), [])

            table = render(args, runs)
            self.assertIn("| MH_05_difficult | disabled | 0.1139 |", table)
            self.assertIn("| MH_05_difficult | enabled (anchor w=10) | 0.0884 |", table)
            self.assertIn("ALL THREE sequences simultaneously", table)
            self.assertIn("REVERSED", table)

    def test_missing_expected_runs_reports_gaps(self) -> None:
        args = Namespace(
            registry_dir=Path("."),
            sequence=["MH_05_difficult"],
            max_frames=400,
        )
        runs: dict = {"MH_05_difficult": {"disabled": {}}}
        missing = missing_expected_runs(args, runs)
        self.assertEqual(missing, [("MH_05_difficult", "enabled")])


if __name__ == "__main__":
    unittest.main()
