from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_sfm_vs_colmap import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    engine: str,
    sequence: str = "MH_03_medium",
    wall_clock: float = 1.0,
    wall_clock_unit: str = "min",
    ate_sim3: float = 0.1,
    ate_se3_metric: float | None = None,
    mean_reprojection: float = 1.0,
    metric_scale: bool = True,
) -> dict:
    metrics = [
        {"name": "wall_clock", "value": wall_clock, "unit": wall_clock_unit, "primary": True},
        {"name": "registered_frames", "value": 2700, "unit": "count", "primary": False},
        {"name": "registration_rate", "value": 1.0, "unit": "ratio", "primary": False},
        {"name": "mean_reprojection", "value": mean_reprojection, "unit": "px", "primary": False},
        {"name": "ate_sim3", "value": ate_sim3, "unit": "m", "primary": False},
        {"name": "metric_scale", "value": metric_scale, "unit": None, "primary": False},
        {"name": "downstream_3dgs", "value": "blurry", "unit": None, "primary": False},
    ]
    if ate_se3_metric is not None:
        metrics.append(
            {"name": "ate_se3_metric", "value": ate_se3_metric, "unit": "m", "primary": False}
        )
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "sfm-vs-colmap"},
        "dataset": {"sequence": sequence},
        "config": {"params": {"engine": engine, "frames": 2700}},
        "metrics": metrics,
    }


class SfmVsColmapSummaryTests(unittest.TestCase):
    def test_load_latest_runs_filters_engine_and_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "visloc-old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="visloc-old",
                        created_utc="2026-07-03T00:00:00Z",
                        engine="visloc",
                        wall_clock=6,
                        ate_sim3=0.13,
                        ate_se3_metric=0.066,
                        mean_reprojection=2.60,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "visloc-new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="visloc-new",
                        created_utc="2026-07-03T12:00:00Z",
                        engine="visloc",
                        wall_clock=6,
                        ate_sim3=0.13,
                        ate_se3_metric=0.066,
                        mean_reprojection=2.60,
                    )
                ),
                encoding="utf-8",
            )
            (registry / "colmap.json").write_text(
                json.dumps(
                    manifest(
                        run_id="colmap",
                        created_utc="2026-07-03T12:00:00Z",
                        engine="colmap",
                        wall_clock=11.7,
                        wall_clock_unit="h",
                        ate_sim3=2.18,
                        mean_reprojection=0.58,
                        metric_scale=False,
                    )
                ),
                encoding="utf-8",
            )
            # Different sequence entirely; must not leak into the MH_03 table.
            (registry / "visloc-other-seq.json").write_text(
                json.dumps(
                    manifest(
                        run_id="visloc-other-seq",
                        created_utc="2026-07-03T12:00:00Z",
                        engine="visloc",
                        sequence="MH_01_easy",
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(registry_dir=registry, sequence="MH_03_medium")
            runs = load_latest_runs(args)

            # Latest by created_utc wins for the visloc engine.
            self.assertEqual(runs["visloc"]["run_id"], "visloc-new")
            self.assertEqual(runs["colmap"]["run_id"], "colmap")
            self.assertEqual(missing_expected_runs(args, runs), [])

            table = render(args, runs)
            self.assertIn("6 min", table)
            self.assertIn("11.7 h", table)
            self.assertIn("0.130 m (Sim3, model) / 0.066 m (SE3, metric VO)", table)
            self.assertIn("2.180 m (Sim3)", table)
            self.assertIn("2.60 px", table)
            self.assertIn("0.58 px", table)
            self.assertIn("blurry", table)
            self.assertIn("~117x faster", table)
            self.assertIn("~17-33x more accurate", table)
            self.assertIn("0.37 cm", table)
            self.assertIn("1.64 cm", table)
            self.assertIn("capture-geometry-limited", table)
            self.assertIn("prior-run reference, not reproduced this session", table)

    def test_missing_expected_runs_reports_gaps(self) -> None:
        args = Namespace(registry_dir=Path("."), sequence="MH_03_medium")
        runs: dict = {"visloc": {}}
        missing = missing_expected_runs(args, runs)
        self.assertEqual(missing, ["colmap"])


if __name__ == "__main__":
    unittest.main()
