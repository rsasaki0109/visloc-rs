from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from summarize_inprocess_deep_slam import (  # noqa: E402
    load_latest_runs,
    missing_expected_runs,
    render,
)


def manifest(
    *,
    run_id: str,
    created_utc: str,
    frontend: str,
    sequence: str = "MH_03_medium",
    wall_clock: float = 1.0,
    verified_loops: int = 1,
    ate_se3: float = 0.1,
    ate_sim3: float = 0.1,
    dependency: str = "single Rust binary",
) -> dict:
    metrics = [
        {"name": "wall_clock", "value": wall_clock, "unit": "s", "primary": True},
        {"name": "verified_loops", "value": verified_loops, "unit": "count", "primary": False},
        {"name": "ate_se3", "value": ate_se3, "unit": "m", "primary": True},
        {"name": "ate_sim3", "value": ate_sim3, "unit": "m", "primary": False},
        {"name": "dependency", "value": dependency, "unit": None, "primary": False},
    ]
    return {
        "run_id": run_id,
        "created_utc": created_utc,
        "status": "success",
        "benchmark": {"id": "inprocess-deep-slam-wallclock"},
        "dataset": {"sequence": sequence},
        "config": {"params": {"frontend": frontend, "frames": 2700}},
        "metrics": metrics,
    }


class InprocessDeepSlamSummaryTests(unittest.TestCase):
    def test_load_latest_runs_filters_frontend_and_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp)
            (registry / "onnx-old.json").write_text(
                json.dumps(
                    manifest(
                        run_id="onnx-old",
                        created_utc="2026-07-03T00:00:00Z",
                        frontend="onnx",
                        wall_clock=199,
                        verified_loops=306,
                        ate_se3=0.051,
                        ate_sim3=0.047,
                        dependency="single Rust binary",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "onnx-new.json").write_text(
                json.dumps(
                    manifest(
                        run_id="onnx-new",
                        created_utc="2026-07-03T13:00:00Z",
                        frontend="onnx",
                        wall_clock=199,
                        verified_loops=306,
                        ate_se3=0.051,
                        ate_sim3=0.047,
                        dependency="single Rust binary",
                    )
                ),
                encoding="utf-8",
            )
            (registry / "filebased.json").write_text(
                json.dumps(
                    manifest(
                        run_id="filebased",
                        created_utc="2026-07-03T13:00:00Z",
                        frontend="file_based",
                        wall_clock=289,
                        verified_loops=319,
                        ate_se3=0.066,
                        ate_sim3=0.057,
                        dependency="Python + PyTorch + ~30 GB feature dump",
                    )
                ),
                encoding="utf-8",
            )
            # Different sequence entirely; must not leak into the MH_03 table.
            (registry / "onnx-other-seq.json").write_text(
                json.dumps(
                    manifest(
                        run_id="onnx-other-seq",
                        created_utc="2026-07-03T13:00:00Z",
                        frontend="onnx",
                        sequence="MH_05_difficult",
                    )
                ),
                encoding="utf-8",
            )

            args = Namespace(registry_dir=registry, sequence="MH_03_medium")
            runs = load_latest_runs(args)

            # Latest by created_utc wins for the onnx frontend.
            self.assertEqual(runs["onnx"]["run_id"], "onnx-new")
            self.assertEqual(runs["file_based"]["run_id"], "filebased")
            self.assertEqual(missing_expected_runs(args, runs), [])

            table = render(args, runs)
            self.assertIn("199 s", table)
            self.assertIn("289 s", table)
            self.assertIn("0.051 m", table)
            self.assertIn("0.066 m", table)
            self.assertIn("306", table)
            self.assertIn("319", table)
            self.assertIn("single Rust binary", table)
            self.assertIn("Python + PyTorch + ~30 GB feature dump", table)
            self.assertIn("1.45x faster", table)
            self.assertIn("135 fps", table)
            self.assertIn("34 fps", table)
            self.assertIn("23.9 fps", table)
            self.assertIn("NOT bit-identical", table)
            self.assertIn("documented prior GPU run, not reproduced this session", table)

    def test_missing_expected_runs_reports_gaps(self) -> None:
        args = Namespace(registry_dir=Path("."), sequence="MH_03_medium")
        runs: dict = {"onnx": {}}
        missing = missing_expected_runs(args, runs)
        self.assertEqual(missing, ["file_based"])


if __name__ == "__main__":
    unittest.main()
