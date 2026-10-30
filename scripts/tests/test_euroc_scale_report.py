from pathlib import Path
import json
import math
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import euroc_scale_report

SCRIPT = Path(__file__).resolve().parents[1] / "euroc_scale_report.py"
T0 = 1403636579.0


def position(t):
    return np.array([3 * math.sin(0.3 * t), 2 * math.sin(0.6 * t), 1 + 0.3 * math.sin(0.2 * t)])


def write_ground_truth(path, seconds=60.0):
    with open(path, "w") as f:
        f.write("#timestamp, p_RS_R_x [m], p_RS_R_y [m], p_RS_R_z [m], q_RS_w [], q_RS_x [], q_RS_y [], q_RS_z []\n")
        for k in range(int(seconds * 200)):
            t = k / 200
            p = position(t)
            f.write(f"{int(round((T0 + t) * 1e9))},{p[0]},{p[1]},{p[2]},1,0,0,0\n")


def write_estimate(path, size_factor, seconds=60.0):
    """A TUM trajectory in seconds, as basalt_euroc_vio_demo writes it, `size_factor(t)`
    times the true one in a rotated, shifted frame."""
    angle = 0.7
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    with open(path, "w") as f:
        for k in range(int(seconds * 20)):
            t = k / 20
            p = rotation @ (position(t) * size_factor(t)) + np.array([5.0, -2.0, 1.0])
            f.write(f"{T0 + t:.9f} {p[0]:.9f} {p[1]:.9f} {p[2]:.9f} 0 0 0 1\n")


class EurocScaleReportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.gt = self.root / "gt.csv"
        write_ground_truth(self.gt)

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_trajectory_has_no_scale_error(self):
        write_estimate(self.root / "exact.tum", lambda t: 1.0)
        r = euroc_scale_report.report(self.gt, self.root / "exact.tum")
        self.assertLess(r["sim3_ate_m"], 1e-6)
        self.assertAlmostEqual(r["scale_error"], 0.0, places=6)
        self.assertEqual(r["settled_s"], 0.0)

    def test_constant_scale_error_is_reported_as_estimate_too_large(self):
        write_estimate(self.root / "large.tum", lambda t: 1.03)
        r = euroc_scale_report.report(self.gt, self.root / "large.tum")
        self.assertAlmostEqual(r["scale_error"], 0.03, places=6)
        self.assertLess(r["sim3_ate_m"], 1e-6)
        self.assertGreater(r["se3_ate_m"], 0.01)

    def test_scale_that_settles_late(self):
        # 30 % too large at the start, +2 % from 20 s on: the 10 s windows fall within
        # 5 % a little before 20 s and stay there.
        write_estimate(self.root / "late.tum", lambda t: 1.02 + 0.28 * max(0.0, 1 - t / 20))
        r = euroc_scale_report.report(self.gt, self.root / "late.tum")
        self.assertGreater(r["settled_s"], 8.0)
        self.assertLessEqual(r["settled_s"], 20.0)
        self.assertAlmostEqual(r["median_abs_window_error_after_settling"], 0.02, places=3)

    def test_trajectory_that_never_settles(self):
        write_estimate(self.root / "drift.tum", lambda t: 1.0 + 0.01 * t)
        r = euroc_scale_report.report(self.gt, self.root / "drift.tum")
        self.assertIsNone(r["settled_s"])
        self.assertIsNone(r["median_abs_window_error_after_settling"])

    def test_cli_rows_and_summary(self):
        write_estimate(self.root / "mono.tum", lambda t: 1.03)
        write_estimate(self.root / "stereo.tum", lambda t: 1.0)
        runs = [("mono", "mono.tum"), ("stereo", "stereo.tum"), ("mono", "missing.tum")]
        outs = []
        for i, (mode, trajectory) in enumerate(runs):
            out = self.root / f"run{i}.json"
            seq = "MH_02_easy" if trajectory == "missing.tum" else "MH_01_easy"
            subprocess.run(
                [sys.executable, str(SCRIPT), "run", "--sequence", seq, "--mode", mode,
                 "--ground-truth-csv", str(self.gt), "--trajectory", str(self.root / trajectory), "--out", str(out)],
                check=True, capture_output=True,
            )
            outs.append(out)
        self.assertIn("error", json.loads(outs[2].read_text()))
        table = subprocess.run(
            [sys.executable, str(SCRIPT), "summarize", *map(str, outs)], check=True, capture_output=True, text=True
        ).stdout
        self.assertIn("| MH_01_easy | mono | ", table)
        self.assertIn("+3.0 %", table)
        self.assertIn("| MH_02_easy | mono | failed:", table)
        self.assertIn("| MH_02_easy | stereo | – |", table)


if __name__ == "__main__":
    unittest.main()
