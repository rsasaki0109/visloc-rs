"""Unit tests for scripts/lamaria_gps_bend.py: a slow heading and scale drift is bent back, a good trajectory is kept."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(SCRIPTS / "tests"))
from test_lamaria_gps_steps import estimate, gps_csv, walk  # noqa: E402


def drift(t, xy, deg_per_min=1.5, pct_per_min=-1.5):
    """The walk's steps turned and stretched by a heading and scale that drift linearly in time."""
    h = np.radians(deg_per_min) * t / 60
    s = np.exp(pct_per_min / 100 * t / 60)
    d = np.diff(xy, axis=0)
    c, sn = np.cos(h[:-1]), np.sin(h[:-1])
    d = np.c_[c * d[:, 0] - sn * d[:, 1], sn * d[:, 0] + c * d[:, 1]] * s[:-1, None]
    return np.vstack([xy[:1], xy[0] + np.cumsum(d, axis=0)])


def error(path, t, xy):
    """Median horizontal error of an estimate against the walk after a 2D similarity."""
    sys.path.insert(0, str(SCRIPTS))
    from lamaria_gps_bend import similarity
    e = np.loadtxt(path)
    c, r, tr = similarity(e[:, 1:3], xy, np.ones(len(xy)))
    return np.median(np.linalg.norm(c * e[:, 1:3] @ r.T + tr - xy, axis=1))


class BendTest(unittest.TestCase):
    def run_bend(self, est_xy, *extra):
        t, xy = walk()
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            gps_csv(d / "g.csv", t, xy)
            estimate(d / "e.txt", t, est_xy(t, xy))
            out = subprocess.run([sys.executable, str(SCRIPTS / "lamaria_gps_bend.py"), str(d / "g.csv"),
                                  str(d / "e.txt"), str(d / "o.txt"), *extra],
                                 capture_output=True, text=True, check=True).stdout
            return out, error(d / "e.txt", t, xy), error(d / "o.txt", t, xy), np.loadtxt(d / "o.txt"), \
                np.loadtxt(d / "e.txt")

    def test_a_drift_is_bent_back(self):
        out, before, after, o, e = self.run_bend(drift)
        self.assertIn("bent", out)
        self.assertGreater(before, 8.0)
        self.assertLess(after, before / 2)
        np.testing.assert_array_equal(o[:, 0], e[:, 0])
        np.testing.assert_allclose(o[:, 3], e[:, 3])

    def test_a_good_trajectory_is_written_as_it_is(self):
        out, _, _, o, e = self.run_bend(lambda t, xy: xy)
        self.assertIn("written as it is", out)
        np.testing.assert_allclose(o, e, atol=1e-6)

    def test_the_gate_can_be_opened(self):
        out, *_ = self.run_bend(lambda t, xy: xy, "--min-residual", "0")
        self.assertIn("bent", out)


if __name__ == "__main__":
    unittest.main()
