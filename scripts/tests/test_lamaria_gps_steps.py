"""Unit tests for scripts/lamaria_gps_steps.py: a missed ride is found and bridged, a whole trajectory is kept."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_gps_steps as steps  # noqa: E402

LAT0, LON0 = 47.3769, 8.5417


def walk(duration=1200.0, rate=10.0):
    """A walk at 1.4 m/s that turns every 120 s; straight from 480 s to 600 s (the ride the tests drop)."""
    t = np.arange(0, duration, 1 / rate)
    heading = np.radians(np.array([0, 70, -20, 50, 120, 120, 30, -40, 10, 80])[(t // 120).astype(int) % 10])
    v = 1.4 * np.c_[np.cos(heading), np.sin(heading)]
    xy = np.vstack([[0, 0], np.cumsum(v[:-1] / rate, axis=0)])
    return t, xy


def gps_csv(path, t, xy, seed=0):
    """1 Hz fixes of the walk in another frame (30 degrees, scale 1.02, offset), 2 m noise, accuracy 5 m."""
    rng = np.random.default_rng(seed)
    th, c = np.radians(30), 1.02
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    tg = np.arange(0, t[-1], 1.0)
    p = np.stack([np.interp(tg, t, xy[:, i]) for i in range(2)], 1)
    en = c * p @ rot.T + [120.0, -40.0] + rng.normal(0, 2.0, (len(tg), 2))
    lat = LAT0 + np.degrees(en[:, 1] / steps.R_EARTH)
    lon = LON0 + np.degrees(en[:, 0] / (steps.R_EARTH * np.cos(np.radians(LAT0))))
    with open(path, "w") as f:
        f.write("t_ns,lat,lon,alt,accuracy,speed,provider\n")
        for k in range(len(tg)):
            f.write(f"{int(1e12 + tg[k] * 1e9)},{lat[k]:.9f},{lon[k]:.9f},400.0,5.00,1.4,gps\n")


def miss(t, xy, s0, e0):
    """The walk as a trajectory that stands still from s0 to e0 and carries on from there, displaced."""
    out = xy.copy()
    i0, i1 = np.searchsorted(t, [s0, e0])
    out[i0:i1] = xy[i0]
    out[i1:] -= xy[i1] - xy[i0]
    return out


def estimate(path, t, xy):
    """Estimate format: integer ns, positions, an identity rotation."""
    with open(path, "w") as f:
        for k in range(len(t)):
            f.write(f"{int(1e12 + round(t[k] * 1e9))} {xy[k, 0]:.6f} {xy[k, 1]:.6f} 1.500000 "
                    f"0.000000000 0.000000000 0.000000000 1.000000000\n")


class StepsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.t, cls.xy = walk()
        gps_csv(cls.dir / "gps.csv", cls.t, cls.xy)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def ramps_of(self, xy):
        estimate(self.dir / "est.txt", self.t, xy)
        _, _, ts, sg, fxy, en = steps.load(self.dir / "gps.csv", self.dir / "est.txt", 10.0)
        return ts, sg, fxy, en, steps.find_ramps(sg, fxy, en, min_step=25.0, log=lambda _: None)

    def test_a_whole_walk_has_no_step(self):
        *_, ramps = self.ramps_of(self.xy)
        self.assertEqual(ramps, [])

    def test_a_missed_ride_is_found_and_bridged(self):
        missed = miss(self.t, self.xy, 500, 590)
        ts, sg, fxy, en, ramps = self.ramps_of(missed)
        self.assertEqual(len(ramps), 1)
        s0, e0 = ramps[0]
        self.assertLess(s0, 590)
        self.assertGreater(e0, 500)
        c, th, levels, _, _ = steps.fit(sg, fxy, en, ramps)
        bridged, moves = steps.bridge(ts, missed, ramps, c, th, levels)
        self.assertAlmostEqual(moves[0], 1.4 * 90, delta=5.0)
        after = ts > max(e0, 590)
        self.assertLess(np.abs(bridged[after] - self.xy[after]).max(), 4.0)
        before = ts < s0
        np.testing.assert_array_equal(bridged[before], missed[before])

    def test_the_script_keeps_the_estimate_format(self):
        missed = miss(self.t, self.xy, 500, 590)
        estimate(self.dir / "missed.txt", self.t, missed)
        out = self.dir / "bridged.txt"
        log = subprocess.run([sys.executable, str(SCRIPTS / "lamaria_gps_steps.py"), str(self.dir / "gps.csv"),
                              str(self.dir / "missed.txt"), str(out)], capture_output=True, text=True, check=True)
        self.assertIn("moved by", log.stdout)
        a, b = np.loadtxt(self.dir / "missed.txt"), np.loadtxt(out)
        first = out.read_text().split("\n", 1)[0].split()
        self.assertEqual(first[0], str(int(1e12)))
        np.testing.assert_array_equal(a[:, [0, 3, 4, 5, 6, 7]], b[:, [0, 3, 4, 5, 6, 7]])
        self.assertGreater(np.abs(a[:, 1:3] - b[:, 1:3]).max(), 100)

    def test_too_few_fixes_leave_the_trajectory(self):
        estimate(self.dir / "short.txt", self.t[:400], self.xy[:400])
        out = self.dir / "short_out.txt"
        log = subprocess.run([sys.executable, str(SCRIPTS / "lamaria_gps_steps.py"), str(self.dir / "gps.csv"),
                              str(self.dir / "short.txt"), str(out), "--max-acc", "1"],
                             capture_output=True, text=True, check=True)
        self.assertIn("too few", log.stdout)
        self.assertEqual(out.read_text(), (self.dir / "short.txt").read_text())


if __name__ == "__main__":
    unittest.main()
