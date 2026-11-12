"""Unit tests for scripts/lamaria_gps_agreement.py and scripts/lamaria_assemble_submission.py: a bent trajectory
agrees worse with the fixes than a straight one, and a submission keeps every entry but the ones replaced."""
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_assemble_submission as assemble  # noqa: E402
import lamaria_gps_agreement as agreement  # noqa: E402

LAT0, LON0, R_EARTH = 47.3769, 8.5417, 6378137.0


def estimate(path, t, xy):
    q = np.tile([0, 0, 0, 1.0], (len(t), 1))
    np.savetxt(path, np.c_[t, xy, np.zeros(len(t)), q], fmt=["%d"] + ["%.9f"] * 7)


def walk(n=600):
    t = (1e18 + np.arange(n) * 1e9).astype(np.int64)
    s = np.arange(n) * 1.3
    return t, np.c_[s * np.cos(0.3), s * np.sin(0.3)] + np.c_[np.zeros(n), 5 * np.sin(s / 60)]


class AgreementTest(unittest.TestCase):
    def test_a_bent_estimate_agrees_worse(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "old").mkdir(), (d / "new").mkdir(), (d / "gps").mkdir()
            t, xy = walk()
            rng = np.random.default_rng(0)
            en = 1.02 * xy @ np.array([[0.6, -0.8], [0.8, 0.6]]).T + [100, -50] + rng.normal(0, 2.0, xy.shape)
            lat = LAT0 + np.degrees(en[:, 1] / R_EARTH)
            lon = LON0 + np.degrees(en[:, 0] / (R_EARTH * np.cos(np.radians(LAT0))))
            with open(d / "gps" / "sequence_1_1.csv", "w") as f:
                f.write("t_ns,lat,lon,alt,accuracy,speed,provider\n")
                for k in range(len(t)):
                    f.write(f"{t[k]},{lat[k]:.9f},{lon[k]:.9f},400.0,5.00,1.3,gps\n")
            bend = np.radians(8) * np.linspace(0, 1, len(t))[:, None]  # a heading drift of 8 degrees
            rel = np.diff(xy, axis=0, prepend=xy[:1])
            bent = np.cumsum(np.c_[np.cos(bend[:, 0]) * rel[:, 0] - np.sin(bend[:, 0]) * rel[:, 1],
                                   np.sin(bend[:, 0]) * rel[:, 0] + np.cos(bend[:, 0]) * rel[:, 1]], axis=0)
            estimate(d / "old" / "sequence_1_1.txt", t, bent)
            estimate(d / "new" / "sequence_1_1.txt", t, xy)
            (row,) = agreement.compare(d / "old", d / "new", d / "gps", ["sequence_1_1"], log=lambda m: None)
            self.assertEqual(row[1], len(t))
            self.assertLess(row[3], 3.0)  # the fixes' own 2 m noise
            self.assertGreater(row[2], row[3] + 1.0)


class AssembleTest(unittest.TestCase):
    def test_only_the_named_estimates_change(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "new").mkdir()
            t, xy = walk(50)
            base = d / "base.zip"
            with zipfile.ZipFile(base, "w") as z:
                for s in ("sequence_1_1", "sequence_1_2"):
                    estimate(d / f"{s}.txt", t, xy)
                    z.write(d / f"{s}.txt", f"slam/{s}.txt")
            estimate(d / "new" / "sequence_1_1.txt", t, xy + 1.0)
            self.assertEqual(assemble.assemble(base, d / "new", d / "out.zip", ["sequence_1_1"]), 2)
            with zipfile.ZipFile(base) as a, zipfile.ZipFile(d / "out.zip") as b:
                self.assertEqual(a.namelist(), b.namelist())
                self.assertEqual(a.read("slam/sequence_1_2.txt"), b.read("slam/sequence_1_2.txt"))
                self.assertEqual(b.read("slam/sequence_1_1.txt"), (d / "new" / "sequence_1_1.txt").read_bytes())

    def test_a_replacement_may_add_poses(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "new").mkdir()
            t, xy = walk(50)
            base = d / "base.zip"
            estimate(d / "a.txt", t[10:], xy[10:])  # a late start
            with zipfile.ZipFile(base, "w") as z:
                z.write(d / "a.txt", "slam/sequence_1_1.txt")
            estimate(d / "new" / "sequence_1_1.txt", t, xy)
            self.assertEqual(assemble.assemble(base, d / "new", d / "out.zip", ["sequence_1_1"]), 1)
            with zipfile.ZipFile(d / "out.zip") as z:
                self.assertEqual(z.read("slam/sequence_1_1.txt"), (d / "new" / "sequence_1_1.txt").read_bytes())
            order = np.r_[1, 0, np.arange(2, 50)]
            estimate(d / "new" / "sequence_1_1.txt", t[order], xy[order])
            with self.assertRaises(SystemExit):
                assemble.assemble(base, d / "new", d / "out2.zip", ["sequence_1_1"])

    def test_a_replacement_must_keep_the_timestamps(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "new").mkdir()
            t, xy = walk(50)
            base = d / "base.zip"
            estimate(d / "a.txt", t, xy)
            with zipfile.ZipFile(base, "w") as z:
                z.write(d / "a.txt", "slam/sequence_1_1.txt")
            estimate(d / "new" / "sequence_1_1.txt", t[:-1], xy[:-1])
            with self.assertRaises(SystemExit):
                assemble.assemble(base, d / "new", d / "out.zip", ["sequence_1_1"])
            xy2 = xy.copy()
            xy2[3, 0] = np.nan
            estimate(d / "new" / "sequence_1_1.txt", t, xy2)
            with self.assertRaises(SystemExit):
                assemble.assemble(base, d / "new", d / "out.zip", ["sequence_1_1"])


if __name__ == "__main__":
    unittest.main()
