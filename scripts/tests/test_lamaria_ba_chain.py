"""Unit tests for scripts/lamaria_ba_chain.py: the steps per track and resuming."""
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lamaria_ba_chain as chain  # noqa: E402


def plan(track):
    return {name: [str(c) for c in cmd]
            for name, cmd, _, _ in chain.steps("mav0", "vio.tum", "imu.csv", "calib.json", track, "w", 4)}


def value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


class StepsTest(unittest.TestCase):
    def test_by_day_the_pass_follows_the_online_windows(self):
        p = plan(2)
        self.assertEqual(list(p), ["features", "match", "loops", "online", "pass solve", "loop-correct",
                                   "pass solve with the revisits"])
        self.assertEqual(value(p["match"], "--tau-slope"), "0.3")
        self.assertEqual(value(p["loops"], "--step"), "4")
        self.assertEqual(value(p["online"], "--gyro-rw-warmup"), "10")
        self.assertNotIn("--gyro-rw-scale", p["online"])
        for name in ("pass solve", "pass solve with the revisits"):
            self.assertEqual(value(p[name], "--gyro-rw-scale"), "10")
            self.assertEqual(value(p[name], "--gyro-rw-warmup"), "10")
            self.assertIn("--est-focal", p[name])
        self.assertEqual(value(p["pass solve"], "--vio"), str(Path("w/online.tum")))
        self.assertIn("--loops", p["pass solve with the revisits"])
        self.assertNotIn("--clahe", p["features"])
        self.assertEqual(chain.result("w", 2), Path("w/ba2.tum"))

    def test_low_light_stops_at_the_online_windows(self):
        p = plan(4)
        self.assertEqual(list(p), ["features", "match", "loops", "online"])
        self.assertIn("--clahe", p["features"])
        self.assertEqual(value(p["features"], "--denoise"), "7")
        self.assertNotIn("--gyro-rw-warmup", p["online"])
        self.assertEqual(chain.result("w", 4), Path("w/online.tum"))

    def test_a_daylit_low_light_sequence_runs_the_day_options(self):
        p = {name: [str(c) for c in cmd]
             for name, cmd, _, _ in chain.steps("mav0", "vio.tum", "imu.csv", "calib.json", 4, "w", 4,
                                                low_light=False)}
        self.assertEqual(list(p), list(plan(2)))
        self.assertNotIn("--clahe", p["features"])
        self.assertEqual(value(p["online"], "--gyro-rw-warmup"), "10")
        self.assertEqual(chain.result("w", 4, low_light=False), Path("w/ba2.tum"))

    def test_no_focal_lengths_on_the_moving_platform(self):
        for cmd in plan(5).values():
            self.assertNotIn("--est-focal", cmd)

    def test_the_rematch_follows_the_pass_by_day_only(self):
        p = {name: [str(c) for c in cmd]
             for name, cmd, _, _ in chain.steps("mav0", "vio.tum", "imu.csv", "calib.json", 2, "w", 4, rematch=True)}
        self.assertEqual(list(p)[-3:], ["rematch", "rematch loops", "solve with the new matches"])
        for name in ("rematch", "rematch loops"):
            self.assertEqual(value(p[name], "--vio"), str(Path("w/ba2.tum")))
            self.assertEqual(value(p[name], "--work"), str(Path("w/rematch")))
        self.assertEqual(value(p["rematch"], "--tau-slope"), "0.3")
        solve = p["solve with the new matches"]
        self.assertEqual(value(solve, "--work"), str(Path("w/rematch")))
        self.assertEqual(value(solve, "--out"), str(Path("w/ba3.tum")))
        self.assertIn("--loops", solve)
        self.assertEqual(value(solve, "--gyro-rw-scale"), "10")
        self.assertEqual(chain.result("w", 2, rematch=True), Path("w/ba3.tum"))
        low = [name for name, *_ in chain.steps("mav0", "vio.tum", "imu.csv", "calib.json", 4, "w", 4, rematch=True)]
        self.assertEqual(low, ["features", "match", "loops", "online"])
        self.assertEqual(chain.result("w", 4, rematch=True), Path("w/online.tum"))


FAKE_BA = textwrap.dedent('''
    import sys
    from pathlib import Path
    a = sys.argv[1:]
    with open(Path(a[a.index("--work") + 1]) / "calls.txt", "a") as f:
        f.write(a[0] + "\\n")
    work = Path(a[a.index("--work") + 1])
    out = {"features": work / "features.npz", "match": work / "matches.npz", "loops": work / "loops.npz"}
    (out[a[0]] if a[0] in out else Path(a[a.index("--out") + 1])).write_text(a[0])
''')


class ResumeTest(unittest.TestCase):
    def test_a_second_run_skips_every_step(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake_ba.py"
            fake.write_text(FAKE_BA)
            work, out = Path(d) / "work", Path(d) / "out.tum"
            for _ in range(2):
                chain.run_chain("mav0", "vio.tum", "imu.csv", "calib.json", 1, work, out, ba=fake, log=lambda m: None)
            calls = (work / "calls.txt").read_text().split()
            self.assertEqual(calls, ["features", "match", "loops", "online", "solve", "loop-correct", "solve"])
            self.assertEqual(out.read_text(), "solve")  # the pass's last solve
            self.assertFalse((work / "matches.npz").exists())

    def test_the_rematch_reads_the_same_features_and_ends_the_chain(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "fake_ba.py"
            fake.write_text(FAKE_BA)
            work, out = Path(d) / "work", Path(d) / "out.tum"
            for _ in range(2):
                chain.run_chain("mav0", "vio.tum", "imu.csv", "calib.json", 1, work, out, ba=fake, log=lambda m: None,
                                rematch=True)
            calls = (work / "calls.txt").read_text().split() + (work / "rematch" / "calls.txt").read_text().split()
            self.assertEqual(calls, ["features", "match", "loops", "online", "solve", "loop-correct", "solve",
                                     "match", "loops", "solve"])
            self.assertEqual((work / "rematch" / "features.npz").read_text(), "features")
            self.assertEqual(out, chain.run_chain("mav0", "vio.tum", "imu.csv", "calib.json", 1, work, out, ba=fake,
                                                  log=lambda m: None, rematch=True))
            self.assertEqual((work / "ba3.tum").read_text(), "solve")
            self.assertFalse((work / "rematch" / "matches.npz").exists())


if __name__ == "__main__":
    unittest.main()
