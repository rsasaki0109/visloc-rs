from pathlib import Path
import argparse
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_lamaria_test_submission as submission


def make_args(**overrides):
    args = argparse.Namespace(vio_exe=Path("vio_demo"), slam_exe=None,
                              config=Path("cfg.json"), threads=12)
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


class EstimatorCommandTests(unittest.TestCase):
    def test_vio_only(self):
        command, trajectories = submission.estimator_command(
            make_args(), Path("seq"), Path("calib.json"), Path("out"))
        self.assertEqual(command[0], Path("vio_demo"))
        for flag in ("--pipeline", "--no-trace", "--no-marg-data"):
            self.assertIn(flag, command)
        self.assertEqual(trajectories, [Path("out/trajectory.tum")])

    def test_online_mapper_takes_precedence(self):
        command, trajectories = submission.estimator_command(
            make_args(slam_exe=Path("slam_demo")), Path("seq"), Path("calib.json"), Path("out"))
        self.assertEqual(command[0], Path("slam_demo"))
        self.assertIn("--no-urgent-keyframes", command)
        self.assertEqual(command[command.index("--loop-match-max-rot-error") + 1], 5)
        self.assertNotIn("--no-marg-data", command)
        self.assertEqual(command[command.index("--euroc-dir") + 1], Path("seq"))
        self.assertEqual(command[command.index("--calibration") + 1], Path("calib.json"))
        self.assertEqual(command[command.index("--threads") + 1], 12)
        self.assertEqual(trajectories, [Path("out/trajectory_online.tum"),
                                        Path("out/trajectory_vio.tum")])

    def test_vio_track_with_the_online_demo_only_submits_its_vio_trajectory(self):
        command, trajectories = submission.estimator_command(
            make_args(vio_exe=None, slam_exe=Path("slam_demo")), Path("seq"),
            Path("calib.json"), Path("out"), submit_vio=True)
        self.assertEqual(command[0], Path("slam_demo"))
        self.assertEqual(trajectories, [Path("out/trajectory_vio.tum")])

    def test_vio_track_prefers_the_vio_only_build(self):
        command, trajectories = submission.estimator_command(
            make_args(slam_exe=Path("slam_demo")), Path("seq"), Path("calib.json"),
            Path("out"), submit_vio=True)
        self.assertEqual(command[0], Path("vio_demo"))
        self.assertEqual(trajectories, [Path("out/trajectory.tum")])

    def test_per_sequence_config_overrides_the_default(self):
        command, _ = submission.estimator_command(
            make_args(), Path("seq"), Path("calib.json"), Path("out"),
            config=Path("lowlight.json"))
        self.assertEqual(command[command.index("--config") + 1], Path("lowlight.json"))


class TrackTests(unittest.TestCase):
    def test_sequence_track(self):
        self.assertEqual(submission.sequence_track("sequence_4_10"), 4)
        self.assertEqual(submission.sequence_track("sequence_1_19"), 1)

    def test_low_light_config_ships_with_the_repo(self):
        repo = Path(__file__).resolve().parents[2]
        self.assertTrue((repo / submission.DEFAULT_LOW_LIGHT_CONFIG).is_file())

    def test_ba_chain_configs_ship_with_the_repo(self):
        repo = Path(__file__).resolve().parents[2]
        for config in (submission.DEFAULT_CONFIG, submission.BA_CHAIN_CONFIG, submission.BA_CHAIN_LOW_LIGHT_CONFIG):
            self.assertTrue((repo / config).is_file(), config)



class DivergenceTests(unittest.TestCase):
    def write(self, speed, seconds=60, rate=20):
        import tempfile
        path = Path(tempfile.mkdtemp()) / "trajectory.tum"
        path.write_text("".join(f"{i / rate:.3f} {speed(i / rate):.3f} 0 0 0 0 0 1\n" for i in range(seconds * rate)))
        return path

    def test_a_walk_or_a_tram_is_not_divergence(self):
        self.assertFalse(submission.diverged(self.write(lambda t: 1.4 * t)))
        self.assertFalse(submission.diverged(self.write(lambda t: 20.0 * t)))

    def test_a_runaway_is(self):
        self.assertTrue(submission.diverged(self.write(lambda t: 3.0 * t * t)))

    def test_a_walk_that_runs_away_for_a_while_fails_on_a_walking_track_only(self):
        # 1.4 m/s, then 10 m/s from 20 s to 40 s (test sequence_4_2), then walking again
        walk = self.write(lambda t: 1.4 * t + 8.6 * min(max(t - 20.0, 0.0), 20.0), seconds=120)
        self.assertFalse(submission.diverged(walk))
        seconds, end = submission.runaway(walk)
        self.assertGreater(seconds, 15)
        self.assertAlmostEqual(end, 45.0, delta=6.0)
        self.assertGreater(submission.vio_failure(walk, 4), 0)
        self.assertEqual(submission.vio_failure(walk, submission.MOVING_TRACK), 0)  # a tram may do that
        self.assertEqual(submission.vio_failure(self.write(lambda t: 1.4 * t, seconds=120), 4), 0)

    def test_the_multicam_fallback_keeps_the_config(self):
        import json
        import tempfile
        repo = Path(__file__).resolve().parents[2]
        out = submission.multicam_config(repo / submission.BA_CHAIN_LOW_LIGHT_CONFIG,
                                         Path(tempfile.mkdtemp()) / "c.json")
        cfg = json.loads(out.read_text())["value0"]
        self.assertTrue(cfg["config.vio_landmarks_all_cameras"])
        self.assertEqual(cfg["config.optical_flow_max_recovered_dist2"], 0.25)  # the low-light setting stays


    def test_mean_brightness_reads_the_manifests_images(self):
        import tempfile
        import numpy as np
        try:
            import cv2
        except ImportError:
            self.skipTest("opencv not installed")
        cam = Path(tempfile.mkdtemp()) / "cam0"
        (cam / "data").mkdir(parents=True)
        rows = []
        for i, level in enumerate((20, 80, 140)):
            cv2.imwrite(str(cam / "data" / f"{i}.png"), np.full((8, 8), level, np.uint8))
            rows.append(f"{i},{i}.png")
        (cam / "data.csv").write_text("#timestamp [ns],filename\n" + "\n".join(rows) + "\n")
        self.assertAlmostEqual(submission.mean_brightness(cam), 80.0)

    def test_a_late_start_trims_the_manifests_and_links_the_images(self):
        import tempfile
        root = Path(tempfile.mkdtemp())
        inner = root / "seq"
        for cam in ("cam0", "cam1"):
            (inner / "mav0" / cam / "data").mkdir(parents=True)
            (inner / "mav0" / cam / "data.csv").write_text(
                "#timestamp [ns],filename\n" + "".join(f"{t * 50_000_000},{t}.png\n" for t in range(1000)))
        (inner / "mav0" / "imu0").mkdir(parents=True)
        (inner / "mav0" / "imu0" / "data.csv").write_text(
            "#timestamp [ns],w\n" + "".join(f"{t * 1_000_000},0\n" for t in range(50_000)))
        late = submission.late_start(inner, root / "late", 10)
        cam = (late / "mav0" / "cam1" / "data.csv").read_text().splitlines()
        self.assertEqual(cam[0], "#timestamp [ns],filename")
        self.assertEqual(cam[1], "10000000000,200.png")
        self.assertEqual(len(cam), 801)
        imu = (late / "mav0" / "imu0" / "data.csv").read_text().splitlines()
        self.assertEqual(imu[1].split(",")[0], "9900000000")  # a tenth of a second before the first image
        self.assertEqual((late / "mav0" / "cam0" / "data").resolve(), (inner / "mav0" / "cam0" / "data").resolve())


if __name__ == "__main__":
    unittest.main()
