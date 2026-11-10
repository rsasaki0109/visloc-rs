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


class PassFeaturesTests(unittest.TestCase):
    def args(self, features="aliked"):
        return make_args(ba_pass_features=features,
                         ba_pass_features_tracks=list(submission.DEFAULT_PASS_FEATURES_TRACKS),
                         ba_pass_features_serials=list(submission.DEFAULT_PASS_FEATURES_SERIALS))

    def test_only_short_sequences_of_the_device_get_learned_keypoints(self):
        self.assertEqual(submission.pass_features(self.args(), 1, "1WM093701G1276"), "aliked")
        self.assertIsNone(submission.pass_features(self.args(), 1, "1WM09370081264"))
        self.assertIsNone(submission.pass_features(self.args(), 2, "1WM093701G1276"))
        self.assertIsNone(submission.pass_features(self.args(), 1, None))

    def test_off_by_default(self):
        self.assertIsNone(submission.pass_features(self.args(None), 1, "1WM093701G1276"))

    def test_the_serial_comes_from_the_factory_calibration(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(submission.device_serial(Path(d), "sequence_1_1"))
            (Path(d) / "sequence_1_1_factory_calib.json").write_text('{"Serial": "1WM093701G1276"}')
            self.assertEqual(submission.device_serial(Path(d), "sequence_1_1"), "1WM093701G1276")



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


class FillLateStartTests(unittest.TestCase):
    """The poses before a late start come from the VIO run backwards, unless that run fails or misses the late one."""

    def fill(self, speed=1.2, drift=0.0, crash=False):
        """fill_late_start on a walk whose VIO started 60 s in; the VIO run backwards walks at `speed` and drifts
        `drift` m/s away from the late one. Returns (the late trajectory, the result, the VIO's --euroc-dir)."""
        import tempfile
        from unittest import mock
        root = Path(tempfile.mkdtemp())
        inner = root / "seq"
        for cam in ("cam0", "cam1"):
            (inner / "mav0" / cam / "data").mkdir(parents=True)
            (inner / "mav0" / cam / "data.csv").write_text(
                "#timestamp [ns],filename\n" + "".join(f"{t * 50_000_000},{t}.png\n" for t in range(2000)))
        (inner / "mav0" / "imu0").mkdir(parents=True)
        (inner / "mav0" / "imu0" / "data.csv").write_text(
            "#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n" + "".join(f"{t * 10_000_000},0,0,0,0,0,9.81\n" for t in range(10_000)))
        late = root / "late.tum"
        late.write_text("".join(f"{t * 0.05:.9f} {1.2 * t * 0.05:.9f} 0 0 0 0 0 1\n" for t in range(1200, 2000)))
        folders = []

        def vio(command, **kwargs):  # the reversed folder's frames, walked back from where the late VIO is at 80 s
            folder = command[command.index("--euroc-dir") + 1]
            folders.append(folder)
            if crash:
                raise RuntimeError("the VIO crashed")
            rows = (folder / "mav0" / "cam0" / "data.csv").read_text().splitlines()[1:]
            first = int(rows[0].split(",")[0])
            out = command[command.index("--out-dir") + 1] / "trajectory.tum"
            out.write_text("".join(
                f"{int(r.split(',')[0]) // 10 ** 9}.{int(r.split(',')[0]) % 10 ** 9:09d} "
                f"{96.0 - speed * (int(r.split(',')[0]) - first) * 1e-9:.9f} "
                f"{drift * (int(r.split(',')[0]) - first) * 1e-9:.9f} 0 0 0 0 1\n" for r in rows))

        with mock.patch.object(submission, "run", vio), mock.patch.object(submission, "log", lambda m: None):
            result = submission.fill_late_start(make_args(), "sequence_4_2", inner, Path("calib.json"),
                                                Path("cfg.json"), late, 60, 4, root, True)
        return late, result, folders[0]

    def test_a_walk_run_backwards_fills_the_first_minute(self):
        late, filled, folder = self.fill()
        self.assertEqual(folder.name, "reversed")
        self.assertNotEqual(filled, late)
        rows = [[float(v) for v in r.split()] for r in filled.read_text().splitlines()]
        self.assertEqual(len(rows), 2000)  # a pose at every frame
        for i in (0, 600, 1299, 1300, 1999):  # joined at 65 s
            self.assertAlmostEqual(rows[i][0], i * 0.05, places=6)
            self.assertAlmostEqual(rows[i][1], 1.2 * i * 0.05, places=4)

    def test_a_run_backwards_that_runs_away_or_misses_the_late_one_is_not_used(self):
        for failed in ({"speed": 6.0}, {"drift": 1.0}, {"crash": True}):
            late, result, _ = self.fill(**failed)
            self.assertEqual(result, late, failed)


if __name__ == "__main__":
    unittest.main()
