"""Unit tests for scripts/lamaria_reversed_start.py: the time-reversed folder, and the join of the two runs."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_reversed_start as fill  # noqa: E402

T0 = 1_000_000_000_000  # the first frame's time (ns)
FRAME_NS = 50_000_000


def recording(root, seconds=100):
    """An EuRoC folder with 20 Hz frames and a 100 Hz IMU whose values count up."""
    inner = Path(root) / "seq"
    for cam in ("cam0", "cam1"):
        (inner / "mav0" / cam / "data").mkdir(parents=True)
        (inner / "mav0" / cam / "data.csv").write_text("#timestamp [ns],filename\n" + "".join(
            f"{T0 + i * FRAME_NS},{T0 + i * FRAME_NS}.png\n" for i in range(seconds * 20)))
    (inner / "mav0" / "imu0").mkdir(parents=True)
    (inner / "mav0" / "imu0" / "data.csv").write_text("#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z\n" + "".join(
        f"{T0 + i * 10_000_000},{0.001 * i},-0.5,0.25,{0.01 * i},9.81,-0.3\n" for i in range(-50, seconds * 100)))
    return inner


def walk(ns):
    """A walk's poses at `ns`: 1.2 m/s forward, swaying, the heading turning (position and quaternion rows)."""
    t = (np.asarray(ns) - T0) * 1e-9
    q = Rotation.from_euler("zyx", np.c_[0.02 * t, 0.1 * np.sin(0.7 * t), 0.05 * np.cos(0.4 * t)]).as_quat()
    return np.c_[1.2 * t, np.sin(0.3 * t), 0.05 * np.sin(2.0 * t), q]


def elsewhere(poses, yaw_deg=40.0, offset=(3.0, -2.0, 0.5)):
    """The same poses in another world frame, as a second VIO run has its own."""
    r = Rotation.from_euler("z", yaw_deg, degrees=True)
    return np.c_[r.apply(poses[:, :3]) + offset, (r * Rotation.from_quat(poses[:, 3:])).as_quat()]


def frame_times(folder):
    return np.array([int(r.split(",")[0]) for r in fill.manifest(folder / "mav0" / "cam0" / "data.csv")[1]], np.int64)


class ReversedStartTest(unittest.TestCase):
    def test_the_frames_run_backwards_under_mirrored_times(self):
        root = Path(tempfile.mkdtemp())
        inner = recording(root)
        c = fill.reversed_start(inner, root / "reversed", 80)
        self.assertEqual(int((root / "reversed" / "mirror_ns.txt").read_text()), c)
        for cam in ("cam0", "cam1"):
            head, rows = fill.manifest(root / "reversed" / "mav0" / cam / "data.csv")
            self.assertEqual(head, ["#timestamp [ns],filename"])
            self.assertEqual(len(rows), 80 * 20 + 1)
            times = [int(r.split(",")[0]) for r in rows]
            self.assertEqual(times, sorted(times))
            self.assertEqual(rows[0].split(",")[1], f"{T0 + 80 * 20 * FRAME_NS}.png")  # the last frame first
            self.assertEqual(rows[-1].split(",")[1], f"{T0}.png")
            for row in rows:  # each image under its mirrored time, never under another frame's own
                t, name = row.split(",")
                self.assertEqual(c - int(t), int(name[:-4]))
            self.assertFalse(set(times) & set(frame_times(inner).tolist()))
            self.assertEqual((root / "reversed" / "mav0" / cam / "data").resolve(),
                             (inner / "mav0" / cam / "data").resolve())

    def test_the_gyroscope_changes_sign_and_the_accelerometer_does_not(self):
        root = Path(tempfile.mkdtemp())
        inner = recording(root)
        c = fill.reversed_start(inner, root / "reversed", 80)
        head, rows = fill.manifest(root / "reversed" / "mav0" / "imu0" / "data.csv")
        self.assertEqual(head, ["#timestamp [ns],w_x,w_y,w_z,a_x,a_y,a_z"])
        times = [int(r.split(",")[0]) for r in rows]
        self.assertEqual(times, sorted(times))
        self.assertEqual(c - times[-1], T0 - 100_000_000)  # a tenth of a second around the frames
        self.assertEqual(c - times[0], T0 + 80_100_000_000)
        at_first = rows[-11].split(",")  # the sample at the first frame: its counting values are 0
        self.assertEqual(c - int(at_first[0]), T0)
        self.assertEqual([float(v) for v in at_first[1:]], [0.0, 0.5, -0.25, 0.0, 9.81, -0.3])
        before = rows[-1].split(",")  # a tenth of a second earlier: the negative rate turns positive
        self.assertEqual([float(v) for v in before[1:]], [0.01, 0.5, -0.25, -0.1, 9.81, -0.3])


class JoinTest(unittest.TestCase):
    def runs(self, root, late=60.0, backward_error=0.0):
        """The late VIO's trajectory, and the reversed run's as the VIO writes it (mirrored times, its own frame)."""
        inner = recording(root)
        ns = frame_times(inner)
        forward = root / "late.tum"
        kept = ns[ns >= T0 + int(late * 1e9)]
        fill.write_tum(forward, kept, walk(kept))
        c = fill.reversed_start(inner, root / "reversed", late + 20)
        mirrored = frame_times(root / "reversed")
        poses = elsewhere(walk(c - mirrored))
        poses[:, 0] += backward_error * (c - mirrored - T0) * 1e-9  # a run that drifts away from the late one
        fill.write_tum(root / "reversed.tum", mirrored, poses)
        return inner, forward, fill.forward_times(root / "reversed.tum", c, root / "backward.tum")

    def test_the_reversed_run_fills_the_frames_before_the_join(self):
        root = Path(tempfile.mkdtemp())
        inner, forward, backward = self.runs(root)
        messages = []
        residual = fill.join(forward, backward, inner / "mav0", root / "filled.tum", log=messages.append)
        self.assertLess(residual, 1e-6)
        self.assertIn("1300 frames before it (0 with the nearest pose)", messages[0])
        ns, poses = fill.read_tum(root / "filled.tum")
        np.testing.assert_array_equal(ns, frame_times(inner))  # every frame, in order
        truth = walk(ns)
        np.testing.assert_allclose(poses[:, :3], truth[:, :3], atol=1e-6)
        turn = (Rotation.from_quat(poses[:, 3:]) * Rotation.from_quat(truth[:, 3:]).inv()).magnitude()
        self.assertLess(turn.max(), 1e-6)
        late_ns, late = fill.read_tum(forward)
        np.testing.assert_array_equal(poses[ns >= T0 + 65_000_000_000], late[late_ns >= T0 + 65_000_000_000])

    def test_runs_that_disagree_leave_a_residual(self):
        root = Path(tempfile.mkdtemp())
        inner, forward, backward = self.runs(root, backward_error=1.0)
        self.assertGreater(fill.join(forward, backward, inner / "mav0", root / "filled.tum", log=lambda m: None), 0.5)

    def test_a_frame_the_reversed_run_skipped_gets_the_nearest_pose(self):
        root = Path(tempfile.mkdtemp())
        inner, forward, backward = self.runs(root)
        ns, poses = fill.read_tum(backward)
        fill.write_tum(backward, ns[3:], poses[3:])  # its last three frames, the recording's first, have no pose
        fill.join(forward, backward, inner / "mav0", root / "filled.tum", log=lambda m: None)
        filled_ns, filled = fill.read_tum(root / "filled.tum")
        np.testing.assert_array_equal(filled_ns, frame_times(inner))
        np.testing.assert_allclose(filled[:3], np.repeat(filled[3:4], 3, axis=0))

    def test_runs_that_do_not_overlap_cannot_be_joined(self):
        root = Path(tempfile.mkdtemp())
        inner, forward, backward = self.runs(root)
        ns, poses = fill.read_tum(backward)
        fill.write_tum(backward, ns[ns < T0 + 60_000_000_000], poses[ns < T0 + 60_000_000_000])
        with self.assertRaises(ValueError):
            fill.join(forward, backward, inner / "mav0", root / "filled.tum")


if __name__ == "__main__":
    unittest.main()
