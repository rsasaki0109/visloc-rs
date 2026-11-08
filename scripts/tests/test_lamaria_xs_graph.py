"""Unit tests for scripts/lamaria_xs_graph.py on synthetic recordings of one loop: the pose graph takes out one
recording's heading drift, the odometry weights loosen that recording, and an unchanged graph leaves a trajectory
as it is."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_xs_common as common  # noqa: E402
import lamaria_xs_graph as graph  # noqa: E402

LAT0, LON0 = 47.3769, 8.5417
TILT = R.from_euler("y", 80, degrees=True).as_matrix()  # the IMU's x axis mostly down, y horizontal, as on Aria


def loop(t, phase):
    """Positions and headings on a 40 m x 30 m loop walked at about 1.3 m/s, starting `phase` metres in."""
    s = (phase + 1.3 * t) % 140.0
    pts = np.array([[0, 0], [40, 0], [40, 30], [0, 30], [0, 0]], float)
    lens = np.array([40, 30, 40, 30], float)
    cum = np.r_[0, np.cumsum(lens)]
    seg = np.searchsorted(cum, s, side="right") - 1
    f = (s - cum[seg]) / lens[seg]
    xy = pts[seg] + f[:, None] * (pts[seg + 1] - pts[seg])
    d = pts[seg + 1] - pts[seg]
    return np.c_[xy, 0.1 * np.sin(t / 7)], np.arctan2(d[:, 1], d[:, 0])


def recording(work, name, phase, drift_deg, frame_deg, offset, seed, duration=220.0):
    """A recording's work directory: its estimate (own frame, heading drifting linearly by drift_deg), its 2 Hz
    keyframes and 1 Hz fixes of the truth (2 m noise). Returns the truth (positions, headings) at the keyframes."""
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 0.1)
    p, phi = loop(t, phase)
    drift = np.radians(drift_deg) * t / duration
    th = np.radians(frame_deg)
    dp = np.diff(p, axis=0)
    est = np.vstack([[0, 0, 0], np.cumsum(np.einsum("nij,nj->ni", common.rz(drift[:-1]), dp), axis=0)])
    est = est @ common.rz(th).T + offset
    rot = np.einsum("nij,jk->nik", common.rz(phi + drift + th), TILT)
    tns = (1e12 + t * 1e9).round().astype(np.int64)
    w = Path(work) / name
    (w / "ba").mkdir(parents=True)
    np.savetxt(w / "ba" / "ba3.tum", np.c_[tns / 1e9, est, R.from_matrix(rot).as_quat()], fmt="%.9f")
    np.savez(w / "ba" / "features.npz", kf_ns=tns[::5])
    tg = np.arange(0, len(t), 10)
    en = p[tg, :2] + rng.normal(0, 2.0, (len(tg), 2))
    lat = LAT0 + np.degrees(en[:, 1] / common.R_EARTH)
    lon = LON0 + np.degrees(en[:, 0] / (common.R_EARTH * np.cos(np.radians(LAT0))))
    with open(w / "gps.csv", "w") as f:
        f.write("t_ns,lat,lon,alt,accuracy,speed,provider\n")
        for k, i in enumerate(tg):
            f.write(f"{tns[i]},{lat[k]:.9f},{lon[k]:.9f},400.0,5.00,1.3,gps\n")
    yaw_true = common.yaw_of(np.einsum("nij,jk->nik", common.rz(phi), TILT), 1)
    return p[::5], yaw_true[::5]


def cross_edges(truth, pairs, rng, every=4):
    """Edges between recordings from the truth (relative pose in A's heading frame), 5 cm and 0.2 deg of noise,
    for keyframes of B within 3 m of a keyframe of A."""
    out = []
    for sA, sB in pairs:
        pa, ya = truth[sA]
        pb, yb = truth[sB]
        for kb in range(0, len(pb), every):
            d = np.linalg.norm(pa[:, :2] - pb[kb, :2], axis=1)
            ka = int(np.argmin(d))
            if d[ka] > 3.0:
                continue
            dp = common.rz(ya[ka]).T @ (pb[kb] - pa[ka]) + rng.normal(0, 0.05, 3)
            out.append((sA, ka, sB, kb, dp, common.wrap(yb[kb] - ya[ka] + np.radians(rng.normal(0, 0.2))), 200))
    return out


def shape_error(xy, truth_xy):
    """RMS of the horizontal positions after their best 2D rotation and translation onto the truth."""
    a, b = xy - xy.mean(0), truth_xy - truth_xy.mean(0)
    u, _, vt = np.linalg.svd(b.T @ a)
    r = u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return float(np.sqrt(((a @ r.T - b) ** 2).sum(1).mean()))


class GraphTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.work = cls.tmp.name
        cls.names = ["sequence_1_1", "sequence_1_2", "sequence_1_3"]
        cls.truth = {
            "sequence_1_1": recording(cls.work, "sequence_1_1", 0.0, 0.0, 20.0, [5.0, -3.0, 0.0], 1),
            "sequence_1_2": recording(cls.work, "sequence_1_2", 50.0, 0.0, -60.0, [-8.0, 2.0, 1.0], 2),
            "sequence_1_3": recording(cls.work, "sequence_1_3", 90.0, 15.0, 140.0, [1.0, 1.0, 0.0], 3),
        }
        rng = np.random.default_rng(7)
        pairs = [(a, b) for a in cls.names for b in cls.names if a != b]
        cls.edges = cross_edges(cls.truth, pairs, rng)
        np.save(Path(cls.work) / "edges.npy", np.array(cls.edges, dtype=object), allow_pickle=True)
        cls.recs = {s: graph.load_recording(cls.work, s) for s in cls.names}
        cls.axis, _ = common.heading_axis([d["rot"] for d in cls.recs.values()])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def solve(self, vce):
        g = graph.Graph(self.recs, self.edges, self.axis, 0.04, 0.005, 0.15, 0.25)
        x, _, _, scale = graph.solve(g, graph.initial_state(self.recs, self.axis), vce=vce, log=lambda m: None)
        return g, x, scale

    def test_heading_axis_is_the_horizontal_one(self):
        self.assertEqual(self.axis, 1)

    def errors(self, g, x):
        return [shape_error(x[g.base[s]:g.base[s] + len(self.recs[s]["kf_ns"]), :2], self.truth[s][0][:, :2])
                for s in self.names]

    def test_drift_is_taken_out(self):
        g, x, _ = self.solve(vce=1)
        before = shape_error(self.recs["sequence_1_3"]["p"][:, :2], self.truth["sequence_1_3"][0][:, :2])
        self.assertGreater(before, 1.0)
        self.assertLess(max(self.errors(g, x)), 0.1 * before)
        mt, my = g.cross_mismatch(x)
        self.assertLess(np.median(mt), 0.1)
        self.assertLess(np.median(my), 0.5)

    def test_the_drifting_recording_is_loosened(self):
        g0, x0, scale0 = self.solve(vce=0)
        g1, x1, scale = self.solve(vce=1)
        np.testing.assert_array_equal(scale0, 1)
        self.assertEqual(int(np.argmax(scale)), self.names.index("sequence_1_3"))
        self.assertGreater(scale[2], 2.0)
        # with one weight for all, the drift bends the recordings that hold too
        e0, e1 = self.errors(g0, x0), self.errors(g1, x1)
        self.assertGreater(min(e0[:2]), 0.2)
        self.assertLess(max(e1), 0.5 * min(e0))

    def test_gps_terms_pull_a_recording_into_place(self):
        s = "sequence_1_1"
        recs = {s: self.recs[s]}
        lat0, lon0 = common.gps_origin([recs[s]["gps"]])
        g = graph.Graph(recs, [], self.axis, 0.04, 0.005, 0.15, 0.25)
        g.add_gps(recs, lat0, lon0, 3.0)
        x0 = graph.initial_state(recs, self.axis)
        x0[:, :2] = x0[:, :2] @ common.rz(0.1)[:2, :2].T + [10.0, -5.0]  # 6 deg and 11 m off
        x0[:, 3] += 0.1
        x, *_ = graph.solve(g, x0, log=lambda m: None)
        shift = common.R_EARTH * np.array([np.cos(lat0) * (np.radians(LON0) - lon0), np.radians(LAT0) - lat0])
        err = x[:, :2] - (self.truth[s][0][:, :2] + shift)
        self.assertLess(np.sqrt((err ** 2).sum(1).mean()), 1.0)
        self.assertLess(np.degrees(np.abs(common.wrap(x[:, 3] - self.truth[s][1])).max()), 1.0)

    def test_an_unchanged_graph_keeps_the_trajectory(self):
        d = self.recs["sequence_1_1"]
        xs = np.c_[d["p"], common.yaw_of(d["rot"], self.axis)]
        out, dpsi = graph.corrected(d, xs, self.axis)
        np.testing.assert_allclose(dpsi, 0, atol=1e-12)
        np.testing.assert_allclose(out[:, :4], d["traj"][:, :4], atol=1e-9)
        same = np.abs(np.abs((out[:, 4:8] * d["traj"][:, 4:8]).sum(1)) - 1)
        self.assertLess(same.max(), 1e-9)

    def test_edges_are_filtered(self):
        path = Path(self.work) / "edges.npy"
        self.assertEqual(len(graph.load_edges([path, path])), 2 * len(self.edges))
        self.assertEqual(len(graph.load_edges([path], min_inliers=201)), 0)
        kept = graph.load_edges([path], exclude={"sequence_1_3"})
        self.assertTrue(kept and all("sequence_1_3" not in (e[0], e[2]) for e in kept))

    def test_a_recording_joins_held_ones(self):
        work = Path(self.work)
        cmd = [sys.executable, str(SCRIPTS / "lamaria_xs_graph.py"), self.work]
        edges = ["--edges", str(work / "edges.npy"), "--vce", "0"]  # the odometry weights depend on the others' bends
        subprocess.run(cmd + [str(work / "all"), *self.names] + edges, check=True, capture_output=True)
        (work / "hold").mkdir()
        for s in self.names[:2]:
            (work / "hold" / f"{s}.tum").write_bytes((work / "all" / f"{s}.tum").read_bytes())
        subprocess.run(cmd + [str(work / "joined"), *self.names] + edges + ["--hold", str(work / "hold")],
                       check=True, capture_output=True)
        self.assertFalse((work / "joined" / f"{self.names[0]}.tum").exists())
        a, b = np.loadtxt(work / "all" / f"{self.names[2]}.tum"), np.loadtxt(work / "joined" / f"{self.names[2]}.tum")
        self.assertLess(np.abs(a[:, 1:4] - b[:, 1:4]).max(), 0.1)

    def test_command_line(self):
        out = Path(self.work) / "out"
        subprocess.run([sys.executable, str(SCRIPTS / "lamaria_xs_graph.py"), self.work, str(out), *self.names,
                        "--edges", str(Path(self.work) / "edges.npy")], check=True, capture_output=True)
        for s in self.names:
            a, b = np.loadtxt(out / f"{s}.tum"), self.recs[s]["traj"]
            self.assertEqual(a.shape, b.shape)
            np.testing.assert_array_equal(a[:, 0], b[:, 0])
            self.assertTrue(np.isfinite(a).all())


if __name__ == "__main__":
    unittest.main()
