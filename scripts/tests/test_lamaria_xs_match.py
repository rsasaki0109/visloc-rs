"""Unit tests for the geometry of scripts/lamaria_xs_map.py and scripts/lamaria_xs_match.py: triangulation with
its checks, the PnP pose taken to the IMU, a cross edge that the pose graph reads back, and the query plan."""
import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import lamaria_xs_common as common  # noqa: E402
import lamaria_xs_graph as graph  # noqa: E402
import lamaria_xs_map as xmap  # noqa: E402
import lamaria_xs_match as match  # noqa: E402

K = np.array([[300.0, 0, 320], [0, 300.0, 240], [0, 0, 1]])
CAM = (R.from_euler("xyz", [-90, 0, -90], degrees=True).as_matrix(), np.array([0.05, -0.02, 0.01]), K)


def project(pose, X):
    r, t, _, k = pose
    pc = X @ r.T + t
    return (pc @ k.T)[:, :2] / pc[:, 2:3]


class MapTest(unittest.TestCase):
    def setUp(self):
        self.pa = xmap.cam_pose(np.eye(3), np.zeros(3), CAM)
        self.pb = xmap.cam_pose(np.eye(3), np.array([0.0, 0.5, 0.0]), CAM)  # 0.5 m to the side

    def test_points_are_recovered(self):
        X = np.array([[4.0, 0.3, 0.2], [8.0, -1.0, 0.5], [12.0, 2.0, -1.0]])
        P, ok = xmap.triangulate(self.pa, self.pb, project(self.pa, X), project(self.pb, X))
        self.assertTrue(ok.all())
        np.testing.assert_allclose(P, X, atol=1e-6)

    def test_bad_points_are_dropped(self):
        X = np.array([[4.0, 0.3, 0.2],  # kept
                      [100.0, 0.0, 0.0],  # parallax below a degree
                      [5.0, 0.0, 0.0]])  # its second observation 6 px off the epipolar line
        xb = project(self.pb, X)
        xb[2] += [0.0, 6.0]
        _, ok = xmap.triangulate(self.pa, self.pb, project(self.pa, X), xb)
        np.testing.assert_array_equal(ok, [True, False, False])


class MatchTest(unittest.TestCase):
    def test_pnp_pose_to_imu(self):
        import cv2
        r_wi = R.from_euler("z", 30, degrees=True).as_matrix()
        p_wi = np.array([2.0, -1.0, 0.3])
        r_cw, t_cw, _, _ = xmap.cam_pose(r_wi, p_wi, CAM)
        rr, pp = match.imu_from_pnp(cv2.Rodrigues(r_cw)[0], t_cw[:, None], CAM)
        np.testing.assert_allclose(rr, r_wi, atol=1e-9)
        np.testing.assert_allclose(pp, p_wi, atol=1e-9)

    def test_an_edge_reads_back_in_the_graph(self):
        tilt = R.from_euler("y", 80, degrees=True).as_matrix()
        ra, pa = common.rz(0.4) @ tilt, np.array([1.0, 2.0, 0.0])
        rb, pb = common.rz(1.1) @ tilt, np.array([3.0, 1.0, 0.2])
        e = match.edge("A", 0, "B", 0, ra, pa, rb, pb, 120, 1)
        g = graph.Graph.__new__(graph.Graph)
        g.I, g.J = np.array([0]), np.array([1])
        g.DP, g.DPSI = np.array([e[4]]), np.array([e[5]])
        g.ST, g.SY = np.array([0.1]), np.array([0.01])
        x = np.array([[*pa, common.yaw_of(ra, 1)], [*pb, common.yaw_of(rb, 1)]])
        rt, ry, *_ = g.residuals(x)
        np.testing.assert_allclose(rt, 0, atol=1e-12)
        np.testing.assert_allclose(ry, 0, atol=1e-12)

    def test_queries_go_to_the_nearest_maps(self):
        line = np.c_[np.arange(0, 50, 1.0), np.zeros(50)]
        offsets = (("A", 0), ("B", 3), ("C", 8), ("D", 30), ("E", -10))  # parallel walks, metres apart
        recs = {s: {"akf": np.arange(50), "en": line + [0, off]} for s, off in offsets}
        for d in recs.values():
            d["tree"] = cKDTree(d["en"])
        plan = match.plan_queries(recs, every=10, radius=12.0, max_maps=2)  # plan[map][query]
        self.assertEqual(sorted(plan["B"]), ["A", "C"])
        self.assertEqual(plan["B"]["A"], [0, 10, 20, 30, 40])
        self.assertEqual(sorted(plan["A"]), ["B", "C", "E"])  # not D, 30 m away
        self.assertEqual(plan["D"], {})
        later = match.plan_queries(recs, every=10, radius=12.0, max_maps=4, map_from=2)
        self.assertEqual({m: sorted(q) for m, q in later.items() if q}, {"E": ["A"]})  # A's third nearest map


if __name__ == "__main__":
    unittest.main()
