"""Unit tests for scripts/lamaria_batch_ba.py (numpy parts, the loop edge's PnP if opencv is there and the IMU
graph's warm start if gtsam is)."""
import importlib.util
import pathlib
import unittest

import numpy as np

try:
    import cv2  # noqa: F401
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False

try:
    import gtsam  # noqa: F401
    HAVE_GTSAM = True
except ImportError:
    HAVE_GTSAM = False

try:
    import torch  # noqa: F401
    from lightglue import LightGlue
    LightGlue(features="aliked")  # the weights must be there (downloaded once)
    HAVE_LIGHTGLUE = True
except Exception:  # noqa: BLE001 - no torch, no LightGlue, or no network for the weights
    HAVE_LIGHTGLUE = False

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "lamaria_batch_ba.py"
spec = importlib.util.spec_from_file_location("lamaria_batch_ba", SCRIPT)
lb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lb)


def rot(rng, scale=1.0):
    return lb.so3_exp(rng.normal(size=3) * scale)


def two_cameras():
    k0 = np.array([[240.0, 0, 390], [0, 240.0, 290], [0, 0, 1]])
    k1 = np.array([[242.0, 0, 385], [0, 242.0, 296], [0, 0, 1]])
    return [{"R_ic": lb.so3_exp(np.array([0.2, -1.3, 0.1])), "t_ic": np.array([0.01, -0.1, 0.07]), "K": k0},
            {"R_ic": lb.so3_exp(np.array([-0.1, 0.4, 0.05])), "t_ic": np.array([0.0, 0.0, -0.006]), "K": k1}]


class ReprojectionJacobianTest(unittest.TestCase):
    def test_jacobians_match_finite_differences(self):
        rng = np.random.default_rng(3)
        cams = two_cameras()
        n_kf, n_lm = 3, 4
        r_wi = np.array([rot(rng, 0.3) for _ in range(n_kf)])
        t_wi = rng.normal(size=(n_kf, 3))
        r_ic = np.array([c["R_ic"] for c in cams])
        t_ic = np.array([c["t_ic"] for c in cams])
        # landmarks in front of every camera: place them along each camera's axis
        obs_kf, obs_cam, obs_lm = [], [], []
        pts = np.zeros((n_lm, 3))
        for j in range(n_lm):
            k, c = j % n_kf, j % 2
            r_wc = r_wi[k] @ r_ic[c]
            center = r_wi[k] @ t_ic[c] + t_wi[k]
            pts[j] = center + r_wc @ np.array([rng.normal() * 0.3, rng.normal() * 0.3, 4.0 + j])
            obs_kf.append(k)
            obs_cam.append(c)
            obs_lm.append(j)
        obs_z = np.zeros((n_lm, 2))
        rp = lb.Reprojection(np.array(obs_kf), np.array(obs_cam), np.array(obs_lm), obs_z, n_lm, cams,
                             [0, 1], 1.0, 1e9, 64, est_focal=True)
        rp.set_focal([0.01, -0.02])
        res, jx, jl, _ = rp.linearize(r_wi, t_wi, r_ic, t_ic, pts)
        eps = 1e-6

        def residuals(r_wi_, t_wi_, r_ic_, pts_):
            return rp._project(r_wi_, t_wi_, r_ic_, t_ic, pts_)[-1]

        for o in range(len(rp.kf)):
            k, c, j = rp.kf[o], rp.cam[o], rp.lm[o]
            for d in range(9):
                r2, t2, ric2 = r_wi.copy(), t_wi.copy(), r_ic.copy()
                step = np.zeros(3)
                step[d % 3] = eps
                if d < 3:      # pose rotation, right perturbation
                    r2[k] = r_wi[k] @ lb.so3_exp(step)
                elif d < 6:    # pose translation in the body frame
                    t2[k] = t_wi[k] + r_wi[k] @ step
                else:          # extrinsic rotation, right perturbation
                    ric2[c] = r_ic[c] @ lb.so3_exp(step)
                num = (residuals(r2, t2, ric2, pts)[o] - res[o]) / eps
                np.testing.assert_allclose(jx[o, :, d], num, rtol=1e-4, atol=1e-3)
            foc = rp.foc.copy()
            rp.set_focal(foc + eps * (np.arange(2) == c))  # focal length scale: f <- f exp(s)
            num = (residuals(r_wi, t_wi, r_ic, pts)[o] - res[o]) / eps
            rp.set_focal(foc)
            np.testing.assert_allclose(jx[o, :, 9], num, rtol=1e-4, atol=1e-3)
            for d in range(3):
                p2 = pts.copy()
                p2[j, d] += eps
                num = (residuals(r_wi, t_wi, r_ic, p2)[o] - res[o]) / eps
                np.testing.assert_allclose(jl[o, :, d], num, rtol=1e-4, atol=1e-3)


class NormalEquationsTest(unittest.TestCase):
    def test_one_pass_loop_matches_linearize(self):
        for focal in (False, True):  # with the focal length scale as a tenth camera parameter
            with self.subTest(focal=focal):
                self._check(focal)

    def _check(self, focal):
        rng = np.random.default_rng(9)
        cams = two_cameras()
        n_kf, n_lm = 4, 6
        r_wi = np.array([rot(rng, 0.3) for _ in range(n_kf)])
        t_wi = rng.normal(size=(n_kf, 3))
        r_ic = np.array([c["R_ic"] for c in cams])
        t_ic = np.array([c["t_ic"] for c in cams])
        obs_kf, obs_cam, obs_lm, pts = [], [], [], np.zeros((n_lm, 3))
        for j in range(n_lm):
            k, c = j % n_kf, j % 2
            pts[j] = (r_wi[k] @ t_ic[c] + t_wi[k]) + r_wi[k] @ r_ic[c] @ np.array([rng.normal() * 0.3, 0.2, 5.0 + j])
            for kk in range(n_kf):  # every keyframe sees every landmark with the same camera
                obs_kf.append(kk)
                obs_cam.append(c)
                obs_lm.append(j)
        obs_z = rng.uniform(100, 600, size=(len(obs_kf), 2))  # far off: some Huber-weighted, some behind
        sigma = rng.uniform(0.5, 2.0, size=len(obs_kf))  # per-observation pixel noise
        rp = lb.Reprojection(np.array(obs_kf), np.array(obs_cam), np.array(obs_lm), obs_z, n_lm, cams, [1], sigma,
                             2.0, 64, numpy_pairs=True, est_focal=focal)
        if focal:
            rp.set_focal([0.0, 0.01])
        rp.active[3] = False
        res, jx, jl, w = rp.linearize(r_wi, t_wi, r_ic, t_ic, pts)
        wjx, wjl = jx * w[:, None, None], jl * w[:, None, None]
        want = [np.einsum("nki,nkj->nij", wjx, jx), np.einsum("nki,nkj->nij", wjx, jl),
                np.einsum("nki,nk->ni", wjx, res), np.einsum("nki,nkj->nij", wjl, jl)]
        h_ll, g_l = np.zeros((n_lm, 3, 3)), np.zeros((n_lm, 3))
        np.add.at(h_ll, rp.lm, want[3])
        np.add.at(g_l, rp.lm, np.einsum("nki,nk->ni", wjl, res))
        want += [h_ll, g_l]
        n_o = len(rp.kf)
        nx = 6 + rp.n_par
        got = [np.zeros((n_o, nx, nx)), np.zeros((n_o, nx, 3)), np.zeros((n_o, nx)), np.zeros((n_o, 3, 3)),
               np.zeros((n_lm, 3, 3)), np.zeros((n_lm, 3))]
        lb._normal_eqs(r_wi, t_wi, r_ic, t_ic, pts, rp.kf, rp.cam, rp.lm, rp.z, rp.kcam, rp.est, rp.active,
                       rp.sigma, rp.huber, *got)
        self.assertTrue(0 < (w > 0).sum() < n_o)
        for g, wv in zip(got, want):
            np.testing.assert_allclose(g, wv, rtol=1e-12, atol=1e-9)
        h_ll_inv = np.linalg.inv(h_ll + np.eye(3)[None])
        m_o = np.zeros_like(got[1])
        lb._landmark_m(got[1], h_ll_inv, rp.lm, m_o)
        np.testing.assert_allclose(m_o, np.einsum("nij,njk->nik", want[1], h_ll_inv[rp.lm]), rtol=1e-12, atol=1e-12)
        # the one-pass cost (numba) against the vectorized projection; keyframe 0 turned away from its
        # landmarks puts some observations behind the camera
        r_wi[0] = r_wi[0] @ np.diag([-1.0, 1.0, -1.0])
        *_, valid, res = rp._project(r_wi, t_wi, r_ic, t_ic, pts)
        e_want, _, cost_want = rp._robust(res, valid)
        e = np.zeros(n_o)
        cost = lb._reproj_cost(r_wi, t_wi, r_ic, t_ic, pts, rp.kf, rp.cam, rp.lm, rp.z, rp.kcam, rp.active, rp.sigma,
                               rp.huber, e)
        self.assertTrue((~valid).any())
        np.testing.assert_allclose(e, e_want, rtol=1e-12)
        self.assertAlmostEqual(cost, cost_want.sum(), delta=1e-9 * cost_want.sum())
        if lb._normal_eqs_par is not None:  # the parallel kernels (numba), chunk by chunk
            par = [np.zeros_like(g) for g in got]
            lb._normal_eqs_par(rp.chunks, r_wi, t_wi, r_ic, t_ic, pts, rp.kf, rp.cam, rp.lm, rp.z, rp.kcam, rp.est,
                               rp.active, rp.sigma, rp.huber, *par)
            ser = [np.zeros_like(g) for g in got]
            lb._normal_eqs(r_wi, t_wi, r_ic, t_ic, pts, rp.kf, rp.cam, rp.lm, rp.z, rp.kcam, rp.est, rp.active,
                           rp.sigma, rp.huber, *ser)
            for g, wv in zip(par, ser):
                np.testing.assert_array_equal(g, wv)
            e_par = np.zeros(n_o)
            self.assertAlmostEqual(lb._reproj_cost_par(rp.chunks, r_wi, t_wi, r_ic, t_ic, pts, rp.kf, rp.cam, rp.lm,
                                                       rp.z, rp.kcam, rp.active, rp.sigma, rp.huber, e_par),
                                   cost, delta=1e-12 * cost)
            np.testing.assert_array_equal(e_par, e)


class OutlierMaskTest(unittest.TestCase):
    def test_inactive_observations_carry_no_weight_or_cost(self):
        cams = two_cameras()
        r_wi, t_wi = np.eye(3)[None].repeat(2, 0), np.array([[0.0, 0, 0], [0.5, 0, 0]])
        r_ic = np.array([c["R_ic"] for c in cams])
        t_ic = np.array([c["t_ic"] for c in cams])
        pts = np.array([r_ic[0] @ np.array([0.1, 0.2, 5.0]) + t_ic[0]])
        z = np.array([[0.0, 0.0], [0.0, 0.0]])  # far off: both observations are Huber-weighted outliers
        rp = lb.Reprojection(np.array([0, 1]), np.array([0, 0]), np.array([0, 0]), z, 1, cams, [0], 1.0, 2.0, 64)
        cost_all, _ = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)
        rp.active[1] = False
        cost_one, err = rp.cost(r_wi, t_wi, r_ic, t_ic, pts)
        _, _, _, w = rp.linearize(r_wi, t_wi, r_ic, t_ic, pts)
        self.assertGreater(cost_all, cost_one)
        self.assertGreater(cost_one, 0.0)
        self.assertEqual(w[1], 0.0)
        self.assertGreater(w[0], 0.0)
        self.assertGreater(err[1], 2.0)  # the error is still reported for an inactive observation


class SegmentPriorTest(unittest.TestCase):
    def test_normal_equations_match_finite_differences(self):
        rng = np.random.default_rng(4)
        # cam1 in three time segments (virtual cameras 3, 4, 5) estimated, cam0's segments (0, 1, 2) not
        base = [rot(rng), rot(rng)]
        r_ic = np.array([base[c] @ lb.so3_exp(rng.normal(size=3) * 0.003) for c in range(2) for _ in range(3)])
        slot = np.array([-1, -1, -1, 0, 1, 2])
        pairs, w = [(3, 4), (4, 5)], 1.0 / np.radians(0.05) ** 2

        def cost(phi):
            ric = r_ic.copy()
            for c in (3, 4, 5):
                ric[c] = r_ic[c] @ lb.so3_exp(phi[slot[c] * 3:slot[c] * 3 + 3])
            return lb.segment_prior(ric, pairs, slot, 3, w)[0]

        def grad(phi, eps=1e-6):
            return np.array([(cost(phi + eps * e) - cost(phi - eps * e)) / (2 * eps) for e in np.eye(9)])

        c0, h, b = lb.segment_prior(r_ic, pairs, slot, 3, w)
        self.assertAlmostEqual(c0, cost(np.zeros(9)))
        np.testing.assert_allclose(b, -grad(np.zeros(9)), rtol=1e-5, atol=1e-3)
        h_num = np.array([(grad(1e-5 * e) - grad(-1e-5 * e)) / 2e-5 for e in np.eye(9)])
        np.testing.assert_allclose(h, h_num, atol=0.01 * w)  # Gauss-Newton: exact up to O(|e0|)


class FocalPriorTest(unittest.TestCase):
    def test_normal_equations_match_finite_differences(self):
        rng = np.random.default_rng(5)
        # both cameras in three segments, 4 parameters per slot (rotation, focal length scale last)
        slot = np.arange(6)
        pairs = [(0, 1), (1, 2), (3, 4), (4, 5)]
        groups = [[0, 1, 2], [3, 4, 5]]
        foc0 = rng.normal(size=6) * 0.003
        w = (1 / 0.0005 ** 2, 1 / 0.01 ** 2, 1 / 0.001 ** 2, groups)

        def cost(x):
            return lb.focal_prior(foc0 + x[3::4], pairs, slot, 4, 6, *w)[0]

        c0, h, b = lb.focal_prior(foc0, pairs, slot, 4, 6, *w)
        self.assertAlmostEqual(c0, cost(np.zeros(24)))
        eps = 1e-7
        grad = np.array([(cost(eps * e) - cost(-eps * e)) / (2 * eps) for e in np.eye(24)])
        np.testing.assert_allclose(b, -grad, rtol=1e-6, atol=1e-6)
        h_num = np.array([[(cost(eps * (ei + ej)) - cost(eps * (ei - ej)) - cost(eps * (ej - ei))
                            + cost(-eps * (ei + ej))) / (4 * eps ** 2) for ej in np.eye(24)] for ei in np.eye(24)])
        np.testing.assert_allclose(h, h_num, rtol=1e-4, atol=1e-6 * np.abs(h).max())  # quadratic: exact
        self.assertTrue(np.all(h[np.ix_(np.arange(24) % 4 != 3, np.arange(24))] == 0))  # rotations untouched


class BandForwardTest(unittest.TestCase):
    def test_matches_a_dense_triangular_solve(self):
        rng = np.random.default_rng(8)
        n, kd = 60, 7
        dense = np.tril(rng.normal(size=(n, n)) * 0.3)
        dense[np.tril_indices(n, -kd - 1)] = 0.0
        dense[np.diag_indices(n)] = 2.0 + rng.uniform(size=n)
        cfac = np.zeros((kd + 1, n))  # LAPACK lower band storage: cfac[i - j, j] = L[i, j]
        for d in range(kd + 1):
            cfac[d, :n - d] = np.diagonal(dense, -d)
        starts = np.array([0, 12, 12, 30, 59, 60])  # sorted; the last column is all zero
        b = rng.normal(size=(n, len(starts)))
        for c, r0 in enumerate(starts):
            b[:r0, c] = 0.0
        z = b.copy()
        lb._band_forward(cfac, z, starts)
        np.testing.assert_allclose(z, np.linalg.solve(dense, b), rtol=1e-12, atol=1e-12)


class TriangulateTest(unittest.TestCase):
    def test_drops_bad_observations_and_narrow_tracks(self):
        rng = np.random.default_rng(8)
        cams = two_cameras()
        n_kf = 6
        kf_p = np.c_[np.arange(n_kf) * 0.5, np.zeros(n_kf), np.zeros(n_kf)]  # 0.5 m steps along x
        kf_r = np.array([rot(rng, 0.05) for _ in range(n_kf)])
        pts = [kf_r[0] @ cams[0]["R_ic"] @ np.array([0.3, -0.2, 4.0]) + kf_p[0],
               kf_r[0] @ cams[0]["R_ic"] @ np.array([-0.4, 0.1, 400.0]) + kf_p[0]]  # far away: too narrow
        xy, image_of, tracks = [], [], []
        for j, pw in enumerate(pts):
            feats = []
            for k in range(n_kf):
                r_wc = kf_r[k] @ cams[0]["R_ic"]
                pc = r_wc.T @ (pw - (kf_r[k] @ cams[0]["t_ic"] + kf_p[k]))
                uv = (cams[0]["K"] @ pc)[:2] / pc[2]
                if j == 0 and k == 3:
                    uv = uv + 8.0  # one wrong match
                feats.append(len(xy))
                xy.append(uv)
                image_of.append(2 * k)
            tracks.append(np.array(feats))
        x, kept = lb.triangulate(tracks, np.array(image_of), np.array(xy), kf_p, kf_r, cams, 4.0, 1.0)
        # the wrong match pulls the first round's point, which drops it and the first observation; the
        # second round keeps the other four; the far point's rays stay within 1 degree
        self.assertEqual(len(kept), 1)
        np.testing.assert_array_equal(kept[0], np.array([1, 2, 4, 5]))
        np.testing.assert_allclose(x[0], pts[0], atol=1e-6)


class SchurAssemblyTest(unittest.TestCase):
    def test_landmark_loop_matches_the_pair_lists(self):
        rng = np.random.default_rng(6)
        cams = two_cameras()
        # 7 keyframes, landmarks seen 2-4 times; band 3, so the longest landmarks also have off-band pairs
        obs_kf, obs_cam, obs_lm = [], [], []
        for j in range(9):
            for k in sorted(rng.choice(7, size=rng.integers(2, 5), replace=False)):
                obs_kf.append(k)
                obs_cam.append(int(rng.integers(2)))
                obs_lm.append(j)
        obs_kf, obs_cam, obs_lm = np.array(obs_kf), np.array(obs_cam), np.array(obs_lm)
        for est, focal in (([0, 1], False), ([1], False), ([0, 1], True)):
            rp = lb.Reprojection(obs_kf, obs_cam, obs_lm, np.zeros((len(obs_kf), 2)), 9, cams, est, 1.0, 2.0, 3,
                                 numpy_pairs=True, est_focal=focal)
            n_o, n, nb, ne, nx = len(rp.kf), 7, 7 * lb.DOF, rp.n_par * rp.n_ext, 6 + rp.n_par
            m_o, h_xl = rng.normal(size=(n_o, nx, 3)), rng.normal(size=(n_o, nx, 3))
            h_xx = rng.normal(size=(n_o, nx, nx))
            h_xx = h_xx + h_xx.transpose(0, 2, 1)
            g_x, g_l = rng.normal(size=(n_o, nx)), rng.normal(size=(9, 3))
            slot = rp.ext_slot[rp.cam]
            u = (rp.band_kf + 1) * lb.DOF - 1
            out = []
            for numpy_path in (True, False):
                ab, border = np.zeros((u + 1, nb)), np.zeros((nb, ne))
                corner, rhs = np.zeros((ne, ne)), np.zeros(nb + ne)
                if numpy_path:
                    direct = np.add.reduceat(h_xx[rp.obs_sort][:, 0:6, 0:6], rp.obs_starts, axis=0)
                    lb._schur_add_numpy(rp, m_o, h_xl, h_xx, g_x, g_l, direct, slot, slot >= 0, n, nb, ne, ab, border,
                                        corner, rhs)
                else:  # the numba kernel's plain-Python body
                    lb._schur_add(m_o, h_xl, h_xx, g_x, g_l, rp.kf, slot, rp.lm, rp.lm_start, rp.band, nb, ab, border,
                                  corner, rhs)
                out.append((ab, border, corner, rhs))
            for x_np, x_loop in zip(*out):
                np.testing.assert_allclose(x_loop, x_np, atol=1e-12)
            self.assertGreater(len(rp.off_keys), 0)


class ModeVectorTest(unittest.TestCase):
    def test_modes_match_small_drifts(self):
        rng = np.random.default_rng(2)
        n = 20
        kf_s = np.arange(n) * 0.5
        r_wi = np.array([rot(rng, 0.5) for _ in range(n)])
        t_wi = np.cumsum(rng.normal(size=(n, 3)), axis=0)
        vel = rng.normal(size=(n, 3))
        pts = rng.normal(size=(5, 3)) * 5
        lm_first = np.array([0, 3, 7, 12, 19])
        vx, vl = lb.mode_vectors(["yaw", "scale"], kf_s, r_wi, t_wi, vel, pts, lm_first)
        s = (kf_s - kf_s[0]) / (kf_s[-1] - kf_s[0])
        s_mid = 0.5 * (s[1:] + s[:-1])
        k_ = 1e-6

        def rz(angle):
            return lb.so3_exp(np.array([0.0, 0.0, angle]))

        # yaw: the heading turns by k_ s, positions follow the turned displacements
        dp = [rz(k_ * s_mid[j]) @ (t_wi[j + 1] - t_wi[j]) for j in range(n - 1)]
        drifted = {
            "yaw": (np.array([rz(k_ * s[k]) @ r_wi[k] for k in range(n)]),
                    np.vstack([t_wi[:1], t_wi[:1] + np.cumsum(dp, axis=0)]),
                    np.array([rz(k_ * s[k]) @ vel[k] for k in range(n)])),
            "scale": (r_wi,
                      np.vstack([t_wi[:1], t_wi[:1] + np.cumsum(np.diff(t_wi, axis=0) * (1 + k_ * s_mid[:, None]),
                                                                  axis=0)]),
                      vel * (1 + k_ * s[:, None])),
        }
        for m, mode in enumerate(("yaw", "scale")):
            r2, t2, v2 = drifted[mode]
            for k in range(n):
                w = np.radians(lb.so3_log_deg(r_wi[k].T @ r2[k])) / k_
                np.testing.assert_allclose(w, vx[m, k, 0:3], atol=1e-4)
                np.testing.assert_allclose(r_wi[k].T @ (t2[k] - t_wi[k]) / k_, vx[m, k, 3:6], atol=1e-4)
                np.testing.assert_allclose((v2[k] - vel[k]) / k_, vx[m, k, 6:9], atol=1e-4)
            turn = (lambda a: rz(k_ * a)) if mode == "yaw" else (lambda a: np.eye(3) * (1 + k_ * a))
            for j, k in enumerate(lm_first):
                moved = t2[k] + turn(s[k]) @ (pts[j] - t_wi[k])
                np.testing.assert_allclose((moved - pts[j]) / k_, vl[m, j], atol=1e-4)
        # deform applies the same drifts exactly: to first order it moves every state along the modes
        bias = rng.normal(size=(n, 6)) * 1e-3
        for m, mode in enumerate(("yaw", "scale")):
            r2, t2, v2, b2, p2 = lb.deform(mode, k_, kf_s, r_wi, t_wi, vel, bias, pts, lm_first)
            for k in range(n):
                w = np.radians(lb.so3_log_deg(r_wi[k].T @ r2[k])) / k_
                np.testing.assert_allclose(w, vx[m, k, 0:3], atol=1e-4)
                np.testing.assert_allclose(r_wi[k].T @ (t2[k] - t_wi[k]) / k_, vx[m, k, 3:6], atol=1e-4)
            np.testing.assert_allclose((v2 - vel) / k_, vx[m, :, 6:9], atol=1e-4)
            np.testing.assert_allclose((b2 - bias) / k_, vx[m, :, 9:15], atol=1e-6)
            np.testing.assert_allclose((p2 - pts) / k_, vl[m], atol=1e-4)
        # the yaw mode shifts the gyro bias by one offset: the mean body vertical per unit heading rate
        up_body = np.einsum("nji,j->ni", r_wi, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(vx[0, :, 12:15], -up_body.mean(0)[None].repeat(n, 0) / (kf_s[-1] - kf_s[0]))
        np.testing.assert_array_equal(vx[1, :, 9:15], 0.0)

    def test_yaw_pieces_add_up_to_the_yaw_mode(self):
        rng = np.random.default_rng(3)
        n = 21
        kf_s = np.arange(n) * 0.5
        r_wi = np.array([rot(rng, 0.5) for _ in range(n)])
        t_wi = np.cumsum(rng.normal(size=(n, 3)), axis=0)
        vel = rng.normal(size=(n, 3))
        pts = rng.normal(size=(4, 3)) * 5
        lm_first = np.array([0, 6, 13, 20])
        pieces = lb.expand_modes("yaw4,scale")
        self.assertEqual(pieces, ["yaw:0:4", "yaw:1:4", "yaw:2:4", "yaw:3:4", "scale"])
        vx, vl = lb.mode_vectors(pieces[:4] + ["yaw"], kf_s, r_wi, t_wi, vel, pts, lm_first)
        # the four pieces together, each turning by one, are the linear heading drift by four
        np.testing.assert_allclose(vx[:4].sum(0), 4 * vx[4], atol=1e-9)
        np.testing.assert_allclose(vl[:4].sum(0), 4 * vl[4], atol=1e-9)
        # a piece turns the heading within its quarter only: nothing before, all of it after
        s1 = lb._mode_profile("yaw:1:4", kf_s)
        np.testing.assert_allclose(s1[:6], 0.0)
        np.testing.assert_allclose(s1[10:], 1.0)
        # and deform follows the piece's tangent to first order
        k_ = 1e-6
        bias = rng.normal(size=(n, 6)) * 1e-3
        r2, t2, _, b2, p2 = lb.deform("yaw:1:4", k_, kf_s, r_wi, t_wi, vel, bias, pts, lm_first)
        for k in range(n):
            w = np.radians(lb.so3_log_deg(r_wi[k].T @ r2[k])) / k_
            np.testing.assert_allclose(w, vx[1, k, 0:3], atol=1e-4)
            np.testing.assert_allclose(r_wi[k].T @ (t2[k] - t_wi[k]) / k_, vx[1, k, 3:6], atol=1e-4)
        np.testing.assert_allclose((b2 - bias) / k_, vx[1, :, 9:15], atol=1e-6)
        np.testing.assert_allclose((p2 - pts) / k_, vl[1], atol=1e-4)


class MatchingTest(unittest.TestCase):
    def test_best_two(self):
        keys = np.array([0, 0, 1, 0, 1])
        vals = np.array([5.0, 2.0, 7.0, 3.0, 1.0], np.float32)
        best, second = lb._best_two(keys, vals, 3)
        self.assertEqual(best.tolist(), [1, 4, -1])
        self.assertEqual(second[:2].tolist(), [3.0, 7.0])
        self.assertTrue(np.isinf(second[2]))

    def test_epipolar_matcher_recovers_true_correspondences(self):
        rng = np.random.default_rng(7)
        cams = two_cameras()
        k = cams[0]["K"]
        # two keyframes 0.8 m apart; points 3-20 m in front of camera 0 of keyframe 0
        kf_r = np.array([np.eye(3), lb.so3_exp(np.array([0.0, 0.05, 0.0]))])
        kf_p = np.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]])
        lb._G.update(cams=[cams[0], cams[0]], kf_p=kf_p, kf_r=kf_r, ratio=0.8, max_desc=300.0)
        r_wc0, c0 = lb._cam_pose(0, 0)
        r_wc1, c1 = lb._cam_pose(1, 0)
        n = 200
        xy0 = rng.uniform([50, 50], [730, 530], size=(n, 2))
        depth = rng.uniform(3, 20, size=n)
        rays = np.c_[xy0, np.ones(n)] @ np.linalg.inv(k).T
        world = c0 + (rays * depth[:, None]) @ r_wc0.T
        cam1 = (world - c1) @ r_wc1
        xy1 = (cam1 @ k.T)[:, :2] / cam1[:, 2:3]
        desc0 = rng.integers(0, 256, size=(n, 128), dtype=np.uint8)
        noise = rng.integers(-3, 4, size=(n, 128))
        desc1 = np.clip(desc0.astype(int) + noise, 0, 255).astype(np.uint8)
        perm = rng.permutation(n)  # shuffle image 1 so a match is not just the identity
        xy = np.concatenate([xy0, xy1[perm]]).astype(np.float32)
        desc = np.concatenate([desc0, desc1[perm]])
        lb._G.update(offsets=np.array([0, n, n, 2 * n, 2 * n]), xy=xy, desc=desc)
        saved = lb.njit
        try:
            lb.njit = None  # numpy path
            _, m = lb._match_worker((0, 0, 1, 0, 2.0))
        finally:
            lb.njit = saved
        self.assertGreater(len(m), 0.9 * n)
        inverse = np.argsort(perm)
        np.testing.assert_array_equal(m[:, 1], inverse[m[:, 0]])

    def test_unguided_matches_numpy_path(self):
        rng = np.random.default_rng(11)
        desc_a = rng.integers(0, 256, size=(80, 128), dtype=np.uint8)
        perm = rng.permutation(80)
        desc_b = desc_a[perm]
        methods = ["numpy"] + (["cv2"] if HAVE_CV2 else [])
        for method in methods:
            m = lb._unguided_matches(desc_a, desc_b, 0.8, 300.0, method=method)
            self.assertEqual(len(m), 80)
            np.testing.assert_array_equal(perm[m[:, 1]], m[:, 0])

    def test_ratio_matcher_takes_learned_descriptors(self):
        rng = np.random.default_rng(5)
        cams = two_cameras()
        k = cams[0]["K"]
        kf_r = np.array([np.eye(3), lb.so3_exp(np.array([0.0, 0.05, 0.0]))])
        kf_p = np.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]])
        lb._G.update(cams=[cams[0], cams[0]], kf_p=kf_p, kf_r=kf_r, ratio=0.8, max_desc=np.inf)
        r_wc0, c0 = lb._cam_pose(0, 0)
        r_wc1, c1 = lb._cam_pose(1, 0)
        n = 150
        xy0 = rng.uniform([50, 50], [730, 530], size=(n, 2))
        rays = np.c_[xy0, np.ones(n)] @ np.linalg.inv(k).T
        world = c0 + (rays * rng.uniform(3, 20, size=n)[:, None]) @ r_wc0.T
        cam1 = (world - c1) @ r_wc1
        xy1 = (cam1 @ k.T)[:, :2] / cam1[:, 2:3]
        desc0 = rng.normal(size=(n, 256))  # unit-length, SuperPoint's length, stored in half precision
        desc0 /= np.linalg.norm(desc0, axis=1, keepdims=True)
        desc1 = desc0 + 0.02 * rng.normal(size=(n, 256))
        desc1 /= np.linalg.norm(desc1, axis=1, keepdims=True)
        perm = rng.permutation(n)
        lb._G.update(offsets=np.array([0, n, n, 2 * n, 2 * n]), xy=np.concatenate([xy0, xy1[perm]]).astype(np.float32),
                     desc=np.concatenate([desc0, desc1[perm]]).astype(np.float16))
        _, m = lb._match_worker((0, 0, 1, 0, 2.0))  # with or without numba
        self.assertGreater(len(m), 0.9 * n)
        np.testing.assert_array_equal(m[:, 1], np.argsort(perm)[m[:, 0]])

    def test_band_of_learned_matches_is_the_ratio_matchers_band(self):
        rng = np.random.default_rng(9)
        cams = two_cameras()
        kf_r = np.array([np.eye(3), lb.so3_exp(np.array([0.0, 0.05, 0.0]))])
        kf_p = np.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]])
        lb._G.update(cams=[cams[0], cams[0]], kf_p=kf_p, kf_r=kf_r)
        k = cams[0]["K"]
        r_wc0, c0 = lb._cam_pose(0, 0)
        r_wc1, c1 = lb._cam_pose(1, 0)
        n = 100
        xy0 = rng.uniform([50, 50], [730, 530], size=(n, 2))
        rays = np.c_[xy0, np.ones(n)] @ np.linalg.inv(k).T
        world = c0 + (rays * rng.uniform(3, 20, size=n)[:, None]) @ r_wc0.T
        cam1 = (world - c1) @ r_wc1
        xy1 = (cam1 @ k.T)[:, :2] / cam1[:, 2:3]
        self.assertLess(lb._band_dist(0, 0, 1, 0, xy0, xy1).max() * k[0, 0], 1e-6)  # true matches: on the line
        # shifted across the epipolar line by 5 px: 5 px off
        r_ba = r_wc1.T @ r_wc0
        line = np.c_[rays] @ (lb.skew(r_wc1.T @ (c0 - c1)) @ r_ba).T
        normal = line[:, :2] / np.hypot(line[:, 0], line[:, 1])[:, None]
        d = lb._band_dist(0, 0, 1, 0, xy0, xy1 + 5 * normal)
        np.testing.assert_allclose(d * k[0, 0], 5.0, rtol=0.02)

    def test_learned_features_default_to_lightglue_without_a_descriptor_limit(self):
        self.assertEqual(lb._matcher("auto", "sift"), "ratio")
        self.assertEqual(lb._matcher("auto", "aliked"), "lightglue")
        self.assertEqual(lb._matcher("ratio", "aliked"), "ratio")
        self.assertEqual(lb._max_desc(None, "sift"), 300.0)
        self.assertTrue(np.isinf(lb._max_desc(None, "disk")))
        self.assertEqual(lb._max_desc(0.9, "disk"), 0.9)
        with self.assertRaises(SystemExit):
            lb._matcher("lightglue", "sift")

    @unittest.skipUnless(HAVE_LIGHTGLUE, "needs torch and LightGlue with its ALIKED weights")
    def test_batched_lightglue_matches_lightglue_pair_by_pair(self):
        import torch
        from lightglue import LightGlue
        torch.manual_seed(0)
        lg = LightGlue(features="aliked", depth_confidence=-1, width_confidence=-1).eval()

        def image(n):
            xy = torch.rand(1, n, 2) * torch.tensor([757.0, 571.0])
            return {"keypoints": xy, "descriptors": torch.nn.functional.normalize(torch.randn(1, n, 128), dim=-1),
                    "image_size": torch.tensor([[758.0, 572.0]])}

        base = [image(96) for _ in range(3)]
        # the second image of a pair: the first one's keypoints moved a little, half of them and new ones
        pairs = []
        for f, n_new in zip(base, (0, 30, 50)):
            keep = torch.arange(96 - n_new)
            g = image(96)
            g["keypoints"][0, keep] = f["keypoints"][0, keep] + torch.randn(len(keep), 2)
            g["descriptors"][0, keep] = torch.nn.functional.normalize(
                f["descriptors"][0, keep] + 0.05 * torch.randn(len(keep), 128), dim=-1)
            pairs.append((f, g))
        with torch.inference_mode():
            ref = [lg({"image0": f, "image1": g})["matches"][0].numpy() for f, g in pairs]
        self.assertGreater(min(len(r) for r in ref), 30)
        same = lb._lightglue_batch(lg, [f for f, _ in pairs], [g for _, g in pairs], False)  # 96 keypoints everywhere
        for r, m in zip(ref, same):
            np.testing.assert_array_equal(m, r)
        trimmed = [(f, {k: v[:, :80] if k != "image_size" else v for k, v in g.items()}) for f, g in pairs]
        trimmed[1] = (trimmed[1][0], pairs[1][1])  # 80, 96 and 80 keypoints in b: padded and masked
        with torch.inference_mode():
            ref = [lg({"image0": f, "image1": g})["matches"][0].numpy() for f, g in trimmed]
        padded = lb._lightglue_batch(lg, [f for f, _ in trimmed], [g for _, g in trimmed], False)
        for r, m in zip(ref, padded):
            np.testing.assert_array_equal(m, r)

    def test_features_written_before_the_extractor_was_recorded_are_sift(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            np.savez(pathlib.Path(d) / "features.npz", kf_ns=np.zeros(1, np.int64), offsets=np.zeros(3, np.int64))
            self.assertEqual(lb.feature_kind(d), ("sift", None))
            np.savez(pathlib.Path(d) / "features.npz", kf_ns=np.zeros(1, np.int64), extractor=np.array("aliked"),
                     image_wh=np.array([[758, 572], [757, 569]], np.int32))
            kind, wh = lb.feature_kind(d)
            self.assertEqual(kind, "aliked")
            self.assertEqual(wh.tolist(), [[758, 572], [757, 569]])


@unittest.skipUnless(HAVE_CV2, "needs opencv-python")
class LoopEdgeTest(unittest.TestCase):
    def test_pnp_body_pose_recovers_the_body(self):
        rng = np.random.default_rng(5)
        cam = two_cameras()[1]
        r_wb, t_wb = rot(rng, 0.5), np.array([3.0, -2.0, 1.5])
        r_wc = r_wb @ cam["R_ic"]
        center = r_wb @ cam["t_ic"] + t_wb
        n = 60
        xy = rng.uniform([60, 60], [720, 520], size=(n, 2))
        depth = rng.uniform(2, 15, size=n)
        rays = np.c_[xy, np.ones(n)] @ np.linalg.inv(cam["K"]).T
        world = center + (rays * depth[:, None]) @ r_wc.T
        seen = xy + rng.normal(scale=0.2, size=xy.shape)
        seen[:6] += rng.uniform(30, 80, size=(6, 2))  # a few wrong matches for RANSAC to reject
        # start the PnP from a guess 0.3 m and a few degrees off
        got = lb.pnp_body_pose(world, seen, cam["K"], cam["R_ic"], cam["t_ic"],
                               lb.so3_exp(np.array([0.03, -0.02, 0.04])) @ r_wb, t_wb + np.array([0.2, 0.2, -0.1]), 15)
        self.assertIsNotNone(got)
        self.assertLess(np.degrees(np.arccos(np.clip((np.trace(got[0].T @ r_wb) - 1) / 2, -1, 1))), 0.1)
        np.testing.assert_allclose(got[1], t_wb, atol=0.02)


@unittest.skipUnless(HAVE_GTSAM, "gtsam not installed")
class WarmStartTest(unittest.TestCase):
    def test_states_of_an_earlier_solve_seed_the_window(self):
        import tempfile

        from gtsam.symbol_shorthand import B, V
        kf_ns = np.arange(5, dtype=np.int64) * 500_000_000
        kf_p = np.c_[np.arange(5) * 0.5, np.zeros((5, 2))]  # 1 m/s along x
        kf_r = np.tile(np.eye(3), (5, 1, 1))
        vio = (kf_ns, kf_p, kf_r)
        imu_ns = np.arange(0, kf_ns[-1] + 1, 5_000_000)
        noise = {"accel_noise_std": 0.01, "gyro_noise_std": 1e-3, "accel_bias_std": 1e-4, "gyro_bias_std": 1e-6}
        # an earlier solve that ended at the third keyframe
        init = {"kf_ns": kf_ns[:3], "vel": np.array([[1.1, 0, 0], [1.2, 0, 0], [1.3, 0, 0]]),
                "bias": np.array([[0.01 * k, 0, 0, 1e-4 * k, 0, 0] for k in range(1, 4)])}
        with tempfile.NamedTemporaryFile("w", suffix=".csv") as f:
            np.savetxt(f, np.c_[imu_ns, np.zeros((len(imu_ns), 3)), np.tile([0, 0, 9.81], (len(imu_ns), 1))],
                       delimiter=",", fmt="%.9g")
            f.flush()
            _, values, _, _ = lb._imu_graph(kf_ns, kf_p, kf_r, vio, f.name, noise, init=init)
        np.testing.assert_allclose(values.atVector(V(1)), [1.2, 0, 0])
        np.testing.assert_allclose(values.atVector(V(4)), [1.0, 0, 0])  # not in init: the trajectory's
        np.testing.assert_allclose(values.atConstantBias(B(2)).gyroscope(), [3e-4, 0, 0])
        np.testing.assert_allclose(values.atConstantBias(B(4)).accelerometer(), [0.03, 0, 0])  # init's last


if __name__ == "__main__":
    unittest.main()
