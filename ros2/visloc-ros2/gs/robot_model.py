"""A sidewalk delivery robot with a camera mast, drawn into splat renders by depth.

    model = Model(camera_height=1.2)
    rgb = shadow(rgb, project, pose, z0, R_wc, c_w)
    rgb = draw(model, rgb, depth, pose, z0, R_wc, c_w, fx)

`draw` rasterizes the robot's triangles at twice the resolution (smooth
normals, a sun and a sky term, a little specular), keeps the pixels where
the robot is in front of the splat's depth, and blends the antialiased
result over the render. For `chase_video.py`'s videos (docs/navigation.md).
"""
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

WHITE, SKIRT, ORANGE, TYRE, HUB = (236, 236, 232), (58, 60, 66), (242, 118, 30), (28, 28, 31), (150, 152, 158)
MAST, CAMBODY, LENS, LIGHT_C, LED = (60, 62, 68), (30, 30, 34), (10, 10, 12), (255, 250, 220), (40, 170, 255)
_T, _N, _C, _E = [], [], [], []   # triangles, vertex normals (per corner), colour, emissive


def _add(tri, nrm, col, emissive=False):
    _T.append(tri); _N.append(nrm); _C.append(col); _E.append(emissive)


def superellipsoid(c, r, e=0.22, col=WHITE, nu=40, nv=20, zmin=-1.0, zmax=1.0, colfn=None):
    """Rounded box: |x/a|^(2/e) + ... = 1, with smooth normals; colfn(z_local) can recolour bands."""
    sgn = lambda v, p: np.sign(v) * np.abs(v) ** p
    us = np.linspace(-math.pi, math.pi, nu + 1)
    vs = np.linspace(-math.pi / 2, math.pi / 2, nv + 1)
    P = np.zeros((nv + 1, nu + 1, 3)); Nn = np.zeros_like(P)
    for i, v in enumerate(vs):
        for j, u in enumerate(us):
            x = r[0] * sgn(math.cos(v), e) * sgn(math.cos(u), e)
            y = r[1] * sgn(math.cos(v), e) * sgn(math.sin(u), e)
            z = r[2] * sgn(math.sin(v), e)
            n = np.array([sgn(math.cos(v), 2 - e) * sgn(math.cos(u), 2 - e) / r[0],
                          sgn(math.cos(v), 2 - e) * sgn(math.sin(u), 2 - e) / r[1],
                          sgn(math.sin(v), 2 - e) / r[2]])
            P[i, j] = (x, y, max(min(z, zmax * r[2]), zmin * r[2])); Nn[i, j] = n / (np.linalg.norm(n) + 1e-12)
    P += c
    for i in range(nv):
        for j in range(nu):
            a, b, cc, d = (i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j)
            zc = (P[a][2] + P[cc][2]) / 2 - c[2]
            colr = colfn(zc) if colfn else col
            _add((P[a], P[b], P[cc]), (Nn[a], Nn[b], Nn[cc]), colr)
            _add((P[a], P[cc], P[d]), (Nn[a], Nn[cc], Nn[d]), colr)


def cylinder(p0, p1, r, col, n=24, caps=True, capcol=None, emissive=False):
    p0, p1 = np.array(p0, float), np.array(p1, float); a = p1 - p0; a /= np.linalg.norm(a)
    u = np.cross(a, [0, 0, 1] if abs(a[2]) < 0.9 else [1, 0, 0]); u /= np.linalg.norm(u); w = np.cross(a, u)
    dirs = [math.cos(2 * math.pi * k / n) * u + math.sin(2 * math.pi * k / n) * w for k in range(n)]
    for k in range(n):
        m = (k + 1) % n
        q = (p0 + r * dirs[k], p0 + r * dirs[m], p1 + r * dirs[m], p1 + r * dirs[k])
        _add((q[0], q[1], q[2]), (dirs[k], dirs[m], dirs[m]), col, emissive)
        _add((q[0], q[2], q[3]), (dirs[k], dirs[m], dirs[k]), col, emissive)
        if caps:
            for p, s in ((p0, -1), (p1, 1)):
                _add((p, p + r * dirs[m], p + r * dirs[k]), (s * a, s * a, s * a), capcol or col, emissive)


class Model:
    """The robot's triangles in its own frame (x forward, y left, z up; origin on the ground
    under the body's centre), with the camera on a mast at `camera_height` over base_link."""

    def __init__(self, camera_height=1.2):
        del _T[:], _N[:], _C[:], _E[:]
        h = camera_height
        band = lambda z: SKIRT if z < -0.10 else (ORANGE if z < -0.06 else WHITE)
        superellipsoid(np.array([0.0, 0, 0.36]), (0.42, 0.30, 0.20), 0.25, colfn=band, nu=32, nv=16)          # body
        superellipsoid(np.array([0.0, 0, 0.555]), (0.38, 0.27, 0.05), 0.3, col=WHITE, zmin=0.0, nu=32, nv=12)  # domed lid
        for x in (-0.27, 0.0, 0.27):                                                              # six wheels
            for y in (-0.31, 0.31):
                s = 1 if y > 0 else -1
                cylinder([x, y - s * 0.035, 0.10], [x, y + s * 0.035, 0.10], 0.10, TYRE, capcol=TYRE)
                cylinder([x, y + s * 0.036, 0.10], [x, y + s * 0.040, 0.10], 0.055, HUB, capcol=HUB)
        for y in (-0.16, 0.16):                                                                   # headlights
            cylinder([0.405, y, 0.40], [0.425, y, 0.40], 0.035, LIGHT_C, emissive=True)
        cylinder([0.41, -0.06, 0.31], [0.43, 0.06, 0.31], 0.012, LED, emissive=True)            # LED bar
        for y in (-0.17, 0.17):                                                                   # tail lights
            cylinder([-0.425, y, 0.42], [-0.405, y, 0.42], 0.03, (235, 30, 30), emissive=True)
        superellipsoid(np.array([-0.40, 0, 0.30]), (0.03, 0.14, 0.035), 0.3, col=(70, 72, 78))  # rear bumper
        cylinder([0.0, 0, 0.58], [0.0, 0, h - 0.075], 0.018, MAST)                                # mast
        cylinder([0.0, 0, 0.58], [0.0, 0, 0.64], 0.04, SKIRT)                                # mast base
        superellipsoid(np.array([0.0, 0, h]), (0.022, 0.032, 0.075), 0.35, col=CAMBODY)   # 360 camera body
        for s in (1, -1):                                                                         # its two lenses
            cylinder([s * 0.020, 0, h + 0.02], [s * 0.030, 0, h + 0.02], 0.017, LENS, capcol=(40, 60, 90))
        self.tris, self.vn = np.array(_T), np.array(_N)
        self.cols, self.emis = np.array(_C, float), np.array(_E)


SUN = np.array([0.45, 0.25, 0.86]); SUN /= np.linalg.norm(SUN)


def draw(model, rgb, depth, pose, z0, R_wc, c_w, fx, ss=2, pitch=0.0):
    """Composite the robot over rgb (H x W x 3 float) where it is in front of the splat's depth.
    pitch: nose up (rad), e.g. with one pair of wheels up a kerb."""
    H, W = depth.shape
    x, y, yaw = pose; c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = Rz @ np.array([[cp, 0, -sp], [0, 1, 0], [sp, 0, cp]])
    tw = model.tris @ Rz.T + np.array([x, y, z0]); nw = model.vn @ Rz.T
    cam = (tw - c_w) @ R_wc
    if cam[..., 2].max() < 0.3:
        return rgb
    F = fx * ss; HH, WW = H * ss, W * ss
    zb = np.full((HH, WW), np.inf); col = np.zeros((HH, WW, 3)); cov = np.zeros((HH, WW), bool)
    view_dir = tw - c_w; view_dir /= np.linalg.norm(view_dir, axis=2, keepdims=True)
    fn = np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0])
    front = np.einsum('ij,ij->i', fn, tw.mean(1) - c_w) < 0      # outward faces towards the camera
    order = np.argsort(-cam[:, :, 2].mean(1))
    for k in order:
        if not front[k] and not model.emis[k]: continue
        p = cam[k]
        if p[:, 2].min() < 0.25: continue
        u = p[:, 0] / p[:, 2] * F + WW / 2; v = p[:, 1] / p[:, 2] * F + HH / 2
        x0, x1 = int(max(u.min(), 0)), int(min(u.max(), WW - 1)) + 1
        y0, y1 = int(max(v.min(), 0)), int(min(v.max(), HH - 1)) + 1
        if x1 <= x0 or y1 <= y0: continue
        d = (v[1] - v[2]) * (u[0] - u[2]) + (u[2] - u[1]) * (v[0] - v[2])
        if abs(d) < 1e-9: continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        l0 = ((v[1] - v[2]) * (gx - u[2]) + (u[2] - u[1]) * (gy - v[2])) / d
        l1 = ((v[2] - v[0]) * (gx - u[2]) + (u[0] - u[2]) * (gy - v[2])) / d
        l2 = 1 - l0 - l1
        inside = (l0 >= -1e-6) & (l1 >= -1e-6) & (l2 >= -1e-6)
        if not inside.any(): continue
        iz = l0 / p[0, 2] + l1 / p[1, 2] + l2 / p[2, 2]; z = 1 / iz
        sd = depth[np.clip((np.arange(y0, y1) // ss), 0, H - 1)][:, np.clip(np.arange(x0, x1) // ss, 0, W - 1)]
        vis = inside & (z < zb[y0:y1, x0:x1]) & ((sd <= 0) | (z < sd + 0.5))
        if not vis.any(): continue
        # perspective-correct normal interpolation
        w0, w1, w2 = l0 / p[0, 2] / iz, l1 / p[1, 2] / iz, l2 / p[2, 2] / iz
        n = w0[..., None] * nw[k, 0] + w1[..., None] * nw[k, 1] + w2[..., None] * nw[k, 2]
        n /= np.linalg.norm(n, axis=2, keepdims=True) + 1e-12
        if model.emis[k]:
            shade = np.ones(n.shape[:2]) * 1.0; spec = 0.0
        else:
            vd = view_dir[k].mean(0)
            lam = np.clip(n @ SUN, 0, 1)
            h = SUN - vd; h /= np.linalg.norm(h)
            spec = 0.35 * np.clip(n @ h, 0, 1) ** 40
            sky = 0.5 + 0.5 * n[..., 2]
            shade = 0.30 + 0.25 * sky + 0.55 * lam
        c_ = model.cols[k] * shade[..., None] + 255 * (spec[..., None] if np.ndim(spec) else spec)
        zb[y0:y1, x0:x1][vis] = z[vis]
        col[y0:y1, x0:x1][vis] = np.clip(c_[vis], 0, 255)
        cov[y0:y1, x0:x1][vis] = True
    # 2x2 average -> colour and coverage (antialiased edges)
    cs = col.reshape(H, ss, W, ss, 3).sum((1, 3)); a = cov.reshape(H, ss, W, ss).sum((1, 3)) / (ss * ss)
    m = a > 0
    out = rgb.copy()
    out[m] = out[m] * (1 - a[m, None]) + cs[m] / (ss * ss)
    return out


def shadow(rgb, project, pose, z0, R_wc, c_w):
    """A soft contact shadow on the ground under the robot."""
    H, W = rgb.shape[:2]
    x, y, yaw = pose; c, s = math.cos(yaw), math.sin(yaw)
    ring = [(x + c * 0.50 * math.cos(t) - s * 0.36 * math.sin(t), y + s * 0.50 * math.cos(t) + c * 0.36 * math.sin(t), z0 + 0.01)
            for t in np.linspace(0, 2 * math.pi, 48)]
    pts, ok = project(ring, R_wc, c_w)
    if not ok.all():
        return rgb
    m = Image.new('L', (W, H), 0)
    ImageDraw.Draw(m).polygon([tuple(p) for p in pts], fill=150)
    m = m.filter(ImageFilter.GaussianBlur(6))
    return rgb * (1 - np.array(m)[..., None] / 255.0)
