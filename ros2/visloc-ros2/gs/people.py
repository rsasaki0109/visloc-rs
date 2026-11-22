"""Walking pedestrians for gs_sim.py: their motion, their lidar returns and their look in the camera.

    crowd = Crowd(path, n=6, grid=grid, seed=0)
    crowd.step(dt, robot_xy)                 # walk; wait while the robot is just ahead
    ranges = crowd.scan(x, y, yaw, n, ranges) # nearer returns where a ray hits a person
    rgb = crowd.draw(rgb, depth, R_wc, c_w, fx, ground_z)

Each pedestrian walks along the scene's route (the capture walk, through
its goals) at their own speed and side offset, turns round at its ends, and
pauses while the robot is within `yield_m` in front of them, as people step
aside or wait on a sidewalk. Robots must still steer round them: they do
not leave the robot's way. In the scan each is a 0.25 m circle; in the
camera a low-poly figure (legs and arms swinging with the gait, clothes in
varied colours) drawn over the splat by depth with `robot_model.draw`.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

import robot_model as rm

SKIN = [(241, 205, 175), (214, 168, 128), (168, 118, 82), (110, 76, 52)]
SHIRT = [(200, 60, 55), (60, 110, 190), (235, 235, 230), (40, 40, 44), (90, 150, 90), (230, 180, 60), (150, 90, 160)]
TROUSERS = [(40, 48, 70), (30, 30, 34), (150, 130, 100), (80, 85, 95), (60, 80, 120)]
HAIR = [(30, 24, 20), (70, 48, 30), (20, 20, 22), (150, 120, 80)]
PHASES = 12
RADIUS = 0.25


class Figure:
    """One pedestrian's triangles at each of PHASES gait phases (x forward, z up, feet at z = 0)."""

    def __init__(self, rng):
        self.frames = self._build(rng, 1.0)
        self.low = self._build(rng, 0.5, same_as=self)

    def _build(self, rng, detail, same_as=None):
        """Meshes at each gait phase; detail scales the tessellation (0.5 for people far away)."""
        if same_as is not None:
            skin, shirt, trousers, hair, s = same_as.look
        else:
            skin, shirt = SKIN[rng.integers(len(SKIN))], SHIRT[rng.integers(len(SHIRT))]
            trousers, hair = TROUSERS[rng.integers(len(TROUSERS))], HAIR[rng.integers(len(HAIR))]
            s = rng.uniform(0.92, 1.08)                   # height scale (about 1.55 - 1.85 m)
            self.look = (skin, shirt, trousers, hair, s)
        q = lambda k: max(4, int(round(k * detail)))  # noqa: E731
        frames = []
        for ph in range(PHASES):
            a = math.sin(2 * math.pi * ph / PHASES)
            del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
            hip, sh = 0.88 * s, 1.40 * s
            for side, sw in ((1, a), (-1, -a)):           # legs: thigh and shin swing about the hip
                y = 0.10 * side
                knee = np.array([0.42 * s * math.sin(0.45 * sw), y, hip - 0.42 * s * math.cos(0.45 * sw)])
                foot = knee + np.array([0.44 * s * math.sin(0.45 * sw - 0.15 * abs(sw)), 0,
                                        -0.44 * s * math.cos(0.45 * sw - 0.15 * abs(sw))])
                rm.cylinder([0, y, hip], knee, 0.075, trousers, n=q(10))
                rm.cylinder(knee, foot, 0.06, trousers, n=q(10))
                rm.superellipsoid(foot + np.array([0.06, 0, 0.03]), (0.12, 0.05, 0.035), 0.5, col=(35, 35, 38),
                                  nu=q(10), nv=q(6))
            rm.superellipsoid(np.array([0, 0, hip + 0.02]), (0.11, 0.17, 0.11), 0.8, col=trousers, nu=q(14), nv=q(8))  # hips
            rm.superellipsoid(np.array([0, 0, hip + 0.17 * s]), (0.10, 0.16, 0.16 * s), 0.9, col=shirt,
                              nu=q(14), nv=q(8))               # waist
            rm.superellipsoid(np.array([0, 0, sh - 0.14 * s]), (0.12, 0.19, 0.17 * s), 0.7, col=shirt,
                              nu=q(16), nv=q(8))               # chest and shoulders
            for side, sw in ((1, -a), (-1, a)):           # arms swing against the legs
                y = 0.20 * side
                hand = np.array([0.58 * s * math.sin(0.35 * sw), y * 1.05, sh - 0.03 - 0.58 * s * math.cos(0.35 * sw)])
                elbow = (np.array([0, y, sh - 0.03]) + hand) / 2
                rm.cylinder([0, y, sh - 0.03], elbow, 0.048, shirt, n=q(8))
                rm.cylinder(elbow, hand, 0.04, skin, n=q(8))
            rm.cylinder([0, 0, sh], [0, 0, sh + 0.08 * s], 0.05, skin, n=q(8))            # neck
            head = np.array([0.01, 0, sh + 0.19 * s])
            rm.superellipsoid(head, (0.10 * s, 0.085 * s, 0.115 * s), 1.0, col=skin, nu=q(12), nv=q(8))
            rm.superellipsoid(head + np.array([-0.015, 0, 0.03 * s]), (0.10 * s, 0.09 * s, 0.10 * s), 1.0, col=hair,
                              zmin=0.0, nu=q(12), nv=q(6))
            m = type('Mesh', (), {})()
            m.tris, m.vn = np.array(rm._T), np.array(rm._N)
            m.cols, m.emis = np.array(rm._C, float), np.array(rm._E)
            frames.append(m)
        del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
        return frames


def _two_link(hip, foot, l1, l2, forward):
    """The knee of a leg from hip to foot (thigh l1, shin l2), bent towards forward."""
    d = foot - hip
    L = min(np.linalg.norm(d), l1 + l2 - 1e-6)
    u = d / (np.linalg.norm(d) + 1e-12)
    a = (l1 * l1 - l2 * l2 + L * L) / (2 * L)
    h = math.sqrt(max(l1 * l1 - a * a, 0.0))
    side = forward - (forward @ u) * u
    side /= np.linalg.norm(side) + 1e-12
    return hip + a * u + h * side


class Bike:
    """A cyclist: bicycle and rider at each of PHASES crank angles (x forward, z up, wheels
    on z = 0). Same shape as Figure: `frames` and `low`."""

    def __init__(self, rng):
        frame_col = [(200, 30, 40), (30, 90, 180), (30, 30, 34), (230, 230, 225), (40, 140, 90)][rng.integers(5)]
        skin, shirt = SKIN[rng.integers(len(SKIN))], SHIRT[rng.integers(len(SHIRT))]
        trousers = TROUSERS[rng.integers(len(TROUSERS))]
        helmet = [(240, 240, 240), (30, 30, 30), (230, 120, 30)][rng.integers(3)]
        self.frames = []
        r_w, hub_x, hub_z = 0.34, 0.52, 0.34
        bb = np.array([0.0, 0, 0.30])
        seat = np.array([-0.18, 0, 0.86])
        head_tube = np.array([0.40, 0, 0.86])
        for ph in range(PHASES):
            th = 2 * math.pi * ph / PHASES
            del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
            for hx in (-hub_x, hub_x):                     # wheels: tyre ring and spokes
                ring = [np.array([hx + r_w * math.cos(2 * math.pi * k / 18), 0, hub_z + r_w * math.sin(2 * math.pi * k / 18)])
                        for k in range(19)]
                for p0, p1 in zip(ring[:-1], ring[1:]):
                    rm.cylinder(p0, p1, 0.022, (25, 25, 28), n=6)
                for k in range(3):
                    a = th / 2 + k * math.pi / 3
                    rm.cylinder([hx - r_w * math.cos(a), 0, hub_z - r_w * math.sin(a)],
                                [hx + r_w * math.cos(a), 0, hub_z + r_w * math.sin(a)], 0.004, (170, 170, 175), n=4,
                                caps=False)
            rear, front = np.array([-hub_x, 0, hub_z]), np.array([hub_x, 0, hub_z])
            for p0, p1 in ((rear, bb), (rear, seat), (bb, seat), (bb, head_tube), (seat + [0.05, 0, -0.04], head_tube),
                           (head_tube, front)):
                rm.cylinder(p0, p1, 0.02, frame_col, n=8)
            bar = head_tube + np.array([-0.04, 0, 0.12])
            rm.cylinder(head_tube, bar, 0.018, (40, 40, 44), n=6)
            rm.cylinder(bar + [0, -0.27, 0], bar + [0, 0.27, 0], 0.016, (40, 40, 44), n=6)
            rm.superellipsoid(seat + np.array([0, 0, 0.03]), (0.13, 0.07, 0.025), 0.6, col=(30, 30, 32), nu=8, nv=4)
            hip = seat + np.array([0.0, 0, 0.09])
            shoulder = np.array([0.16, 0, 1.36])
            for side, a in ((1, th), (-1, th + math.pi)):  # legs on the pedals
                y = 0.11 * side
                pedal = bb + np.array([0.17 * math.cos(a), y, 0.17 * math.sin(a)])
                h = hip + np.array([0, y, 0])
                knee = _two_link(h, pedal, 0.46, 0.46, np.array([1.0, 0, 0.3]))
                rm.cylinder(h, knee, 0.07, trousers, n=8)
                rm.cylinder(knee, pedal, 0.055, trousers, n=8)
                rm.superellipsoid(pedal + np.array([0.04, 0, -0.02]), (0.11, 0.045, 0.035), 0.5, col=(35, 35, 38),
                                  nu=8, nv=4)
            d = shoulder - hip
            rm.superellipsoid(hip + np.array([0, 0, 0.04]), (0.12, 0.17, 0.10), 0.8, col=trousers, nu=10, nv=6)
            rm.cylinder(hip + np.array([0, 0, 0.05]), shoulder - 0.08 * d / np.linalg.norm(d), 0.15, shirt, n=12)
            rm.superellipsoid(shoulder - np.array([0.02, 0, 0.02]), (0.11, 0.19, 0.08), 0.7, col=shirt, nu=12, nv=6)
            for side in (1, -1):                               # arms to the bar
                sh = shoulder + np.array([0, 0.19 * side, -0.02])
                hand = bar + np.array([0, 0.24 * side, 0.02])
                elbow = _two_link(sh, hand, 0.30, 0.30, np.array([-0.3, 0.6 * side, -1.0]))
                rm.cylinder(sh, elbow, 0.045, shirt, n=8)
                rm.cylinder(elbow, hand, 0.038, skin, n=8)
            head = shoulder + np.array([0.08, 0, 0.22])
            rm.cylinder(shoulder, head - [0, 0, 0.09], 0.05, skin, n=8)
            rm.superellipsoid(head, (0.10, 0.085, 0.115), 1.0, col=skin, nu=10, nv=6)
            rm.superellipsoid(head + np.array([-0.01, 0, 0.04]), (0.13, 0.11, 0.10), 1.0, col=helmet, zmin=0.0,
                              nu=10, nv=5)
            d = shoulder - hip
            m = type('Mesh', (), {})()
            m.tris, m.vn = np.array(rm._T), np.array(rm._N)
            m.cols, m.emis = np.array(rm._C, float), np.array(rm._E)
            self.frames.append(m)
        del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
        self.low = self.frames


def _rot_to_quat(R):
    """(w, x, y, z) of rotation matrices R (N x 3 x 3)."""
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    q = np.empty((len(R), 4))
    w = np.sqrt(np.maximum(1 + tr, 1e-12)) / 2
    big = w > 0.1
    q[:, 0] = w
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * w)
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * w)
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)
    for i in np.where(~big)[0]:               # small w: Shepperd's other branches
        m = R[i]
        k = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
        a, b, c = k, (k + 1) % 3, (k + 2) % 3
        r = np.sqrt(max(1 + m[a, a] - m[b, b] - m[c, c], 1e-12))
        v = np.zeros(4)
        v[1 + a] = r / 2
        v[1 + b] = (m[b, a] + m[a, b]) / (2 * r)
        v[1 + c] = (m[c, a] + m[a, c]) / (2 * r)
        v[0] = (m[c, b] - m[b, c]) / (2 * r)
        q[i] = v
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def mesh_splats(fig):
    """One flat gaussian per triangle of a mesh (its own frame): N x 14 rows of
    x y z, log scales, quaternion (w x y z), opacity logit, r g b in 0..1. The
    colour carries the same sky-and-sun shading as the drawn figures."""
    t = fig.tris
    c = t.mean(1)
    e1 = t[:, 1] - t[:, 0]
    e2 = t[:, 2] - t[:, 0]
    n = np.cross(e1, e2)
    area = np.linalg.norm(n, axis=1)
    keep = area > 1e-10
    t, c, e1, e2, n, area = t[keep], c[keep], e1[keep], e2[keep], n[keep], area[keep]
    n /= area[:, None]
    # In-plane axes: the longest edge and its perpendicular.
    edges = np.stack([t[:, 1] - t[:, 0], t[:, 2] - t[:, 1], t[:, 0] - t[:, 2]], 1)
    lens = np.linalg.norm(edges, axis=2)
    a = edges[np.arange(len(t)), lens.argmax(1)]
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = np.cross(n, a)
    span_a = np.ptp(np.einsum('nkj,nj->nk', t - c[:, None], a), axis=1)
    span_b = np.ptp(np.einsum('nkj,nj->nk', t - c[:, None], b), axis=1)
    # A gaussian shows to about 3 sigma: long thin triangles (spokes, tubes) are
    # covered by several along their length, each sigma ~ a third of its piece.
    pieces = np.maximum(1, np.ceil(span_a / np.maximum(2.5 * span_b, 0.05))).astype(int)
    R = np.stack([a, b, n], 2)                   # columns: the gaussian's axes
    q = _rot_to_quat(R)
    vn = fig.vn[keep].mean(1)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
    shade = 0.30 + 0.25 * (0.5 + 0.5 * vn[:, 2]) + 0.55 * np.clip(vn @ rm.SUN, 0, 1)
    col = np.clip(fig.cols[keep] * shade[:, None] / 255.0, 0, 1)
    idx = np.repeat(np.arange(len(t)), pieces)
    k = np.concatenate([np.arange(m) for m in pieces])
    frac = (k + 0.5) / pieces[idx] - 0.5           # piece centres along the long axis
    # Centre the pieces on the triangle's extent along a.
    proj = np.einsum('nkj,nj->nk', t - c[:, None], a)
    mid_a = (proj.max(1) + proj.min(1)) / 2
    out = np.zeros((len(idx), 14), np.float32)
    out[:, 0:3] = c[idx] + (mid_a[idx] + frac * span_a[idx])[:, None] * a[idx]
    sa = np.maximum(span_a[idx] / pieces[idx] * 0.36, 0.004)
    sb = np.maximum(span_b[idx] * 0.36, 0.004)
    out[:, 3:6] = np.log(np.stack([sa, sb, np.full(len(idx), 0.003)], 1))
    out[:, 6:10] = q[idx]
    out[:, 10] = 4.0
    out[:, 11:14] = col[idx]
    return out


def place_splats(splats, x, y, yaw, z0):
    """Splat rows of a figure moved to (x, y, z0), turned by yaw about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    out = splats.copy()
    p = splats[:, 0:3]
    out[:, 0] = c * p[:, 0] - s * p[:, 1] + x
    out[:, 1] = s * p[:, 0] + c * p[:, 1] + y
    out[:, 2] = p[:, 2] + z0
    hw, hz = math.cos(yaw / 2), math.sin(yaw / 2)   # q_yaw = (hw, 0, 0, hz), times q
    w, qx, qy, qz = splats[:, 6], splats[:, 7], splats[:, 8], splats[:, 9]
    out[:, 6] = hw * w - hz * qz
    out[:, 7] = hw * qx - hz * qy
    out[:, 8] = hw * qy + hz * qx
    out[:, 9] = hw * qz + hz * w
    return out


def draw_fast(fig, rgb, depth, pose, z0, R_wc, c_w, fx):
    """Flat-shaded polygons filled by PIL, back to front, hidden where the splat is nearer than the
    person: a quick look for the simulator's camera (robot_model.draw is the smooth one for videos)."""
    H, W = depth.shape
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    tw = fig.tris @ Rz.T + np.array([x, y, z0])
    cam = (tw - c_w) @ R_wc
    fn = np.cross(tw[:, 1] - tw[:, 0], tw[:, 2] - tw[:, 0])
    keep = (np.einsum('ij,ij->i', fn, tw.mean(1) - c_w) < 0) & (cam[:, :, 2].min(1) > 0.25)
    if not keep.any():
        return rgb
    cam, n = cam[keep], (fig.vn[keep].mean(1) @ Rz.T)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    shade = 0.30 + 0.25 * (0.5 + 0.5 * n[:, 2]) + 0.55 * np.clip(n @ rm.SUN, 0, 1)
    col = np.clip(fig.cols[keep] * shade[:, None], 0, 255).astype(int)
    uv = np.stack([cam[:, :, 0] / cam[:, :, 2] * fx + W / 2, cam[:, :, 1] / cam[:, :, 2] * fx + H / 2], 2)
    layer = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for k in np.argsort(-cam[:, :, 2].mean(1)):
        d.polygon([tuple(p) for p in uv[k]], fill=(*col[k], 255))
    a = np.asarray(layer)
    m = a[..., 3] > 0
    dist = float(np.linalg.norm(np.array([x, y, z0 + 0.9]) - c_w))
    m &= (depth <= 0) | (dist < depth + 0.4)
    out = rgb.copy()
    out[m] = a[..., :3][m]
    return out


class Crowd:
    def __init__(self, path, n, grid, seed=0, speed=1.1, yield_m=1.2, cyclists=0, cyclist_speed=4.0):
        rng = np.random.default_rng(seed + 1000)
        self.path = np.asarray(path, float)[:, :2]
        seg = np.linalg.norm(np.diff(self.path, axis=0), axis=1)
        self.cum = np.concatenate([[0], np.cumsum(seg)])
        self.length = self.cum[-1]
        self.grid, self.yield_m = grid, yield_m
        self.s = rng.uniform(0, self.length, n)                     # arc position along the route
        self.dir = rng.choice([-1.0, 1.0], n)
        self.v = speed * rng.uniform(0.8, 1.2, n)
        self.side = rng.uniform(-1.0, 1.0, n)
        self.phase = rng.uniform(0, 1, n)
        self.figures = [Figure(rng) for _ in range(n)]
        self.kind = np.zeros(n, int)                                # 0: on foot, 1: on a bicycle
        if cyclists:                                                # drawn after the walkers: their looks stay
            self.s = np.concatenate([self.s, rng.uniform(0, self.length, cyclists)])
            self.dir = np.concatenate([self.dir, rng.choice([-1.0, 1.0], cyclists)])
            self.v = np.concatenate([self.v, cyclist_speed * rng.uniform(0.8, 1.2, cyclists)])
            self.side = np.concatenate([self.side, rng.uniform(-1.0, 1.0, cyclists)])
            self.phase = np.concatenate([self.phase, rng.uniform(0, 1, cyclists)])
            self.figures += [Bike(rng) for _ in range(cyclists)]
            self.kind = np.concatenate([self.kind, np.ones(cyclists, int)])
            n += cyclists
        self.xy, self.yaw = np.zeros((n, 2)), np.zeros(n)
        self._place()

    def _at(self, s):
        k = np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.path) - 2)
        t = (s - self.cum[k]) / np.maximum(self.cum[k + 1] - self.cum[k], 1e-9)
        d = self.path[k + 1] - self.path[k]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        return self.path[k] + t[:, None] * (self.path[k + 1] - self.path[k]), d

    def _place(self):
        p, d = self._at(self.s)
        normal = np.stack([-d[:, 1], d[:, 0]], 1)
        for i in range(len(self.s)):
            off = self.side[i]                    # shrink the side offset until the spot is free
            while abs(off) > 0.05 and self.grid.hits(*(p[i] + off * normal[i]), RADIUS):
                off *= 0.7
            self.xy[i] = p[i] + off * normal[i]
            self.yaw[i] = math.atan2(d[i, 1] * self.dir[i], d[i, 0] * self.dir[i])

    def step(self, dt, robot_xy):
        for i in range(len(self.s)):
            rel = np.asarray(robot_xy) - self.xy[i]
            ahead = rel @ [math.cos(self.yaw[i]), math.sin(self.yaw[i])]
            gap = self.yield_m * (2.5 if self.kind[i] else 1.0)
            if 0 < ahead < gap and abs(rel @ [-math.sin(self.yaw[i]), math.cos(self.yaw[i])]) < 0.7:
                continue                          # the robot is just in front: wait
            self.s[i] += self.dir[i] * self.v[i] * dt
            # one stride per 1.3 m on foot, one crank turn per 4.5 m on a bicycle
            self.phase[i] = (self.phase[i] + self.v[i] * dt / (4.5 if self.kind[i] else 1.3)) % 1.0
            if self.s[i] < 0 or self.s[i] > self.length:
                self.s[i] = min(max(self.s[i], 0), self.length)
                self.dir[i] *= -1
        self._place()

    def circles(self):
        """Footprints as circles: (centres M x 2, radii M). A person is one circle, a
        bicycle three along its length."""
        cs, rs = [], []
        for p, yw, k in zip(self.xy, self.yaw, self.kind):
            if k:
                for o in (-0.6, 0.0, 0.6):
                    cs.append(p + o * np.array([math.cos(yw), math.sin(yw)]))
                    rs.append(0.3)
            else:
                cs.append(p)
                rs.append(RADIUS)
        return np.array(cs).reshape(-1, 2), np.array(rs)

    def hits(self, x, y, r):
        c, rad = self.circles()
        return bool(len(c)) and bool((np.hypot(c[:, 0] - x, c[:, 1] - y) < r + rad).any())

    def scan(self, x, y, yaw, n, ranges):
        a = yaw + np.linspace(-math.pi, math.pi, n, endpoint=False)
        d = np.stack([np.cos(a), np.sin(a)], 1)
        out = np.array(ranges, float)
        centres, radii = self.circles()
        for p, R in zip(centres, radii):
            f = p - [x, y]
            tca = d @ f
            d2 = f @ f - tca ** 2
            hit = (tca > 0) & (d2 < R ** 2)
            t = tca - np.sqrt(np.maximum(R ** 2 - d2, 0))
            out = np.where(hit & (t < out), np.maximum(t, 0.0), out)
        return out

    def splats(self, c_w, ground_z, max_m=25.0, only=None):
        """The people within max_m of c_w as gaussians (K x 14 float32, see
        mesh_splats), for gsplat_render_server --dynamic."""
        rows = []
        for i in range(len(self.xy)):
            if only is not None and not only[i]:
                continue
            if math.hypot(self.xy[i, 0] - c_w[0], self.xy[i, 1] - c_w[1]) > max_m:
                continue
            fig = self.figures[i]
            k = int(self.phase[i] * PHASES) % PHASES
            if not hasattr(fig, 'splat_frames'):
                fig.splat_frames = [mesh_splats(m) for m in fig.frames]
            rows.append(place_splats(fig.splat_frames[k], self.xy[i, 0], self.xy[i, 1], self.yaw[i],
                                     ground_z(*self.xy[i])))
        return np.concatenate(rows) if rows else np.zeros((0, 14), np.float32)

    def draw(self, rgb, depth, R_wc, c_w, fx, ground_z, max_m=25.0, fast=False, only=None):
        """rgb (H x W x 3 float) with the pedestrians in front of the splat drawn over it."""
        fwd = R_wc[:, 2]
        order = np.argsort(-np.hypot(self.xy[:, 0] - c_w[0], self.xy[:, 1] - c_w[1]))
        for i in order:
            if only is not None and not only[i]:
                continue
            rel = np.array([*self.xy[i], 0.0]) - c_w
            rel[2] = 0
            dist = np.linalg.norm(rel)
            if dist > max_m or rel @ fwd < -0.5:
                continue
            z0 = ground_z(*self.xy[i])
            box = np.array([[self.xy[i, 0] + dx, self.xy[i, 1] + dy, z0 + dz]
                            for dx in (-0.4, 0.4) for dy in (-0.4, 0.4) for dz in (0, 1.9)])
            cam = (box - c_w) @ R_wc
            if (cam[:, 2] < 0.3).all():
                continue
            H, W = depth.shape
            z = np.maximum(cam[:, 2], 0.3)
            u, v = cam[:, 0] / z * fx + W / 2, cam[:, 1] / z * fx + H / 2
            if u.max() < 0 or u.min() >= W or v.max() < 0 or v.min() >= H:
                continue                          # outside the image
            far = dist > 8.0
            fig = (self.figures[i].low if far else self.figures[i].frames)[int(self.phase[i] * PHASES) % PHASES]
            if fast:
                rgb = draw_fast(fig, rgb, depth, (self.xy[i, 0], self.xy[i, 1], self.yaw[i]), z0, R_wc, c_w, fx)
                continue
            rgb = rm.draw(fig, rgb, depth, (self.xy[i, 0], self.xy[i, 1], self.yaw[i]), z0, R_wc, c_w, fx,
                          ss=1 if far else 2)
        return rgb
