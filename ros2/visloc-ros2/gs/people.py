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
        skin, shirt = SKIN[rng.integers(len(SKIN))], SHIRT[rng.integers(len(SHIRT))]
        trousers, hair = TROUSERS[rng.integers(len(TROUSERS))], HAIR[rng.integers(len(HAIR))]
        s = rng.uniform(0.92, 1.08)                       # height scale (about 1.55 - 1.85 m)
        self.frames = []
        for ph in range(PHASES):
            a = math.sin(2 * math.pi * ph / PHASES)
            del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
            hip, sh = 0.88 * s, 1.40 * s
            for side, sw in ((1, a), (-1, -a)):           # legs: thigh and shin swing about the hip
                y = 0.10 * side
                knee = np.array([0.42 * s * math.sin(0.45 * sw), y, hip - 0.42 * s * math.cos(0.45 * sw)])
                foot = knee + np.array([0.44 * s * math.sin(0.45 * sw - 0.15 * abs(sw)), 0,
                                        -0.44 * s * math.cos(0.45 * sw - 0.15 * abs(sw))])
                rm.cylinder([0, y, hip], knee, 0.075, trousers, n=10)
                rm.cylinder(knee, foot, 0.06, trousers, n=10)
                rm.superellipsoid(foot + np.array([0.06, 0, 0.03]), (0.12, 0.05, 0.035), 0.5, col=(35, 35, 38),
                                  nu=10, nv=6)
            rm.superellipsoid(np.array([0, 0, hip + 0.02]), (0.11, 0.17, 0.11), 0.8, col=trousers, nu=14, nv=8)  # hips
            rm.superellipsoid(np.array([0, 0, hip + 0.17 * s]), (0.10, 0.16, 0.16 * s), 0.9, col=shirt,
                              nu=14, nv=8)               # waist
            rm.superellipsoid(np.array([0, 0, sh - 0.14 * s]), (0.12, 0.19, 0.17 * s), 0.7, col=shirt,
                              nu=16, nv=8)               # chest and shoulders
            for side, sw in ((1, -a), (-1, a)):           # arms swing against the legs
                y = 0.20 * side
                hand = np.array([0.58 * s * math.sin(0.35 * sw), y * 1.05, sh - 0.03 - 0.58 * s * math.cos(0.35 * sw)])
                elbow = (np.array([0, y, sh - 0.03]) + hand) / 2
                rm.cylinder([0, y, sh - 0.03], elbow, 0.048, shirt, n=8)
                rm.cylinder(elbow, hand, 0.04, skin, n=8)
            rm.cylinder([0, 0, sh], [0, 0, sh + 0.08 * s], 0.05, skin, n=8)            # neck
            head = np.array([0.01, 0, sh + 0.19 * s])
            rm.superellipsoid(head, (0.10 * s, 0.085 * s, 0.115 * s), 1.0, col=skin, nu=12, nv=8)
            rm.superellipsoid(head + np.array([-0.015, 0, 0.03 * s]), (0.10 * s, 0.09 * s, 0.10 * s), 1.0, col=hair,
                              zmin=0.0, nu=12, nv=6)
            m = type('Mesh', (), {})()
            m.tris, m.vn = np.array(rm._T), np.array(rm._N)
            m.cols, m.emis = np.array(rm._C, float), np.array(rm._E)
            self.frames.append(m)
        del rm._T[:], rm._N[:], rm._C[:], rm._E[:]


class Crowd:
    def __init__(self, path, n, grid, seed=0, speed=1.1, yield_m=1.2):
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
            if 0 < ahead < self.yield_m and abs(rel @ [-math.sin(self.yaw[i]), math.cos(self.yaw[i])]) < 0.6:
                continue                          # the robot is just in front: wait
            self.s[i] += self.dir[i] * self.v[i] * dt
            self.phase[i] = (self.phase[i] + self.v[i] * dt / 1.3) % 1.0   # one stride cycle per 1.3 m
            if self.s[i] < 0 or self.s[i] > self.length:
                self.s[i] = min(max(self.s[i], 0), self.length)
                self.dir[i] *= -1
        self._place()

    def hits(self, x, y, r):
        return bool(len(self.xy)) and bool((np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y) < r + RADIUS).any())

    def scan(self, x, y, yaw, n, ranges):
        a = yaw + np.linspace(-math.pi, math.pi, n, endpoint=False)
        d = np.stack([np.cos(a), np.sin(a)], 1)
        out = np.array(ranges, float)
        for p in self.xy:
            f = p - [x, y]
            tca = d @ f
            d2 = f @ f - tca ** 2
            hit = (tca > 0) & (d2 < RADIUS ** 2)
            t = tca - np.sqrt(np.maximum(RADIUS ** 2 - d2, 0))
            out = np.where(hit & (t < out), np.maximum(t, 0.0), out)
        return out

    def draw(self, rgb, depth, R_wc, c_w, fx, ground_z, max_m=25.0):
        """rgb (H x W x 3 float) with the pedestrians in front of the splat drawn over it."""
        fwd = R_wc[:, 2]
        order = np.argsort(-np.hypot(self.xy[:, 0] - c_w[0], self.xy[:, 1] - c_w[1]))
        for i in order:
            rel = np.array([*self.xy[i], 0.0]) - c_w
            rel[2] = 0
            dist = np.linalg.norm(rel)
            if dist > max_m or rel @ fwd < -0.5:
                continue
            fig = self.figures[i].frames[int(self.phase[i] * PHASES) % PHASES]
            rgb = rm.draw(fig, rgb, depth, (self.xy[i, 0], self.xy[i, 1], self.yaw[i]),
                          ground_z(*self.xy[i]), R_wc, c_w, fx)
        return rgb
