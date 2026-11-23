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

    def __init__(self, rng, extra=None):
        """extra: None, 'dog' (a dog on a lead at their right) or 'stroller' (pushed ahead)."""
        self.extra = extra
        self.frames = self._build(rng, 1.0)
        if extra == 'dog':
            self.dog_col = [(150, 100, 55), (30, 28, 26), (235, 225, 205), (120, 120, 125)][rng.integers(4)]
        elif extra == 'stroller':
            self.stroller_col = [(40, 60, 110), (150, 30, 40), (60, 60, 64), (70, 110, 80)][rng.integers(4)]
        if extra:
            self.frames = self._build(rng, 1.0, same_as=self)
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
            hands = []
            for side, sw in ((1, -a), (-1, a)):           # arms swing against the legs
                y = 0.20 * side
                hand = np.array([0.58 * s * math.sin(0.35 * sw), y * 1.05, sh - 0.03 - 0.58 * s * math.cos(0.35 * sw)])
                if self.extra == 'stroller' and same_as is not None:      # both hands on the handle
                    hand = np.array([0.42, 0.19 * side, 0.98 * s])
                elif self.extra == 'dog' and same_as is not None and side < 0:   # the lead in the right hand
                    hand = np.array([0.22, -0.26, 0.92 * s])
                hands.append(hand)
                elbow = (np.array([0, y, sh - 0.03]) + hand) / 2
                rm.cylinder([0, y, sh - 0.03], elbow, 0.048, shirt, n=q(8))
                rm.cylinder(elbow, hand, 0.04, skin, n=q(8))
            rm.cylinder([0, 0, sh], [0, 0, sh + 0.08 * s], 0.05, skin, n=q(8))            # neck
            head = np.array([0.01, 0, sh + 0.19 * s])
            rm.superellipsoid(head, (0.10 * s, 0.085 * s, 0.115 * s), 1.0, col=skin, nu=q(12), nv=q(8))
            rm.superellipsoid(head + np.array([-0.015, 0, 0.03 * s]), (0.10 * s, 0.09 * s, 0.10 * s), 1.0, col=hair,
                              zmin=0.0, nu=q(12), nv=q(6))
            if self.extra == 'dog' and same_as is not None:
                _dog(np.array([0.85, -0.35, 0.0]), 2 * math.pi * ph / PHASES, self.dog_col, hands[1], q)
            elif self.extra == 'stroller' and same_as is not None:
                _stroller(np.array([0.75, 0.0, 0.0]), 2 * math.pi * ph / PHASES, self.stroller_col, q)
            m = type('Mesh', (), {})()
            m.tris, m.vn = np.array(rm._T), np.array(rm._N)
            m.cols, m.emis = np.array(rm._C, float), np.array(rm._E)
            frames.append(m)
        del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
        return frames


def _dog(base, th, col, hand, q):
    """A medium dog trotting at base (its frame: x forward), legs by gait angle th, on a lead."""
    body = base + np.array([0, 0, 0.42])
    rm.superellipsoid(body, (0.32, 0.11, 0.12), 0.8, col=col, nu=q(14), nv=q(8))
    head = base + np.array([0.36, 0, 0.58])
    rm.superellipsoid(head, (0.11, 0.08, 0.09), 0.9, col=col, nu=q(10), nv=q(6))
    rm.superellipsoid(head + np.array([0.1, 0, -0.03]), (0.06, 0.045, 0.045), 0.9, col=col, nu=q(8), nv=q(4))  # muzzle
    for y in (-0.05, 0.05):                                    # ears
        rm.superellipsoid(head + np.array([-0.02, y, 0.09]), (0.03, 0.02, 0.05), 0.9, col=col, nu=6, nv=4)
    for x, y, ph in ((0.22, 0.07, 0.0), (0.22, -0.07, math.pi), (-0.22, 0.07, math.pi), (-0.22, -0.07, 0.0)):
        top = base + np.array([x, y, 0.38])
        foot = base + np.array([x + 0.09 * math.sin(th + ph), y, 0.02])
        rm.cylinder(top, foot, 0.035, col, n=q(6))
    rm.cylinder(base + np.array([-0.3, 0, 0.47]), base + np.array([-0.42, 0, 0.62 + 0.03 * math.sin(2 * th)]),
                0.025, col, n=q(6))                            # tail
    collar = head + np.array([-0.08, 0, -0.06])
    rm.cylinder(hand, collar, 0.006, (200, 40, 40), n=4, caps=False)   # the lead


def _stroller(base, th, col, q):
    """A pushchair ahead of its pusher (frame: x forward): bassinet, hood, frame, turning wheels."""
    rm.superellipsoid(base + np.array([0.05, 0, 0.62]), (0.34, 0.22, 0.14), 0.6, col=col, nu=q(14), nv=q(8))
    rm.superellipsoid(base + np.array([-0.12, 0, 0.78]), (0.2, 0.23, 0.2), 0.9, col=col, zmin=0.0,
                      nu=q(12), nv=q(6))                       # hood
    grey = (60, 60, 64)
    for y in (-0.19, 0.19):
        rm.cylinder(base + np.array([-0.33, y, 0.98]), base + np.array([-0.05, y, 0.48]), 0.012, grey, n=q(6))
        rm.cylinder(base + np.array([-0.05, y, 0.48]), base + np.array([0.28, y, 0.12]), 0.012, grey, n=q(6))
        rm.cylinder(base + np.array([-0.05, y, 0.48]), base + np.array([-0.25, y, 0.12]), 0.012, grey, n=q(6))
        for x in (0.28, -0.25):                               # wheels
            rm.cylinder(base + np.array([x, y - 0.02, 0.12]), base + np.array([x, y + 0.02, 0.12]), 0.11,
                        (25, 25, 28), n=q(12))
    rm.cylinder(base + np.array([-0.33, -0.22, 0.98]), base + np.array([-0.33, 0.22, 0.98]), 0.016, grey, n=q(6))
    del th


class Car:
    """A small car (frame: x forward, z up, wheels on z = 0), its wheels turned by PHASES steps."""

    def __init__(self, rng):
        paint = [(200, 30, 35), (235, 235, 232), (30, 32, 36), (60, 90, 150), (150, 155, 160), (40, 90, 60),
                 (220, 180, 50)][rng.integers(7)]
        glass, tyre, rim = (35, 45, 55), (22, 22, 24), (175, 178, 182)
        self.frames = []
        for ph in range(PHASES):
            th = 2 * math.pi * ph / PHASES
            del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
            rm.superellipsoid(np.array([0.0, 0, 0.62]), (2.2, 0.88, 0.33), 0.35, col=paint, nu=24, nv=10)   # body
            rm.superellipsoid(np.array([-0.25, 0, 1.06]), (1.25, 0.8, 0.3), 0.45, col=paint, zmin=0.0,
                              nu=20, nv=8)                                                               # roof
            rm.superellipsoid(np.array([-0.25, 0, 1.08]), (1.18, 0.82, 0.22), 0.5, col=glass, zmin=-0.2,
                              zmax=0.6, nu=20, nv=8)                                                     # windows
            for x in (-1.35, 1.35):
                for y in (-0.8, 0.8):
                    s_ = 1 if y > 0 else -1
                    rm.cylinder([x, y - s_ * 0.12, 0.33], [x, y + s_ * 0.1, 0.33], 0.33, tyre, n=16)
                    hub = np.array([x, y + s_ * 0.105, 0.33])
                    for k in range(3):                                        # spokes turn with the phase
                        a = th + k * math.pi / 3
                        rm.cylinder(hub - [0.2 * math.cos(a), 0, 0.2 * math.sin(a)],
                                    hub + [0.2 * math.cos(a), 0, 0.2 * math.sin(a)], 0.03, rim, n=4)
            for y in (-0.6, 0.6):
                rm.cylinder([2.18, y, 0.7], [2.22, y, 0.7], 0.11, (255, 250, 225), emissive=True)   # headlights
                rm.cylinder([-2.22, y, 0.75], [-2.18, y, 0.75], 0.09, (220, 25, 25), emissive=True)  # tail lights
            m = type('Mesh', (), {})()
            m.tris, m.vn = np.array(rm._T), np.array(rm._N)
            m.cols, m.emis = np.array(rm._C, float), np.array(rm._E)
            self.frames.append(m)
        del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
        self.low = self.frames


def estimate_road(xyz, rgb, opacity, route, ground_z, search=(2.5, 12.0), min_score=0.25):
    """Where a road runs beside the route, from the splat's colours: per route segment, the
    side offset (m, + to the left of the route's direction) of the densest band of
    asphalt-like ground (grey, near the ground, opaque), or None where there is no such band.
    Returns [(s_start, s_end, offset or None)] along the route. A guess: pass
    --road-offset when the scene says otherwise."""
    route = np.asarray(route, float)[:, :2]
    gz = np.array([ground_z(x, y) for x, y in xyz[:, :2]])
    mx, mn = rgb.max(1), rgb.min(1)
    grey = ((mx - mn) / np.maximum(mx, 1e-3) < 0.12) & (mx > 0.25) & (mx < 0.65)
    ground = (np.abs(xyz[:, 2] - gz) < 0.15) & (opacity > 0.5)
    bins = np.arange(-search[1], search[1] + 0.01, 1.0)
    centre = (bins[:-1] + bins[1:]) / 2
    out, s0 = [], 0.0
    for a, b in zip(route[:-1], route[1:]):
        d = b - a
        L = float(np.linalg.norm(d))
        u = d / max(L, 1e-9)
        n = np.array([-u[1], u[0]])
        rel = xyz[:, :2] - a
        m = ground & (rel @ u > 0) & (rel @ u < L)
        lat = rel @ n
        tot = np.histogram(lat[m], bins)[0]
        asp = np.histogram(lat[m & grey], bins)[0]
        # a 3 m wide band: the road, not a strip of shadow
        tot3 = np.convolve(tot, np.ones(3), 'same')
        asp3 = np.convolve(asp, np.ones(3), 'same')
        ok = (np.abs(centre) >= search[0]) & (tot3 > 60)
        frac = np.where(ok, asp3 / np.maximum(tot3, 1), 0)
        k = int(np.argmax(frac))
        out.append((s0, s0 + L, float(centre[k]) if frac[k] >= min_score else None))
        s0 += L
    return out


def check_road(road, route, ground_z, render, min_sharpness=4.0, relative=0.4, samples=3):
    """Drops the stretches of an estimated road where the splat is broken up: from points
    along each stretch, it looks towards the road (camera 1.6 m up) with render(R_cw, t_cw)
    -> grey image, and keeps the stretch only when the lower half of those views is sharp:
    their median variance of the Laplacian at least min_sharpness and at least `relative`
    times the sharpest stretch's (the scale depends on the camera). Returns (road,
    sharpness per stretch)."""
    route = np.asarray(route, float)[:, :2]
    out, sharp = [], []
    for k, (a, b, off) in enumerate(road):
        if off is None:
            out.append((a, b, None))
            sharp.append(None)
            continue
        p0, p1 = route[k], route[k + 1]
        u = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        n = np.array([-u[1], u[0]])
        values = []
        for f in (np.arange(samples) + 0.5) / samples:
            x, y = p0 + f * (p1 - p0)
            lx, ly = np.array([x, y]) + off * n
            yaw = math.atan2(ly - y, lx - x)
            c, s_ = math.cos(yaw), math.sin(yaw)
            centre = np.array([x, y, 1.6 + ground_z(x, y)])
            R_wc = np.array([[s_, 0, c], [-c, 0, s_], [0, -1, 0]])
            img = np.asarray(render(R_wc.T, -R_wc.T @ centre), float)
            half = img[img.shape[0] // 2:]
            lap = 4 * half[1:-1, 1:-1] - half[:-2, 1:-1] - half[2:, 1:-1] - half[1:-1, :-2] - half[1:-1, 2:]
            values.append(float(lap.var()))
        sharp.append(float(np.median(values)))
        out.append((a, b, off))
    best = max((v for v in sharp if v is not None), default=0.0)
    out = [(a, b, off if v is not None and v >= max(min_sharpness, relative * best) else None)
           for (a, b, off), v in zip(out, sharp)]
    return out, sharp


# Footprints (along, left, radius) in each kind's frame: walker, cyclist, dog walker, stroller, car.
FOOT = {0: ((0.0, 0.0, RADIUS),),
        1: ((-0.6, 0.0, 0.3), (0.0, 0.0, 0.3), (0.6, 0.0, 0.3)),
        2: ((0.0, 0.0, RADIUS), (0.85, -0.35, 0.22)),
        3: ((0.0, 0.0, RADIUS), (0.75, 0.0, 0.32)),
        4: ((-1.4, 0.0, 0.9), (0.0, 0.0, 0.9), (1.4, 0.0, 0.9))}
LANE_HALF = 1.6     # a lane's centre from the road's centre line (m)


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
    def __init__(self, path, n, grid, seed=0, speed=1.1, yield_m=1.2, cyclists=0, cyclist_speed=4.0,
                 dog_walkers=0, strollers=0, pauses=True, cars=0, road_offset=None, car_speed=6.0,
                 crossing=False):
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
        for kind, count, extra, v in ((2, dog_walkers, 'dog', 0.9 * speed), (3, strollers, 'stroller', 0.8 * speed)):
            if count:                                               # after the cyclists, likewise
                self.s = np.concatenate([self.s, rng.uniform(0, self.length, count)])
                self.dir = np.concatenate([self.dir, rng.choice([-1.0, 1.0], count)])
                self.v = np.concatenate([self.v, v * rng.uniform(0.8, 1.2, count)])
                self.side = np.concatenate([self.side, rng.uniform(-1.0, 1.0, count)])
                self.phase = np.concatenate([self.phase, rng.uniform(0, 1, count)])
                self.figures += [Figure(rng, extra) for _ in range(count)]
                self.kind = np.concatenate([self.kind, np.full(count, kind)])
                n += count
        if cars:                                                    # last, likewise
            if road_offset is None:
                raise ValueError('cars need road_offset (the road centre line beside the route)')
            # A number: one road all along; else [(s_start, s_end, offset or None)] per segment,
            # and the cars keep to the longest stretch of road.
            road = [(0.0, self.length, float(road_offset))] if np.isscalar(road_offset) else road_offset
            runs, cur = [], []
            for seg in road:
                if seg[2] is None:
                    cur and runs.append(cur)
                    cur = []
                else:
                    cur.append(seg)
            cur and runs.append(cur)
            if not runs:
                raise ValueError('no road beside the route')
            self.road = max(runs, key=lambda r: r[-1][1] - r[0][0])
            self.road_offset = road_offset
            self.road_s = (self.road[0][0] + 1.0, self.road[-1][1] - 1.0)
            # The two lanes as polylines offset from the route's vertices along their corner
            # bisectors (mitred), so a car's path stays continuous round the corners.
            seg_off = []
            for a_, b_ in zip(self.cum[:-1], self.cum[1:]):
                mid = (a_ + b_) / 2
                o = next((o for a, b, o in self.road if a <= mid <= b), None)
                seg_off.append(o)
            last = next(o for o in seg_off if o is not None)
            for k, o in enumerate(seg_off):       # stretches without a road: the nearest road's offset
                seg_off[k] = last if o is None else o
                last = seg_off[k]
            dirs = np.diff(self.path, axis=0)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-12
            normals = np.stack([-dirs[:, 1], dirs[:, 0]], 1)
            self.lanes = {}
            for d_ in (1.0, -1.0):
                verts = []
                for k in range(len(self.path)):
                    na = normals[max(k - 1, 0)]
                    nb = normals[min(k, len(normals) - 1)]
                    oa = seg_off[max(k - 1, 0)] - LANE_HALF * d_
                    ob = seg_off[min(k, len(normals) - 1)] - LANE_HALF * d_
                    m = na + nb
                    m /= np.linalg.norm(m) + 1e-12
                    scale = 1.0 / max(m @ nb, 0.5)          # mitre, capped at a 120-degree corner
                    verts.append(self.path[k] + m * scale * (oa + ob) / 2)
                self.lanes[d_] = np.array(verts)
            self.lane_mix = None   # set below: per car, -1 .. 1, the lane it is in (moves on a U-turn)
            car_s = rng.uniform(*self.road_s, cars)
            self.s = np.concatenate([self.s, car_s])
            car_dir = rng.choice([-1.0, 1.0], cars)
            self.dir = np.concatenate([self.dir, car_dir])
            self.v = np.concatenate([self.v, car_speed * rng.uniform(0.8, 1.2, cars)])
            self.side = np.concatenate([self.side, [self.lane(s_, d_) for s_, d_ in zip(car_s, car_dir)]])
            self.phase = np.concatenate([self.phase, rng.uniform(0, 1, cars)])
            self.figures += [Car(rng) for _ in range(cars)]
            self.kind = np.concatenate([self.kind, np.full(cars, 4)])
            n += cars
        self.lane_mix = self.dir.copy()
        # A zebra crossing with a traffic light in the middle of the road's stretch: cars stop
        # at its line while their light is not green, walkers cross while it is red.
        self.crossing_s = (self.road_s[0] + self.road_s[1]) / 2 if (crossing and cars) else None
        self.clock = 0.0
        self.cross = np.zeros(n, int)          # 0 walking, 1 crossing, 2 over there, 3 back
        self.cross_home = np.zeros(n)
        self.cross_wait = np.zeros(n)
        self.cross_tried = np.zeros(n, bool)
        self._decor = {}
        # Behaviour: walkers now and then stand still a while (a chat, a phone); anyone on foot
        # now and then drifts across to the other side.
        self.brng = np.random.default_rng(seed + 2000)
        self.pauses = pauses
        self.pause = np.zeros(n)
        self.side_goal = self.side.copy()
        self.xy, self.yaw = np.zeros((n, 2)), np.zeros(n)
        self.normal = np.zeros((n, 2))
        self.dodge = np.zeros(n)        # extra side offset while stepping round the robot (m)
        self.waited = np.zeros(n)       # seconds blocked by the robot
        self.stuck = np.zeros(n)        # seconds without moving at all
        self.robot_radius = 0.2
        self._place()
        for _ in range(50):             # start nobody inside anybody: move them along until clear
            bad = [i for i in range(n) if not self._clear(i, self.xy[i], self.yaw[i], (1e9, 1e9))]
            if not bad:
                break
            for i in bad[1:] or bad:
                self.s[i] = (self.s[i] + 1.0) % self.length
            self._place()

    def light(self):
        """The cars' light: 'green' (15 s), 'amber' (3 s), 'red' (12 s; walkers cross)."""
        t = self.clock % 30.0
        return 'green' if t < 15 else 'amber' if t < 18 else 'red'

    def _road_side(self):
        """+1 when the road is to the left of the route there, -1 to the right."""
        return 1.0 if self.lane(self.crossing_s, 1.0) + LANE_HALF > 0 else -1.0

    def decor(self, ground_z):
        """The crossing's stripes and its traffic light (lit as now) as gaussians."""
        if self.crossing_s is None:
            return np.zeros((0, 14), np.float32)
        state = self.light()
        if state not in self._decor:
            p, d = self._at(np.array([self.crossing_s]))
            p, d = p[0], d[0]
            n = np.array([-d[1], d[0]])
            mid = self.lane(self.crossing_s, 1.0) + LANE_HALF      # the road's centre line offset
            rows = []
            for k in range(-5, 6):                                 # bars across the road, 0.5 m on, 0.5 off
                if k % 2:
                    continue
                for a in np.arange(-1.4, 1.41, 0.14):            # along the road, 3 m wide crossing
                    for b in np.arange(-0.2, 0.21, 0.14):
                        q = p + (mid + k * 0.5 + b) * n + a * d
                        rows.append([q[0], q[1], ground_z(*q) + 0.02, math.log(0.09), math.log(0.09),
                                     math.log(0.004), 1, 0, 0, 0, 4.0, 0.93, 0.93, 0.9])
            del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
            side = self._road_side()
            base = p + (mid - side * (LANE_HALF + 1.4)) * n - 2.0 * d
            z0 = ground_z(*base)
            rm.cylinder([0, 0, 0], [0, 0, 2.6], 0.06, (60, 62, 66), n=10)
            rm.superellipsoid(np.array([0.0, 0, 2.95]), (0.16, 0.16, 0.42), 0.35, col=(30, 30, 32), nu=10, nv=6)
            lit = {'red': (255, 40, 30), 'amber': (255, 170, 20), 'green': (40, 230, 90)}
            for z, name in ((3.25, 'red'), (2.95, 'amber'), (2.65, 'green')):
                col = lit[name] if name == state else (55, 55, 58)
                rm.cylinder([-0.17, 0, z], [-0.15, 0, z], 0.1, col, emissive=name == state, n=12)
                rm.cylinder([0.15, 0, z], [0.17, 0, z], 0.1, col, emissive=name == state, n=12)
            m = type('Mesh', (), {})()
            m.tris, m.vn = np.array(rm._T), np.array(rm._N)
            m.cols, m.emis = np.array(rm._C, float), np.array(rm._E)
            del rm._T[:], rm._N[:], rm._C[:], rm._E[:]
            pole = place_splats(mesh_splats(m), base[0], base[1], math.atan2(d[1], d[0]), z0)
            self._decor[state] = np.concatenate([np.array(rows, np.float32), pole])
        return self._decor[state]

    def lane(self, s, direction):
        """A car's side offset at arc position s going this way: the road's offset there,
        keeping right."""
        road = self.road
        k = next((n for n, (a, b, _) in enumerate(road) if a <= s <= b), 0 if s < road[0][0] else len(road) - 1)
        off = road[k][2]
        # Blend across a change of offset over 4 m either side of the joint.
        for j, edge in ((k - 1, road[k][0]), (k + 1, road[k][1])):
            if 0 <= j < len(road) and abs(s - edge) < 4.0:
                w = 0.5 * (1 - abs(s - edge) / 4.0)
                off = (1 - w) * off + w * road[j][2]
        return off - LANE_HALF * direction

    def _at(self, s):
        k = np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.path) - 2)
        t = (s - self.cum[k]) / np.maximum(self.cum[k + 1] - self.cum[k], 1e-9)
        d = self.path[k + 1] - self.path[k]
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
        return self.path[k] + t[:, None] * (self.path[k + 1] - self.path[k]), d

    def _car_pose(self, i, s):
        """A car at route arc position s, in its lane (blended while it changes lanes)."""
        k = int(np.clip(np.searchsorted(self.cum, s) - 1, 0, len(self.path) - 2))
        t = (s - self.cum[k]) / max(self.cum[k + 1] - self.cum[k], 1e-9)
        w = (self.lane_mix[i] + 1) / 2                       # 1: the +1 lane, 0: the -1 lane
        pts = {d_: (1 - t) * L[k] + t * L[k + 1] for d_, L in self.lanes.items()}
        xy = w * pts[1.0] + (1 - w) * pts[-1.0]
        seg = w * (self.lanes[1.0][k + 1] - self.lanes[1.0][k]) + (1 - w) * (self.lanes[-1.0][k + 1] - self.lanes[-1.0][k])
        seg /= np.linalg.norm(seg) + 1e-12
        yaw = math.atan2(seg[1] * self.dir[i], seg[0] * self.dir[i])
        return xy, yaw, np.array([-seg[1], seg[0]])

    def _pose(self, i, s, dodge, direction):
        """Where agent i would stand at arc position s with this dodge: (xy, yaw, normal)."""
        if self.kind[i] == 4:
            return self._car_pose(i, s)
        p, d = self._at(np.array([s]))
        p, d = p[0], d[0]
        normal = np.array([-d[1], d[0]])
        off = self.side[i] + dodge
        while self.kind[i] != 4 and not self.cross[i] and abs(off) > 0.05 and self.grid.hits(*(p + off * normal), RADIUS):
            off *= 0.7                            # (cars keep to their lane: the road is off the grid)
        return p + off * normal, math.atan2(d[1] * direction, d[0] * direction), normal

    def _footprint(self, i, xy, yaw):
        h = np.array([math.cos(yaw), math.sin(yaw)])
        left = np.array([-h[1], h[0]])
        f = FOOT[int(self.kind[i])]
        return np.array([xy + a * h + b * left for a, b, _ in f]), np.array([r for _, _, r in f])

    def _overlap(self, i, xy, yaw, robot):
        """How deep agent i standing at (xy, yaw) is in anyone (or within 5 cm of the robot), m."""
        c, r = self._footprint(i, xy, yaw)
        worst = float((r + self.robot_radius + 0.05 - np.hypot(c[:, 0] - robot[0], c[:, 1] - robot[1])).max())
        for j in range(len(self.s)):
            if j == i:
                continue
            cj, rj = self._footprint(j, self.xy[j], self.yaw[j])
            d = np.hypot(c[:, None, 0] - cj[None, :, 0], c[:, None, 1] - cj[None, :, 1])
            worst = max(worst, float((r[:, None] + rj[None, :] - d).max()))
        return worst

    def _clear(self, i, xy, yaw, robot, slack=0.02):
        """Whether agent i standing at (xy, yaw) overlaps nobody by more than slack, and
        stays 5 cm clear of the robot."""
        c, r = self._footprint(i, xy, yaw)
        # the robot is kept 5 cm clear, so it is never inside anyone when it next moves
        if (np.hypot(c[:, 0] - robot[0], c[:, 1] - robot[1]) < r + self.robot_radius + 0.05).any():
            return False
        for j in range(len(self.s)):
            if j == i:
                continue
            cj, rj = self._footprint(j, self.xy[j], self.yaw[j])
            d = np.hypot(c[:, None, 0] - cj[None, :, 0], c[:, None, 1] - cj[None, :, 1])
            if (d < r[:, None] + rj[None, :] - slack).any():
                return False
        return True

    def _place(self):
        p, d = self._at(self.s)
        normal = np.stack([-d[:, 1], d[:, 0]], 1)
        self.normal = normal
        for i in range(len(self.s)):
            if self.kind[i] == 4:
                self.xy[i], self.yaw[i], self.normal[i] = self._car_pose(i, self.s[i])
                continue
            off = self.side[i] + self.dodge[i]    # shrink the side offset until the spot is free
            while self.kind[i] != 4 and not self.cross[i] and abs(off) > 0.05 and self.grid.hits(*(p[i] + off * normal[i]), RADIUS):
                off *= 0.7
            self.xy[i] = p[i] + off * normal[i]
            self.yaw[i] = math.atan2(d[i, 1] * self.dir[i], d[i, 0] * self.dir[i])

    def step(self, dt, robot_xy):
        """Walk (or ride) on, keeping clear of the robot and of each other.

        Whoever has someone just ahead (the robot, or another person or cyclist
        within their look-ahead and overlapping their lane) steps aside, away
        from them, and slows down (a wall on that side: after 3 s they try the
        other side). Behind someone going the same way who is still in the
        lane, they follow at that one's speed. Facing the robot or someone
        coming the other way, they stop only just before touching. No move
        ever makes two footprints (or one and the robot's) overlap by more
        than 2 cm: a step that would is not taken. Whoever cannot move at all
        for 8 s turns back, as people do in a lane too narrow to pass. Once
        past, they drift back to their own line. Cyclists look further ahead
        (3 m) and so overtake walkers."""
        robot = np.asarray(robot_xy, float)
        n = len(self.s)
        self.clock += dt
        light = self.light() if self.crossing_s is not None else None
        rad = np.array([max(abs(b) + r for _, b, r in FOOT[int(k)]) for k in self.kind])    # half-width
        half = np.array([max(abs(a) + r for a, _, r in FOOT[int(k)]) for k in self.kind]) - rad  # length beyond
        heading = np.stack([np.cos(self.yaw), np.sin(self.yaw)], 1)
        vel = heading * (self.v * self.moving)[:, None] if hasattr(self, 'moving') else heading * self.v[:, None]
        moving = np.ones(n)
        for i in range(n):
            bike = self.kind[i] == 1
            car = self.kind[i] == 4
            look = 12.0 if car else self.yield_m * (2.5 if bike else 1.0)
            # (ahead, lateral, needed clearance, same-direction speed or None for "wait")
            best = None
            rel = robot - self.xy[i]
            a, lat = rel @ heading[i] - half[i], rel @ self.normal[i]    # gap from my front
            need = rad[i] + 0.45
            if 0 < a < look + 0.45 and abs(lat) < need + 0.25:
                best = (a, lat, need, None, True)
            for j in range(n):
                if j == i:
                    continue
                rel = self.xy[j] - self.xy[i]
                a = rel @ heading[i] - half[i] - half[j] * abs(heading[j] @ heading[i])
                need = rad[i] + rad[j]
                if not -rad[j] < a < look + rad[j]:
                    continue
                lat = rel @ self.normal[i]
                if abs(lat) >= need + 0.25:
                    continue
                along = vel[j] @ heading[i]                 # their speed along my way
                follow = along if along > 0.2 else None     # same way: follow
                if best is None or a < best[0]:
                    best = (a, lat, need, follow, False)
            speed = self.v[i]
            hold = False                                         # a light or a crossing holds them
            old_dodge, old_side = self.dodge[i], self.side[i]
            if car:                                              # into its lane, 1 m/s across on a U-turn
                step = float(np.clip(self.dir[i] - self.lane_mix[i], -dt * 1.0 / LANE_HALF, dt * 1.0 / LANE_HALF))
                self.lane_mix[i] += step
                if step:
                    speed *= 0.3
            elif self.cross[i]:                                  # crossing the road and back
                side = self._road_side()
                target = (self.lane(self.crossing_s, 1.0) + LANE_HALF + side * (LANE_HALF + 1.2)
                          if self.cross[i] in (1, 2) else self.cross_home[i])
                step = float(np.clip(target - self.side[i], -1.2 * dt, 1.2 * dt))
                self.side[i] += step
                hold = True
                if self.cross[i] == 1 and not step:
                    self.cross[i], self.cross_wait[i] = 2, self.brng.uniform(4.0, 10.0)
                elif self.cross[i] == 2:
                    self.cross_wait[i] -= dt
                    if self.cross_wait[i] <= 0 and light == 'red':
                        self.cross[i] = 3
                elif self.cross[i] == 3 and not step:
                    self.cross[i] = 0
                    self.side_goal[i] = self.side[i]
            elif not bike:                                       # drift across now and then
                if self.brng.random() < dt / 90.0:
                    self.side_goal[i] = self.brng.uniform(-1.6, 1.6)
                step = float(np.clip(self.side_goal[i] - self.side[i], -0.5 * dt, 0.5 * dt))
                self.side[i] += step
                if step:
                    speed *= 0.8
            if self.crossing_s is not None and self.kind[i] == 0 and not self.cross[i]:
                near = abs(self.s[i] - self.crossing_s) < 0.4
                if near and not self.cross_tried[i] and light == 'red':
                    self.cross_tried[i] = True
                    if self.brng.random() < 0.6:                 # most of those who reach it on red cross
                        self.cross[i], self.cross_home[i] = 1, self.side[i]
                elif not near:
                    self.cross_tried[i] = False
            if car and light in ('amber', 'red'):                # wait at the line
                to_line = self.dir[i] * (self.crossing_s - self.s[i]) - 2.3 - 3.0
                if 0.0 <= to_line < 2.0:
                    hold = True
            pausing = False
            if self.pauses and self.kind[i] == 0:                # a walker stands still a while now and then
                if self.pause[i] > 0:
                    self.pause[i] -= dt
                    pausing = True
                elif self.brng.random() < dt / 60.0:
                    self.pause[i] = self.brng.uniform(8.0, 20.0)
            if best is not None:
                a, lat, need, follow, is_robot = best
                away = -1.0 if lat > 0 else 1.0
                if self.waited[i] > 3.0:
                    away = -away                                 # that side is blocked: try the other
                if self.waited[i] > 6.0:
                    self.waited[i] = 0.0
                self.dodge[i] = float(np.clip(self.dodge[i] + away * (0.0 if car else 1.2 if bike else 0.8) * dt,
                                              -2.0, 2.0))
                self.waited[i] += dt if a < need + 0.3 else 0.0
                in_way = abs(lat) < need + 0.05
                if in_way and follow is not None:
                    speed = min(self.v[i], follow)               # behind someone: follow
                elif in_way and a < need + 0.3:                  # about to touch: stop
                    speed = 0.0
                else:
                    speed = (0.35 if is_robot else 0.6) * self.v[i]
            if pausing or hold:
                speed = 0.0
            if best is None:
                self.waited[i] = 0.0
                self.dodge[i] -= np.sign(self.dodge[i]) * min(abs(self.dodge[i]), 0.3 * dt)
            # Never step into someone: try the move with the new dodge, then without moving
            # on, then without the new dodge; else stay put.
            s_new = min(max(self.s[i] + self.dir[i] * speed * dt, 0.0), self.length)
            moved = False
            # Already inside someone (placed there, or the robot drove up): any step that
            # gets less deep is allowed, so nobody stays stuck inside another.
            now = self._overlap(i, self.xy[i], self.yaw[i], robot)
            for s_try, dodge_try in ((s_new, self.dodge[i]), (self.s[i], self.dodge[i]), (s_new, old_dodge)):
                xy, yaw, normal = self._pose(i, s_try, dodge_try, self.dir[i])
                if self._clear(i, xy, yaw, robot) or (now > 0.02 and self._overlap(i, xy, yaw, robot) < now - 1e-4):
                    moved = s_try != self.s[i]
                    self.s[i], self.dodge[i] = s_try, dodge_try
                    self.xy[i], self.yaw[i], self.normal[i] = xy, yaw, normal
                    break
            else:
                self.dodge[i], self.side[i] = old_dodge, old_side
                self.side_goal[i] = old_side                     # that way is taken: stay on this line
            if moved or pausing or hold:
                self.stuck[i] = 0.0
                # one stride per 1.3 m on foot, one crank turn per 4.5 m on a bicycle
                self.phase[i] = (self.phase[i] + speed * dt / (2.1 if car else 4.5 if bike else 1.3)) % 1.0
            else:
                self.stuck[i] += dt
                speed = 0.0
            moving[i] = speed / max(self.v[i], 1e-6)
            lo, hi = self.road_s if car else (0.0, self.length)
            if (self.s[i] <= lo and self.dir[i] < 0) or (self.s[i] >= hi and self.dir[i] > 0) or self.stuck[i] > 8.0:
                self.dir[i] *= -1                            # the end of the route, or no way through: turn back
                self.stuck[i] = 0.0
                self.waited[i] = 0.0
        self.moving = moving

    def overlaps(self, robot_xy, robot_radius):
        """(person-person, person-robot) pairs whose footprints overlap now."""
        c, r = self.circles()
        owner = np.concatenate([[i] * len(FOOT[int(k)]) for i, k in enumerate(self.kind)]) if len(r) else np.zeros(0, int)
        d = np.hypot(c[:, None, 0] - c[None, :, 0], c[:, None, 1] - c[None, :, 1])
        hit = (d < r[:, None] + r[None, :]) & (owner[:, None] < owner[None, :])
        pp = len({(owner[a], owner[b]) for a, b in zip(*np.nonzero(hit))})
        dr = np.hypot(c[:, 0] - robot_xy[0], c[:, 1] - robot_xy[1])
        pr = len(set(owner[dr < r + robot_radius]))
        return pp, pr

    def circles(self):
        """Footprints as circles: (centres M x 2, radii M). A person is one circle, a
        bicycle three along its length."""
        cs, rs = [], []
        for i in range(len(self.xy)):
            c, r = self._footprint(i, self.xy[i], self.yaw[i])
            cs += list(c)
            rs += list(r)
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
        rows.append(self.decor(ground_z))
        return np.concatenate(rows)

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
