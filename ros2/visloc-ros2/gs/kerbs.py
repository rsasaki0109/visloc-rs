"""Kerbs where the robot crosses a street: found in the splat's colours, a lowered road
between them, and the robot's pitch and speed over them (docs/navigation.md).

The splat's ground is too sparse and noisy (several cm) to show a 10 cm kerb
in its geometry, but its colours do: pavement here is warm (beige flags),
road is neutral grey (asphalt, setts). `find_kerbs` walks each crossing
(`lane_map.py`'s cuts, where the robot goes over a street) and takes the
longest run of road-coloured ground along it as the road; its ends are the
kerbs. `Kerbs` lowers the ground between them by `drop` (10 cm), so the
robot steps down off one kerb and up the other: its pitch and height come
from its front and rear wheels (`pose`), and the sim slows it to a crawl
while a wheel is on a kerb edge (`on_edge`).
"""

import math

import numpy as np


def find_kerbs(xyz, rgb, opacity, ground_z, crossings, half_width=3.0, drop=0.10, step=0.25, radius=0.5,
               warm=8.0, min_road=2.0):
    """[{'a': [x, y], 'b': [x, y], 'half_width', 'drop'}]: per crossing (x0, y0, x1, y1), the road
    between its kerbs a and b, or nothing where the colours show no road."""
    from scipy.spatial import cKDTree  # noqa: PLC0415
    gz = np.array([ground_z(x, y) for x, y in xyz[:, :2]])
    m = (np.abs(xyz[:, 2] - gz) < 0.35) & (opacity > 0.2)
    P, C = xyz[m, :2], rgb[m] * 255.0
    tree = cKDTree(P)
    out = []
    for x0, y0, x1, y1 in crossings:
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        L = float(np.linalg.norm(b - a))
        ts = np.arange(0.0, L + 1e-9, step)
        road = []
        for t in ts:
            idx = tree.query_ball_point(a + (b - a) * t / max(L, 1e-9), radius)
            if len(idx) < 5:
                road.append(None)              # no ground seen: decided by its neighbours
                continue
            c = np.median(C[idx], 0)
            road.append(bool(c[0] - c[2] < warm))     # neutral grey, not warm pavement
        # unknown samples take the nearest known one's class
        known = [i for i, r in enumerate(road) if r is not None]
        if not known:
            continue
        road = [road[i] if road[i] is not None else road[min(known, key=lambda k: abs(k - i))]
                for i in range(len(road))]
        best, run = None, None
        for i, r in enumerate(road + [False]):
            if r and run is None:
                run = i
            elif not r and run is not None:
                if best is None or i - run > best[1] - best[0]:
                    best = (run, i)
                run = None
        if best is None or (best[1] - best[0]) * step < min_road:
            continue
        ka = a + (b - a) * ts[best[0]] / max(L, 1e-9)
        kb = a + (b - a) * min(ts[best[1] - 1] + step, L) / max(L, 1e-9)
        out.append({'a': [round(float(v), 3) for v in ka], 'b': [round(float(v), 3) for v in kb],
                    'half_width': half_width, 'drop': drop})
    return out


class Kerbs:
    """The roads between kerbs (scene.json's 'kerbs'): how far the ground drops there, and the
    robot's height, pitch and kerb contact."""

    def __init__(self, kerbs, edge=0.06):
        self.k = [(np.array(r['a'], float), np.array(r['b'], float), float(r['half_width']), float(r['drop']))
                  for r in (kerbs or [])]
        self.edge = edge                       # the kerb's face: the drop eases over this many m

    def __bool__(self):
        return bool(self.k)

    def drop_at(self, x, y):
        """How far the ground at (x, y) is below the pavement (m)."""
        p = np.array([x, y], float)
        for a, b, hw, drop in self.k:
            d = b - a
            L = float(np.linalg.norm(d))
            u = d / max(L, 1e-9)
            t = float((p - a) @ u)
            lat = abs(float(u[0] * (p - a)[1] - u[1] * (p - a)[0]))
            if lat > hw:
                continue
            inside = min(t, L - t)              # m inside the road from the nearer kerb
            if inside > -self.edge / 2:
                return drop * min(1.0, (inside + self.edge / 2) / self.edge)
        return 0.0

    def pose(self, x, y, yaw, wheelbase=0.5):
        """(drop of the body's middle, pitch) from the ground under the front and rear wheels;
        pitch > 0 is nose up."""
        c, s = math.cos(yaw), math.sin(yaw)
        h = wheelbase / 2
        zf = -self.drop_at(x + c * h, y + s * h)
        zr = -self.drop_at(x - c * h, y - s * h)
        return (zf + zr) / 2, math.atan2(zf - zr, wheelbase)

    def on_edge(self, x, y, yaw, wheelbase=0.5, margin=0.12):
        """Whether a wheel is on (or about to reach) a kerb: the sim slows the robot to a crawl."""
        c, s = math.cos(yaw), math.sin(yaw)
        for off in (wheelbase / 2, -wheelbase / 2):
            px, py = x + c * off, y + s * off
            if abs(self.drop_at(px + c * margin, py + s * margin) - self.drop_at(px - c * margin, py - s * margin)) > 1e-3:
                return True
        return False
