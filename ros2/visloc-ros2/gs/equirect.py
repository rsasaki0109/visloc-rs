#!/usr/bin/env python3
"""360-degree views of a splat: six 90-degree faces stitched into an equirectangular image.

    from equirect import Panorama
    pano = Panorama(server, 'scene.ply', face=512, width=1536)
    img = pano.render(x, y, z, yaw)          # H x W x 3 uint8, yaw at the centre column

Used for videos of `gs_sim.py`'s robot (docs/navigation.md): the robot's
localization uses one pinhole camera; this shows everything around it.
"""

import math
import subprocess

import numpy as np

def _face(fwd, down):
    """Camera axes (columns x right, y down, z forward) in the robot's yaw frame."""
    z, y = np.array(fwd, float), np.array(down, float)
    return np.stack([np.cross(y, z), y, z], 1)


# The robot's yaw frame: x forward, y left, z up.
_FACES = {
    'front': _face([1, 0, 0], [0, 0, -1]),
    'left': _face([0, 1, 0], [0, 0, -1]),
    'back': _face([-1, 0, 0], [0, 0, -1]),
    'right': _face([0, -1, 0], [0, 0, -1]),
    'up': _face([0, 0, 1], [1, 0, 0]),
    'down': _face([0, 0, -1], [-1, 0, 0]),
}


class Panorama:
    def __init__(self, server, ply, face=512, width=1536):
        self.face, self.w, self.h = face, width, width // 2
        f = face / 2
        self.p = subprocess.Popen([server, '--ply', ply, '--camera', f'PINHOLE {face} {face} {f} {f} {f} {f}'],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        # each output pixel's direction in the yaw frame, and which face sees it
        lon = (0.5 - (np.arange(self.w) + 0.5) / self.w) * 2 * math.pi       # +lon = left
        lat = (0.5 - (np.arange(self.h) + 0.5) / self.h) * math.pi
        lon, lat = np.meshgrid(lon, lat)
        d = np.stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)], -1)
        self.lookup = []
        best = np.full(lon.shape, -np.inf)
        which = np.zeros(lon.shape, int)
        for k, R in enumerate(_FACES.values()):
            z = d @ R[:, 2]
            better = z > best
            best[better], which[better] = z[better], k
        for k, R in enumerate(_FACES.values()):
            m = which == k
            c = d[m] @ R                                  # camera coords
            u = (c[:, 0] / c[:, 2] * f + f).clip(0, face - 1).astype(int)
            v = (c[:, 1] / c[:, 2] * f + f).clip(0, face - 1).astype(int)
            self.lookup.append((m, v, u))

    def _face(self, R_wc, centre):
        R_cw = R_wc.T
        t = -R_cw @ centre
        self.p.stdin.write((' '.join(f'{v:.7f}' for v in [*R_cw.reshape(-1), *t]) + '\n').encode())
        self.p.stdin.flush()
        n = self.face * self.face * 3
        return np.frombuffer(self.p.stdout.read(n), np.uint8).reshape(self.face, self.face, 3)

    def render(self, x, y, z, yaw):
        c, s = math.cos(yaw), math.sin(yaw)
        R_yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        out = np.zeros((self.h, self.w, 3), np.uint8)
        centre = np.array([x, y, z])
        for (m, v, u), R in zip(self.lookup, _FACES.values()):
            img = self._face(R_yaw @ R, centre)
            out[m] = img[v, u]
        return out

    def close(self):
        self.p.stdin.close()
        self.p.wait()
