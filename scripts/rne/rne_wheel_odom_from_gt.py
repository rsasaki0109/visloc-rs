# Local helper for docs/dynamic_obstacle_localization.md: RNE example 113 wheel odometry from gt.tum.
"""Port of RNE example 113's wheel_odom.rs, from an existing export's gt.tum
(for episodes rendered before the exporter wrote wheel_odom.tum)."""
import math
import sys
from pathlib import Path

MASK = (1 << 64) - 1
SCALE_ERROR, TURN_ERROR, NOISE_M, NOISE_RAD = 0.02, 0.03, 0.0005, 0.0005


class Wheel:
    def __init__(self, seed):
        self.x = self.y = self.h = 0.0
        self.prev = None
        self.rng = (seed | 1) & MASK

    def noise(self):
        r = self.rng
        r ^= r >> 12
        r ^= (r << 25) & MASK
        r ^= r >> 27
        self.rng = r
        value = ((r * 0x2545F4914F6CDD1D) & MASK) >> 11
        return value / float(1 << 52) - 1.0

    def step(self, x, z, yaw):
        if self.prev is not None:
            px, pz, pyaw = self.prev
            dx, dz = x - px, z - pz
            forward = dx * math.cos(pyaw) - dz * math.sin(pyaw)
            turn = yaw - pyaw
            while turn > math.pi:
                turn -= 2 * math.pi
            while turn < -math.pi:
                turn += 2 * math.pi
            moving = abs(forward) > 1e-9 or abs(turn) > 1e-9
            nm, nr = (NOISE_M * self.noise(), NOISE_RAD * self.noise()) if moving else (0.0, 0.0)
            d = forward * (1 + SCALE_ERROR) + nm
            t = turn * (1 + TURN_ERROR) + nr
            mid = self.h + 0.5 * t
            self.x += d * math.cos(mid)
            self.y += d * math.sin(mid)
            self.h += t
        self.prev = (x, z, yaw)
        return self.x, self.y, self.h


run = Path(sys.argv[1])
w = Wheel(11)
out = ["# timestamp tx ty tz qx qy qz qw"]
for line in (run / "gt.tum").read_text().splitlines():
    v = line.split()
    if len(v) != 8:
        continue
    ts = int(v[0])
    x, z = float(v[1]), float(v[3])
    qy, qw = float(v[5]), float(v[7])
    yaw = 2 * math.atan2(qy, qw)
    px, py, ph = w.step(x, z, yaw)
    out.append(f"{ts * 1e-9:.9f} {px:.6f} {py:.6f} 0 0 0 {math.sin(0.5 * ph):.9f} {math.cos(0.5 * ph):.9f}")
(run / "wheel_odom.tum").write_text("\n".join(out) + "\n")
print(len(out) - 1, "wheel poses ->", run / "wheel_odom.tum")
