# Local helper for docs/dynamic_obstacle_localization.md (RNE example 113 runs).
"""RNE example 113's gt.tum (ns, base_link = IMU, RNE y-up world) -> EuRoC
mav0/state_groundtruth_estimate0/data.csv, and a seconds TUM gt_s.tum."""
import sys
from pathlib import Path

run = Path(sys.argv[1])
rows = [l.split() for l in (run / "gt.tum").read_text().splitlines() if l.strip() and not l.startswith("#")]
out = run / "mav0" / "state_groundtruth_estimate0"
out.mkdir(parents=True, exist_ok=True)
lines = ["#timestamp,p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z [],"
         "v_RS_R_x [m s^-1],v_RS_R_y [m s^-1],v_RS_R_z [m s^-1],b_w_RS_S_x [rad s^-1],b_w_RS_S_y [rad s^-1],"
         "b_w_RS_S_z [rad s^-1],b_a_RS_S_x [m s^-2],b_a_RS_S_y [m s^-2],b_a_RS_S_z [m s^-2]"]
tum = []
for ts, x, y, z, qx, qy, qz, qw in rows:
    lines.append(f"{ts},{x},{y},{z},{qw},{qx},{qy},{qz},0,0,0,0,0,0,0,0,0")
    tum.append(f"{int(ts) * 1e-9:.9f} {x} {y} {z} {qx} {qy} {qz} {qw}")
(out / "data.csv").write_text("\n".join(lines) + "\n")
(run / "gt_s.tum").write_text("\n".join(tum) + "\n")
print(len(rows), "poses")
