"""Post-hoc baseline: Sim(3)-align a model built without priors to the GPS, then score it.

usage: posthoc_alignment.py MODEL_DIR TRUTH_CSV

The GPS jumps are left out of the fit (an oracle advantage for this baseline).
"""
import sys
import numpy as np
model, truth_csv = sys.argv[1:3]
truth = {}; gps = {}; out = set()
for line in open(truth_csv).read().splitlines()[1:]:
    t = line.split(","); truth[t[0]] = np.array(list(map(float, t[1:4]))); gps[t[0]] = np.array(list(map(float, t[4:7])))
    if t[7] == "1": out.add(t[0])
est = {}
for line in open(model + "/images.txt"):
    t = line.split()
    if line.startswith("#") or len(t) < 10 or not t[9].lower().endswith(".jpg"): continue
    w, x, y, z = map(float, t[1:5]); tr = np.array(list(map(float, t[5:8])))
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    est[t[9]] = -R.T @ tr
fit = [n for n in sorted(est) if n in gps and n not in out]
src = np.array([est[n] for n in fit]); dst = np.array([gps[n] for n in fit])
ms, md = src.mean(0), dst.mean(0); X, Y = src - ms, dst - md
U, D, Vt = np.linalg.svd(Y.T @ X / len(fit)); Sg = np.eye(3)
if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
s = np.trace(np.diag(D) @ Sg) / (X ** 2).sum(1).mean(); Rf = U @ Sg @ Vt
err = np.array([np.linalg.norm(s * Rf @ (est[n] - ms) + md - truth[n]) for n in sorted(est) if n in truth])
print(f"post-hoc Sim3 to GPS: RMSE {np.sqrt((err**2).mean()):.3f} m, median {np.median(err):.3f} m, max {err.max():.3f} m ({len(err)} cameras)")
