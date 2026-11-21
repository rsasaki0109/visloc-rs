"""Score a model's camera centres against make_exif_gps.py's truth, with no alignment.

usage: eval_centres.py MODEL_DIR TRUTH_CSV [v]

The model is read in the ENU frame of MODEL_DIR/enu_origin.txt (written by
the mappers when GPS priors were used). Also prints the best-Sim(3) shape
error; a third argument lists every camera.
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
    if line.startswith("#") or len(t) < 10 or not t[9].lower().endswith(".jpg"):
        continue
    w, x, y, z = map(float, t[1:5]); tr = np.array(list(map(float, t[5:8])))
    R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    est[t[9]] = -R.T @ tr
# Truth/GPS are ENU at the generator's origin; the model's ENU origin is in enu_origin.txt.
import math, os
A = 6378137.0; F = 1 / 298.257223563; E2 = F * (2 - F)
def lla_to_ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo), (n + alt) * math.cos(la) * math.sin(lo), (n * (1 - E2) + alt) * math.sin(la)])
def enu_rot(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0], [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)], [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])
G = (47.3769, 8.5417, 408.0)
origin_file = os.path.join(model, "enu_origin.txt")
if os.path.exists(origin_file):
    M = tuple(map(float, [l for l in open(origin_file) if not l.startswith("#")][0].split()))
    to_model = lambda e: enu_rot(*M[:2]) @ (lla_to_ecef(*G) + enu_rot(*G[:2]).T @ e - lla_to_ecef(*M))
    truth = {k: to_model(v) for k, v in truth.items()}; gps = {k: to_model(v) for k, v in gps.items()}
common = sorted(set(est) & set(truth))
err = np.array([np.linalg.norm(est[n] - truth[n]) for n in common])
gerr = np.array([np.linalg.norm(gps[n] - truth[n]) for n in common if n not in out])
print(f"registered {len(est)}, with truth {len(common)}")
print(f"model vs truth (no alignment): RMSE {np.sqrt((err**2).mean()):.3f} m, median {np.median(err):.3f} m, max {err.max():.3f} m")
print(f"raw GPS (inliers) vs truth:    RMSE {np.sqrt((gerr**2).mean()):.3f} m")
for n in sorted(out & set(est)):
    print(f"  outlier {n}: model error {np.linalg.norm(est[n]-truth[n]):.3f} m (GPS error {np.linalg.norm(gps[n]-truth[n]):.1f} m)")

src = np.array([est[n] for n in common]); dst = np.array([truth[n] for n in common])
ms, md = src.mean(0), dst.mean(0); X, Y = src - ms, dst - md
U, D, Vt = np.linalg.svd(Y.T @ X / len(common)); Sg = np.eye(3)
if np.linalg.det(U) * np.linalg.det(Vt) < 0: Sg[2, 2] = -1
scale = np.trace(np.diag(D) @ Sg) / (X ** 2).sum(1).mean()
Rf = U @ Sg @ Vt
res = dst - (scale * (Rf @ X.T).T + md)
print(f"model shape vs truth (best Sim3): scale {scale:.4f}, RMSE {np.sqrt((res**2).sum(1).mean()):.3f} m")
if len(sys.argv) > 3:
    for n, r in zip(common, res):
        print(f"  {n}: shape residual {np.linalg.norm(r):.2f} m, direct error {np.linalg.norm(est[n]-truth[n]):.2f} m, gps err {np.linalg.norm(gps[n]-truth[n]):.2f}")
