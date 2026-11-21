"""Camera-centre errors of COLMAP models in a Cartesian truth frame.

usage: eval_metric.py TRUTH_CSV MODEL_DIR[:posthoc] ...
`:posthoc` first Sim(3)-aligns the model to the GPS columns of TRUTH_CSV.
Reports absolute error, error after the best translation (removes a common
GPS bias), and after the best Sim(3) (shape only).
"""
import sys
import numpy as np
truth, gps = {}, {}
for l in open(sys.argv[1]).read().splitlines()[1:]:
    t = l.split(","); truth[t[0]] = np.array(list(map(float, t[1:4]))); gps[t[0]] = np.array(list(map(float, t[4:7])))
def centres(d):
    out = {}
    for l in open(d + "/images.txt"):
        t = l.split()
        if l.startswith("#") or len(t) < 10 or not t[9].lower().endswith(".jpg"): continue
        w, x, y, z = map(float, t[1:5]); tr = np.array(list(map(float, t[5:8])))
        R = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        out[t[9]] = -R.T @ tr
    return out
def sim3(src, dst):
    ms, md = src.mean(0), dst.mean(0); X, Y = src - ms, dst - md
    U, D, Vt = np.linalg.svd(Y.T @ X / len(src)); S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0: S[2, 2] = -1
    s = np.trace(np.diag(D) @ S) / (X ** 2).sum(1).mean(); R = U @ S @ Vt
    return lambda p: s * (R @ (p - ms).T).T + md, s
def stats(e):
    return f"RMSE {np.sqrt((e**2).mean()):6.3f}  median {np.median(e):6.3f}  max {e.max():6.3f}"
print(f"{'run':<12} {'n':>3} | absolute                            | bias removed (translation)          | shape (Sim3)")
names_all = None
for arg in sys.argv[2:]:
    d, _, mode = arg.partition(":")
    est = centres(d)
    names = sorted(n for n in est if n in truth)
    E = np.array([est[n] for n in names]); T = np.array([truth[n] for n in names]); Gp = np.array([gps[n] for n in names])
    if mode == "posthoc":
        f, _ = sim3(E, Gp); E = f(E)
    a = np.linalg.norm(E - T, axis=1)
    b = np.linalg.norm(E - E.mean(0) + T.mean(0) - T, axis=1)
    f, sc = sim3(E, T); c = np.linalg.norm(f(E) - T, axis=1)
    label = d.rstrip("/").split("/")[-1] + (":ph" if mode else "")
    print(f"{label:<12} {len(names):>3} | {stats(a)} | {stats(b)} | {stats(c)} (scale {sc:.4f})")
# The GPS itself on the union of registered images of the first run.
est = centres(sys.argv[2].split(":")[0]); names = sorted(n for n in est if n in truth)
g = np.array([gps[n] for n in names]); t = np.array([truth[n] for n in names])
print(f"{'gnss':<12} {len(names):>3} | {stats(np.linalg.norm(g - t, axis=1))} | {stats(np.linalg.norm(g - g.mean(0) + t.mean(0) - t, axis=1))}")
