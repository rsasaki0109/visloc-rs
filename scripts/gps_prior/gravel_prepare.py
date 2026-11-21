"""Gravel-mine UAV set: undistorted 1/SCALE pinhole JPEGs, GPS prior file, truth CSV.

usage: gravel_prepare.py SUPP_DIR RAW_ZIP OUT_DIR [SCALE]

The calibration (RefinedIOPs) is the dataset's Brown model in mm: for a
measured pixel (u, v), x = s(u+0.5) - s w/2 - xp, y = -s(v+0.5) + s h/2 - yp,
and the ideal photo coordinate is (x, y) + alpha(x, y). This convention
reprojects the dataset's own tie points at 0.29 px median. Output: images/
(grayscale JPEG), camera.txt (pinhole fx fy cx cy w h), gps_priors.txt
(DirectEOPs centres, Cartesian, minus OFFSET), truth.csv (RefinedEOPs
centres, same offset).
"""
import io, math, os, sys, zipfile
import numpy as np
from PIL import Image
supp, raw_zip, out = sys.argv[1:4]
scale = int(sys.argv[4]) if len(sys.argv) > 4 else 4
OFFSET = np.array([193000.0, 5025000.0, 0.0])
iop = {}
for line in open(f"{supp}/RefinedIOPs.txt"):
    k, v = line.strip().split("=")
    for kk, vv in zip(k.split(","), v.split(",")): iop[kk] = float(vv)
s, w, h, f = iop["s"], int(iop["w"]), int(iop["h"]), iop["f"]
W, H = w // scale, h // scale
F = f / (s * scale)
os.makedirs(f"{out}/images", exist_ok=True)
with open(f"{out}/camera.txt", "w") as c:
    c.write(f"{F:.4f} {F:.4f} {W/2:.4f} {H/2:.4f} {W} {H}\n")

def alpha(xb, yb):
    r2 = xb*xb + yb*yb; rad = iop["k1"]*r2 + iop["k2"]*r2**2 + iop["k3"]*r2**3
    ax = xb*rad + iop["p1"]*(r2 + 2*xb*xb) + 2*iop["p2"]*xb*yb + iop["b1"]*xb + iop["b2"]*yb
    ay = yb*rad + iop["p2"]*(r2 + 2*yb*yb) + 2*iop["p1"]*xb*yb
    return ax, ay

# Source pixel of every output pixel: ideal photo coords at the output grid,
# then invert x + alpha(x) = x_ideal by fixed-point iteration.
U, V = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
xi = (U - W/2) * s * scale
yi = -(V - H/2) * s * scale
xb, yb = xi.copy(), yi.copy()
for _ in range(20):
    ax, ay = alpha(xb, yb)
    xb, yb = xi - ax, yi - ay
ax, ay = alpha(xb, yb)
print(f"undistortion fixed point residual {max(np.abs(xb+ax-xi).max(), np.abs(yb+ay-yi).max())/s:.2e} px")
src_u = (xb + iop["xp"] + s*w/2) / s - 0.5
src_v = (s*h/2 - iop["yp"] - yb) / s - 0.5

def sample(img, u, v):
    """Box-filter the full-res image by `scale` first, then bilinear-sample it."""
    small = img.reshape(h // scale, scale, w // scale, scale).mean(axis=(1, 3))
    us = (u + 0.5) / scale - 0.5; vs = (v + 0.5) / scale - 0.5
    u0 = np.clip(np.floor(us).astype(int), 0, small.shape[1] - 2); v0 = np.clip(np.floor(vs).astype(int), 0, small.shape[0] - 2)
    du = np.clip(us - u0, 0, 1); dv = np.clip(vs - v0, 0, 1)
    a = small[v0, u0]; b = small[v0, u0 + 1]; c = small[v0 + 1, u0]; d = small[v0 + 1, u0 + 1]
    out_ = (a*(1-du)*(1-dv) + b*du*(1-dv) + c*(1-du)*dv + d*du*dv)
    outside = (us < -0.5) | (vs < -0.5) | (us > small.shape[1] - 0.5) | (vs > small.shape[0] - 0.5)
    out_[outside] = 0
    return out_

names = {}
with zipfile.ZipFile(raw_zip) as z:
    members = [m for m in z.namelist() if m.lower().endswith((".tif", ".tiff"))]
    print(f"{len(members)} TIFFs in the zip")
    for m in sorted(members):
        label = int("".join(ch for ch in os.path.splitext(os.path.basename(m))[0] if ch.isdigit()))
        name = f"img_{label:03d}.jpg"
        names[label] = name
        if os.path.exists(f"{out}/images/{name}"):
            continue
        img = Image.open(io.BytesIO(z.read(m))).convert("L")
        assert img.size == (w, h), (m, img.size)
        gray = np.asarray(img, dtype=np.float32)
        undist = sample(gray, src_u, src_v)
        Image.fromarray(np.clip(undist, 0, 255).astype(np.uint8)).save(f"{out}/images/{name}", quality=95)
        print("wrote", name, flush=True)

def centres(path):
    return {int(l.split()[0]): np.array(list(map(float, l.split()[1:4]))) - OFFSET for l in open(path) if l.strip()}
direct = centres(f"{supp}/DirectEOPs.txt"); refined = centres(f"{supp}/RefinedEOPs.txt")
with open(f"{out}/gps_priors.txt", "w") as g:
    g.write("# NAME X Y Z (DirectEOPs: onboard low-cost GNSS, NAD83/MTM7 minus (193000, 5025000, 0))\n")
    for label, c in sorted(direct.items()):
        if label in names: g.write(f"{names[label]} {c[0]:.3f} {c[1]:.3f} {c[2]:.3f}\n")
with open(f"{out}/truth.csv", "w") as t:
    t.write("name,e,n,u,gps_e,gps_n,gps_u,outlier\n")
    for label, c in sorted(refined.items()):
        if label in names and label in direct:
            g_ = direct[label]
            t.write(f"{names[label]},{c[0]},{c[1]},{c[2]},{g_[0]},{g_[1]},{g_[2]},0\n")
err = np.array([np.linalg.norm(direct[k] - refined[k]) for k in refined if k in direct])
dxyz = np.array([direct[k] - refined[k] for k in refined if k in direct])
print(f"onboard GNSS vs refined: RMSE {np.sqrt((err**2).mean()):.2f} m, median {np.median(err):.2f} m, max {err.max():.2f} m; mean offset {dxyz.mean(0).round(2)}, per-axis std {dxyz.std(0).round(2)}")
