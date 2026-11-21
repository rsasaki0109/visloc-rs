"""Check the gravel-mine camera convention: reproject ObjectPoints/GCPs through RefinedEOPs (expect ~0.29 px for R, alpha_sign=+1, half=0.5).

usage: gravel_model_check.py SUPP_DIR
"""
import itertools, math, sys
import numpy as np
d = sys.argv[1]
iop = {}
for line in open(f"{d}/RefinedIOPs.txt"):
    k, v = line.strip().split("=")
    for kk, vv in zip(k.split(","), v.split(",")): iop[kk] = float(vv)
eop = {int(l.split()[0]): np.array(list(map(float, l.split()[1:]))) for l in open(f"{d}/RefinedEOPs.txt") if l.strip()}
obj = {int(l.split()[0]): np.array(list(map(float, l.split()[1:4]))) for l in open(f"{d}/ObjectPoints.txt") if l.strip()}
for l in open(f"{d}/GCPs.txt"):
    if l.strip(): obj[int(l.split()[0])] = np.array(list(map(float, l.split()[1:4])))
tie = [l.split() for l in open(f"{d}/TiePoints.txt") if l.strip()]
tie = [(int(a), int(b), float(c), float(e)) for a, b, c, e in tie]
print("points", len(obj), "obs", len(tie), "with 3D", sum(1 for t in tie if t[1] in obj))
def R3(o, p, k):
    o, p, k = map(math.radians, (o, p, k))
    Rx = np.array([[1,0,0],[0,math.cos(o),-math.sin(o)],[0,math.sin(o),math.cos(o)]])
    Ry = np.array([[math.cos(p),0,math.sin(p)],[0,1,0],[-math.sin(p),0,math.cos(p)]])
    Rz = np.array([[math.cos(k),-math.sin(k),0],[math.sin(k),math.cos(k),0],[0,0,1]])
    return (Rx @ Ry @ Rz).T  # world -> camera (camera-to-world is Rx(w) Ry(p) Rz(k))
s, w, h, f = iop["s"], iop["w"], iop["h"], iop["f"]
def alpha(xb, yb):
    r2 = xb*xb + yb*yb; rad = iop["k1"]*r2 + iop["k2"]*r2**2 + iop["k3"]*r2**3
    ax = xb*rad + iop["p1"]*(r2 + 2*xb*xb) + 2*iop["p2"]*xb*yb + iop["b1"]*xb + iop["b2"]*yb
    ay = yb*rad + iop["p2"]*(r2 + 2*yb*yb) + 2*iop["p1"]*xb*yb
    return ax, ay
sample = [t for t in tie if t[1] in obj and t[0] in eop][:4000]
for transpose, sign, half in itertools.product([False], [1, -1, 0], [0.0, 0.5, -0.5]):
    res = []
    for im, pt, u, v in sample:
        e = eop[im]; R = R3(*e[3:]); R = R.T if transpose else R
        dcam = R @ (obj[pt] - e[:3])
        xb = s*(u + half) - s*w/2 - iop["xp"]; yb = -s*(v + half) + s*h/2 - iop["yp"]
        ax, ay = alpha(xb, yb)
        xu, yu = xb + sign*ax, yb + sign*ay
        xp_, yp_ = -f*dcam[0]/dcam[2], -f*dcam[1]/dcam[2]
        res.append(math.hypot(xu - xp_, yu - yp_) / s)
    res = np.array(res)
    print(f"R{'^T' if transpose else ''} alpha_sign={sign:+d} half={half}: median {np.median(res):.3f} px, p90 {np.percentile(res,90):.3f} px")
