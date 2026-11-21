"""Simulate GPS: write EXIF GPS fixes into copies of JPEGs from a COLMAP model's camera centres.

usage: make_exif_gps.py MODEL_DIR IMAGES_DIR OUT_IMAGES_DIR TRUTH_CSV

MODEL_DIR/images.txt gives the true camera centres (matched to the JPEGs by
lower-cased file name). They are moved by a fixed rotation and offset into an
East-North-Up frame at 47.3769 N 8.5417 E, given Gaussian noise (sigma 0.5 m
east/north, 1.0 m up, scaled by GPS_NOISE) and two 30 m jumps (the 4th and
16th image), converted to WGS84 and spliced into each JPEG as an APP1 Exif
segment (pixels untouched). TRUTH_SCALE scales the centres first (default:
the camera spread becomes 20 m; use 1 for a metric ground truth such as
ETH3D's). TRUTH_CSV gets the true and the GPS ENU positions.
"""
import math, os, random, sys
import numpy as np
from PIL import Image

model, src, dst, truth_csv = sys.argv[1:5]
A = 6378137.0; F = 1 / 298.257223563; E2 = F * (2 - F)

def lla_to_ecef(lat, lon, alt):
    la, lo = math.radians(lat), math.radians(lon)
    n = A / math.sqrt(1 - E2 * math.sin(la) ** 2)
    return np.array([(n + alt) * math.cos(la) * math.cos(lo), (n + alt) * math.cos(la) * math.sin(lo), (n * (1 - E2) + alt) * math.sin(la)])

def enu_rot(lat, lon):
    la, lo = math.radians(lat), math.radians(lon)
    return np.array([[-math.sin(lo), math.cos(lo), 0],
                     [-math.sin(la) * math.cos(lo), -math.sin(la) * math.sin(lo), math.cos(la)],
                     [math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)]])

def ecef_to_lla(x, y, z):
    lon = math.atan2(y, x); p = math.hypot(x, y); lat = math.atan2(z, p * (1 - E2))
    for _ in range(10):
        n = A / math.sqrt(1 - E2 * math.sin(lat) ** 2)
        alt = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1 - E2 * n / (n + alt)))
    return math.degrees(lat), math.degrees(lon), alt

LAT0, LON0, ALT0 = 47.3769, 8.5417, 408.0
O = lla_to_ecef(LAT0, LON0, ALT0); R = enu_rot(LAT0, LON0)
def enu_to_lla(e):
    return ecef_to_lla(*(O + R.T @ e))

centres = {}
for line in open(os.path.join(model, "images.txt")):
    t = line.split()
    if line.startswith("#") or len(t) < 10 or not t[9].lower().endswith((".jpg", ".png")):
        continue
    q = np.array(list(map(float, t[1:5]))); tr = np.array(list(map(float, t[5:8])))
    w, x, y, z = q
    Rw2c = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    centres[os.path.basename(t[9]).lower()] = -Rw2c.T @ tr
names = sorted(centres)
C = np.array([centres[n] for n in names])
extent = np.linalg.norm(C - C.mean(0), axis=1).max()
scale = float(os.environ['TRUTH_SCALE']) if 'TRUTH_SCALE' in os.environ else 20.0 / extent
a, b, c = 0.3, -0.5, 1.7
Rx = np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
Ry = np.array([[math.cos(b),0,math.sin(b)],[0,1,0],[-math.sin(b),0,math.cos(b)]])
Rz = np.array([[math.cos(c),-math.sin(c),0],[math.sin(c),math.cos(c),0],[0,0,1]])
Rs = Rz @ Ry @ Rx; ts = np.array([25.0, -40.0, 3.0])
truth = {n: scale * Rs @ centres[n] + ts for n in names}
rng = random.Random(7)
NOISE = float(os.environ.get("GPS_NOISE", "1"))
outliers = set(names[k] for k in (3, 15) if k < len(names)) if NOISE > 0 else set()
os.makedirs(dst, exist_ok=True)
with open(truth_csv, "w") as f:
    f.write("name,e,n,u,gps_e,gps_n,gps_u,outlier\n")
    for fname in sorted(os.listdir(src)):
        if not fname.lower().endswith(".jpg"):
            continue
        data = open(os.path.join(src, fname), "rb").read()
        if fname.lower() in truth:
            e = truth[fname.lower()]
            g = e + NOISE * np.array([rng.gauss(0, 0.5), rng.gauss(0, 0.5), rng.gauss(0, 1.0)])
            if fname.lower() in outliers:
                g = g + np.array([30.0, 0.0, 0.0])
            f.write(f"{fname},{e[0]},{e[1]},{e[2]},{g[0]},{g[1]},{g[2]},{int(fname.lower() in outliers)}\n")
            lat, lon, alt = enu_to_lla(g)
            def dms(v):
                v = abs(v); d = int(v); m = int((v - d) * 60); s = (v - d - m / 60) * 3600
                return (d, m, round(s, 6))
            ex = Image.Exif()
            ex[0x8825] = {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon), 5: b"\x00" if alt >= 0 else b"\x01", 6: abs(alt)}
            payload = ex.tobytes()
            assert payload.startswith(b"Exif\x00\x00")
            data = data[:2] + b"\xff\xe1" + (len(payload) + 2).to_bytes(2, "big") + payload + data[2:]
        open(os.path.join(dst, fname), "wb").write(data)
print(f"truth for {len(truth)} images, scale {scale:.4f}, outliers {sorted(outliers)}")
