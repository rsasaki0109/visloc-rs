"""Refine a reconstruction's alignment to an ETH3D ground-truth scan by ICP.

Usage: eth3d_icp_align.py RECON_PLY SCAN_MLP OUT_PLY [--max-dist 0.05]
                          [--min-dist 0.02] [--iterations 40] [--samples 300000]

An end-to-end reconstruction (SfM poses, then MVS) is moved onto the
ground-truth cameras by a Sim(3) on the camera centres
(align_colmap_model.py), which leaves millimetres of misregistration that
the ETH3D evaluator's 1 and 2 cm tolerances count as reconstruction error.
As Tanks and Temples' evaluation does, this refines that alignment by
rigid point-to-point ICP against the scan (the MLP's scans, each moved by
its matrix), so the score measures the reconstruction rather than the
registration. Only pairs closer than the distance threshold count; the
threshold shrinks from --max-dist to --min-dist over the iterations.

Writes RECON_PLY's points (and normals) moved by the refined transform to
OUT_PLY, same header, and prints the transform and the fraction of sampled
points within 1 and 2 cm of the scan before and after.
"""
import re
import sys

import numpy as np
from scipy.spatial import cKDTree

TYPES = {"float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
         "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
         "short": "<i2", "ushort": "<u2", "int": "<i4", "int32": "<i4",
         "uint": "<u4", "uint32": "<u4"}


def read_ply(path):
    """(header bytes, vertex structured array, trailing bytes) of a binary
    little-endian PLY whose first element is the vertices."""
    data = open(path, "rb").read()
    end = data.index(b"end_header") + len(b"end_header")
    while data[end:end + 1] in (b"\r", b"\n"):
        end += 1
    header = data[:end]
    lines = header.decode("ascii", "replace").splitlines()
    if not any("binary_little_endian" in l for l in lines):
        sys.exit(f"{path}: only binary little-endian PLY is supported")
    count, props, first = None, [], True
    for l in lines:
        t = l.split()
        if t[:1] == ["element"]:
            if count is not None:
                first = False
            if t[1] == "vertex" and first:
                count = int(t[2])
        elif t[:1] == ["property"] and count is not None and first:
            if t[1] == "list":
                sys.exit(f"{path}: list properties in the vertex element")
            props.append((t[2], TYPES[t[1]]))
    dtype = np.dtype(props)
    vertices = np.frombuffer(data, dtype=dtype, count=count, offset=end).copy()
    return header, vertices, data[end + count * dtype.itemsize:]


def xyz(vertices):
    return np.stack([vertices["x"], vertices["y"], vertices["z"]], 1).astype(np.float64)


def read_scan(mlp_path, samples, rng):
    """The MLP's scans in the common frame, at most `samples` points."""
    text = open(mlp_path).read()
    base = mlp_path.rsplit("/", 1)[0] + "/" if "/" in mlp_path else ""
    parts = []
    for name, matrix in re.findall(
            r'filename="([^"]+)".*?<MLMatrix44>(.*?)</MLMatrix44>', text, re.S):
        m = np.array([float(v) for v in matrix.split()]).reshape(4, 4)
        p = xyz(read_ply(base + name)[1])
        # Some scans (relief_2's) hold non-finite points.
        p = p[np.isfinite(p).all(1)]
        parts.append(p @ m[:3, :3].T + m[:3, 3])
    scan = np.concatenate(parts)
    if len(scan) > samples:
        scan = scan[rng.choice(len(scan), samples, replace=False)]
    return scan


def rigid(a, b):
    """R, t minimising |R a + t - b| (Kabsch)."""
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return r, cb - r @ ca


def main():
    args = sys.argv[1:]
    opts = {"--max-dist": 0.05, "--min-dist": 0.02, "--iterations": 40, "--samples": 300000}
    pos = []
    i = 0
    while i < len(args):
        if args[i] in opts:
            opts[args[i]] = type(opts[args[i]])(args[i + 1])
            i += 2
        else:
            pos.append(args[i])
            i += 1
    if len(pos) != 3:
        sys.exit(__doc__)
    recon_path, mlp_path, out_path = pos
    rng = np.random.default_rng(0)
    header, vertices, rest = read_ply(recon_path)
    points = xyz(vertices)
    finite = points[np.isfinite(points).all(1)]
    sample = finite[rng.choice(len(finite), min(opts["--samples"], len(finite)), replace=False)]
    scan = read_scan(mlp_path, 4 * opts["--samples"], rng)
    tree = cKDTree(scan)

    def within(p):
        d, _ = tree.query(p, k=1, distance_upper_bound=0.05)
        return (d < 0.01).mean(), (d < 0.02).mean()

    before = within(sample)
    r_total, t_total = np.eye(3), np.zeros(3)
    current = sample.copy()
    n = opts["--iterations"]
    for k in range(n):
        dist = opts["--max-dist"] + (opts["--min-dist"] - opts["--max-dist"]) * min(1.0, k / max(1, n // 2))
        d, idx = tree.query(current, k=1, distance_upper_bound=dist)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            break
        r, t = rigid(current[ok], scan[idx[ok]])
        current = current @ r.T + t
        r_total, t_total = r @ r_total, r @ t_total + t
        if np.linalg.norm(t) < 1e-6 and abs(np.trace(r) - 3) < 1e-12:
            break
    after = within(current)
    angle = np.degrees(np.arccos(np.clip((np.trace(r_total) - 1) / 2, -1, 1)))
    print(f"icp: rotation {angle:.4f} deg, translation {np.linalg.norm(t_total) * 1000:.2f} mm; "
          f"sampled points within 1 / 2 cm of the scan: {100 * before[0]:.1f}% / "
          f"{100 * before[1]:.1f}% -> {100 * after[0]:.1f}% / {100 * after[1]:.1f}%")

    moved = points @ r_total.T + t_total
    out = vertices.copy()
    for j, c in enumerate("xyz"):
        out[c] = moved[:, j].astype(out.dtype[c])
    if all(c in out.dtype.names for c in ("nx", "ny", "nz")):
        normals = np.stack([out["nx"], out["ny"], out["nz"]], 1).astype(np.float64) @ r_total.T
        for j, c in enumerate(("nx", "ny", "nz")):
            out[c] = normals[:, j].astype(out.dtype[c])
    with open(out_path, "wb") as f:
        f.write(header)
        f.write(out.tobytes())
        f.write(rest)


if __name__ == "__main__":
    main()
