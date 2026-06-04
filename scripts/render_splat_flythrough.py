#!/usr/bin/env python3
"""Render a flythrough GIF of a .splat along its COLMAP training camera poses.

Rendering at (smoothed, subsampled) training poses keeps every frame on the
manifold the splat was actually supervised on, so the flythrough stays sharp —
and it doubles as a replay of the camera trajectory visloc-rs estimates through
the scene. Uses the CUDA gsplat rasterizer (headless WebGL is unavailable here).

  python3 scripts/render_splat_flythrough.py <model_dir> <splat> <out.gif> [stride] [fps]
"""
import sys, os, struct, math
import numpy as np
import torch
from gsplat import rasterization
from PIL import Image

MODEL, SPLAT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
STRIDE = int(sys.argv[4]) if len(sys.argv) > 4 else 3
FPS = int(sys.argv[5]) if len(sys.argv) > 5 else 20
dev = "cuda"


def load_splat(p):
    b = np.fromfile(p, dtype=np.uint8).reshape(-1, 32)
    pos = b[:, 0:12].copy().view(np.float32).reshape(-1, 3)
    scl = b[:, 12:24].copy().view(np.float32).reshape(-1, 3)
    rgba = b[:, 24:28].astype(np.float32) / 255.0
    quat = (b[:, 28:32].astype(np.float32) - 128.0) / 128.0
    return pos, scl, rgba[:, :3], rgba[:, 3], quat


def read_cam(sp):
    for l in open(sp + "/cameras.txt"):
        if l.startswith("#"):
            continue
        v = l.split()
        return int(v[2]), int(v[3]), float(v[4]), float(v[5]), float(v[6]), float(v[7])


def Q2R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def main():
    sp = os.path.join(MODEL, "undistorted", "sparse")
    W, H, fx, fy, cx, cy = read_cam(sp)
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], device=dev)[None]
    rows = [l.split() for l in open(sp + "/images.txt")
            if not l.startswith("#") and l.strip().endswith(".png")]
    rows.sort(key=lambda r: r[9])  # chronological by timestamp filename
    rows = rows[::STRIDE]

    pos, scl, col, op, quat = load_splat(SPLAT)
    means = torch.tensor(pos, device=dev)
    scales = torch.tensor(scl, device=dev)
    quats = torch.nn.functional.normalize(torch.tensor(quat, device=dev), dim=-1)
    opac = torch.tensor(op, device=dev)
    colors = torch.tensor(col, device=dev)

    # render at the EXACT training poses — smoothing the centres pushes the
    # camera off the supervised manifold and reveals floaters.
    frames = []
    for r in rows:
        q = list(map(float, r[1:5])); t = np.array(list(map(float, r[5:8])), dtype=np.float32)
        vm = np.eye(4, dtype=np.float32); vm[:3, :3] = Q2R(*q); vm[:3, 3] = t
        out, _, _ = rasterization(means, quats, scales, opac, colors,
                                  torch.tensor(vm[None], device=dev), K, W, H, sh_degree=None)
        img = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img))
    # downscale for a lean GIF
    sc = 640.0 / W
    frames = [f.resize((640, int(H * sc))) for f in frames]
    frames[0].save(OUT, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"wrote {OUT}: {len(frames)} frames {640}x{int(H*sc)} ({os.path.getsize(OUT)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
