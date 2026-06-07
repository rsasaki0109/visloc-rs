#!/usr/bin/env python3
"""Production-style 3DGS trainer: gsplat official DefaultStrategy (gradient-driven
adaptive densify + opacity/scale prune) + degree-3 SH, starting from the SfM
points and GROWING. The minimal MCMC trainer distributes a fixed budget nearly
uniformly and prunes weakly, so edges stay soft and floaters linger; DefaultStrategy
clones/splits Gaussians where the image gradient demands detail and prunes
transparent floaters — the standard recipe that makes 3DGS crisp.

MODEL OUT_SPLAT [ITERS] [CMP_OUT] [CMP_VIEWS]
Reads <model>/undistorted/{images, sparse/...}. Images cached on CPU, moved to
GPU per step (lets us train at full-ish res without OOM as N grows).
"""
import sys, os, math, struct
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

MODEL = sys.argv[1]
OUT_SPLAT = sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
CMP_OUT = sys.argv[4] if len(sys.argv) > 4 else None
CMP_VIEWS = [int(x) for x in sys.argv[5].split(",")] if len(sys.argv) > 5 else [30, 64, 100]
dev = "cuda"
C0 = 0.28209479177387814
SH_DEG = 3
SH_INTERVAL = 1000


def _gw(ch, k=11, s=1.5):
    c = (torch.arange(k, device=dev) - k // 2).float()
    g = torch.exp(-(c ** 2) / (2 * s ** 2)); g = g / g.sum()
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(ch, 1, k, k).contiguous()


def ssim(a, b, win):
    pad = win.shape[-1] // 2; ch = a.shape[1]
    ma = F.conv2d(a, win, padding=pad, groups=ch); mb = F.conv2d(b, win, padding=pad, groups=ch)
    ma2, mb2, mab = ma * ma, mb * mb, ma * mb
    sa = F.conv2d(a * a, win, padding=pad, groups=ch) - ma2
    sb = F.conv2d(b * b, win, padding=pad, groups=ch) - mb2
    sab = F.conv2d(a * b, win, padding=pad, groups=ch) - mab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mab + c1) * (2 * sab + c2)) / ((ma2 + mb2 + c1) * (sa + sb + c2))).mean()


def Q2R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_colmap(base):
    sp = os.path.join(base, "undistorted", "sparse"); imdir = os.path.join(base, "undistorted", "images")
    for l in open(sp + "/cameras.txt"):
        if l.startswith("#"): continue
        v = l.split(); W, H = int(v[2]), int(v[3]); fx, fy, cx, cy = map(float, v[4:8]); break
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], np.float32)
    rows = [l.split() for l in open(sp + "/images.txt")
            if not l.startswith("#") and len(l.split()) >= 10 and l.strip().endswith(".png")]
    views = []
    for r in rows:
        q = list(map(float, r[1:5])); t = np.array(list(map(float, r[5:8])), np.float32)
        vm = np.eye(4, dtype=np.float32); vm[:3, :3] = Q2R(*q); vm[:3, 3] = t
        views.append((vm, r[9]))
    pts, cols = [], []
    for l in open(sp + "/points3D.txt"):
        if l.startswith("#"): continue
        v = l.split()
        if len(v) >= 7:
            pts.append([float(v[1]), float(v[2]), float(v[3])])
            cols.append([float(v[4]) / 255, float(v[5]) / 255, float(v[6]) / 255])
    return K, W, H, views, imdir, np.array(pts, np.float32), np.array(cols, np.float32)


def knn_scale(pts):
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(pts)
        d, _ = tree.query(pts, k=4)
        return np.clip(d[:, 1:].mean(1), 1e-4, None).astype(np.float32)
    except Exception:
        span = float(np.linalg.norm(pts.max(0) - pts.min(0)))
        return np.full(len(pts), span / (len(pts) ** (1 / 3)) * 0.5, np.float32)


def main():
    K, W, H, views, imdir, pts, pcol = load_colmap(MODEL)
    print(f"loaded {len(views)} views, {len(pts)} points, image {W}x{H}", flush=True)
    imgs_cpu = [torch.tensor(np.asarray(Image.open(os.path.join(imdir, n)).convert("RGB"), np.float32) / 255.0)
                for _, n in views]
    viewmats = torch.tensor(np.stack([v[0] for v in views]), device=dev)
    Ks = torch.tensor(K, device=dev)[None]
    cam_c = np.stack([(-v[0][:3, :3].T @ v[0][:3, 3]) for v in views])
    scene_scale = float(np.linalg.norm(cam_c - cam_c.mean(0), axis=1).max()) * 1.1

    N0 = len(pts)
    sc0 = knn_scale(pts)
    means = torch.nn.Parameter(torch.tensor(pts, device=dev))
    scales = torch.nn.Parameter(torch.log(torch.tensor(sc0, device=dev))[:, None].repeat(1, 3))
    quats = torch.nn.Parameter(torch.zeros(N0, 4, device=dev)); quats.data[:, 0] = 1.0
    opacities = torch.nn.Parameter(torch.logit(torch.full((N0,), 0.1, device=dev)))
    n_sh = (SH_DEG + 1) ** 2
    sh0 = torch.nn.Parameter(((torch.tensor(pcol, device=dev) - 0.5) / C0)[:, None, :])
    shN = torch.nn.Parameter(torch.zeros(N0, n_sh - 1, 3, device=dev))
    params = torch.nn.ParameterDict({"means": means, "scales": scales, "quats": quats,
                                     "opacities": opacities, "sh0": sh0, "shN": shN}).to(dev)
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20.0}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k], "name": k}], eps=1e-15)
                  for k in params}

    strategy = DefaultStrategy(verbose=True, refine_start_iter=500,
                               refine_stop_iter=int(ITERS * 0.5), reset_every=3000, refine_every=100)
    strategy.check_sanity(params, optimizers)
    state = strategy.initialize_state(scene_scale=scene_scale)
    win = _gw(3)
    rng = np.random.default_rng(0)

    def colors():
        return torch.cat([params["sh0"], params["shN"]], dim=1)

    for step in range(ITERS):
        i = int(rng.integers(0, len(views)))
        sh_deg = min(SH_DEG, step // SH_INTERVAL)
        render, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1), torch.exp(params["scales"]),
            torch.sigmoid(params["opacities"]), colors(), viewmats[i:i + 1], Ks, W, H,
            sh_degree=sh_deg, render_mode="RGB", packed=False)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = render[0].clamp(0, 1)
        gt = imgs_cpu[i].to(dev)
        l1 = (pred - gt).abs().mean()
        s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], win)
        loss = 0.8 * l1 + 0.2 * (1 - s)
        for o in optimizers.values():
            o.zero_grad(set_to_none=True)
        loss.backward()
        for o in optimizers.values():
            o.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)
        if step % 1000 == 0 or step == ITERS - 1:
            print(f"  [{step:6d}/{ITERS}] l1={l1.item():.4f} sh={sh_deg} N={params['means'].shape[0]}", flush=True)

    # spatial floater prune: drop Gaussians outside the SfM-point bounding box
    # (expanded by a margin). Sky/background has no SfM support, so MCMC/Default
    # floaters that drift above the building fall outside the hull.
    with torch.no_grad():
        lo = torch.tensor(pts.min(0) - 0.30 * (pts.max(0) - pts.min(0)), device=dev)
        hi = torch.tensor(pts.max(0) + 0.30 * (pts.max(0) - pts.min(0)), device=dev)
        m = params["means"]
        inside = ((m >= lo) & (m <= hi)).all(dim=1)
        n_before = m.shape[0]
        for k in params:
            params[k] = torch.nn.Parameter(params[k].data[inside])
        print(f"floater prune: {n_before} -> {params['means'].shape[0]} (dropped {n_before - int(inside.sum())} outside SfM hull)", flush=True)

    if CMP_OUT:
        # clamp requested comparison views to what was actually registered (a
        # sparser scene may register fewer images than the default indices assume)
        cmp_views = sorted({min(max(v, 0), len(views) - 1) for v in CMP_VIEWS})
        with torch.no_grad():
            tiles = []
            for vi in cmp_views:
                out, _, _ = rasterization(params["means"], F.normalize(params["quats"], dim=-1),
                    torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), colors(),
                    viewmats[vi:vi + 1], Ks, W, H, sh_degree=SH_DEG, render_mode="RGB", packed=False)
                ren = (out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
                gtv = (imgs_cpu[vi].numpy() * 255).astype(np.uint8)
                tiles.append(np.concatenate([gtv, ren], axis=1))
            grid = Image.fromarray(np.concatenate(tiles, axis=0))
            grid.resize((1600, int(grid.height * 1600 / grid.width))).save(CMP_OUT)
            print(f"wrote comparison {CMP_OUT} (left=GT, right=3DGS DefaultStrategy; views {cmp_views})", flush=True)
            # flythrough GIF at the (name-sorted) training poses
            order = sorted(range(len(views)), key=lambda k: views[k][1])
            frames = []
            for vi in order:
                out, _, _ = rasterization(params["means"], F.normalize(params["quats"], dim=-1),
                    torch.exp(params["scales"]), torch.sigmoid(params["opacities"]), colors(),
                    viewmats[vi:vi + 1], Ks, W, H, sh_degree=SH_DEG, render_mode="RGB", packed=False)
                frames.append(Image.fromarray((out[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)).resize((640, int(640 * H / W))))
            gif = CMP_OUT.replace(".png", "_flythrough.gif")
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=80, loop=0)
            print(f"wrote flythrough {gif}: {len(frames)} frames", flush=True)

    with torch.no_grad():
        m = params["means"].cpu().numpy(); sc = torch.exp(params["scales"]).cpu().numpy()
        op = torch.sigmoid(params["opacities"]).cpu().numpy()
        col = np.clip(params["sh0"][:, 0, :].cpu().numpy() * C0 + 0.5, 0, 1)
        qt = F.normalize(params["quats"], dim=-1).cpu().numpy()
    keep = op > 0.02
    m, sc, op, col, qt = m[keep], sc[keep], op[keep], col[keep], qt[keep]
    order2 = np.argsort(-(op * sc.prod(1)))
    buf = bytearray()
    for j in order2:
        buf += struct.pack("<3f", *m[j]); buf += struct.pack("<3f", *sc[j])
        buf += bytes([int(np.clip(col[j][c] * 255, 0, 255)) for c in range(3)] + [int(np.clip(op[j] * 255, 0, 255))])
        q = qt[j]; buf += bytes([int(np.clip(q[c] * 128 + 128, 0, 255)) for c in range(4)])
    open(OUT_SPLAT, "wb").write(buf)
    print(f"SPLAT {OUT_SPLAT} {len(order2)} gaussians ({len(buf)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
