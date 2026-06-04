#!/usr/bin/env python3
"""Minimal, correct 3D Gaussian Splatting trainer on a COLMAP model, using the
official gsplat MCMC strategy.

The gs-mapper project's hand-rolled densify/prune loop renders uniform gray fog
on EuRoC regardless of init points or pose set (verified: 3 different inits, both
GT- and visloc-posed models, all plateau at l1~0.24 with loss RISING during
densification). Pose convention and point projection were verified correct, so
the trainer itself is the fault. gsplat's MCMCStrategy is init-robust (samples a
fixed gaussian budget, relocates dead gaussians) and battle-tested, so we drive
rasterization directly with it.

Reads <model>/undistorted/{images, sparse/{cameras,images,points3D}.txt}, trains,
and writes a .splat (antimatter15 32-byte format) for the web viewer.
"""
import sys, os, math, struct
import numpy as np
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import MCMCStrategy

MODEL = sys.argv[1]
OUT_SPLAT = sys.argv[2]
ITERS = int(sys.argv[3]) if len(sys.argv) > 3 else 7000
CAP = int(sys.argv[4]) if len(sys.argv) > 4 else 300000
dev = "cuda"


def _gauss_win(ch, ksize=11, sigma=1.5, device="cuda"):
    c = (torch.arange(ksize, device=device) - ksize // 2).float()
    g = torch.exp(-(c ** 2) / (2 * sigma ** 2)); g = (g / g.sum())
    w = (g[:, None] @ g[None, :])[None, None]
    return w.expand(ch, 1, ksize, ksize).contiguous()


def ssim(a, b, win):
    # a,b: [1,3,H,W] in [0,1]
    pad = win.shape[-1] // 2; ch = a.shape[1]
    mu_a = F.conv2d(a, win, padding=pad, groups=ch)
    mu_b = F.conv2d(b, win, padding=pad, groups=ch)
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    sa = F.conv2d(a * a, win, padding=pad, groups=ch) - mu_a2
    sb = F.conv2d(b * b, win, padding=pad, groups=ch) - mu_b2
    sab = F.conv2d(a * b, win, padding=pad, groups=ch) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_ab + c1) * (2 * sab + c2)) / ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))
    return s.mean()


def Q2R(w, x, y, z):
    n = math.sqrt(w * w + x * x + y * y + z * z); w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_colmap(base):
    sp = os.path.join(base, "undistorted", "sparse")
    imdir = os.path.join(base, "undistorted", "images")
    for l in open(sp + "/cameras.txt"):
        if l.startswith("#"): continue
        v = l.split(); W, H = int(v[2]), int(v[3]); fx, fy, cx, cy = map(float, v[4:8]); break
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]], dtype=np.float32)
    rows = [l.split() for l in open(sp + "/images.txt")
            if not l.startswith("#") and len(l.split()) >= 10 and l.strip().endswith(".png")]
    views = []
    for r in rows:
        q = list(map(float, r[1:5])); t = np.array(list(map(float, r[5:8])), dtype=np.float32)
        vm = np.eye(4, dtype=np.float32); vm[:3, :3] = Q2R(*q); vm[:3, 3] = t
        views.append((vm, r[9]))
    pts = []
    for l in open(sp + "/points3D.txt"):
        if l.startswith("#"): continue
        v = l.split()
        if len(v) >= 4:
            pts.append([float(v[1]), float(v[2]), float(v[3])])
    return K, W, H, views, imdir, np.array(pts, dtype=np.float32)


def main():
    import matplotlib.image as mpimg
    K, W, H, views, imdir, pts = load_colmap(MODEL)
    print(f"loaded {len(views)} views, {len(pts)} colmap points, image {W}x{H}", flush=True)

    # load all images (grayscale -> RGB), cache on GPU
    imgs = []
    for vm, name in views:
        im = mpimg.imread(os.path.join(imdir, name))
        if im.ndim == 2: im = np.stack([im] * 3, -1)
        if im.shape[-1] == 4: im = im[..., :3]
        if im.dtype == np.uint8: im = im.astype(np.float32) / 255.0
        imgs.append(torch.tensor(im, dtype=torch.float32, device=dev))
    viewmats = torch.tensor(np.stack([v[0] for v in views]), device=dev)
    Ks = torch.tensor(K, device=dev)[None]

    # camera centres -> scene scale; init gaussians randomly in the camera+points hull
    cam_c = np.stack([(-v[0][:3, :3].T @ v[0][:3, 3]) for v in views])
    anchor = pts if len(pts) > 2000 else cam_c
    lo, hi = anchor.min(0), anchor.max(0)
    ctr = (lo + hi) / 2
    N0 = min(CAP, max(50000, len(pts)))
    rng = np.random.default_rng(0)
    if len(pts) > 5000:
        idx = rng.integers(0, len(pts), N0)
        init = pts[idx] + rng.normal(0, 0.02, (N0, 3)).astype(np.float32)
    else:
        init = (ctr + (rng.random((N0, 3)).astype(np.float32) - 0.5) * (hi - lo) * 1.2)
    # mean nearest-neighbour spacing for a sane initial scale
    span = float(np.linalg.norm(hi - lo))
    init_scale = max(span / (N0 ** (1 / 3)) * 0.5, 1e-3)

    means = torch.nn.Parameter(torch.tensor(init, device=dev))
    scales = torch.nn.Parameter(torch.full((N0, 3), math.log(init_scale), device=dev))
    quats = torch.nn.Parameter(torch.zeros(N0, 4, device=dev)); quats.data[:, 0] = 1.0
    opacities = torch.nn.Parameter(torch.logit(torch.full((N0,), 0.1, device=dev)))
    colors = torch.nn.Parameter(torch.full((N0, 3), 0.5, device=dev))
    params = torch.nn.ParameterDict({
        "means": means, "scales": scales, "quats": quats,
        "opacities": opacities, "colors": colors}).to(dev)

    lrs = {"means": 1.6e-4 * span, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "colors": 2.5e-3}
    optimizers = {k: torch.optim.Adam([{"params": params[k], "lr": lrs[k]}], eps=1e-15)
                  for k in params}

    strategy = MCMCStrategy(cap_max=CAP, refine_stop_iter=int(ITERS * 0.9), verbose=False)
    state = strategy.initialize_state()

    win = _gauss_win(3, device=dev)
    order = list(range(len(views)))
    for step in range(ITERS):
        i = order[rng.integers(0, len(order))]
        render, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            params["colors"], viewmats[i:i + 1], Ks, W, H, sh_degree=None,
            render_mode="RGB", packed=True)
        strategy.step_pre_backward(params, optimizers, state, step, info)
        pred = render[0].clamp(0, 1)
        gt = imgs[i]
        l1 = (pred - gt).abs().mean()
        # SSIM sharpens the reconstruction (penalises the smeary low-frequency
        # haze L1 alone tolerates). gsplat-standard 0.8*L1 + 0.2*(1-SSIM).
        s = ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None], win)
        loss = 0.8 * l1 + 0.2 * (1.0 - s)
        for opt in optimizers.values():
            opt.zero_grad(set_to_none=True)
        loss.backward()
        for opt in optimizers.values():
            opt.step()
        strategy.step_post_backward(params, optimizers, state, step, info,
                                    lr=lrs["means"])
        if step % 500 == 0 or step == ITERS - 1:
            print(f"  [{step:5d}/{ITERS}] l1={l1.item():.4f} N={params['means'].shape[0]}", flush=True)

    # export .splat (antimatter15 32-byte: pos[3]f32 scale[3]f32 rgba[4]u8 quat[4]u8)
    with torch.no_grad():
        m = params["means"].cpu().numpy()
        sc = torch.exp(params["scales"]).cpu().numpy()
        op = torch.sigmoid(params["opacities"]).cpu().numpy()
        col = params["colors"].clamp(0, 1).cpu().numpy()
        qt = F.normalize(params["quats"], dim=-1).cpu().numpy()
    keep = op > 0.02
    m, sc, op, col, qt = m[keep], sc[keep], op[keep], col[keep], qt[keep]
    order2 = np.argsort(-(op * sc.prod(1)))  # big, opaque first
    buf = bytearray()
    for j in order2:
        buf += struct.pack("<3f", *m[j])
        buf += struct.pack("<3f", *sc[j])
        buf += bytes([int(np.clip(col[j][0] * 255, 0, 255)),
                      int(np.clip(col[j][1] * 255, 0, 255)),
                      int(np.clip(col[j][2] * 255, 0, 255)),
                      int(np.clip(op[j] * 255, 0, 255))])
        q = qt[j]  # stored (w,x,y,z) -> bytes (w,x,y,z)*128+128
        buf += bytes([int(np.clip(q[k] * 128 + 128, 0, 255)) for k in range(4)])
    open(OUT_SPLAT, "wb").write(buf)
    print(f"SPLAT {OUT_SPLAT} {len(m)} gaussians ({len(buf)/1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
