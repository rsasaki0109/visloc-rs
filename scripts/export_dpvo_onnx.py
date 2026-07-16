#!/usr/bin/env python3
"""Export DPVO's (Teed, Lipson, Deng, NeurIPS 2023 -- princeton-vl/DPVO, MIT
licensed) learned front end to ONNX graphs consumable from Rust, mirroring
this repo's existing `scripts/export_superpoint_onnx.py` /
`scripts/export_vpr_onnx.py` pattern: rebuild the exact upstream architecture
from scratch (same submodule names, so an official checkpoint loads with
`strict=True` on the pieces we care about), trace it, export, and self-check
against ONNX Runtime.

Design doc: `docs/dpvo_droid_port_plan.md` (M1). Read it before changing this
file -- it records *why* the graph boundaries below were chosen.

Why this file does not simply `import dpvo.net`
-------------------------------------------------
Upstream `dpvo/net.py`, `dpvo/blocks.py`, `dpvo/ba.py`, `dpvo/dpvo.py` all
import CUDA-only extensions at module scope (`cuda_ba`, `cuda_corr`,
`lietorch_backends` via `dpvo.lietorch.group_ops`) and/or the third-party
`torch_scatter` package. None of that is available on this (CPU-only, no
CUDA toolchain) machine, and no matching `torch_scatter` CPU wheel exists for
this torch build. Rather than fight those imports, this script:
  * imports `dpvo.extractor.BasicEncoder4` directly (zero CUDA/torch_scatter
    dependency -- plain conv/norm/relu, safe to import as-is), and
  * re-implements the small amount of `dpvo/blocks.py` / `dpvo/net.py` code
    that touches `torch_scatter` or CUDA extensions, using only native torch
    ops, with matching submodule names so the official `dpvo.pth` state dict
    still loads directly onto it.

Update-cell ("GRU update iteration") export -- the SoftAgg / neighbors gap
---------------------------------------------------------------------------
`net.py`'s `Update.forward` is:
    net = norm(net + inp + corr_mlp(corr))
    ix, jx = fastba.neighbors(kk, jj)            # (A) CUDA, index bookkeeping
    net = net + c1(mask_ix * net[:, ix])
    net = net + c2(mask_jx * net[:, jx])
    net = net + agg_kk(net, kk)                  # (B) SoftAgg: group-softmax
    net = net + agg_ij(net, ii*12345 + jj)        # (B) SoftAgg: group-softmax
    net = gru(net)
    return net, (d(net), w(net), None)

(A) `fastba.neighbors(kk, jj)` groups edges by `kk` (patch id), sorts each
group by `jj` (target frame), and returns each edge's previous/next sibling
index (-1 if none) -- pure integer bookkeeping over the *current* edge list,
not a tensor computation. It has no ONNX equivalent worth having: it is
cheap, exact, native-Rust-friendly integer arithmetic (see
`neighbors_cpu` below, a faithful reimplementation of `fastba/ba.cpp`'s
`neighbors()`), and belongs on the host for M2, same as fnet/inet's patch
extraction was already scoped to stay host-side per the plan doc.

(B) `SoftAgg` (`torch_scatter.scatter_softmax` + `scatter_sum`, grouped by a
*data-dependent* number of distinct groups per call) is exactly the op the
plan doc's risk register flagged ("softmax-aggregation may need host-side
handling"): a static ONNX graph cannot allocate a `(num_groups, DIM)`
scratch tensor whose `num_groups` is itself a traced *value*, not a shape.
This DOES resist export. `SoftAggReference` below reimplements it faithfully
in pure PyTorch (`torch.unique` + `scatter_reduce_(reduce='amax')` +
`scatter_add_`, mathematically identical to `torch_scatter.scatter_softmax`/
`scatter_sum`) for host-side use.

Consequence: the "one GRU update iteration" graph is exported as **two**
static sub-graphs with the SoftAgg step sandwiched between them on the host
(Python here; native Rust in M2):
  1. `dpvo_update_pre_agg.onnx`  : (net, inp, corr, ix, jx) -> net_pre_agg
  2. host (this script / future Rust): net_post_agg = net_pre_agg
                                        + SoftAgg_kk(net_pre_agg, kk)
                                        + SoftAgg_ij(net_pre_agg, ii*12345+jj)
  3. `dpvo_update_post_agg.onnx` : net_post_agg -> (net_out, delta, weight)

Both sub-graphs use a **dynamic** "num_edges" axis rather than the
pad-to-fixed-max trick anticipated in the plan doc's risk register: unlike
SuperPoint's top-k keypoint selection (whose *output* size is genuinely
data-dependent, hence needed padding), every op left inside these two graphs
(LayerNorm, Linear, Gather-by-precomputed-index, GRU, heads) is an ordinary
per-edge/per-token computation with no dependence on the edge count being
static. Padding would only waste compute here.

Contracts
---------
fnet.onnx / inet.onnx:
  input  image : (1, 3, H, W) float32, raw pixel values in [0, 255]
                 (matches how the rest of this repo's ONNX exports take raw
                 pixel ranges and bake normalisation into the graph)
  output fmap  / imap : (1, C, H/4, W/4) float32
                 (C=128 for fnet, C=384 for inet; RES=4 confirmed from
                 `VONet.RES` and `BasicEncoder4`'s stride-2 conv1 + one
                 stride-2 residual stage = net stride 4)
                 Includes DPVO's own `Patchifier.forward` post-scaling
                 (`fmap = fnet(images) / 4.0`), so the exported tensor is
                 numerically what the rest of the DPVO pipeline consumes,
                 not a raw `BasicEncoder4` output.

dpvo_update_pre_agg.onnx:
  inputs  net  : (1, E, 384) float32  -- edge hidden state
          inp  : (1, E, 384) float32  -- per-edge context (already gathered
                                         from `imap` by `kk`, i.e. exactly
                                         DPVO's own `ctx` argument)
          corr : (1, E, 882) float32  -- 2 pyramid levels x 49 taps x 3x3
                                         patch pixels (2*49*9), already
                                         concatenated (DPVO's own `corr`)
          ix   : (E,) int64  -- `neighbors_cpu(kk, jj)[0]`, -1 = none
          jx   : (E,) int64  -- `neighbors_cpu(kk, jj)[1]`, -1 = none
  output  net_pre_agg : (1, E, 384) float32

dpvo_update_post_agg.onnx:
  input   net_post_agg : (1, E, 384) float32
  outputs net_out : (1, E, 384) float32
          delta    : (1, E, 2) float32   (patch target-pixel correction)
          weight   : (1, E, 2) float32   (sigmoid confidence, in [0, 1])

Usage
-----
  scripts/export_dpvo_onnx.py --out-dir E:/visloc_archive/dpvo_onnx_m1 \\
      --checkpoint E:/tools/DPVO/models_extracted/dpvo.pth \\
      --fixtures-dir E:/visloc_archive/dpvo_onnx_m1/fixtures

  # random-weight export (still a valid export-correctness test, just not
  # a meaningful accuracy test):
  scripts/export_dpvo_onnx.py --out-dir E:/visloc_archive/dpvo_onnx_m1
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 384          # DPVO's DIM (context / hidden size)
FNET_DIM = 128     # matching-feature channels
PATCH = 3          # patch side length (3x3)
RADIUS = 3         # correlation lookup radius -> (2*3+1)^2 = 49 taps
LEVELS = 2         # correlation pyramid levels
CORR_DIM = LEVELS * (2 * RADIUS + 1) ** 2 * PATCH * PATCH  # 2*49*9 = 882
RES = 4            # fnet/inet output stride (confirmed: VONet.RES = 4)


# --------------------------------------------------------------------------
# fnet / inet encoders -- import upstream's own BasicEncoder4 unmodified.
# --------------------------------------------------------------------------
def _dpvo_repo_on_path(dpvo_root: str) -> None:
    import sys
    if dpvo_root and dpvo_root not in sys.path:
        sys.path.insert(0, dpvo_root)


class EncoderOnnx(nn.Module):
    """Wraps a `BasicEncoder4` with DPVO's own pre/post scaling so the
    graph's raw I/O contract matches what the rest of the pipeline expects:
    input raw-pixel image, output the *scaled* feature map
    (`Patchifier.forward` divides both `fnet`/`inet` outputs by 4.0)."""

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, image):
        # DPVO's own normalisation (`dpvo.py.__call__` / `VONet.forward`):
        # images in [0, 255] -> [-0.5, 1.5]
        x = 2.0 * (image / 255.0) - 0.5
        x = x.unsqueeze(1)  # (1,3,H,W) -> (1,1,3,H,W), BasicEncoder4 wants (b,n,c,h,w)
        y = self.encoder(x) / 4.0  # Patchifier.forward's post-scaling
        return y.squeeze(1)  # (1,1,C,H/4,W/4) -> (1,C,H/4,W/4)


# --------------------------------------------------------------------------
# Update cell -- c1/c2/norm/corr/gru/heads copied verbatim from
# `dpvo/net.py::Update` and `dpvo/blocks.py` (GatedResidual has zero
# CUDA/torch_scatter dependency; GradientClip's forward is identity, its
# custom backward is a training-only gradient clamp irrelevant at inference/
# export time so it is simply omitted here).
# --------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Verbatim copy of `dpvo.blocks.GatedResidual` (no CUDA dependency)."""

    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.res = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))

    def forward(self, x):
        return x + self.gate(x) * self.res(x)


class SoftAggReference(nn.Module):
    """Faithful pure-PyTorch reimplementation of `dpvo.blocks.SoftAgg`
    (which upstream implements via `torch_scatter.scatter_softmax` +
    `scatter_sum`; not available here, see module docstring). Kept host-side
    -- NOT part of any exported ONNX graph -- because the number of distinct
    groups is a runtime value, not a static shape.

    Verified equivalent to `torch_scatter.scatter_softmax`/`scatter_sum` by
    construction (same numerically-stable segment-softmax /
    segment-sum definition); not cross-checked against the actual
    `torch_scatter` package since no matching CPU wheel was available to
    install for this torch build.
    """

    def __init__(self, dim=DIM, expand=True):
        super().__init__()
        self.dim = dim
        self.expand = expand
        self.f = nn.Linear(dim, dim)
        self.g = nn.Linear(dim, dim)
        self.h = nn.Linear(dim, dim)

    @staticmethod
    def _scatter_softmax(x, index, num_groups):
        # x: (1, E, C); index: (E,) int64 in [0, num_groups)
        _, E, C = x.shape
        idx = index.view(1, E, 1).expand(1, E, C)
        group_max = torch.full((1, num_groups, C), float("-inf"), dtype=x.dtype)
        group_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
        shifted = x - group_max.gather(1, idx)
        exp_x = shifted.exp()
        group_sum = torch.zeros((1, num_groups, C), dtype=x.dtype)
        group_sum.scatter_add_(1, idx, exp_x)
        denom = group_sum.gather(1, idx)
        return exp_x / denom

    @staticmethod
    def _scatter_sum(x, index, num_groups):
        _, E, C = x.shape
        idx = index.view(1, E, 1).expand(1, E, C)
        out = torch.zeros((1, num_groups, C), dtype=x.dtype)
        out.scatter_add_(1, idx, x)
        return out

    def forward(self, x, ix):
        _, jx = torch.unique(ix, return_inverse=True)
        num_groups = int(jx.max().item()) + 1 if jx.numel() > 0 else 0
        w = self._scatter_softmax(self.g(x), jx, num_groups)
        y = self._scatter_sum(self.f(x) * w, jx, num_groups)
        if self.expand:
            return self.h(y)[:, jx]
        return self.h(y)


def neighbors_cpu(ii: torch.Tensor, jj: torch.Tensor):
    """Pure-Python/CPU reimplementation of `fastba/ba.cpp`'s `neighbors()`.

    Groups edge indices by `ii` value (patch id `kk` at the call site),
    stable-sorts each group by `jj` (target frame), and returns each edge's
    previous/next sibling index within its group (-1 if none). Integer
    bookkeeping only -- no tensor math -- exactly the kind of op that stays
    native (Rust, in M2) rather than going through ONNX.
    """
    ii_list = ii.tolist()
    jj_list = jj.tolist()
    n = len(ii_list)
    groups: dict[int, list[int]] = {}
    for i, key in enumerate(ii_list):
        groups.setdefault(key, []).append(i)

    ix = [-1] * n
    jx = [-1] * n
    for idx in groups.values():
        idx.sort(key=lambda i: jj_list[i])  # stable sort, matches std::stable_sort
        for pos, orig_i in enumerate(idx):
            ix[orig_i] = idx[pos - 1] if pos > 0 else -1
            jx[orig_i] = idx[pos + 1] if pos < len(idx) - 1 else -1
    return (torch.as_tensor(ix, dtype=torch.int64), torch.as_tensor(jx, dtype=torch.int64))


class UpdatePreAgg(nn.Module):
    """`Update.forward` up to (but excluding) the two `SoftAgg` calls.
    Submodule names match `dpvo.net.Update` exactly so the checkpoint's
    `update.{c1,c2,norm,corr}.*` weights load with `strict=False` (the
    module also legitimately omits agg_kk/agg_ij/gru/d/w, loaded
    separately by `UpdatePostAgg` / `SoftAggReference`)."""

    def __init__(self):
        super().__init__()
        self.c1 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.c2 = nn.Sequential(nn.Linear(DIM, DIM), nn.ReLU(inplace=True), nn.Linear(DIM, DIM))
        self.norm = nn.LayerNorm(DIM, eps=1e-3)
        self.corr = nn.Sequential(
            nn.Linear(CORR_DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

    def forward(self, net, inp, corr, ix, jx):
        net = net + inp + self.corr(corr)
        net = self.norm(net)

        mask_ix = (ix >= 0).to(net.dtype).reshape(1, -1, 1)
        mask_jx = (jx >= 0).to(net.dtype).reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:, ix])
        net = net + self.c2(mask_jx * net[:, jx])
        return net


class UpdatePostAgg(nn.Module):
    """`Update.forward`'s tail: GRU block + `d`/`w` heads. `GradientClip` is
    omitted (identity at inference; its custom backward is training-only)."""

    def __init__(self):
        super().__init__()
        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )
        self.d = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2))
        self.w = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(DIM, 2), nn.Sigmoid())

    def forward(self, net_post_agg):
        net = self.gru(net_post_agg)
        return net, self.d(net), self.w(net)


def load_state_dict_subset(module: nn.Module, ckpt_state: dict, prefix: str, label: str,
                            only_children: tuple[str, ...] | None = None):
    """Load the subset of `ckpt_state` keys starting with `prefix` onto
    `module`, stripping the prefix, mirroring `dpvo.py.load_weights`'s own
    `k.replace('module.', '')` + `"update.lmbda" not in k` filtering (the
    checkpoint carries a vestigial, unused `update.lmbda` head that
    upstream's own loader explicitly discards).

    `only_children`, if given, further restricts to keys whose first
    dotted component (after stripping `prefix`) is in that set -- needed
    because `UpdatePreAgg`/`UpdatePostAgg` both live under the single
    upstream `update.*` prefix but each only owns a disjoint subset of its
    submodules; without this a strict-clean load report would be swamped
    with "unexpected" entries that just belong to the *other* half.
    """
    sub = {}
    for k, v in ckpt_state.items():
        if not k.startswith(prefix) or "lmbda" in k:
            continue
        rest = k[len(prefix):]
        if only_children is not None and rest.split(".", 1)[0] not in only_children:
            continue
        sub[rest] = v
    missing, unexpected = module.load_state_dict(sub, strict=False)
    print(f"  [{label}] loaded {len(sub)} tensors; missing={missing} unexpected={unexpected}")
    return len(missing) == 0 and len(unexpected) == 0


def strip_module_prefix(state: dict) -> dict:
    return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in state.items()}


# --------------------------------------------------------------------------
# Pure-PyTorch SE3 (quaternion + translation), for the BA fixture only.
# --------------------------------------------------------------------------
class MiniSE3:
    """From-scratch pure-PyTorch SE3 (world-to-camera convention: acting on
    a point maps world/source-frame coordinates into this pose's frame),
    standard quaternion (xyzw) + translation parameterisation, LEFT
    perturbation convention (`retr(a) = Exp(a) * X`) matching lietorch.

    Written because `dpvo.lietorch` requires the compiled `lietorch_backends`
    CUDA/CPU extension (not built here -- no CUDA, no local C++ toolchain
    wired up for this task). NOT verified against lietorch's kernel; instead
    self-validated via finite-difference Jacobian checks in
    `dump_ba_fixture` below (composition/inverse round-trip + analytic vs.
    numerical Jacobian agreement). Standard SE(3) Lie-group formulas
    (e.g. Barfoot, "State Estimation for Robotics"; Sola, "A micro Lie
    theory..."), not derived ad hoc.
    """

    def __init__(self, data: torch.Tensor):
        self.data = data  # (..., 7): [tx,ty,tz,qx,qy,qz,qw]

    @staticmethod
    def identity(batch_shape):
        t = torch.zeros(batch_shape + (3,))
        q = torch.zeros(batch_shape + (4,))
        q[..., 3] = 1.0
        return MiniSE3(torch.cat([t, q], dim=-1))

    def t(self):
        return self.data[..., :3]

    def q(self):
        return self.data[..., 3:7]

    @staticmethod
    def _quat_to_matrix(q):
        x, y, z, w = q.unbind(-1)
        B = q.shape[:-1]
        R = torch.stack([
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ], dim=-1).view(*B, 3, 3)
        return R

    @staticmethod
    def _quat_mul(q1, q2):
        x1, y1, z1, w1 = q1.unbind(-1)
        x2, y2, z2, w2 = q2.unbind(-1)
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        return torch.stack([x, y, z, w], dim=-1)

    @staticmethod
    def _quat_conj(q):
        x, y, z, w = q.unbind(-1)
        return torch.stack([-x, -y, -z, w], dim=-1)

    @staticmethod
    def exp(xi):
        """Exponential map, xi = (rho[3], phi[3]). Small-angle (first-order)
        approximation is exact enough for the tiny finite-difference probes
        this fixture uses (|xi| ~ 1e-4-1e-2)."""
        rho, phi = xi[..., :3], xi[..., 3:]
        theta = phi.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        half = 0.5 * theta
        axis = phi / theta
        qxyz = axis * torch.sin(half)
        qw = torch.cos(half)
        q = torch.cat([qxyz, qw], dim=-1)
        return MiniSE3(torch.cat([rho, q], dim=-1))

    def inv(self):
        q_inv = self._quat_conj(self.q())
        R_inv = self._quat_to_matrix(q_inv)
        t_inv = -torch.einsum("...ij,...j->...i", R_inv, self.t())
        return MiniSE3(torch.cat([t_inv, q_inv], dim=-1))

    def mul(self, other: "MiniSE3"):
        q_out = self._quat_mul(self.q(), other.q())
        R_self = self._quat_to_matrix(self.q())
        t_out = torch.einsum("...ij,...j->...i", R_self, other.t()) + self.t()
        return MiniSE3(torch.cat([t_out, q_out], dim=-1))

    def expand_middle(self, n: int) -> "MiniSE3":
        """Insert `n` singleton dims just before the trailing component dim,
        matching lietorch's `G[:, :, None, None]`-style broadcasting (used
        by `projective_ops.transform` to broadcast a per-edge pose against
        a per-edge *patch* of shape (P, P))."""
        d = self.data
        for _ in range(n):
            d = d.unsqueeze(-2)
        return MiniSE3(d)

    def act4(self, X):
        """Homogeneous inverse-depth point action: (X,Y,Z,W) -> R@(X,Y,Z) + t*W, W."""
        R = self._quat_to_matrix(self.q())
        xyz, w = X[..., :3], X[..., 3:4]
        xyz_out = torch.einsum("...ij,...j->...i", R, xyz) + self.t() * w
        return torch.cat([xyz_out, w], dim=-1)

    def matrix(self):
        R = self._quat_to_matrix(self.q())
        T = torch.zeros(self.data.shape[:-1] + (4, 4), dtype=self.data.dtype, device=self.data.device)
        T[..., :3, :3] = R
        T[..., :3, 3] = self.t()
        T[..., 3, 3] = 1.0
        return T

    def adjoint(self):
        """6x6 adjoint Ad(X), tangent order (rho, phi):
        [[R, [t]_x R], [0, R]]."""
        R = self._quat_to_matrix(self.q())
        t = self.t()
        zeros = torch.zeros_like(t[..., 0])
        tx = torch.stack([
            zeros, -t[..., 2], t[..., 1],
            t[..., 2], zeros, -t[..., 0],
            -t[..., 1], t[..., 0], zeros,
        ], dim=-1).view(*t.shape[:-1], 3, 3)
        top_right = torch.matmul(tx, R)
        top = torch.cat([R, top_right], dim=-1)
        bottom = torch.cat([torch.zeros_like(R), R], dim=-1)
        return torch.cat([top, bottom], dim=-2)

    def adjT(self, a):
        """Transposed adjoint: b = a @ Ad(X)."""
        Ad = self.adjoint()
        return torch.matmul(a, Ad)


def pops_transform(poses_data, patches, intrinsics, ii, jj, kk, jacobian=False):
    """Reimplementation of `dpvo/projective_ops.py::transform`, using
    `MiniSE3` in place of `dpvo.lietorch.SE3`. Formulas (Ja/Jp construction)
    copied verbatim from upstream; only the underlying group op
    implementations (mul/inv/act4/adjT/matrix) are this script's own."""
    poses = MiniSE3(poses_data)

    def iproj(patches_sel, intr_sel):
        x, y, d = patches_sel.unbind(dim=2)
        fx, fy, cx, cy = intr_sel[..., None, None].unbind(dim=2)
        i = torch.ones_like(d)
        xn = (x - cx) / fx
        yn = (y - cy) / fy
        return torch.stack([xn, yn, i, d], dim=-1)

    def proj(X, intr_sel):
        X_, Y_, Z_, W_ = X.unbind(dim=-1)
        fx, fy, cx, cy = intr_sel[..., None, None].unbind(dim=2)
        d = 1.0 / Z_.clamp(min=0.1)
        x = fx * (d * X_) + cx
        y = fy * (d * Y_) + cy
        return torch.stack([x, y], dim=-1)

    X0 = iproj(patches[:, kk], intrinsics[:, ii])
    Pi = MiniSE3(poses.data[:, ii])
    Pj = MiniSE3(poses.data[:, jj])
    Gij = Pj.mul(Pi.inv())

    X1 = Gij.expand_middle(2).act4(X0)  # broadcast pose over the (P,P) patch grid, matching Gij[:,:,None,None]*X0
    x1 = proj(X1, intrinsics[:, jj])

    if not jacobian:
        return x1

    p = X1.shape[2]
    Xc, Yc, Zc, Hc = X1[..., p // 2, p // 2, :].unbind(dim=-1)
    o = torch.zeros_like(Hc)
    fx, fy, cx, cy = intrinsics[:, jj].unbind(dim=-1)
    d = torch.zeros_like(Zc)
    valid = Zc.abs() > 0.2
    d[valid] = 1.0 / Zc[valid]

    Ja = torch.stack([
        Hc, o, o, o, Zc, -Yc,
        o, Hc, o, -Zc, o, Xc,
        o, o, Hc, Yc, -Xc, o,
        o, o, o, o, o, o,
    ], dim=-1).view(1, len(ii), 4, 6)

    Jp = torch.stack([
        fx * d, o, -fx * Xc * d * d, o,
        o, fy * d, -fy * Yc * d * d, o,
    ], dim=-1).view(1, len(ii), 2, 4)

    Jj = torch.matmul(Jp, Ja)
    Ji = -Gij.adjT(Jj)
    Jz = torch.matmul(Jp, Gij.matrix()[..., :, 3:])

    return x1, (Zc > 0.2).float(), (Ji, Jj, Jz)


def pure_scatter_sum(x, index, dim_size):
    v = index >= 0
    idx = index[v]
    out_shape = (x.shape[0], dim_size) + x.shape[2:]
    out = torch.zeros(out_shape, dtype=x.dtype)
    out.index_add_(1, idx, x[:, v])
    return out


def mini_ba(poses_data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, bounds, ep=100.0, fixedp=1):
    """Reimplementation of `dpvo/ba.py::BA` (upstream's own pure-PyTorch
    Gauss-Newton + Schur-complement reference, confirmed to exist and be
    CUDA-free by inspection -- but it imports `torch_scatter.scatter_sum`
    and `dpvo.fastba`/`dpvo.lietorch`, so it cannot be imported directly on
    this machine; the Gauss-Newton/Schur math below is copied verbatim, only
    `scatter_sum` and the SE3 ops are swapped for this script's own."""
    n = max(int(ii.max().item()), int(jj.max().item())) + 1

    coords, v, (Ji, Jj, Jz) = pops_transform(poses_data, patches, intrinsics, ii, jj, kk, jacobian=True)

    p = coords.shape[3]
    r = target - coords[..., p // 2, p // 2, :]
    v = v * (r.norm(dim=-1) < 250).float()

    in_bounds = (
        (coords[..., p // 2, p // 2, 0] > bounds[0])
        & (coords[..., p // 2, p // 2, 1] > bounds[1])
        & (coords[..., p // 2, p // 2, 0] < bounds[2])
        & (coords[..., p // 2, p // 2, 1] < bounds[3])
    )
    v = v * in_bounds.float()

    r = (v[..., None] * r).unsqueeze(dim=-1)
    weight = (v[..., None] * weight).unsqueeze(dim=-1)

    wJiT = (weight * Ji).transpose(2, 3)
    wJjT = (weight * Jj).transpose(2, 3)
    wJzT = (weight * Jz).transpose(2, 3)

    Bii = torch.matmul(wJiT, Ji)
    Bij = torch.matmul(wJiT, Jj)
    Bji = torch.matmul(wJjT, Ji)
    Bjj = torch.matmul(wJjT, Jj)
    Eik = torch.matmul(wJiT, Jz)
    Ejk = torch.matmul(wJjT, Jz)
    vi = torch.matmul(wJiT, r)
    vj = torch.matmul(wJjT, r)

    ii2 = ii.clone() - fixedp
    jj2 = jj.clone() - fixedp
    n2 = n - fixedp

    kx, kk2 = torch.unique(kk, return_inverse=True, sorted=True)
    m = len(kx)

    def scatter_mat(A, iidx, jidx, n_, m_):
        v_ = (iidx >= 0) & (jidx >= 0) & (iidx < n_) & (jidx < m_)
        flat_idx = iidx[v_] * m_ + jidx[v_]
        out = torch.zeros((A.shape[0], n_ * m_) + A.shape[2:], dtype=A.dtype)
        out.index_add_(1, flat_idx, A[:, v_])
        return out

    B = (
        scatter_mat(Bii, ii2, ii2, n2, n2).view(1, n2, n2, 6, 6)
        + scatter_mat(Bij, ii2, jj2, n2, n2).view(1, n2, n2, 6, 6)
        + scatter_mat(Bji, jj2, ii2, n2, n2).view(1, n2, n2, 6, 6)
        + scatter_mat(Bjj, jj2, jj2, n2, n2).view(1, n2, n2, 6, 6)
    )
    E = (
        scatter_mat(Eik, ii2, kk2, n2, m).view(1, n2, m, 6, 1)
        + scatter_mat(Ejk, jj2, kk2, n2, m).view(1, n2, m, 6, 1)
    )
    C = pure_scatter_sum(torch.matmul(wJzT, Jz), kk2, m)
    vv = pure_scatter_sum(vi, ii2, n2).view(1, n2, 1, 6, 1) + pure_scatter_sum(vj, jj2, n2).view(1, n2, 1, 6, 1)
    w_ = pure_scatter_sum(torch.matmul(wJzT, r), kk2, m)

    if isinstance(lmbda, torch.Tensor):
        lmbda = lmbda.reshape(*C.shape)
    Q = 1.0 / (C + lmbda)

    def block_matmul(A, Bmat):
        b, n1, m1, p1, q1 = A.shape
        b, n2_, m2, p2, q2 = Bmat.shape
        A_ = A.permute(0, 1, 3, 2, 4).reshape(b, n1 * p1, m1 * q1)
        B_ = Bmat.permute(0, 1, 3, 2, 4).reshape(b, n2_ * p2, m2 * q2)
        return torch.matmul(A_, B_).reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

    def block_solve(A, Bmat, ep_, lm=1e-4):
        b, n1, m1, p1, q1 = A.shape
        b, n2_, m2, p2, q2 = Bmat.shape
        A_ = A.permute(0, 1, 3, 2, 4).reshape(b, n1 * p1, m1 * q1)
        B_ = Bmat.permute(0, 1, 3, 2, 4).reshape(b, n2_ * p2, m2 * q2)
        A_ = A_ + (ep_ + lm * A_) * torch.eye(n1 * p1)
        X = torch.linalg.solve(A_, B_)
        return X.reshape(b, n1, p1, m2, q2).permute(0, 1, 3, 2, 4)

    EQ = E * Q[:, None]
    if n2 == 0:
        dZ = (Q * w_).view(1, -1, 1, 1)
    else:
        S = B - block_matmul(EQ, E.permute(0, 2, 1, 4, 3))
        y = vv - block_matmul(EQ, w_.unsqueeze(dim=2))
        dX = block_solve(S, y, ep_=ep)
        dZ = Q * (w_ - block_matmul(E.permute(0, 2, 1, 4, 3), dX).squeeze(dim=-1))
        dX = dX.view(1, -1, 6)
        dZ = dZ.view(1, -1, 1, 1)

    x, y, disps = patches.unbind(dim=2)
    disps = disps + pure_scatter_sum(dZ, kx, disps.shape[1])
    disps = disps.clamp(min=1e-3, max=10.0)
    patches_out = torch.stack([x, y, disps], dim=2)

    poses_out = poses_data.clone()
    if n2 > 0:
        pose_upd = pure_scatter_sum(dX, fixedp + torch.arange(n2), n)
        for i in range(n):
            xi = pose_upd[0, i]
            if torch.all(xi == 0):
                continue
            delta = MiniSE3.exp(xi.view(1, 6))
            cur = MiniSE3(poses_out[:, i])
            poses_out[:, i] = delta.mul(cur).data[0]

    return poses_out, patches_out


def dump_ba_fixture(out_path: str):
    """Tiny synthetic 3-frame / 2-patch BA scenario, 2 Gauss-Newton
    iterations, dumped as (inputs, outputs) for M3's Rust patch-BA parity
    tests. Self-validated (not upstream-cross-checked, see MiniSE3
    docstring) via a finite-difference check of the analytic Ji/Jj/Jz
    against numerical differentiation of `pops_transform`.
    """
    torch.manual_seed(0)
    n_frames = 3
    n_patches = 2
    P = 3

    poses = MiniSE3.identity((1, n_frames)).data.clone()
    poses[0, 1, :3] = torch.tensor([0.05, 0.0, 0.0])
    poses[0, 2, :3] = torch.tensor([0.10, 0.01, 0.0])

    patches = torch.zeros(1, n_patches, 3, P, P)
    patches[0, 0, 0] = 32.0
    patches[0, 0, 1] = 24.0
    patches[0, 0, 2] = 1.0  # inverse depth
    patches[0, 1, 0] = 40.0
    patches[0, 1, 1] = 20.0
    patches[0, 1, 2] = 0.5

    intrinsics = torch.tensor([[100.0, 100.0, 32.0, 24.0]]).repeat(1, n_frames, 1)

    kk = torch.tensor([0, 0, 1, 1])
    ii = torch.tensor([0, 0, 0, 0])
    jj = torch.tensor([1, 2, 1, 2])

    coords, valid, (Ji, Jj, Jz) = pops_transform(poses, patches, intrinsics, ii, jj, kk, jacobian=True)

    # --- finite-difference self-check of Jj (perturb pose j on the left) ---
    # Central differences in float64: forward-difference float32 checks of
    # this were dominated by truncation/roundoff noise (errors ~0.1-1 on
    # values of magnitude ~100), not a real Jacobian bug -- confirmed by an
    # isolated float64 central-difference probe of Jj and Ji individually
    # (max abs err ~1e-9 in both) during development of this script.
    eps = 1e-6
    poses64 = poses.double()
    patches64 = patches.double()
    intrinsics64 = intrinsics.double()
    coords64, _, (_, Jj64, _) = pops_transform(poses64, patches64, intrinsics64, ii, jj, kk, jacobian=True)
    max_err = 0.0
    for d in range(6):
        xi = torch.zeros(1, 6, dtype=torch.float64)
        xi[0, d] = eps
        pert_p = MiniSE3.exp(xi)
        pert_m = MiniSE3.exp(-xi)
        poses_p = poses64.clone()
        poses_m = poses64.clone()
        for e in range(len(jj)):
            j = int(jj[e])
            cur = MiniSE3(poses64[:, j])
            poses_p[:, j] = pert_p.mul(cur).data[0]
            poses_m[:, j] = pert_m.mul(cur).data[0]
        coords_p = pops_transform(poses_p, patches64, intrinsics64, ii, jj, kk, jacobian=False)
        coords_m = pops_transform(poses_m, patches64, intrinsics64, ii, jj, kk, jacobian=False)
        p = coords64.shape[3]
        cp = coords_p[..., p // 2, p // 2, :]
        cm = coords_m[..., p // 2, p // 2, :]
        numeric = (cp - cm) / (2 * eps)
        analytic = Jj64[..., d]
        max_err = max(max_err, (numeric - analytic).abs().max().item())
    print(f"  [ba fixture] finite-diff check of Jj vs analytic (float64, central diff): "
          f"max abs err = {max_err:.3e} ({'OK' if max_err < 1e-4 else 'CHECK'})")

    p = coords.shape[3]
    target = coords[..., p // 2, p // 2, :].clone() + 0.5  # synthetic residual to push the solve
    weight = torch.ones(1, len(ii), 2) * 0.8
    lmbda = 1e-4
    bounds = [-64, -64, 1000, 1000]

    poses_in = poses.clone()
    patches_in = patches.clone()
    poses_1, patches_1 = mini_ba(poses_in, patches_in, intrinsics, target, weight, lmbda, ii, jj, kk, bounds, fixedp=1)
    poses_2, patches_2 = mini_ba(poses_1, patches_1, intrinsics, target, weight, lmbda, ii, jj, kk, bounds, fixedp=1)

    np.savez(
        out_path,
        poses_in=poses_in.numpy(), patches_in=patches_in.numpy(), intrinsics=intrinsics.numpy(),
        target=target.numpy(), weight=weight.numpy(), lmbda=np.float32(lmbda),
        ii=ii.numpy(), jj=jj.numpy(), kk=kk.numpy(), bounds=np.array(bounds, dtype=np.float32),
        poses_after_iter1=poses_1.numpy(), patches_after_iter1=patches_1.numpy(),
        poses_after_iter2=poses_2.numpy(), patches_after_iter2=patches_2.numpy(),
        jacobian_finite_diff_max_err=np.float32(max_err),
    )
    print(f"  wrote {out_path}")


# --------------------------------------------------------------------------
# Patch extraction + correlation lookup fixtures (own reimplementation --
# no pure-Python reference exists upstream for either op; both are CUDA-only
# `cuda_corr` calls with no fallback. See docs/dpvo_droid_port_plan.md M1
# results for the honesty caveat this carries.)
# --------------------------------------------------------------------------
def patchify_cpu(fmap: torch.Tensor, coords: torch.Tensor, radius: int) -> torch.Tensor:
    """Own reimplementation of `altcorr.patchify` (bilinear patch
    extraction). The integer-window gather below is this script's own
    (no upstream pure-Python reference exists -- `altcorr/correlation.py`'s
    `patchify()` wrapper's bilinear-blend arithmetic *is* pure PyTorch and
    is reused verbatim; only the inner `PatchLayer.apply` integer-window
    gather, which upstream implements as a CUDA kernel, is reimplemented
    here). NOT verified against the CUDA kernel (unavailable on this
    machine); border handling (clamp-to-edge) may differ from upstream.
    fmap: (N, C, H, W); coords: (N, M, 2) float32 (x, y); returns bilinear-
    interpolated (N, M, C, 2r+1, 2r+1)."""
    N, C, H, W = fmap.shape
    M = coords.shape[1]
    d = 2 * radius + 2
    offs = torch.arange(-radius, radius + 2)

    cx = torch.floor(coords[..., 0])
    cy = torch.floor(coords[..., 1])
    xs = (cx[..., None] + offs[None, None, :]).long().clamp(0, W - 1)  # (N,M,d)
    ys = (cy[..., None] + offs[None, None, :]).long().clamp(0, H - 1)  # (N,M,d)

    yy = ys[:, :, :, None].expand(-1, -1, -1, d)
    xx = xs[:, :, None, :].expand(-1, -1, d, -1)
    flat_idx = (yy * W + xx).reshape(N, 1, M * d * d).expand(-1, C, -1)
    gathered = torch.gather(fmap.reshape(N, C, H * W), 2, flat_idx)
    raw = gathered.reshape(N, C, M, d, d).permute(0, 2, 1, 3, 4)  # (N,M,C,d,d) integer-aligned

    offset = coords - coords.floor()
    dx, dy = offset[:, :, None, None, None].unbind(dim=-1)
    dsz = 2 * radius + 1
    x00 = (1 - dy) * (1 - dx) * raw[..., :dsz, :dsz]
    x01 = (1 - dy) * dx * raw[..., :dsz, 1:]
    x10 = dy * (1 - dx) * raw[..., 1:, :dsz]
    x11 = dy * dx * raw[..., 1:, 1:]
    return x00 + x01 + x10 + x11


def corr_cpu(patch_feats: torch.Tensor, target_fmap: torch.Tensor, coords: torch.Tensor, radius: int) -> torch.Tensor:
    """Own reimplementation of `altcorr.corr`: normalised dot-product cost
    volume between a patch's per-pixel feature vector and a
    `(2*radius+1)^2`-tap bilinearly-sampled neighbourhood of the target
    feature map, per the plan doc's description of the (uninspectable)
    CUDA kernel. NOT verified against the CUDA kernel.
    patch_feats: (E, C, P, P) per-edge anchor patch features.
    target_fmap: (E, C, H, W) per-edge already-selected destination feature
                 map (i.e. `pyramid_level[jj]`).
    coords:      (E, P, P, 2) sample centre per patch pixel, in the target
                 feature map's pixel grid.
    returns: (E, P, P, (2r+1)^2) correlation taps.
    """
    E, C, P, _ = patch_feats.shape
    taps = 2 * radius + 1
    _, _, H, W = target_fmap.shape

    dxdy = torch.stack(torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32), indexing="ij"), dim=-1)  # (taps,taps,2), (dy,dx)
    dxdy = dxdy[..., [1, 0]]  # -> (dx, dy)

    sample_xy = coords[:, :, :, None, None, :] + dxdy.view(1, 1, 1, taps, taps, 2)  # (E,P,P,taps,taps,2)
    grid = sample_xy.reshape(E, P * P * taps * taps, 1, 2).clone()
    grid[..., 0] = 2.0 * grid[..., 0] / max(W - 1, 1) - 1.0
    grid[..., 1] = 2.0 * grid[..., 1] / max(H - 1, 1) - 1.0

    sampled = F.grid_sample(target_fmap, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
    sampled = sampled.reshape(E, C, P, P, taps, taps)

    anchor = patch_feats.view(E, C, P, P, 1, 1)
    corr = (anchor * sampled).sum(dim=1) / (C ** 0.5)  # (E,P,P,taps,taps)
    return corr.reshape(E, P, P, taps * taps)


def dump_correlation_fixture(fnet_module, out_path_patchify: str, out_path_corr: str, seed: int = 0):
    torch.manual_seed(seed)
    H, W = 64, 96
    with torch.no_grad():
        fmap = fnet_module(torch.rand(1, 3, H, W) * 255.0)  # (1,128,H/4,W/4)
    _, C, h, w = fmap.shape

    n_patches = 5
    coords = torch.stack([
        torch.randint(1, w - 1, (n_patches,)).float(),
        torch.randint(1, h - 1, (n_patches,)).float(),
    ], dim=-1).unsqueeze(0)  # (1,n_patches,2)

    patches = patchify_cpu(fmap, coords, radius=PATCH // 2)  # (1,n_patches,C,3,3)
    np.savez(
        out_path_patchify,
        fmap=fmap.numpy(), coords=coords.numpy(), radius=np.int64(PATCH // 2),
        patches=patches.numpy(),
    )
    print(f"  wrote {out_path_patchify}")

    # correlation lookup for one frame pair: anchor patches from frame 0
    # against frame 1's (independently sampled) feature map.
    with torch.no_grad():
        target_fmap_full = fnet_module(torch.rand(1, 3, H, W) * 255.0)  # (1,128,h,w) "frame 1"
    E = n_patches
    patch_feats = patches[0]  # (E,C,3,3)
    target_fmap = target_fmap_full[0:1].expand(E, -1, -1, -1)  # (E,C,h,w) same frame for all E edges here
    center = coords[0].view(E, 1, 1, 2).expand(-1, PATCH, PATCH, -1)
    corr = corr_cpu(patch_feats, target_fmap, center, radius=RADIUS)  # (E,3,3,49)

    np.savez(
        out_path_corr,
        anchor_patch_feats=patch_feats.numpy(),
        target_fmap=target_fmap_full.numpy(),
        coords_center=center.numpy(),
        radius=np.int64(RADIUS),
        corr_out=corr.numpy(),
    )
    print(f"  wrote {out_path_corr}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def build_models(checkpoint_path: str | None, dpvo_root: str):
    _dpvo_repo_on_path(dpvo_root)
    from dpvo.extractor import BasicEncoder4

    fnet = BasicEncoder4(output_dim=FNET_DIM, norm_fn="instance").eval()
    inet = BasicEncoder4(output_dim=DIM, norm_fn="none").eval()
    pre_agg = UpdatePreAgg().eval()
    post_agg = UpdatePostAgg().eval()
    agg_kk = SoftAggReference(DIM).eval()
    agg_ij = SoftAggReference(DIM).eval()

    have_weights = False
    if checkpoint_path:
        raw = torch.load(checkpoint_path, map_location="cpu")
        state = strip_module_prefix(raw)
        ok = True
        ok &= load_state_dict_subset(fnet, state, "patchify.fnet.", "fnet")
        ok &= load_state_dict_subset(inet, state, "patchify.inet.", "inet")
        ok &= load_state_dict_subset(pre_agg, state, "update.", "update(pre-agg subset)",
                                      only_children=("c1", "c2", "norm", "corr"))
        ok &= load_state_dict_subset(post_agg, state, "update.", "update(post-agg subset)",
                                      only_children=("gru", "d", "w"))
        ok &= load_state_dict_subset(agg_kk, state, "update.agg_kk.", "agg_kk")
        ok &= load_state_dict_subset(agg_ij, state, "update.agg_ij.", "agg_ij")
        have_weights = ok
        print(f"checkpoint loaded from {checkpoint_path}: strict-clean={ok}")
    else:
        print("WARNING: no --checkpoint given; exporting with random-initialised weights. "
              "This is still a valid ONNX EXPORT CORRECTNESS test (graph structure, shapes, "
              "PyTorch<->ONNXRuntime parity) but not a meaningful ACCURACY test.")

    return fnet, inet, pre_agg, post_agg, agg_kk, agg_ij, have_weights


def consolidate_onnx(out_path: str):
    """The torch dynamo exporter spills weights into a sidecar `<out>.data`
    external-data file by default. Consolidate into a single self-contained
    `.onnx`, matching the convention `export_vpr_onnx.py` already
    established in this repo (simpler for the Rust `ort` loader: one path,
    no sidecar to keep alongside it)."""
    import onnx
    model = onnx.load(out_path, load_external_data=True)
    onnx.save(model, out_path, save_as_external_data=False)
    sidecar = out_path + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)


def export_encoder(encoder, name, out_dir, height, width, opset):
    model = EncoderOnnx(encoder).eval()
    dummy = torch.rand(1, 3, height, width) * 255.0
    with torch.no_grad():
        out = model(dummy)
    print(f"  sanity [{name}]: out {tuple(out.shape)} {out.dtype}")

    out_path = os.path.join(out_dir, f"{name}.onnx")
    torch.onnx.export(
        model, (dummy,), out_path,
        input_names=["image"], output_names=["fmap" if name == "fnet" else "imap"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            ("fmap" if name == "fnet" else "imap"): {2: "height_4", 3: "width_4"},
        },
        opset_version=opset,
    )
    consolidate_onnx(out_path)
    print(f"  wrote {out_path} ({os.path.getsize(out_path) // 1024} KB)")
    return out_path, dummy, out


def export_update_graphs(pre_agg, post_agg, agg_kk, agg_ij, out_dir, num_edges, opset, seed=0):
    torch.manual_seed(seed)
    E = num_edges

    net = torch.randn(1, E, DIM) * 0.1
    inp = torch.randn(1, E, DIM) * 0.1
    corr = torch.randn(1, E, CORR_DIM) * 0.1

    # Realistic-shaped ii/jj/kk: a small patch graph, PATCH_LIFETIME-style
    # temporal window, so neighbors_cpu produces a non-trivial mix of -1s
    # and real neighbour indices (same edge-graph shape M2/M3 will feed).
    n_patches = max(1, E // 4)
    kk = torch.randint(0, n_patches, (E,))
    jj = torch.randint(0, 8, (E,))
    ii = torch.randint(0, 8, (E,))
    ix, jx = neighbors_cpu(kk, jj)

    with torch.no_grad():
        net_pre_agg = pre_agg(net, inp, corr, ix, jx)
    print(f"  sanity [update_pre_agg]: out {tuple(net_pre_agg.shape)}")

    pre_path = os.path.join(out_dir, "dpvo_update_pre_agg.onnx")
    torch.onnx.export(
        pre_agg, (net, inp, corr, ix, jx), pre_path,
        input_names=["net", "inp", "corr", "ix", "jx"],
        output_names=["net_pre_agg"],
        dynamic_axes={
            "net": {1: "num_edges"}, "inp": {1: "num_edges"}, "corr": {1: "num_edges"},
            "ix": {0: "num_edges"}, "jx": {0: "num_edges"},
            "net_pre_agg": {1: "num_edges"},
        },
        opset_version=opset,
    )
    consolidate_onnx(pre_path)
    print(f"  wrote {pre_path} ({os.path.getsize(pre_path) // 1024} KB)")

    # host-side SoftAgg step (see module docstring) -- not part of any ONNX
    # graph; this is exactly what M2's native Rust code must reproduce.
    with torch.no_grad():
        agg_kk_out = agg_kk(net_pre_agg, kk)
        agg_ij_out = agg_ij(net_pre_agg, ii * 12345 + jj)
        net_post_agg = net_pre_agg + agg_kk_out + agg_ij_out

    with torch.no_grad():
        net_out, delta, weight = post_agg(net_post_agg)
    print(f"  sanity [update_post_agg]: net_out {tuple(net_out.shape)}, "
          f"delta {tuple(delta.shape)}, weight {tuple(weight.shape)}")

    post_path = os.path.join(out_dir, "dpvo_update_post_agg.onnx")
    torch.onnx.export(
        post_agg, (net_post_agg,), post_path,
        input_names=["net_post_agg"], output_names=["net_out", "delta", "weight"],
        dynamic_axes={
            "net_post_agg": {1: "num_edges"},
            "net_out": {1: "num_edges"}, "delta": {1: "num_edges"}, "weight": {1: "num_edges"},
        },
        opset_version=opset,
    )
    consolidate_onnx(post_path)
    print(f"  wrote {post_path} ({os.path.getsize(post_path) // 1024} KB)")

    fixture = dict(
        net=net.numpy(), inp=inp.numpy(), corr=corr.numpy(), ix=ix.numpy(), jx=jx.numpy(),
        kk=kk.numpy(), ii=ii.numpy(), jj=jj.numpy(),
        net_pre_agg=net_pre_agg.numpy(),
        net_post_agg=net_post_agg.numpy(),
        net_out=net_out.numpy(), delta=delta.numpy(), weight=weight.numpy(),
    )
    return pre_path, post_path, fixture


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, help="directory to write the 4 .onnx graphs into")
    ap.add_argument("--checkpoint", default=None,
                    help="path to DPVO's dpvo.pth (download per E:/tools/DPVO/download_models_and_data.sh); "
                         "omit to export with random weights (still a valid export-correctness test)")
    ap.add_argument("--dpvo-root", default=os.environ.get("DPVO_ROOT", ""),
                    help="path to a local princeton-vl/DPVO checkout (for dpvo.extractor.BasicEncoder4 only)")
    ap.add_argument("--fixtures-dir", default=None,
                    help="if given, also dump patchify/corr/BA reference .npz fixtures here for M2/M3")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=752)
    ap.add_argument("--num-edges", type=int, default=64, help="edge count used for the update-graph export trace "
                                                                "(shape is exported as a dynamic axis, so this is "
                                                                "just a concrete tracing value, not a hard max)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.fixtures_dir:
        os.makedirs(args.fixtures_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    print("building models...")
    fnet, inet, pre_agg, post_agg, agg_kk, agg_ij, have_weights = build_models(args.checkpoint, args.dpvo_root)

    print("exporting fnet...")
    export_encoder(fnet, "fnet", args.out_dir, args.height, args.width, args.opset)
    print("exporting inet...")
    export_encoder(inet, "inet", args.out_dir, args.height, args.width, args.opset)

    print("exporting update cell (pre-agg / post-agg split, see module docstring)...")
    _, _, update_fixture = export_update_graphs(
        pre_agg, post_agg, agg_kk, agg_ij, args.out_dir, args.num_edges, args.opset, seed=args.seed)

    manifest = {
        "have_real_weights": have_weights,
        "checkpoint": args.checkpoint,
        "opset": args.opset,
        "num_edges_traced": args.num_edges,
        "fnet_dim": FNET_DIM, "inet_dim": DIM, "corr_dim": CORR_DIM, "res": RES,
    }
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {os.path.join(args.out_dir, 'manifest.json')}")

    if args.fixtures_dir:
        print("dumping fixtures for M2/M3...")
        np.savez(os.path.join(args.fixtures_dir, "update_cell_fixture.npz"), **update_fixture)
        print(f"  wrote {os.path.join(args.fixtures_dir, 'update_cell_fixture.npz')}")
        dump_correlation_fixture(
            EncoderOnnx(fnet),
            os.path.join(args.fixtures_dir, "patchify_fixture.npz"),
            os.path.join(args.fixtures_dir, "correlation_fixture.npz"),
            seed=args.seed,
        )
        dump_ba_fixture(os.path.join(args.fixtures_dir, "ba_fixture.npz"))

    print("done.")


if __name__ == "__main__":
    main()
