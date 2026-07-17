#!/usr/bin/env python3
"""Export the LightGlue matcher (SuperPoint weights) to an ONNX graph with a
Rust-friendly I/O contract for the in-process matcher
(`crates/vision/src/features/lightglue_onnx.rs`).

Contract (batch size 1):
  inputs  kpts0 : (1, M, 2) float32   (x, y) pixel coordinates, image 0
          desc0 : (1, M, 256) float32 L2-normalised descriptors, image 0
          kpts1 : (1, N, 2) float32
          desc1 : (1, N, 256) float32
  outputs matches0 : (M,) int64   for each kpt in image 0, the matched index in
                                   image 1, or -1 if unmatched
          mscores0 : (M,) float32 matching confidence in (0, 1]

LightGlue's adaptive depth/width pruning and FlashAttention introduce
data-dependent control flow that does not trace to a static ONNX graph; we
disable them (`depth_confidence=-1, width_confidence=-1, flash=False`), which
runs all 9 transformer layers unconditionally — a pure speed knob, the matches
are unchanged. Keypoints are normalised by a baked-in image size (LightGlue's
own `normalize_keypoints`), so the model is exported per image resolution
(re-export for a different camera). Everything else (input projection, rotary
positional encoding, the 9 self/cross attention layers, the log-assignment
head, and `filter_matches`) is the reference implementation verbatim.

Weights: `LightGlue(features="superpoint")` downloads cvg/LightGlue's own
released checkpoint (Apache-2.0, see `docs/colmap_port_plan.md`'s "M6
results" for the license verdict) on first use; nothing is committed to this
repo (same "user-run export script, no weights in the repo" convention as
`export_superpoint_onnx.py`).

**Exporter: `dynamo=True` is required, not optional, on this graph.** The
default (legacy TorchScript-tracing) exporter fails on every configuration
tried (`torch.onnx.symbolic_opset9.transpose`: `IndexError: list index out of
range`, inside LightGlue's rotary-positional-encoding/attention code — a
legacy-exporter tracing bug, not a bug in LightGlue or in this script: the
same module runs correctly in eager PyTorch and the *dynamo* exporter
(`torch.export`-based FX capture, not per-op symbolic translation) traces the
identical module without incident). This reproduces on both torch 2.5.1+cu121
and (with `dynamo=True`) succeeds on torch 2.9.1 — see the plan doc for the
full repro and why a **dedicated venv** (`E:/tools/venvs/lightglue_export`,
not the repo's existing `E:/tools/venv-cu`) was used: pinning a newer torch
just for this export without disturbing whatever other in-flight work already
depends on `venv-cu`'s existing (older) torch pin.

Usage:
  scripts/export_lightglue_onnx.py --out models/lightglue.onnx \
      --width 752 --height 480
"""
import argparse
import os

import torch
from lightglue import LightGlue
from lightglue.lightglue import filter_matches, normalize_keypoints


class LightGlueOnnx(torch.nn.Module):
    """ONNX-exportable LightGlue head (batch 1, no pruning, no flash)."""

    def __init__(self, lg: LightGlue, width: int, height: int):
        super().__init__()
        self.lg = lg
        self.register_buffer(
            "size", torch.tensor([[float(width), float(height)]], dtype=torch.float32)
        )

    def forward(self, kpts0, desc0, kpts1, desc1):
        lg = self.lg
        k0 = normalize_keypoints(kpts0, self.size)
        k1 = normalize_keypoints(kpts1, self.size)
        d0 = lg.input_proj(desc0)
        d1 = lg.input_proj(desc1)
        enc0 = lg.posenc(k0)
        enc1 = lg.posenc(k1)
        for i in range(lg.conf.n_layers):
            d0, d1 = lg.transformers[i](d0, d1, enc0, enc1)
        scores, _ = lg.log_assignment[i](d0, d1)
        m0, _m1, mscores0, _mscores1 = filter_matches(scores, lg.conf.filter_threshold)
        return m0[0].to(torch.int64), mscores0[0]


def build_model(width: int, height: int) -> "LightGlueOnnx":
    """Build the (untraced) PyTorch reference module for a given baked-in
    image size. Factored out of `main()` so `check_lightglue_onnx_parity.py`
    can import the *exact* module that was (or would be) traced for export,
    the same "import model-building code directly" pattern
    `scripts/check_dpvo_onnx_parity.py` uses for the DPVO port (M1,
    `docs/dpvo_droid_port_plan.md`) — the PyTorch reference is guaranteed to
    be the same architecture/weights the export traces, not a
    hand-reconstructed approximation of it.
    """
    lg = LightGlue(
        features="superpoint",
        depth_confidence=-1,
        width_confidence=-1,
        flash=False,
    ).eval()
    return LightGlueOnnx(lg, width, height).eval()


def dummy_inputs(width: int, height: int, m: int = 512, n: int = 480):
    """Seeded-shape (not seeded-*value*: caller controls RNG state) dummy
    `(kpts0, desc0, kpts1, desc1)` tensors, distinct `M != N` so tracing does
    not silently bake `M == N` into the graph. Shared by `main()`'s own
    export-time sanity check and `check_lightglue_onnx_parity.py`'s
    "seeded random" fixture.
    """
    kpts0 = torch.rand(1, m, 2) * torch.tensor([float(width), float(height)])
    kpts1 = torch.rand(1, n, 2) * torch.tensor([float(width), float(height)])
    desc0 = torch.nn.functional.normalize(torch.rand(1, m, 256), dim=-1)
    desc1 = torch.nn.functional.normalize(torch.rand(1, n, 256), dim=-1)
    return kpts0, desc0, kpts1, desc1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=752)
    ap.add_argument("--height", type=int, default=480)
    # 18, not 16/17: the dynamo/`torch.export`-based exporter (see the
    # `dynamo=True` call below) only ships opset-18 implementations for a
    # couple of ops this graph touches; requesting 16 or 17 makes it emit a
    # non-fatal warning and attempt (and fail) a version downversion before
    # falling back to 18 anyway. Asking for 18 directly avoids the noise.
    # Still satisfies M6's own "opset >= 16" requirement.
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    model = build_model(args.width, args.height)
    kpts0, desc0, kpts1, desc1 = dummy_inputs(args.width, args.height)
    with torch.no_grad():
        mo, sc = model(kpts0, desc0, kpts1, desc1)
    print(f"sanity: matches0 {tuple(mo.shape)} {mo.dtype}, mscores0 {tuple(sc.shape)} {sc.dtype}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model, (kpts0, desc0, kpts1, desc1), args.out,
        input_names=["kpts0", "desc0", "kpts1", "desc1"],
        output_names=["matches0", "mscores0"],
        dynamic_axes={
            "kpts0": {1: "m"}, "desc0": {1: "m"},
            "kpts1": {1: "n"}, "desc1": {1: "n"},
            "matches0": {0: "m"}, "mscores0": {0: "m"},
        },
        opset_version=args.opset,
        # PyTorch >= 2.5's FX/`torch.export`-based exporter ("dynamo=True").
        # The legacy TorchScript-tracing exporter (the default) fails on this
        # graph on torch 2.5.1: `torch.onnx.symbolic_opset9.transpose` raises
        # `IndexError: list index out of range` inside LightGlue's rotary
        # positional-encoding / attention code (a transpose on more axes than
        # the legacy tracer's symbolic shape inference believes the tensor
        # has -- a legacy-exporter tracing bug, not a LightGlue bug: the same
        # module runs and produces correct output directly in eager PyTorch).
        # The dynamo exporter traces via `torch.export` (a from-scratch FX
        # capture, not per-op symbolic translation) and does not hit this
        # code path. See `docs/colmap_port_plan.md`'s "M6 results" for the
        # full repro.
        dynamo=True,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
