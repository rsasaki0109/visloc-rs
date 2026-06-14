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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=752)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    lg = LightGlue(
        features="superpoint",
        depth_confidence=-1,
        width_confidence=-1,
        flash=False,
    ).eval()
    model = LightGlueOnnx(lg, args.width, args.height).eval()

    # Dummy inputs with distinct M, N so the export does not bake M == N.
    m, n = 512, 480
    kpts0 = torch.rand(1, m, 2) * torch.tensor([args.width, args.height])
    kpts1 = torch.rand(1, n, 2) * torch.tensor([args.width, args.height])
    desc0 = torch.nn.functional.normalize(torch.rand(1, m, 256), dim=-1)
    desc1 = torch.nn.functional.normalize(torch.rand(1, n, 256), dim=-1)
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
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
