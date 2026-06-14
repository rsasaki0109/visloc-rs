#!/usr/bin/env python3
"""Export the LightGlue SuperPoint front-end to an ONNX graph whose I/O
contract matches the in-Rust runtime (`crates/vision/src/features/superpoint_onnx.rs`).

Contract:
  input   image       : (1, 1, H, W) float32 in [0, 1]
  outputs keypoints   : (N, 2) int64   (x, y) pixel coordinates
          scores      : (N,)   float32
          descriptors : (N, 256) float32  (L2-normalised, row per keypoint)

The reference `lightglue.SuperPoint.forward` selects keypoints with a dynamic
`torch.where(scores > threshold)`, which does not trace cleanly to a static
ONNX graph. We reimplement the head for batch size 1 with a fixed `top-k`
selection (k = max_keypoints) so the graph has a constant keypoint count and
only H/W stay dynamic. NMS, border removal and bilinear descriptor sampling are
kept bit-for-bit from the reference implementation.

Usage:
  scripts/export_superpoint_onnx.py --out models/superpoint_1500.onnx \
      --max-keypoints 1500 --nms-radius 4 --height 480 --width 752
"""
import argparse
import os

import torch
import torch.nn.functional as F
from lightglue.superpoint import SuperPoint, simple_nms, sample_descriptors


class SuperPointOnnx(torch.nn.Module):
    """ONNX-exportable SuperPoint head (batch size 1, fixed top-k)."""

    def __init__(self, sp: SuperPoint, max_keypoints: int, nms_radius: int,
                 remove_borders: int):
        super().__init__()
        self.sp = sp
        self.max_keypoints = max_keypoints
        self.nms_radius = nms_radius
        self.remove_borders = remove_borders

    def forward(self, image):
        sp = self.sp
        x = sp.relu(sp.conv1a(image))
        x = sp.relu(sp.conv1b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv2a(x))
        x = sp.relu(sp.conv2b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv3a(x))
        x = sp.relu(sp.conv3b(x))
        x = sp.pool(x)
        x = sp.relu(sp.conv4a(x))
        x = sp.relu(sp.conv4b(x))

        # Dense keypoint scores
        cPa = sp.relu(sp.convPa(x))
        scores = sp.convPb(cPa)
        scores = F.softmax(scores, 1)[:, :-1]
        b, _, h, w = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(b, h, w, 8, 8)
        scores = scores.permute(0, 1, 3, 2, 4).reshape(b, h * 8, w * 8)
        scores = simple_nms(scores, self.nms_radius)

        pad = self.remove_borders
        if pad:
            scores[:, :pad] = -1
            scores[:, :, :pad] = -1
            scores[:, -pad:] = -1
            scores[:, :, -pad:] = -1

        # Fixed top-k selection over the flattened score map (b == 1).
        flat = scores.reshape(-1)
        topk_scores, idx = torch.topk(flat, self.max_keypoints, dim=0, sorted=True)
        full_w = scores.shape[-1]
        # Avoid aten::remainder (the dynamo ONNX exporter cannot translate it
        # when the divisor is a symbolic dim): recover x via subtraction.
        ys = torch.div(idx, full_w, rounding_mode="floor")
        xs = idx - ys * full_w
        keypoints_xy = torch.stack([xs, ys], dim=-1).float()  # (N, 2) (x, y)

        # Dense descriptors, sampled at keypoints (reference path).
        cDa = sp.relu(sp.convDa(x))
        descriptors = sp.convDb(cDa)
        descriptors = F.normalize(descriptors, p=2, dim=1)  # (1, 256, h, w)
        desc = sample_descriptors(keypoints_xy[None], descriptors, 8)[0]  # (256, N)
        desc = desc.transpose(0, 1).contiguous()  # (N, 256)

        keypoints_out = keypoints_xy.to(torch.int64)  # (N, 2) int64
        return keypoints_out, topk_scores, desc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-keypoints", type=int, default=1500)
    ap.add_argument("--nms-radius", type=int, default=4)
    ap.add_argument("--remove-borders", type=int, default=4)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=752)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    sp = SuperPoint(max_num_keypoints=args.max_keypoints,
                    nms_radius=args.nms_radius,
                    detection_threshold=0.0).eval()
    model = SuperPointOnnx(sp, args.max_keypoints, args.nms_radius,
                           args.remove_borders).eval()

    dummy = torch.rand(1, 1, args.height, args.width)
    with torch.no_grad():
        kp, sc, de = model(dummy)
    print(f"sanity: kp{tuple(kp.shape)} {kp.dtype}, sc{tuple(sc.shape)} {sc.dtype}, "
          f"de{tuple(de.shape)} {de.dtype}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model, (dummy,), args.out,
        input_names=["image"],
        output_names=["keypoints", "scores", "descriptors"],
        dynamic_axes={
            "image": {2: "height", 3: "width"},
            "keypoints": {0: "num_keypoints"},
            "scores": {0: "num_keypoints"},
            "descriptors": {0: "num_keypoints"},
        },
        opset_version=args.opset,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
