#!/usr/bin/env python3
"""Export an EigenPlaces global-descriptor (visual place recognition) model to
an ONNX graph with a Rust-friendly I/O contract for the in-process global
descriptor extractor (`crates/vision/src/features/global_descriptor_onnx.rs`).

EigenPlaces (Berton et al., ICCV 2023) and its predecessor CosPlace produce a
single L2-normalised global descriptor per image, trained for visual place
recognition (VPR). visloc-rs uses it as a *learned* loop-closure / relocalisation
retrieval front-end, replacing the hand-built k-means VLAD over local SuperPoint
descriptors — the same "learned front-end is the lever" pattern that already paid
off for SuperPoint + LightGlue.

Contract (batch size 1):
  input   image : (1, 3, H, W) float32   RGB in [0, 1] (NOT ImageNet-normalised;
                                          the graph applies the ImageNet mean/std
                                          internally, so the Rust caller only has
                                          to divide pixel values by 255)
  output  descriptor : (1, D) float32     L2-normalised global descriptor
                                          (D = fc_output_dim)

H and W are dynamic axes, so one exported model handles any input resolution
(the ResNet backbone is fully convolutional and GeM pools over the whole spatial
extent). The Rust side resizes each frame to a fixed size before inference for
determinism.

This script deliberately does NOT call `torch.hub.load("gmberton/...")`: that
would execute the upstream repo's `hubconf.py`. Instead it rebuilds the exact
`GeoLocalizationNet_` architecture from **torchvision** components plus the three
trivial aggregation layers (GeM / L2Norm / Flatten) reproduced inline, then loads
the cached EigenPlaces checkpoint as a plain weights-only state_dict. The result
is byte-for-byte the same network with zero third-party code execution.

Default weights are read from the torch.hub checkpoint cache populated earlier
(`~/.cache/torch/hub/checkpoints/ResNet50_2048_eigenplaces.pth`). If absent,
download it once from
  https://github.com/gmberton/EigenPlaces/releases/download/v1.0/ResNet50_2048_eigenplaces.pth

Usage:
  scripts/export_vpr_onnx.py --out models/eigenplaces_r50_2048.onnx
  scripts/export_vpr_onnx.py --backbone ResNet18 --fc-output-dim 512 \
      --out models/eigenplaces_r18_512.onnx
"""
import argparse
import os

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

CHANNELS_NUM_IN_LAST_CONV = {
    "ResNet18": 512,
    "ResNet50": 2048,
    "ResNet101": 2048,
    "VGG16": 512,
}

# Output dimensions for which the EigenPlaces authors released weights.
AVAILABLE = {
    "VGG16": [512],
    "ResNet18": [256, 512],
    "ResNet50": [128, 256, 512, 1024, 2048],
    "ResNet101": [128, 256, 512, 1024, 2048],
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --- aggregation layers, reproduced verbatim from the upstream layers.py -----
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))
        return x.pow(1.0 / self.p)


class Flatten(nn.Module):
    def forward(self, x):
        return x[:, :, 0, 0]


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


def build_backbone(backbone_name):
    """Truncated torchvision backbone matching upstream `_get_backbone`."""
    model = getattr(torchvision.models, backbone_name.lower())()
    if backbone_name.startswith("ResNet"):
        layers = list(model.children())[:-2]  # drop avgpool + fc
    elif backbone_name == "VGG16":
        layers = list(model.features.children())[:-2]
    else:
        raise ValueError(backbone_name)
    return nn.Sequential(*layers)


class GeoLocalizationNet(nn.Module):
    """Re-implementation of upstream `GeoLocalizationNet_`, same module names so
    the released state_dict loads with strict=True."""

    def __init__(self, backbone_name, fc_output_dim):
        super().__init__()
        self.backbone = build_backbone(backbone_name)
        features_dim = CHANNELS_NUM_IN_LAST_CONV[backbone_name]
        self.aggregation = nn.Sequential(
            L2Norm(),
            GeM(),
            Flatten(),
            nn.Linear(features_dim, fc_output_dim),
            L2Norm(),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.aggregation(x)
        return x


class EigenPlacesOnnx(nn.Module):
    """Wraps the net with baked-in ImageNet normalisation so the Rust caller
    feeds raw RGB in [0, 1]."""

    def __init__(self, net):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, image):
        x = (image - self.mean) / self.std
        return self.net(x)


def default_ckpt(backbone, dim):
    cache = os.path.expanduser("~/.cache/torch/hub/checkpoints")
    return os.path.join(cache, f"{backbone}_{dim}_eigenplaces.pth")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="ResNet50",
                    choices=list(CHANNELS_NUM_IN_LAST_CONV.keys()))
    ap.add_argument("--fc-output-dim", type=int, default=2048)
    ap.add_argument("--checkpoint", default=None,
                    help="path to <Backbone>_<dim>_eigenplaces.pth "
                         "(default: torch.hub checkpoint cache)")
    ap.add_argument("--height", type=int, default=480,
                    help="dummy export height (axis is dynamic)")
    ap.add_argument("--width", type=int, default=640,
                    help="dummy export width (axis is dynamic)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    if args.fc_output_dim not in AVAILABLE.get(args.backbone, []):
        raise SystemExit(
            f"no released weights for {args.backbone} dim {args.fc_output_dim}; "
            f"available: {AVAILABLE[args.backbone]}")

    ckpt = args.checkpoint or default_ckpt(args.backbone, args.fc_output_dim)
    if not os.path.exists(ckpt):
        raise SystemExit(
            f"checkpoint not found: {ckpt}\n"
            f"download once from https://github.com/gmberton/EigenPlaces/"
            f"releases/download/v1.0/{args.backbone}_{args.fc_output_dim}_eigenplaces.pth")

    net = GeoLocalizationNet(args.backbone, args.fc_output_dim)
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    missing, unexpected = net.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"WARNING state_dict mismatch: missing={missing} unexpected={unexpected}")
    else:
        print("state_dict loaded strict-clean (all keys matched)")
    net.eval()

    model = EigenPlacesOnnx(net).eval()
    dummy = torch.rand(1, 3, args.height, args.width)
    with torch.no_grad():
        out = model(dummy)
    print(f"sanity: descriptor {tuple(out.shape)} {out.dtype}, "
          f"L2 norm {out.norm(dim=1).item():.6f}")
    assert out.shape == (1, args.fc_output_dim)
    assert abs(out.norm(dim=1).item() - 1.0) < 1e-4, "output must be L2-normalised"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.onnx.export(
        model, (dummy,), args.out,
        input_names=["image"],
        output_names=["descriptor"],
        dynamic_axes={"image": {2: "h", 3: "w"}},
        opset_version=args.opset,
    )

    # The torch dynamo exporter spills weights into a sidecar `<out>.data` file.
    # Consolidate into a single self-contained .onnx (matches the single-file
    # convention of superpoint_1500.onnx / lightglue.onnx and avoids the runner
    # having to ship a sidecar). Then delete the sidecar.
    import onnx
    consolidated = onnx.load(args.out, load_external_data=True)
    onnx.save(consolidated, args.out, save_as_external_data=False)
    sidecar = args.out + ".data"
    if os.path.exists(sidecar):
        os.remove(sidecar)
    print(f"wrote {args.out} (single-file, {os.path.getsize(args.out) // (1024 * 1024)} MB)")

    # ONNX-runtime parity check (CPU): the exported graph must reproduce torch.
    try:
        import numpy as np
        import onnxruntime as ort
        sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {"image": dummy.numpy()})[0]
        max_abs = float(np.abs(ort_out - out.numpy()).max())
        print(f"onnxruntime parity: max abs diff {max_abs:.2e} "
              f"({'OK' if max_abs < 1e-4 else 'MISMATCH'})")
    except ImportError:
        print("onnxruntime not available; skipped parity check")


if __name__ == "__main__":
    main()
