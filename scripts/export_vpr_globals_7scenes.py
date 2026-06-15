#!/usr/bin/env python3
"""Export per-frame learned VPR global descriptors (EigenPlaces) for selected
7-Scenes frames, one file per frame named  seq-SS_frame-NNNNNN.txt  in the
output directory (mirroring scripts/export_superpoint_7scenes.py), each a single
whitespace-separated line of float32 values.

`examples/relocalization_7scenes_demo.rs --global-descriptor-dir <dir>` consumes
these as the retrieval-gating descriptor, replacing the hand-built
`normalized_mean` of SuperPoint descriptors — the same learned global descriptor
the in-process Rust `GlobalDescriptorOnnxExtractor` runs. This is the *learned*
side of the retrieval A/B; the SuperPoint features (for PnP) are exported
separately by export_superpoint_7scenes.py.

Uses the ONNX model exported by scripts/export_vpr_onnx.py (no upstream-repo
code execution). ImageNet normalisation is baked into the graph, so the image is
fed as RGB in [0, 1].

Usage:
  scripts/export_vpr_globals_7scenes.py \
      --dataset /path/to/7scenes/chess --seqs 1,2,4,6 --stride 20 \
      --model models/eigenplaces_r50_2048.onnx --out-dir /tmp/vpr_7scenes_chess
"""
import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--seqs", required=True, help="comma list, e.g. 1,2,4,6")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--frames-per-seq", type=int, default=1000)
    ap.add_argument("--model", default="models/eigenplaces_r50_2048.onnx")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    avail = ort.get_available_providers()
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail
        else ["CPUExecutionProvider"]
    )
    sess = ort.InferenceSession(args.model, providers=providers)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    total = 0
    for seq in seqs:
        seq_dir = args.dataset / f"seq-{seq:02d}"
        for idx in range(0, args.frames_per_seq, args.stride):
            color = seq_dir / f"frame-{idx:06d}.color.png"
            if not color.exists():
                continue
            im = Image.open(color).convert("RGB").resize((args.width, args.height))
            x = (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None]
            desc = sess.run(None, {"image": x})[0].ravel()
            out = args.out_dir / f"seq-{seq:02d}_frame-{idx:06d}.txt"
            out.write_text(" ".join(f"{v:.7f}" for v in desc) + "\n")
            total += 1
    print(f"DONE: {total} frames -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
