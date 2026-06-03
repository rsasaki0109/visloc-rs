#!/usr/bin/env python3
"""Export SuperPoint features for selected 7-Scenes frames.

Writes one file per frame named  seq-SS_frame-NNNNNN.txt  in the
"X Y SCORE D0 D1 ..." format consumed by visloc_io::external_deep.
"""
import argparse
from pathlib import Path

import torch
from lightglue import SuperPoint
from lightglue.utils import load_image, rbd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--seqs", required=True, help="comma list, e.g. 1,2,4,6")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--frames-per-seq", type=int, default=1000)
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seqs = [int(s) for s in args.seqs.split(",") if s.strip()]
    total = 0
    for seq in seqs:
        seq_dir = args.dataset / f"seq-{seq:02d}"
        for idx in range(0, args.frames_per_seq, args.stride):
            color = seq_dir / f"frame-{idx:06d}.color.png"
            if not color.exists():
                continue
            image = load_image(color).to(device)
            with torch.no_grad():
                feats = rbd(extractor.extract(image))
            kpts = feats["keypoints"].cpu().numpy()
            scores = feats["keypoint_scores"].cpu().numpy()
            descs = feats["descriptors"].cpu().numpy()
            out = args.out_dir / f"seq-{seq:02d}_frame-{idx:06d}.txt"
            with open(out, "w") as f:
                for (x, y), s, d in zip(kpts, scores, descs):
                    row = f"{x:.3f} {y:.3f} {float(s):.5f} " + " ".join(
                        f"{v:.5f}" for v in d
                    )
                    f.write(row + "\n")
            total += 1
            if total % 25 == 0:
                print(f"exported {total} frames (seq {seq} idx {idx})", flush=True)
    print(f"DONE: {total} frames -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
