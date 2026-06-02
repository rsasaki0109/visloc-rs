#!/usr/bin/env python3
"""Hierarchical localization correspondences for 7-Scenes via SuperPoint+LightGlue.

For each test frame: retrieve the top-K appearance-nearest train keyframes,
match the query against each with LightGlue, lift the matched keyframe keypoints
to 3D world points using that keyframe's depth + ground-truth pose, keep the best
match per query keypoint (per-query dedup), and write one correspondence file
per query — "x y X Y Z conf" — consumed by the Rust
`relocalization_7scenes_demo --correspondences-dir`. Only the matcher differs
from the in-Rust BruteForce baseline; the Rust PnP+RANSAC still estimates pose.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def read_pose(path):
    v = np.loadtxt(path)  # 4x4 camera-to-world
    return v[:3, :3], v[:3, 3]


def normalized_mean(desc):  # desc: [N,256]
    if len(desc) == 0:
        return np.zeros(256, np.float32)
    m = desc.mean(axis=0)
    n = np.linalg.norm(m)
    return (m / n) if n > 0 else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--train-seqs", default="1,2,4,6")
    ap.add_argument("--test-seqs", default="3,5")
    ap.add_argument("--train-stride", type=int, default=10)
    ap.add_argument("--test-stride", type=int, default=10)
    ap.add_argument("--frames-per-seq", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--max-keypoints", type=int, default=1024)
    ap.add_argument("--min-depth", type=float, default=0.3)
    ap.add_argument("--max-depth", type=float, default=4.0)
    ap.add_argument("--focal", type=float, default=585.0)
    ap.add_argument("--cx", type=float, default=320.0)
    ap.add_argument("--cy", type=float, default=240.0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--grouped", action="store_true",
                    help="emit 'KF x y X Y Z conf' (per-keyframe groups, no dedup) "
                         "for Rust per-keyframe PnP via --grouped-corrs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fx = fy = args.focal
    cx, cy = args.cx, args.cy

    def frames(seqs, stride):
        for seq in [int(s) for s in seqs.split(",") if s.strip()]:
            sd = args.dataset / f"seq-{seq:02d}"
            for idx in range(0, args.frames_per_seq, stride):
                base = sd / f"frame-{idx:06d}"
                if (base.with_suffix(".color.png")).exists():
                    yield seq, idx, base

    # ---- Build train keyframes: SP feats (kept on device) + depth + pose. ----
    keyframes = []  # dict: feats(batched,device), globaldesc, depth, R, t
    for seq, idx, base in frames(args.train_seqs, args.train_stride):
        img = load_image(base.with_suffix(".color.png")).to(device)
        with torch.no_grad():
            feats = extractor.extract(img)
        desc = rbd(feats)["descriptors"].cpu().numpy()
        depth = np.asarray(Image.open(base.with_suffix(".depth.png"))).astype(np.float32)
        R, t = read_pose(base.with_suffix(".pose.txt"))
        keyframes.append(
            dict(feats=feats, g=normalized_mean(desc), depth=depth, R=R, t=t)
        )
    G = np.stack([k["g"] for k in keyframes])  # [K,256]
    print(f"train keyframes: {len(keyframes)}", flush=True)

    # ---- Per test frame: retrieve, LightGlue match, lift, dedup, write. ----
    n_written = 0
    for seq, idx, base in frames(args.test_seqs, args.test_stride):
        img = load_image(base.with_suffix(".color.png")).to(device)
        with torch.no_grad():
            qfeats = extractor.extract(img)
        q = rbd(qfeats)
        qk = q["keypoints"].cpu().numpy()
        qg = normalized_mean(q["descriptors"].cpu().numpy())
        order = np.argsort(-(G @ qg))[: args.topk]

        # query keypoint index -> (score, X, Y, Z) for pooled/dedup output, and
        # grouped rows (kf_rank, qi, X, Y, Z, score) for --grouped output.
        best = {}
        grouped_rows = []
        for kf_rank, ki in enumerate(order):
            kf = keyframes[ki]
            with torch.no_grad():
                m01 = rbd(matcher({"image0": qfeats, "image1": kf["feats"]}))
            matches = m01["matches"].cpu().numpy()  # [M,2] (query_idx, kf_idx)
            scores = m01["scores"].cpu().numpy() if "scores" in m01 else np.ones(len(matches))
            kfk = rbd(kf["feats"])["keypoints"].cpu().numpy()
            for (qi, ti), sc in zip(matches, scores):
                u, v = kfk[ti]
                iu, iv = int(round(u)), int(round(v))
                if not (0 <= iu < 640 and 0 <= iv < 480):
                    continue
                d = kf["depth"][iv, iu] / 1000.0
                if d <= 0 or d >= 65.0 or d < args.min_depth or d > args.max_depth:
                    continue
                pc = np.array([(u - cx) / fx * d, (v - cy) / fy * d, d])
                pw = kf["R"] @ pc + kf["t"]
                if args.grouped:
                    grouped_rows.append((kf_rank, int(qi), pw[0], pw[1], pw[2], float(sc)))
                else:
                    prev = best.get(int(qi))
                    if prev is None or sc > prev[0]:
                        best[int(qi)] = (float(sc), pw[0], pw[1], pw[2])

        out = args.out_dir / f"seq-{seq:02d}_frame-{idx:06d}.corr.txt"
        with open(out, "w") as f:
            if args.grouped:
                for kf_rank, qi, X, Y, Z, sc in grouped_rows:
                    x, y = qk[qi]
                    f.write(f"{kf_rank} {x:.3f} {y:.3f} {X:.5f} {Y:.5f} {Z:.5f} {sc:.5f}\n")
            else:
                for qi, (sc, X, Y, Z) in best.items():
                    x, y = qk[qi]
                    f.write(f"{x:.3f} {y:.3f} {X:.5f} {Y:.5f} {Z:.5f} {sc:.5f}\n")
        n_written += 1
        if n_written % 20 == 0:
            n_corrs = len(grouped_rows) if args.grouped else len(best)
            print(f"wrote {n_written} queries (seq {seq} idx {idx}, {n_corrs} corrs)", flush=True)
    print(f"DONE: {n_written} query correspondence files -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
