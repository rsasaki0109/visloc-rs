#!/usr/bin/env python3
"""Relocalization *retrieval* recall benchmark: learned VPR global descriptor
(EigenPlaces, via the same ONNX the in-process Rust extractor loads) vs the
hand-built `normalized_mean` of SuperPoint descriptors that
`examples/relocalization_7scenes_demo.rs` currently uses for retrieval gating.

This isolates the retrieval stage — the binding constraint in relocalization,
where a single query must find its place in a map with no temporal/odometry
prior. (Contrast VO loop closure, where retrieval is easy and the geometric
verifier is the lever — see docs/ and the seq02 finding.) For each test query it
asks: is a *geometrically correct* train keyframe (camera centre within
`--pos-thresh` metres of the query) ranked in the descriptor's top-K?

Both descriptors are evaluated on the *same* frames so the only variable is the
descriptor. Models are the ONNX graphs exported by
`scripts/export_vpr_onnx.py` (EigenPlaces) and `scripts/export_superpoint_onnx.py`
(SuperPoint), i.e. exactly what the Rust pipeline runs.

Usage (7-Scenes chess; reads TrainSplit.txt / TestSplit.txt):
  scripts/eval_relocalization_recall.py \
      --dataset /path/to/7scenes/chess \
      --vpr-model models/eigenplaces_r50_2048.onnx \
      --superpoint-model models/superpoint_1500.onnx \
      --train-stride 20 --test-stride 40 --pos-thresh 0.30
"""
import argparse
import glob
import re

import numpy as np
import onnxruntime as ort
from PIL import Image


def providers():
    avail = ort.get_available_providers()
    return (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in avail
        else ["CPUExecutionProvider"]
    )


def read_split(path):
    return [int(re.search(r"(\d+)", ln).group(1)) for ln in open(path) if re.search(r"\d", ln)]


def vpr_embed(sess, path, w, h):
    im = Image.open(path).convert("RGB").resize((w, h))
    x = (np.asarray(im, np.float32) / 255.0).transpose(2, 0, 1)[None]
    return sess.run(None, {"image": x})[0].ravel()


def sp_normalized_mean(sess, path, w, h):
    im = Image.open(path).convert("L").resize((w, h))
    x = (np.asarray(im, np.float32) / 255.0)[None, None]
    desc = None
    for o in sess.run(None, {"image": x}):
        if o.ndim == 2 and 256 in o.shape:
            desc = o if o.shape[1] == 256 else o.T
    if desc is None or len(desc) == 0:
        return np.zeros(256, np.float32)
    m = desc.mean(0)
    n = np.linalg.norm(m)
    return m / n if n > 0 else m


def pose_centre(path):
    return np.loadtxt(path)[:3, 3]


def collect(seqs, stride, dataset, embed_fns):
    embs = {name: [] for name in embed_fns}
    centres = []
    for s in seqs:
        frames = sorted(glob.glob(f"{dataset}/seq-{s:02d}/frame-*.color.png"))[::stride]
        for c in frames:
            for name, fn in embed_fns.items():
                embs[name].append(fn(c))
            centres.append(pose_centre(c.replace(".color.png", ".pose.txt")))
    return {k: np.array(v) for k, v in embs.items()}, np.array(centres)


def l2norm(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def recall_at_k(q_emb, db_emb, gt, valid, ks):
    sim = l2norm(q_emb) @ l2norm(db_emb).T
    order = np.argsort(-sim, axis=1)
    out = {}
    for k in ks:
        top = order[:, :k]
        hit = sum(gt[i, top[i]].any() for i in range(len(q_emb)) if valid[i])
        out[k] = hit / valid.sum()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="7-Scenes scene dir (with seq-NN/, *Split.txt)")
    ap.add_argument("--vpr-model", default="models/eigenplaces_r50_2048.onnx")
    ap.add_argument("--superpoint-model", default="models/superpoint_1500.onnx")
    ap.add_argument("--train-stride", type=int, default=20)
    ap.add_argument("--test-stride", type=int, default=40)
    ap.add_argument("--pos-thresh", type=float, default=0.30, help="correct-retrieval radius (m)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 20])
    args = ap.parse_args()

    tr = read_split(f"{args.dataset}/TrainSplit.txt")
    te = read_split(f"{args.dataset}/TestSplit.txt")
    print(f"train seqs {tr}  test seqs {te}")

    vpr = ort.InferenceSession(args.vpr_model, providers=providers())
    sp = ort.InferenceSession(args.superpoint_model, providers=providers())
    embed_fns = {
        "EigenPlaces": lambda p: vpr_embed(vpr, p, args.width, args.height),
        "normalized_mean": lambda p: sp_normalized_mean(sp, p, args.width, args.height),
    }

    db, c_db = collect(tr, args.train_stride, args.dataset, embed_fns)
    q, c_q = collect(te, args.test_stride, args.dataset, embed_fns)
    print(f"train keyframes {len(c_db)}  test queries {len(c_q)}")

    dist = np.linalg.norm(c_db[None] - c_q[:, None], axis=2)
    gt = dist < args.pos_thresh
    valid = gt.any(1)
    print(f"queries with a <{args.pos_thresh}m train keyframe: {valid.sum()}/{len(c_q)}\n")

    print(f"{'recall@K':<18}" + "".join(f"@{k:<7}" for k in args.ks))
    for name in embed_fns:
        r = recall_at_k(q[name], db[name], gt, valid, args.ks)
        print(f"{name:<18}" + "".join(f"{100 * r[k]:<8.1f}" for k in args.ks))


if __name__ == "__main__":
    main()
