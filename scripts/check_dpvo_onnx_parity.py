#!/usr/bin/env python3
"""Check numerical parity between the PyTorch modules and the ONNX graphs
produced by `scripts/export_dpvo_onnx.py` (fnet, inet, dpvo_update_pre_agg,
dpvo_update_post_agg), on identical seeded random inputs run through both
PyTorch (CPU) and ONNX Runtime (CPUExecutionProvider).

Mirrors the parity-check style already used inline by
`scripts/export_vpr_onnx.py` (max-abs-diff against a 1e-4 threshold), pulled
out into its own script per `docs/dpvo_droid_port_plan.md`'s M1 spec.

This script imports model-building code directly from
`export_dpvo_onnx.py` (same directory) rather than duplicating it, so the
PyTorch reference is guaranteed to be the exact same architecture/weights
that were traced for export.

Usage:
  scripts/check_dpvo_onnx_parity.py --onnx-dir E:/visloc_archive/dpvo_onnx_m1 \\
      --checkpoint E:/tools/DPVO/models_extracted/dpvo.pth \\
      --dpvo-root E:/tools/DPVO

  # random-weight parity (still meaningful: checks the ONNX graph is a
  # faithful translation of the traced PyTorch graph, independent of which
  # weights were baked in). IMPORTANT: random weights are never persisted to
  # disk, so this only reconstructs the SAME weights as export time if you
  # pass the SAME --seed used for `export_dpvo_onnx.py` (both default to 0):
  scripts/check_dpvo_onnx_parity.py --onnx-dir E:/visloc_archive/dpvo_onnx_m1 --seed 0
"""
import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_dpvo_onnx as dpvo_export  # noqa: E402

PASS_THRESHOLD = 1e-4


def report(name, torch_out, ort_out):
    torch_np = torch_out.detach().numpy() if hasattr(torch_out, "detach") else np.asarray(torch_out)
    ort_np = np.asarray(ort_out)
    if torch_np.shape != ort_np.shape:
        print(f"  {name:14s} SHAPE MISMATCH torch={torch_np.shape} onnx={ort_np.shape}  FAIL")
        return False
    max_abs = float(np.abs(torch_np - ort_np).max()) if torch_np.size else 0.0
    denom = np.abs(torch_np).max()
    rel = max_abs / denom if denom > 1e-12 else max_abs
    ok = max_abs < PASS_THRESHOLD
    print(f"  {name:14s} shape={torch_np.shape!s:20s} max_abs={max_abs:.3e}  max_rel={rel:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def check_encoder(onnx_path, torch_model, dummy_input, out_name):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_out = torch_model(dummy_input)
    ort_out = sess.run(None, {"image": dummy_input.numpy()})[0]
    return report(out_name, torch_out, ort_out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", required=True, help="directory containing the 4 exported .onnx graphs")
    ap.add_argument("--checkpoint", default=None, help="same checkpoint used at export time (or omit for both)")
    ap.add_argument("--dpvo-root", default=os.environ.get("DPVO_ROOT", ""))
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=384)
    ap.add_argument("--num-edges", type=int, default=97, help="a different edge count than export time, to also "
                                                                "exercise the dynamic 'num_edges' axis")
    ap.add_argument("--seed", type=int, default=0,
                     help="module-init seed; MUST match export_dpvo_onnx.py's --seed (also default 0) when "
                          "--checkpoint is omitted, since random weights are not persisted to disk")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print("rebuilding PyTorch reference modules (identical to export_dpvo_onnx.py)...")
    fnet, inet, pre_agg, post_agg, agg_kk, agg_ij, have_weights = dpvo_export.build_models(
        args.checkpoint, args.dpvo_root)

    manifest_path = os.path.join(args.onnx_dir, "manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        print(f"manifest: {manifest}")

    all_ok = True

    print("\n[fnet]")
    dummy = torch.rand(1, 3, args.height, args.width) * 255.0
    all_ok &= check_encoder(os.path.join(args.onnx_dir, "fnet.onnx"),
                             dpvo_export.EncoderOnnx(fnet), dummy, "fmap")

    print("\n[inet]")
    all_ok &= check_encoder(os.path.join(args.onnx_dir, "inet.onnx"),
                             dpvo_export.EncoderOnnx(inet), dummy, "imap")

    print("\n[dpvo_update_pre_agg] (different num_edges than export trace, exercising the dynamic axis)")
    E = args.num_edges
    net = torch.randn(1, E, dpvo_export.DIM) * 0.1
    inp = torch.randn(1, E, dpvo_export.DIM) * 0.1
    corr = torch.randn(1, E, dpvo_export.CORR_DIM) * 0.1
    n_patches = max(1, E // 4)
    kk = torch.randint(0, n_patches, (E,))
    jj = torch.randint(0, 8, (E,))
    ii = torch.randint(0, 8, (E,))
    ix, jx = dpvo_export.neighbors_cpu(kk, jj)

    with torch.no_grad():
        net_pre_agg = pre_agg(net, inp, corr, ix, jx)

    sess_pre = ort.InferenceSession(os.path.join(args.onnx_dir, "dpvo_update_pre_agg.onnx"),
                                     providers=["CPUExecutionProvider"])
    ort_pre = sess_pre.run(None, {
        "net": net.numpy(), "inp": inp.numpy(), "corr": corr.numpy(),
        "ix": ix.numpy(), "jx": jx.numpy(),
    })[0]
    all_ok &= report("net_pre_agg", net_pre_agg, ort_pre)

    print("\n[host-side SoftAgg step -- not ONNX, sanity-only]")
    with torch.no_grad():
        agg_kk_out = agg_kk(net_pre_agg, kk)
        agg_ij_out = agg_ij(net_pre_agg, ii * 12345 + jj)
        net_post_agg = net_pre_agg + agg_kk_out + agg_ij_out
    print(f"  net_post_agg shape={tuple(net_post_agg.shape)} (computed host-side, no ONNX Runtime involved; "
          f"see export_dpvo_onnx.py module docstring)")

    print("\n[dpvo_update_post_agg]")
    with torch.no_grad():
        net_out, delta, weight = post_agg(net_post_agg)
    sess_post = ort.InferenceSession(os.path.join(args.onnx_dir, "dpvo_update_post_agg.onnx"),
                                      providers=["CPUExecutionProvider"])
    ort_net_out, ort_delta, ort_weight = sess_post.run(None, {"net_post_agg": net_post_agg.numpy()})
    all_ok &= report("net_out", net_out, ort_net_out)
    all_ok &= report("delta", delta, ort_delta)
    all_ok &= report("weight", weight, ort_weight)

    print(f"\noverall: {'ALL PASS' if all_ok else 'SOME FAILED'} (threshold {PASS_THRESHOLD:.0e} max-abs), "
          f"real_weights={have_weights}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
