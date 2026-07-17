#!/usr/bin/env python3
"""Check numerical parity between the PyTorch `lightglue` reference module and
the ONNX graph produced by `scripts/export_lightglue_onnx.py`, on both a
seeded-random fixture and a real-descriptor fixture (two adjacent ETH3D
`terrace` images' cached SuperPoint features), run through both PyTorch (CPU)
and ONNX Runtime (`CPUExecutionProvider`).

Mirrors `scripts/check_dpvo_onnx_parity.py`'s own style for Milestone M6 of
`docs/colmap_port_plan.md` (LightGlue-as-ONNX-matcher): max-abs-diff on the
match-score output against a `1e-4` threshold, plus an exact-match-index
agreement rate on the discrete `matches0` output (an integer argmax-derived
index, so "parity" there means *identical*, not *close* -- any real
floating-point noise between the two backends shows up as an occasional
disagreeing index, not a small numeric delta, which is why this script
reports an agreement rate for `matches0` and a numeric diff for `mscores0`
rather than diffing `matches0` as if it were continuous).

This script imports `build_model`/`dummy_inputs` directly from
`export_lightglue_onnx.py` (same directory), so the PyTorch reference is
guaranteed to be the exact module traced for export -- the same "import
model-building code, don't hand-reconstruct it" pattern
`check_dpvo_onnx_parity.py` already established for the DPVO port.

Every fixture this script *checks* is also *dumped* to `--fixtures-dir` as a
`.npz` archive (`kpts0, desc0, kpts1, desc1, matches0, mscores0`), consumed by
`crates/vision/tests/lightglue_onnx_parity.rs`'s `#[ignore]`-gated Rust
parity tests -- so a single Python run produces both the parity verdict and
the fixtures the Rust side re-checks against the *already-exported* ONNX
graph via `ort`.

Usage:
  # seeded-random fixture only (no real feature files needed):
  scripts/check_lightglue_onnx_parity.py \\
      --onnx E:/visloc_archive/lightglue_onnx_m6/models/lightglue_terrace_6205x4136.onnx \\
      --width 6205 --height 4136 \\
      --fixtures-dir E:/visloc_archive/lightglue_onnx_m6/fixtures

  # + a real-descriptor fixture from two already-exported SuperPoint feature
  # files (the repo's `X Y SCORE D0..D255` per-line text format):
  scripts/check_lightglue_onnx_parity.py \\
      --onnx E:/visloc_archive/lightglue_onnx_m6/models/lightglue_terrace_6205x4136.onnx \\
      --width 6205 --height 4136 \\
      --real-features0 E:/datasets/eth3d/battle/terrace/visloc_run/features/DSC_0259_features.txt \\
      --real-features1 E:/datasets/eth3d/battle/terrace/visloc_run/features/DSC_0260_features.txt \\
      --fixtures-dir E:/visloc_archive/lightglue_onnx_m6/fixtures
"""
import argparse
import os
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import export_lightglue_onnx as lg_export  # noqa: E402

PASS_THRESHOLD = 1e-4
DESCRIPTOR_DIM = 256


def load_features_txt(path: str):
    """Parse the repo's `X Y SCORE D0 D1 ... D255` per-keypoint feature text
    format (`read_external_deep_features_txt` in Rust) into `(keypoints,
    descriptors)` numpy arrays, L2-normalising each descriptor the same way
    `export_lightglue_onnx.py`'s dummy inputs are normalised (real SuperPoint
    descriptors from this repo's exporter are already unit-norm, so this is a
    no-op in practice, but kept explicit rather than assumed).
    """
    keypoints = []
    descriptors = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if not parts:
                continue
            x, y = float(parts[0]), float(parts[1])
            desc = np.asarray(parts[3 : 3 + DESCRIPTOR_DIM], dtype=np.float32)
            assert desc.shape[0] == DESCRIPTOR_DIM, (
                f"{path}: expected {DESCRIPTOR_DIM}-d descriptor, got {desc.shape[0]}"
            )
            keypoints.append((x, y))
            descriptors.append(desc)
    kpts = np.asarray(keypoints, dtype=np.float32)
    desc = np.stack(descriptors, axis=0)
    norm = np.linalg.norm(desc, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    desc = desc / norm
    return kpts, desc


def run_case(name, model, sess, kpts0, desc0, kpts1, desc1, fixtures_dir):
    with torch.no_grad():
        torch_m0, torch_sc0 = model(
            torch.from_numpy(kpts0)[None],
            torch.from_numpy(desc0)[None],
            torch.from_numpy(kpts1)[None],
            torch.from_numpy(desc1)[None],
        )
    torch_m0 = torch_m0.numpy()
    torch_sc0 = torch_sc0.numpy()

    ort_m0, ort_sc0 = sess.run(
        ["matches0", "mscores0"],
        {
            "kpts0": kpts0[None],
            "desc0": desc0[None],
            "kpts1": kpts1[None],
            "desc1": desc1[None],
        },
    )
    # ORT/onnxruntime may return a leading batch dim of 1 depending on the
    # exporter's opset lowering of the final `[0]` index in the traced
    # module -- squeeze defensively so the comparison below is shape-safe
    # regardless (matches `lightglue_onnx.rs`'s own `squeeze_to_1d_*`
    # leniency for the same reason).
    ort_m0 = np.asarray(ort_m0).reshape(-1)
    ort_sc0 = np.asarray(ort_sc0).reshape(-1)
    torch_m0 = torch_m0.reshape(-1)
    torch_sc0 = torch_sc0.reshape(-1)

    agree = float(np.mean(torch_m0 == ort_m0)) if torch_m0.size else 1.0
    max_abs = float(np.abs(torch_sc0 - ort_sc0).max()) if torch_sc0.size else 0.0
    ok = agree == 1.0 and max_abs < PASS_THRESHOLD
    print(
        f"  [{name}] M={kpts0.shape[0]} N={kpts1.shape[0]} "
        f"matched(py)={int((torch_m0 >= 0).sum())} matched(ort)={int((ort_m0 >= 0).sum())} "
        f"index_agreement={agree * 100:.2f}%  mscores0_max_abs={max_abs:.3e}  "
        f"{'PASS' if ok else 'FAIL'}"
    )

    if fixtures_dir:
        os.makedirs(fixtures_dir, exist_ok=True)
        out_path = os.path.join(fixtures_dir, f"{name}_fixture.npz")
        np.savez(
            out_path,
            kpts0=kpts0, desc0=desc0, kpts1=kpts1, desc1=desc1,
            matches0=torch_m0.astype(np.int64), mscores0=torch_sc0.astype(np.float32),
        )
        print(f"  [{name}] wrote fixture {out_path}")

    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="the exported .onnx graph to check")
    ap.add_argument("--width", type=int, required=True, help="MUST match the --width the graph was exported with")
    ap.add_argument("--height", type=int, required=True, help="MUST match the --height the graph was exported with")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--m", type=int, default=512, help="seeded-random fixture: keypoint count, image 0")
    ap.add_argument("--n", type=int, default=480, help="seeded-random fixture: keypoint count, image 1")
    ap.add_argument("--real-features0", default=None, help="optional: real SuperPoint feature .txt, image 0")
    ap.add_argument("--real-features1", default=None, help="optional: real SuperPoint feature .txt, image 1")
    ap.add_argument("--fixtures-dir", default=None, help="if set, dump .npz fixtures here for the Rust tests")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    print(f"rebuilding PyTorch reference module (width={args.width}, height={args.height})...")
    model = lg_export.build_model(args.width, args.height)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    all_ok = True

    print("\n[seeded random]")
    kpts0, desc0, kpts1, desc1 = lg_export.dummy_inputs(args.width, args.height, args.m, args.n)
    all_ok &= run_case(
        "random", model, sess,
        kpts0[0].numpy(), desc0[0].numpy(), kpts1[0].numpy(), desc1[0].numpy(),
        args.fixtures_dir,
    )

    if args.real_features0 and args.real_features1:
        print("\n[real descriptors]")
        kpts0, desc0 = load_features_txt(args.real_features0)
        kpts1, desc1 = load_features_txt(args.real_features1)
        all_ok &= run_case("real", model, sess, kpts0, desc0, kpts1, desc1, args.fixtures_dir)
    else:
        print("\n[real descriptors] skipped (pass --real-features0/--real-features1 to include)")

    print(f"\noverall: {'ALL PASS' if all_ok else 'SOME FAILED'} (threshold {PASS_THRESHOLD:.0e} max-abs "
          f"on mscores0, exact-match required on matches0)")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
