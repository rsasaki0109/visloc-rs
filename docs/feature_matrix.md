# Feature Matrix

`visloc-rs` keeps the default build lightweight. Optional image and model
runtimes are feature-gated so visual-localization users can choose the smallest
dependency surface that matches their deployment.

| Feature set | Support tier | CI expectation | MSRV | Notes |
| --- | --- | --- | --- | --- |
| `--no-default-features` | Tier 1 | Linux + Windows `cargo check --workspace --all-targets --no-default-features` | Rust 1.82 | Core crates, algorithms, pipelines, and dependency-light examples. |
| default | Tier 1 | Linux + Windows `cargo check --workspace --all-targets` | Rust 1.82 | Same lightweight surface; default features are intentionally empty. |
| `image-io` | Tier 1 | Linux + Windows `cargo check --workspace --all-targets --features image-io`; MSRV check in `scripts/check_msrv.sh` | Rust 1.82 | PNG/JPEG loading, common image sequences, KITTI image-directory helpers, and user-facing image examples. |
| `onnx-inference` | Tier 2 opt-in | `VISLOC_CHECK_ONNX=1 sh scripts/check_feature_matrix.sh` on a current stable Rust | Not part of the Rust 1.82 MSRV guarantee | Pulls ONNX Runtime through `ort`; model files and hashes must be recorded in benchmark manifests. |
| `onnx-cuda` | Tier 2 hardware-gated | `VISLOC_CHECK_ONNX_CUDA=1 sh scripts/check_feature_matrix.sh` on CUDA-capable hosts | Not part of the Rust 1.82 MSRV guarantee | Requires CUDA/cuDNN-compatible ONNX Runtime setup and remains outside the default CI gate. |

## Policy

- Tier 1 features must not require OpenCV, PyTorch, ONNX Runtime, CUDA, or
  implicit dataset/model downloads.
- Optional native/runtime features must stay behind explicit Cargo features.
- Benchmark claims that use optional learned models must record the enabled
  features, command, model hash, dataset identity, commit, and metric
  implementation in the benchmark registry.
- Adding a public feature requires updating this file, `docs/api_stability.md`,
  and the CI or opt-in validation command that covers it.
- `tests/test_feature_matrix.py` keeps this table aligned with the root
  `Cargo.toml`, `scripts/check_feature_matrix.sh`, and the Tier 1 CI matrix.

## Local Checks

```bash
python -m unittest tests.test_feature_matrix
sh scripts/check_feature_matrix.sh
VISLOC_CHECK_ONNX=1 sh scripts/check_feature_matrix.sh
VISLOC_CHECK_ONNX_CUDA=1 sh scripts/check_feature_matrix.sh
```
