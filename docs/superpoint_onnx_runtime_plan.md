# Phase-27 — In-Rust SuperPoint ONNX runtime plan

**Status:** implementation shipped behind opt-in feature flag
(2026-05-19, follow-up activation). `ort = "2.0.0-rc.12"` is wired
as an optional dependency under the `onnx-inference` feature on
`crates/vision`; the in-Rust extractor body is implemented; the
EuRoC demo accepts `--feature-extractor superpoint-onnx
--superpoint-onnx-model <path>`. **No model file is bundled in the
repo** — operators must download a SuperPoint ONNX export
themselves and pass its path on the CLI. Validation against a real
model file (bit-identical descriptor regression + V1_01 strict
empirical re-run) remains the next contributor step; see
*Validation plan* below.

Activation status checklist:

- ☑ `onnx-inference` feature on `crates/vision`; pass-through on root.
- ☑ `SuperPointOnnxExtractor::load_from_path` constructs an
  `ort::session::Session` at optimization Level 3.
- ☑ `DeepFeatureExtractor::extract_deep` preprocess + infer +
  postprocess (auto-detected output shapes, min-score filter,
  descending-score sort, top-K truncation, defensive L2-norm).
- ☑ EuRoC demo wiring: `FeatureExtractorKind::SuperPointOnnx`,
  `DemoExtractor::SuperPointOnnx`, `--superpoint-onnx-model
  <path>`, audit-log line, kind-match string.
- ☐ Drop a SuperPoint ONNX model into a known location (operator
  step — not bundled).
- ☐ Bit-identical descriptor regression test vs Python pre-export.
- ☐ EuRoC V1_01 strict empirical re-run reproducing 0.0029 m
  rigid ATE.
- ☐ Per-frame inference latency benchmark.

## Why this is a low-priority deliverable

Phase-26 #1 (Python SuperPoint offline pre-export) produces
descriptors that are bit-identical to what an in-Rust ONNX
implementation would produce, given the same model weights and
the same input image. The empirical signal for the EuRoC bench
is therefore unchanged between offline pre-export and in-Rust
online inference. **Phase-27 is a deployment / latency / dependency
concern, not a research signal lift.**

Concrete trade-offs:

| Path                          | Pros                                                  | Cons                                                      |
|-------------------------------|-------------------------------------------------------|-----------------------------------------------------------|
| Python `--mono-dir` pre-export (Phase-26 #1) | Empirically validated; no Rust deps changed; no model download in build | Two-step workflow; external Python + PyTorch + LightGlue dependency; ~6 min per V2_01 sequence pre-export wall on RTX-class GPU |
| In-Rust ONNX inference (Phase-27)            | One-step workflow; deployable as a single binary; per-frame online (no batch pre-export step) | New `ort` crate dependency; new ONNX Runtime native library (~50 MB if `download-binaries`, system-install otherwise); model file (~10 MB SuperPoint ONNX) needs distribution path |

Neither side is a clear win. Phase-27 should be shipped when there
is a concrete consumer that needs the one-step workflow
(e.g., a deployable binary, an integration that can't run Python at
inference time). Until then, Phase-26 #1 is the recommended path
for EuRoC benchmarking.

## Architecture (drop-in via existing trait)

The Rust side already has the right abstraction:
`DeepFeatureExtractor` trait in `crates/vision/src/features/deep.rs:137`
with output `DeepFeatureSet { keypoints, scores, descriptors }` —
**designed in Phase-13 explicitly to support an ONNX backend
without changing consumers**. The current implementer is
`HogLikeFeatureExtractor` (classical-but-deep-shaped). Phase-27
adds a second implementer: `SuperPointOnnxExtractor`.

Downstream wiring (EuRoC demo, bootstrap, tracker) already accepts
any `FeatureExtractor` via `DemoExtractor` enum. Adding a
`DemoExtractor::SuperPointOnnx` variant is mechanical (mirror the
existing `DemoExtractor::SuperPointOffline` wiring).

## Cargo dependency

Add to `Cargo.toml` of the crate that owns the new extractor
(recommended: `crates/vision/`):

```toml
[features]
default = []
# Phase-27: in-Rust ONNX inference for SuperPoint (and future
# learned-feature backends). Off by default; pulls ~50 MB of
# ONNX Runtime native libraries via the `download-binaries`
# sub-feature (alternative: install onnxruntime system-wide
# and drop that sub-feature to use the system copy).
onnx-inference = ["dep:ort"]

[dependencies]
ort = { version = "2.0", optional = true, default-features = false, features = ["download-binaries", "ndarray"] }
ndarray = { version = "0.16", optional = true }
```

The `ort` crate at v2.x (as of 2024+) uses ONNX Runtime 1.17+. Pin
the version when shipping to avoid CI breakage from API churn.
`download-binaries` is the cheapest path for first-time users;
production deployments should switch to system-installed ORT for
faster builds and smaller binaries.

## Model file

SuperPoint ONNX is available from:

- `magic-leap-research/SuperPoint` (the original SuperPoint authors)
  - Released as PyTorch checkpoint; ONNX conversion is well-documented
- `fabio-sim/LightGlue-ONNX` releases include a SuperPoint ONNX
  matched against their LightGlue ONNX (good if a future Phase-28
  ports LightGlue too)
- `pytorch/vision` and HuggingFace hubs also host community SuperPoint
  ONNX exports

The model file is ~10 MB. **Do not bundle in this repo.** Document
in the extractor:

```rust
/// Path to the SuperPoint ONNX model. Download once from the
/// upstream release (e.g. `magic-leap-research/SuperPoint`) and
/// pass the path via `SuperPointOnnxExtractor::load_from_path`.
/// A sanity check ensures the loaded model's input/output shapes
/// match SuperPoint's contract (input `(1, 1, H, W) f32`, outputs
/// `keypoints (N, 2) i64`, `scores (N,) f32`, `descriptors (256, N) f32`).
```

For the EuRoC bench wiring add a CLI flag
`--superpoint-onnx-model <path>` on `examples/euroc_online_slam_vi_image_demo.rs`.
A future helper script `scripts/fetch_superpoint_onnx.sh` could
download the model into `models/superpoint.onnx` with a checksum
verification step.

## Extractor implementation sketch

File: `crates/vision/src/features/superpoint_onnx.rs` (skeleton
already in this commit).

```rust
#[cfg(feature = "onnx-inference")]
pub struct SuperPointOnnxExtractor {
    session: ort::Session,
    config: SuperPointOnnxConfig,
}

#[cfg(feature = "onnx-inference")]
impl SuperPointOnnxExtractor {
    pub fn load_from_path<P: AsRef<std::path::Path>>(
        path: P,
        config: SuperPointOnnxConfig,
    ) -> Result<Self, SuperPointOnnxError> {
        let session = ort::Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .commit_from_file(path)?;
        validate_session_io(&session)?; // shape sanity checks
        Ok(Self { session, config })
    }
}

#[cfg(feature = "onnx-inference")]
impl DeepFeatureExtractor for SuperPointOnnxExtractor {
    type Image = GrayscaleImage;
    type Error = SuperPointOnnxError;

    fn extract_deep(&self, image: &Self::Image) -> Result<DeepFeatureSet, Self::Error> {
        // 1. Preprocess: grayscale f32 in [0, 1], shape (1, 1, H, W).
        let input_tensor = preprocess_image(image, &self.config)?;
        // 2. Run inference.
        let outputs = self.session.run(ort::inputs![input_tensor]?)?;
        // 3. Postprocess: read keypoints (i64 N,2), scores (f32 N), descriptors (f32 256,N).
        //    L2-normalise descriptors so the existing MutualSoftmaxMatcher
        //    treats inner products as cosine similarity (the offline
        //    Python pre-export already does this — match it).
        let (keypoints, scores, descriptors) = postprocess_outputs(outputs)?;
        DeepFeatureSet::new(keypoints, scores, descriptors).map_err(Into::into)
    }
}
```

The preprocessing / postprocessing helpers should match
`scripts/export_superpoint_lightglue.py` byte-for-byte so that
the in-Rust path produces descriptors interchangeable with the
offline replay. **A regression test fixture** (one small image,
one expected DeepFeatureSet) is the first thing to add when this
ships.

## EuRoC demo wiring

Mirror the existing `SuperPointOffline` integration in
`examples/euroc_online_slam_vi_image_demo.rs`:

1. `enum FeatureExtractorKind` gains `SuperPointOnnx`.
2. `enum DemoExtractor` gains `SuperPointOnnx(SuperPointOnnxExtractor)`.
3. New CLI flag `--superpoint-onnx-model <path>` (required when
   `--feature-extractor superpoint-onnx` is set).
4. The seed-frame extraction + per-frame extraction wiring is
   already trait-generic; nothing else changes.
5. New audit line `superpoint_onnx_model_path=<path>`.

Cam0+cam1 wiring is identical to the offline variant (both cameras
run through the same extractor; the seed-frame helper picks the
camera via `set_camera`).

## Validation plan

After implementation, before claiming "in-Rust SuperPoint works":

1. **Bit-identical descriptor regression test.** Pre-export
   features for V2_01 cam0 frame 0 via Python; run the in-Rust
   extractor on the same image with the same ONNX model; assert
   per-keypoint position match within 0.01 px and descriptor
   match within 1e-4 (the two paths share the same underlying
   ONNX Runtime computation, so differences should be in the
   pre/post-processing only).
2. **EuRoC V1_01 strict empirical re-run.** Rerun the Phase-26 #1
   V1_01 strict config with `--feature-extractor superpoint-onnx`
   instead of `--feature-extractor superpoint-offline`. Assert
   rigid ATE matches the offline result within numerical noise
   (10⁻⁴ m).
3. **Performance benchmark.** Per-frame inference latency on the
   target deployment hardware. SuperPoint ONNX with ORT's CPU
   provider is ~40-80 ms / frame on modern x86 CPUs; with the
   CUDA execution provider ~5-10 ms / frame. Compare against the
   Python pre-export wall-time (~250 ms / frame on RTX 3060 for
   the full export-to-text pipeline) to characterise the
   real-time deployment win.

## Out of scope

- LightGlue ONNX. The existing `MutualSoftmaxMatcher` is a
  LightGlue-style stand-in and Phase-26 #3b showed it does not
  improve V-class cliff matching beyond `BruteForceMatcher` +
  cross-check. A LightGlue ONNX port may be useful for
  empirically validating Phase-26 #3b on a real LightGlue
  forward pass, but the empirical signal is unlikely to change.
- Multi-resolution / image-pyramid SuperPoint. The Phase-15 / #26
  pre-export runs single-scale at the EuRoC native 752×480
  resolution; the ONNX path should match.
- Mobile / quantised SuperPoint. Phase-27 targets the desktop /
  server profile. Mobile deployment is a separate workstream.
- TensorRT / OpenVINO / CoreML execution providers. ORT's CPU
  and CUDA providers cover the common cases; specialised EPs
  are a downstream optimisation.

## Decision gates

Before merging Phase-27, the contributor should be able to answer:

- Does the in-Rust extractor produce DeepFeatureSets that match
  the Python pre-export byte-for-byte (within FP tolerance)?
- Does the EuRoC V1_01 strict run reproduce the Phase-26 #1
  0.0029 m rigid ATE under the same binary build?
- Does the inference latency meet the deployment target?
- Is the `ort` crate dependency footprint (build time + runtime
  library size) acceptable for the use case?

If any answer is "no", the offline pre-export remains the
recommended path; ship the integration as opt-in only.

## Path forward

This document is the contract. The minimal sequence to complete
Phase-27 once the use case demands it:

1. Add `ort` + `ndarray` to `crates/vision/Cargo.toml` under the
   `onnx-inference` feature.
2. Activate `crates/vision/src/features/superpoint_onnx.rs`
   (currently a skeleton — see the file).
3. Add cam0+cam1 plumbing in the EuRoC demo (mirror
   `SuperPointOffline`).
4. Drop a SuperPoint ONNX model into a known location.
5. Validate per the validation plan above.
6. Update Phase-26 #1's recommended-config documentation with
   the one-step alternative.

Reasonable wall-time estimate for an experienced Rust + ML
engineer with `ort` familiarity: 4-8 hours including model
download, pre/post-processing parity work, and validation runs.
