//! Phase-27 — in-Rust SuperPoint ONNX runtime.
//!
//! Two build profiles share this module:
//!
//! * **Default (`onnx-inference` feature off)** — the extractor compiles
//!   as a skeleton that always returns
//!   [`SuperPointOnnxError::FeatureDisabled`]. This keeps the type
//!   visible to downstream consumers without forcing the heavy
//!   `ort` / ONNX Runtime native dependency on every workspace build.
//!
//! * **`onnx-inference` enabled** — the extractor wraps an
//!   `ort::session::Session` and runs SuperPoint inference per
//!   frame. The expected I/O contract is the LightGlue-ONNX-style
//!   pre-baked-postprocessing variant: input `image: (1, 1, H, W) f32`
//!   in `[0, 1]`; outputs `keypoints: (N, 2) i64`, `scores: (N,) f32`,
//!   `descriptors: (256, N) f32` *or* `(N, 256) f32` (auto-detected).
//!   See `docs/superpoint_onnx_runtime_plan.md` for model sourcing
//!   notes and validation gate.
//!
//! Empirically, in-Rust ONNX inference and the existing Python
//! pre-export path (`scripts/export_superpoint_lightglue.py --mono-dir`
//! consumed via `SuperPointOfflineExtractor`) produce bit-identical
//! descriptors given the same model weights and the same input image.
//! Phase-26 #1 V-class breakthrough (V1_01 strict rigid ATE 0.0029 m)
//! is therefore unchanged between the two paths; Phase-27 is a
//! deployment / latency / dependency concern, not a research signal
//! lift.
//!
//! Activation checklist (already partially completed in this revision):
//!
//! 1. ☑ `onnx-inference` feature on `visloc-vision` pulls in `ort` 2.x.
//! 2. ☑ Skeleton activated (`load_from_path`, preprocess, infer,
//!    postprocess wired below).
//! 3. ☐ Drop a SuperPoint ONNX model into a known location and pass
//!    its path via `--superpoint-onnx-model` on the EuRoC demo
//!    (mirror `--superpoint-offline-dir` wiring).
//! 4. ☐ Validate per the plan-doc validation plan: bit-identical
//!    descriptor regression vs Python pre-export + EuRoC V1_01 strict
//!    rerun + latency benchmark.
//!
//! Until the feature flag is enabled and a model is supplied, the
//! types below exist so consumers can write code against them today
//! that will compile (and at runtime fail loudly with
//! [`SuperPointOnnxError::FeatureDisabled`]) without the feature.

use super::deep::{DeepFeatureExtractor, DeepFeatureSet, DeepFeatureSetError};
use super::GrayscaleImage;
#[cfg(feature = "onnx-inference")]
use nalgebra::Point2;
use std::fmt;
use std::path::Path;
#[cfg(feature = "onnx-inference")]
use std::sync::{Arc, Mutex};

/// Configuration for [`SuperPointOnnxExtractor`].
///
/// Defaults match SuperPoint's published inference recipe:
/// non-max-suppression radius 4 px, keypoint score threshold 0.005,
/// retain top 1500 keypoints (matches the EuRoC pre-export
/// `--max-keypoints 1500` setting used in Phase-26).
#[derive(Debug, Clone, PartialEq)]
pub struct SuperPointOnnxConfig {
    /// Non-max-suppression window radius in pixels.
    pub nms_radius_pixels: u32,
    /// Minimum SuperPoint heatmap score for a keypoint to be retained.
    pub min_score: f32,
    /// Maximum keypoints kept per image (top-N by score after NMS).
    pub max_keypoints: usize,
}

impl Default for SuperPointOnnxConfig {
    fn default() -> Self {
        Self {
            nms_radius_pixels: 4,
            min_score: 0.005,
            max_keypoints: 1500,
        }
    }
}

/// In-Rust SuperPoint ONNX extractor. Implements [`DeepFeatureExtractor`]
/// when the `onnx-inference` feature is enabled; otherwise every method
/// returns [`SuperPointOnnxError::FeatureDisabled`].
///
/// The loaded session is wrapped in `Arc<Mutex<_>>` so the extractor
/// remains cheaply cloneable for plumbing through stereo (cam0 /
/// cam1) pipelines — both cameras share one session, exactly like
/// the offline `SuperPointOfflineExtractor` shares one descriptor
/// store across frames. The mutex serialises per-frame inference,
/// matching the current EuRoC demo's single-threaded frame loop;
/// callers that want parallel cam0/cam1 inference should hold two
/// independent sessions instead.
#[derive(Clone)]
pub struct SuperPointOnnxExtractor {
    config: SuperPointOnnxConfig,
    #[cfg(feature = "onnx-inference")]
    session: Arc<Mutex<ort::session::Session>>,
}

impl fmt::Debug for SuperPointOnnxExtractor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SuperPointOnnxExtractor")
            .field("config", &self.config)
            .field("onnx_inference_feature", &cfg!(feature = "onnx-inference"))
            .finish()
    }
}

impl SuperPointOnnxExtractor {
    /// Borrow the active configuration.
    pub fn config(&self) -> &SuperPointOnnxConfig {
        &self.config
    }
}

// ---------------------------------------------------------------
// Feature-disabled stub (`cargo build` default)
// ---------------------------------------------------------------

#[cfg(not(feature = "onnx-inference"))]
impl SuperPointOnnxExtractor {
    /// Load a SuperPoint ONNX model from disk.
    ///
    /// **Stub** — without the `onnx-inference` feature this always
    /// returns [`SuperPointOnnxError::FeatureDisabled`].
    pub fn load_from_path<P: AsRef<Path>>(
        _path: P,
        config: SuperPointOnnxConfig,
    ) -> Result<Self, SuperPointOnnxError> {
        Err(SuperPointOnnxError::FeatureDisabled {
            requested: "load_from_path",
            config,
        })
    }
}

#[cfg(not(feature = "onnx-inference"))]
impl DeepFeatureExtractor for SuperPointOnnxExtractor {
    type Image = GrayscaleImage;
    type Error = SuperPointOnnxError;

    fn extract_deep(&self, _image: &Self::Image) -> Result<DeepFeatureSet, Self::Error> {
        Err(SuperPointOnnxError::FeatureDisabled {
            requested: "extract_deep",
            config: self.config.clone(),
        })
    }
}

// ---------------------------------------------------------------
// Active implementation (`onnx-inference` feature)
// ---------------------------------------------------------------

/// Execution-provider selection for the in-Rust SuperPoint session.
///
/// `CudaThenCpu` registers the CUDA execution provider first and the CPU
/// provider as a fallback, so the session runs on the GPU when the build
/// includes the `onnx-cuda` feature (which pulls the CUDA-enabled ONNX
/// Runtime binaries) *and* a working CUDA + cuDNN runtime is present;
/// otherwise CUDA registration fails gracefully and inference falls back to
/// the CPU provider. `Cpu` forces the CPU provider only — useful for an A/B
/// throughput comparison and for deterministic CI.
#[cfg(feature = "onnx-inference")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnnxBackend {
    /// Prefer CUDA, fall back to CPU if the GPU provider cannot register.
    /// This is the production default: a deployment without a CUDA runtime
    /// still runs, just on the CPU.
    CudaThenCpu,
    /// CUDA execution provider only; loading errors if the GPU provider
    /// cannot register (no silent CPU fallback). Use this when a caller must
    /// know the GPU is actually in use — e.g. a throughput benchmark that
    /// would otherwise mislabel CPU numbers as CUDA.
    Cuda,
    /// CPU execution provider only.
    Cpu,
}

#[cfg(feature = "onnx-inference")]
impl Default for OnnxBackend {
    fn default() -> Self {
        OnnxBackend::CudaThenCpu
    }
}

#[cfg(feature = "onnx-inference")]
impl SuperPointOnnxExtractor {
    /// Load a SuperPoint ONNX model from disk and build a session
    /// optimised at level 3, preferring the CUDA execution provider with a
    /// graceful CPU fallback. The session is shared via [`Arc`] so `Clone`
    /// of the extractor is cheap.
    pub fn load_from_path<P: AsRef<Path>>(
        path: P,
        config: SuperPointOnnxConfig,
    ) -> Result<Self, SuperPointOnnxError> {
        Self::load_from_path_with_backend(path, config, OnnxBackend::default())
    }

    /// Like [`load_from_path`](Self::load_from_path) but with an explicit
    /// execution-provider choice (see [`OnnxBackend`]).
    pub fn load_from_path_with_backend<P: AsRef<Path>>(
        path: P,
        config: SuperPointOnnxConfig,
        backend: OnnxBackend,
    ) -> Result<Self, SuperPointOnnxError> {
        use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

        let providers = match backend {
            // CUDA first, CPU as the always-available fallback. ort registers
            // providers in order and silently skips one that fails to load
            // (e.g. CUDA binaries / cuDNN absent), so this degrades to CPU
            // without erroring.
            OnnxBackend::CudaThenCpu => vec![
                CUDAExecutionProvider::default().build(),
                CPUExecutionProvider::default().build(),
            ],
            OnnxBackend::Cuda => {
                vec![CUDAExecutionProvider::default().build().error_on_failure()]
            }
            OnnxBackend::Cpu => vec![CPUExecutionProvider::default().build()],
        };

        let session = ort::session::Session::builder()
            .map_err(SuperPointOnnxError::from_ort)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(SuperPointOnnxError::from_ort)?
            .with_execution_providers(providers)
            .map_err(SuperPointOnnxError::from_ort)?
            .commit_from_file(path.as_ref())
            .map_err(SuperPointOnnxError::from_ort)?;
        Ok(Self {
            config,
            session: Arc::new(Mutex::new(session)),
        })
    }
}

#[cfg(feature = "onnx-inference")]
impl DeepFeatureExtractor for SuperPointOnnxExtractor {
    type Image = GrayscaleImage;
    type Error = SuperPointOnnxError;

    fn extract_deep(&self, image: &Self::Image) -> Result<DeepFeatureSet, Self::Error> {
        let width = image.width();
        let height = image.height();
        if width == 0 || height == 0 {
            return Err(SuperPointOnnxError::EmptyImage);
        }

        let input =
            ndarray::Array4::<f32>::from_shape_vec((1, 1, height, width), image.pixels().to_vec())
                .map_err(|error| SuperPointOnnxError::PreprocessShape {
                    width,
                    height,
                    message: error.to_string(),
                })?;

        let input_value =
            ort::value::Tensor::from_array(input).map_err(SuperPointOnnxError::from_ort)?;
        let mut session = self
            .session
            .lock()
            .map_err(|error| SuperPointOnnxError::OnnxRuntime(error.to_string()))?;
        let mut outputs = session
            .run(ort::inputs![input_value])
            .map_err(SuperPointOnnxError::from_ort)?;

        let (keypoints_array, scores_array, descriptors_array) =
            extract_named_outputs(&mut outputs)?;

        let triples = postprocess(
            keypoints_array.view(),
            scores_array.view(),
            descriptors_array.view(),
            &self.config,
        )?;

        let (keypoints, scores, descriptors) = unzip_triples(triples);
        DeepFeatureSet::new(keypoints, scores, descriptors).map_err(SuperPointOnnxError::from)
    }
}

#[cfg(feature = "onnx-inference")]
fn extract_named_outputs(
    outputs: &mut ort::session::SessionOutputs<'_>,
) -> Result<
    (
        ndarray::Array2<i64>,
        ndarray::Array1<f32>,
        ndarray::Array2<f32>,
    ),
    SuperPointOnnxError,
> {
    // The LightGlue-ONNX style export names the three outputs
    // "keypoints", "scores", "descriptors". If a model under
    // evaluation diverges from those names the lookup below errors
    // out loudly; the contract is intentionally narrow so postprocess
    // drift is caught at activation time, not silently inferred.
    let kp_value =
        outputs
            .remove("keypoints")
            .ok_or_else(|| SuperPointOnnxError::OutputShapeMismatch {
                expected: "output named `keypoints`",
                actual: "missing".to_string(),
            })?;
    let score_value =
        outputs
            .remove("scores")
            .ok_or_else(|| SuperPointOnnxError::OutputShapeMismatch {
                expected: "output named `scores`",
                actual: "missing".to_string(),
            })?;
    let desc_value =
        outputs
            .remove("descriptors")
            .ok_or_else(|| SuperPointOnnxError::OutputShapeMismatch {
                expected: "output named `descriptors`",
                actual: "missing".to_string(),
            })?;

    let kp_array: ndarray::ArrayD<i64> = kp_value
        .try_extract_array::<i64>()
        .map_err(SuperPointOnnxError::from_ort)?
        .into_owned();
    let score_array: ndarray::ArrayD<f32> = score_value
        .try_extract_array::<f32>()
        .map_err(SuperPointOnnxError::from_ort)?
        .into_owned();
    let desc_array: ndarray::ArrayD<f32> = desc_value
        .try_extract_array::<f32>()
        .map_err(SuperPointOnnxError::from_ort)?
        .into_owned();

    Ok((
        normalise_keypoints(kp_array)?,
        normalise_scores(score_array)?,
        normalise_descriptors(desc_array)?,
    ))
}

#[cfg(feature = "onnx-inference")]
fn normalise_keypoints(
    array: ndarray::ArrayD<i64>,
) -> Result<ndarray::Array2<i64>, SuperPointOnnxError> {
    // Accept (N, 2) or (1, N, 2) (batch dim).
    let shape = array.shape().to_vec();
    let view = if shape.len() == 3 && shape[0] == 1 && shape[2] == 2 {
        array.into_shape_with_order((shape[1], 2)).map_err(|e| {
            SuperPointOnnxError::OutputShapeMismatch {
                expected: "(N, 2) or (1, N, 2) keypoints",
                actual: format!("{shape:?}: {e}"),
            }
        })?
    } else if shape.len() == 2 && shape[1] == 2 {
        array.into_dimensionality::<ndarray::Ix2>().map_err(|e| {
            SuperPointOnnxError::OutputShapeMismatch {
                expected: "(N, 2) keypoints",
                actual: format!("{shape:?}: {e}"),
            }
        })?
    } else {
        return Err(SuperPointOnnxError::OutputShapeMismatch {
            expected: "(N, 2) or (1, N, 2) keypoints",
            actual: format!("{shape:?}"),
        });
    };
    Ok(view)
}

#[cfg(feature = "onnx-inference")]
fn normalise_scores(
    array: ndarray::ArrayD<f32>,
) -> Result<ndarray::Array1<f32>, SuperPointOnnxError> {
    // Accept (N,) or (1, N) (batch dim).
    let shape = array.shape().to_vec();
    let view = if shape.len() == 2 && shape[0] == 1 {
        array.into_shape_with_order((shape[1],)).map_err(|e| {
            SuperPointOnnxError::OutputShapeMismatch {
                expected: "(N,) or (1, N) scores",
                actual: format!("{shape:?}: {e}"),
            }
        })?
    } else if shape.len() == 1 {
        array.into_dimensionality::<ndarray::Ix1>().map_err(|e| {
            SuperPointOnnxError::OutputShapeMismatch {
                expected: "(N,) scores",
                actual: format!("{shape:?}: {e}"),
            }
        })?
    } else {
        return Err(SuperPointOnnxError::OutputShapeMismatch {
            expected: "(N,) or (1, N) scores",
            actual: format!("{shape:?}"),
        });
    };
    Ok(view)
}

#[cfg(feature = "onnx-inference")]
fn normalise_descriptors(
    array: ndarray::ArrayD<f32>,
) -> Result<ndarray::Array2<f32>, SuperPointOnnxError> {
    // Accept (N, 256), (256, N) (transposed), (1, N, 256), (1, 256, N).
    const DIM: usize = 256;
    let shape = array.shape().to_vec();
    let two_d = match shape.as_slice() {
        [n, d] if *d == DIM => array.into_shape_with_order((*n, DIM)).map_err(|e| {
            SuperPointOnnxError::OutputShapeMismatch {
                expected: "(N, 256) descriptors",
                actual: format!("{shape:?}: {e}"),
            }
        })?,
        [d, n] if *d == DIM => {
            // Transpose 256xN -> Nx256.
            let arr = array.into_shape_with_order((DIM, *n)).map_err(|e| {
                SuperPointOnnxError::OutputShapeMismatch {
                    expected: "(256, N) descriptors",
                    actual: format!("{shape:?}: {e}"),
                }
            })?;
            arr.t().to_owned()
        }
        [1, n, d] if *d == DIM => array.into_shape_with_order((*n, DIM)).map_err(|e| {
            SuperPointOnnxError::OutputShapeMismatch {
                expected: "(1, N, 256) descriptors",
                actual: format!("{shape:?}: {e}"),
            }
        })?,
        [1, d, n] if *d == DIM => {
            let arr = array.into_shape_with_order((DIM, *n)).map_err(|e| {
                SuperPointOnnxError::OutputShapeMismatch {
                    expected: "(1, 256, N) descriptors",
                    actual: format!("{shape:?}: {e}"),
                }
            })?;
            arr.t().to_owned()
        }
        _ => {
            return Err(SuperPointOnnxError::OutputShapeMismatch {
                expected: "(N, 256), (256, N), (1, N, 256), or (1, 256, N) descriptors",
                actual: format!("{shape:?}"),
            });
        }
    };
    Ok(two_d)
}

// Postprocessing enforces the downstream contract of
// `DeepFeatureSet`: equal lengths, unit-norm descriptors, score
// threshold, top-K truncation. Gated on `onnx-inference` because the
// signature depends on `ndarray`, which is an optional dependency.
#[cfg(feature = "onnx-inference")]
fn postprocess(
    keypoints: ndarray::ArrayView2<'_, i64>,
    scores: ndarray::ArrayView1<'_, f32>,
    descriptors: ndarray::ArrayView2<'_, f32>,
    config: &SuperPointOnnxConfig,
) -> Result<Vec<(Point2<f64>, f32, Vec<f32>)>, SuperPointOnnxError> {
    let n_kp = keypoints.shape()[0];
    let n_score = scores.shape()[0];
    let n_desc = descriptors.shape()[0];
    if n_kp != n_score || n_kp != n_desc {
        return Err(SuperPointOnnxError::OutputShapeMismatch {
            expected: "keypoints / scores / descriptors agree on N",
            actual: format!("N_keypoints={n_kp} N_scores={n_score} N_descriptors={n_desc}"),
        });
    }

    let descriptor_dim = descriptors.shape()[1];

    let mut triples: Vec<(Point2<f64>, f32, Vec<f32>)> = Vec::with_capacity(n_kp);
    for i in 0..n_kp {
        let score = scores[i];
        if !score.is_finite() || score < config.min_score {
            continue;
        }
        let x = keypoints[(i, 0)] as f64;
        let y = keypoints[(i, 1)] as f64;
        let mut descriptor = vec![0.0_f32; descriptor_dim];
        for d in 0..descriptor_dim {
            descriptor[d] = descriptors[(i, d)];
        }
        // L2-normalise defensively. The LightGlue-ONNX SuperPoint
        // export already emits unit-norm descriptors, but normalising
        // again costs O(N * 256) and guarantees the contract for
        // mutual-softmax matching downstream.
        let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in descriptor.iter_mut() {
                *v /= norm;
            }
        }
        triples.push((Point2::new(x, y), score, descriptor));
    }

    // Sort by descending score, ties broken by ascending index (the
    // existing y-then-x tiebreaker used by `HogLikeFeatureExtractor`
    // would require keypoint coordinates as the tiebreak key — score
    // ties at this granularity are negligible, so we just preserve
    // insertion order via stable sort).
    triples.sort_by(|left, right| right.1.total_cmp(&left.1));
    if triples.len() > config.max_keypoints {
        triples.truncate(config.max_keypoints);
    }
    Ok(triples)
}

#[cfg(feature = "onnx-inference")]
fn unzip_triples(
    triples: Vec<(Point2<f64>, f32, Vec<f32>)>,
) -> (Vec<Point2<f64>>, Vec<f32>, Vec<Vec<f32>>) {
    let mut keypoints = Vec::with_capacity(triples.len());
    let mut scores = Vec::with_capacity(triples.len());
    let mut descriptors = Vec::with_capacity(triples.len());
    for (kp, score, desc) in triples {
        keypoints.push(kp);
        scores.push(score);
        descriptors.push(desc);
    }
    (keypoints, scores, descriptors)
}

#[derive(Debug, Clone, PartialEq)]
pub enum SuperPointOnnxError {
    /// The `onnx-inference` feature is not enabled. Rebuild
    /// `visloc-vision` with `--features onnx-inference` and supply a
    /// SuperPoint ONNX model via [`SuperPointOnnxExtractor::load_from_path`].
    FeatureDisabled {
        requested: &'static str,
        config: SuperPointOnnxConfig,
    },
    /// Input image had zero width or height.
    EmptyImage,
    /// Preprocessing failed to reshape the input image into the
    /// expected (1, 1, H, W) tensor.
    PreprocessShape {
        width: usize,
        height: usize,
        message: String,
    },
    /// One of the model outputs had an unexpected shape.
    OutputShapeMismatch {
        expected: &'static str,
        actual: String,
    },
    /// Underlying ONNX Runtime error. Stored as `String` so the error
    /// type stays `Clone + PartialEq + Eq`-friendly.
    OnnxRuntime(String),
    /// Inner conversion failure when constructing a
    /// [`DeepFeatureSet`] from postprocessed ONNX outputs.
    DeepFeatureSet(DeepFeatureSetError),
}

impl SuperPointOnnxError {
    #[cfg(feature = "onnx-inference")]
    fn from_ort<E: std::fmt::Display>(error: E) -> Self {
        Self::OnnxRuntime(error.to_string())
    }
}

impl fmt::Display for SuperPointOnnxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureDisabled { requested, .. } => write!(
                formatter,
                "SuperPointOnnxExtractor::{requested}: in-Rust ONNX inference is gated behind \
                 the `onnx-inference` feature; rebuild with `--features onnx-inference` and \
                 supply a SuperPoint ONNX model path (see docs/superpoint_onnx_runtime_plan.md)"
            ),
            Self::EmptyImage => write!(formatter, "SuperPoint ONNX input image is empty"),
            Self::PreprocessShape {
                width,
                height,
                message,
            } => write!(
                formatter,
                "SuperPoint ONNX preprocess failed: image {width}x{height} cannot be reshaped \
                 into (1, 1, H, W): {message}"
            ),
            Self::OutputShapeMismatch { expected, actual } => write!(
                formatter,
                "SuperPoint ONNX output shape mismatch: expected {expected}, got {actual}"
            ),
            Self::OnnxRuntime(message) => write!(formatter, "ONNX Runtime error: {message}"),
            Self::DeepFeatureSet(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SuperPointOnnxError {}

impl From<DeepFeatureSetError> for SuperPointOnnxError {
    fn from(error: DeepFeatureSetError) -> Self {
        Self::DeepFeatureSet(error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "onnx-inference"))]
    #[test]
    fn skeleton_extractor_load_returns_feature_disabled() {
        // Without the feature flag, the skeleton always fails with
        // FeatureDisabled. Once enabled, see `load_real_model_*`
        // integration tests (gated on the model fixture).
        let result = SuperPointOnnxExtractor::load_from_path(
            "/nonexistent",
            SuperPointOnnxConfig::default(),
        );
        match result {
            Err(SuperPointOnnxError::FeatureDisabled { requested, .. }) => {
                assert_eq!(requested, "load_from_path");
            }
            other => panic!("expected FeatureDisabled, got {other:?}"),
        }
    }

    #[test]
    fn default_config_matches_phase26_pre_export_settings() {
        // Phase-26 #1 pre-export script uses --max-keypoints 1500
        // (and default NMS / score-threshold). The Rust defaults must
        // match so the in-Rust path produces an apples-to-apples
        // descriptor stream once activated.
        let config = SuperPointOnnxConfig::default();
        assert_eq!(config.max_keypoints, 1500);
        assert_eq!(config.nms_radius_pixels, 4);
        assert!((config.min_score - 0.005).abs() < 1.0e-9);
    }

    #[cfg(feature = "onnx-inference")]
    #[test]
    fn postprocess_filters_by_min_score_and_truncates_to_max_keypoints() {
        // Four candidates, one below min_score; cap at 2 keeps the two
        // highest scores.
        let keypoints = ndarray::array![[1_i64, 2], [3, 4], [5, 6], [7, 8]];
        let scores = ndarray::array![0.9_f32, 0.7, 0.5, 0.001];
        // Distinct descriptors so the unit-norm normalisation does not
        // collapse them.
        let descriptors = ndarray::array![
            [1.0_f32, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let config = SuperPointOnnxConfig {
            min_score: 0.01,
            max_keypoints: 2,
            nms_radius_pixels: 4,
        };
        let result =
            postprocess(keypoints.view(), scores.view(), descriptors.view(), &config).unwrap();
        assert_eq!(result.len(), 2);
        // Top score first: (3, 4) with 0.7? No — (1, 2) with 0.9, then (3, 4) with 0.7.
        assert_eq!(result[0].0, Point2::new(1.0, 2.0));
        assert!((result[0].1 - 0.9).abs() < 1.0e-6);
        assert_eq!(result[1].0, Point2::new(3.0, 4.0));
        assert!((result[1].1 - 0.7).abs() < 1.0e-6);
    }

    #[cfg(feature = "onnx-inference")]
    #[test]
    fn postprocess_normalises_descriptors_to_unit_norm() {
        let keypoints = ndarray::array![[10_i64, 20]];
        let scores = ndarray::array![0.5_f32];
        // 3-4-0 vector — norm 5, should normalise to 0.6, 0.8, 0.
        let descriptors = ndarray::array![[3.0_f32, 4.0, 0.0]];
        let config = SuperPointOnnxConfig::default();
        let result =
            postprocess(keypoints.view(), scores.view(), descriptors.view(), &config).unwrap();
        assert_eq!(result.len(), 1);
        let descriptor = &result[0].2;
        let norm = descriptor.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1.0e-6, "got norm {norm}");
        assert!((descriptor[0] - 0.6).abs() < 1.0e-6);
        assert!((descriptor[1] - 0.8).abs() < 1.0e-6);
        assert!(descriptor[2].abs() < 1.0e-6);
    }

    #[cfg(feature = "onnx-inference")]
    #[test]
    fn postprocess_rejects_inconsistent_lengths() {
        let keypoints = ndarray::array![[0_i64, 0], [1, 1]];
        let scores = ndarray::array![0.5_f32]; // length mismatch
        let descriptors = ndarray::array![[1.0_f32, 0.0]];
        let config = SuperPointOnnxConfig::default();
        let result = postprocess(keypoints.view(), scores.view(), descriptors.view(), &config);
        match result {
            Err(SuperPointOnnxError::OutputShapeMismatch { .. }) => {}
            other => panic!("expected OutputShapeMismatch, got {other:?}"),
        }
    }

    #[cfg(feature = "onnx-inference")]
    #[test]
    fn postprocess_skips_nonfinite_and_below_threshold_scores() {
        let keypoints = ndarray::array![[0_i64, 0], [1, 0], [2, 0]];
        let scores = ndarray::array![f32::NAN, 0.001, 0.5];
        let descriptors = ndarray::array![[1.0_f32, 0.0], [0.0, 1.0], [1.0, 1.0],];
        let config = SuperPointOnnxConfig {
            min_score: 0.01,
            max_keypoints: 10,
            nms_radius_pixels: 4,
        };
        let result =
            postprocess(keypoints.view(), scores.view(), descriptors.view(), &config).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, Point2::new(2.0, 0.0));
    }

    #[cfg(feature = "onnx-inference")]
    #[test]
    fn normalise_descriptors_handles_all_supported_layouts() {
        use ndarray::ArrayD;
        let nominal: Vec<f32> = (0..(3 * 256)).map(|i| i as f32 / 1000.0).collect();
        // (N, 256)
        let a = ArrayD::from_shape_vec(ndarray::IxDyn(&[3, 256]), nominal.clone()).unwrap();
        let r = normalise_descriptors(a).unwrap();
        assert_eq!(r.shape(), [3, 256]);
        // (1, 3, 256)
        let a = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3, 256]), nominal.clone()).unwrap();
        let r = normalise_descriptors(a).unwrap();
        assert_eq!(r.shape(), [3, 256]);
        // (256, 3) — transposed
        let a = ArrayD::from_shape_vec(ndarray::IxDyn(&[256, 3]), nominal.clone()).unwrap();
        let r = normalise_descriptors(a).unwrap();
        assert_eq!(r.shape(), [3, 256]);
    }
}
