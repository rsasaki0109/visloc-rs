//! In-Rust LightGlue matcher via ONNX Runtime.
//!
//! The sibling of [`super::superpoint_onnx`]: where that module runs the
//! SuperPoint feature *extractor* in-process, this runs the learned
//! **LightGlue matcher** in-process, so a full deep front-end (extract +
//! match) needs no Python feature-export pass.
//!
//! Two build profiles share this module, exactly like `superpoint_onnx`:
//!
//! * **Default (`onnx-inference` off)** — a skeleton whose every method
//!   returns [`LightGlueOnnxError::FeatureDisabled`].
//! * **`onnx-inference` enabled** — wraps an `ort::session::Session`.
//!
//! Expected ONNX I/O contract (batch size 1), produced by
//! `scripts/export_lightglue_onnx.py`:
//!
//! * inputs  `kpts0 (1, M, 2) f32`, `desc0 (1, M, 256) f32`,
//!   `kpts1 (1, N, 2) f32`, `desc1 (1, N, 256) f32`
//! * outputs `matches0 (M,) i64` (matched index in image 1, or `-1`),
//!   `mscores0 (M,) f32` (confidence in `(0, 1]`)
//!
//! Keypoints are pixel coordinates; the exported graph normalises them by the
//! baked-in image size. The export disables LightGlue's adaptive
//! depth/width pruning and FlashAttention (a speed knob, not a quality one);
//! the ONNX matches are bit-identical to the Python reference on real EuRoC
//! pairs.

#[cfg(feature = "onnx-inference")]
use super::superpoint_onnx::OnnxBackend;
use nalgebra::Point2;
use std::fmt;
use std::path::Path;
#[cfg(feature = "onnx-inference")]
use std::sync::{Arc, Mutex};

/// Expected descriptor dimensionality (SuperPoint / LightGlue).
pub const DESCRIPTOR_DIM: usize = 256;

/// A single LightGlue match: keypoint `query_index` in image 0 matched to
/// keypoint `train_index` in image 1 with confidence `score` in `(0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LightGlueMatch {
    pub query_index: usize,
    pub train_index: usize,
    pub score: f32,
}

/// Errors from the in-Rust LightGlue ONNX matcher.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LightGlueOnnxError {
    /// The `onnx-inference` feature is not enabled.
    FeatureDisabled { requested: &'static str },
    /// A descriptor did not have the expected [`DESCRIPTOR_DIM`] length, or
    /// the keypoint / descriptor counts disagreed for one image.
    DimensionMismatch { message: String },
    /// A model output had an unexpected shape.
    OutputShapeMismatch {
        expected: &'static str,
        actual: String,
    },
    /// Underlying ONNX Runtime error (stored as `String` to stay cheaply
    /// comparable).
    OnnxRuntime(String),
}

impl fmt::Display for LightGlueOnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LightGlueOnnxError::FeatureDisabled { requested } => write!(
                f,
                "LightGlue ONNX `{requested}` requires the `onnx-inference` feature"
            ),
            LightGlueOnnxError::DimensionMismatch { message } => {
                write!(f, "LightGlue ONNX input dimension mismatch: {message}")
            }
            LightGlueOnnxError::OutputShapeMismatch { expected, actual } => {
                write!(f, "LightGlue ONNX output shape: expected {expected}, got {actual}")
            }
            LightGlueOnnxError::OnnxRuntime(message) => {
                write!(f, "LightGlue ONNX Runtime error: {message}")
            }
        }
    }
}

impl std::error::Error for LightGlueOnnxError {}

#[cfg(feature = "onnx-inference")]
impl LightGlueOnnxError {
    fn from_ort<E: std::fmt::Display>(error: E) -> Self {
        Self::OnnxRuntime(error.to_string())
    }
}

/// In-Rust LightGlue matcher. Wraps a shared `ort` session so `Clone` is cheap
/// (mirrors [`super::superpoint_onnx::SuperPointOnnxExtractor`]).
#[derive(Clone)]
pub struct LightGlueOnnxMatcher {
    #[cfg(feature = "onnx-inference")]
    session: Arc<Mutex<ort::session::Session>>,
}

impl fmt::Debug for LightGlueOnnxMatcher {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("LightGlueOnnxMatcher")
            .field("onnx_inference_feature", &cfg!(feature = "onnx-inference"))
            .finish()
    }
}

// ---------------------------------------------------------------
// Feature-disabled stub
// ---------------------------------------------------------------

#[cfg(not(feature = "onnx-inference"))]
impl LightGlueOnnxMatcher {
    /// **Stub** — without `onnx-inference` this returns
    /// [`LightGlueOnnxError::FeatureDisabled`].
    pub fn load_from_path<P: AsRef<Path>>(_path: P) -> Result<Self, LightGlueOnnxError> {
        Err(LightGlueOnnxError::FeatureDisabled {
            requested: "load_from_path",
        })
    }

    /// **Stub** — see [`load_from_path`](Self::load_from_path).
    pub fn match_features(
        &self,
        _keypoints0: &[Point2<f64>],
        _descriptors0: &[Vec<f32>],
        _keypoints1: &[Point2<f64>],
        _descriptors1: &[Vec<f32>],
    ) -> Result<Vec<LightGlueMatch>, LightGlueOnnxError> {
        Err(LightGlueOnnxError::FeatureDisabled {
            requested: "match_features",
        })
    }
}

// ---------------------------------------------------------------
// Active implementation (`onnx-inference`)
// ---------------------------------------------------------------

#[cfg(feature = "onnx-inference")]
impl LightGlueOnnxMatcher {
    /// Load a LightGlue ONNX model, preferring CUDA with a CPU fallback.
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, LightGlueOnnxError> {
        Self::load_from_path_with_backend(path, OnnxBackend::default())
    }

    /// Like [`load_from_path`](Self::load_from_path) with an explicit
    /// execution-provider choice (shares [`OnnxBackend`] with SuperPoint).
    pub fn load_from_path_with_backend<P: AsRef<Path>>(
        path: P,
        backend: OnnxBackend,
    ) -> Result<Self, LightGlueOnnxError> {
        use ort::execution_providers::{CPUExecutionProvider, CUDAExecutionProvider};

        let providers = match backend {
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
            .map_err(LightGlueOnnxError::from_ort)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(LightGlueOnnxError::from_ort)?
            .with_execution_providers(providers)
            .map_err(LightGlueOnnxError::from_ort)?
            .commit_from_file(path.as_ref())
            .map_err(LightGlueOnnxError::from_ort)?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
        })
    }

    /// Match keypoints/descriptors between two images. Returns one
    /// [`LightGlueMatch`] per matched keypoint in image 0 (unmatched
    /// keypoints are omitted).
    pub fn match_features(
        &self,
        keypoints0: &[Point2<f64>],
        descriptors0: &[Vec<f32>],
        keypoints1: &[Point2<f64>],
        descriptors1: &[Vec<f32>],
    ) -> Result<Vec<LightGlueMatch>, LightGlueOnnxError> {
        if keypoints0.is_empty() || keypoints1.is_empty() {
            return Ok(Vec::new());
        }
        let kpts0 = build_keypoint_tensor(keypoints0)?;
        let desc0 = build_descriptor_tensor(descriptors0, keypoints0.len(), "image 0")?;
        let kpts1 = build_keypoint_tensor(keypoints1)?;
        let desc1 = build_descriptor_tensor(descriptors1, keypoints1.len(), "image 1")?;

        let kpts0 = ort::value::Tensor::from_array(kpts0).map_err(LightGlueOnnxError::from_ort)?;
        let desc0 = ort::value::Tensor::from_array(desc0).map_err(LightGlueOnnxError::from_ort)?;
        let kpts1 = ort::value::Tensor::from_array(kpts1).map_err(LightGlueOnnxError::from_ort)?;
        let desc1 = ort::value::Tensor::from_array(desc1).map_err(LightGlueOnnxError::from_ort)?;

        let mut session = self
            .session
            .lock()
            .map_err(|error| LightGlueOnnxError::OnnxRuntime(error.to_string()))?;
        // Positional binding follows the export's input order:
        // kpts0, desc0, kpts1, desc1.
        let mut outputs = session
            .run(ort::inputs![kpts0, desc0, kpts1, desc1])
            .map_err(LightGlueOnnxError::from_ort)?;

        let matches0 = outputs
            .remove("matches0")
            .ok_or_else(|| LightGlueOnnxError::OutputShapeMismatch {
                expected: "output named `matches0`",
                actual: "missing".to_string(),
            })?;
        let mscores0 = outputs
            .remove("mscores0")
            .ok_or_else(|| LightGlueOnnxError::OutputShapeMismatch {
                expected: "output named `mscores0`",
                actual: "missing".to_string(),
            })?;

        let matches0 = matches0
            .try_extract_array::<i64>()
            .map_err(LightGlueOnnxError::from_ort)?;
        let mscores0 = mscores0
            .try_extract_array::<f32>()
            .map_err(LightGlueOnnxError::from_ort)?;
        let matches0 = squeeze_to_1d_i64(matches0.view())?;
        let mscores0 = squeeze_to_1d_f32(mscores0.view())?;

        let mut matches = Vec::new();
        for (query_index, (&target, &score)) in
            matches0.iter().zip(mscores0.iter()).enumerate()
        {
            if target >= 0 {
                matches.push(LightGlueMatch {
                    query_index,
                    train_index: target as usize,
                    score,
                });
            }
        }
        Ok(matches)
    }
}

#[cfg(feature = "onnx-inference")]
fn build_keypoint_tensor(
    keypoints: &[Point2<f64>],
) -> Result<ndarray::Array3<f32>, LightGlueOnnxError> {
    let m = keypoints.len();
    let mut array = ndarray::Array3::<f32>::zeros((1, m, 2));
    for (i, p) in keypoints.iter().enumerate() {
        array[(0, i, 0)] = p.x as f32;
        array[(0, i, 1)] = p.y as f32;
    }
    Ok(array)
}

#[cfg(feature = "onnx-inference")]
fn build_descriptor_tensor(
    descriptors: &[Vec<f32>],
    expected_count: usize,
    which: &str,
) -> Result<ndarray::Array3<f32>, LightGlueOnnxError> {
    if descriptors.len() != expected_count {
        return Err(LightGlueOnnxError::DimensionMismatch {
            message: format!(
                "{which}: {} descriptors for {expected_count} keypoints",
                descriptors.len()
            ),
        });
    }
    let mut array = ndarray::Array3::<f32>::zeros((1, expected_count, DESCRIPTOR_DIM));
    for (i, desc) in descriptors.iter().enumerate() {
        if desc.len() != DESCRIPTOR_DIM {
            return Err(LightGlueOnnxError::DimensionMismatch {
                message: format!(
                    "{which}: descriptor {i} has length {} (expected {DESCRIPTOR_DIM})",
                    desc.len()
                ),
            });
        }
        for (j, &value) in desc.iter().enumerate() {
            array[(0, i, j)] = value;
        }
    }
    Ok(array)
}

#[cfg(feature = "onnx-inference")]
fn squeeze_to_1d_i64(
    array: ndarray::ArrayView<i64, ndarray::IxDyn>,
) -> Result<Vec<i64>, LightGlueOnnxError> {
    let shape = array.shape().to_vec();
    match shape.as_slice() {
        [_m] => Ok(array.iter().copied().collect()),
        [1, _m] => Ok(array.iter().copied().collect()),
        _ => Err(LightGlueOnnxError::OutputShapeMismatch {
            expected: "(M,) or (1, M) matches0",
            actual: format!("{shape:?}"),
        }),
    }
}

#[cfg(feature = "onnx-inference")]
fn squeeze_to_1d_f32(
    array: ndarray::ArrayView<f32, ndarray::IxDyn>,
) -> Result<Vec<f32>, LightGlueOnnxError> {
    let shape = array.shape().to_vec();
    match shape.as_slice() {
        [_m] => Ok(array.iter().copied().collect()),
        [1, _m] => Ok(array.iter().copied().collect()),
        _ => Err(LightGlueOnnxError::OutputShapeMismatch {
            expected: "(M,) or (1, M) mscores0",
            actual: format!("{shape:?}"),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn match_struct_is_copy_and_comparable() {
        let a = LightGlueMatch {
            query_index: 1,
            train_index: 2,
            score: 0.9,
        };
        let b = a;
        assert_eq!(a, b);
    }

    #[cfg(not(feature = "onnx-inference"))]
    #[test]
    fn stub_reports_feature_disabled() {
        let err = LightGlueOnnxMatcher::load_from_path("nope.onnx").unwrap_err();
        assert_eq!(
            err,
            LightGlueOnnxError::FeatureDisabled {
                requested: "load_from_path"
            }
        );
    }
}
