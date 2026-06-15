//! In-Rust learned global-descriptor (visual place recognition) ONNX runtime.
//!
//! Where [`superpoint_onnx`](super::superpoint_onnx) produces *local* features
//! per keypoint, this module produces a single *global* descriptor per image —
//! one L2-normalised vector that summarises the whole frame for appearance-based
//! retrieval (loop-closure candidate search, relocalisation). It wraps an
//! EigenPlaces / CosPlace model exported by `scripts/export_vpr_onnx.py`.
//!
//! The motivation is the recurring "learned front-end is the lever" result:
//! replacing hand-built corner descriptors with SuperPoint, and hand-built
//! matching with LightGlue, each closed a large gap. The loop-closure retrieval
//! front-end is the last hand-built stage — a k-means VLAD over local SuperPoint
//! descriptors (`place_recognition::vlad`). A learned VPR descriptor is the
//! natural next substitution.
//!
//! Two build profiles share this module, mirroring `superpoint_onnx`:
//!
//! * **Default (`onnx-inference` off)** — a skeleton whose every method returns
//!   [`GlobalDescriptorOnnxError::FeatureDisabled`], so downstream code compiles
//!   without the heavy `ort` dependency.
//! * **`onnx-inference` on** — wraps an `ort::session::Session` and runs the VPR
//!   model per frame.
//!
//! I/O contract (see `scripts/export_vpr_onnx.py`):
//!   input  `image` : (1, 3, H, W) f32, RGB in [0, 1] (the graph applies the
//!                    ImageNet mean/std internally)
//!   output `descriptor` : (1, D) f32, L2-normalised
//!
//! H and W are dynamic, so one model handles any resolution. KITTI and EuRoC
//! frames are single-channel; [`extract_global`](GlobalDescriptorOnnxExtractor::extract_global)
//! takes a [`GrayscaleImage`] and replicates the luma channel across RGB, which
//! is lossless for those grayscale cameras.

use super::GrayscaleImage;
use std::fmt;
use std::path::Path;
#[cfg(feature = "onnx-inference")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "onnx-inference")]
pub use super::superpoint_onnx::OnnxBackend;

/// In-Rust learned global-descriptor extractor. Implements per-frame VPR
/// inference when `onnx-inference` is enabled; otherwise every method returns
/// [`GlobalDescriptorOnnxError::FeatureDisabled`].
///
/// The loaded session is wrapped in `Arc<Mutex<_>>` so the extractor stays
/// cheaply cloneable (shared across a frame loop) and inference is serialised,
/// matching the SuperPoint / LightGlue extractors in this crate.
#[derive(Clone)]
pub struct GlobalDescriptorOnnxExtractor {
    #[cfg(feature = "onnx-inference")]
    session: Arc<Mutex<ort::session::Session>>,
    #[cfg(not(feature = "onnx-inference"))]
    _private: (),
}

impl fmt::Debug for GlobalDescriptorOnnxExtractor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GlobalDescriptorOnnxExtractor")
            .field("onnx_inference_feature", &cfg!(feature = "onnx-inference"))
            .finish()
    }
}

// ---------------------------------------------------------------
// Feature-disabled stub (`cargo build` default)
// ---------------------------------------------------------------

#[cfg(not(feature = "onnx-inference"))]
impl GlobalDescriptorOnnxExtractor {
    /// Load a VPR global-descriptor ONNX model from disk.
    ///
    /// **Stub** — without the `onnx-inference` feature this always returns
    /// [`GlobalDescriptorOnnxError::FeatureDisabled`].
    pub fn load_from_path<P: AsRef<Path>>(_path: P) -> Result<Self, GlobalDescriptorOnnxError> {
        Err(GlobalDescriptorOnnxError::FeatureDisabled {
            requested: "load_from_path",
        })
    }

    /// Compute the global descriptor for one frame.
    ///
    /// **Stub** — always returns [`GlobalDescriptorOnnxError::FeatureDisabled`].
    pub fn extract_global(
        &self,
        _image: &GrayscaleImage,
    ) -> Result<Vec<f32>, GlobalDescriptorOnnxError> {
        Err(GlobalDescriptorOnnxError::FeatureDisabled {
            requested: "extract_global",
        })
    }
}

// ---------------------------------------------------------------
// Active implementation (`onnx-inference` feature)
// ---------------------------------------------------------------

#[cfg(feature = "onnx-inference")]
impl GlobalDescriptorOnnxExtractor {
    /// Load a VPR global-descriptor ONNX model from disk, preferring the CUDA
    /// execution provider with a graceful CPU fallback (see [`OnnxBackend`]).
    pub fn load_from_path<P: AsRef<Path>>(path: P) -> Result<Self, GlobalDescriptorOnnxError> {
        Self::load_from_path_with_backend(path, OnnxBackend::default())
    }

    /// Like [`load_from_path`](Self::load_from_path) but with an explicit
    /// execution-provider choice.
    pub fn load_from_path_with_backend<P: AsRef<Path>>(
        path: P,
        backend: OnnxBackend,
    ) -> Result<Self, GlobalDescriptorOnnxError> {
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
            .map_err(GlobalDescriptorOnnxError::from_ort)?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)
            .map_err(GlobalDescriptorOnnxError::from_ort)?
            .with_execution_providers(providers)
            .map_err(GlobalDescriptorOnnxError::from_ort)?
            .commit_from_file(path.as_ref())
            .map_err(GlobalDescriptorOnnxError::from_ort)?;
        Ok(Self {
            session: Arc::new(Mutex::new(session)),
        })
    }

    /// Compute the L2-normalised global descriptor for one frame.
    ///
    /// The grayscale luma channel (already in `[0, 1]`) is replicated across the
    /// three RGB input channels — lossless for KITTI / EuRoC grayscale cameras.
    /// The returned vector is the model's output verbatim (already L2-normalised
    /// by the exported graph), so a dot product between two descriptors is their
    /// cosine similarity.
    pub fn extract_global(
        &self,
        image: &GrayscaleImage,
    ) -> Result<Vec<f32>, GlobalDescriptorOnnxError> {
        let width = image.width();
        let height = image.height();
        if width == 0 || height == 0 {
            return Err(GlobalDescriptorOnnxError::EmptyImage);
        }

        // Build (1, 3, H, W): the luma plane copied into R, G, B.
        let plane = image.pixels();
        let mut chw = Vec::with_capacity(3 * plane.len());
        for _ in 0..3 {
            chw.extend_from_slice(plane);
        }
        let input = ndarray::Array4::<f32>::from_shape_vec((1, 3, height, width), chw).map_err(
            |error| GlobalDescriptorOnnxError::PreprocessShape {
                width,
                height,
                message: error.to_string(),
            },
        )?;

        let input_value =
            ort::value::Tensor::from_array(input).map_err(GlobalDescriptorOnnxError::from_ort)?;
        let mut session = self
            .session
            .lock()
            .map_err(|error| GlobalDescriptorOnnxError::OnnxRuntime(error.to_string()))?;
        let mut outputs = session
            .run(ort::inputs![input_value])
            .map_err(GlobalDescriptorOnnxError::from_ort)?;

        let value = outputs.remove("descriptor").ok_or_else(|| {
            GlobalDescriptorOnnxError::OutputShapeMismatch {
                expected: "output named `descriptor`",
                actual: "missing".to_string(),
            }
        })?;
        let array: ndarray::ArrayD<f32> = value
            .try_extract_array::<f32>()
            .map_err(GlobalDescriptorOnnxError::from_ort)?
            .into_owned();

        // Accept (1, D) or (D,).
        let shape = array.shape().to_vec();
        let descriptor: Vec<f32> = match shape.as_slice() {
            [1, _d] => array.iter().copied().collect(),
            [_d] => array.iter().copied().collect(),
            _ => {
                return Err(GlobalDescriptorOnnxError::OutputShapeMismatch {
                    expected: "(1, D) or (D,) descriptor",
                    actual: format!("{shape:?}"),
                })
            }
        };
        Ok(descriptor)
    }
}

// ---------------------------------------------------------------
// Error type
// ---------------------------------------------------------------

/// Errors from the in-Rust VPR global-descriptor runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GlobalDescriptorOnnxError {
    /// The `onnx-inference` feature was not compiled in.
    FeatureDisabled { requested: &'static str },
    /// The input image had zero width or height.
    EmptyImage,
    /// The input tensor could not be shaped to `(1, 3, H, W)`.
    PreprocessShape {
        width: usize,
        height: usize,
        message: String,
    },
    /// The model output was missing or had an unexpected shape.
    OutputShapeMismatch {
        expected: &'static str,
        actual: String,
    },
    /// An ONNX Runtime error.
    OnnxRuntime(String),
}

#[cfg(feature = "onnx-inference")]
impl GlobalDescriptorOnnxError {
    fn from_ort<E: std::fmt::Display>(error: E) -> Self {
        GlobalDescriptorOnnxError::OnnxRuntime(error.to_string())
    }
}

impl fmt::Display for GlobalDescriptorOnnxError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GlobalDescriptorOnnxError::FeatureDisabled { requested } => write!(
                formatter,
                "global-descriptor ONNX runtime unavailable: `{requested}` requires the \
                 `onnx-inference` feature"
            ),
            GlobalDescriptorOnnxError::EmptyImage => {
                write!(formatter, "input image has zero width or height")
            }
            GlobalDescriptorOnnxError::PreprocessShape {
                width,
                height,
                message,
            } => write!(
                formatter,
                "failed to build (1, 3, {height}, {width}) input tensor: {message}"
            ),
            GlobalDescriptorOnnxError::OutputShapeMismatch { expected, actual } => {
                write!(
                    formatter,
                    "unexpected model output: expected {expected}, got {actual}"
                )
            }
            GlobalDescriptorOnnxError::OnnxRuntime(message) => {
                write!(formatter, "ONNX Runtime error: {message}")
            }
        }
    }
}

impl std::error::Error for GlobalDescriptorOnnxError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(feature = "onnx-inference"))]
    #[test]
    fn load_without_feature_reports_disabled() {
        let result = GlobalDescriptorOnnxExtractor::load_from_path("/nonexistent.onnx");
        assert_eq!(
            result.err(),
            Some(GlobalDescriptorOnnxError::FeatureDisabled {
                requested: "load_from_path"
            })
        );
    }

    #[test]
    fn error_display_is_descriptive() {
        let error = GlobalDescriptorOnnxError::OutputShapeMismatch {
            expected: "(1, D) or (D,) descriptor",
            actual: "[2, 3, 4]".to_string(),
        };
        let text = error.to_string();
        assert!(text.contains("unexpected model output"));
        assert!(text.contains("(1, D)"));
    }
}
