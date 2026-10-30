//! YOLOX inference with ONNX Runtime.
//!
//! With the `onnx-inference` feature, [`YoloxOnnxDetector`] wraps an
//! `ort` session on the CPU and runs an official YOLOX ONNX model (see
//! [`yolox`](crate::yolox) for the contract). Without it, every method returns
//! [`YoloxOnnxError::FeatureDisabled`], so callers compile without `ort`.
//!
//! The models are not bundled. The official ones are on the YOLOX releases
//! page (`yolox_nano.onnx` … `yolox_x.onnx`). `ort` loads the ONNX Runtime
//! library at run time, from `ORT_DYLIB_PATH` or the system library path,
//! as for the other ONNX models in this repository.

use std::fmt;
use std::path::Path;

use crate::bbox::Detection;
use crate::detector::ObjectDetector;
use crate::image::RgbImage;
use crate::yolox::{YoloxConfig, YoloxOutputError};

/// A YOLOX ONNX model behind the [`ObjectDetector`] trait.
pub struct YoloxOnnxDetector {
    config: YoloxConfig,
    #[cfg(feature = "onnx-inference")]
    session: ort::session::Session,
}

impl fmt::Debug for YoloxOnnxDetector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("YoloxOnnxDetector")
            .field("config", &self.config)
            .field("onnx_inference_feature", &cfg!(feature = "onnx-inference"))
            .finish()
    }
}

/// Errors of [`YoloxOnnxDetector`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum YoloxOnnxError {
    /// The `onnx-inference` feature was not compiled in.
    FeatureDisabled,
    /// The image has zero width or height.
    EmptyImage,
    /// The model output is missing or does not fit the configured input size.
    Output(String),
    /// An ONNX Runtime error, for example an input size the model rejects.
    OnnxRuntime(String),
}

impl fmt::Display for YoloxOnnxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FeatureDisabled => write!(
                f,
                "YOLOX ONNX inference needs the `onnx-inference` feature of visloc-objects"
            ),
            Self::EmptyImage => write!(f, "the image has zero width or height"),
            Self::Output(message) => write!(f, "unexpected YOLOX output: {message}"),
            Self::OnnxRuntime(message) => write!(f, "ONNX Runtime error: {message}"),
        }
    }
}

impl std::error::Error for YoloxOnnxError {}

#[cfg(feature = "onnx-inference")]
fn runtime_error<E: fmt::Display>(error: E) -> YoloxOnnxError {
    YoloxOnnxError::OnnxRuntime(error.to_string())
}

impl From<YoloxOutputError> for YoloxOnnxError {
    fn from(error: YoloxOutputError) -> Self {
        Self::Output(error.to_string())
    }
}

impl YoloxOnnxDetector {
    pub fn config(&self) -> &YoloxConfig {
        &self.config
    }

    /// Change the thresholds; the input size must stay the model's.
    pub fn config_mut(&mut self) -> &mut YoloxConfig {
        &mut self.config
    }
}

#[cfg(not(feature = "onnx-inference"))]
impl YoloxOnnxDetector {
    /// Load a YOLOX ONNX model. `config` must have the model's input size
    /// (416 × 416 for nano and tiny, 640 × 640 for the others).
    ///
    /// **Stub:** without the `onnx-inference` feature this always fails.
    pub fn load_from_path<P: AsRef<Path>>(
        _path: P,
        _config: YoloxConfig,
    ) -> Result<Self, YoloxOnnxError> {
        Err(YoloxOnnxError::FeatureDisabled)
    }
}

#[cfg(feature = "onnx-inference")]
impl YoloxOnnxDetector {
    /// Load a YOLOX ONNX model. `config` must have the model's input size
    /// (416 × 416 for nano and tiny, 640 × 640 for the others).
    pub fn load_from_path<P: AsRef<Path>>(
        path: P,
        config: YoloxConfig,
    ) -> Result<Self, YoloxOnnxError> {
        let session = ort::session::Session::builder()
            .map_err(runtime_error)?
            // `All` rather than `Level3`, which ONNX Runtime 1.22 rejects (see
            // visloc-vision's SuperPoint session).
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::All)
            .map_err(runtime_error)?
            .commit_from_file(path.as_ref())
            .map_err(runtime_error)?;
        Ok(Self { config, session })
    }

    fn run(&mut self, image: &RgbImage) -> Result<Vec<Detection>, YoloxOnnxError> {
        if image.width() == 0 || image.height() == 0 {
            return Err(YoloxOnnxError::EmptyImage);
        }
        let (width, height) = (self.config.input_width, self.config.input_height);
        let input = crate::yolox::letterbox(image, width, height);
        let tensor = ort::value::Tensor::from_array((
            [1usize, 3, height, width],
            input.tensor.into_boxed_slice(),
        ))
        .map_err(runtime_error)?;
        let outputs = self
            .session
            .run(ort::inputs![tensor])
            .map_err(runtime_error)?;
        if outputs.len() == 0 {
            return Err(YoloxOnnxError::Output(
                "the model has no outputs".to_string(),
            ));
        }
        let (_, values) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(runtime_error)?;
        Ok(crate::yolox::decode(values, &self.config, input.scale)?)
    }
}

impl ObjectDetector for YoloxOnnxDetector {
    type Error = YoloxOnnxError;

    /// Detections in `image` pixels, sorted by score.
    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>, Self::Error> {
        #[cfg(feature = "onnx-inference")]
        {
            self.run(image)
        }
        #[cfg(not(feature = "onnx-inference"))]
        {
            let _ = image;
            Err(YoloxOnnxError::FeatureDisabled)
        }
    }
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "onnx-inference"))]
    #[test]
    fn without_the_feature_loading_fails_cleanly() {
        let error = super::YoloxOnnxDetector::load_from_path("missing.onnx", Default::default())
            .unwrap_err();
        assert_eq!(error, super::YoloxOnnxError::FeatureDisabled);
        assert!(error.to_string().contains("onnx-inference"));
    }
}
