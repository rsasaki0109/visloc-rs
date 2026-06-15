pub mod deep;
pub mod global_descriptor_onnx;
pub mod lightglue_onnx;
pub mod superpoint_onnx;

pub use deep::{
    build_pyramid, CornerDeepAdapter, DeepFeatureExtractor, DeepFeatureSet, DeepFeatureSetError,
    HogLikeFeatureConfig, HogLikeFeatureError, HogLikeFeatureExtractor, MultiScaleDeepConfig,
    MultiScaleDeepExtractor, HOG_BINS, HOG_CELLS_PER_SIDE, HOG_CELL_SIZE, HOG_DESCRIPTOR_DIM,
};

use nalgebra::Point2;
use std::convert::Infallible;
use std::fmt;
use std::marker::PhantomData;

pub trait FeatureExtractor {
    type Image;
    type Error;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureSet {
    pub keypoints: Vec<Point2<f64>>,
    pub descriptors: Vec<Vec<f32>>,
}

impl FeatureSet {
    pub fn new(
        keypoints: Vec<Point2<f64>>,
        descriptors: Vec<Vec<f32>>,
    ) -> Result<Self, FeatureSetError> {
        let features = Self {
            keypoints,
            descriptors,
        };
        features.validate()?;
        Ok(features)
    }

    pub fn validate(&self) -> Result<(), FeatureSetError> {
        if self.keypoints.len() != self.descriptors.len() {
            return Err(FeatureSetError::ShapeMismatch {
                keypoint_count: self.keypoints.len(),
                descriptor_count: self.descriptors.len(),
            });
        }

        let Some(first_descriptor) = self.descriptors.first() else {
            return Ok(());
        };
        let descriptor_dimension = first_descriptor.len();
        for (index, descriptor) in self.descriptors.iter().enumerate() {
            if descriptor.len() != descriptor_dimension {
                return Err(FeatureSetError::DescriptorDimensionMismatch {
                    descriptor_index: index,
                    expected_dimension: descriptor_dimension,
                    actual_dimension: descriptor.len(),
                });
            }
        }

        Ok(())
    }

    pub fn len(&self) -> usize {
        self.keypoints.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keypoints.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureSetError {
    ShapeMismatch {
        keypoint_count: usize,
        descriptor_count: usize,
    },
    DescriptorDimensionMismatch {
        descriptor_index: usize,
        expected_dimension: usize,
        actual_dimension: usize,
    },
}

impl fmt::Display for FeatureSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                keypoint_count,
                descriptor_count,
            } => write!(
                formatter,
                "feature shape mismatch: {keypoint_count} keypoint(s), {descriptor_count} descriptor(s)"
            ),
            Self::DescriptorDimensionMismatch {
                descriptor_index,
                expected_dimension,
                actual_dimension,
            } => write!(
                formatter,
                "descriptor {descriptor_index} has dimension {actual_dimension}, expected {expected_dimension}"
            ),
        }
    }
}

impl std::error::Error for FeatureSetError {}

#[derive(Debug, Clone, PartialEq)]
pub struct GrayscaleImage {
    width: usize,
    height: usize,
    pixels: Vec<f32>,
}

impl GrayscaleImage {
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> Result<Self, GrayscaleImageError> {
        if width == 0 || height == 0 {
            return Err(GrayscaleImageError::EmptyImage);
        }
        let expected_len = width * height;
        if pixels.len() != expected_len {
            return Err(GrayscaleImageError::PixelCountMismatch {
                width,
                height,
                expected_len,
                actual_len: pixels.len(),
            });
        }
        Ok(Self {
            width,
            height,
            pixels,
        })
    }

    pub fn from_luma_u8(
        width: usize,
        height: usize,
        pixels: Vec<u8>,
    ) -> Result<Self, GrayscaleImageError> {
        let pixels = pixels
            .into_iter()
            .map(|value| value as f32 / 255.0)
            .collect();
        Self::new(width, height, pixels)
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn pixels(&self) -> &[f32] {
        &self.pixels
    }

    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x >= self.width || y >= self.height {
            return None;
        }
        Some(self.pixels[y * self.width + x])
    }

    fn pixel_unchecked(&self, x: usize, y: usize) -> f32 {
        self.pixels[y * self.width + x]
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GrayscaleImageError {
    EmptyImage,
    PixelCountMismatch {
        width: usize,
        height: usize,
        expected_len: usize,
        actual_len: usize,
    },
}

impl fmt::Display for GrayscaleImageError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyImage => write!(formatter, "grayscale image dimensions must be non-zero"),
            Self::PixelCountMismatch {
                width,
                height,
                expected_len,
                actual_len,
            } => write!(
                formatter,
                "grayscale image {width}x{height} expects {expected_len} pixel(s), got {actual_len}"
            ),
        }
    }
}

impl std::error::Error for GrayscaleImageError {}

#[derive(Debug, Clone, PartialEq)]
pub struct CornerFeatureConfig {
    pub max_features: usize,
    pub min_score: f32,
    pub descriptor_radius: usize,
}

impl Default for CornerFeatureConfig {
    fn default() -> Self {
        Self {
            max_features: 256,
            min_score: 0.05,
            descriptor_radius: 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CornerFeatureExtractor {
    pub config: CornerFeatureConfig,
}

impl CornerFeatureExtractor {
    pub fn new(config: CornerFeatureConfig) -> Self {
        Self { config }
    }

    fn required_margin(&self) -> usize {
        self.config.descriptor_radius.max(1)
    }

    fn corner_score(image: &GrayscaleImage, x: usize, y: usize) -> f32 {
        let dx = image.pixel_unchecked(x + 1, y) - image.pixel_unchecked(x - 1, y);
        let dy = image.pixel_unchecked(x, y + 1) - image.pixel_unchecked(x, y - 1);
        let dxy1 = image.pixel_unchecked(x + 1, y + 1) - image.pixel_unchecked(x - 1, y - 1);
        let dxy2 = image.pixel_unchecked(x + 1, y - 1) - image.pixel_unchecked(x - 1, y + 1);
        (dx * dx + dy * dy + 0.5 * (dxy1 * dxy1 + dxy2 * dxy2)).sqrt()
    }

    fn is_local_maximum(image: &GrayscaleImage, x: usize, y: usize, score: f32) -> bool {
        for neighbor_y in (y - 1)..=(y + 1) {
            for neighbor_x in (x - 1)..=(x + 1) {
                if neighbor_x == x && neighbor_y == y {
                    continue;
                }
                if neighbor_x == 0
                    || neighbor_y == 0
                    || neighbor_x + 1 >= image.width
                    || neighbor_y + 1 >= image.height
                {
                    continue;
                }
                if Self::corner_score(image, neighbor_x, neighbor_y) > score {
                    return false;
                }
            }
        }
        true
    }

    fn descriptor(&self, image: &GrayscaleImage, x: usize, y: usize) -> Vec<f32> {
        let radius = self.config.descriptor_radius;
        let center = image.pixel_unchecked(x, y);
        let side = radius * 2 + 1;
        let mut descriptor = Vec::with_capacity(side * side);
        for yy in (y - radius)..=(y + radius) {
            for xx in (x - radius)..=(x + radius) {
                descriptor.push(image.pixel_unchecked(xx, yy) - center);
            }
        }
        descriptor
    }

    /// Compute the patch descriptor centred at the integer pixel
    /// `(x, y)`, skipping corner detection. Returns `None` when the
    /// requested centre is too close to the image border for a full
    /// descriptor patch to fit.
    pub fn describe_at(&self, image: &GrayscaleImage, x: usize, y: usize) -> Option<Vec<f32>> {
        let radius = self.required_margin();
        if x < radius || y < radius || x + radius >= image.width() || y + radius >= image.height() {
            return None;
        }
        Some(self.descriptor(image, x, y))
    }
}

impl Default for CornerFeatureExtractor {
    fn default() -> Self {
        Self::new(CornerFeatureConfig::default())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CornerFeatureError {
    ImageTooSmall {
        width: usize,
        height: usize,
        required_margin: usize,
    },
    FeatureSet(FeatureSetError),
}

impl fmt::Display for CornerFeatureError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImageTooSmall {
                width,
                height,
                required_margin,
            } => write!(
                formatter,
                "image {width}x{height} is too small for feature extraction with margin {required_margin}"
            ),
            Self::FeatureSet(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for CornerFeatureError {}

impl From<FeatureSetError> for CornerFeatureError {
    fn from(error: FeatureSetError) -> Self {
        Self::FeatureSet(error)
    }
}

impl FeatureExtractor for CornerFeatureExtractor {
    type Image = GrayscaleImage;
    type Error = CornerFeatureError;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        let margin = self.required_margin();
        if image.width <= margin * 2 || image.height <= margin * 2 {
            return Err(CornerFeatureError::ImageTooSmall {
                width: image.width,
                height: image.height,
                required_margin: margin,
            });
        }

        let mut candidates = Vec::new();
        for y in margin..(image.height - margin) {
            for x in margin..(image.width - margin) {
                let score = Self::corner_score(image, x, y);
                if score < self.config.min_score {
                    continue;
                }
                if Self::is_local_maximum(image, x, y, score) {
                    candidates.push((score, x, y));
                }
            }
        }

        candidates.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.2.cmp(&right.2))
                .then_with(|| left.1.cmp(&right.1))
        });
        candidates.truncate(self.config.max_features);

        let mut keypoints = Vec::with_capacity(candidates.len());
        let mut descriptors = Vec::with_capacity(candidates.len());
        for (_score, x, y) in candidates {
            keypoints.push(Point2::new(x as f64, y as f64));
            descriptors.push(self.descriptor(image, x, y));
        }

        FeatureSet::new(keypoints, descriptors).map_err(CornerFeatureError::from)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProvidedFeatureExtractor {
    pub features: FeatureSet,
}

impl ProvidedFeatureExtractor {
    pub fn new(features: FeatureSet) -> Self {
        Self { features }
    }
}

impl FeatureExtractor for ProvidedFeatureExtractor {
    type Image = ();
    type Error = Infallible;

    fn extract(&self, _image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        Ok(self.features.clone())
    }
}

#[derive(Debug, Clone)]
pub struct FnFeatureExtractor<F, I, E> {
    extract_fn: F,
    _phantom: PhantomData<fn(&I) -> E>,
}

impl<F, I, E> FnFeatureExtractor<F, I, E> {
    pub fn new(extract_fn: F) -> Self {
        Self {
            extract_fn,
            _phantom: PhantomData,
        }
    }
}

impl<F, I, E> FeatureExtractor for FnFeatureExtractor<F, I, E>
where
    F: Fn(&I) -> Result<FeatureSet, E>,
{
    type Image = I;
    type Error = E;

    fn extract(&self, image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        (self.extract_fn)(image)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CornerFeatureConfig, CornerFeatureError, CornerFeatureExtractor, FeatureExtractor,
        FeatureSet, FeatureSetError, FnFeatureExtractor, GrayscaleImage, GrayscaleImageError,
        ProvidedFeatureExtractor,
    };
    use nalgebra::Point2;

    #[test]
    fn feature_set_new_validates_shapes() {
        let error = FeatureSet::new(vec![Point2::new(1.0, 2.0)], Vec::new()).unwrap_err();

        assert_eq!(
            error,
            FeatureSetError::ShapeMismatch {
                keypoint_count: 1,
                descriptor_count: 0,
            }
        );
    }

    #[test]
    fn feature_set_new_validates_descriptor_dimensions() {
        let error = FeatureSet::new(
            vec![Point2::new(1.0, 2.0), Point2::new(3.0, 4.0)],
            vec![vec![1.0, 2.0], vec![3.0]],
        )
        .unwrap_err();

        assert_eq!(
            error,
            FeatureSetError::DescriptorDimensionMismatch {
                descriptor_index: 1,
                expected_dimension: 2,
                actual_dimension: 1,
            }
        );
    }

    #[test]
    fn provided_feature_extractor_returns_static_features() {
        let features = FeatureSet::new(vec![Point2::new(1.0, 2.0)], vec![vec![0.5]]).unwrap();
        let extractor = ProvidedFeatureExtractor::new(features.clone());

        assert_eq!(extractor.extract(&()).unwrap(), features);
    }

    #[test]
    fn fn_feature_extractor_wraps_external_extractor_logic() {
        let extractor = FnFeatureExtractor::new(|image: &&str| {
            FeatureSet::new(
                vec![Point2::new(image.len() as f64, 0.0)],
                vec![vec![image.len() as f32]],
            )
        });

        let features = extractor.extract(&"abc").unwrap();

        assert_eq!(features.len(), 1);
        assert_eq!(features.keypoints[0], Point2::new(3.0, 0.0));
        assert_eq!(features.descriptors[0], vec![3.0]);
    }

    #[test]
    fn grayscale_image_validates_pixel_count() {
        let error = GrayscaleImage::new(2, 2, vec![0.0; 3]).unwrap_err();

        assert_eq!(
            error,
            GrayscaleImageError::PixelCountMismatch {
                width: 2,
                height: 2,
                expected_len: 4,
                actual_len: 3,
            }
        );
    }

    #[test]
    fn corner_feature_extractor_detects_synthetic_corner() {
        let mut pixels = vec![0_u8; 11 * 11];
        for y in 5..11 {
            for x in 5..11 {
                pixels[y * 11 + x] = 255;
            }
        }
        let image = GrayscaleImage::from_luma_u8(11, 11, pixels).unwrap();
        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
            max_features: 4,
            min_score: 0.5,
            descriptor_radius: 1,
        });

        let features = extractor.extract(&image).unwrap();

        assert!(!features.is_empty());
        assert!(features.len() <= 4);
        assert_eq!(features.descriptors[0].len(), 9);
        assert!(features
            .keypoints
            .iter()
            .any(|point| (point.x - 5.0).abs() <= 1.0 && (point.y - 5.0).abs() <= 1.0));
    }

    #[test]
    fn corner_feature_extractor_rejects_too_small_images() {
        let image = GrayscaleImage::new(3, 3, vec![0.0; 9]).unwrap();
        let extractor = CornerFeatureExtractor::new(CornerFeatureConfig {
            max_features: 8,
            min_score: 0.0,
            descriptor_radius: 2,
        });

        let error = extractor.extract(&image).unwrap_err();

        assert_eq!(
            error,
            CornerFeatureError::ImageTooSmall {
                width: 3,
                height: 3,
                required_margin: 2,
            }
        );
    }
}
