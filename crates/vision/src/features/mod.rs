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
        FeatureExtractor, FeatureSet, FeatureSetError, FnFeatureExtractor, ProvidedFeatureExtractor,
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
}
