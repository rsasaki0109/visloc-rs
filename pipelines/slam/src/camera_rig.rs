//! Per-image pinhole calibration support for the SfM frontends.
//!
//! The historical incremental/global APIs take one [`Camera`].  A camera rig
//! lets callers retain one calibration per image while using those APIs: the
//! feature pixels are converted to the rig's first-camera pixel convention,
//! so the existing normalized-ray, PnP, triangulation, and BA code sees the
//! exact same rays.  The conversion is lossless for pinhole cameras and does
//! not touch descriptor rows or feature indices.  The legacy single-camera
//! APIs remain unchanged.

use nalgebra::{Point2, Point3};
use thiserror::Error;
use visloc_core::geometry::Pose;
use visloc_core::types::{Camera, CameraModel};
use visloc_vision::features::FeatureSet;

use crate::global_sfm::{
    reconstruct_global_sfm, GlobalReconstructionError, GlobalReconstructionTuning,
};
use crate::incremental_sfm::{
    incremental_sfm, IncrementalSfmConfig, IncrementalSfmError, IncrementalSfmResult,
    PairwiseMatches, SfmTrack,
};

/// Validated, index-aligned per-image pinhole calibrations.
#[derive(Debug, Clone, PartialEq)]
pub struct PerImageCameras {
    cameras: Vec<Camera>,
    reference_index: usize,
}

/// Errors returned while constructing or applying a per-image calibration
/// set.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum PerImageCameraError {
    #[error("per-image calibration contains no cameras")]
    Empty,
    #[error("camera {index} uses unsupported model {model}; only PINHOLE is supported")]
    UnsupportedModel { index: usize, model: String },
    #[error("camera {index} has {actual} parameters; PINHOLE requires exactly 4 [fx, fy, cx, cy]")]
    ParameterCount { index: usize, actual: usize },
    #[error("camera {index} has invalid parameter {name}={value}")]
    InvalidParameter {
        index: usize,
        name: &'static str,
        value: f64,
    },
    #[error("camera index {index} is outside 0..{len}")]
    ImageIndex { index: usize, len: usize },
    #[error("per-image camera count {cameras} does not match feature count {features}")]
    FeatureCount { cameras: usize, features: usize },
    #[error("image dimension count {actual} does not match camera count {expected}")]
    DimensionCount { actual: usize, expected: usize },
    #[error(
        "image {index} dimensions {actual_width}x{actual_height} do not match calibration {expected_width}x{expected_height}"
    )]
    DimensionMismatch {
        index: usize,
        actual_width: u32,
        actual_height: u32,
        expected_width: u32,
        expected_height: u32,
    },
    #[error(
        "image {image} keypoint {keypoint} is non-finite or outside {width}x{height}: ({x}, {y})"
    )]
    InvalidKeypoint {
        image: usize,
        keypoint: usize,
        x: f64,
        y: f64,
        width: u32,
        height: u32,
    },
    #[error("feature set {image} is invalid: {message}")]
    InvalidFeatures { image: usize, message: String },
}

/// Error from the per-image-camera incremental convenience wrapper.
#[derive(Debug, Error)]
pub enum PerImageCameraIncrementalError {
    #[error("invalid per-image calibration: {0}")]
    Camera(#[from] PerImageCameraError),
    #[error("incremental SfM failed: {0}")]
    Incremental(#[from] IncrementalSfmError),
}

/// Error from the per-image-camera global convenience wrapper.
#[derive(Debug, Error)]
pub enum PerImageCameraGlobalError {
    #[error("invalid per-image calibration: {0}")]
    Camera(#[from] PerImageCameraError),
    #[error("global SfM failed: {0}")]
    Global(#[from] GlobalReconstructionError),
}

impl PerImageCameras {
    /// Validate and construct a camera set.  The first camera is the internal
    /// reference convention; all cameras remain available through
    /// [`Self::camera`].
    pub fn new(cameras: Vec<Camera>) -> Result<Self, PerImageCameraError> {
        if cameras.is_empty() {
            return Err(PerImageCameraError::Empty);
        }
        for (index, camera) in cameras.iter().enumerate() {
            if camera.model != CameraModel::Pinhole {
                return Err(PerImageCameraError::UnsupportedModel {
                    index,
                    model: format!("{:?}", camera.model),
                });
            }
            if camera.params.len() != 4 {
                return Err(PerImageCameraError::ParameterCount {
                    index,
                    actual: camera.params.len(),
                });
            }
            if camera.width == 0 {
                return Err(PerImageCameraError::InvalidParameter {
                    index,
                    name: "width",
                    value: camera.width as f64,
                });
            }
            if camera.height == 0 {
                return Err(PerImageCameraError::InvalidParameter {
                    index,
                    name: "height",
                    value: camera.height as f64,
                });
            }
            let names = ["fx", "fy", "cx", "cy"];
            for (name, value) in names.into_iter().zip(camera.params.iter().copied()) {
                if !value.is_finite() || ((name == "fx" || name == "fy") && value <= 0.0) {
                    return Err(PerImageCameraError::InvalidParameter { index, name, value });
                }
            }
        }
        Ok(Self {
            cameras,
            reference_index: 0,
        })
    }

    /// Construct from a borrowed slice without changing camera ids or order.
    pub fn from_slice(cameras: &[Camera]) -> Result<Self, PerImageCameraError> {
        Self::new(cameras.to_vec())
    }

    pub fn len(&self) -> usize {
        self.cameras.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cameras.is_empty()
    }

    pub fn cameras(&self) -> &[Camera] {
        &self.cameras
    }

    pub fn camera(&self, image_index: usize) -> Result<&Camera, PerImageCameraError> {
        self.cameras
            .get(image_index)
            .ok_or(PerImageCameraError::ImageIndex {
                index: image_index,
                len: self.cameras.len(),
            })
    }

    /// Camera used as the internal pixel convention by the compatibility
    /// wrappers.  It is also a valid output camera for the shared legacy API.
    pub fn reference_camera(&self) -> &Camera {
        &self.cameras[self.reference_index]
    }

    /// Whether every image has exactly the same geometry (camera ids may still
    /// differ).  This lets wrappers preserve the byte-identical legacy path.
    pub fn has_shared_geometry(&self) -> bool {
        let reference = self.reference_camera();
        self.cameras.iter().all(|camera| {
            camera.model == reference.model
                && camera.width == reference.width
                && camera.height == reference.height
                && camera.params == reference.params
        })
    }

    /// Validate decoded image dimensions against the corresponding COLMAP
    /// camera blocks before any pixel conversion is attempted.
    pub fn validate_image_dimensions(
        &self,
        dimensions: &[(u32, u32)],
    ) -> Result<(), PerImageCameraError> {
        if dimensions.len() != self.cameras.len() {
            return Err(PerImageCameraError::DimensionCount {
                actual: dimensions.len(),
                expected: self.cameras.len(),
            });
        }
        for (index, (&(actual_width, actual_height), camera)) in
            dimensions.iter().zip(&self.cameras).enumerate()
        {
            if actual_width != camera.width || actual_height != camera.height {
                return Err(PerImageCameraError::DimensionMismatch {
                    index,
                    actual_width,
                    actual_height,
                    expected_width: camera.width,
                    expected_height: camera.height,
                });
            }
        }
        Ok(())
    }

    /// Validate feature shape and pixel bounds against this set.  This is
    /// useful for feature-file input where image decoding is intentionally
    /// avoided; callers with decoded images should also call
    /// [`Self::validate_image_dimensions`].
    pub fn validate_features(&self, features: &[FeatureSet]) -> Result<(), PerImageCameraError> {
        if features.len() != self.cameras.len() {
            return Err(PerImageCameraError::FeatureCount {
                cameras: self.cameras.len(),
                features: features.len(),
            });
        }
        for (image, (feature_set, camera)) in features.iter().zip(&self.cameras).enumerate() {
            feature_set
                .validate()
                .map_err(|error| PerImageCameraError::InvalidFeatures {
                    image,
                    message: error.to_string(),
                })?;
            for (keypoint, point) in feature_set.keypoints.iter().enumerate() {
                if !point.x.is_finite()
                    || !point.y.is_finite()
                    || point.x < 0.0
                    || point.y < 0.0
                    || point.x >= camera.width as f64
                    || point.y >= camera.height as f64
                {
                    return Err(PerImageCameraError::InvalidKeypoint {
                        image,
                        keypoint,
                        x: point.x,
                        y: point.y,
                        width: camera.width,
                        height: camera.height,
                    });
                }
            }
        }
        Ok(())
    }

    /// Convert one image's pixel to the reference camera's pixel convention.
    /// For PINHOLE cameras this is simply normalize-then-project and therefore
    /// preserves the underlying bearing exactly (up to floating point roundoff).
    pub fn to_reference_pixel(
        &self,
        image_index: usize,
        pixel: &Point2<f64>,
    ) -> Result<Point2<f64>, PerImageCameraError> {
        let camera = self.camera(image_index)?;
        if self.has_shared_geometry() {
            return Ok(*pixel);
        }
        let normalized =
            camera
                .normalize_pixel(pixel)
                .ok_or_else(|| PerImageCameraError::InvalidKeypoint {
                    image: image_index,
                    keypoint: 0,
                    x: pixel.x,
                    y: pixel.y,
                    width: camera.width,
                    height: camera.height,
                })?;
        self.reference_camera()
            .project(&Point3::new(normalized.x, normalized.y, 1.0))
            .ok_or_else(|| PerImageCameraError::InvalidKeypoint {
                image: image_index,
                keypoint: 0,
                x: pixel.x,
                y: pixel.y,
                width: camera.width,
                height: camera.height,
            })
    }

    /// Convert a reference-convention pixel back to an image's native
    /// calibration.  This is primarily for exporters and diagnostics.
    pub fn from_reference_pixel(
        &self,
        image_index: usize,
        pixel: &Point2<f64>,
    ) -> Result<Point2<f64>, PerImageCameraError> {
        let camera = self.camera(image_index)?;
        if self.has_shared_geometry() {
            return Ok(*pixel);
        }
        let normalized = self
            .reference_camera()
            .normalize_pixel(pixel)
            .ok_or_else(|| PerImageCameraError::InvalidKeypoint {
                image: image_index,
                keypoint: 0,
                x: pixel.x,
                y: pixel.y,
                width: camera.width,
                height: camera.height,
            })?;
        camera
            .project(&Point3::new(normalized.x, normalized.y, 1.0))
            .ok_or_else(|| PerImageCameraError::InvalidKeypoint {
                image: image_index,
                keypoint: 0,
                x: pixel.x,
                y: pixel.y,
                width: camera.width,
                height: camera.height,
            })
    }

    /// Clone feature sets while changing only keypoint pixels.  Descriptors,
    /// row order, and indices are untouched.
    pub fn canonicalize_features(
        &self,
        features: &[FeatureSet],
    ) -> Result<Vec<FeatureSet>, PerImageCameraError> {
        // Preserve the borrowed-input API and its atomic-on-error behavior,
        // while letting owned callers canonicalize without copying the
        // descriptor bank.  Only the keypoint vectors are temporary here.
        let mut canonical = features.to_vec();
        self.canonicalize_features_in_place(&mut canonical)?;
        Ok(canonical)
    }

    /// Canonicalize feature pixels in place without cloning descriptors.
    ///
    /// The per-image calibration changes only pixel coordinates; descriptor
    /// rows and their indices remain byte-for-byte untouched.  All converted
    /// keypoints are computed before the input is modified, so an invalid
    /// conversion leaves the caller's feature sets unchanged, matching the
    /// error behavior of [`Self::canonicalize_features`].
    pub fn canonicalize_features_in_place(
        &self,
        features: &mut [FeatureSet],
    ) -> Result<(), PerImageCameraError> {
        self.validate_features(features)?;
        if self.has_shared_geometry() {
            return Ok(());
        }
        let canonical_keypoints = features
            .iter()
            .enumerate()
            .map(|(image, feature_set)| {
                feature_set
                    .keypoints
                    .iter()
                    .map(|pixel| self.to_reference_pixel(image, pixel))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (feature_set, keypoints) in features.iter_mut().zip(canonical_keypoints) {
            feature_set.keypoints = keypoints;
        }
        Ok(())
    }
}

/// Run the existing incremental mapper using one validated camera per image.
///
/// Pairwise match indices remain valid because only feature pixels are
/// canonicalized; descriptor rows and their order are untouched.  For a
/// shared-geometry set this delegates with the original feature slice, so the
/// historical path is an exact no-op.
pub fn incremental_sfm_with_per_image_cameras(
    cameras: &PerImageCameras,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> Result<IncrementalSfmResult, PerImageCameraIncrementalError> {
    let canonical = cameras.canonicalize_features(features)?;
    Ok(incremental_sfm(
        cameras.reference_camera(),
        &canonical,
        pairwise,
        config,
    )?)
}

/// Run the existing global mapper using one validated camera per image.  The
/// returned poses/tracks are in the reference-camera ray convention; callers
/// can use [`PerImageCameras::from_reference_pixel`] for native-pixel output.
#[allow(clippy::type_complexity)]
pub fn reconstruct_global_sfm_with_per_image_cameras(
    cameras: &PerImageCameras,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tuning: &GlobalReconstructionTuning,
    mapper: &IncrementalSfmConfig,
) -> Result<(Vec<Option<Pose>>, Vec<SfmTrack>, f64), PerImageCameraGlobalError> {
    let canonical = cameras.canonicalize_features(features)?;
    Ok(reconstruct_global_sfm(
        cameras.reference_camera(),
        &canonical,
        pairwise,
        tuning,
        mapper,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;
    use visloc_vision::pnp::Correspondence2D3D;
    use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
    use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;

    fn camera(id: u64, width: u32, height: u32, fx: f64, fy: f64) -> Camera {
        Camera::pinhole(
            id,
            width,
            height,
            fx,
            fy,
            width as f64 / 2.0,
            height as f64 / 2.0,
        )
    }

    fn features(points: Vec<Point2<f64>>) -> FeatureSet {
        FeatureSet::new(points, vec![vec![1.0, 2.0]; 1]).unwrap()
    }

    #[test]
    fn shared_geometry_is_an_exact_feature_noop() {
        let cameras = PerImageCameras::new(vec![camera(4, 100, 80, 50.0, 50.0); 2]).unwrap();
        let input = vec![
            features(vec![Point2::new(10.0, 20.0)]),
            features(vec![Point2::new(30.0, 40.0)]),
        ];
        assert_eq!(cameras.canonicalize_features(&input).unwrap(), input);
    }

    #[test]
    fn different_focal_lengths_preserve_normalized_bearing() {
        let first = camera(1, 100, 80, 50.0, 50.0);
        let second = camera(2, 200, 160, 100.0, 80.0);
        let cameras = PerImageCameras::new(vec![first.clone(), second.clone()]).unwrap();
        let input = vec![
            features(vec![Point2::new(50.0, 40.0)]),
            features(vec![Point2::new(100.0, 80.0)]),
        ];
        let converted = cameras.canonicalize_features(&input).unwrap();
        let normalized_before = second.normalize_pixel(&input[1].keypoints[0]).unwrap();
        let normalized_after = first.normalize_pixel(&converted[1].keypoints[0]).unwrap();
        assert!((normalized_before.x - normalized_after.x).abs() < 1e-12);
        assert!((normalized_before.y - normalized_after.y).abs() < 1e-12);
    }

    #[test]
    fn in_place_canonicalization_keeps_descriptor_storage_and_matches_owned() {
        let first = camera(1, 100, 80, 50.0, 50.0);
        let second = camera(2, 200, 160, 100.0, 80.0);
        let cameras = PerImageCameras::new(vec![first, second]).unwrap();
        let input = vec![
            features(vec![Point2::new(50.0, 40.0)]),
            features(vec![Point2::new(100.0, 80.0)]),
        ];
        let mut in_place = input.clone();
        let descriptor_storage: Vec<*const Vec<f32>> = in_place
            .iter()
            .map(|feature_set| feature_set.descriptors.as_ptr())
            .collect();
        cameras
            .canonicalize_features_in_place(&mut in_place)
            .unwrap();
        let owned = cameras.canonicalize_features(&input).unwrap();

        assert_eq!(in_place, owned);
        for (feature_set, storage) in in_place.iter().zip(descriptor_storage) {
            assert_eq!(feature_set.descriptors.as_ptr(), storage);
        }
        assert_eq!(in_place[0].descriptors, input[0].descriptors);
        assert_eq!(in_place[1].descriptors, input[1].descriptors);
    }

    #[test]
    fn dimensions_and_parameters_are_checked() {
        let invalid_camera = Camera {
            id: 1,
            model: CameraModel::Pinhole,
            width: 100,
            height: 80,
            params: vec![50.0, f64::NAN, 50.0, 40.0],
        };
        assert!(matches!(
            PerImageCameras::new(vec![invalid_camera]),
            Err(PerImageCameraError::InvalidParameter { name: "fy", .. })
        ));
        let cameras = PerImageCameras::new(vec![camera(1, 100, 80, 50.0, 50.0)]).unwrap();
        assert!(matches!(
            cameras.validate_image_dimensions(&[(99, 80)]),
            Err(PerImageCameraError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn distinct_cameras_preserve_triangulation_and_pnp_rays() {
        let left = camera(7, 640, 480, 500.0, 500.0);
        let right = camera(9, 800, 600, 700.0, 680.0);
        let rig = PerImageCameras::new(vec![left.clone(), right.clone()]).unwrap();
        let left_to_right = SE3::new(UnitQuaternion::identity(), Vector3::new(-0.25, 0.0, 0.0));
        let points = [
            Point3::new(-0.8, -0.6, 4.0),
            Point3::new(0.7, -0.5, 4.5),
            Point3::new(-0.6, 0.7, 5.0),
            Point3::new(0.8, 0.6, 5.5),
            Point3::new(-0.2, 0.1, 6.0),
            Point3::new(0.4, -0.2, 6.5),
            Point3::new(-0.5, 0.3, 7.0),
            Point3::new(0.2, 0.5, 7.5),
        ];
        let mut pnp = Vec::with_capacity(points.len());
        for point in points {
            let left_pixel = left.project(&point).unwrap();
            let right_pixel = right
                .project(&left_to_right.transform_point(&point))
                .unwrap();
            let triangulated = triangulate_two_view_left_frame(
                &left,
                &right,
                &left_to_right,
                &left_pixel,
                &right_pixel,
            )
            .unwrap();
            assert!((triangulated - point).norm() < 1.0e-8);
            pnp.push(Correspondence2D3D {
                point2d: rig.to_reference_pixel(1, &right_pixel).unwrap(),
                point3d: point,
                confidence: None,
            });
        }
        let report = PnPRansac::default()
            .estimate(&pnp, rig.reference_camera())
            .expect("canonicalized rays must support PnP");
        assert!(report.inliers.len() >= 6);
        assert!(
            report
                .pose
                .world_to_camera
                .rotation
                .rotation_to(&left_to_right.rotation)
                .angle()
                < 1.0e-6
        );
        assert!(
            (report.pose.world_to_camera.translation - left_to_right.translation).norm() < 1.0e-6,
            "PnP translation drifted: {:?}",
            report.pose.world_to_camera.translation
        );
    }
}
