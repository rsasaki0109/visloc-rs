#![allow(clippy::useless_vec)]

use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{
    Camera, Frame, Landmark, LandmarkDescriptorStore, LocalizationFailureReason,
    PoseEstimationFailureReason, PoseEstimatorDiagnostics, QueryImage, VisualMap,
};
use visloc_rs::vision::features::{FeatureExtractor, FeatureSet};
use visloc_rs::vision::matching::{BruteForceMatcher, DescriptorMatch, Matcher};
use visloc_rs::vision::pnp::Correspondence2D3D;
use visloc_rs::vision::ransac::RansacReport;
use visloc_rs::{
    localize, localize_frame, localize_frames, localize_with_descriptor_store,
    CorrespondenceBuildError, CorrespondenceBuilder, CrossCheckMatcher, FixedLandmarkSelector,
    ImageLocalizer, InMemoryMapProvider, IntersectCandidateSelector, LocalizationConfig,
    LocalizationPipeline, LocalizationPrior, PriorSubmapSelector, ProjectionCorrespondenceBuilder,
    RadiusLandmarkSelector, RadiusSubmapSelector, RobustPoseEstimator, SelectableMapProvider,
};

#[derive(Debug, Clone)]
struct IdentityPoseEstimator;

impl RobustPoseEstimator for IdentityPoseEstimator {
    fn estimate(
        &self,
        correspondences: &[Correspondence2D3D],
        _camera: &Camera,
    ) -> Option<RansacReport> {
        Some(RansacReport {
            pose: Pose::identity(),
            inliers: (0..correspondences.len()).collect(),
            inlier_reprojection_errors: vec![0.0; correspondences.len()],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
            diagnostics: PoseEstimatorDiagnostics {
                refinement_applied: false,
                pre_refinement_mean_reprojection_error: None,
                post_refinement_mean_reprojection_error: Some(0.0),
                refinement_error_delta: None,
            },
        })
    }
}

#[derive(Debug, Clone)]
struct FixedConfidenceMatcher {
    confidences: Vec<Option<f32>>,
}

impl Matcher for FixedConfidenceMatcher {
    fn match_descriptors(&self, query: &[Vec<f32>], train: &[Vec<f32>]) -> Vec<DescriptorMatch> {
        let count = query.len().min(train.len()).min(self.confidences.len());
        (0..count)
            .map(|index| DescriptorMatch {
                query_index: index,
                train_index: index,
                distance: 0.0,
                second_best_distance: None,
                ratio: None,
                confidence: self.confidences[index],
            })
            .collect()
    }
}

#[derive(Debug, Clone, Default)]
struct RecordingPoseEstimator {
    last_weights: Arc<Mutex<Option<Option<Vec<f32>>>>>,
}

impl RobustPoseEstimator for RecordingPoseEstimator {
    fn estimate(
        &self,
        correspondences: &[Correspondence2D3D],
        _camera: &Camera,
    ) -> Option<RansacReport> {
        *self.last_weights.lock().unwrap() = Some(None);
        Some(identity_report(correspondences))
    }

    fn estimate_with_weights(
        &self,
        correspondences: &[Correspondence2D3D],
        _camera: &Camera,
        weights: &[f32],
    ) -> Option<RansacReport> {
        *self.last_weights.lock().unwrap() = Some(Some(weights.to_vec()));
        Some(identity_report(correspondences))
    }
}

fn identity_report(correspondences: &[Correspondence2D3D]) -> RansacReport {
    RansacReport {
        pose: Pose::identity(),
        inliers: (0..correspondences.len()).collect(),
        inlier_reprojection_errors: vec![0.0; correspondences.len()],
        mean_reprojection_error: 0.0,
        median_reprojection_error: 0.0,
        max_reprojection_error: 0.0,
        diagnostics: PoseEstimatorDiagnostics {
            refinement_applied: false,
            pre_refinement_mean_reprojection_error: None,
            post_refinement_mean_reprojection_error: Some(0.0),
            refinement_error_delta: None,
        },
    }
}

#[derive(Debug, Clone)]
struct StaticFeatureExtractor {
    features: FeatureSet,
}

impl FeatureExtractor for StaticFeatureExtractor {
    type Image = ();
    type Error = Infallible;

    fn extract(&self, _image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        Ok(self.features.clone())
    }
}

#[test]
fn localizes_dummy_query_against_descriptor_map() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 0.0];
        let mut landmark = Landmark::new(index as u64, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let result = localize(
        QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        map,
    );

    assert!(result.success);
    assert_eq!(result.failure_reason, None);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.match_count, 6);
    assert_eq!(result.correspondence_count, 6);
    assert_eq!(result.inlier_count, 6);
    assert_eq!(result.outlier_count, 0);
    assert_eq!(result.inlier_ratio, 1.0);
    assert_eq!(result.inlier_query_indices, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(result.inlier_landmark_ids, vec![0, 1, 2, 3, 4, 5]);
    assert!(result.reprojection_error.unwrap() < 1.0e-6);
    assert!(result.median_reprojection_error.unwrap() < 1.0e-6);
    assert!(result.max_reprojection_error.unwrap() < 1.0e-6);
    assert_eq!(result.inlier_reprojection_errors.len(), 6);
    let diagnostics = result.estimator_diagnostics.unwrap();
    assert!(diagnostics.refinement_applied);
    assert!(diagnostics.post_refinement_mean_reprojection_error.unwrap() < 1.0e-6);
}

#[test]
fn localization_pipeline_extracts_features_from_image_input() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 11.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let extractor = StaticFeatureExtractor {
        features: FeatureSet {
            keypoints,
            descriptors,
        },
    };
    let result = LocalizationPipeline::default()
        .localize_image_with_extractor(&(), camera, &map, &extractor)
        .unwrap();

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn localization_pipeline_extracts_features_for_frame_image_input() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 13.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let extractor = StaticFeatureExtractor {
        features: FeatureSet {
            keypoints,
            descriptors,
        },
    };
    let frame_result = LocalizationPipeline::default()
        .localize_frame_image_with_extractor(42, camera.id, &(), &map, &extractor)
        .unwrap();

    assert_eq!(frame_result.frame_id, 42);
    assert!(frame_result.result.success);
    assert_eq!(frame_result.result.inlier_count, 6);
}

#[test]
fn image_localizer_wraps_extractor_and_pipeline() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 17.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let localizer = ImageLocalizer::new(StaticFeatureExtractor {
        features: FeatureSet {
            keypoints,
            descriptors,
        },
    });
    let result = localizer.localize_image(&(), camera, &map).unwrap();

    assert!(result.success);
    assert_eq!(result.match_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn localization_pipeline_reports_missing_camera_before_extracting_frame_image() {
    let extractor = StaticFeatureExtractor {
        features: FeatureSet {
            keypoints: Vec::new(),
            descriptors: Vec::new(),
        },
    };
    let frame_result = LocalizationPipeline::default()
        .localize_frame_image_with_extractor(42, 999, &(), &VisualMap::new(), &extractor)
        .unwrap();

    assert_eq!(frame_result.frame_id, 42);
    assert_eq!(
        frame_result.result.failure_reason,
        Some(LocalizationFailureReason::MissingCamera { camera_id: 999 })
    );
}

#[test]
fn localizes_with_pixel_noise_and_outlier_matches() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
        Point3::new(-0.5, 0.4, 6.5),
        Point3::new(0.25, 0.75, 8.0),
    ];

    let mut map = VisualMap::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 1.0, 0.25];
        let mut landmark = Landmark::new(index as u64, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);

        let mut keypoint = camera.project(&pose.transform_world_point(point)).unwrap();
        keypoint.x += ((index % 3) as f64 - 1.0) * 0.15;
        keypoint.y += ((index % 2) as f64 - 0.5) * 0.2;
        keypoints.push(keypoint);
        descriptors.push(descriptor);
    }

    for index in 0..3 {
        let descriptor = vec![100.0 + index as f32, 0.0, 0.0];
        let mut landmark = Landmark::new(
            100 + index as u64,
            Point3::new(4.0 + index as f64, -3.0, 9.0),
        );
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);

        keypoints.push(Point2::new(50.0 + index as f64 * 150.0, 420.0));
        descriptors.push(descriptor);
    }

    let result = localize(
        QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        map,
    );

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 11);
    assert_eq!(result.match_count, 11);
    assert_eq!(result.correspondence_count, 11);
    assert!(result.inlier_count >= 8);
    assert_eq!(
        result.outlier_count,
        result.correspondence_count - result.inlier_count
    );
    assert!(result.inlier_ratio >= 8.0 / 11.0);
    let reprojection_error = result.reprojection_error.unwrap();
    assert!(
        reprojection_error < 4.0,
        "mean reprojection error should stay within the RANSAC threshold, got {reprojection_error}"
    );

    let estimated_pose = result.pose.unwrap();
    let translation_error = estimated_pose.world_to_camera.translation.norm();
    assert!(
        translation_error < 0.05,
        "translation should stay near the synthetic ground truth, got {translation_error}"
    );
}

#[test]
fn localizes_with_external_landmark_descriptor_store() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = 100 + index as u64;
        let descriptor = vec![index as f32, 2.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let result = localize_with_descriptor_store(
        QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        map,
        descriptor_store,
    );

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.match_count, 6);
    assert_eq!(result.correspondence_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn localizes_with_in_memory_map_provider_and_external_descriptors() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = 200 + index as u64;
        let descriptor = vec![index as f32, 23.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let provider = InMemoryMapProvider::with_descriptor_store(map, descriptor_store);
    let result = LocalizationPipeline::default().localize_with_provider(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &provider,
    );

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn in_memory_map_provider_builds_radius_submap_from_provider() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let near_points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];
    let far_points = vec![
        Point3::new(100.0, 0.0, 4.0),
        Point3::new(101.0, 0.0, 4.5),
        Point3::new(102.0, 0.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in near_points.iter().chain(far_points.iter()).enumerate() {
        let landmark_id = 300 + index as u64;
        let descriptor = vec![index as f32, 29.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        if index < near_points.len() {
            keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
            descriptors.push(descriptor);
        }
    }

    let full_provider = InMemoryMapProvider::with_descriptor_store(map, descriptor_store);
    let submap_provider =
        InMemoryMapProvider::from_provider_radius(&full_provider, Point3::new(0.0, 0.0, 5.0), 3.0);
    let result = LocalizationPipeline::default().localize_with_provider(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &submap_provider,
    );

    assert_eq!(submap_provider.map.landmarks.len(), 6);
    assert_eq!(
        submap_provider
            .descriptor_store
            .as_ref()
            .expect("submap descriptors should be retained")
            .len(),
        6
    );
    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn selectable_map_provider_applies_radius_submap_selector() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let near_points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];
    let far_points = vec![Point3::new(100.0, 0.0, 4.0), Point3::new(101.0, 0.0, 4.5)];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in near_points.iter().chain(far_points.iter()).enumerate() {
        let landmark_id = 400 + index as u64;
        let descriptor = vec![index as f32, 37.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        if index < near_points.len() {
            keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
            descriptors.push(descriptor);
        }
    }

    let full_provider = InMemoryMapProvider::with_descriptor_store(map, descriptor_store);
    let selected_provider = SelectableMapProvider::new(
        full_provider,
        RadiusSubmapSelector::new(Point3::new(0.0, 0.0, 5.0), 3.0),
    );
    let result = LocalizationPipeline::default().localize_with_provider(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &selected_provider,
    );

    assert_eq!(selected_provider.selected_provider().map.landmarks.len(), 6);
    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn localization_prior_builds_radius_submap_selector() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let near_points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];
    let far_points = vec![Point3::new(80.0, 0.0, 4.0), Point3::new(81.0, 0.0, 4.5)];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in near_points.iter().chain(far_points.iter()).enumerate() {
        let landmark_id = 500 + index as u64;
        let descriptor = vec![index as f32, 41.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        if index < near_points.len() {
            keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
            descriptors.push(descriptor);
        }
    }

    let prior = LocalizationPrior::from_position(Point3::new(0.0, 0.0, 5.0), 3.0);
    let selected_provider = SelectableMapProvider::new(
        InMemoryMapProvider::with_descriptor_store(map, descriptor_store),
        PriorSubmapSelector::new(prior.clone()),
    );
    let result = LocalizationPipeline::default().localize_with_provider(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &selected_provider,
    );

    assert_eq!(prior.center_world(), Some(Point3::new(0.0, 0.0, 5.0)));
    assert!(prior.to_radius_submap_selector().is_some());
    assert_eq!(selected_provider.selected_provider().map.landmarks.len(), 6);
    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
}

#[test]
fn prior_submap_selector_falls_back_to_all_map_without_radius_prior() {
    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    for index in 0..3 {
        let landmark_id = 600 + index;
        let descriptor = vec![index as f32, 43.0];
        map.landmarks.insert(
            landmark_id,
            Landmark::new(landmark_id, Point3::new(index as f64, 0.0, 4.0)),
        );
        descriptor_store.insert(landmark_id, descriptor);
    }

    let provider = SelectableMapProvider::new(
        InMemoryMapProvider::with_descriptor_store(map, descriptor_store),
        PriorSubmapSelector::new(LocalizationPrior::none()),
    );

    assert_eq!(provider.selected_provider().map.landmarks.len(), 3);
    assert_eq!(
        provider
            .selected_provider()
            .descriptor_store
            .as_ref()
            .expect("all-map fallback should retain descriptors")
            .len(),
        3
    );
}

#[test]
fn localizes_frame_using_camera_from_visual_map() {
    let camera = Camera::pinhole(7, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut frame = Frame::new(42, camera.id);

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 7.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }

    let result = localize_frame(frame, map);

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.match_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn localize_frame_reports_missing_camera() {
    let mut frame = Frame::new(42, 999);
    frame.keypoints.push(Point2::new(10.0, 20.0));
    frame.descriptors.push(vec![1.0, 0.0]);

    let result = localize_frame(frame, VisualMap::new());

    assert!(!result.success);
    assert_eq!(
        result.failure_reason,
        Some(LocalizationFailureReason::MissingCamera { camera_id: 999 })
    );
    assert_eq!(result.match_count, 0);
    assert_eq!(result.correspondence_count, 0);
}

#[test]
fn localizes_frame_sequence_statelessly() {
    let camera = Camera::pinhole(7, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut frame_a = Frame::new(100, camera.id);
    let mut frame_b = Frame::new(101, camera.id);

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 8.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        let keypoint = camera.project(&pose.transform_world_point(point)).unwrap();
        frame_a.keypoints.push(keypoint);
        frame_a.descriptors.push(descriptor.clone());
        frame_b.keypoints.push(keypoint);
        frame_b.descriptors.push(descriptor);
    }

    let results = localize_frames(vec![frame_a, frame_b], map);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].frame_id, 100);
    assert_eq!(results[1].frame_id, 101);
    assert!(results
        .iter()
        .all(|frame_result| frame_result.result.success));
    assert!(results
        .iter()
        .all(|frame_result| frame_result.result.inlier_count == 6));
}

#[test]
fn builds_correspondences_without_running_pose_estimation() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 0.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints,
        descriptors,
    };
    let builder = CorrespondenceBuilder::new(BruteForceMatcher { ratio: Some(0.8) });
    let correspondence_set = builder.build(&query, &map, &descriptor_store).unwrap();

    assert_eq!(correspondence_set.candidate_landmark_count, 3);
    assert_eq!(correspondence_set.match_count, 3);
    assert_eq!(correspondence_set.descriptor_matches.len(), 3);
    assert_eq!(correspondence_set.descriptor_matches[0].query_index, 0);
    assert_eq!(correspondence_set.descriptor_matches[0].train_index, 0);
    assert_eq!(correspondence_set.correspondences.len(), 3);
    assert_eq!(correspondence_set.query_indices, vec![0, 1, 2]);
    assert_eq!(correspondence_set.landmark_ids, vec![1, 2, 3]);
    assert_eq!(correspondence_set.correspondences[0].point3d, points[0]);
    assert!(correspondence_set
        .correspondences
        .iter()
        .all(|correspondence| correspondence.confidence.is_none()));
}

#[test]
fn correspondence_builder_copies_match_confidence_to_correspondences() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 0.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints,
        descriptors,
    };
    let builder = CorrespondenceBuilder::new(FixedConfidenceMatcher {
        confidences: vec![Some(0.2), None, Some(0.9)],
    });
    let correspondence_set = builder.build(&query, &map, &descriptor_store).unwrap();

    let confidences = correspondence_set
        .correspondences
        .iter()
        .map(|correspondence| correspondence.confidence)
        .collect::<Vec<_>>();
    assert_eq!(confidences, vec![Some(0.2), None, Some(0.9)]);
}

#[test]
fn correspondence_builder_accepts_cross_check_matcher() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [Point3::new(-1.0, -1.0, 4.0), Point3::new(1.0, -1.0, 4.5)];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();
    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 0.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints,
        descriptors,
    };
    let builder =
        CorrespondenceBuilder::new(CrossCheckMatcher::new(BruteForceMatcher { ratio: None }));

    let correspondence_set = builder.build(&query, &map, &descriptor_store).unwrap();

    assert_eq!(correspondence_set.match_count, 2);
    assert_eq!(correspondence_set.descriptor_matches.len(), 2);
    assert!(correspondence_set
        .descriptor_matches
        .iter()
        .all(|descriptor_match| descriptor_match.ratio.is_some()));
}

#[test]
fn correspondence_builder_reports_shape_mismatch() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let query = QueryImage {
        camera,
        keypoints: vec![Point2::new(10.0, 20.0)],
        descriptors: Vec::new(),
    };
    let builder = CorrespondenceBuilder::new(BruteForceMatcher::default());
    let error = builder
        .build(&query, &VisualMap::new(), &LandmarkDescriptorStore::new())
        .unwrap_err();

    assert_eq!(
        error,
        CorrespondenceBuildError::QueryFeatureShapeMismatch {
            keypoint_count: 1,
            descriptor_count: 0,
        }
    );
}

#[test]
fn correspondence_builder_uses_candidate_selector() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut projected_keypoints = Vec::new();
    let mut map_descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 0.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        projected_keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        map_descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints: vec![projected_keypoints[1], projected_keypoints[3]],
        descriptors: vec![map_descriptors[1].clone(), map_descriptors[3].clone()],
    };
    let builder = CorrespondenceBuilder::with_candidate_selector(
        BruteForceMatcher { ratio: Some(0.8) },
        FixedLandmarkSelector::new(vec![2, 4]),
    );
    let correspondence_set = builder.build(&query, &map, &descriptor_store).unwrap();

    assert_eq!(correspondence_set.candidate_landmark_count, 2);
    assert_eq!(correspondence_set.match_count, 2);
    assert_eq!(correspondence_set.correspondences.len(), 2);
    assert_eq!(correspondence_set.query_indices, vec![0, 1]);
    assert_eq!(correspondence_set.landmark_ids, vec![2, 4]);
    assert_eq!(correspondence_set.correspondences[0].point3d, points[1]);
    assert_eq!(correspondence_set.correspondences[1].point3d, points[3]);
}

#[test]
fn localization_pipeline_uses_candidate_selector() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
        Point3::new(-0.5, 0.4, 6.5),
        Point3::new(0.25, 0.75, 8.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut projected_keypoints = Vec::new();
    let mut map_descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 1.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        projected_keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        map_descriptors.push(descriptor);
    }

    let pipeline = LocalizationPipeline::with_candidate_selector(
        BruteForceMatcher { ratio: Some(0.8) },
        FixedLandmarkSelector::new(vec![1, 2, 3, 4, 5, 6]),
        LocalizationConfig::default(),
    );
    let result = pipeline.localize_with_descriptor_store(
        &QueryImage {
            camera,
            keypoints: projected_keypoints[..6].to_vec(),
            descriptors: map_descriptors[..6].to_vec(),
        },
        &map,
        &descriptor_store,
    );

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 6);
    assert_eq!(result.match_count, 6);
    assert_eq!(result.correspondence_count, 6);
    assert_eq!(result.inlier_count, 6);
}

#[test]
fn correspondence_builder_reports_empty_candidate_selector() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let query = QueryImage {
        camera,
        keypoints: vec![Point2::new(10.0, 20.0)],
        descriptors: vec![vec![1.0, 0.0]],
    };
    let builder = CorrespondenceBuilder::with_candidate_selector(
        BruteForceMatcher::default(),
        FixedLandmarkSelector::new(Vec::new()),
    );
    let error = builder
        .build(&query, &VisualMap::new(), &LandmarkDescriptorStore::new())
        .unwrap_err();

    assert_eq!(error, CorrespondenceBuildError::NoCandidateLandmarks);
}

#[test]
fn radius_landmark_selector_filters_candidates_by_world_distance() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-0.5, 0.0, 4.0),
        Point3::new(0.5, 0.0, 4.2),
        Point3::new(0.0, 0.5, 4.4),
        Point3::new(8.0, 0.0, 4.0),
        Point3::new(9.0, 0.0, 4.5),
        Point3::new(10.0, 0.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut projected_keypoints = Vec::new();
    let mut map_descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 3.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        projected_keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        map_descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints: projected_keypoints[..3].to_vec(),
        descriptors: map_descriptors[..3].to_vec(),
    };
    let builder = CorrespondenceBuilder::with_candidate_selector(
        BruteForceMatcher { ratio: Some(0.8) },
        RadiusLandmarkSelector::new(Point3::new(0.0, 0.0, 4.2), 1.0),
    );
    let correspondence_set = builder.build(&query, &map, &descriptor_store).unwrap();

    assert_eq!(correspondence_set.candidate_landmark_count, 3);
    assert_eq!(correspondence_set.match_count, 3);
    assert_eq!(correspondence_set.correspondences.len(), 3);
}

#[test]
fn intersect_candidate_selector_combines_existing_selection_with_radius_prior() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [
        Point3::new(0.0, 0.0, 4.0),
        Point3::new(0.5, 0.0, 4.2),
        Point3::new(8.0, 0.0, 4.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 7.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let query = QueryImage {
        camera,
        keypoints: keypoints[1..2].to_vec(),
        descriptors: descriptors[1..2].to_vec(),
    };
    let selector = IntersectCandidateSelector::new(
        FixedLandmarkSelector::new(vec![2, 3]),
        RadiusLandmarkSelector::new(Point3::new(0.0, 0.0, 4.1), 1.0),
    );
    let correspondence_set = CorrespondenceBuilder::with_candidate_selector(
        BruteForceMatcher { ratio: Some(0.8) },
        selector,
    )
    .build(&query, &map, &descriptor_store)
    .unwrap();

    assert_eq!(correspondence_set.candidate_landmark_count, 1);
    assert_eq!(correspondence_set.landmark_ids, vec![2]);
}

#[test]
fn localization_pipeline_accepts_custom_pose_estimator() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 5.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let pipeline = LocalizationPipeline::with_pose_estimator(
        BruteForceMatcher { ratio: Some(0.8) },
        FixedLandmarkSelector::new(vec![1, 2, 3]),
        IdentityPoseEstimator,
        LocalizationConfig::default(),
    );
    let result = pipeline.localize_with_descriptor_store(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &map,
        &descriptor_store,
    );

    assert!(result.success);
    assert_eq!(result.candidate_landmark_count, 3);
    assert_eq!(result.match_count, 3);
    assert_eq!(result.correspondence_count, 3);
    assert_eq!(result.inlier_count, 3);
    assert_eq!(result.outlier_count, 0);
    assert_eq!(result.inlier_ratio, 1.0);
    assert_eq!(result.inlier_query_indices, vec![0, 1, 2]);
    assert_eq!(result.inlier_landmark_ids, vec![1, 2, 3]);
    assert_eq!(result.inlier_reprojection_errors, vec![0.0, 0.0, 0.0]);
    assert_eq!(
        result.estimator_diagnostics.unwrap(),
        PoseEstimatorDiagnostics {
            refinement_applied: false,
            pre_refinement_mean_reprojection_error: None,
            post_refinement_mean_reprojection_error: Some(0.0),
            refinement_error_delta: None,
        }
    );
    assert_eq!(result.pose.unwrap(), Pose::identity());
}

#[test]
fn localization_pipeline_routes_correspondence_confidence_to_pose_estimator() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 5.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let estimator = RecordingPoseEstimator::default();
    let recorded_weights = estimator.last_weights.clone();
    let pipeline = LocalizationPipeline::with_pose_estimator(
        FixedConfidenceMatcher {
            confidences: vec![Some(0.1), None, Some(0.8)],
        },
        FixedLandmarkSelector::new(vec![1, 2, 3]),
        estimator,
        LocalizationConfig::default(),
    );
    let result = pipeline.localize_with_descriptor_store(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &map,
        &descriptor_store,
    );

    assert!(result.success);
    assert_eq!(
        *recorded_weights.lock().unwrap(),
        Some(Some(vec![0.1, 0.0, 0.8]))
    );
}

#[test]
fn quality_gate_can_reject_estimated_pose_but_keep_diagnostics() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
    ];

    let mut map = VisualMap::new();
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut keypoints = Vec::new();
    let mut descriptors = Vec::new();

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 6.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        keypoints.push(camera.project(&pose.transform_world_point(point)).unwrap());
        descriptors.push(descriptor);
    }

    let config = LocalizationConfig {
        min_inliers: 4,
        ..LocalizationConfig::default()
    };
    let pipeline = LocalizationPipeline::with_pose_estimator(
        BruteForceMatcher { ratio: Some(0.8) },
        FixedLandmarkSelector::new(vec![1, 2, 3]),
        IdentityPoseEstimator,
        config,
    );
    let result = pipeline.localize_with_descriptor_store(
        &QueryImage {
            camera,
            keypoints,
            descriptors,
        },
        &map,
        &descriptor_store,
    );

    assert!(!result.success);
    assert_eq!(
        result.failure_reason,
        Some(LocalizationFailureReason::QualityGateFailed)
    );
    assert!(result.pose.is_some());
    assert_eq!(result.inlier_count, 3);
    assert_eq!(result.inlier_ratio, 1.0);
    assert_eq!(result.inlier_landmark_ids, vec![1, 2, 3]);
}

#[test]
fn reports_query_feature_shape_mismatch() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let query = QueryImage {
        camera,
        keypoints: vec![Point2::new(10.0, 20.0)],
        descriptors: Vec::new(),
    };

    let result = localize(query, VisualMap::new());

    assert!(!result.success);
    assert_eq!(
        result.failure_reason,
        Some(LocalizationFailureReason::QueryFeatureShapeMismatch {
            keypoint_count: 1,
            descriptor_count: 0,
        })
    );
    assert_eq!(result.match_count, 0);
    assert_eq!(result.correspondence_count, 0);
    assert_eq!(result.candidate_landmark_count, 0);
}

#[test]
fn reports_missing_map_descriptors() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut map = VisualMap::new();
    map.landmarks
        .insert(1, Landmark::new(1, Point3::new(0.0, 0.0, 5.0)));
    let query = QueryImage {
        camera,
        keypoints: vec![Point2::new(10.0, 20.0)],
        descriptors: vec![vec![1.0, 0.0]],
    };

    let result = localize(query, map);

    assert!(!result.success);
    assert_eq!(
        result.failure_reason,
        Some(LocalizationFailureReason::NoMapDescriptors)
    );
    assert_eq!(result.match_count, 0);
    assert_eq!(result.correspondence_count, 0);
    assert_eq!(result.candidate_landmark_count, 1);
}

#[test]
fn reports_pose_estimation_failure_with_too_few_correspondences() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut map = VisualMap::new();
    let mut landmark = Landmark::new(1, Point3::new(0.0, 0.0, 5.0));
    landmark.descriptor = Some(vec![1.0, 0.0]);
    map.landmarks.insert(1, landmark);

    let query = QueryImage {
        camera,
        keypoints: vec![Point2::new(320.0, 240.0)],
        descriptors: vec![vec![1.0, 0.0]],
    };

    let result = localize(query, map);

    assert!(!result.success);
    assert_eq!(
        result.failure_reason,
        Some(LocalizationFailureReason::PoseEstimationFailed {
            correspondence_count: 1,
        })
    );
    assert_eq!(result.match_count, 1);
    assert_eq!(result.correspondence_count, 1);
    assert_eq!(result.candidate_landmark_count, 1);
    let diagnostics = result.pose_failure_diagnostics.unwrap();
    assert_eq!(
        diagnostics.reason,
        PoseEstimationFailureReason::InsufficientCorrespondences
    );
    assert_eq!(diagnostics.minimum_correspondence_count, Some(6));
    assert_eq!(diagnostics.ransac_iterations, Some(128));
    assert_eq!(diagnostics.ransac_reprojection_threshold, Some(4.0));
}

#[test]
fn projection_window_matching_finds_the_same_pose_as_appearance_matching_on_a_clean_frame() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut descriptor_store = LandmarkDescriptorStore::new();
    let mut frame = Frame::new(10, camera.id);

    for (index, point) in points.iter().enumerate() {
        let landmark_id = index as u64 + 1;
        let descriptor = vec![index as f32, 3.0];
        map.landmarks
            .insert(landmark_id, Landmark::new(landmark_id, *point));
        descriptor_store.insert(landmark_id, descriptor.clone());
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }
    let query = QueryImage::from_frame(&frame, camera.clone());

    let appearance_set = CorrespondenceBuilder::new(BruteForceMatcher { ratio: Some(0.8) })
        .build(&query, &map, &descriptor_store)
        .unwrap();
    let projection_set =
        ProjectionCorrespondenceBuilder::new(BruteForceMatcher { ratio: Some(0.8) })
            .build_with_pose_prior(&query, &map, &descriptor_store, &pose, 15.0)
            .unwrap();

    // Same query-keypoint <-> landmark pairing, independent of the two
    // builders' different iteration order (appearance iterates by query
    // index, projection iterates by landmark index).
    let mut appearance_pairs: Vec<(usize, u64)> = appearance_set
        .query_indices
        .iter()
        .copied()
        .zip(appearance_set.landmark_ids.iter().copied())
        .collect();
    let mut projection_pairs: Vec<(usize, u64)> = projection_set
        .query_indices
        .iter()
        .copied()
        .zip(projection_set.landmark_ids.iter().copied())
        .collect();
    appearance_pairs.sort_unstable();
    projection_pairs.sort_unstable();
    assert_eq!(appearance_pairs, projection_pairs);
    assert_eq!(appearance_set.correspondences.len(), points.len());
    assert_eq!(projection_set.correspondences.len(), points.len());

    let pipeline = LocalizationPipeline::default();
    let appearance_result =
        pipeline.localize_frame_with_descriptor_store(&frame, &map, &descriptor_store);
    let projection_result = pipeline.localize_frame_with_projection_window_and_descriptor_store(
        &frame,
        &map,
        &descriptor_store,
        &pose,
        None,
        15.0,
    );

    assert!(appearance_result.success);
    assert!(projection_result.success);
    assert_eq!(
        appearance_result.inlier_count,
        projection_result.inlier_count
    );
    let appearance_center = appearance_result.pose.unwrap().camera_center_world();
    let projection_center = projection_result.pose.unwrap().camera_center_world();
    assert!((appearance_center - projection_center).norm() < 1.0e-9);
}

#[test]
fn projection_window_matching_disambiguates_identical_descriptors_at_different_locations() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    // Two landmarks, well separated in the image, sharing the exact same
    // descriptor: appearance-only matching cannot tell them apart.
    let landmark_1_point = Point3::new(-2.0, 0.0, 5.0);
    let landmark_2_point = Point3::new(2.0, 0.0, 5.0);
    let shared_descriptor = vec![1.0, 0.0];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut landmark_1 = Landmark::new(1, landmark_1_point);
    landmark_1.descriptor = Some(shared_descriptor.clone());
    let mut landmark_2 = Landmark::new(2, landmark_2_point);
    landmark_2.descriptor = Some(shared_descriptor.clone());
    map.landmarks.insert(1, landmark_1);
    map.landmarks.insert(2, landmark_2);

    let mut descriptor_store = LandmarkDescriptorStore::new();
    descriptor_store.insert(1, shared_descriptor.clone());
    descriptor_store.insert(2, shared_descriptor.clone());

    // Query keypoint 0 sits where landmark 2 projects; keypoint 1 sits
    // where landmark 1 projects (deliberately "swapped" relative to
    // landmark id order, so a naive first-wins tie-break would pair them
    // wrong).
    let keypoint_near_landmark_2 = camera
        .project(&pose.transform_world_point(&landmark_2_point))
        .unwrap();
    let keypoint_near_landmark_1 = camera
        .project(&pose.transform_world_point(&landmark_1_point))
        .unwrap();
    let query = QueryImage {
        camera: camera.clone(),
        keypoints: vec![keypoint_near_landmark_2, keypoint_near_landmark_1],
        descriptors: vec![shared_descriptor.clone(), shared_descriptor.clone()],
    };

    // Appearance-only matching (no ratio test, so identical-distance ties
    // don't get filtered out): both query keypoints tie-break to the same
    // (first) landmark, producing an inconsistent pairing.
    let appearance_set = CorrespondenceBuilder::new(BruteForceMatcher { ratio: None })
        .build(&query, &map, &descriptor_store)
        .unwrap();
    assert_eq!(appearance_set.landmark_ids, vec![1, 1]);

    // Projection-guided matching uses the (correct) pose prior to restrict
    // each landmark's descriptor search to its own projection window,
    // correctly disambiguating despite the identical descriptors.
    let projection_set = ProjectionCorrespondenceBuilder::new(BruteForceMatcher { ratio: None })
        .build_with_pose_prior(&query, &map, &descriptor_store, &pose, 5.0)
        .unwrap();

    let pairs: std::collections::HashMap<u64, usize> = projection_set
        .landmark_ids
        .iter()
        .copied()
        .zip(projection_set.query_indices.iter().copied())
        .collect();
    assert_eq!(pairs.get(&1), Some(&1));
    assert_eq!(pairs.get(&2), Some(&0));
}

#[test]
fn projection_window_matching_selects_one_keypoint_and_applies_ratio_within_window() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::identity();
    let point = Point3::new(0.0, 0.0, 5.0);
    let projected = camera.project(&point).unwrap();
    let mut landmark = Landmark::new(1, point);
    landmark.descriptor = Some(vec![0.0]);
    let mut map = VisualMap::new();
    map.landmarks.insert(1, landmark);
    let mut descriptor_store = LandmarkDescriptorStore::new();
    descriptor_store.insert(1, vec![0.0]);
    let query = QueryImage {
        camera,
        keypoints: vec![projected, Point2::new(projected.x + 1.0, projected.y)],
        descriptors: vec![vec![0.10], vec![0.11]],
    };

    // The best and second-best descriptors are too similar, so a landmark ->
    // window ratio test must reject the ambiguous assignment.
    let ratio_result = ProjectionCorrespondenceBuilder::new(BruteForceMatcher { ratio: Some(0.8) })
        .build_with_pose_prior(&query, &map, &descriptor_store, &pose, 5.0);
    assert!(ratio_result.is_err());

    // With the ratio gate disabled the same window still contributes only its
    // single nearest keypoint, never both keypoints for one landmark.
    let result = ProjectionCorrespondenceBuilder::new(BruteForceMatcher { ratio: None })
        .build_with_pose_prior(&query, &map, &descriptor_store, &pose, 5.0)
        .unwrap();
    assert_eq!(result.correspondences.len(), 1);
    assert_eq!(result.query_indices, vec![0]);
    assert_eq!(result.landmark_ids, vec![1]);
}
