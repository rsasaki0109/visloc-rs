#![allow(clippy::useless_vec)]

use std::convert::Infallible;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{
    ConstantVelocityMotionModel, FeatureExtractor, FeatureSet, ImageTracker, InMemoryMapProvider,
    LocalizationPipeline, MapProviderStats, MotionModel, PriorSubmapSelector,
    SelectableMapProvider, Tracker, TrackingConfig, TrackingEvent, TrackingResult, TrackingState,
};

#[derive(Debug, Clone)]
struct FixedPoseMotionModel {
    pose: Pose,
}

impl MotionModel for FixedPoseMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        _last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        Some(self.pose.clone())
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

fn pose_with_identity_rotation_at_center(center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
}

fn extractor_from_frame(frame: &Frame) -> StaticFeatureExtractor {
    StaticFeatureExtractor {
        features: FeatureSet {
            keypoints: frame.keypoints.clone(),
            descriptors: frame.descriptors.clone(),
        },
    }
}

fn successful_tracking_result(frame_id: u64, pose: Pose) -> TrackingResult {
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event: TrackingEvent::Tracked,
        successive_failures: 0,
        pose_prior: None,
        used_pose_prior: false,
        map_landmark_count: 0,
        map_stats: MapProviderStats::default(),
        localization: visloc_rs::core::types::LocalizationResult::success(
            visloc_rs::core::types::LocalizationSuccess {
                pose,
                candidate_landmark_count: 0,
                match_count: 0,
                correspondence_count: 0,
                inliers: Vec::new(),
                inlier_query_indices: Vec::new(),
                inlier_landmark_ids: Vec::new(),
                inlier_reprojection_errors: Vec::new(),
                mean_reprojection_error: 0.0,
                median_reprojection_error: 0.0,
                max_reprojection_error: 0.0,
            },
        ),
    }
}

fn build_map_and_frame(frame_id: u64, camera_id: u64) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
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
    let mut frame = Frame::new(frame_id, camera.id);

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }

    (map, frame)
}

#[test]
fn constant_velocity_motion_model_extrapolates_camera_center() {
    let mut model = ConstantVelocityMotionModel::new();
    let frame = Frame::new(3, 1);
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(2.0, 0.0, 0.0));

    model.observe(&successful_tracking_result(1, pose_a));
    model.observe(&successful_tracking_result(2, pose_b));
    let prediction = model.predict_pose(&frame, None, None).unwrap();

    let predicted_center = prediction.camera_center_world();
    assert!((predicted_center - Point3::new(4.0, 0.0, 0.0)).norm() < 1.0e-9);
}

#[test]
fn tracker_enters_tracking_after_successful_localization() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let result = tracker.track_frame(&frame, &map);

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert_eq!(result.event, TrackingEvent::Initialized);
    assert_eq!(result.successive_failures, 0);
    assert!(result.pose_prior.is_none());
    assert!(!result.used_pose_prior);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert!(result.localization.success);
    assert_eq!(tracker.state(), TrackingState::Tracking);
    assert_eq!(tracker.last_result().unwrap().frame_id, 10);
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
    assert!(tracker.last_successful_pose().is_some());
    assert_eq!(tracker.stats().frame_count, 1);
    assert_eq!(tracker.stats().successful_frame_count, 1);
    assert_eq!(tracker.stats().failed_frame_count, 0);
}

#[test]
fn tracker_tracks_frame_with_map_provider() {
    let (map, frame) = build_map_and_frame(10, 1);
    let provider = InMemoryMapProvider::new(map);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let result = tracker.track_frame_with_provider(&frame, &provider);

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(tracker.stats().successful_frame_count, 1);
}

#[test]
fn tracker_predicts_localization_prior_for_next_frame() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let first = tracker.track_frame(&frame, &map);
    let prior = tracker.localization_prior_for_frame(&frame, 8.0);
    let provider = SelectableMapProvider::new(
        InMemoryMapProvider::new(map),
        PriorSubmapSelector::new(prior.clone()),
    );

    assert!(first.localization.success);
    assert_eq!(prior.radius, Some(8.0));
    assert!(prior.pose.is_some());
    assert!(prior.to_radius_submap_selector().is_some());
    assert_eq!(provider.selected_provider().map.landmarks.len(), 6);
}

#[test]
fn tracker_tracks_with_motion_prior_submap_provider() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let first = tracker.track_frame_with_prior_submap_provider(&frame, &provider, 8.0);
    let second = tracker.track_frame_with_prior_submap_provider(&frame, &provider, 8.0);

    assert!(first.localization.success);
    assert_eq!(first.map_landmark_count, 12);
    assert_eq!(first.map_stats.landmark_count, 12);
    assert_eq!(first.map_stats.descriptor_count, 12);
    assert_eq!(first.localization.candidate_landmark_count, 12);
    assert!(first.pose_prior.is_none());
    assert!(second.localization.success);
    assert_eq!(second.map_landmark_count, 6);
    assert_eq!(second.map_stats.landmark_count, 6);
    assert_eq!(second.map_stats.descriptor_count, 6);
    assert_eq!(second.localization.candidate_landmark_count, 6);
    assert!(second.pose_prior.is_some());
}

#[test]
fn image_tracker_tracks_extracted_frame_image() {
    let (map, frame) = build_map_and_frame(10, 1);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let result = image_tracker
        .track_frame_image(10, frame.camera_id, &(), &map)
        .unwrap();

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert_eq!(result.event, TrackingEvent::Initialized);
    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(result.localization.inlier_count, 6);
    assert_eq!(image_tracker.tracker().stats().successful_frame_count, 1);
}

#[test]
fn image_tracker_tracks_with_motion_prior_submap_provider() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let first = image_tracker
        .track_frame_image_with_prior_submap_provider(10, frame.camera_id, &(), &provider, 8.0)
        .unwrap();
    let second = image_tracker
        .track_frame_image_with_prior_submap_provider(11, frame.camera_id, &(), &provider, 8.0)
        .unwrap();

    assert!(first.localization.success);
    assert_eq!(first.map_landmark_count, 12);
    assert_eq!(first.map_stats.landmark_count, 12);
    assert_eq!(first.map_stats.descriptor_count, 12);
    assert_eq!(first.localization.candidate_landmark_count, 12);
    assert!(second.localization.success);
    assert_eq!(second.map_landmark_count, 6);
    assert_eq!(second.map_stats.landmark_count, 6);
    assert_eq!(second.map_stats.descriptor_count, 6);
    assert_eq!(second.localization.candidate_landmark_count, 6);
    assert!(second.pose_prior.is_some());
}

#[test]
fn tracking_result_exposes_pose_prior_as_localization_prior() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
        },
    );

    let first = tracker.track_frame(&frame, &map);
    let second = tracker.track_frame(&frame, &map);
    let first_prior = first.localization_prior(8.0);
    let second_prior = second.localization_prior(8.0);

    assert!(first_prior.pose.is_none());
    assert_eq!(first_prior.radius, None);
    assert!(second_prior.pose.is_some());
    assert_eq!(second_prior.radius, Some(8.0));
    assert!(second_prior.to_radius_submap_selector().is_some());
}

#[test]
fn image_tracker_tracks_frame_image_with_map_provider() {
    let (map, frame) = build_map_and_frame(10, 1);
    let provider = InMemoryMapProvider::new(map);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let result = image_tracker
        .track_frame_image_with_provider(10, frame.camera_id, &(), &provider)
        .unwrap();

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(image_tracker.tracker().stats().successful_frame_count, 1);
}

#[test]
fn tracker_becomes_lost_after_successive_failures() {
    let (_map, frame) = build_map_and_frame(10, 1);
    let empty_map = VisualMap::new();
    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            ..TrackingConfig::default()
        },
    );

    let first = tracker.track_frame(&frame, &empty_map);
    let second = tracker.track_frame(&frame, &empty_map);

    assert_eq!(first.state, TrackingState::Uninitialized);
    assert_eq!(first.event, TrackingEvent::TrackingFailed);
    assert!(!first.localization.success);
    assert_eq!(first.successive_failures, 1);
    assert_eq!(second.state, TrackingState::Lost);
    assert_eq!(second.event, TrackingEvent::Lost);
    assert!(first.pose_prior.is_none());
    assert!(!first.used_pose_prior);
    assert!(second.pose_prior.is_none());
    assert!(!second.used_pose_prior);
    assert!(!second.localization.success);
    assert_eq!(second.map_landmark_count, 0);
    assert_eq!(second.map_stats.landmark_count, 0);
    assert_eq!(second.map_stats.descriptor_count, 0);
    assert_eq!(second.successive_failures, 2);
    assert_eq!(tracker.state(), TrackingState::Lost);
    assert_eq!(tracker.last_result().unwrap().event, TrackingEvent::Lost);
    assert_eq!(tracker.last_successful_frame_id(), None);
    assert!(tracker.last_successful_pose().is_none());
    assert_eq!(tracker.stats().frame_count, 2);
    assert_eq!(tracker.stats().successful_frame_count, 0);
    assert_eq!(tracker.stats().failed_frame_count, 2);
    assert_eq!(tracker.stats().lost_count, 1);
}

#[test]
fn tracker_reports_relocalized_after_lost_success() {
    let (map, frame) = build_map_and_frame(10, 1);
    let empty_map = VisualMap::new();
    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 1,
            ..TrackingConfig::default()
        },
    );

    let lost = tracker.track_frame(&frame, &empty_map);
    let relocalized = tracker.track_frame(&frame, &map);

    assert_eq!(lost.state, TrackingState::Lost);
    assert_eq!(lost.event, TrackingEvent::Lost);
    assert_eq!(relocalized.state, TrackingState::Tracking);
    assert_eq!(relocalized.event, TrackingEvent::Relocalized);
    assert!(relocalized.pose_prior.is_none());
    assert!(!relocalized.used_pose_prior);
    assert!(relocalized.localization.success);
    assert_eq!(relocalized.map_landmark_count, 6);
    assert_eq!(relocalized.map_stats.landmark_count, 6);
    assert_eq!(relocalized.map_stats.descriptor_count, 6);
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
    assert!(tracker.last_successful_pose().is_some());
    assert_eq!(tracker.stats().frame_count, 2);
    assert_eq!(tracker.stats().successful_frame_count, 1);
    assert_eq!(tracker.stats().failed_frame_count, 1);
    assert_eq!(tracker.stats().lost_count, 1);
    assert_eq!(tracker.stats().relocalization_count, 1);
}

#[test]
fn tracker_uses_last_pose_prior_to_limit_landmark_candidates() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    let far_points = [
        Point3::new(100.0, 0.0, 5.0),
        Point3::new(101.0, 0.0, 5.0),
        Point3::new(102.0, 0.0, 5.0),
        Point3::new(103.0, 0.0, 5.0),
        Point3::new(104.0, 0.0, 5.0),
        Point3::new(105.0, 0.0, 5.0),
    ];
    for (index, point) in far_points.iter().enumerate() {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
        },
    );

    let first = tracker.track_frame(&frame, &map);
    let second = tracker.track_frame(&frame, &map);

    assert!(first.localization.success);
    assert_eq!(first.map_landmark_count, 12);
    assert!(first.pose_prior.is_none());
    assert!(!first.used_pose_prior);
    assert_eq!(first.localization.candidate_landmark_count, 12);
    assert!(second.localization.success);
    assert_eq!(second.map_landmark_count, 12);
    assert!(second.pose_prior.is_some());
    assert!(second.used_pose_prior);
    assert_eq!(second.localization.candidate_landmark_count, 6);
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
}

#[test]
fn tracker_accepts_custom_motion_model_for_pose_prior() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    let far_points = [
        Point3::new(100.0, 0.0, 5.0),
        Point3::new(101.0, 0.0, 5.0),
        Point3::new(102.0, 0.0, 5.0),
        Point3::new(103.0, 0.0, 5.0),
        Point3::new(104.0, 0.0, 5.0),
        Point3::new(105.0, 0.0, 5.0),
    ];
    for (index, point) in far_points.iter().enumerate() {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let far_prior =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-100.0, 0.0, 0.0));
    let mut tracker = Tracker::with_motion_model(
        LocalizationPipeline::default(),
        FixedPoseMotionModel { pose: far_prior },
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
        },
    );

    let result = tracker.track_frame(&frame, &map);

    assert!(!result.localization.success);
    assert_eq!(result.map_landmark_count, 12);
    assert!(result.pose_prior.is_some());
    assert!(result.used_pose_prior);
    assert_eq!(result.localization.candidate_landmark_count, 6);
    assert_eq!(tracker.last_successful_frame_id(), None);
}
