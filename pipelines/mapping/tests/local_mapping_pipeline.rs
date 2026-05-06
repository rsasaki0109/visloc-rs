use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::{
    Camera, Frame, Keyframe, LocalizationResult, LocalizationSuccess, VisualMap,
};
use visloc_mapping::{
    LandmarkCandidate, LandmarkCandidateMappingFailureReason, LandmarkCandidateObservation,
    LinearTriangulator, LocalMapWindow, LocalMappingPipeline, LocalRefinementReason,
    LocalRefinementResult, LocalRefiner, NoopLocalRefiner, SimpleKeyframePolicy, StagedMapUpdate,
};
use visloc_tracking::{TrackingEvent, TrackingResult, TrackingState};

fn pose_at_center(center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
}

fn keyframe(frame_id: u64, pose: Pose, keypoint: Point2<f64>) -> Keyframe {
    let mut frame = Frame::new(frame_id, 1);
    frame.pose = Some(pose);
    frame.keypoints.push(keypoint);
    Keyframe {
        frame,
        observations: Vec::new(),
    }
}

fn tracking_success(frame_id: u64, pose: Pose) -> TrackingResult {
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event: TrackingEvent::Initialized,
        successive_failures: 0,
        pose_prior: None,
        used_pose_prior: false,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: Default::default(),
        localization: LocalizationResult::success(LocalizationSuccess {
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
        }),
    }
}

fn tracking_failure(frame_id: u64) -> TrackingResult {
    let mut result = tracking_success(frame_id, Pose::identity());
    result.localization = LocalizationResult::failure(
        visloc_core::types::LocalizationFailureReason::NoDescriptorMatches,
        0,
        0,
        0,
    );
    result
}

fn map_keyframe_candidate() -> (VisualMap, Keyframe, TrackingResult, LandmarkCandidate) {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let point = Point3::new(0.2, -0.1, 5.0);
    let pose_a = pose_at_center(Vector3::new(0.0, 0.0, 0.0));
    let pose_b = pose_at_center(Vector3::new(1.0, 0.0, 0.0));
    let pixel_a = camera
        .project(&pose_a.transform_world_point(&point))
        .unwrap();
    let pixel_b = camera
        .project(&pose_b.transform_world_point(&point))
        .unwrap();

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera);
    map.keyframes.insert(1, keyframe(1, pose_a, pixel_a));

    let selected_keyframe = keyframe(2, pose_b.clone(), pixel_b);
    let tracking = tracking_success(2, pose_b);
    let candidate = LandmarkCandidate::new(100)
        .with_observation(LandmarkCandidateObservation::new(1, 0, pixel_a))
        .with_observation(LandmarkCandidateObservation::new(2, 0, pixel_b));

    (map, selected_keyframe, tracking, candidate)
}

#[derive(Debug, Clone, PartialEq)]
struct CountingRefiner;

impl LocalRefiner for CountingRefiner {
    fn refine(
        &self,
        _map: &VisualMap,
        _local_window: &LocalMapWindow,
        staged_update: &mut StagedMapUpdate,
    ) -> LocalRefinementResult {
        LocalRefinementResult {
            refined: true,
            reason: LocalRefinementReason::Refined,
            keyframe_count: staged_update.keyframes.len(),
            landmark_count: staged_update.landmarks.len(),
        }
    }
}

#[test]
fn local_mapping_pipeline_stages_selected_keyframe_and_triangulated_landmark() {
    let (mut map, selected_keyframe, tracking, candidate) = map_keyframe_candidate();
    let mut pipeline = LocalMappingPipeline::default();

    let result = pipeline.process_keyframe(&map, &tracking, selected_keyframe, [candidate]);

    assert!(result.keyframe_decision.selected);
    assert_eq!(result.local_window.keyframe_ids, vec![1, 2]);
    assert_eq!(result.triangulated_landmarks.len(), 1);
    assert!(result.candidate_failures.is_empty());
    assert!(result.staged_update_validation.is_valid());
    assert_eq!(result.refinement.reason, LocalRefinementReason::Noop);
    assert!(result.is_ready_to_apply());
    assert_eq!(result.staged_update.keyframes.len(), 1);
    assert_eq!(result.staged_update.landmarks.len(), 1);
    assert_eq!(result.staged_update.observations.len(), 2);

    let applied = result.staged_update.apply_to(&mut map).unwrap();

    assert_eq!(applied.keyframe_count, 1);
    assert_eq!(applied.landmark_count, 1);
    assert_eq!(applied.observation_count, 2);
    assert!(map.validate().is_valid());
}

#[test]
fn local_mapping_pipeline_does_not_stage_when_keyframe_is_rejected() {
    let (map, selected_keyframe, _tracking, candidate) = map_keyframe_candidate();
    let mut pipeline = LocalMappingPipeline::default();
    let tracking = tracking_failure(2);

    let result = pipeline.process_keyframe(&map, &tracking, selected_keyframe, [candidate]);

    assert!(!result.keyframe_decision.selected);
    assert!(result.staged_update.is_empty());
    assert!(result.triangulated_landmarks.is_empty());
    assert!(result.candidate_failures.is_empty());
    assert_eq!(
        result.refinement.reason,
        LocalRefinementReason::NoSelectedKeyframe
    );
    assert!(!result.is_ready_to_apply());
}

#[test]
fn local_mapping_pipeline_reports_candidate_failures_without_blocking_keyframe_stage() {
    let (map, selected_keyframe, tracking, _candidate) = map_keyframe_candidate();
    let invalid_candidate = LandmarkCandidate::new(100)
        .with_observation(LandmarkCandidateObservation::new(
            999,
            0,
            Point2::new(1.0, 2.0),
        ))
        .with_observation(LandmarkCandidateObservation::new(
            2,
            0,
            selected_keyframe.frame.keypoints[0],
        ));
    let mut pipeline = LocalMappingPipeline::default();

    let result = pipeline.process_keyframe(&map, &tracking, selected_keyframe, [invalid_candidate]);

    assert!(result.keyframe_decision.selected);
    assert_eq!(result.staged_update.keyframes.len(), 1);
    assert!(result.triangulated_landmarks.is_empty());
    assert_eq!(result.candidate_failures.len(), 1);
    assert!(matches!(
        result.candidate_failures[0].reason,
        LandmarkCandidateMappingFailureReason::CandidateValidationFailed(_)
    ));
    assert!(!result.is_ready_to_apply());
}

#[test]
fn local_mapping_pipeline_runs_custom_local_refiner_before_validation() {
    let (map, selected_keyframe, tracking, candidate) = map_keyframe_candidate();
    let mut pipeline = LocalMappingPipeline::with_refiner(
        SimpleKeyframePolicy::default(),
        LinearTriangulator::default(),
        CountingRefiner,
        Default::default(),
        Default::default(),
    );

    let result = pipeline.process_keyframe(&map, &tracking, selected_keyframe, [candidate]);

    assert!(result.refinement.refined);
    assert_eq!(result.refinement.reason, LocalRefinementReason::Refined);
    assert_eq!(result.refinement.keyframe_count, 1);
    assert_eq!(result.refinement.landmark_count, 1);
}

#[test]
fn local_mapping_pipeline_new_uses_noop_refiner() {
    let pipeline = LocalMappingPipeline::new(
        SimpleKeyframePolicy::default(),
        LinearTriangulator::default(),
        Default::default(),
        Default::default(),
    );

    assert_eq!(pipeline.local_refiner, NoopLocalRefiner);
}
