use nalgebra::{UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::{LocalizationFailureReason, LocalizationResult, LocalizationSuccess};
use visloc_mapping::{
    KeyframeDecisionReason, KeyframePolicy, KeyframePolicyConfig, SimpleKeyframePolicy,
};
use visloc_tracking::{TrackingEvent, TrackingResult, TrackingState};

fn pose_at(center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
}

fn success_result(frame_id: u64, event: TrackingEvent, center: Vector3<f64>) -> TrackingResult {
    success_result_with_inliers(frame_id, event, center, 0)
}

fn success_result_with_inliers(
    frame_id: u64,
    event: TrackingEvent,
    center: Vector3<f64>,
    inlier_count: usize,
) -> TrackingResult {
    let inliers: Vec<usize> = (0..inlier_count).collect();
    let inlier_landmark_ids: Vec<u64> = (0..inlier_count).map(|idx| idx as u64 + 1).collect();
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event,
        successive_failures: 0,
        pose_prior: None,
        used_pose_prior: false,
        used_external_localization_prior: false,
        external_localization_prior_radius: None,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: Default::default(),
        localization: LocalizationResult::success(LocalizationSuccess {
            pose: pose_at(center),
            candidate_landmark_count: inlier_count,
            match_count: inlier_count,
            correspondence_count: inlier_count,
            inliers: inliers.clone(),
            inlier_query_indices: inliers,
            inlier_landmark_ids,
            inlier_reprojection_errors: vec![0.0; inlier_count],
            mean_reprojection_error: 0.0,
            median_reprojection_error: 0.0,
            max_reprojection_error: 0.0,
        }),
        covisibility_local_map_size: None,
    }
}

fn failed_result(frame_id: u64) -> TrackingResult {
    TrackingResult {
        frame_id,
        state: TrackingState::Lost,
        event: TrackingEvent::Lost,
        successive_failures: 1,
        pose_prior: None,
        used_pose_prior: false,
        used_external_localization_prior: false,
        external_localization_prior_radius: None,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: Default::default(),
        localization: LocalizationResult::failure(
            LocalizationFailureReason::NoDescriptorMatches,
            0,
            0,
            0,
        ),
        covisibility_local_map_size: None,
    }
}

#[test]
fn selects_first_successful_frame() {
    let mut policy = SimpleKeyframePolicy::default();

    let decision = policy.evaluate(&success_result(
        10,
        TrackingEvent::Initialized,
        Vector3::zeros(),
    ));

    assert!(decision.selected);
    assert_eq!(
        decision.reason,
        KeyframeDecisionReason::FirstSuccessfulFrame
    );
    assert_eq!(decision.last_keyframe_frame_id, Some(10));
    assert_eq!(decision.selected_keyframe_count, 1);
    assert_eq!(policy.last_keyframe_frame_id(), Some(10));
}

#[test]
fn rejects_failed_tracking_result() {
    let mut policy = SimpleKeyframePolicy::default();

    let decision = policy.evaluate(&failed_result(11));

    assert!(!decision.selected);
    assert_eq!(decision.reason, KeyframeDecisionReason::NotLocalized);
    assert_eq!(policy.selected_keyframe_count(), 0);
}

#[test]
fn applies_frame_gap_and_translation_thresholds() {
    let mut policy = SimpleKeyframePolicy::new(KeyframePolicyConfig {
        min_frame_id_gap: 3,
        min_translation: 1.0,
        select_relocalized_frames: true,
        ..KeyframePolicyConfig::default()
    });

    assert!(
        policy
            .evaluate(&success_result(
                10,
                TrackingEvent::Initialized,
                Vector3::zeros()
            ))
            .selected
    );

    let gap_rejection = policy.evaluate(&success_result(
        12,
        TrackingEvent::Tracked,
        Vector3::new(3.0, 0.0, 0.0),
    ));
    assert!(!gap_rejection.selected);
    assert_eq!(
        gap_rejection.reason,
        KeyframeDecisionReason::FrameIdGapTooSmall {
            frame_id_gap: 2,
            min_frame_id_gap: 3,
        }
    );

    let translation_rejection = policy.evaluate(&success_result(
        13,
        TrackingEvent::Tracked,
        Vector3::new(0.5, 0.0, 0.0),
    ));
    assert!(!translation_rejection.selected);
    assert_eq!(
        translation_rejection.reason,
        KeyframeDecisionReason::TranslationTooSmall {
            translation: 0.5,
            min_translation: 1.0,
        }
    );

    let selected = policy.evaluate(&success_result(
        13,
        TrackingEvent::Tracked,
        Vector3::new(2.0, 0.0, 0.0),
    ));
    assert!(selected.selected);
    assert_eq!(
        selected.reason,
        KeyframeDecisionReason::ThresholdsMet {
            frame_id_gap: 3,
            translation: 2.0,
        }
    );
    assert_eq!(policy.selected_keyframe_count(), 2);
}

#[test]
fn can_select_relocalized_frame_immediately() {
    let mut policy = SimpleKeyframePolicy::new(KeyframePolicyConfig {
        min_frame_id_gap: 100,
        min_translation: 100.0,
        select_relocalized_frames: true,
        ..KeyframePolicyConfig::default()
    });

    assert!(
        policy
            .evaluate(&success_result(
                10,
                TrackingEvent::Initialized,
                Vector3::zeros()
            ))
            .selected
    );
    let relocalized = policy.evaluate(&success_result(
        11,
        TrackingEvent::Relocalized,
        Vector3::new(0.1, 0.0, 0.0),
    ));

    assert!(relocalized.selected);
    assert_eq!(relocalized.reason, KeyframeDecisionReason::Relocalized);
    assert_eq!(policy.last_keyframe_frame_id(), Some(11));
}

#[test]
fn reset_clears_keyframe_policy_state() {
    let mut policy = SimpleKeyframePolicy::default();

    assert!(
        policy
            .evaluate(&success_result(
                10,
                TrackingEvent::Initialized,
                Vector3::zeros()
            ))
            .selected
    );
    policy.reset();

    assert_eq!(policy.last_keyframe_frame_id(), None);
    assert!(policy.last_keyframe_pose().is_none());
    assert_eq!(policy.last_keyframe_tracked_landmark_count(), None);
    assert_eq!(policy.selected_keyframe_count(), 0);
}

#[test]
fn can_select_when_tracked_landmarks_drop_after_frame_gap() {
    let mut policy = SimpleKeyframePolicy::new(KeyframePolicyConfig {
        min_frame_id_gap: 3,
        min_translation: 10.0,
        select_relocalized_frames: true,
        tracked_landmark_keyframe_ratio: Some(0.9),
        min_tracked_landmarks_for_quality_keyframe: 20,
    });

    assert!(
        policy
            .evaluate(&success_result_with_inliers(
                10,
                TrackingEvent::Initialized,
                Vector3::zeros(),
                100
            ))
            .selected
    );
    assert_eq!(policy.last_keyframe_tracked_landmark_count(), Some(100));

    let gap_rejection = policy.evaluate(&success_result_with_inliers(
        12,
        TrackingEvent::Tracked,
        Vector3::new(0.1, 0.0, 0.0),
        50,
    ));
    assert!(!gap_rejection.selected);
    assert_eq!(
        gap_rejection.reason,
        KeyframeDecisionReason::FrameIdGapTooSmall {
            frame_id_gap: 2,
            min_frame_id_gap: 3,
        }
    );

    let selected = policy.evaluate(&success_result_with_inliers(
        13,
        TrackingEvent::Tracked,
        Vector3::new(0.1, 0.0, 0.0),
        80,
    ));

    assert!(selected.selected);
    assert_eq!(
        selected.reason,
        KeyframeDecisionReason::TrackedLandmarkDrop {
            frame_id_gap: 3,
            tracked_landmarks: 80,
            last_keyframe_tracked_landmarks: 100,
            min_tracked_landmark_ratio: 0.9,
        }
    );
    assert_eq!(policy.last_keyframe_tracked_landmark_count(), Some(80));
    assert_eq!(policy.selected_keyframe_count(), 2);
}

#[test]
fn tracked_landmark_drop_respects_reference_count_floor() {
    let mut policy = SimpleKeyframePolicy::new(KeyframePolicyConfig {
        min_frame_id_gap: 1,
        min_translation: 10.0,
        select_relocalized_frames: true,
        tracked_landmark_keyframe_ratio: Some(0.9),
        min_tracked_landmarks_for_quality_keyframe: 20,
    });

    assert!(
        policy
            .evaluate(&success_result_with_inliers(
                10,
                TrackingEvent::Initialized,
                Vector3::zeros(),
                10
            ))
            .selected
    );

    let rejected = policy.evaluate(&success_result_with_inliers(
        11,
        TrackingEvent::Tracked,
        Vector3::new(0.1, 0.0, 0.0),
        5,
    ));

    assert!(!rejected.selected);
    assert_eq!(
        rejected.reason,
        KeyframeDecisionReason::TranslationTooSmall {
            translation: 0.1,
            min_translation: 10.0,
        }
    );
}

#[test]
fn tracked_landmark_drop_respects_current_count_floor() {
    let mut policy = SimpleKeyframePolicy::new(KeyframePolicyConfig {
        min_frame_id_gap: 1,
        min_translation: 10.0,
        select_relocalized_frames: true,
        tracked_landmark_keyframe_ratio: Some(0.9),
        min_tracked_landmarks_for_quality_keyframe: 20,
    });

    assert!(
        policy
            .evaluate(&success_result_with_inliers(
                10,
                TrackingEvent::Initialized,
                Vector3::zeros(),
                100
            ))
            .selected
    );

    let rejected = policy.evaluate(&success_result_with_inliers(
        11,
        TrackingEvent::Tracked,
        Vector3::new(0.1, 0.0, 0.0),
        5,
    ));

    assert!(!rejected.selected);
    assert_eq!(
        rejected.reason,
        KeyframeDecisionReason::TranslationTooSmall {
            translation: 0.1,
            min_translation: 10.0,
        }
    );
    assert_eq!(policy.last_keyframe_tracked_landmark_count(), Some(100));
}
