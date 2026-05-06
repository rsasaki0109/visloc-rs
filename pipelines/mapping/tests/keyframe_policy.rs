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
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event,
        successive_failures: 0,
        pose_prior: None,
        used_pose_prior: false,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: Default::default(),
        localization: LocalizationResult::success(LocalizationSuccess {
            pose: pose_at(center),
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

fn failed_result(frame_id: u64) -> TrackingResult {
    TrackingResult {
        frame_id,
        state: TrackingState::Lost,
        event: TrackingEvent::Lost,
        successive_failures: 1,
        pose_prior: None,
        used_pose_prior: false,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: Default::default(),
        localization: LocalizationResult::failure(
            LocalizationFailureReason::NoDescriptorMatches,
            0,
            0,
            0,
        ),
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
    assert_eq!(policy.selected_keyframe_count(), 0);
}
