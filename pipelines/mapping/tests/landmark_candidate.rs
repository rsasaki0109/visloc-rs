use nalgebra::{Point2, Point3};
use visloc_core::types::{Camera, Frame, Keyframe, Landmark, VisualMap};
use visloc_mapping::{
    LandmarkCandidate, LandmarkCandidateObservation, LandmarkCandidateValidationConfig,
    LandmarkCandidateValidationIssue, LocalMapWindow,
};

fn keyframe(frame_id: u64, keypoint_count: usize) -> Keyframe {
    let mut frame = Frame::new(frame_id, 1);
    for index in 0..keypoint_count {
        frame
            .keypoints
            .push(Point2::new(100.0 + index as f64, 120.0));
    }
    Keyframe {
        frame,
        observations: Vec::new(),
    }
}

fn map_with_keyframes() -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras
        .insert(1, Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
    map.landmarks
        .insert(100, Landmark::new(100, Point3::new(0.0, 0.0, 5.0)));
    map.keyframes.insert(1, keyframe(1, 2));
    map.keyframes.insert(2, keyframe(2, 2));
    map.keyframes.insert(3, keyframe(3, 2));
    map
}

fn candidate_observation(frame_id: u64, keypoint_index: usize) -> LandmarkCandidateObservation {
    LandmarkCandidateObservation::new(
        frame_id,
        keypoint_index,
        Point2::new(100.0 + keypoint_index as f64, 120.0),
    )
}

#[test]
fn valid_landmark_candidate_requires_two_observations() {
    let map = map_with_keyframes();
    let candidate = LandmarkCandidate::new(7)
        .with_observation(candidate_observation(1, 0))
        .with_observation(candidate_observation(2, 1))
        .with_descriptor(vec![0.1, 0.2, 0.3]);

    let report =
        candidate.validate_against(&map, None, &LandmarkCandidateValidationConfig::default());

    assert!(report.is_valid());
    assert_eq!(candidate.observation_count(), 2);
    assert!(candidate.is_triangulatable(2));
    assert!(candidate.descriptor.is_some());
}

#[test]
fn reports_too_few_observations() {
    let map = map_with_keyframes();
    let candidate = LandmarkCandidate::new(7).with_observation(candidate_observation(1, 0));

    let report =
        candidate.validate_against(&map, None, &LandmarkCandidateValidationConfig::default());

    assert!(!report.is_valid());
    assert_eq!(report.issue_count(), 1);
    assert!(report
        .issues
        .contains(&LandmarkCandidateValidationIssue::TooFewObservations {
            observation_count: 1,
            min_observations: 2,
        }));
    assert!(!candidate.is_triangulatable(2));
}

#[test]
fn reports_missing_keyframe_and_keypoint_bounds() {
    let map = map_with_keyframes();
    let candidate = LandmarkCandidate::new(7)
        .with_observation(candidate_observation(1, 99))
        .with_observation(candidate_observation(999, 0));

    let report =
        candidate.validate_against(&map, None, &LandmarkCandidateValidationConfig::default());

    assert!(report
        .issues
        .contains(&LandmarkCandidateValidationIssue::KeypointOutOfBounds {
            frame_id: 1,
            keypoint_index: 99,
            keypoint_count: 2,
        }));
    assert!(report
        .issues
        .contains(&LandmarkCandidateValidationIssue::MissingKeyframe { frame_id: 999 }));
}

#[test]
fn reports_duplicate_observations() {
    let map = map_with_keyframes();
    let candidate = LandmarkCandidate::new(7)
        .with_observation(candidate_observation(1, 0))
        .with_observation(candidate_observation(1, 0));

    let report =
        candidate.validate_against(&map, None, &LandmarkCandidateValidationConfig::default());

    assert!(report
        .issues
        .contains(&LandmarkCandidateValidationIssue::DuplicateObservation {
            frame_id: 1,
            keypoint_index: 0,
        }));
}

#[test]
fn reports_observations_outside_local_window() {
    let map = map_with_keyframes();
    let window = LocalMapWindow::from_keyframe_ids(&map, Some(2), vec![1, 2]);
    let candidate = LandmarkCandidate::new(7)
        .with_observation(candidate_observation(2, 0))
        .with_observation(candidate_observation(3, 0));

    let report = candidate.validate_against(
        &map,
        Some(&window),
        &LandmarkCandidateValidationConfig::default(),
    );

    assert!(report.issues.contains(
        &LandmarkCandidateValidationIssue::ObservationOutsideLocalWindow { frame_id: 3 }
    ));
}
