use nalgebra::{Point2, Point3};
use visloc_core::types::{Camera, Frame, Keyframe, Landmark, Observation, VisualMap};
use visloc_mapping::{MapUpdateValidationIssue, StagedMapUpdate};

fn base_map() -> VisualMap {
    let mut map = VisualMap::new();
    map.cameras
        .insert(1, Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0));
    map
}

fn keyframe(frame_id: u64, camera_id: u64) -> Keyframe {
    let mut frame = Frame::new(frame_id, camera_id);
    frame.keypoints.push(Point2::new(100.0, 120.0));
    Keyframe {
        frame,
        observations: Vec::new(),
    }
}

fn landmark(landmark_id: u64) -> Landmark {
    Landmark::new(landmark_id, Point3::new(0.0, 0.0, 5.0))
}

fn observation(frame_id: u64, landmark_id: u64, keypoint_index: usize) -> Observation {
    Observation {
        frame_id,
        landmark_id,
        keypoint_index,
        xy: Point2::new(100.0, 120.0),
    }
}

#[test]
fn validates_and_applies_staged_keyframe_landmark_and_observation() {
    let mut map = base_map();
    let update = StagedMapUpdate::new()
        .with_keyframe(keyframe(10, 1))
        .with_landmark(landmark(100))
        .with_observation(observation(10, 100, 0));

    let report = update.validate_against(&map);
    assert!(report.is_valid());

    let applied = update.apply_to(&mut map).unwrap();

    assert_eq!(applied.keyframe_count, 1);
    assert_eq!(applied.landmark_count, 1);
    assert_eq!(applied.observation_count, 1);
    assert!(map.keyframes.contains_key(&10));
    assert!(map.landmarks.contains_key(&100));
    assert_eq!(map.keyframes.get(&10).unwrap().observations.len(), 1);
    assert_eq!(map.landmarks.get(&100).unwrap().observations.len(), 1);
    assert!(map.validate().is_valid());
}

#[test]
fn embedded_keyframe_observation_is_mirrored_to_landmark_without_duplicates() {
    let mut map = base_map();
    map.landmarks.insert(100, landmark(100));
    let obs = observation(10, 100, 0);
    let mut staged_keyframe = keyframe(10, 1);
    staged_keyframe.observations.push(obs.clone());
    let update = StagedMapUpdate::new()
        .with_keyframe(staged_keyframe)
        // Some producers stage the same relation explicitly; application must
        // keep both VisualMap indices set-like.
        .with_observation(obs);

    assert!(update.validate_against(&map).is_valid());
    update.apply_to(&mut map).unwrap();

    assert_eq!(map.keyframes[&10].observations.len(), 1);
    assert_eq!(map.landmarks[&100].observations.len(), 1);
    assert!(map.validate().is_valid());
}

#[test]
fn reports_duplicate_staged_entities_and_existing_map_conflicts() {
    let mut map = base_map();
    map.keyframes.insert(10, keyframe(10, 1));
    map.landmarks.insert(100, landmark(100));
    let update = StagedMapUpdate::new()
        .with_keyframe(keyframe(10, 1))
        .with_keyframe(keyframe(10, 1))
        .with_landmark(landmark(100))
        .with_landmark(landmark(100));

    let report = update.validate_against(&map);

    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::KeyframeAlreadyExists { frame_id: 10 }));
    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::DuplicateStagedKeyframe { frame_id: 10 }));
    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::LandmarkAlreadyExists { landmark_id: 100 }));
    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::DuplicateStagedLandmark { landmark_id: 100 }));
}

#[test]
fn rejects_observations_without_referenced_entities() {
    let map = base_map();
    let update = StagedMapUpdate::new().with_observation(observation(10, 100, 0));

    let report = update.validate_against(&map);

    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::ObservationMissingKeyframe {
            frame_id: 10,
            landmark_id: 100,
            keypoint_index: 0,
        }));
    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::ObservationMissingLandmark {
            frame_id: 10,
            landmark_id: 100,
            keypoint_index: 0,
        }));
}

#[test]
fn rejects_observations_outside_keypoint_bounds_and_duplicates() {
    let map = base_map();
    let update = StagedMapUpdate::new()
        .with_keyframe(keyframe(10, 1))
        .with_landmark(landmark(100))
        .with_observation(observation(10, 100, 1))
        .with_observation(observation(10, 100, 1));

    let report = update.validate_against(&map);

    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::ObservationKeypointOutOfBounds {
            frame_id: 10,
            landmark_id: 100,
            keypoint_index: 1,
            keypoint_count: 1,
        }));
    assert!(report
        .issues
        .contains(&MapUpdateValidationIssue::DuplicateStagedObservation {
            frame_id: 10,
            landmark_id: 100,
            keypoint_index: 1,
        }));
}

#[test]
fn apply_rejects_invalid_update_without_mutating_map() {
    let mut map = base_map();
    let update = StagedMapUpdate::new().with_keyframe(keyframe(10, 999));

    let error = update.apply_to(&mut map).unwrap_err();

    assert_eq!(error.issue_count(), 1);
    assert!(error
        .issues
        .contains(&MapUpdateValidationIssue::MissingCameraForKeyframe {
            frame_id: 10,
            camera_id: 999,
        }));
    assert!(!map.keyframes.contains_key(&10));
}
