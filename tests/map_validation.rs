use nalgebra::{Matrix3, Point2, Point3};
use visloc_rs::core::geometry::SE3;
use visloc_rs::core::types::{
    Camera, Frame, Keyframe, Landmark, LandmarkDescriptorStore, Observation, StereoObservation,
    VisualMap, VisualMapValidationIssue,
};

fn valid_map() -> VisualMap {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let mut frame = Frame::new(10, camera.id);
    frame.keypoints.push(Point2::new(320.0, 240.0));

    let observation = Observation {
        frame_id: frame.id,
        landmark_id: 100,
        keypoint_index: 0,
        xy: frame.keypoints[0],
    };

    let mut landmark = Landmark::new(100, Point3::new(0.0, 0.0, 5.0));
    landmark.descriptor = Some(vec![1.0, 2.0]);
    landmark.observations.push(observation.clone());

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera);
    map.landmarks.insert(landmark.id, landmark);
    map.keyframes.insert(
        frame.id,
        Keyframe {
            frame,
            observations: vec![observation],
        },
    );
    map
}

#[test]
fn visual_map_validation_accepts_consistent_map() {
    let map = valid_map();

    let structure_report = map.validate();
    let descriptor_report = map.validate_with_descriptors(None);

    assert!(structure_report.is_valid());
    assert_eq!(structure_report.issue_count(), 0);
    assert!(structure_report.into_result().is_ok());
    assert!(descriptor_report.is_valid());
}

#[test]
fn visual_map_validation_reports_broken_references() {
    let mut map = VisualMap::new();
    let mut frame = Frame::new(10, 999);
    frame.keypoints.push(Point2::new(1.0, 2.0));
    let observation = Observation {
        frame_id: 11,
        landmark_id: 200,
        keypoint_index: 4,
        xy: Point2::new(1.0, 2.0),
    };
    map.keyframes.insert(
        99,
        Keyframe {
            frame,
            observations: vec![observation],
        },
    );

    let mut landmark = Landmark::new(100, Point3::new(0.0, 0.0, 5.0));
    landmark.observations.push(Observation {
        frame_id: 404,
        landmark_id: 100,
        keypoint_index: 0,
        xy: Point2::new(0.0, 0.0),
    });
    map.landmarks.insert(landmark.id, landmark);

    let report = map.validate();

    assert!(!report.is_valid());
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::KeyframeIdMismatch {
            keyframe_id: 99,
            frame_id: 10,
        }));
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::MissingCameraForKeyframe {
            frame_id: 10,
            camera_id: 999,
        }));
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::ObservationFrameMismatch {
            expected_frame_id: 10,
            actual_frame_id: 11,
            landmark_id: 200,
            keypoint_index: 4,
        }));
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::ObservationMissingLandmark {
            frame_id: 10,
            landmark_id: 200,
            keypoint_index: 4,
        }));
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::ObservationKeypointOutOfBounds {
            frame_id: 10,
            landmark_id: 200,
            keypoint_index: 4,
            keypoint_count: 1,
        }));
    assert!(report.issues.contains(
        &VisualMapValidationIssue::LandmarkObservationMissingKeyframe {
            landmark_id: 100,
            frame_id: 404,
        }
    ));
}

#[test]
fn visual_map_validation_reports_descriptor_gaps() {
    let mut map = valid_map();
    map.landmarks
        .insert(200, Landmark::new(200, Point3::new(1.0, 0.0, 5.0)));

    let embedded_report = map.validate_with_descriptors(None);

    assert!(embedded_report
        .issues
        .contains(&VisualMapValidationIssue::MissingDescriptorForLandmark { landmark_id: 200 }));

    let mut descriptor_store = LandmarkDescriptorStore::new();
    descriptor_store.insert(100, vec![1.0, 2.0]);
    descriptor_store.insert(999, vec![9.0, 9.0]);

    let external_report = map.validate_with_descriptors(Some(&descriptor_store));

    assert!(external_report
        .issues
        .contains(&VisualMapValidationIssue::MissingDescriptorForLandmark { landmark_id: 200 }));
    assert!(external_report
        .issues
        .contains(&VisualMapValidationIssue::DescriptorForMissingLandmark { landmark_id: 999 }));
}

#[test]
fn visual_map_validation_checks_landmark_covariance_sidecar() {
    let mut map = valid_map();
    map.landmark_position_covariances
        .insert(100, Matrix3::from_diagonal_element(0.01));
    assert!(map.validate().is_valid());

    map.landmark_position_covariances
        .insert(999, Matrix3::identity());
    map.landmark_position_covariances.insert(
        100,
        Matrix3::new(1.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0),
    );
    let report = map.validate();
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::CovarianceForMissingLandmark { landmark_id: 999 }));
    assert!(report
        .issues
        .contains(&VisualMapValidationIssue::InvalidLandmarkCovariance { landmark_id: 100 }));
}

#[test]
fn visual_map_validation_checks_stereo_observation_sidecar() {
    let mut map = valid_map();
    map.cameras
        .insert(2, Camera::pinhole(2, 640, 480, 505.0, 498.0, 318.0, 241.0));
    map.stereo_observations.push(StereoObservation {
        frame_id: 10,
        landmark_id: 100,
        right_camera_id: 2,
        xy_right: Point2::new(305.0, 240.5),
        left_to_right: SE3::identity(),
    });
    assert!(map.validate().is_valid());

    map.stereo_observations.push(StereoObservation {
        frame_id: 404,
        landmark_id: 999,
        right_camera_id: 77,
        xy_right: Point2::new(1.0, 2.0),
        left_to_right: SE3::identity(),
    });
    let report = map.validate();
    assert!(report.issues.contains(
        &VisualMapValidationIssue::StereoObservationMissingKeyframe {
            frame_id: 404,
            landmark_id: 999,
        }
    ));
    assert!(report.issues.contains(
        &VisualMapValidationIssue::StereoObservationMissingLandmark {
            frame_id: 404,
            landmark_id: 999,
        }
    ));
    assert!(report.issues.contains(
        &VisualMapValidationIssue::StereoObservationMissingRightCamera {
            frame_id: 404,
            landmark_id: 999,
            camera_id: 77,
        }
    ));
}
