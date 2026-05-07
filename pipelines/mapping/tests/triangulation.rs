use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::{Camera, Frame, Keyframe, VisualMap};
use visloc_mapping::{
    LandmarkCandidate, LandmarkCandidateObservation, LandmarkCandidateValidationIssue,
    LinearTriangulator, TriangulationFailureReason, Triangulator,
};

fn pose_at_center(center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
}

fn keyframe(frame_id: u64, camera_id: u64, pose: Pose, keypoint: Point2<f64>) -> Keyframe {
    let mut frame = Frame::new(frame_id, camera_id);
    frame.pose = Some(pose);
    frame.keypoints.push(keypoint);
    Keyframe {
        frame,
        observations: Vec::new(),
    }
}

fn map_and_candidate(point: Point3<f64>) -> (VisualMap, LandmarkCandidate) {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
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
    map.keyframes.insert(1, keyframe(1, 1, pose_a, pixel_a));
    map.keyframes.insert(2, keyframe(2, 1, pose_b, pixel_b));

    let candidate = LandmarkCandidate::new(100)
        .with_observation(LandmarkCandidateObservation::new(1, 0, pixel_a))
        .with_observation(LandmarkCandidateObservation::new(2, 0, pixel_b))
        .with_descriptor(vec![0.1, 0.2]);

    (map, candidate)
}

#[test]
fn linear_triangulator_converts_candidate_to_landmark() {
    let expected = Point3::new(0.2, -0.1, 5.0);
    let (map, candidate) = map_and_candidate(expected);

    let triangulated = LinearTriangulator::default()
        .triangulate(&candidate, &map)
        .unwrap();

    assert_eq!(triangulated.landmark.id, 100);
    assert!((triangulated.landmark.position - expected).norm() < 1.0e-9);
    assert_eq!(triangulated.landmark.descriptor, Some(vec![0.1, 0.2]));
    assert_eq!(triangulated.landmark.observations.len(), 2);
    assert_eq!(triangulated.observation_count, 2);
    assert!(triangulated.mean_reprojection_error < 1.0e-9);
    assert!(triangulated.max_reprojection_error < 1.0e-9);
}

#[test]
fn triangulator_rejects_invalid_candidate() {
    let expected = Point3::new(0.0, 0.0, 5.0);
    let (map, candidate) = map_and_candidate(expected);
    let candidate = LandmarkCandidate::new(candidate.id)
        .with_observation(candidate.observations.first().unwrap().clone());

    let error = LinearTriangulator::default()
        .triangulate(&candidate, &map)
        .unwrap_err();

    let TriangulationFailureReason::CandidateValidationFailed(report) = error else {
        panic!("expected candidate validation failure");
    };
    assert!(report
        .issues
        .contains(&LandmarkCandidateValidationIssue::TooFewObservations {
            observation_count: 1,
            min_observations: 2,
        }));
}

#[test]
fn triangulator_reports_missing_keyframe_pose() {
    let expected = Point3::new(0.0, 0.0, 5.0);
    let (mut map, candidate) = map_and_candidate(expected);
    map.keyframes.get_mut(&2).unwrap().frame.pose = None;

    let error = LinearTriangulator::default()
        .triangulate(&candidate, &map)
        .unwrap_err();

    assert_eq!(
        error,
        TriangulationFailureReason::MissingPose { frame_id: 2 }
    );
}

#[test]
fn triangulator_reports_missing_camera() {
    let expected = Point3::new(0.0, 0.0, 5.0);
    let (mut map, candidate) = map_and_candidate(expected);
    map.cameras.clear();

    let error = LinearTriangulator::default()
        .triangulate(&candidate, &map)
        .unwrap_err();

    assert_eq!(
        error,
        TriangulationFailureReason::MissingCamera {
            frame_id: 1,
            camera_id: 1,
        }
    );
}
