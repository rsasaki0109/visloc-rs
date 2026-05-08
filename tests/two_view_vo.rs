use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::{
    parse_two_view_matches_txt, Frame, Pose, TwoViewMatchVisualOdometryConfig,
    TwoViewMatchVisualOdometryFrontend, VisualOdometryFrontend, VisualOdometryPriorProvider,
};

#[test]
fn two_view_match_frontend_estimates_translation_prior_from_inlier_flow() {
    let matches = parse_two_view_matches_txt(
        r#"
        0 0 100.0 120.0 70.0 121.0 0.99
        1 1 200.0 130.0 169.5 131.0 0.98
        2 2 300.0 140.0 270.5 141.2 0.97
        3 3 400.0 150.0 370.0 151.1 0.96
        4 4 120.0 220.0 90.2 221.0 0.95
        5 5 220.0 230.0 190.1 231.2 0.94
        6 6 320.0 240.0 520.0 40.0 0.10
        "#,
    )
    .unwrap();
    let previous = Frame::new(10, 1);
    let current = Frame::new(11, 1);
    let frontend = TwoViewMatchVisualOdometryFrontend::new(TwoViewMatchVisualOdometryConfig {
        min_matches: 6,
        min_inliers: 5,
        max_residual_pixels: 2.0,
        pixel_translation_scale: 0.01,
        forward_translation: 0.2,
    })
    .with_matches(previous.id, current.id, matches);

    let estimate = frontend
        .estimate_relative_pose(&previous, &current)
        .unwrap()
        .unwrap();

    assert_eq!(estimate.match_count, 7);
    assert_eq!(estimate.inlier_count, 6);
    assert!((estimate.previous_to_current.translation.x + 0.2995).abs() < 1.0e-9);
    assert!((estimate.previous_to_current.translation.y - 0.010833333333333334).abs() < 1.0e-9);
    assert!((estimate.previous_to_current.translation.z - 0.2).abs() < 1.0e-9);
    assert!(estimate.mean_reprojection_error.unwrap() < 0.6);
}

#[test]
fn two_view_match_frontend_returns_none_without_enough_inliers() {
    let matches = parse_two_view_matches_txt(
        r#"
        0 0 100.0 120.0 70.0 121.0
        1 1 200.0 130.0 500.0 80.0
        2 2 300.0 140.0 10.0 441.0
        "#,
    )
    .unwrap();
    let previous = Frame::new(10, 1);
    let current = Frame::new(11, 1);
    let frontend = TwoViewMatchVisualOdometryFrontend::new(TwoViewMatchVisualOdometryConfig {
        min_matches: 3,
        min_inliers: 3,
        max_residual_pixels: 2.0,
        pixel_translation_scale: 0.01,
        forward_translation: 0.0,
    })
    .with_matches(previous.id, current.id, matches);

    assert!(frontend
        .estimate_relative_pose(&previous, &current)
        .unwrap()
        .is_none());
}

#[test]
fn two_view_match_prior_provider_converts_estimate_to_pose_prior() {
    let matches = parse_two_view_matches_txt(
        r#"
        0 0 100.0 120.0 55.0 120.0
        1 1 200.0 130.0 155.0 130.0
        2 2 300.0 140.0 255.0 140.0
        3 3 400.0 150.0 355.0 150.0
        4 4 120.0 220.0 75.0 220.0
        5 5 220.0 230.0 175.0 230.0
        "#,
    )
    .unwrap();
    let previous = Frame::new(10, 1);
    let current = Frame::new(11, 1);
    let provider = VisualOdometryPriorProvider::new(
        TwoViewMatchVisualOdometryFrontend::new(TwoViewMatchVisualOdometryConfig {
            min_matches: 6,
            min_inliers: 6,
            max_residual_pixels: 1.0,
            pixel_translation_scale: 0.01,
            forward_translation: 0.0,
        })
        .with_matches(previous.id, current.id, matches),
    );
    let previous_pose =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));

    let prior = provider
        .predict_pose_prior(&previous, &previous_pose, &current)
        .unwrap()
        .unwrap();

    assert!((prior.pose.camera_center_world() - Point3::new(0.45, 0.0, 0.0)).norm() < 1.0e-9);
}
