use nalgebra::{Point2, UnitQuaternion, Vector3};
use visloc_rs::{
    parse_two_view_matches_txt, Frame, Pose, TwoViewMatchVisualOdometryConfig,
    TwoViewMatchVisualOdometryFrontend, VisualOdometryPriorProvider,
};

fn main() {
    let matches = parse_two_view_matches_txt(
        r#"
        # PREV_IDX CURR_IDX PREV_X PREV_Y CURR_X CURR_Y SCORE
        0 0 120.0 140.0 75.0 140.2 0.99
        1 1 260.0 180.0 215.2 180.1 0.98
        2 2 410.0 220.0 365.1 220.3 0.97
        3 3 180.0 300.0 135.0 300.1 0.96
        4 4 320.0 260.0 275.2 260.2 0.95
        5 5 480.0 210.0 435.1 210.0 0.94
        6 6 100.0 100.0 220.0 320.0 0.10
        "#,
    )
    .expect("dummy two-view matches must parse");
    let previous_frame = frame_with_keypoints(10, &matches.matched_previous_keypoints());
    let current_frame = frame_with_keypoints(11, &matches.matched_current_keypoints());
    let frontend = TwoViewMatchVisualOdometryFrontend::new(TwoViewMatchVisualOdometryConfig {
        min_matches: 6,
        min_inliers: 5,
        max_residual_pixels: 2.0,
        pixel_translation_scale: 0.01,
        forward_translation: 0.0,
    })
    .with_matches(previous_frame.id, current_frame.id, matches);
    let provider = VisualOdometryPriorProvider::new(frontend);
    let previous_pose =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
    let prior = provider
        .predict_pose_prior(&previous_frame, &previous_pose, &current_frame)
        .expect("two-view match frontend is infallible")
        .expect("dummy matches should produce a VO prior");
    let center = prior.pose.camera_center_world();

    println!(
        "vo_prior previous={} current={} matches={} inliers={} mean_flow_residual_px={:?}",
        prior.estimate.previous_frame_id,
        prior.estimate.current_frame_id,
        prior.estimate.match_count,
        prior.estimate.inlier_count,
        prior.estimate.mean_reprojection_error
    );
    println!(
        "previous_to_current_translation=[{:.3}, {:.3}, {:.3}]",
        prior.estimate.previous_to_current.translation.x,
        prior.estimate.previous_to_current.translation.y,
        prior.estimate.previous_to_current.translation.z
    );
    println!(
        "predicted_camera_center_world=[{:.3}, {:.3}, {:.3}]",
        center.x, center.y, center.z
    );
}

fn frame_with_keypoints(frame_id: u64, keypoints: &[Point2<f64>]) -> Frame {
    let mut frame = Frame::new(frame_id, 1);
    frame.keypoints = keypoints.to_vec();
    frame
}
