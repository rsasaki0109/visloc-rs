use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::{
    Pose, PoseTrajectory, TrackingEvent, TrackingState, TrajectoryErrorSummary, TrajectorySample,
};

fn main() {
    let estimated = trajectory_from_centers(&[
        (0, Vector3::new(0.00, 0.00, 0.00)),
        (1, Vector3::new(1.05, 0.02, 0.00)),
        (2, Vector3::new(2.10, 0.04, 0.01)),
        (4, Vector3::new(4.30, 0.10, 0.00)),
    ]);
    let reference = trajectory_from_centers(&[
        (0, Vector3::new(0.0, 0.0, 0.0)),
        (1, Vector3::new(1.0, 0.0, 0.0)),
        (2, Vector3::new(2.0, 0.0, 0.0)),
        (3, Vector3::new(3.0, 0.0, 0.0)),
    ]);

    let errors = estimated.translation_errors_against(&reference);
    let summary = estimated.translation_error_summary_against(&reference);

    println!("per_frame_translation_errors:");
    for error in &errors {
        println!(
            "frame={} translation_error={:.6}",
            error.frame_id, error.translation_error
        );
    }

    print_summary(&summary);
    println!("summary_json:\n{}", summary.to_json());
}

fn trajectory_from_centers(centers: &[(u64, Vector3<f64>)]) -> PoseTrajectory {
    let mut trajectory = PoseTrajectory::new();
    for (frame_id, center) in centers {
        trajectory.push_sample(TrajectorySample {
            frame_id: *frame_id,
            pose: pose_with_identity_rotation_at_center(*center),
            state: TrackingState::Tracking,
            event: TrackingEvent::Tracked,
            inlier_count: 20,
            inlier_ratio: 1.0,
            reprojection_error: Some(0.2),
        });
    }
    trajectory
}

fn pose_with_identity_rotation_at_center(center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
}

fn print_summary(summary: &TrajectoryErrorSummary) {
    println!(
        "summary estimated={} reference={} matched={} missing_reference={} missing_estimate={} mean={:?} rmse={:?} max={:?}",
        summary.estimated_pose_count,
        summary.reference_pose_count,
        summary.matched_pose_count,
        summary.missing_reference_count,
        summary.missing_estimate_count,
        summary.mean_translation_error,
        summary.rmse_translation_error,
        summary.max_translation_error,
    );
}
