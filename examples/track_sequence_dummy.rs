use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{Camera, Frame, Landmark, VisualMap};
use visloc_rs::{LocalizationPipeline, PoseTrajectory, Tracker, TrackingConfig};

fn main() {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = [
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());

    let mut good_frame_a = Frame::new(100, camera.id);
    let mut good_frame_b = Frame::new(101, camera.id);
    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);

        let keypoint = camera
            .project(&pose.transform_world_point(point))
            .expect("dummy point must be in front of the camera");
        good_frame_a.keypoints.push(keypoint);
        good_frame_a.descriptors.push(descriptor.clone());
        good_frame_b.keypoints.push(keypoint);
        good_frame_b.descriptors.push(descriptor);
    }

    let mut bad_frame_a = good_frame_a.clone();
    bad_frame_a.id = 102;
    bad_frame_a.camera_id = 999;
    let mut bad_frame_b = bad_frame_a.clone();
    bad_frame_b.id = 103;
    let mut relocalize_frame = good_frame_a.clone();
    relocalize_frame.id = 104;

    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
            ..TrackingConfig::default()
        },
    );

    let frames = [
        good_frame_a,
        good_frame_b,
        bad_frame_a,
        bad_frame_b,
        relocalize_frame,
    ];

    let results = tracker.track_frames(&frames, &map);
    for tracking in &results {
        println!(
            "frame={} state={:?} event={:?} success={} failures={} prior={} reason={:?} map_landmarks={} descriptors={} inliers={} ratio={:.3}",
            tracking.frame_id,
            tracking.state,
            tracking.event,
            tracking.localization.success,
            tracking.successive_failures,
            tracking.used_pose_prior,
            tracking.tracking_failure_reason,
            tracking.map_landmark_count,
            tracking.map_stats.descriptor_count,
            tracking.localization.inlier_count,
            tracking.localization.inlier_ratio,
        );
    }

    let trajectory = PoseTrajectory::from_tracking_results(&results);
    println!(
        "trajectory poses={} path_length={:.3} mean_reprojection_error={:?}",
        trajectory.len(),
        trajectory.total_path_length(),
        trajectory.mean_reprojection_error(),
    );
    println!("trajectory_csv:\n{}", trajectory.to_csv());
    println!("trajectory_kitti_poses:\n{}", trajectory.to_kitti_poses());

    let stats = tracker.stats();
    println!(
        "stats first={:?} last={:?} frames={} ok={} failed={} success_rate={:.3} prior_rate={:.3} inlier_ratio={:.3} mean_inliers={:.1} lost={} relocalized={} quality_gate_failures={}",
        stats.first_frame_id,
        stats.last_frame_id,
        stats.frame_count,
        stats.successful_frame_count,
        stats.failed_frame_count,
        stats.success_rate(),
        stats.pose_prior_usage_rate(),
        stats.overall_inlier_ratio(),
        stats.mean_inliers_per_successful_frame(),
        stats.lost_count,
        stats.relocalization_count,
        stats.tracking_quality_gate_failure_count,
    );
}
