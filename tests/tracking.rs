#![allow(clippy::useless_vec)]

use std::{convert::Infallible, fs};

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::{
    Camera, Frame, Landmark, LandmarkDescriptorStore, LocalizationFailureReason,
    LocalizationResult, LocalizationSuccess, VisualMap,
};
use visloc_rs::{
    tracking_results_to_csv, tracking_results_to_html_report, write_tracking_results_csv,
    write_tracking_results_html_report, ConstantVelocityMotionModel, FeatureExtractor, FeatureSet,
    FrameLocalizer, ImageTracker, InMemoryMapProvider, LocalizationPipeline, LocalizationPrior,
    MapProviderStats, MotionModel, PoseTrajectory, PriorSubmapSelector, SelectableMapProvider,
    Tracker, TrackingConfig, TrackingEvent, TrackingFailureReason, TrackingResult, TrackingState,
    TrackingStats, TrajectoryAlignment,
};

#[derive(Debug, Clone)]
struct FixedPoseMotionModel {
    pose: Pose,
}

impl MotionModel for FixedPoseMotionModel {
    fn predict_pose(
        &self,
        _frame: &Frame,
        _last_result: Option<&TrackingResult>,
        _last_successful_pose: Option<&Pose>,
    ) -> Option<Pose> {
        Some(self.pose.clone())
    }
}

#[derive(Debug, Clone)]
struct StaticFeatureExtractor {
    features: FeatureSet,
}

#[derive(Debug, Clone)]
struct FixedFrameLocalizer {
    result: LocalizationResult,
}

impl FrameLocalizer for FixedFrameLocalizer {
    fn localize_frame_with_descriptor_store(
        &self,
        _frame: &Frame,
        _map: &VisualMap,
        _descriptor_store: &LandmarkDescriptorStore,
    ) -> LocalizationResult {
        self.result.clone()
    }
}

impl FeatureExtractor for StaticFeatureExtractor {
    type Image = ();
    type Error = Infallible;

    fn extract(&self, _image: &Self::Image) -> Result<FeatureSet, Self::Error> {
        Ok(self.features.clone())
    }
}

fn pose_with_identity_rotation_at_center(center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -center)
}

fn extractor_from_frame(frame: &Frame) -> StaticFeatureExtractor {
    StaticFeatureExtractor {
        features: FeatureSet {
            keypoints: frame.keypoints.clone(),
            descriptors: frame.descriptors.clone(),
        },
    }
}

fn successful_tracking_result(frame_id: u64, pose: Pose) -> TrackingResult {
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event: TrackingEvent::Tracked,
        successive_failures: 0,
        pose_prior: None,
        used_pose_prior: false,
        used_external_localization_prior: false,
        external_localization_prior_radius: None,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: MapProviderStats::default(),
        localization: visloc_rs::core::types::LocalizationResult::success(
            visloc_rs::core::types::LocalizationSuccess {
                pose,
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
            },
        ),
    }
}

fn failed_tracking_result(frame_id: u64) -> TrackingResult {
    TrackingResult {
        frame_id,
        state: TrackingState::Tracking,
        event: TrackingEvent::TrackingFailed,
        successive_failures: 1,
        pose_prior: None,
        used_pose_prior: false,
        used_external_localization_prior: false,
        external_localization_prior_radius: None,
        tracking_failure_reason: None,
        map_landmark_count: 0,
        map_stats: MapProviderStats::default(),
        localization: LocalizationResult::failure(
            LocalizationFailureReason::NoDescriptorMatches,
            0,
            0,
            0,
        ),
    }
}

fn successful_localization_result(
    inlier_count: usize,
    correspondence_count: usize,
    mean_reprojection_error: f64,
) -> LocalizationResult {
    LocalizationResult::success(LocalizationSuccess {
        pose: Pose::identity(),
        candidate_landmark_count: correspondence_count,
        match_count: correspondence_count,
        correspondence_count,
        inliers: (0..inlier_count).collect(),
        inlier_query_indices: (0..inlier_count).collect(),
        inlier_landmark_ids: (1..=inlier_count as u64).collect(),
        inlier_reprojection_errors: vec![mean_reprojection_error; inlier_count],
        mean_reprojection_error,
        median_reprojection_error: mean_reprojection_error,
        max_reprojection_error: mean_reprojection_error,
    })
}

fn build_map_and_frame(frame_id: u64, camera_id: u64) -> (VisualMap, Frame) {
    let camera = Camera::pinhole(camera_id, 640, 480, 500.0, 500.0, 320.0, 240.0);
    let pose = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
    let points = vec![
        Point3::new(-1.0, -1.0, 4.0),
        Point3::new(1.0, -1.0, 4.5),
        Point3::new(-1.0, 1.0, 5.0),
        Point3::new(1.0, 1.0, 5.5),
        Point3::new(0.0, 0.0, 6.0),
        Point3::new(0.5, -0.25, 7.0),
    ];

    let mut map = VisualMap::new();
    map.cameras.insert(camera.id, camera.clone());
    let mut frame = Frame::new(frame_id, camera.id);

    for (index, point) in points.iter().enumerate() {
        let descriptor = vec![index as f32, 9.0];
        let mut landmark = Landmark::new(index as u64 + 1, *point);
        landmark.descriptor = Some(descriptor.clone());
        map.landmarks.insert(landmark.id, landmark);
        frame
            .keypoints
            .push(camera.project(&pose.transform_world_point(point)).unwrap());
        frame.descriptors.push(descriptor);
    }

    (map, frame)
}

#[test]
fn constant_velocity_motion_model_extrapolates_camera_center() {
    let mut model = ConstantVelocityMotionModel::new();
    let frame = Frame::new(3, 1);
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(2.0, 0.0, 0.0));

    model.observe(&successful_tracking_result(1, pose_a));
    model.observe(&successful_tracking_result(2, pose_b));
    let prediction = model.predict_pose(&frame, None, None).unwrap();

    let predicted_center = prediction.camera_center_world();
    assert!((predicted_center - Point3::new(4.0, 0.0, 0.0)).norm() < 1.0e-9);
}

#[test]
fn constant_velocity_motion_model_reset_clears_history() {
    let mut model = ConstantVelocityMotionModel::new();
    let frame = Frame::new(3, 1);
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(2.0, 0.0, 0.0));

    model.observe(&successful_tracking_result(1, pose_a));
    model.observe(&successful_tracking_result(2, pose_b));
    assert!(model.predict_pose(&frame, None, None).is_some());

    model.reset();

    assert!(model.predict_pose(&frame, None, None).is_none());
}

#[test]
fn pose_trajectory_keeps_successful_tracking_poses() {
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(3.0, 4.0, 0.0));
    let pose_c = pose_with_identity_rotation_at_center(Vector3::new(3.0, 4.0, 12.0));
    let results = vec![
        successful_tracking_result(1, pose_a),
        failed_tracking_result(2),
        successful_tracking_result(3, pose_b),
        successful_tracking_result(4, pose_c),
    ];

    let trajectory = PoseTrajectory::from_tracking_results(&results);

    assert_eq!(trajectory.len(), 3);
    assert_eq!(trajectory.frame_ids(), vec![1, 3, 4]);
    assert!((trajectory.total_path_length() - 17.0).abs() < 1.0e-9);
    assert_eq!(trajectory.mean_reprojection_error(), Some(0.0));
    assert_eq!(trajectory.samples()[0].inlier_count, 0);
    assert_eq!(
        trajectory.camera_centers_world(),
        vec![
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(3.0, 4.0, 0.0),
            Point3::new(3.0, 4.0, 12.0)
        ]
    );
}

#[test]
fn pose_trajectory_exports_csv() {
    let pose = pose_with_identity_rotation_at_center(Vector3::new(1.0, 2.0, 3.0));
    let trajectory = PoseTrajectory::from_tracking_results(&[successful_tracking_result(42, pose)]);

    let csv = trajectory.to_csv();

    assert!(csv.starts_with(
        "frame_id,camera_center_x,camera_center_y,camera_center_z,qw,qx,qy,qz,tx,ty,tz,state,event,inlier_count,inlier_ratio,reprojection_error\n"
    ));
    assert!(csv.contains("42,1,2,3,1,0,0,0,-1,-2,-3,Tracking,Tracked,0,0,0\n"));
}

#[test]
fn pose_trajectory_exports_kitti_poses() {
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(1.0, 2.0, 3.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(4.0, 5.0, 6.0));
    let trajectory = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(1, pose_a),
        failed_tracking_result(2),
        successful_tracking_result(3, pose_b),
    ]);

    let kitti = trajectory.to_kitti_poses();

    assert_eq!(kitti, "1 0 0 1 0 1 0 2 0 0 1 3\n1 0 0 4 0 1 0 5 0 0 1 6\n");
}

#[test]
fn pose_trajectory_parses_kitti_poses() {
    let trajectory = PoseTrajectory::from_kitti_poses_str(
        "# r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz\n1 0 0 1 0 1 0 2 0 0 1 3\n1 0 0 4 0 1 0 5 0 0 1 6\n",
    )
    .unwrap();

    assert_eq!(trajectory.frame_ids(), vec![0, 1]);
    assert_eq!(
        trajectory.camera_centers_world(),
        vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]
    );
    assert_eq!(
        trajectory.to_kitti_poses(),
        "1 0 0 1 0 1 0 2 0 0 1 3\n1 0 0 4 0 1 0 5 0 0 1 6\n"
    );
}

#[test]
fn pose_trajectory_reads_kitti_pose_file() {
    let path = std::env::temp_dir().join(format!(
        "visloc_tracking_kitti_{}_{}.txt",
        std::process::id(),
        "pose_file"
    ));
    fs::write(&path, "1 0 0 1 0 1 0 2 0 0 1 3\n").unwrap();

    let trajectory = PoseTrajectory::read_kitti_poses(&path).unwrap();

    assert_eq!(trajectory.frame_ids(), vec![0]);
    assert_eq!(
        trajectory.camera_centers_world(),
        vec![Point3::new(1.0, 2.0, 3.0)]
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn pose_trajectory_rejects_invalid_kitti_pose_lines() {
    let error = PoseTrajectory::from_kitti_poses_str("1 2 3\n").unwrap_err();

    assert_eq!(error.line_number, 1);
    assert!(error.message.contains("expected 12 fields"));
}

#[test]
fn pose_trajectory_exports_tum_poses() {
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(1.0, 2.0, 3.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(4.0, 5.0, 6.0));
    let trajectory = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(1, pose_a),
        failed_tracking_result(2),
        successful_tracking_result(3, pose_b),
    ]);

    let tum = trajectory.to_tum_poses();

    assert_eq!(tum, "1 1 2 3 0 0 0 1\n3 4 5 6 0 0 0 1\n");
}

#[test]
fn pose_trajectory_parses_tum_poses() {
    let trajectory = PoseTrajectory::from_tum_poses_str(
        "# frame_id tx ty tz qx qy qz qw\n1 1 2 3 0 0 0 1\n3 4 5 6 0 0 0 1\n",
    )
    .unwrap();

    assert_eq!(trajectory.frame_ids(), vec![1, 3]);
    assert_eq!(
        trajectory.camera_centers_world(),
        vec![Point3::new(1.0, 2.0, 3.0), Point3::new(4.0, 5.0, 6.0)]
    );
    assert_eq!(
        trajectory.to_tum_poses(),
        "1 1 2 3 0 0 0 1\n3 4 5 6 0 0 0 1\n"
    );
}

#[test]
fn pose_trajectory_reads_tum_pose_file() {
    let path = std::env::temp_dir().join(format!(
        "visloc_tracking_tum_{}_{}.txt",
        std::process::id(),
        "pose_file"
    ));
    fs::write(&path, "7 1 2 3 0 0 0 1\n").unwrap();

    let trajectory = PoseTrajectory::read_tum_poses(&path).unwrap();

    assert_eq!(trajectory.frame_ids(), vec![7]);
    assert_eq!(
        trajectory.camera_centers_world(),
        vec![Point3::new(1.0, 2.0, 3.0)]
    );
    fs::remove_file(path).unwrap();
}

#[test]
fn pose_trajectory_rejects_invalid_tum_pose_lines() {
    let error = PoseTrajectory::from_tum_poses_str("1 2 3\n").unwrap_err();

    assert_eq!(error.line_number, 1);
    assert!(error.message.contains("expected 8 fields"));
}

#[test]
fn pose_trajectory_exports_summary_json() {
    let pose_a = pose_with_identity_rotation_at_center(Vector3::new(1.0, 2.0, 3.0));
    let pose_b = pose_with_identity_rotation_at_center(Vector3::new(4.0, 6.0, 8.0));
    let trajectory = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(10, pose_a),
        failed_tracking_result(11),
        successful_tracking_result(12, pose_b),
    ]);

    let summary = trajectory.summary();
    let json = trajectory.to_summary_json();

    assert_eq!(summary.pose_count, 2);
    assert_eq!(summary.first_frame_id, Some(10));
    assert_eq!(summary.last_frame_id, Some(12));
    assert!(
        (summary.total_path_length - (3.0_f64 * 3.0 + 4.0 * 4.0 + 5.0 * 5.0).sqrt()).abs() < 1.0e-9
    );
    assert_eq!(summary.mean_inlier_count, Some(0.0));
    assert_eq!(summary.mean_inlier_ratio, Some(0.0));
    assert_eq!(summary.mean_reprojection_error, Some(0.0));
    assert_eq!(summary.min_camera_center_world, Some([1.0, 2.0, 3.0]));
    assert_eq!(summary.max_camera_center_world, Some([4.0, 6.0, 8.0]));
    assert!(json.contains("\"pose_count\": 2"));
    assert!(json.contains("\"first_frame_id\": 10"));
    assert!(json.contains("\"min_camera_center_world\": [1, 2, 3]"));
}

#[test]
fn pose_trajectory_exports_html_report_without_reference() {
    let trajectory = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            7,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            9,
            pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 1.0)),
        ),
    ]);

    let html = trajectory.to_html_report();

    assert!(html.contains("<!doctype html>"));
    assert!(html.contains("visloc-rs trajectory report"));
    assert!(html.contains("<span class=\"label\">Poses</span><span class=\"value\">2</span>"));
    assert!(html.contains("<span class=\"label\">First frame</span><span class=\"value\">7</span>"));
    assert!(html.contains("<span class=\"label\">Last frame</span><span class=\"value\">9</span>"));
    assert!(html.contains("estimated trajectory"));
    assert!(html.contains("<svg viewBox=\"0 0 900 520\""));
}

#[test]
fn pose_trajectory_writes_html_report_without_reference() {
    let trajectory = PoseTrajectory::from_tracking_results(&[successful_tracking_result(
        7,
        pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
    )]);
    let path = std::env::temp_dir().join(format!(
        "visloc-rs-trajectory-only-report-{}.html",
        std::process::id()
    ));

    trajectory.write_html_report(&path).unwrap();
    let html = fs::read_to_string(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert!(html.contains("Estimated camera-center trajectory"));
    assert!(html.contains("Path length"));
}

#[test]
fn pose_trajectory_reports_translation_errors_against_reference() {
    let estimated = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            1,
            pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            2,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 2.0, 0.0)),
        ),
        successful_tracking_result(
            4,
            pose_with_identity_rotation_at_center(Vector3::new(9.0, 0.0, 0.0)),
        ),
    ]);
    let reference = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            1,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            2,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            3,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 3.0)),
        ),
    ]);

    let errors = estimated.translation_errors_against(&reference);
    let summary = estimated.translation_error_summary_against(&reference);
    let csv = estimated.translation_errors_csv_against(&reference);
    let json = summary.to_json();

    assert_eq!(errors.len(), 2);
    assert_eq!(errors[0].to_csv_record(), "1,1");
    assert_eq!(errors[0].frame_id, 1);
    assert_eq!(errors[0].translation_error, 1.0);
    assert_eq!(errors[1].frame_id, 2);
    assert_eq!(errors[1].translation_error, 2.0);
    assert_eq!(summary.estimated_pose_count, 3);
    assert_eq!(summary.reference_pose_count, 3);
    assert_eq!(summary.matched_pose_count, 2);
    assert_eq!(summary.missing_reference_count, 1);
    assert_eq!(summary.missing_estimate_count, 1);
    assert_eq!(summary.mean_translation_error, Some(1.5));
    assert!((summary.rmse_translation_error.unwrap() - (2.5_f64).sqrt()).abs() < 1.0e-9);
    assert_eq!(summary.max_translation_error, Some(2.0));
    assert_eq!(csv, "frame_id,translation_error\n1,1\n2,2\n");
    assert!(json.contains("\"matched_pose_count\": 2"));
    assert!(json.contains("\"mean_translation_error\": 1.5"));
}

#[test]
fn pose_trajectory_can_align_first_matching_translation_before_error_summary() {
    let estimated = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            1,
            pose_with_identity_rotation_at_center(Vector3::new(10.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            2,
            pose_with_identity_rotation_at_center(Vector3::new(11.0, 1.0, 0.0)),
        ),
    ]);
    let reference = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            1,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            2,
            pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 0.0)),
        ),
    ]);

    let unaligned = estimated.translation_error_summary_against(&reference);
    let aligned = estimated.translation_error_summary_against_with_alignment(
        &reference,
        TrajectoryAlignment::FirstMatchedTranslation,
    );
    let aligned_errors = estimated.translation_errors_against_with_alignment(
        &reference,
        TrajectoryAlignment::FirstMatchedTranslation,
    );
    let aligned_csv = estimated.translation_errors_csv_against_with_alignment(
        &reference,
        TrajectoryAlignment::FirstMatchedTranslation,
    );

    assert!((unaligned.mean_translation_error.unwrap() - 10.024937810560445).abs() < 1.0e-9);
    assert_eq!(aligned.mean_translation_error, Some(0.5));
    assert_eq!(aligned.rmse_translation_error, Some((0.5_f64).sqrt()));
    assert_eq!(aligned.max_translation_error, Some(1.0));
    assert_eq!(aligned_errors[0].translation_error, 0.0);
    assert_eq!(aligned_errors[1].translation_error, 1.0);
    assert_eq!(aligned_csv, "frame_id,translation_error\n1,0\n2,1\n");
}

#[test]
fn pose_trajectory_exports_html_report_against_reference() {
    let estimated = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            1,
            pose_with_identity_rotation_at_center(Vector3::new(10.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            2,
            pose_with_identity_rotation_at_center(Vector3::new(11.0, 1.0, 0.0)),
        ),
    ]);
    let reference = PoseTrajectory::from_tracking_results(&[
        successful_tracking_result(
            1,
            pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
        ),
        successful_tracking_result(
            2,
            pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 0.0)),
        ),
    ]);

    let html = estimated.to_html_report_against_with_alignment(
        &reference,
        TrajectoryAlignment::FirstMatchedTranslation,
    );

    assert!(html.contains("<!doctype html>"));
    assert!(html.contains("visloc-rs trajectory evaluation"));
    assert!(html.contains("Alignment: <code>FirstMatchedTranslation</code>"));
    assert!(html.contains("<svg viewBox=\"0 0 900 520\""));
    assert!(html.contains("<span class=\"value\">0.5000 m</span>"));
    assert!(html.contains("<td>2</td><td>1.0000 m</td>"));
}

#[test]
fn pose_trajectory_writes_html_report_against_reference() {
    let estimated = PoseTrajectory::from_tracking_results(&[successful_tracking_result(
        1,
        pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 0.0)),
    )]);
    let reference = PoseTrajectory::from_tracking_results(&[successful_tracking_result(
        1,
        pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0)),
    )]);
    let path = std::env::temp_dir().join(format!(
        "visloc-rs-trajectory-report-{}.html",
        std::process::id()
    ));

    estimated
        .write_html_report_against(&reference, &path)
        .unwrap();
    let html = fs::read_to_string(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert!(html.contains("Mean error"));
    assert!(html.contains("1.0000 m"));
}

#[test]
fn pose_trajectory_error_summary_handles_no_matching_frames() {
    let estimated = PoseTrajectory::from_tracking_results(&[successful_tracking_result(
        1,
        pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 0.0)),
    )]);
    let reference = PoseTrajectory::from_tracking_results(&[successful_tracking_result(
        2,
        pose_with_identity_rotation_at_center(Vector3::new(1.0, 0.0, 0.0)),
    )]);

    let summary = estimated.translation_error_summary_against(&reference);

    assert_eq!(summary.matched_pose_count, 0);
    assert_eq!(summary.missing_reference_count, 1);
    assert_eq!(summary.missing_estimate_count, 1);
    assert_eq!(summary.mean_translation_error, None);
    assert_eq!(summary.rmse_translation_error, None);
    assert_eq!(summary.max_translation_error, None);
}

#[test]
fn tracking_stats_can_be_rebuilt_from_results() {
    let pose = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let mut success = successful_tracking_result(10, pose);
    success.event = TrackingEvent::Initialized;
    success.used_pose_prior = true;
    success.used_external_localization_prior = true;
    success.external_localization_prior_radius = Some(8.0);
    let mut lost = failed_tracking_result(11);
    lost.state = TrackingState::Lost;
    lost.event = TrackingEvent::Lost;

    let stats = TrackingStats::from_results(&[success, lost]);

    assert_eq!(stats.first_frame_id, Some(10));
    assert_eq!(stats.last_frame_id, Some(11));
    assert_eq!(stats.frame_count, 2);
    assert_eq!(stats.successful_frame_count, 1);
    assert_eq!(stats.failed_frame_count, 1);
    assert_eq!(stats.lost_count, 1);
    assert_eq!(stats.pose_prior_used_count, 1);
    assert_eq!(stats.external_localization_prior_used_count, 1);
    assert_eq!(stats.success_rate(), 0.5);
    assert_eq!(stats.external_localization_prior_usage_rate(), 0.5);
}

#[test]
fn tracking_stats_exports_json() {
    let pose = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let mut success = successful_tracking_result(10, pose);
    success.used_pose_prior = true;
    success.used_external_localization_prior = true;
    success.external_localization_prior_radius = Some(8.0);
    let failed = failed_tracking_result(11);
    let stats = TrackingStats::from_results(&[success, failed]);

    let json = stats.to_json();

    assert!(json.contains("\"first_frame_id\": 10"));
    assert!(json.contains("\"last_frame_id\": 11"));
    assert!(json.contains("\"frame_count\": 2"));
    assert!(json.contains("\"success_rate\": 0.5"));
    assert!(json.contains("\"pose_prior_usage_rate\": 0.5"));
    assert!(json.contains("\"external_localization_prior_usage_rate\": 0.5"));
}

#[test]
fn tracking_stats_writes_json() {
    let path = std::env::temp_dir().join(format!(
        "visloc-rs-tracking-summary-{}.json",
        std::process::id()
    ));
    let stats = TrackingStats::from_results(&[failed_tracking_result(12)]);

    stats.write_json(&path).unwrap();
    let json = fs::read_to_string(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert!(json.contains("\"frame_count\": 1"));
    assert!(json.contains("\"failure_rate\": 1"));
}

#[test]
fn tracking_results_export_html_report() {
    let pose = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let mut initialized = successful_tracking_result(10, pose);
    initialized.event = TrackingEvent::Initialized;
    initialized.used_pose_prior = true;
    initialized.used_external_localization_prior = true;
    initialized.external_localization_prior_radius = Some(8.0);
    let mut lost = failed_tracking_result(11);
    lost.state = TrackingState::Lost;
    lost.event = TrackingEvent::Lost;

    let html = tracking_results_to_html_report(&[initialized, lost]);

    assert!(html.contains("<!doctype html>"));
    assert!(html.contains("visloc-rs tracking report"));
    assert!(html.contains("<span class=\"label\">Frames</span><span class=\"value\">2</span>"));
    assert!(html
        .contains("<span class=\"label\">Success rate</span><span class=\"value\">50.0%</span>"));
    assert!(html.contains("NoDescriptorMatches"));
    assert!(html.contains("motion + external(8.000m)"));
    assert!(html.contains("Lost"));
    assert!(html.contains("<svg viewBox=\"0 0 900 190\""));
}

#[test]
fn tracking_results_write_html_report() {
    let path = std::env::temp_dir().join(format!(
        "visloc-rs-tracking-report-{}.html",
        std::process::id()
    ));
    let result = failed_tracking_result(12);

    write_tracking_results_html_report(&[result], &path).unwrap();
    let html = fs::read_to_string(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert!(html.contains("Frame-by-frame sequence-localization state"));
    assert!(html.contains("failed"));
}

#[test]
fn tracking_results_export_csv() {
    let pose = pose_with_identity_rotation_at_center(Vector3::new(0.0, 0.0, 0.0));
    let mut initialized = successful_tracking_result(10, pose);
    initialized.event = TrackingEvent::Initialized;
    initialized.used_pose_prior = true;
    initialized.used_external_localization_prior = true;
    initialized.external_localization_prior_radius = Some(8.0);
    let mut lost = failed_tracking_result(11);
    lost.state = TrackingState::Lost;
    lost.event = TrackingEvent::Lost;

    let csv = tracking_results_to_csv(&[initialized, lost]);

    assert!(csv.starts_with("frame_id,state,event,success,successive_failures"));
    assert!(csv.contains("10,Tracking,Initialized,true,0,true,true,8"));
    assert!(csv.contains("11,Lost,Lost,false,1,false,false,"));
    assert!(csv.contains("NoDescriptorMatches"));
}

#[test]
fn tracking_results_write_csv() {
    let path = std::env::temp_dir().join(format!("visloc-rs-tracking-{}.csv", std::process::id()));
    let result = failed_tracking_result(12);

    write_tracking_results_csv(&[result], &path).unwrap();
    let csv = fs::read_to_string(&path).unwrap();
    fs::remove_file(path).unwrap();

    assert!(csv.contains("frame_id,state,event"));
    assert!(csv.contains("12,Tracking,TrackingFailed,false"));
}

#[test]
fn tracker_enters_tracking_after_successful_localization() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let result = tracker.track_frame(&frame, &map);

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert_eq!(result.event, TrackingEvent::Initialized);
    assert_eq!(result.successive_failures, 0);
    assert!(result.pose_prior.is_none());
    assert!(!result.used_pose_prior);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert!(result.localization.success);
    assert_eq!(tracker.state(), TrackingState::Tracking);
    assert_eq!(tracker.last_result().unwrap().frame_id, 10);
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
    assert!(tracker.last_successful_pose().is_some());
    assert_eq!(tracker.stats().first_frame_id, Some(10));
    assert_eq!(tracker.stats().last_frame_id, Some(10));
    assert_eq!(tracker.stats().frame_count, 1);
    assert_eq!(tracker.stats().successful_frame_count, 1);
    assert_eq!(tracker.stats().failed_frame_count, 0);
    assert_eq!(tracker.stats().success_rate(), 1.0);
    assert_eq!(tracker.stats().failure_rate(), 0.0);
    assert_eq!(tracker.stats().overall_inlier_ratio(), 1.0);
    assert_eq!(tracker.stats().mean_inliers_per_successful_frame(), 6.0);
}

#[test]
fn tracker_reset_clears_sequence_state_history_and_stats() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let result = tracker.track_frame(&frame, &map);
    assert!(result.localization.success);
    assert_eq!(tracker.state(), TrackingState::Tracking);
    assert!(tracker.last_result().is_some());
    assert!(tracker.last_successful_pose().is_some());
    assert_eq!(tracker.stats().frame_count, 1);

    tracker.reset();

    assert_eq!(tracker.state(), TrackingState::Uninitialized);
    assert_eq!(tracker.successive_failures(), 0);
    assert!(tracker.last_result().is_none());
    assert_eq!(tracker.last_successful_frame_id(), None);
    assert!(tracker.last_successful_pose().is_none());
    assert_eq!(tracker.stats().frame_count, 0);
    assert_eq!(tracker.stats().successful_frame_count, 0);
    assert_eq!(tracker.stats().first_frame_id, None);
    assert_eq!(tracker.stats().last_frame_id, None);
}

#[test]
fn tracker_tracks_frame_with_map_provider() {
    let (map, frame) = build_map_and_frame(10, 1);
    let provider = InMemoryMapProvider::new(map);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let result = tracker.track_frame_with_provider(&frame, &provider);

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(tracker.stats().successful_frame_count, 1);
}

#[test]
fn tracker_tracks_frame_sequence_with_convenience_api() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut second_frame = frame.clone();
    second_frame.id = 11;
    let frames = vec![frame, second_frame];
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let results = tracker.track_frames(&frames, &map);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].event, TrackingEvent::Initialized);
    assert_eq!(results[1].event, TrackingEvent::Tracked);
    assert!(results.iter().all(|result| result.localization.success));
    assert_eq!(tracker.stats().first_frame_id, Some(10));
    assert_eq!(tracker.stats().last_frame_id, Some(11));
    assert_eq!(tracker.stats().frame_count, 2);
    assert_eq!(tracker.stats().success_rate(), 1.0);
}

#[test]
fn tracker_tracks_provider_sequence_with_convenience_api() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut second_frame = frame.clone();
    second_frame.id = 11;
    let frames = vec![frame, second_frame];
    let provider = InMemoryMapProvider::new(map);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let results = tracker.track_frames_with_provider(&frames, &provider);

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.localization.success));
    assert_eq!(results[0].map_stats.landmark_count, 6);
    assert_eq!(results[1].map_stats.descriptor_count, 6);
    assert_eq!(tracker.stats().successful_frame_count, 2);
}

#[test]
fn tracker_predicts_localization_prior_for_next_frame() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let first = tracker.track_frame(&frame, &map);
    let prior = tracker.localization_prior_for_frame(&frame, 8.0);
    let provider = SelectableMapProvider::new(
        InMemoryMapProvider::new(map),
        PriorSubmapSelector::new(prior.clone()),
    );

    assert!(first.localization.success);
    assert_eq!(prior.radius, Some(8.0));
    assert!(prior.pose.is_some());
    assert!(prior.to_radius_submap_selector().is_some());
    assert_eq!(provider.selected_provider().map.landmarks.len(), 6);
}

#[test]
fn tracker_tracks_with_motion_prior_submap_provider() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let first = tracker.track_frame_with_prior_submap_provider(&frame, &provider, 8.0);
    let second = tracker.track_frame_with_prior_submap_provider(&frame, &provider, 8.0);

    assert!(first.localization.success);
    assert_eq!(first.map_landmark_count, 12);
    assert_eq!(first.map_stats.landmark_count, 12);
    assert_eq!(first.map_stats.descriptor_count, 12);
    assert_eq!(first.localization.candidate_landmark_count, 12);
    assert!(first.pose_prior.is_none());
    assert!(second.localization.success);
    assert_eq!(second.map_landmark_count, 6);
    assert_eq!(second.map_stats.landmark_count, 6);
    assert_eq!(second.map_stats.descriptor_count, 6);
    assert_eq!(second.localization.candidate_landmark_count, 6);
    assert!(second.pose_prior.is_some());
}

#[test]
fn tracker_tracks_with_external_localization_prior_submap_provider() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let prior = LocalizationPrior::from_position(Point3::origin(), 8.0);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let result =
        tracker.track_frame_with_localization_prior_submap_provider(&frame, &provider, &prior);

    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(result.localization.candidate_landmark_count, 6);
    assert!(result.pose_prior.is_none());
    assert!(!result.used_pose_prior);
    assert!(result.used_external_localization_prior);
    assert_eq!(result.external_localization_prior_radius, Some(8.0));
    assert_eq!(
        tracker
            .last_result()
            .unwrap()
            .external_localization_prior_radius,
        Some(8.0)
    );
    assert_eq!(tracker.stats().external_localization_prior_used_count, 1);
}

#[test]
fn tracker_tracks_sequence_with_optional_external_localization_priors() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    let mut second_frame = frame.clone();
    second_frame.id = 11;
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let prior = LocalizationPrior::from_position(Point3::origin(), 8.0);
    let mut tracker = Tracker::new(LocalizationPipeline::default(), TrackingConfig::default());

    let results = tracker.track_frames_with_localization_prior_submap_provider(
        [(&frame, Some(&prior)), (&second_frame, None)],
        &provider,
    );

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.localization.success));
    assert_eq!(results[0].map_landmark_count, 6);
    assert_eq!(results[0].localization.candidate_landmark_count, 6);
    assert!(results[0].used_external_localization_prior);
    assert_eq!(results[0].external_localization_prior_radius, Some(8.0));
    assert_eq!(results[1].map_landmark_count, 12);
    assert_eq!(results[1].localization.candidate_landmark_count, 12);
    assert!(!results[1].used_external_localization_prior);
    assert_eq!(tracker.stats().external_localization_prior_used_count, 1);
}

#[test]
fn image_tracker_tracks_extracted_frame_image() {
    let (map, frame) = build_map_and_frame(10, 1);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let result = image_tracker
        .track_frame_image(10, frame.camera_id, &(), &map)
        .unwrap();

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert_eq!(result.event, TrackingEvent::Initialized);
    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(result.localization.inlier_count, 6);
    assert_eq!(image_tracker.tracker().stats().successful_frame_count, 1);
}

#[test]
fn image_tracker_tracks_image_sequence_with_convenience_api() {
    let (map, frame) = build_map_and_frame(10, 1);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());
    let images = [(), ()];

    let results = image_tracker
        .track_frame_images(
            [
                (10, frame.camera_id, &images[0]),
                (11, frame.camera_id, &images[1]),
            ],
            &map,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].event, TrackingEvent::Initialized);
    assert_eq!(results[1].event, TrackingEvent::Tracked);
    assert!(results.iter().all(|result| result.localization.success));
    assert_eq!(image_tracker.tracker().stats().first_frame_id, Some(10));
    assert_eq!(image_tracker.tracker().stats().last_frame_id, Some(11));
    assert_eq!(image_tracker.tracker().stats().successful_frame_count, 2);
}

#[test]
fn image_tracker_reset_clears_inner_tracker_state() {
    let (map, frame) = build_map_and_frame(10, 1);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    image_tracker
        .track_frame_image(10, frame.camera_id, &(), &map)
        .unwrap();
    assert_eq!(image_tracker.tracker().state(), TrackingState::Tracking);

    image_tracker.reset();

    assert_eq!(
        image_tracker.tracker().state(),
        TrackingState::Uninitialized
    );
    assert!(image_tracker.tracker().last_result().is_none());
    assert_eq!(image_tracker.tracker().stats().frame_count, 0);
}

#[test]
fn image_tracker_tracks_with_motion_prior_submap_provider() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let first = image_tracker
        .track_frame_image_with_prior_submap_provider(10, frame.camera_id, &(), &provider, 8.0)
        .unwrap();
    let second = image_tracker
        .track_frame_image_with_prior_submap_provider(11, frame.camera_id, &(), &provider, 8.0)
        .unwrap();

    assert!(first.localization.success);
    assert_eq!(first.map_landmark_count, 12);
    assert_eq!(first.map_stats.landmark_count, 12);
    assert_eq!(first.map_stats.descriptor_count, 12);
    assert_eq!(first.localization.candidate_landmark_count, 12);
    assert!(second.localization.success);
    assert_eq!(second.map_landmark_count, 6);
    assert_eq!(second.map_stats.landmark_count, 6);
    assert_eq!(second.map_stats.descriptor_count, 6);
    assert_eq!(second.localization.candidate_landmark_count, 6);
    assert!(second.pose_prior.is_some());
}

#[test]
fn image_tracker_tracks_with_external_localization_prior_submap_provider() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    for index in 0..6 {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, Point3::new(100.0 + index as f64, 0.0, 5.0));
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }
    let provider = InMemoryMapProvider::new(map);
    let prior = LocalizationPrior::from_position(Point3::origin(), 8.0);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let result = image_tracker
        .track_frame_image_with_localization_prior_submap_provider(
            10,
            frame.camera_id,
            &(),
            &provider,
            &prior,
        )
        .unwrap();

    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.localization.candidate_landmark_count, 6);
    assert!(result.pose_prior.is_none());
    assert!(result.used_external_localization_prior);
    assert_eq!(result.external_localization_prior_radius, Some(8.0));
    assert_eq!(
        image_tracker
            .tracker()
            .stats()
            .external_localization_prior_used_count,
        1
    );
}

#[test]
fn tracking_result_exposes_pose_prior_as_localization_prior() {
    let (map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
            max_pose_prior_translation_error: None,
            ..TrackingConfig::default()
        },
    );

    let first = tracker.track_frame(&frame, &map);
    let second = tracker.track_frame(&frame, &map);
    let first_prior = first.localization_prior(8.0);
    let second_prior = second.localization_prior(8.0);

    assert!(first_prior.pose.is_none());
    assert_eq!(first_prior.radius, None);
    assert!(second_prior.pose.is_some());
    assert_eq!(second_prior.radius, Some(8.0));
    assert!(second_prior.to_radius_submap_selector().is_some());
}

#[test]
fn image_tracker_tracks_frame_image_with_map_provider() {
    let (map, frame) = build_map_and_frame(10, 1);
    let provider = InMemoryMapProvider::new(map);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());

    let result = image_tracker
        .track_frame_image_with_provider(10, frame.camera_id, &(), &provider)
        .unwrap();

    assert_eq!(result.frame_id, 10);
    assert_eq!(result.state, TrackingState::Tracking);
    assert!(result.localization.success);
    assert_eq!(result.map_landmark_count, 6);
    assert_eq!(result.map_stats.landmark_count, 6);
    assert_eq!(result.map_stats.descriptor_count, 6);
    assert_eq!(image_tracker.tracker().stats().successful_frame_count, 1);
}

#[test]
fn image_tracker_tracks_provider_image_sequence_with_convenience_api() {
    let (map, frame) = build_map_and_frame(10, 1);
    let provider = InMemoryMapProvider::new(map);
    let extractor = extractor_from_frame(&frame);
    let mut image_tracker = ImageTracker::new(extractor, TrackingConfig::default());
    let images = [(), ()];

    let results = image_tracker
        .track_frame_images_with_provider(
            [
                (10, frame.camera_id, &images[0]),
                (11, frame.camera_id, &images[1]),
            ],
            &provider,
        )
        .unwrap();

    assert_eq!(results.len(), 2);
    assert!(results.iter().all(|result| result.localization.success));
    assert_eq!(results[0].map_stats.landmark_count, 6);
    assert_eq!(results[1].map_stats.descriptor_count, 6);
    assert_eq!(image_tracker.tracker().stats().successful_frame_count, 2);
}

#[test]
fn tracker_becomes_lost_after_successive_failures() {
    let (_map, frame) = build_map_and_frame(10, 1);
    let empty_map = VisualMap::new();
    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            ..TrackingConfig::default()
        },
    );

    let first = tracker.track_frame(&frame, &empty_map);
    let second = tracker.track_frame(&frame, &empty_map);

    assert_eq!(first.state, TrackingState::Uninitialized);
    assert_eq!(first.event, TrackingEvent::TrackingFailed);
    assert!(!first.localization.success);
    assert_eq!(first.successive_failures, 1);
    assert_eq!(second.state, TrackingState::Lost);
    assert_eq!(second.event, TrackingEvent::Lost);
    assert!(first.pose_prior.is_none());
    assert!(!first.used_pose_prior);
    assert!(second.pose_prior.is_none());
    assert!(!second.used_pose_prior);
    assert!(!second.localization.success);
    assert_eq!(second.map_landmark_count, 0);
    assert_eq!(second.map_stats.landmark_count, 0);
    assert_eq!(second.map_stats.descriptor_count, 0);
    assert_eq!(second.successive_failures, 2);
    assert_eq!(tracker.state(), TrackingState::Lost);
    assert_eq!(tracker.last_result().unwrap().event, TrackingEvent::Lost);
    assert_eq!(tracker.last_successful_frame_id(), None);
    assert!(tracker.last_successful_pose().is_none());
    assert_eq!(tracker.stats().frame_count, 2);
    assert_eq!(tracker.stats().successful_frame_count, 0);
    assert_eq!(tracker.stats().failed_frame_count, 2);
    assert_eq!(tracker.stats().lost_count, 1);
    assert_eq!(tracker.stats().success_rate(), 0.0);
    assert_eq!(tracker.stats().failure_rate(), 1.0);
}

#[test]
fn tracker_reports_relocalized_after_lost_success() {
    let (map, frame) = build_map_and_frame(10, 1);
    let empty_map = VisualMap::new();
    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 1,
            ..TrackingConfig::default()
        },
    );

    let lost = tracker.track_frame(&frame, &empty_map);
    let relocalized = tracker.track_frame(&frame, &map);

    assert_eq!(lost.state, TrackingState::Lost);
    assert_eq!(lost.event, TrackingEvent::Lost);
    assert_eq!(relocalized.state, TrackingState::Tracking);
    assert_eq!(relocalized.event, TrackingEvent::Relocalized);
    assert!(relocalized.pose_prior.is_none());
    assert!(!relocalized.used_pose_prior);
    assert!(relocalized.localization.success);
    assert_eq!(relocalized.map_landmark_count, 6);
    assert_eq!(relocalized.map_stats.landmark_count, 6);
    assert_eq!(relocalized.map_stats.descriptor_count, 6);
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
    assert!(tracker.last_successful_pose().is_some());
    assert_eq!(tracker.stats().frame_count, 2);
    assert_eq!(tracker.stats().successful_frame_count, 1);
    assert_eq!(tracker.stats().failed_frame_count, 1);
    assert_eq!(tracker.stats().lost_count, 1);
    assert_eq!(tracker.stats().relocalization_count, 1);
    assert_eq!(tracker.stats().first_frame_id, Some(10));
    assert_eq!(tracker.stats().last_frame_id, Some(10));
    assert_eq!(tracker.stats().success_rate(), 0.5);
    assert_eq!(tracker.stats().failure_rate(), 0.5);
}

#[test]
fn tracker_uses_last_pose_prior_to_limit_landmark_candidates() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    let far_points = [
        Point3::new(100.0, 0.0, 5.0),
        Point3::new(101.0, 0.0, 5.0),
        Point3::new(102.0, 0.0, 5.0),
        Point3::new(103.0, 0.0, 5.0),
        Point3::new(104.0, 0.0, 5.0),
        Point3::new(105.0, 0.0, 5.0),
    ];
    for (index, point) in far_points.iter().enumerate() {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
            max_pose_prior_translation_error: None,
            ..TrackingConfig::default()
        },
    );

    let first = tracker.track_frame(&frame, &map);
    let second = tracker.track_frame(&frame, &map);

    assert!(first.localization.success);
    assert_eq!(first.map_landmark_count, 12);
    assert!(first.pose_prior.is_none());
    assert!(!first.used_pose_prior);
    assert_eq!(first.localization.candidate_landmark_count, 12);
    assert!(second.localization.success);
    assert_eq!(second.map_landmark_count, 12);
    assert!(second.pose_prior.is_some());
    assert!(second.used_pose_prior);
    assert_eq!(second.localization.candidate_landmark_count, 6);
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
}

#[test]
fn tracker_quality_gate_rejects_large_jump_from_pose_prior() {
    let (map, first_frame) = build_map_and_frame(10, 1);
    let camera = map.cameras.get(&first_frame.camera_id).unwrap();
    let shifted_pose = pose_with_identity_rotation_at_center(Vector3::new(0.5, 0.0, 0.0));
    let mut shifted_frame = Frame::new(11, camera.id);
    let mut landmarks = map.landmarks.values().collect::<Vec<_>>();
    landmarks.sort_by_key(|landmark| landmark.id);

    for landmark in landmarks {
        shifted_frame.keypoints.push(
            camera
                .project(&shifted_pose.transform_world_point(&landmark.position))
                .unwrap(),
        );
        shifted_frame
            .descriptors
            .push(landmark.descriptor.clone().unwrap());
    }

    let mut tracker = Tracker::new(
        LocalizationPipeline::default(),
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
            max_pose_prior_translation_error: Some(0.1),
            ..TrackingConfig::default()
        },
    );

    let first = tracker.track_frame(&first_frame, &map);
    let second = tracker.track_frame(&shifted_frame, &map);

    assert!(first.localization.success);
    assert_eq!(first.event, TrackingEvent::Initialized);
    assert!(second.pose_prior.is_some());
    assert!(second.used_pose_prior);
    assert!(!second.localization.success);
    assert_eq!(second.event, TrackingEvent::TrackingFailed);
    assert_eq!(second.state, TrackingState::Tracking);
    assert_eq!(second.successive_failures, 1);
    assert!(matches!(
        second.tracking_failure_reason,
        Some(TrackingFailureReason::PosePriorTranslationErrorExceeded {
            translation_error,
            max_translation_error: 0.1,
        }) if translation_error > 0.1
    ));
    assert_eq!(tracker.last_successful_frame_id(), Some(10));
    assert_eq!(tracker.stats().successful_frame_count, 1);
    assert_eq!(tracker.stats().failed_frame_count, 1);
    assert_eq!(tracker.stats().pose_prior_used_count, 1);
    assert_eq!(tracker.stats().pose_prior_usage_rate(), 0.5);
    assert_eq!(tracker.stats().tracking_quality_gate_failure_count, 1);
    assert_eq!(tracker.stats().total_inlier_count, 12);
    assert_eq!(tracker.stats().total_correspondence_count, 12);
    assert_eq!(tracker.stats().overall_inlier_ratio(), 1.0);
}

#[test]
fn tracker_quality_gate_rejects_too_few_inliers() {
    let (_map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(
        FixedFrameLocalizer {
            result: successful_localization_result(3, 3, 0.0),
        },
        TrackingConfig {
            min_inliers: 4,
            ..TrackingConfig::default()
        },
    );

    let result = tracker.track_frame(&frame, &VisualMap::new());

    assert!(!result.localization.success);
    assert_eq!(result.event, TrackingEvent::TrackingFailed);
    assert!(matches!(
        result.tracking_failure_reason,
        Some(TrackingFailureReason::InsufficientInliers {
            inlier_count: 3,
            min_inliers: 4,
        })
    ));
    assert_eq!(tracker.stats().tracking_quality_gate_failure_count, 1);
    assert_eq!(tracker.stats().failed_frame_count, 1);
}

#[test]
fn tracker_quality_gate_rejects_low_inlier_ratio() {
    let (_map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(
        FixedFrameLocalizer {
            result: successful_localization_result(3, 6, 0.0),
        },
        TrackingConfig {
            min_inlier_ratio: 0.75,
            ..TrackingConfig::default()
        },
    );

    let result = tracker.track_frame(&frame, &VisualMap::new());

    assert!(!result.localization.success);
    assert!(matches!(
        result.tracking_failure_reason,
        Some(TrackingFailureReason::InlierRatioTooLow {
            inlier_ratio: 0.5,
            min_inlier_ratio: 0.75,
        })
    ));
    assert_eq!(tracker.stats().tracking_quality_gate_failure_count, 1);
}

#[test]
fn tracker_quality_gate_rejects_high_reprojection_error() {
    let (_map, frame) = build_map_and_frame(10, 1);
    let mut tracker = Tracker::new(
        FixedFrameLocalizer {
            result: successful_localization_result(6, 6, 2.5),
        },
        TrackingConfig {
            max_mean_reprojection_error: Some(1.0),
            ..TrackingConfig::default()
        },
    );

    let result = tracker.track_frame(&frame, &VisualMap::new());

    assert!(!result.localization.success);
    assert!(matches!(
        result.tracking_failure_reason,
        Some(TrackingFailureReason::MeanReprojectionErrorTooHigh {
            reprojection_error: 2.5,
            max_reprojection_error: 1.0,
        })
    ));
    assert_eq!(tracker.stats().tracking_quality_gate_failure_count, 1);
}

#[test]
fn tracker_accepts_custom_motion_model_for_pose_prior() {
    let (mut map, frame) = build_map_and_frame(10, 1);
    let far_points = [
        Point3::new(100.0, 0.0, 5.0),
        Point3::new(101.0, 0.0, 5.0),
        Point3::new(102.0, 0.0, 5.0),
        Point3::new(103.0, 0.0, 5.0),
        Point3::new(104.0, 0.0, 5.0),
        Point3::new(105.0, 0.0, 5.0),
    ];
    for (index, point) in far_points.iter().enumerate() {
        let landmark_id = index as u64 + 100;
        let mut landmark = Landmark::new(landmark_id, *point);
        landmark.descriptor = Some(vec![index as f32 + 100.0, 9.0]);
        map.landmarks.insert(landmark_id, landmark);
    }

    let far_prior =
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-100.0, 0.0, 0.0));
    let mut tracker = Tracker::with_motion_model(
        LocalizationPipeline::default(),
        FixedPoseMotionModel { pose: far_prior },
        TrackingConfig {
            min_successive_failures_to_lost: 2,
            last_pose_candidate_radius: Some(8.0),
            max_pose_prior_translation_error: None,
            ..TrackingConfig::default()
        },
    );

    let result = tracker.track_frame(&frame, &map);

    assert!(!result.localization.success);
    assert_eq!(result.map_landmark_count, 12);
    assert!(result.pose_prior.is_some());
    assert!(result.used_pose_prior);
    assert_eq!(result.localization.candidate_landmark_count, 6);
    assert_eq!(tracker.last_successful_frame_id(), None);
}
