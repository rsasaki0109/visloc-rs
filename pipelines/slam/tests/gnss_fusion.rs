//! End-to-end tests of the joint GNSS + visual-odometry estimator on
//! deterministic synthetic drives (drifting VO, noisy GNSS with a dropout and
//! multipath outliers).

use nalgebra::{Point3, Vector3};
use visloc_core::geometry::Pose;
use visloc_fusion::{GnssMeasurement, MeasurementBuffer, Timed, TimedPose, Timestamp};
use visloc_slam::gnss_fusion::{
    aligned_position_rmse, fuse_gnss_with_visual_odometry, position_rmse, transform_trajectory,
    FrameGnssStatus, GnssFusionError, GnssRobustMode, GnssVoFusion, GnssVoFusionConfig,
    GnssVoFusionResult,
};
use visloc_slam::gnss_synthetic::{
    generate_gnss_vo_scenario, SyntheticGnssVoConfig, SyntheticGnssVoScenario, SyntheticOutlier,
};

/// A 60 s, 5 Hz drive (300 frames, one lap) with an 8 s dropout and three
/// multipath bursts — small enough for unoptimized test builds.
fn short_drive(base: SyntheticGnssVoConfig) -> SyntheticGnssVoConfig {
    SyntheticGnssVoConfig {
        frame_count: 300,
        lap_seconds: 60.0,
        dropout: Some((25.0, 8.0)),
        outliers: vec![
            SyntheticOutlier {
                time_seconds: 11.0,
                burst_length: 1,
                offset_enu: Vector3::new(20.0, -12.0, 4.0),
            },
            SyntheticOutlier {
                time_seconds: 19.6,
                burst_length: 2,
                offset_enu: Vector3::new(-15.0, 18.0, -6.0),
            },
            SyntheticOutlier {
                time_seconds: 47.2,
                burst_length: 1,
                offset_enu: Vector3::new(22.0, 10.0, 8.0),
            },
        ],
        vo_yaw_drift_per_frame: 0.02_f64.to_radians(),
        ..base
    }
}

fn fuse(scenario: &SyntheticGnssVoScenario, mut config: GnssVoFusionConfig) -> GnssVoFusionResult {
    config.lever_arm = scenario.lever_arm;
    fuse_gnss_with_visual_odometry(&scenario.vo, &scenario.gnss, &config).expect("fusion")
}

/// Frames within one GNSS period of a corrupted fix (they may interpolate it).
fn near_outlier_mask(scenario: &SyntheticGnssVoScenario, result: &GnssVoFusionResult) -> Vec<bool> {
    result
        .frames
        .iter()
        .map(|frame| {
            scenario.outlier_fix_timestamps.iter().any(|t| {
                (t.as_nanoseconds() - frame.timestamp.as_nanoseconds()).abs() < 500_000_000
            })
        })
        .collect()
}

fn max_error_where(estimate: &[TimedPose], truth: &[TimedPose], mask: &[bool]) -> f64 {
    estimate
        .iter()
        .zip(truth)
        .zip(mask)
        .filter(|(_, selected)| **selected)
        .map(|((a, b), _)| (a.value.camera_center_world() - b.value.camera_center_world()).norm())
        .fold(0.0, f64::max)
}

#[test]
fn metric_fusion_beats_vo_rejects_outliers_and_bridges_dropout() {
    let scenario = generate_gnss_vo_scenario(&short_drive(SyntheticGnssVoConfig::default()));
    let truth = &scenario.ground_truth_enu;
    let result = fuse(&scenario, GnssVoFusionConfig::metric_gravity_aligned());

    let vo_bootstrap = transform_trajectory(&result.bootstrap.alignment, &scenario.vo);
    let vo_bootstrap_ate = position_rmse(&vo_bootstrap, truth).unwrap();
    let vo_oracle_ate = aligned_position_rmse(&scenario.vo, truth, false).unwrap();
    let fused_ate = position_rmse(&result.trajectory_enu, truth).unwrap();
    eprintln!("metric: VO bootstrap {vo_bootstrap_ate:.3} / oracle {vo_oracle_ate:.3} -> fused {fused_ate:.3}");
    assert!(
        fused_ate < 0.25 * vo_bootstrap_ate && fused_ate < 0.5 * vo_oracle_ate,
        "fused {fused_ate} vs VO bootstrap {vo_bootstrap_ate} / oracle {vo_oracle_ate}"
    );
    assert!(fused_ate < 1.2, "fused ATE {fused_ate}");

    // Alignment recovered (the true alignment is defined at frame 0).
    let yaw_error = (result.alignment.yaw() - scenario.true_alignment.yaw()).abs();
    assert!(
        yaw_error.to_degrees() < 1.5,
        "yaw error {}",
        yaw_error.to_degrees()
    );
    assert_eq!(result.alignment.scale, 1.0);

    // Every frame that consumed a corrupted fix is flagged; nothing else is.
    let near = near_outlier_mask(&scenario, &result);
    for (frame, near) in result.frames.iter().zip(&near) {
        if !near {
            assert!(
                !matches!(frame.gnss, FrameGnssStatus::Outlier { .. }),
                "false outlier at {:?}",
                frame.timestamp
            );
        }
    }
    assert!(result.summary.frames_with_outlier_fix >= scenario.outlier_fix_timestamps.len());
    // Outliers (15-25 m jumps) must not pull the trajectory.
    let near_error = max_error_where(&result.trajectory_enu, truth, &near);
    eprintln!("metric: max error near outliers {near_error:.3}");
    assert!(near_error < 2.5, "max error near outliers {near_error}");

    // The dropout is bridged by VO: those frames have no fix but stay close.
    let dropout: Vec<bool> = result
        .frames
        .iter()
        .map(|frame| matches!(frame.gnss, FrameGnssStatus::NoFix))
        .collect();
    let dropout_frames = dropout.iter().filter(|d| **d).count();
    assert!(
        (30..=55).contains(&dropout_frames),
        "dropout frames {dropout_frames}"
    );
    let dropout_error = max_error_where(&result.trajectory_enu, truth, &dropout);
    eprintln!("metric: {dropout_frames} dropout frames, max error {dropout_error:.3}");
    assert!(dropout_error < 4.0, "max dropout error {dropout_error}");
}

#[test]
fn robust_solve_resists_outliers_that_pull_least_squares() {
    let scenario = generate_gnss_vo_scenario(&short_drive(SyntheticGnssVoConfig::default()));
    let truth = &scenario.ground_truth_enu;
    let robust = fuse(&scenario, GnssVoFusionConfig::default());
    let mut plain_config = GnssVoFusionConfig::default();
    plain_config.optimizer.robust = GnssRobustMode::None;
    plain_config.optimizer.refine_without_outliers = false;
    let plain = fuse(&scenario, plain_config);

    let near = near_outlier_mask(&scenario, &robust);
    let robust_error = max_error_where(&robust.trajectory_enu, truth, &near);
    let plain_error = max_error_where(&plain.trajectory_enu, truth, &near);
    eprintln!("outliers: robust {robust_error:.3} vs least squares {plain_error:.3}");
    assert!(
        robust_error < 0.6 * plain_error,
        "robust {robust_error} vs least squares {plain_error}"
    );
}

#[test]
fn monocular_fusion_recovers_scale_and_beats_sim3_aligned_vo() {
    let scenario = generate_gnss_vo_scenario(&short_drive(SyntheticGnssVoConfig::monocular()));
    let truth = &scenario.ground_truth_enu;
    let result = fuse(&scenario, GnssVoFusionConfig::monocular());

    let vo_oracle_ate = aligned_position_rmse(&scenario.vo, truth, true).unwrap();
    let fused_ate = position_rmse(&result.trajectory_enu, truth).unwrap();
    assert!(
        fused_ate < 0.5 * vo_oracle_ate && fused_ate < 1.2,
        "fused {fused_ate} vs Sim(3)-aligned VO {vo_oracle_ate}"
    );
    let scale_error = result.alignment.scale / scenario.true_alignment.scale - 1.0;
    eprintln!("monocular: Sim(3) VO {vo_oracle_ate:.3} -> fused {fused_ate:.3}, scale error {scale_error:.4}");
    assert!(scale_error.abs() < 0.03, "scale error {scale_error}");
    let rotation_error = result
        .alignment
        .rotation
        .angle_to(&scenario.true_alignment.rotation);
    assert!(
        rotation_error.to_degrees() < 2.0,
        "rotation error {}",
        rotation_error.to_degrees()
    );
}

#[test]
fn sliding_window_fusion_tracks_ground_truth() {
    let scenario = generate_gnss_vo_scenario(&short_drive(SyntheticGnssVoConfig::default()));
    let truth = &scenario.ground_truth_enu;
    let config = GnssVoFusionConfig {
        window: Some(100),
        window_update_stride: 25,
        ..GnssVoFusionConfig::default()
    };
    let result = fuse(&scenario, config);
    assert!(result.summary.optimizer_runs > 3);
    let vo_bootstrap = transform_trajectory(&result.bootstrap.alignment, &scenario.vo);
    let vo_bootstrap_ate = position_rmse(&vo_bootstrap, truth).unwrap();
    let fused_ate = position_rmse(&result.trajectory_enu, truth).unwrap();
    eprintln!("window: VO {vo_bootstrap_ate:.3} -> fused {fused_ate:.3}");
    assert!(
        fused_ate < 0.35 * vo_bootstrap_ate && fused_ate < 1.6,
        "window fused {fused_ate} vs VO {vo_bootstrap_ate}"
    );
}

#[test]
fn fusion_reports_unobservable_alignment_and_bad_timestamps() {
    let poses: Vec<TimedPose> = (0..20)
        .map(|i| {
            Timed::new(
                Timestamp::from_nanoseconds(i * 100_000_000),
                Pose::from_world_to_camera(
                    Default::default(),
                    Vector3::new(-0.1 * i as f64, 0.0, 0.0),
                ),
            )
        })
        .collect();
    // No GNSS at all.
    let empty = MeasurementBuffer::new();
    assert_eq!(
        fuse_gnss_with_visual_odometry(&poses, &empty, &GnssVoFusionConfig::default()),
        Err(GnssFusionError::AlignmentNotObservable)
    );
    // GNSS present but only 2 m of motion: yaw is unobservable.
    let gnss = MeasurementBuffer::from_measurements(poses.iter().map(|pose| {
        GnssMeasurement::new(
            pose.timestamp,
            Point3::from(pose.value.camera_center_world().coords),
        )
        .with_accuracy(Some(0.5), Some(1.0))
    }));
    assert_eq!(
        fuse_gnss_with_visual_odometry(&poses, &gnss, &GnssVoFusionConfig::default()),
        Err(GnssFusionError::AlignmentNotObservable)
    );

    let mut fusion = GnssVoFusion::new(GnssVoFusionConfig::default());
    fusion.push_vo_pose(&poses[3]).unwrap();
    assert!(matches!(
        fusion.push_vo_pose(&poses[2]),
        Err(GnssFusionError::NonMonotonicTimestamp { .. })
    ));
    assert_eq!(fusion.optimize(), Err(GnssFusionError::TooFewFrames));
}
