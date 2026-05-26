//! Integration tests for outlier-robust bundle adjustment via Graduated
//! Non-Convexity ([`BundleAdjustment::optimize_gnc`]).
//!
//! Each test builds a rigid, fully-observed synthetic pinhole scene whose
//! inlier reprojections are exact at truth, then corrupts a handful of
//! observations (a wrong feature match shifts the measured pixel far from
//! the projection). This is a pipeline-correctness fixture — controlled
//! ground truth with injected outliers — not a performance benchmark; the
//! real-data robustness numbers live in the `pgo`/`ba` g2o demos.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_slam::gnc::{GncConfig, GncKernel};
use visloc_slam::{BaConfig, BaError, BaObservation, BundleAdjustment, LinearSolver};

fn pinhole() -> Camera {
    Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
}

fn pose_with_yaw(center: Vector3<f64>, yaw_rad: f64) -> Pose {
    let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_rad);
    let t = -(r.transform_vector(&center));
    Pose::from_world_to_camera(r, t)
}

/// Five keyframes along a short forward arc, each yawing slightly.
fn truth_poses() -> Vec<(u64, Pose)> {
    (0..5u64)
        .map(|i| {
            let center = Vector3::new(0.3 * i as f64, 0.0, 0.04 * i as f64);
            (10 + i, pose_with_yaw(center, 0.015 * i as f64))
        })
        .collect()
}

/// A 3×4 grid of landmarks 4.5–7 m ahead, all inside the image from every
/// keyframe.
fn truth_landmarks() -> Vec<(u64, Point3<f64>)> {
    let mut v = Vec::new();
    let mut id = 1u64;
    for r in 0..3 {
        for c in 0..4 {
            let x = -1.2 + 0.8 * c as f64;
            let y = -0.8 + 0.8 * r as f64;
            let z = 4.5 + 0.4 * (r + c) as f64;
            v.push((id, Point3::new(x, y, z)));
            id += 1;
        }
    }
    v
}

/// Build a BA sitting exactly at truth (every inlier residual is zero),
/// every landmark observed by every pose, with the two **end** poses
/// (`10` and `14`) fixed for gauge so every free pose is interior and
/// over-determined. Observations whose row-major insertion index is in
/// `outliers` get their measured pixel shifted by `shift_px` (a wrong
/// correspondence). Returns the BA and the total observation count.
fn build_with_outliers(outliers: &[usize], shift_px: f64) -> (BundleAdjustment, usize) {
    let camera = pinhole();
    let mut ba = BundleAdjustment::new(camera.clone());
    let poses = truth_poses();
    let landmarks = truth_landmarks();
    for (id, pose) in &poses {
        ba.add_pose(*id, pose.clone());
    }
    for (id, point) in &landmarks {
        ba.add_landmark(*id, *point);
    }
    let mut idx = 0usize;
    for (kf_id, pose) in &poses {
        for (lm_id, point) in &landmarks {
            let xc = pose.transform_world_point(point);
            let mut uv = camera.project(&xc).expect("landmark in front of camera");
            if outliers.contains(&idx) {
                uv.x += shift_px;
                uv.y -= shift_px;
            }
            ba.add_observation(BaObservation {
                keyframe_id: *kf_id,
                landmark_id: *lm_id,
                xy: uv,
            });
            idx += 1;
        }
    }
    ba.fix_pose(10);
    ba.fix_pose(14);
    (ba, idx)
}

fn ba_config() -> BaConfig {
    BaConfig {
        linear_solver: LinearSolver::Dense,
        ..BaConfig::default()
    }
}

/// `c = 5 px` cleanly separates the ~0 px inlier residuals from the
/// 70–90 px injected outliers.
fn gnc_config(kernel: GncKernel) -> GncConfig {
    GncConfig {
        kernel,
        c: 5.0,
        ..GncConfig::default()
    }
}

fn max_pose_center_error(ba: &BundleAdjustment) -> f64 {
    truth_poses()
        .iter()
        .map(|(id, truth)| {
            (ba.poses[id].camera_center_world() - truth.camera_center_world()).norm()
        })
        .fold(0.0_f64, f64::max)
}

#[test]
fn gnc_geman_mcclure_identifies_every_outlier() {
    let outliers = [7usize, 18, 33, 41, 52];
    let (mut ba, n) = build_with_outliers(&outliers, 70.0);

    let res = ba
        .optimize_gnc(&ba_config(), &gnc_config(GncKernel::GemanMcClure))
        .expect("GNC bundle adjustment must succeed");

    assert_eq!(res.observation_count, n);
    assert!(res.converged, "GNC should reach its terminal μ level");

    // Every injected outlier is driven to a vanishing weight (perfect
    // recall), separated from the lightest inlier by orders of magnitude.
    let max_outlier_w = outliers
        .iter()
        .map(|&i| res.observation_weights[i])
        .fold(0.0_f64, f64::max);
    let min_inlier_w = (0..n)
        .filter(|i| !outliers.contains(i))
        .map(|i| res.observation_weights[i])
        .fold(f64::INFINITY, f64::min);
    assert!(
        max_outlier_w < 1e-3,
        "all outliers should be rejected, worst weight = {max_outlier_w}"
    );
    assert!(
        min_inlier_w > 1000.0 * max_outlier_w,
        "outliers ({max_outlier_w}) should sit orders of magnitude below the \
         lightest inlier ({min_inlier_w})"
    );

    // The inlier-only cost is far below the outlier-laden cost at the input
    // estimate. (The smooth Geman-McClure surrogate also lightly shrinks
    // inlier weights, which loosens the weakly-observable monocular depth
    // direction; the hard 0/1 truncated-least-squares kernel is what
    // additionally recovers truth exactly — see the TLS tests.)
    assert!(
        res.inlier_cost < 0.05 * res.initial_cost,
        "inlier cost {} should be far below initial cost {}",
        res.inlier_cost,
        res.initial_cost
    );
}

#[test]
fn gnc_recovers_truth_where_plain_least_squares_is_dragged_off() {
    let outliers = [7usize, 18, 33, 41, 52];
    let shift = 70.0;

    let (mut l2, _) = build_with_outliers(&outliers, shift);
    l2.optimize(&ba_config()).expect("plain BA must succeed");
    let l2_err = max_pose_center_error(&l2);

    let (mut gnc, _) = build_with_outliers(&outliers, shift);
    gnc.optimize_gnc(&ba_config(), &gnc_config(GncKernel::TruncatedLeastSquares))
        .expect("GNC bundle adjustment must succeed");
    let gnc_err = max_pose_center_error(&gnc);

    assert!(
        gnc_err < l2_err,
        "GNC error {gnc_err} should beat plain L2 error {l2_err}"
    );
    assert!(
        gnc_err < 1e-3,
        "truncated-least-squares GNC should recover truth near-exactly, got {gnc_err}"
    );
}

#[test]
fn gnc_truncated_least_squares_hard_rejects_outliers() {
    let outliers = [25usize, 38, 50];
    let (mut ba, n) = build_with_outliers(&outliers, 90.0);

    let res = ba
        .optimize_gnc(&ba_config(), &gnc_config(GncKernel::TruncatedLeastSquares))
        .expect("GNC bundle adjustment must succeed");

    // TLS gives a hard 0/1 verdict: outliers reach machine-zero weight and
    // inliers stay at exactly one.
    for &i in &outliers {
        assert!(
            res.observation_weights[i] < 1e-6,
            "TLS outlier obs {i} weight = {}",
            res.observation_weights[i]
        );
    }
    let inliers_exactly_one = (0..n)
        .filter(|i| !outliers.contains(i))
        .all(|i| (res.observation_weights[i] - 1.0).abs() < 1e-9);
    assert!(inliers_exactly_one, "TLS inliers should keep weight 1.0");

    assert!(max_pose_center_error(&ba) < 1e-4);
}

#[test]
fn gnc_on_clean_scene_keeps_every_observation() {
    let (mut ba, n) = build_with_outliers(&[], 0.0);

    let res = ba
        .optimize_gnc(&ba_config(), &gnc_config(GncKernel::TruncatedLeastSquares))
        .expect("GNC bundle adjustment must succeed");

    assert_eq!(res.inlier_count(0.5), n);
    assert_eq!(res.outlier_count(0.5), 0);
    assert!(max_pose_center_error(&ba) < 1e-6);
}

#[test]
fn gnc_propagates_bundle_validation_errors() {
    let mut empty = BundleAdjustment::new(pinhole());
    assert_eq!(
        empty.optimize_gnc(&ba_config(), &gnc_config(GncKernel::GemanMcClure)),
        Err(BaError::NoPoses)
    );
}
