//! Bundle adjustment demo: synthetic scene → drift → BA recovery.
//!
//! Builds a 5-keyframe / 30-landmark synthetic pinhole scene where every
//! landmark is observed from every keyframe, projects measurements through
//! truth, then perturbs the last three keyframes (translation + small yaw)
//! and a third of the landmarks (3D shifts). Fixes the first two keyframes
//! for gauge, runs `BundleAdjustment::optimize`, and prints the iteration
//! trace plus per-keyframe pose error and per-landmark recovery against
//! truth.
//!
//! Asset-free, runs in CI.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::core::types::Camera;
use visloc_rs::{BaConfig, BaObservation, BundleAdjustment, LinearSolver};

fn pose_with_yaw(camera_center: Vector3<f64>, yaw_rad: f64) -> Pose {
    let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_rad);
    let t = -(r.transform_vector(&camera_center));
    Pose::from_world_to_camera(r, t)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);

    // Truth: 5 keyframes along a small forward arc, each yawing slightly.
    let truth_poses: Vec<(u64, Pose)> = (0..5u64)
        .map(|i| {
            let center = Vector3::new(0.4 * i as f64, 0.0, 0.05 * i as f64);
            let yaw = 0.04 * i as f64;
            (10 + i * 10, pose_with_yaw(center, yaw))
        })
        .collect();

    // Truth: 30 landmarks on a 5x6 grid 4–8 m ahead, with depth jitter.
    let mut truth_landmarks: Vec<(u64, Point3<f64>)> = Vec::new();
    for r in 0..5 {
        for c in 0..6 {
            let id = (r * 6 + c) as u64 + 1;
            let x = -1.5 + 0.6 * c as f64;
            let y = -1.0 + 0.5 * r as f64;
            let z = 5.0 + 0.4 * ((r * 6 + c) as f64).sin();
            truth_landmarks.push((id, Point3::new(x, y, z)));
        }
    }

    // Build a truth-consistent BA: every landmark observed from every keyframe.
    let mut truth_ba = BundleAdjustment::new(camera.clone());
    for (id, pose) in &truth_poses {
        truth_ba.add_pose(*id, pose.clone());
    }
    for (id, point) in &truth_landmarks {
        truth_ba.add_landmark(*id, *point);
    }
    let mut observation_count = 0usize;
    for (kf_id, pose) in &truth_poses {
        for (lm_id, point) in &truth_landmarks {
            let xc = pose.transform_world_point(point);
            if xc.z <= 0.0 {
                continue;
            }
            if let Some(uv) = camera.project(&xc) {
                truth_ba.add_observation(BaObservation {
                    keyframe_id: *kf_id,
                    landmark_id: *lm_id,
                    xy: uv,
                });
                observation_count += 1;
            }
        }
    }
    println!(
        "scene keyframes={} landmarks={} observations={} cost_at_truth={:.3e}",
        truth_ba.poses.len(),
        truth_ba.landmarks.len(),
        observation_count,
        truth_ba.cost(),
    );

    // Inject pose drift on the last three keyframes (translation + yaw).
    let mut drifted_ba = truth_ba.clone();
    drifted_ba.fix_pose(truth_poses[0].0);
    drifted_ba.fix_pose(truth_poses[1].0);
    let pose_drifts = [
        (truth_poses[2].0, Vector3::new(0.04, -0.01, 0.02), 0.02_f64),
        (truth_poses[3].0, Vector3::new(-0.03, 0.02, -0.01), -0.025),
        (truth_poses[4].0, Vector3::new(0.05, -0.02, 0.03), 0.03),
    ];
    for (id, delta_t, delta_yaw) in &pose_drifts {
        let truth = drifted_ba.poses[id].clone();
        let drifted_center = truth.camera_center_world().coords + delta_t;
        let drifted_yaw = {
            // Recover truth yaw via axis-angle decomposition (we only ever
            // build yaw-only rotations in this demo).
            let truth_axis_angle = truth.world_to_camera.rotation.scaled_axis();
            truth_axis_angle.y + delta_yaw
        };
        drifted_ba.add_pose(*id, pose_with_yaw(drifted_center, drifted_yaw));
    }

    // Inject landmark drift on a third of the landmarks (3D shifts).
    for (i, (id, truth_point)) in truth_landmarks.iter().enumerate() {
        if i % 3 != 0 {
            continue;
        }
        let delta = Vector3::new(
            0.03 * (i as f64 * 0.7).sin(),
            0.04 * (i as f64 * 1.1).cos(),
            0.05 * (i as f64 * 0.3).sin(),
        );
        drifted_ba.add_landmark(*id, *truth_point + delta);
    }
    println!("ba_drift cost_before={:.3}", drifted_ba.cost());

    // Run BA. LM damping + sparse Cholesky on the reduced camera system.
    let result = drifted_ba.optimize(&BaConfig {
        linear_solver: LinearSolver::Sparse,
        max_iterations: 30,
        ..BaConfig::default()
    })?;
    println!(
        "ba_done initial_cost={:.3} final_cost={:.3e} iterations={} converged={}",
        result.initial_cost,
        result.final_cost,
        result.iterations.len(),
        result.converged,
    );
    for stats in &result.iterations {
        println!(
            "  iter={} cost_before={:.3e} cost_after={:.3e} max_pose_step={:.3e} max_lm_step={:.3e} lambda={:.3e} accepted={}",
            stats.iteration,
            stats.cost_before,
            stats.cost_after,
            stats.max_pose_step,
            stats.max_landmark_step,
            stats.lambda,
            stats.step_accepted,
        );
    }

    // Per-pose error after BA.
    let mut max_translation_err: f64 = 0.0;
    let mut max_rotation_err: f64 = 0.0;
    for (id, truth) in &truth_poses {
        let recovered = &drifted_ba.poses[id];
        let translation_err =
            (recovered.camera_center_world() - truth.camera_center_world()).norm();
        let rotation_err = recovered
            .world_to_camera
            .rotation
            .rotation_to(&truth.world_to_camera.rotation)
            .angle();
        max_translation_err = max_translation_err.max(translation_err);
        max_rotation_err = max_rotation_err.max(rotation_err);
        println!(
            "  pose keyframe={} t_err={:.3e} rot_err={:.3e}",
            id, translation_err, rotation_err,
        );
    }
    let mut max_landmark_err: f64 = 0.0;
    for (id, truth_point) in &truth_landmarks {
        let recovered = drifted_ba.landmarks[id];
        let err = (recovered - truth_point).norm();
        if err > max_landmark_err {
            max_landmark_err = err;
        }
    }
    println!(
        "ba_summary max_pose_t_err={:.3e} max_pose_rot_err={:.3e} max_landmark_err={:.3e}",
        max_translation_err, max_rotation_err, max_landmark_err,
    );
    Ok(())
}
