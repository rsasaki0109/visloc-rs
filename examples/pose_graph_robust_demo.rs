//! Robust pose-graph SE(3) optimization demo.
//!
//! Builds a three-keyframe synthetic loop and adds two loop-closure
//! constraints: one consistent with the truth path (the inlier) and one with
//! a deliberately wrong relative pose (the outlier). The example then runs:
//!
//! 1. Pure Gauss-Newton (no robust kernel, no LM damping) — the outlier pulls
//!    the third keyframe several decimeters off truth.
//! 2. Levenberg-Marquardt with a Huber kernel — the outlier is down-weighted
//!    and the third keyframe converges close to truth.
//!
//! The demo is meant to make the robustness benefit of LM + Huber visible on
//! a runnable, deterministic, in-memory fixture.

use nalgebra::{UnitQuaternion, Vector3};
use visloc_rs::core::geometry::Pose;
use visloc_rs::{
    relative_world_to_camera, LoopClosureConstraint, PoseGraph, PoseGraphSe3Config, RobustKernel,
};

fn pose_at(camera_center: Vector3<f64>) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -camera_center)
}

fn main() {
    let truth_10 = pose_at(Vector3::new(0.0, 0.0, 0.0));
    let truth_20 = pose_at(Vector3::new(1.0, 0.0, 0.0));
    let truth_30 = pose_at(Vector3::new(0.5, 0.0, 0.5));

    let bogus_pose = pose_at(Vector3::new(5.0, 0.0, 5.0));

    let mut graph = PoseGraph::new();
    graph.add_pose(10, truth_10.clone());
    graph.add_pose(20, truth_20.clone());
    graph.add_pose(30, truth_30.clone());
    graph.anchor(10);
    graph.add_sequential_edge(10, 20, relative_world_to_camera(&truth_10, &truth_20));
    graph.add_sequential_edge(20, 30, relative_world_to_camera(&truth_20, &truth_30));
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 10,
        to_keyframe_id: 30,
        relative_pose: relative_world_to_camera(&truth_10, &truth_30),
        inlier_count: 24,
        inlier_ratio: 1.0,
        mean_sampson_error: 1.0e-4,
        score: 240.0,
    });
    graph.add_loop_closure_constraint(&LoopClosureConstraint {
        from_keyframe_id: 20,
        to_keyframe_id: 30,
        relative_pose: relative_world_to_camera(&truth_20, &bogus_pose),
        inlier_count: 8,
        inlier_ratio: 0.6,
        mean_sampson_error: 1.0e-2,
        score: 30.0,
    });

    println!("# Outlier-prone three-keyframe loop");
    println!(
        "# initial se3_cost = {:.6} (with truth poses still in graph)",
        graph.se3_cost()
    );

    println!("\n## Pure Gauss-Newton (no robust kernel, no LM)");
    let mut graph_gn = graph.clone();
    let result_gn = graph_gn
        .optimize_se3_iterative(&PoseGraphSe3Config::default())
        .expect("GN must succeed");
    print_result("GN", &graph_gn, &result_gn, &truth_30);

    println!("\n## Levenberg-Marquardt + Huber kernel");
    let robust_config = PoseGraphSe3Config {
        robust_kernel: RobustKernel::Huber { delta: 0.05 },
        initial_lambda: Some(1.0e-4),
        max_iterations: 50,
        ..PoseGraphSe3Config::default()
    };
    let mut graph_robust = graph.clone();
    let result_robust = graph_robust
        .optimize_se3_iterative(&robust_config)
        .expect("LM + Huber must succeed");
    print_result("LM+Huber", &graph_robust, &result_robust, &truth_30);
}

fn print_result(
    label: &str,
    graph: &PoseGraph,
    result: &visloc_rs::PoseGraphSe3Result,
    truth_30: &Pose,
) {
    println!(
        "{label}: initial_cost={:.6} final_cost={:.6} iterations={} converged={}",
        result.initial_cost,
        result.final_cost,
        result.iterations.len(),
        result.converged,
    );
    for stats in &result.iterations {
        println!(
            "  iter={} cost_before={:.6} cost_after={:.6} max_step={:.6} lambda={:.2e} accepted={}",
            stats.iteration,
            stats.cost_before,
            stats.cost_after,
            stats.max_step_norm,
            stats.lambda,
            stats.step_accepted,
        );
    }
    let center_30 = graph.poses[&30].camera_center_world();
    let truth_center = truth_30.camera_center_world();
    let drift = (center_30 - truth_center).norm();
    println!(
        "  KF30 center=[{:.3}, {:.3}, {:.3}] truth=[{:.3}, {:.3}, {:.3}] drift={:.4}",
        center_30.x,
        center_30.y,
        center_30.z,
        truth_center.x,
        truth_center.y,
        truth_center.z,
        drift,
    );
}
