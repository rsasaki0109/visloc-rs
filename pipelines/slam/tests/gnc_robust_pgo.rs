//! Integration tests for Graduated Non-Convexity (GNC) outlier-robust SE(3)
//! pose-graph optimization (`PoseGraph::optimize_se3_gnc`).
//!
//! These use small hand-built graphs (pipeline-correctness unit tests): a
//! ground-truth chain whose sequential edges alone determine the trajectory
//! exactly, plus an injected *wrong* loop closure. Injecting controlled
//! outliers into an otherwise-consistent graph is the standard robust-PGO
//! evaluation protocol; the end-to-end accuracy contrast against real SE-Sync
//! `.g2o` graphs lives in the `pgo_g2o_benchmark` example.

use nalgebra::{Matrix6, UnitQuaternion, Vector3};
use visloc_core::geometry::{Pose, SE3};
use visloc_slam::gnc::{GncConfig, GncKernel};
use visloc_slam::{
    relative_world_to_camera, LinearSolver, PoseGraph, PoseGraphEdgeKind, PoseGraphSe3Config,
};

fn pose_with_yaw(camera_center: Vector3<f64>, yaw_rad: f64) -> Pose {
    let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw_rad);
    let translation = -(rotation.transform_vector(&camera_center));
    Pose::from_world_to_camera(rotation, translation)
}

/// Five-pose ground-truth path. The sequential edges between consecutive poses
/// fully determine the trajectory given the anchor at node 0, so rejecting any
/// extra (wrong) loop closure must recover these poses exactly.
fn truth_chain() -> Vec<(u64, Pose)> {
    vec![
        (0, pose_with_yaw(Vector3::new(0.0, 0.0, 0.0), 0.0)),
        (1, pose_with_yaw(Vector3::new(1.0, 0.0, 0.0), 0.3)),
        (2, pose_with_yaw(Vector3::new(2.0, 0.0, 0.3), 0.6)),
        (3, pose_with_yaw(Vector3::new(2.0, 0.0, 1.3), 1.2)),
        (4, pose_with_yaw(Vector3::new(1.0, 0.0, 1.3), 1.8)),
    ]
}

/// Build a graph seeded *at* the ground truth with exact unit-information
/// edges: the sequential chain `0-1-2-3-4` plus a correct loop closure `4-0`
/// that closes the cycle. The cycle makes the trajectory **rigid** (every node
/// is held by ≥ 2 consistent constraints), so an extra wrong chord stands out
/// as the clear inconsistency. Starting at truth isolates the injected
/// outlier's effect: the plain L2 solve is pulled off truth to compromise with
/// the bad edge, while GNC should reject it and stay put.
fn graph_at_truth(truth: &[(u64, Pose)]) -> PoseGraph {
    let mut graph = PoseGraph::new();
    for (id, pose) in truth {
        graph.add_pose(*id, pose.clone());
    }
    graph.anchor(truth[0].0);
    for pair in truth.windows(2) {
        let (from, a) = &pair[0];
        let (to, b) = &pair[1];
        graph.add_edge_with_information(
            *from,
            *to,
            relative_world_to_camera(a, b),
            PoseGraphEdgeKind::Sequential,
            Matrix6::identity(),
        );
    }
    // Correct loop closure last -> first, closing the cycle for rigidity.
    let last = truth.len() - 1;
    graph.add_edge_with_information(
        truth[last].0,
        truth[0].0,
        relative_world_to_camera(&truth[last].1, &truth[0].1),
        PoseGraphEdgeKind::LoopClosure,
        Matrix6::identity(),
    );
    graph
}

/// A grossly wrong loop closure on the interior chord `1 -> 3`: a relative pose
/// to a phantom node 3 displaced 3 m from the truth. Both endpoints are pinned
/// by the rigid consistent cycle, so this edge is the unambiguous outlier.
fn outlier_chord(truth: &[(u64, Pose)]) -> SE3 {
    let phantom_3 = pose_with_yaw(Vector3::new(5.0, 0.0, 1.3), 1.2);
    relative_world_to_camera(&truth[1].1, &phantom_3)
}

fn max_center_error(graph: &PoseGraph, truth: &[(u64, Pose)]) -> f64 {
    truth
        .iter()
        .map(|(id, pose)| {
            (graph.poses[id].camera_center_world() - pose.camera_center_world()).norm()
        })
        .fold(0.0_f64, f64::max)
}

fn no_chordal_config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        // Seed exactly at truth; do not let the chordal step (which trusts the
        // outlier) move the starting point, so the test isolates GNC.
        chordal_init: false,
        linear_solver: LinearSolver::Dense,
        ..PoseGraphSe3Config::default()
    }
}

#[test]
fn gnc_rejects_outlier_loop_closure_and_recovers_truth() {
    let truth = truth_chain();
    let mut graph = graph_at_truth(&truth);

    // Inject the wrong interior chord 1 -> 3 as the last edge.
    graph.add_edge_with_information(
        1,
        3,
        outlier_chord(&truth),
        PoseGraphEdgeKind::LoopClosure,
        Matrix6::identity(),
    );
    let outlier_idx = graph.edges.len() - 1;
    let edge_count = graph.edges.len();

    let config = no_chordal_config();
    let gnc = GncConfig {
        kernel: GncKernel::GemanMcClure,
        c: 0.3,
        anneal_factor: 1.4,
        max_outer: 100,
        inner_iterations: 10,
    };

    let result = graph
        .optimize_se3_gnc(&config, &gnc)
        .expect("GNC solve must succeed");

    assert_eq!(result.edge_count, edge_count);
    assert_eq!(result.variable_count, 4);
    assert!(result.converged, "GNC mu schedule should reach terminal");

    // The outlier is rejected (weight -> 0); every consistent edge is kept.
    assert!(
        result.edge_weights[outlier_idx] < 0.05,
        "outlier should be rejected, weight = {}",
        result.edge_weights[outlier_idx]
    );
    for (i, &w) in result.edge_weights.iter().enumerate() {
        if i != outlier_idx {
            assert!(w > 0.9, "inlier edge {i} should be kept, weight = {w}");
        }
    }
    assert_eq!(result.outlier_count(0.5), 1);
    assert_eq!(result.inlier_count(0.5), edge_count - 1);

    // With the outlier rejected, the rigid cycle recovers ground truth. The
    // Geman-McClure surrogate is smooth, so the outlier keeps a tiny residual
    // weight (it never goes hard to zero like TLS) and the recovery is
    // near-exact rather than exact — still ~10⁴× better than the 3 m outlier
    // and far better than the L2 fit below.
    let gnc_err = max_center_error(&graph, &truth);
    assert!(
        gnc_err < 1.0e-3,
        "GNC should recover truth, max center error = {gnc_err}"
    );
    assert!(
        result.inlier_cost < 1.0e-4,
        "inlier residual should be near zero, got {}",
        result.inlier_cost
    );

    // Contrast: a plain (non-robust) L2 solve on the same corrupted graph is
    // pulled away from truth by the outlier.
    let mut l2_graph = graph_at_truth(&truth);
    l2_graph.add_edge_with_information(
        1,
        3,
        outlier_chord(&truth),
        PoseGraphEdgeKind::LoopClosure,
        Matrix6::identity(),
    );
    l2_graph
        .optimize_se3_iterative(&config)
        .expect("L2 solve must succeed");
    let l2_err = max_center_error(&l2_graph, &truth);
    assert!(
        l2_err > 0.1,
        "plain L2 should be corrupted by the outlier, max center error = {l2_err}"
    );
    assert!(
        gnc_err < l2_err,
        "GNC ({gnc_err}) should be strictly more accurate than L2 ({l2_err})"
    );
}

#[test]
fn gnc_truncated_least_squares_also_rejects_outlier() {
    let truth = truth_chain();
    let mut graph = graph_at_truth(&truth);
    graph.add_edge_with_information(
        1,
        3,
        outlier_chord(&truth),
        PoseGraphEdgeKind::LoopClosure,
        Matrix6::identity(),
    );
    let outlier_idx = graph.edges.len() - 1;

    let gnc = GncConfig {
        kernel: GncKernel::TruncatedLeastSquares,
        c: 0.3,
        anneal_factor: 1.4,
        max_outer: 100,
        inner_iterations: 10,
    };
    let result = graph
        .optimize_se3_gnc(&no_chordal_config(), &gnc)
        .expect("GNC-TLS solve must succeed");

    // TLS gives a hard verdict: the outlier weight collapses to 0.
    assert!(result.edge_weights[outlier_idx] < 1.0e-6);
    assert_eq!(result.outlier_count(0.5), 1);
    assert!(max_center_error(&graph, &truth) < 1.0e-6);
}

#[test]
fn gnc_on_clean_graph_keeps_every_edge_and_recovers_truth() {
    // No outliers (the rigid consistent cycle only): GNC must not reject
    // anything and must leave the truth-seeded estimate at truth.
    let truth = truth_chain();
    let mut graph = graph_at_truth(&truth);

    let gnc = GncConfig {
        kernel: GncKernel::GemanMcClure,
        c: 0.3,
        anneal_factor: 1.4,
        max_outer: 100,
        inner_iterations: 10,
    };
    let result = graph
        .optimize_se3_gnc(&no_chordal_config(), &gnc)
        .expect("GNC solve must succeed");

    for (i, &w) in result.edge_weights.iter().enumerate() {
        assert!(
            w > 0.9,
            "edge {i} of a clean graph should stay, weight = {w}"
        );
    }
    assert_eq!(result.outlier_count(0.5), 0);
    assert!(max_center_error(&graph, &truth) < 1.0e-6);
    assert!(result.inlier_cost < 1.0e-9);
}

#[test]
fn gnc_propagates_graph_errors() {
    // No anchor set -> same validation errors as the plain optimizer.
    let mut graph = PoseGraph::new();
    graph.add_pose(0, pose_with_yaw(Vector3::zeros(), 0.0));
    graph.add_pose(1, pose_with_yaw(Vector3::new(1.0, 0.0, 0.0), 0.0));
    graph.add_edge_with_information(
        0,
        1,
        SE3::identity(),
        PoseGraphEdgeKind::Sequential,
        Matrix6::identity(),
    );
    assert!(graph
        .optimize_se3_gnc(&no_chordal_config(), &GncConfig::default())
        .is_err());
}
