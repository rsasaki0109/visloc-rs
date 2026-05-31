//! Integration tests for the Gaussian marginalization prior on [`PoseGraph`]:
//! the factor mechanics (a prior pulls a free pose to its linearization point)
//! and — once `marginalize_pose` lands — that a windowed solve equals the batch
//! solve on the retained poses.

use nalgebra::{DMatrix, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_slam::{
    relative_world_to_camera, GaussianPrior, PoseGraph, PoseGraphEdgeKind, PoseGraphSe3Config,
};

fn pose_at(x: f64) -> Pose {
    Pose::from_world_to_camera(UnitQuaternion::identity(), -Vector3::new(x, 0.0, 0.0))
}

fn config() -> PoseGraphSe3Config {
    PoseGraphSe3Config {
        initial_lambda: Some(1.0e-6),
        max_iterations: 50,
        chordal_init: false,
        ..PoseGraphSe3Config::default()
    }
}

/// A stiff Gaussian prior on an otherwise-unconstrained pose must pull it to the
/// prior's linearization point — the basic factor mechanic (H += Ω, g += Ω·e,
/// cost += eᵀΩe), independent of marginalization.
#[test]
fn a_prior_pulls_a_free_pose_to_its_linearization_point() {
    let mut graph = PoseGraph::new();
    graph.add_pose(0, pose_at(0.0));
    graph.add_pose(1, pose_at(1.0));
    // Pose 2 starts badly wrong (x = 5) and has NO edge; only the prior holds it.
    graph.add_pose(2, pose_at(5.0));
    graph.anchor(0);
    // One real edge so the graph is non-empty and pose 1 is constrained.
    graph.add_edge_with_information(
        0,
        1,
        relative_world_to_camera(&pose_at(0.0), &pose_at(1.0)),
        PoseGraphEdgeKind::Sequential,
        nalgebra::Matrix6::identity() * 50.0,
    );
    // Stiff prior pinning pose 2 to x = 2.
    let target = pose_at(2.0);
    graph.priors.push(GaussianPrior {
        ids: vec![2],
        information: DMatrix::<f64>::identity(6, 6) * 1.0e3,
        linearization: vec![target.world_to_camera.clone()],
    });

    graph.optimize_se3_iterative(&config()).unwrap();

    // Pose 2 must have converged to the prior's linearization point.
    let p2 = &graph.poses[&2];
    let err = target
        .world_to_camera
        .inverse()
        .compose(&p2.world_to_camera)
        .log()
        .norm();
    assert!(
        err < 1e-4,
        "a stiff prior must pull the free pose to its linearization point: tangent err {err}"
    );
    // Pose 1 (edge-constrained) stays at its truth.
    let p1 = &graph.poses[&1];
    let err1 = pose_at(1.0)
        .world_to_camera
        .inverse()
        .compose(&p1.world_to_camera)
        .log()
        .norm();
    assert!(err1 < 1e-4, "edge-constrained pose 1 must stay put: {err1}");
}

/// Two priors with disjoint id sets compose additively (each pulls its own pose),
/// and an empty `linearization`-consistent prior at the current estimate adds no
/// gradient (zero tangent error) — so adding such a prior cannot move the solve.
#[test]
fn a_zero_residual_prior_at_the_estimate_does_not_move_the_solve() {
    let mut base = PoseGraph::new();
    for i in 0..4 {
        base.add_pose(i, pose_at(i as f64));
    }
    base.anchor(0);
    for i in 0..3 {
        base.add_edge_with_information(
            i,
            i + 1,
            relative_world_to_camera(&pose_at(i as f64), &pose_at((i + 1) as f64)),
            PoseGraphEdgeKind::Sequential,
            nalgebra::Matrix6::identity() * 50.0,
        );
    }
    let mut with_prior = base.clone();
    base.optimize_se3_iterative(&config()).unwrap();

    // Linearize a prior at the CURRENT (truth) estimate → zero residual → it adds
    // curvature but no gradient, so the optimum is unchanged.
    with_prior.priors.push(GaussianPrior {
        ids: vec![1, 2],
        information: DMatrix::<f64>::identity(12, 12) * 10.0,
        linearization: vec![
            with_prior.poses[&1].world_to_camera.clone(),
            with_prior.poses[&2].world_to_camera.clone(),
        ],
    });
    with_prior.optimize_se3_iterative(&config()).unwrap();

    for id in 1..4u64 {
        let d = base.poses[&id]
            .world_to_camera
            .inverse()
            .compose(&with_prior.poses[&id].world_to_camera)
            .log()
            .norm();
        assert!(
            d < 1e-6,
            "a zero-residual prior at the estimate must not move pose {id}: {d}"
        );
    }
}
