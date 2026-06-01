//! Integration tests for the Gaussian marginalization prior on [`PoseGraph`]:
//! the factor mechanics (a prior pulls a free pose to its linearization point)
//! and — once `marginalize_pose` lands — that a windowed solve equals the batch
//! solve on the retained poses.

use nalgebra::{DMatrix, Matrix6, UnitQuaternion, Vector3, Vector6};
use visloc_core::geometry::{Pose, SE3};
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
        gradient: nalgebra::DVector::zeros(6),
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
        gradient: nalgebra::DVector::zeros(12),
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

/// The fixed-lag correctness gate: marginalizing an interior pose at the batch
/// optimum, then re-solving the windowed graph, must leave the retained poses at
/// the batch optimum. This is exact only if the prior carries BOTH the curvature
/// Λ' AND the linear term b' (the marginalized edges' gradient) and is built from
/// only the id-incident factors — a missing b' or a double-counted edge would
/// shift the windowed optimum. The inconsistent loop closure makes the optimum
/// carry distributed residual, so any such error is detectable.
#[test]
fn windowed_solve_matches_batch_after_marginalizing_an_interior_pose() {
    let build = || {
        let mut g = PoseGraph::new();
        for i in 0..5 {
            g.add_pose(i, pose_at(i as f64));
        }
        g.anchor(0);
        let info = Matrix6::identity() * 50.0;
        for i in 0..4 {
            g.add_edge_with_information(
                i,
                i + 1,
                relative_world_to_camera(&pose_at(i as f64), &pose_at((i + 1) as f64)),
                PoseGraphEdgeKind::Sequential,
                info,
            );
        }
        // Inconsistent loop 0→4: the truth-relative pose twisted by a small δ, so
        // the optimum cannot satisfy every edge and residual spreads over the chain.
        let twist = SE3::exp(&Vector6::new(0.1, -0.05, 0.08, 0.1, 0.0, -0.1));
        let loop_meas = relative_world_to_camera(&pose_at(0.0), &pose_at(4.0)).compose(&twist);
        g.add_edge_with_information(0, 4, loop_meas, PoseGraphEdgeKind::LoopClosure, info);
        g
    };

    let mut batch = build();
    batch.optimize_se3_iterative(&config()).unwrap();

    // Start the windowed solve from the batch optimum, then marginalize pose 2.
    let mut windowed = batch.clone();
    windowed.marginalize_pose(2).unwrap();
    assert!(!windowed.poses.contains_key(&2), "pose 2 must be removed");
    assert_eq!(
        windowed.priors.len(),
        1,
        "one prior over the blanket {{1,3}}"
    );
    assert_eq!(windowed.priors[0].ids, vec![1, 3]);
    // The id-incident edges (1-2, 2-3) are gone; 0-1, 3-4, 0-4 remain.
    assert!(windowed.edges.iter().all(|e| e.from != 2 && e.to != 2));

    windowed.optimize_se3_iterative(&config()).unwrap();

    for id in [1u64, 3, 4] {
        let d = batch.poses[&id]
            .world_to_camera
            .inverse()
            .compose(&windowed.poses[&id].world_to_camera)
            .log()
            .norm();
        assert!(
            d < 1e-6,
            "windowed pose {id} must equal the batch optimum (prior preserves the marginal): err {d}"
        );
    }
}

/// With a 2-pose blanket (an interior chain pose), the dense Schur prior is
/// already a 2-clique = a tree, so `marginalize_pose_sparsified` must be
/// bit-identical to the dense `marginalize_pose` (Chow-Liu has nothing to drop).
#[test]
fn sparsified_marginalize_is_exact_for_a_two_pose_blanket() {
    let build = || {
        let mut g = PoseGraph::new();
        for i in 0..5 {
            g.add_pose(i, pose_at(i as f64));
        }
        g.anchor(0);
        let info = Matrix6::identity() * 50.0;
        for i in 0..4 {
            g.add_edge_with_information(
                i,
                i + 1,
                relative_world_to_camera(&pose_at(i as f64), &pose_at((i + 1) as f64)),
                PoseGraphEdgeKind::Sequential,
                info,
            );
        }
        let twist = SE3::exp(&Vector6::new(0.1, -0.05, 0.08, 0.1, 0.0, -0.1));
        let loop_meas = relative_world_to_camera(&pose_at(0.0), &pose_at(4.0)).compose(&twist);
        g.add_edge_with_information(0, 4, loop_meas, PoseGraphEdgeKind::LoopClosure, info);
        g.optimize_se3_iterative(&config()).unwrap();
        g
    };
    let mut dense = build();
    let mut sparse = dense.clone();
    dense.marginalize_pose(2).unwrap();
    sparse.marginalize_pose_sparsified(2).unwrap();

    assert_eq!(dense.priors[0].ids, sparse.priors[0].ids);
    let d = (&dense.priors[0].information - &sparse.priors[0].information)
        .abs()
        .max();
    assert!(
        d < 1e-12,
        "2-clique prior must be unchanged by sparsification: {d}"
    );
    let dg = (&dense.priors[0].gradient - &sparse.priors[0].gradient)
        .abs()
        .max();
    assert!(dg < 1e-12, "gradient must be unchanged: {dg}");
}

/// With a ≥3-pose blanket the dense Schur prior is a full clique; sparsification
/// drops it to the Chow-Liu tree. Marginalizing the centre of a 3-leaf star
/// leaves a 3-clique prior over the leaves; the sparsified prior must (a) zero
/// exactly one of the three leaf-pair couplings (tree keeps 2 of 3 edges) and
/// (b) preserve every leaf's marginal covariance exactly.
#[test]
fn sparsified_marginalize_drops_to_a_tree_and_preserves_node_marginals() {
    use nalgebra::DMatrix;
    // Star: centre 10 connected to anchor 0 and three leaves 1,2,3, with distinct
    // edge stiffnesses so the three leaf-pair mutual informations differ (no MST
    // tie) and the dropped edge is well-defined.
    let mut g = PoseGraph::new();
    g.add_pose(0, pose_at(0.0));
    g.add_pose(10, pose_at(1.0));
    g.add_pose(1, pose_at(2.0));
    g.add_pose(2, pose_at(1.0));
    g.add_pose(3, pose_at(0.5));
    g.anchor(0);
    let edge = |g: &mut PoseGraph, a: u64, b: u64, xa: f64, xb: f64, w: f64| {
        g.add_edge_with_information(
            a,
            b,
            relative_world_to_camera(&pose_at(xa), &pose_at(xb)),
            PoseGraphEdgeKind::Sequential,
            Matrix6::identity() * w,
        );
    };
    edge(&mut g, 0, 10, 0.0, 1.0, 40.0);
    edge(&mut g, 10, 1, 1.0, 2.0, 60.0);
    edge(&mut g, 10, 2, 1.0, 1.0, 30.0);
    edge(&mut g, 10, 3, 1.0, 0.5, 90.0);
    g.optimize_se3_iterative(&config()).unwrap();

    let mut dense = g.clone();
    let mut sparse = g.clone();
    dense.marginalize_pose(10).unwrap();
    sparse.marginalize_pose_sparsified(10).unwrap();

    assert_eq!(sparse.priors[0].ids, vec![1, 2, 3]);
    let block = |m: &DMatrix<f64>, i: usize, j: usize| m.view((i * 6, j * 6), (6, 6)).into_owned();
    // Sparse prior: exactly one of the three off-diagonal leaf-pairs is zero.
    let info = &sparse.priors[0].information;
    let zero_pairs = [(0usize, 1usize), (0, 2), (1, 2)]
        .iter()
        .filter(|&&(i, j)| block(info, i, j).abs().max() < 1e-10)
        .count();
    assert_eq!(
        zero_pairs, 1,
        "Chow-Liu tree of a 3-clique drops exactly one edge"
    );
    // The dense prior couples all three pairs (none zero).
    let dinfo = &dense.priors[0].information;
    assert!(
        [(0usize, 1usize), (0, 2), (1, 2)]
            .iter()
            .all(|&(i, j)| block(dinfo, i, j).abs().max() > 1e-10),
        "the dense clique prior couples every leaf pair"
    );
    // Node marginals preserved: per-leaf 6×6 covariance equals the dense prior's.
    let cov_dense = dinfo.clone().cholesky().unwrap().inverse();
    let cov_sparse = info.clone().cholesky().unwrap().inverse();
    for i in 0..3 {
        let d = (block(&cov_dense, i, i) - block(&cov_sparse, i, i))
            .abs()
            .max();
        assert!(d < 1e-9, "leaf {i} marginal covariance not preserved: {d}");
    }
}

/// Marginalizing a leaf pose whose only neighbour is the anchor just drops it
/// (its information was purely relative to the fixed gauge) — no prior added.
#[test]
fn marginalizing_an_anchor_only_leaf_adds_no_prior() {
    let mut g = PoseGraph::new();
    g.add_pose(0, pose_at(0.0));
    g.add_pose(1, pose_at(1.0));
    g.anchor(0);
    g.add_edge_with_information(
        0,
        1,
        relative_world_to_camera(&pose_at(0.0), &pose_at(1.0)),
        PoseGraphEdgeKind::Sequential,
        Matrix6::identity() * 50.0,
    );
    g.marginalize_pose(1).unwrap();
    assert!(!g.poses.contains_key(&1));
    assert!(g.priors.is_empty(), "anchor-only leaf leaves no prior");
    assert!(g.edges.is_empty(), "its only edge is removed");
}

#[test]
fn marginalize_pose_rejects_the_anchor_and_absent_ids() {
    use visloc_slam::PoseGraphError;
    let mut g = PoseGraph::new();
    g.add_pose(0, pose_at(0.0));
    g.add_pose(1, pose_at(1.0));
    g.anchor(0);
    g.add_edge_with_information(
        0,
        1,
        relative_world_to_camera(&pose_at(0.0), &pose_at(1.0)),
        PoseGraphEdgeKind::Sequential,
        Matrix6::identity() * 50.0,
    );
    assert_eq!(g.marginalize_pose(0), Err(PoseGraphError::MissingNode(0)));
    assert_eq!(g.marginalize_pose(9), Err(PoseGraphError::MissingNode(9)));
}

/// The sliding-window driver: marginalizing the oldest poses down to a window at
/// the batch optimum keeps the graph bounded AND leaves the retained recent
/// poses at the batch optimum — chained marginalizations compose without drift.
#[test]
fn marginalize_oldest_keeps_a_bounded_window_matching_batch() {
    let build = || {
        let mut g = PoseGraph::new();
        for i in 0..8 {
            g.add_pose(i, pose_at(i as f64));
        }
        g.anchor(0);
        let info = Matrix6::identity() * 50.0;
        for i in 0..7 {
            g.add_edge_with_information(
                i,
                i + 1,
                relative_world_to_camera(&pose_at(i as f64), &pose_at((i + 1) as f64)),
                PoseGraphEdgeKind::Sequential,
                info,
            );
        }
        // An inconsistent loop closure so the optimum carries distributed residual.
        let twist = SE3::exp(&Vector6::new(0.06, -0.04, 0.05, 0.08, 0.02, -0.07));
        let loop_meas = relative_world_to_camera(&pose_at(0.0), &pose_at(7.0)).compose(&twist);
        g.add_edge_with_information(0, 7, loop_meas, PoseGraphEdgeKind::LoopClosure, info);
        g
    };

    let mut batch = build();
    batch.optimize_se3_iterative(&config()).unwrap();

    let mut windowed = batch.clone();
    // Keep a window of 4 poses (anchor 0 + the three most recent: 5, 6, 7).
    let removed = windowed.marginalize_oldest(4).unwrap();
    assert_eq!(
        removed,
        vec![1, 2, 3, 4],
        "oldest non-anchor poses removed in order"
    );
    assert_eq!(windowed.poses.len(), 4, "window bounded to 4 poses");
    assert!(windowed.poses.contains_key(&0)); // anchor retained
    for kept in [5u64, 6, 7] {
        assert!(windowed.poses.contains_key(&kept));
    }

    windowed.optimize_se3_iterative(&config()).unwrap();
    for id in [5u64, 6, 7] {
        let d = batch.poses[&id]
            .world_to_camera
            .inverse()
            .compose(&windowed.poses[&id].world_to_camera)
            .log()
            .norm();
        assert!(
            d < 1e-6,
            "windowed pose {id} must match the batch optimum after chained marginalization: err {d}"
        );
    }
}
