//! GLOMAP-style global structure-from-motion back-end: rotation averaging
//! followed by translation-direction position averaging.
//!
//! The incremental mapper ([`crate::incremental_sfm::incremental_sfm`]) grows one
//! reconstruction image by image; the global SfM lineage (GLOMAP, Stumberger
//! et al. 2024, and its ancestors) instead exploits verified pairwise
//! relative poses directly: average all relative rotations into one
//! consistent world frame, then solve camera positions jointly from the
//! relative translation **directions**, and only afterwards run bundle
//! adjustment. Global estimation avoids the sequential drift an incremental
//! chain accumulates and parallelises trivially.
//!
//! This module implements that two-stage geometry pipeline in pure Rust:
//!
//! 1. **Rotation averaging** — maximum-spanning-tree propagation seeds every
//!    camera of the root component, then iterative geodesic consensus sweeps
//!    pull each orientation towards the weighted SO(3) mean of its
//!    neighbours' predictions (the dominant eigenvector of each camera's
//!    weighted quaternion outer-product sum, tracked by warm-started power
//!    iteration — chordal-L2 relaxation evaluated lazily, no dense system).
//! 2. **Position averaging** — with rotations fixed, every edge contributes
//!    four least-squares rows on the unknown camera centres: three
//!    perpendicular rows `(c_i − c_j) ⊥ d_ij` and one **unit-displacement**
//!    row `d̂_ij · (c_i − c_j) = 1`. The perpendicular rows make displacements
//!    parallel to the measured bearings; the unit rows give the problem a
//!    non-trivial scale (each edge votes its own displacement magnitude),
//!    which the joint solve arbitrates globally. Conjugate gradient on the
//!    normal equations; a weak origin prior absorbs the free-translation
//!    nullspace and the solved field is re-anchored so the seed centre sits
//!    exactly at the origin.
//!
//! Monocular relative translations are scale-free per edge, so positions
//! come out up to one global scale — the same gauge freedom every global SfM
//! system carries into its similarity alignment against metric sources.
//!
//! Only the connected component containing the seed camera is solved;
//! images outside it stay unposed.

use std::collections::{HashMap, HashSet};

use nalgebra::{DMatrix, DVector, Matrix3, Point2, Point3, UnitQuaternion, Vector3};

use visloc_core::geometry::{Pose, Sim3, SE3};
use visloc_core::types::Camera;
use visloc_tracking::umeyama_similarity_transform;
use visloc_vision::features::FeatureSet;
use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_vision::two_view::{
    recover_relative_pose_with_options, CheiralityOptions, ConfigurationType, RelativePose,
    RelativePoseEstimator, TwoViewCorrespondence,
};

use crate::incremental_sfm::{
    build_track_output, post_refinement_registration_pass, reprojection_error_px,
    run_bundle_adjustment, triangulate_pending, PairwiseMatches, PnpSolver, SfmTrack,
};
use visloc_vision::pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};

/// One verified pairwise constraint between two views of the view graph.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalSfmEdge {
    /// First image index.
    pub image_i: usize,
    /// Second image index.
    pub image_j: usize,
    /// Relative rotation `R_ij` mapping camera-`i`-frame coordinates to
    /// camera-`j`-frame coordinates (`X_j = R_ij · X_i + t_ij`). Composed
    /// with world-to-camera orientations: `R_w2c_j = R_ij · R_w2c_i`.
    pub rotation_ij: UnitQuaternion<f64>,
    /// Relative translation **direction** from camera `i` to camera `j`,
    /// expressed in camera `i`'s frame (unit length; essential-matrix
    /// decompositions recover this up to per-edge scale only).
    pub direction_ij: Vector3<f64>,
    /// Edge weight (e.g., verified inlier count).
    pub weight: f64,
    /// Sample of inlier pixel correspondences retained so translation can be
    /// re-estimated under globally-averaged rotations. Empty in synthetic
    /// fixtures that only exercise averaging.
    pub inlier_sample: Vec<(Point2<f64>, Point2<f64>)>,
    /// Optional runner-up relative rotation from chirality-ambiguous essential
    /// decomposition. Rotation averaging picks primary vs alternate by
    /// agreement with the emerging global solution.
    pub rotation_alt: Option<UnitQuaternion<f64>>,
    /// Bearing paired with [`Self::rotation_alt`] (camera-j centre in cam-i).
    pub direction_alt: Option<Vector3<f64>>,
}

impl GlobalSfmEdge {
    /// Relative rotation step leaving `from` under the primary hypothesis.
    fn step_primary(&self, from: usize) -> UnitQuaternion<f64> {
        if self.image_i == from {
            self.rotation_ij
        } else {
            self.rotation_ij.inverse()
        }
    }

    /// Relative rotation step leaving `from`, choosing primary or alternate
    /// by which better predicts `to_hint` when supplied.
    fn step_adaptive(
        &self,
        from: usize,
        from_q: UnitQuaternion<f64>,
        to_hint: Option<UnitQuaternion<f64>>,
    ) -> UnitQuaternion<f64> {
        let primary = self.step_primary(from);
        let Some(hint) = to_hint else {
            return primary;
        };
        let Some(r_alt) = self.rotation_alt else {
            return primary;
        };
        let alt = if self.image_i == from {
            r_alt
        } else {
            r_alt.inverse()
        };
        let primary_pred = primary * from_q;
        let alt_pred = alt * from_q;
        if (alt_pred.inverse() * hint).angle() + 1e-12 < (primary_pred.inverse() * hint).angle() {
            alt
        } else {
            primary
        }
    }

    /// Select primary or alternate (R, direction) under current global
    /// orientations of both endpoints.
    fn select_pose(
        &self,
        qi: UnitQuaternion<f64>,
        qj: UnitQuaternion<f64>,
    ) -> (UnitQuaternion<f64>, Vector3<f64>) {
        let primary_err = ((self.rotation_ij * qi).inverse() * qj).angle();
        if let (Some(r_alt), Some(d_alt)) = (self.rotation_alt, self.direction_alt) {
            let alt_err = ((r_alt * qi).inverse() * qj).angle();
            if alt_err + 1e-12 < primary_err {
                return (r_alt, d_alt);
            }
        }
        (self.rotation_ij, self.direction_ij)
    }

    /// Swap primary ↔ alternate in place. No-op when no alternate is stored.
    fn swap_primary_alternate(&mut self) {
        let Some(r_alt) = self.rotation_alt.take() else {
            return;
        };
        let Some(d_alt) = self.direction_alt.take() else {
            self.rotation_alt = Some(r_alt);
            return;
        };
        self.rotation_alt = Some(self.rotation_ij);
        self.direction_alt = Some(self.direction_ij);
        self.rotation_ij = r_alt;
        self.direction_ij = d_alt;
    }
}

/// Result of the global pose solve.
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalSfmPoses {
    /// One pose per input image; `None` for images outside the solved
    /// component (or everything when no usable edges exist).
    pub poses: Vec<Option<Pose>>,
    /// Mean angular residual over solved edges, in radians: the angle
    /// between each edge's solved centre displacement and its measured
    /// bearing direction.
    pub mean_bearing_residual_rad: f64,
    /// How many edge translation directions were flipped by
    /// [`refine_edge_directions_under_rotations`] (0 when refine is off).
    pub translation_refine_flips: usize,
}

type JointTrackPositioningInput<'a> = (&'a Camera, &'a [FeatureSet], &'a [Vec<(usize, usize)>]);

/// Solve global camera poses from verified pairwise relative-pose edges.
///
/// `num_images` sizes the output; `seed` is the gauge camera (identity
/// rotation, centre pinned at the origin); `sweeps` bounds the rotation
/// consensus iterations (8–16 is ample on well-connected graphs);
/// `cg_iterations` bounds the position least-squares solver.
/// `max_edge_rotation_error_deg` trims edges whose implied relative rotation
/// disagrees with the emerging global solution by more than this angle
/// (IRLS-style outlier rejection; 10° is a reasonable default).
///
/// Returns `None` when no valid edge touches any image.
///
/// When `refine_translations` is true and `camera` is supplied, after
/// rotation averaging each edge's `direction_ij` is re-scored under the
/// consensus relative rotation (and optionally re-estimated from its
/// inlier sample via a fixed-R epipolar nullspace). This is the lever that
/// can flip chirality-ambiguous pairwise translations once the rotation
/// graph has left the wrong basin.
///
/// `pose_priors`, when supplied, pins those cameras' world-to-camera
/// orientations during averaging. When `pin_prior_centres` is true (default
/// hybrid), Sim(3)-aligns the solved centres onto the prior centres and
/// writes the full incremental pose back. When false (`--hybrid-rotation-priors-only`),
/// incremental orientations stay pinned but camera centres come from global
/// bearing averaging instead of the incremental solve.
///
/// `joint_tracks`, when supplied with `joint_global_positioning`, refines
/// centres via GLOMAP-style camera+point ray IRLS after pairwise init.
#[allow(clippy::too_many_arguments)]
pub fn solve_global_sfm(
    num_images: usize,
    edges: &mut [GlobalSfmEdge],
    seed: usize,
    sweeps: usize,
    cg_iterations: usize,
    max_edge_rotation_error_deg: f64,
    refine_translations: bool,
    camera: Option<&Camera>,
    pose_priors: Option<&[Option<Pose>]>,
    pin_prior_centres: bool,
    joint_tracks: Option<JointTrackPositioningInput<'_>>,
    repair_prior_edges: bool,
    metric_prior_scale: bool,
) -> Option<GlobalSfmPoses> {
    solve_global_sfm_with_options(
        num_images,
        edges,
        seed,
        sweeps,
        cg_iterations,
        max_edge_rotation_error_deg,
        refine_translations,
        camera,
        pose_priors,
        pin_prior_centres,
        joint_tracks,
        repair_prior_edges,
        metric_prior_scale,
        false,
    )
}

/// Internal global pose solve with experimental position-averaging options.
/// The public [`solve_global_sfm`] wrapper intentionally keeps the historical
/// unit-displacement solver as its default.
#[allow(clippy::too_many_arguments)]
fn solve_global_sfm_with_options(
    num_images: usize,
    edges: &mut [GlobalSfmEdge],
    seed: usize,
    sweeps: usize,
    cg_iterations: usize,
    max_edge_rotation_error_deg: f64,
    refine_translations: bool,
    camera: Option<&Camera>,
    pose_priors: Option<&[Option<Pose>]>,
    pin_prior_centres: bool,
    joint_tracks: Option<JointTrackPositioningInput<'_>>,
    repair_prior_edges: bool,
    metric_prior_scale: bool,
    independent_edge_scales: bool,
) -> Option<GlobalSfmPoses> {
    let adjacency = build_adjacency(num_images, edges)?;
    let fixed_rots: Option<Vec<Option<UnitQuaternion<f64>>>> = pose_priors.map(|priors| {
        priors
            .iter()
            .map(|p| p.as_ref().map(|pose| pose.world_to_camera.rotation))
            .collect()
    });
    let (mut rotations, mut edge_weights) = average_rotations_with_priors(
        num_images,
        edges,
        &adjacency,
        seed,
        sweeps,
        max_edge_rotation_error_deg,
        fixed_rots.as_deref(),
    );
    let component: Vec<usize> = rotations
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_some())
        .map(|(i, _)| i)
        .collect();
    if component.is_empty() {
        return None;
    }
    let mut translation_refine_flips = 0usize;
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        let mut errs: Vec<f64> = Vec::new();
        let kept = edge_weights.iter().filter(|&&w| w > 0.0).count();
        for (index, e) in edges.iter().enumerate() {
            if edge_weights[index] <= 0.0 {
                continue;
            }
            if let (Some(qi), Some(qj)) = (rotations[e.image_i], rotations[e.image_j]) {
                let predicted = step_from(edges, index, e.image_i) * qi;
                errs.push((predicted.inverse() * qj).angle());
            }
        }
        errs.sort_by(|a, b| a.total_cmp(b));
        let med = errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN);
        eprintln!(
            "global-sfm debug: post-average rotation err median={:.4} deg max={:.4} deg kept={}/{}",
            med.to_degrees(),
            errs.last().map(|e| e.to_degrees()).unwrap_or(f64::NAN),
            kept,
            edges.len()
        );
    }
    if refine_translations {
        if let Some(cam) = camera {
            translation_refine_flips =
                refine_edge_directions_under_rotations(edges, &rotations, &edge_weights, cam);
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: global-R translation refine flipped {translation_refine_flips} edge directions"
                );
            }
        }
    }
    if repair_prior_edges {
        if let Some(priors) = pose_priors {
            let (repaired, flipped) =
                repair_edges_from_pose_priors(edges, &mut edge_weights, priors);
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: prior-edge repair rewrote {repaired} edges (flipped {flipped})"
                );
            }
        }
    }
    let mut centers = if independent_edge_scales {
        average_positions_with_independent_edge_scales(
            edges,
            &edge_weights,
            &rotations,
            seed,
            cg_iterations,
        )
        .unwrap_or_else(|| {
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: independent edge-scale solve was rank-deficient; falling back to legacy position averaging"
                );
            }
            average_positions_with_priors(
                edges,
                &edge_weights,
                &rotations,
                seed,
                cg_iterations,
                if metric_prior_scale {
                    pose_priors
                } else {
                    None
                },
            )
        })
    } else {
        average_positions_with_priors(
            edges,
            &edge_weights,
            &rotations,
            seed,
            cg_iterations,
            if metric_prior_scale {
                pose_priors
            } else {
                None
            },
        )
    };
    if let Some((cam, feats, tracks)) = joint_tracks {
        let pinned = if pin_prior_centres { pose_priors } else { None };
        centers =
            refine_centers_joint_tracks(cam, feats, tracks, &rotations, centers, seed, pinned);
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            let posed = centers.iter().filter(|c| c.is_some()).count();
            eprintln!("global-sfm debug: joint track positioning kept {posed} centres");
        }
    }
    if let Some(priors) = pose_priors {
        if pin_prior_centres {
            align_solution_to_pose_priors(&mut rotations, &mut centers, priors);
        } else {
            align_rotation_only_gauge(&mut centers, priors, seed);
        }
    }
    let mut poses: Vec<Option<Pose>> = vec![None; num_images];
    for &image in &component {
        if let Some(priors) = pose_priors {
            if let Some(Some(prior)) = priors.get(image) {
                if pin_prior_centres {
                    poses[image] = Some(prior.clone());
                    continue;
                }
                let Some(center) = centers[image] else {
                    continue;
                };
                let q = prior.world_to_camera.rotation;
                let t_w2c = -q.transform_vector(&center.coords);
                poses[image] = Some(Pose::from_world_to_camera(q, t_w2c));
                continue;
            }
        }
        let (Some(q_w2c), Some(center)) = (rotations[image], centers[image]) else {
            continue;
        };
        let t_w2c = -q_w2c.transform_vector(&center.coords);
        poses[image] = Some(Pose::from_world_to_camera(q_w2c, t_w2c));
    }
    let (mut sum, mut count) = (0.0f64, 0usize);
    for (edge_index, edge) in edges.iter().enumerate() {
        if edge_weights.get(edge_index).copied().unwrap_or(1.0) <= 0.0 {
            continue;
        }
        let (Some(ci), Some(cj)) = (
            centers.get(edge.image_i).copied().flatten(),
            centers.get(edge.image_j).copied().flatten(),
        ) else {
            continue;
        };
        let Some(q_i) = rotations[edge.image_i] else {
            continue;
        };
        let measured = q_i
            .inverse()
            .transform_vector(&edge.direction_ij)
            .normalize();
        let displacement = cj - ci;
        let norm = displacement.norm();
        if norm < 1e-12 {
            continue;
        }
        let cos = (displacement.dot(&measured) / norm).clamp(-1.0, 1.0);
        sum += cos.acos();
        count += 1;
    }
    Some(GlobalSfmPoses {
        mean_bearing_residual_rad: sum / count.max(1) as f64,
        translation_refine_flips,
        poses,
    })
}

/// Scale + seed translation for hybrid rotation-only priors: robust inter-prior
/// distance ratios set metric scale; the seed prior's incremental centre anchors
/// translation. Prior cameras are not snapped to their incremental centres.
fn align_rotation_only_gauge(
    centers: &mut [Option<Point3<f64>>],
    priors: &[Option<Pose>],
    seed: usize,
) {
    let prior_indices: Vec<usize> = priors
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.as_ref().map(|_| i))
        .collect();
    if prior_indices.len() < 2 {
        return;
    }
    let mut ratios = Vec::new();
    for a in 0..prior_indices.len() {
        for b in (a + 1)..prior_indices.len() {
            let i = prior_indices[a];
            let j = prior_indices[b];
            let (Some(ci), Some(cj), Some(pi), Some(pj)) = (
                centers.get(i).copied().flatten(),
                centers.get(j).copied().flatten(),
                priors.get(i).and_then(|p| p.as_ref()),
                priors.get(j).and_then(|p| p.as_ref()),
            ) else {
                continue;
            };
            let da = (ci - cj).norm();
            let dp = (pi.camera_center_world() - pj.camera_center_world()).norm();
            if da > 1e-9 && dp > 1e-9 {
                ratios.push(dp / da);
            }
        }
    }
    if ratios.is_empty() {
        return;
    }
    ratios.sort_by(f64::total_cmp);
    // Need several prior pairs for incremental layout to constrain scale; with
    // only one pair a wrong prior distance would corrupt the whole solve.
    let scale = if ratios.len() >= 3 {
        let s = ratios[ratios.len() / 2];
        if s.is_finite() && s > 0.0 {
            s
        } else {
            1.0
        }
    } else {
        1.0
    };
    if !(scale.is_finite() && scale > 0.0) {
        return;
    }
    for c in centers.iter_mut().flatten() {
        *c = Point3::from(scale * c.coords);
    }
    let Some(seed_prior) = priors.get(seed).and_then(|p| p.as_ref()) else {
        return;
    };
    let Some(seed_center) = centers.get(seed).copied().flatten() else {
        return;
    };
    let delta = seed_prior.camera_center_world() - seed_center;
    for c in centers.iter_mut().flatten() {
        *c = Point3::from(c.coords + delta);
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: rotation-only gauge scale={scale:.4} from {} prior pairs (seed={seed})",
            ratios.len()
        );
    }
}

/// Sim(3)-align solved centres (and free rotations) onto absolute pose priors.
fn align_solution_to_pose_priors(
    rotations: &mut [Option<UnitQuaternion<f64>>],
    centers: &mut [Option<Point3<f64>>],
    priors: &[Option<Pose>],
) {
    let mut src = Vec::new();
    let mut dst = Vec::new();
    let n = rotations.len().min(centers.len()).min(priors.len());
    for i in 0..n {
        let (Some(c), Some(prior)) = (centers[i], priors[i].as_ref()) else {
            continue;
        };
        src.push(c);
        dst.push(prior.camera_center_world());
    }
    if src.len() < 3 {
        return;
    }
    let Some(fit) = umeyama_similarity_transform(&src, &dst, true) else {
        return;
    };
    if !(fit.scale.is_finite() && fit.scale > 0.0) {
        return;
    }
    let sim = Sim3::new(
        UnitQuaternion::from_rotation_matrix(&fit.rotation),
        fit.translation,
        fit.scale,
    );
    let r_inv = UnitQuaternion::from_rotation_matrix(&fit.rotation).inverse();
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: aligned to {} pose priors (Sim3 scale={:.4})",
            src.len(),
            fit.scale
        );
    }
    for i in 0..n {
        if priors.get(i).and_then(|p| p.as_ref()).is_some() {
            // Exact prior pose is written later; keep centre/rotation coherent.
            if let Some(prior) = priors[i].as_ref() {
                centers[i] = Some(prior.camera_center_world());
                rotations[i] = Some(prior.world_to_camera.rotation);
            }
            continue;
        }
        if let Some(c) = centers[i] {
            centers[i] = Some(sim.transform_point(&c));
        }
        if let Some(q) = rotations[i] {
            rotations[i] = Some(q * r_inv);
        }
    }
}

/// Adjacency: image → [(neighbor, edge_index)].
type Adjacency = HashMap<usize, Vec<(usize, usize)>>;

fn build_adjacency(num_images: usize, edges: &[GlobalSfmEdge]) -> Option<Adjacency> {
    if num_images == 0 || edges.is_empty() {
        return None;
    }
    let mut adjacency: Adjacency = HashMap::new();
    for (index, edge) in edges.iter().enumerate() {
        if edge.image_i >= num_images || edge.image_j >= num_images || edge.image_i == edge.image_j
        {
            continue;
        }
        if !(edge.weight.is_finite() && edge.weight > 0.0) {
            continue;
        }
        let n2 = edge.direction_ij.norm_squared();
        if !n2.is_finite() || n2 < 1e-18 {
            continue;
        }
        adjacency
            .entry(edge.image_i)
            .or_default()
            .push((edge.image_j, index));
        adjacency
            .entry(edge.image_j)
            .or_default()
            .push((edge.image_i, index));
    }
    (!adjacency.is_empty()).then_some(adjacency)
}

/// Rotation step along an edge leaving `from`: composes onto `from`'s
/// orientation to predict its neighbour's.
fn step_from(edges: &[GlobalSfmEdge], index: usize, from: usize) -> UnitQuaternion<f64> {
    edges[index].step_primary(from)
}

/// Maximum-spanning-tree rotation seeding (Prim-style over descending weights).
/// Returns per-image orientations plus the reached member list; images not
/// connected to the root (or prior set) stay `None`.
///
/// When `fixed_rotations` supplies absolute orientations (e.g. from an
/// incremental solve), those cameras are pinned and the tree grows outward
/// from their union instead of from a single identity seed.
fn tree_seed_rotations(
    num_images: usize,
    edges: &[GlobalSfmEdge],
    adjacency: &Adjacency,
    root: usize,
    fixed_rotations: Option<&[Option<UnitQuaternion<f64>>]>,
) -> (Vec<Option<UnitQuaternion<f64>>>, Vec<usize>, Vec<bool>) {
    // Maximum-spanning-tree growth from the root, Prim-style: always extend
    // through the highest-weight edge leaving the reached set. A Kruskal
    // pass cannot be used here because it consumes tree edges inside the
    // unreached mass before the frontier ever reaches them, stranding the
    // seed (observed as posed=4 on courtyard's 37-image component).
    let mut reached = vec![false; num_images];
    let mut rotations: Vec<Option<UnitQuaternion<f64>>> = vec![None; num_images];
    let mut fixed_mask = vec![false; num_images];
    let mut members = Vec::new();
    if let Some(fixed) = fixed_rotations {
        for (i, prior) in fixed.iter().enumerate().take(num_images) {
            if let Some(q) = prior {
                rotations[i] = Some(*q);
                reached[i] = true;
                fixed_mask[i] = true;
                members.push(i);
            }
        }
    }
    if members.is_empty() {
        reached[root] = true;
        rotations[root] = Some(UnitQuaternion::identity());
        members.push(root);
    } else if !reached[root] {
        // Priors define the gauge; keep `root` as a free camera grown from them.
    }
    // Frontier heap of (weight key, tie-break index, from, edge_index).
    let mut heap: std::collections::BinaryHeap<(u64, usize, usize, usize)> =
        std::collections::BinaryHeap::new();
    let push_frontier = |heap: &mut std::collections::BinaryHeap<(u64, usize, usize, usize)>,
                         node: usize,
                         reached: &[bool]| {
        if let Some(neighbors) = adjacency.get(&node) {
            for &(neighbor, edge_index) in neighbors {
                if !reached[neighbor] {
                    let w = edges[edge_index].weight;
                    let key = if w.is_finite() && w > 0.0 { w } else { 1.0 };
                    // Quantize the weight into the u64 key (nanogram precision
                    // is plenty for inlier counts) so the heap is total.
                    let key_bits = (key.max(0.0) * 1e6).round().max(0.0) as u64;
                    heap.push((key_bits, usize::MAX - edge_index, node, edge_index));
                }
            }
        }
    };
    for &node in &members {
        push_frontier(&mut heap, node, &reached);
    }
    while let Some((_, _, known, edge_index)) = heap.pop() {
        let unknown = {
            let e = &edges[edge_index];
            if reached[e.image_i] && !reached[e.image_j] {
                e.image_j
            } else if reached[e.image_j] && !reached[e.image_i] {
                e.image_i
            } else {
                continue;
            }
        };
        let base = rotations[known].expect("frontier source carries a seeded orientation");
        rotations[unknown] = Some(step_from(edges, edge_index, known) * base);
        reached[unknown] = true;
        members.push(unknown);
        push_frontier(&mut heap, unknown, &reached);
    }
    members.sort_unstable();
    (rotations, members, fixed_mask)
}

/// Iterative geodesic-consensus rotation averaging over the root's
/// component. Each sweep recomputes every non-seed / non-fixed orientation
/// as the weighted SO(3) mean of its neighbours' predictions.
///
/// `fixed_rotations`, when supplied, pins those cameras for the whole solve
/// (hybrid incremental→global gauge).
pub fn average_rotations(
    num_images: usize,
    edges: &[GlobalSfmEdge],
    adjacency: &Adjacency,
    seed: usize,
    sweeps: usize,
    max_edge_rotation_error_deg: f64,
) -> (Vec<Option<UnitQuaternion<f64>>>, Vec<f64>) {
    average_rotations_with_priors(
        num_images,
        edges,
        adjacency,
        seed,
        sweeps,
        max_edge_rotation_error_deg,
        None,
    )
}

fn average_rotations_with_priors(
    num_images: usize,
    edges: &[GlobalSfmEdge],
    adjacency: &Adjacency,
    seed: usize,
    sweeps: usize,
    max_edge_rotation_error_deg: f64,
    fixed_rotations: Option<&[Option<UnitQuaternion<f64>>]>,
) -> (Vec<Option<UnitQuaternion<f64>>>, Vec<f64>) {
    // IRLS edge weights: start from the verified-inlier weight; each sweep
    // zeroes edges whose implied relative rotation disagrees with the
    // emerging global solution beyond the trim threshold.
    let mut weights: Vec<f64> = edges.iter().map(|e| e.weight.max(1.0)).collect();
    let max_error_rad = max_edge_rotation_error_deg.to_radians();
    let (mut rotations, members, fixed_mask) =
        tree_seed_rotations(num_images, edges, adjacency, seed, fixed_rotations);
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        let n_fixed = fixed_mask.iter().filter(|&&f| f).count();
        eprintln!(
            "global-sfm debug: tree seed reached {} of {} images from root {} ({} rotation priors)",
            members.len(),
            num_images,
            seed,
            n_fixed
        );
    }
    let mut ordered_members = members.clone();
    ordered_members.sort_unstable();
    for _ in 0..sweeps {
        // Gauss-Seidel style: consume this sweep's own updates immediately
        // (Jacobi snapshots can oscillate on noisy view graphs).
        for &image in &ordered_members {
            if image == seed || fixed_mask.get(image).copied().unwrap_or(false) {
                continue;
            }
            let Some(neighbors) = adjacency.get(&image) else {
                continue;
            };
            if neighbors.is_empty() {
                continue;
            }
            // Weighted quaternion outer-product matrix M = Σ w q qᵀ over
            // neighbour-implied predictions of this camera's orientation.
            let mut m = [[0.0f64; 4]; 4];
            for &(neighbor, edge_index) in neighbors {
                let w = weights[edge_index];
                if w <= 0.0 {
                    continue;
                }
                let Some(neighbor_q) = rotations[neighbor] else {
                    continue;
                };
                let predicted =
                    edges[edge_index].step_adaptive(neighbor, neighbor_q, rotations[image])
                        * neighbor_q;
                let q = predicted.coords;
                let w = w.max(1.0);
                for r in 0..4 {
                    for c in 0..4 {
                        m[r][c] += w * q[r] * q[c];
                    }
                }
            }
            let Some(current) = rotations[image] else {
                continue;
            };
            let mut v = *current.as_vector();
            for _ in 0..8 {
                let next = [
                    m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + m[0][3] * v.w,
                    m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + m[1][3] * v.w,
                    m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + m[2][3] * v.w,
                    m[3][0] * v.x + m[3][1] * v.y + m[3][2] * v.z + m[3][3] * v.w,
                ];
                let norm =
                    (next[0] * next[0] + next[1] * next[1] + next[2] * next[2] + next[3] * next[3])
                        .sqrt();
                if !norm.is_finite() || norm < 1e-15 {
                    break;
                }
                v = nalgebra::Vector4::new(next[0], next[1], next[2], next[3]) / norm;
            }
            if v.dot(current.as_vector()) < 0.0 {
                v = -v;
            }
            rotations[image] = Some(UnitQuaternion::from_quaternion(
                nalgebra::Quaternion::from_vector(v),
            ));
        }
        if max_error_rad.is_finite() && max_error_rad > 0.0 {
            for (edge_index, e) in edges.iter().enumerate() {
                if weights[edge_index] <= 0.0 {
                    continue;
                }
                if let (Some(qi), Some(qj)) = (rotations[e.image_i], rotations[e.image_j]) {
                    let (r_sel, _) = e.select_pose(qi, qj);
                    let predicted = r_sel * qi;
                    let err = (predicted.inverse() * qj).angle();
                    if err > max_error_rad {
                        weights[edge_index] = 0.0;
                    }
                }
            }
        }
    }
    (rotations, weights)
}

pub fn average_positions(
    edges: &[GlobalSfmEdge],
    edge_weights: &[f64],
    rotations: &[Option<UnitQuaternion<f64>>],
    seed: usize,
    cg_iterations: usize,
) -> Vec<Option<Point3<f64>>> {
    average_positions_with_priors(edges, edge_weights, rotations, seed, cg_iterations, None)
}

/// Like [`average_positions`], but when `pose_priors` is supplied the global
/// scale row is taken from the highest-weight prior–prior edge at its true
/// metric length (instead of a unit displacement on a seed edge). Hybrid
/// mappers use this so free cameras inherit the incremental metric frame.
pub fn average_positions_with_priors(
    edges: &[GlobalSfmEdge],
    edge_weights: &[f64],
    rotations: &[Option<UnitQuaternion<f64>>],
    seed: usize,
    cg_iterations: usize,
    pose_priors: Option<&[Option<Pose>]>,
) -> Vec<Option<Point3<f64>>> {
    // World-frame bearings from every kept edge.
    struct Bearing {
        i: usize,
        j: usize,
        /// World-frame unit bearing from camera i towards camera j.
        direction: Vector3<f64>,
        weight: f64,
    }
    let mut bearings: Vec<Bearing> = Vec::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        let weight = edge_weights.get(edge_index).copied().unwrap_or(1.0);
        if weight <= 0.0 {
            continue;
        }
        let Some(q) = rotations.get(edge.image_i).copied().flatten() else {
            continue;
        };
        let Some(qj) = rotations.get(edge.image_j).copied().flatten() else {
            continue;
        };
        let (_, direction_ij) = edge.select_pose(q, qj);
        let world = q.inverse().transform_vector(&direction_ij);
        let norm = world.norm();
        if !norm.is_finite() || norm < 1e-12 {
            continue;
        }
        bearings.push(Bearing {
            i: edge.image_i,
            j: edge.image_j,
            direction: world / norm,
            weight,
        });
    }
    bearings.retain(|b| {
        rotations.get(b.i).copied().flatten().is_some()
            && rotations.get(b.j).copied().flatten().is_some()
    });
    if bearings.is_empty() {
        return vec![None; rotations.len()];
    }

    // ---- Translation-sign repair via MST placement --------------------------
    // Chirality-ambiguous essentials can flip `direction_ij` 180° while still
    // passing per-pair RANSAC. Build a maximum-weight spanning tree, place
    // cameras by walking unit steps along tree bearings, then flip any
    // (tree or off-tree) bearing that is anti-aligned with the preliminary
    // displacement. Self-consistent flip basins stay flipped together — this
    // only repairs *isolated* sign errors against a trusted skeleton.
    {
        let members: Vec<usize> = (0..rotations.len())
            .filter(|&i| rotations[i].is_some())
            .collect();
        let local_of: HashMap<usize, usize> =
            members.iter().enumerate().map(|(k, &i)| (i, k)).collect();
        let mut parent: Vec<usize> = (0..members.len()).collect();
        fn find_sign(parent: &mut [usize], x: usize) -> usize {
            let mut x = x;
            while parent[x] != x {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            x
        }
        let mut tree_edge: Vec<usize> = Vec::new();
        let mut order: Vec<usize> = (0..bearings.len()).collect();
        order.sort_by(|&a, &b| {
            bearings[b]
                .weight
                .total_cmp(&bearings[a].weight)
                .then_with(|| a.cmp(&b))
        });
        for &k in &order {
            let Some(&li) = local_of.get(&bearings[k].i) else {
                continue;
            };
            let Some(&lj) = local_of.get(&bearings[k].j) else {
                continue;
            };
            let (ri, rj) = (find_sign(&mut parent, li), find_sign(&mut parent, lj));
            if ri != rj {
                parent[ri] = rj;
                tree_edge.push(k);
            }
        }
        // BFS place from seed along tree edges.
        let mut adj: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
        for &k in &tree_edge {
            let b = &bearings[k];
            adj.entry(b.i).or_default().push((b.j, k));
            adj.entry(b.j).or_default().push((b.i, k));
        }
        let mut prelim: HashMap<usize, Vector3<f64>> = HashMap::new();
        if rotations.get(seed).copied().flatten().is_some() {
            prelim.insert(seed, Vector3::zeros());
            let mut stack = vec![seed];
            while let Some(node) = stack.pop() {
                let Some(nbrs) = adj.get(&node) else {
                    continue;
                };
                let c_node = prelim[&node];
                for &(other, k) in nbrs {
                    if prelim.contains_key(&other) {
                        continue;
                    }
                    let b = &bearings[k];
                    // Walking i→j along +direction; j→i along −direction.
                    let step = if b.i == node {
                        b.direction
                    } else {
                        -b.direction
                    };
                    prelim.insert(other, c_node + step);
                    stack.push(other);
                }
            }
        }
        let mut flipped = 0usize;
        for b in bearings.iter_mut() {
            let (Some(ci), Some(cj)) = (prelim.get(&b.i), prelim.get(&b.j)) else {
                continue;
            };
            let disp = *cj - *ci;
            if disp.norm() < 1e-12 {
                continue;
            }
            if disp.dot(&b.direction) < 0.0 {
                b.direction = -b.direction;
                flipped += 1;
            }
        }
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && flipped > 0 {
            eprintln!(
                "global-sfm debug: translation-sign repair flipped {flipped}/{} bearings",
                bearings.len()
            );
        }
    }

    const PRIOR_WEIGHT: f64 = 1e-6;
    let members: Vec<usize> = (0..rotations.len())
        .filter(|&i| rotations[i].is_some())
        .collect();

    // One scale-setting displacement row. Default: unit length on the
    // highest-weight seed-touching edge. With pose priors: the highest-weight
    // prior–prior bearing at its true metric length (all prior–prior metric
    // rows A/B'd worse on courtyard: ~353 cm vs single-row ~311 cm).
    struct MetricEdge {
        i: usize,
        j: usize,
        direction: Vector3<f64>,
        weight: f64,
        length: f64,
    }
    let mut metric_edges: Vec<MetricEdge> = Vec::new();
    if let Some(priors) = pose_priors {
        let best = bearings.iter().filter_map(|b| {
            let pi = priors.get(b.i).and_then(|p| p.as_ref())?;
            let pj = priors.get(b.j).and_then(|p| p.as_ref())?;
            let len = (pj.camera_center_world() - pi.camera_center_world()).norm();
            if !(len.is_finite() && len > 1e-6) {
                return None;
            }
            Some((b, len))
        });
        if let Some((b, len)) = best.max_by(|(a, _), (b, _)| {
            a.weight
                .total_cmp(&b.weight)
                .then_with(|| a.i.cmp(&b.i))
                .then_with(|| a.j.cmp(&b.j))
        }) {
            metric_edges.push(MetricEdge {
                i: b.i,
                j: b.j,
                direction: b.direction,
                weight: b.weight.max(1.0),
                length: len,
            });
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: metric scale from 1 prior–prior edge \
                     (i={}, j={}, len={len:.4}, w={:.3})",
                    b.i, b.j, b.weight
                );
            }
        }
    }
    if metric_edges.is_empty() {
        if let Some((_, b)) = bearings
            .iter()
            .enumerate()
            .filter(|(_, b)| b.i == seed || b.j == seed)
            .max_by(|(ia, a), (ib, b)| {
                a.weight
                    .total_cmp(&b.weight)
                    .then_with(|| a.j.cmp(&b.j).reverse())
                    .then_with(|| ib.cmp(ia))
            })
        {
            metric_edges.push(MetricEdge {
                i: b.i,
                j: b.j,
                direction: b.direction,
                weight: b.weight.max(1.0),
                length: 1.0,
            });
        }
    }

    // ---- Trust-hierarchy graduated Huber ------------------------------------
    // Bent shapes come from systematically wrong bearings pulling the least
    // squares; the errors are not outlier-like, so plain robust weighting
    // cannot separate them. What does distinguish them is TRUST: a maximum-
    // weight spanning tree (Kruskal) marks the edges most likely to be right.
    // We therefore run graduated Huber rounds where every edge's threshold is
    // scaled by its trust tier — tree edges tolerate twice the angular error
    // of off-tree edges at every round — so systematically wrong off-tree
    // edges are demoted FIRST while the trusted skeleton survives to anchor
    // the shape. On clean data nothing ever exceeds even the tight threshold,
    // making the whole schedule an identity.
    let mut parent: Vec<usize> = (0..members.len()).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut x = x;
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let local_of: HashMap<usize, usize> =
        members.iter().enumerate().map(|(k, &i)| (i, k)).collect();
    let mut tree_flags = vec![false; bearings.len()];
    {
        let mut order: Vec<usize> = (0..bearings.len()).collect();
        order.sort_by(|&a, &b| {
            bearings[b]
                .weight
                .total_cmp(&bearings[a].weight)
                .then_with(|| a.cmp(&b))
        });
        for &k in &order {
            let li = *local_of.get(&bearings[k].i).unwrap();
            let lj = *local_of.get(&bearings[k].j).unwrap();
            let (ri, rj) = (find(&mut parent, li), find(&mut parent, lj));
            if ri != rj {
                parent[ri] = rj;
                tree_flags[k] = true;
            }
        }
    }

    const WEIGHT_FLOOR: f64 = 1e-6;
    let mut bearing_weights: Vec<f64> = bearings.iter().map(|b| b.weight.max(1.0)).collect();

    let mut rhs: HashMap<usize, Vector3<f64>> = HashMap::new();
    for m in &metric_edges {
        *rhs.entry(m.i).or_default() += m.direction.scale(-m.weight * m.length);
        *rhs.entry(m.j).or_default() += m.direction.scale(m.weight * m.length);
    }

    let hess_vec = |bs: &[Bearing],
                    ws: &[f64],
                    metrics: &[MetricEdge],
                    v: &HashMap<usize, Vector3<f64>>|
     -> HashMap<usize, Vector3<f64>> {
        let mut out: HashMap<usize, Vector3<f64>> = HashMap::new();
        for (k, b) in bs.iter().enumerate() {
            let w = ws[k];
            if w <= 0.0 {
                continue;
            }
            let ci = v.get(&b.i).copied().unwrap_or(Vector3::zeros());
            let cj = v.get(&b.j).copied().unwrap_or(Vector3::zeros());
            // J_i v = d × (vi − vj); Aᵀ_i y = −w·(d × y), Aᵀ_j y = +w·(d×y).
            let y = b.direction.cross(&(ci - cj));
            *out.entry(b.i).or_default() += b.direction.cross(&y).scale(-w);
            *out.entry(b.j).or_default() += b.direction.cross(&y).scale(w);
        }
        // Parallel (metric) blocks: (H v)_i = w d (d·(vi−vj)).
        for m in metrics {
            let ci = v.get(&m.i).copied().unwrap_or(Vector3::zeros());
            let cj = v.get(&m.j).copied().unwrap_or(Vector3::zeros());
            let proj = m.direction.scale(m.direction.dot(&(ci - cj)) * m.weight);
            *out.entry(m.i).or_default() += proj;
            *out.entry(m.j).or_default() -= proj;
        }
        for (&node, val) in v {
            *out.entry(node).or_default() += val.scale(PRIOR_WEIGHT);
        }
        out
    };

    let solve_cg = |ws: &[f64], cg_iterations: usize| -> HashMap<usize, Vector3<f64>> {
        let zero_field: HashMap<usize, Vector3<f64>> =
            members.iter().map(|&i| (i, Vector3::zeros())).collect();
        let mut out = zero_field.clone();
        let h0 = hess_vec(
            bearings.as_slice(),
            ws,
            metric_edges.as_slice(),
            &zero_field,
        );
        let mut r: HashMap<usize, Vector3<f64>> = rhs
            .iter()
            .map(|(k, v)| (*k, *v - h0.get(k).copied().unwrap_or(Vector3::zeros())))
            .collect();
        let mut p = r.clone();
        let mut rs_old: f64 = r.values().map(|g| g.norm_squared()).sum();
        for _ in 0..cg_iterations.max(1) {
            if !rs_old.is_finite() || rs_old < 1e-24 {
                break;
            }
            let hp = hess_vec(bearings.as_slice(), ws, metric_edges.as_slice(), &p);
            let denom: f64 = p
                .iter()
                .map(|(k, v)| v.dot(&hp.get(k).copied().unwrap_or(Vector3::zeros())))
                .sum();
            if !(denom.is_finite() && denom > 1e-30) {
                break;
            }
            let alpha = rs_old / denom;
            for (k, v) in &p {
                *out.entry(*k).or_default() += *v * alpha;
            }
            for (k, v) in &hp {
                *r.entry(*k).or_default() -= *v * alpha;
            }
            let rs_new: f64 = r.values().map(|g| g.norm_squared()).sum();
            if !rs_new.is_finite() {
                break;
            }
            let beta = rs_new / rs_old;
            for (k, v) in r.iter() {
                let pk = p.get(k).copied().unwrap_or(Vector3::zeros());
                *p.entry(*k).or_default() = *v + pk.scale(beta);
            }
            rs_old = rs_new;
        }
        out
    };

    let bearing_angle_error = |positions: &HashMap<usize, Vector3<f64>>, k: usize| -> f64 {
        let b = &bearings[k];
        let ci = positions.get(&b.i).copied().unwrap_or(Vector3::zeros());
        let cj = positions.get(&b.j).copied().unwrap_or(Vector3::zeros());
        let disp = cj - ci;
        let norm = disp.norm();
        if norm < 1e-9 {
            return f64::INFINITY;
        }
        (disp.dot(&b.direction) / norm).clamp(-1.0, 1.0).acos()
    };

    let metric_bearing_keys: HashSet<(usize, usize)> = metric_edges
        .iter()
        .map(|m| (m.i.min(m.j), m.i.max(m.j)))
        .collect();
    let is_metric_bearing = |k: usize| -> bool {
        let b = &bearings[k];
        metric_bearing_keys.contains(&(b.i.min(b.j), b.i.max(b.j)))
    };

    let mut positions = solve_cg(bearing_weights.as_slice(), cg_iterations);

    // Graduated rounds: the per-edge effective threshold is the round's base
    // relaxed over ~0.05 rad → ~0.4 rad, doubled for tree-tier edges.
    const ROUNDS: usize = 8;
    let bases: Vec<f64> = (0..ROUNDS)
        .map(|round| 0.05 * (8.0f64).powf(round as f64 / (ROUNDS - 1) as f64))
        .collect();
    for (round, &base) in bases.iter().enumerate() {
        let mut changed = false;
        for k in 0..bearings.len() {
            if is_metric_bearing(k) {
                continue;
            }
            let threshold = if tree_flags[k] { base * 2.0 } else { base };
            let err = bearing_angle_error(&positions, k);
            let factor = if err.is_finite() && err > threshold {
                (threshold / err).max(WEIGHT_FLOOR / bearing_weights[k].max(1e-12))
            } else {
                1.0
            };
            let target = bearing_weights[k] * factor;
            if (target - bearing_weights[k]).abs() > 1e-9 {
                bearing_weights[k] = target;
                changed = true;
            }
        }
        positions = solve_cg(bearing_weights.as_slice(), cg_iterations);
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            let mut errs: Vec<f64> = (0..bearings.len())
                .filter(|&k| !is_metric_bearing(k))
                .map(|k| bearing_angle_error(&positions, k))
                .collect();
            errs.sort_by(|a, b| a.total_cmp(b));
            eprintln!(
                "global-sfm debug: position round {round} base={:.3} rad                  median residual {:.2} deg",
                base,
                errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN).to_degrees()
            );
        }
        if !changed {
            break;
        }
    }

    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        let mut errs: Vec<f64> = (0..bearings.len())
            .filter(|&k| !is_metric_bearing(k))
            .map(|k| bearing_angle_error(&positions, k))
            .collect();
        errs.sort_by(|a, b| a.total_cmp(b));
        eprintln!(
            "global-sfm debug: positions final: {} nodes, {} bearings, median angular residual {:.2} deg",
            positions.len(),
            errs.len(),
            errs.get(errs.len() / 2).copied().unwrap_or(f64::NAN).to_degrees()
        );
    }

    // Re-anchor: seed centre becomes exactly the origin.
    let anchor = positions.get(&seed).copied().unwrap_or(Vector3::zeros());
    let mut out: Vec<Option<Point3<f64>>> = vec![None; rotations.len()];
    for (image, center) in positions {
        out[image] = Some(Point3::from(center - anchor));
    }
    out
}

/// Position averaging with one independent (signed) scale per bearing edge.
/// The legacy solver adds a unit-displacement row to every edge; that silently
/// forces every baseline to have the same length.  This variant eliminates the
/// per-edge scales analytically: each edge contributes the two-dimensional
/// constraint `(I - d dᵀ)(c_j - c_i) = 0`, while one highest-support edge gets a
/// single longitudinal unit row to fix the global gauge.  The resulting solve
/// has only camera-centre unknowns, so dense view graphs do not become
/// rank-deficient merely because every edge has its own nuisance scale.  A
/// deterministic Huber IRLS loop downweights inconsistent bearing lines.  It is
/// intentionally separate from [`average_positions_with_priors`] so callers
/// can A/B the correction. `None` means the bearing graph is not sufficiently
/// constrained for a full-rank solve; callers should retain their existing
/// fallback in that case.
pub fn average_positions_with_independent_edge_scales(
    edges: &[GlobalSfmEdge],
    edge_weights: &[f64],
    rotations: &[Option<UnitQuaternion<f64>>],
    seed: usize,
    iterations: usize,
) -> Option<Vec<Option<Point3<f64>>>> {
    #[derive(Clone)]
    struct Bearing {
        i: usize,
        j: usize,
        direction: Vector3<f64>,
        base_weight: f64,
    }

    let _seed_rotation = rotations.get(seed).copied().flatten()?;
    let mut bearings = Vec::new();
    for (edge_index, edge) in edges.iter().enumerate() {
        if edge.image_i == edge.image_j {
            continue;
        }
        let (Some(q_i), Some(q_j)) = (
            rotations.get(edge.image_i).copied().flatten(),
            rotations.get(edge.image_j).copied().flatten(),
        ) else {
            continue;
        };
        let (_, selected_direction) = edge.select_pose(q_i, q_j);
        let direction = q_i.inverse().transform_vector(&selected_direction);
        let Some(direction) = direction.try_normalize(1e-12) else {
            continue;
        };
        let base_weight = edge_weights
            .get(edge_index)
            .copied()
            .unwrap_or(edge.weight)
            .max(0.0);
        if !(base_weight.is_finite() && base_weight > 0.0) {
            continue;
        }
        bearings.push(Bearing {
            i: edge.image_i,
            j: edge.image_j,
            direction,
            base_weight,
        });
    }
    if bearings.is_empty() {
        return None;
    }

    let members: Vec<usize> = rotations
        .iter()
        .enumerate()
        .filter_map(|(image, rotation)| rotation.as_ref().map(|_| image))
        .collect();
    if members.len() < 2 || !members.contains(&seed) {
        return None;
    }
    // A two-view edge is enough to set the gauge, but a larger tree has
    // unconstrained bending modes (one independent bearing line per edge).
    // Keep the established sparse-tree fallback while allowing cyclic graphs
    // to use the SVD's deterministic minimum-norm solution when the scene
    // itself leaves a small geometric nullspace.
    if members.len() > 2 && bearings.len() < members.len() {
        return None;
    }
    let mut connected = vec![false; rotations.len()];
    let mut adjacency = vec![Vec::new(); rotations.len()];
    for bearing in &bearings {
        adjacency[bearing.i].push(bearing.j);
        adjacency[bearing.j].push(bearing.i);
    }
    let mut stack = vec![seed];
    connected[seed] = true;
    while let Some(image) = stack.pop() {
        for &neighbor in &adjacency[image] {
            if !connected[neighbor] {
                connected[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }
    if members.iter().any(|&image| !connected[image]) {
        return None;
    }
    let mut center_offset = vec![None; rotations.len()];
    let mut center_count = 0usize;
    for &image in &members {
        if image != seed {
            center_offset[image] = Some(center_count);
            center_count += 1;
        }
    }

    // One longitudinal row fixes the otherwise-free global scale.  The
    // perpendicular constraints deliberately allow a signed edge scale: an
    // essential decomposition's antipodal translation has the same bearing
    // line, and forcing positivity here would reintroduce a discrete,
    // order-dependent sign decision before the graph has a pose solution.
    let anchor = bearings
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| {
            a.base_weight
                .total_cmp(&b.base_weight)
                .then_with(|| b.i.cmp(&a.i))
                .then_with(|| b.j.cmp(&a.j))
                .then_with(|| ib.cmp(ia))
        })
        .map(|(index, _)| index)?;
    let variable_count = center_count * 3;
    if variable_count == 0 {
        return None;
    }

    let solve = |weights: &[f64], bearings: &[Bearing]| {
        // Three rows per edge are harmless (the projection has rank two) and
        // keep the matrix construction simple and deterministic.  The final
        // row is the gauge/scale anchor.
        let mut a = DMatrix::<f64>::zeros(bearings.len() * 3 + 1, variable_count);
        let mut b = DVector::<f64>::zeros(bearings.len() * 3 + 1);
        for (edge_index, edge) in bearings.iter().enumerate() {
            let sqrt_weight = weights[edge_index].max(1e-12).sqrt();
            let projection =
                Matrix3::<f64>::identity() - edge.direction * edge.direction.transpose();
            for projected_axis in 0..3 {
                let row = edge_index * 3 + projected_axis;
                for coordinate_axis in 0..3 {
                    let coefficient = sqrt_weight * projection[(projected_axis, coordinate_axis)];
                    if let Some(offset) = center_offset[edge.i] {
                        a[(row, offset * 3 + coordinate_axis)] -= coefficient;
                    }
                    if let Some(offset) = center_offset[edge.j] {
                        a[(row, offset * 3 + coordinate_axis)] += coefficient;
                    }
                }
            }
        }
        let anchor_edge = &bearings[anchor];
        let anchor_row = bearings.len() * 3;
        for axis in 0..3 {
            if let Some(offset) = center_offset[anchor_edge.i] {
                a[(anchor_row, offset * 3 + axis)] -= anchor_edge.direction[axis];
            }
            if let Some(offset) = center_offset[anchor_edge.j] {
                a[(anchor_row, offset * 3 + axis)] += anchor_edge.direction[axis];
            }
        }
        b[anchor_row] = 1.0;
        let svd = a.svd(true, true);
        let largest = svd.singular_values.iter().copied().fold(0.0f64, f64::max);
        let tolerance = (largest * 1e-10).max(1e-12);
        let rank = svd
            .singular_values
            .iter()
            .filter(|&&value| value > tolerance)
            .count();
        if rank < variable_count && std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            let mut tail = svd.singular_values.as_slice().to_vec();
            tail.sort_by(f64::total_cmp);
            tail.truncate(8);
            eprintln!(
                "global-sfm debug: independent edge-scale geometric nullspace rank={rank}/{variable_count} singular_tail={tail:?} tolerance={tolerance:.3e} (minimum-norm solve)"
            );
        }
        let solution = svd
            .solve(&b, tolerance)
            .ok()?
            .iter()
            .copied()
            .collect::<Vec<_>>();
        if solution.iter().all(|value| value.is_finite()) {
            Some((solution, rank))
        } else {
            None
        }
    };

    let mut weights: Vec<f64> = bearings.iter().map(|edge| edge.base_weight).collect();
    let mut final_solution = None;
    let mut final_rank = 0usize;
    let mut final_residuals = Vec::new();
    let rounds = iterations.clamp(1, 8);
    for _round in 0..rounds {
        let (solution, rank) = solve(&weights, &bearings)?;
        let mut scale_magnitudes = Vec::with_capacity(bearings.len());
        let mut residuals = Vec::with_capacity(bearings.len());
        for edge in &bearings {
            let ci = center_offset[edge.i]
                .map(|offset| {
                    Vector3::new(
                        solution[offset * 3],
                        solution[offset * 3 + 1],
                        solution[offset * 3 + 2],
                    )
                })
                .unwrap_or_else(Vector3::zeros);
            let cj = center_offset[edge.j]
                .map(|offset| {
                    Vector3::new(
                        solution[offset * 3],
                        solution[offset * 3 + 1],
                        solution[offset * 3 + 2],
                    )
                })
                .unwrap_or_else(Vector3::zeros);
            let displacement = cj - ci;
            let scale = displacement.dot(&edge.direction);
            let perpendicular = displacement - edge.direction * scale;
            if scale.is_finite() && scale.abs() > 1e-9 {
                scale_magnitudes.push(scale.abs());
            }
            residuals.push(perpendicular.norm());
        }
        if scale_magnitudes.is_empty() || residuals.iter().any(|value| !value.is_finite()) {
            return None;
        }
        scale_magnitudes.sort_by(f64::total_cmp);
        let scale_reference = scale_magnitudes[scale_magnitudes.len() / 2].max(1e-9);
        for residual in &mut residuals {
            *residual /= scale_reference;
        }
        let mut sorted = residuals.clone();
        sorted.sort_by(f64::total_cmp);
        let median = sorted[sorted.len() / 2];
        let mut deviations: Vec<f64> = residuals
            .iter()
            .map(|value| (value - median).abs())
            .collect();
        deviations.sort_by(f64::total_cmp);
        let mad = deviations[deviations.len() / 2];
        // 1.345 is the standard 95%-efficiency Huber transition for Gaussian
        // residuals; 1.4826 converts MAD to a robust sigma estimate.
        let huber_delta = (1.345 * 1.4826 * mad).max(1e-3);
        let next_weights: Vec<f64> = residuals
            .iter()
            .enumerate()
            .map(|(index, residual)| {
                let factor = if *residual > huber_delta {
                    huber_delta / *residual
                } else {
                    1.0
                };
                bearings[index].base_weight * factor.max(1e-6)
            })
            .collect();
        let stable = weights
            .iter()
            .zip(next_weights.iter())
            .all(|(old, new)| (old - new).abs() <= old.max(1.0) * 1e-3);
        final_solution = Some(solution);
        final_rank = rank;
        final_residuals = residuals;
        if stable {
            break;
        }
        weights = next_weights;
    }
    let solution = final_solution?;
    let mut centers = vec![None; rotations.len()];
    centers[seed] = Some(Point3::origin());
    for &image in &members {
        let Some(offset) = center_offset[image] else {
            continue;
        };
        centers[image] = Some(Point3::new(
            solution[offset * 3],
            solution[offset * 3 + 1],
            solution[offset * 3 + 2],
        ));
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        let median = {
            let mut values = final_residuals.clone();
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        };
        let downweighted = weights
            .iter()
            .zip(bearings.iter())
            .filter(|(weight, edge)| **weight < edge.base_weight * 0.999)
            .count();
        let anchor_edge = &bearings[anchor];
        eprintln!(
            "global-sfm debug: independent edge-scale positions edges={} nodes={} rank={}/{} anchor={}-{} median_norm_residual={:.6} downweighted={}",
            bearings.len(),
            members.len(),
            final_rank,
            variable_count,
            anchor_edge.i,
            anchor_edge.j,
            median,
            downweighted,
        );
    }
    Some(centers)
}

/// GLOMAP-style joint camera+point positioning (simplified): with rotations
/// fixed, alternate least-squares updates of 3D points and camera centres so
/// each observation's world bearing aligns with `(X - c)`. Pairwise
/// translation averaging alone cannot resolve chirality-bent façades; track
/// rays supply the missing multi-view scale.
///
/// `init_centers` seeds the solve (typically from [`average_positions`]).
/// `pinned_priors`, when set, freezes those cameras' centres at their prior
/// centres for the whole IRLS (hybrid gauge).
fn refine_centers_joint_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    rotations: &[Option<UnitQuaternion<f64>>],
    init_centers: Vec<Option<Point3<f64>>>,
    seed: usize,
    pinned_priors: Option<&[Option<Pose>]>,
) -> Vec<Option<Point3<f64>>> {
    let n = rotations.len();
    let seed_init = init_centers.get(seed).copied().flatten();
    let mut centers = init_centers;
    if centers.len() < n {
        centers.resize(n, None);
    }
    // World-frame unit bearings per track observation.
    struct Obs {
        image: usize,
        bearing_w: Vector3<f64>,
    }
    let mut track_obs: Vec<Vec<Obs>> = Vec::new();
    for track in tracks {
        if track.len() < 2 {
            continue;
        }
        let mut obs = Vec::with_capacity(track.len());
        for &(image, kp) in track {
            let Some(q) = rotations.get(image).copied().flatten() else {
                continue;
            };
            let Some(feats) = features.get(image) else {
                continue;
            };
            let Some(px) = feats.keypoints.get(kp) else {
                continue;
            };
            let Some(nrm) = camera.normalize_pixel(px) else {
                continue;
            };
            let cam_ray = Vector3::new(nrm.x, nrm.y, 1.0);
            let Some(unit) = cam_ray.try_normalize(1e-12) else {
                continue;
            };
            // Bearing in world: R_w2c^{-1} * ray_cam
            let bearing_w = q.inverse().transform_vector(&unit);
            let Some(bearing_w) = bearing_w.try_normalize(1e-12) else {
                continue;
            };
            obs.push(Obs { image, bearing_w });
        }
        if obs.len() >= 2 {
            track_obs.push(obs);
        }
    }
    if track_obs.is_empty() {
        return centers;
    }

    let pin_center = |image: usize| -> Option<Point3<f64>> {
        pinned_priors
            .and_then(|p| p.get(image))
            .and_then(|o| o.as_ref())
            .map(|pose| pose.camera_center_world())
    };
    for (i, center) in centers.iter_mut().enumerate().take(n) {
        if let Some(c) = pin_center(i) {
            *center = Some(c);
        }
    }

    // Init points: midpoints of first two rays with known centres.
    let mut points: Vec<Option<Point3<f64>>> = vec![None; track_obs.len()];
    for (ti, obs) in track_obs.iter().enumerate() {
        let mut mid = Vector3::zeros();
        let mut count = 0usize;
        for o in obs {
            let Some(c) = centers.get(o.image).copied().flatten() else {
                continue;
            };
            // Unit depth along ray as a crude init; refined below.
            mid += c.coords + o.bearing_w;
            count += 1;
        }
        if count >= 2 {
            points[ti] = Some(Point3::from(mid / count as f64));
        }
    }

    let huber = |r: f64, delta: f64| -> f64 {
        let a = r.abs();
        if a <= delta {
            1.0
        } else {
            delta / a
        }
    };

    const ROUNDS: usize = 8;
    for round in 0..ROUNDS {
        let delta = 0.05 + 0.05 * round as f64; // ~3°→~25° in sin-approx units
                                                // ---- Point update: X minimizes Σ w || û × (X - c) ||^2 ----
        for (ti, obs) in track_obs.iter().enumerate() {
            let mut ata = Matrix3::zeros();
            let mut atb = Vector3::zeros();
            let mut used = 0usize;
            for o in obs {
                let Some(c) = centers.get(o.image).copied().flatten() else {
                    continue;
                };
                let u = o.bearing_w;
                // Skew(u) * X = Skew(u) * c  → two independent rows
                let skew = Matrix3::new(0.0, -u.z, u.y, u.z, 0.0, -u.x, -u.y, u.x, 0.0);
                let residual = if let Some(x) = points[ti] {
                    (u.cross(&(x.coords - c.coords))).norm()
                } else {
                    1.0
                };
                let w = huber(residual, delta).max(1e-3);
                ata += skew.transpose() * skew * w;
                atb += skew.transpose() * (skew * c.coords) * w;
                used += 1;
            }
            if used < 2 {
                continue;
            }
            if let Some(x) = ata.try_inverse().map(|inv| inv * atb) {
                if x.iter().all(|v| v.is_finite()) {
                    points[ti] = Some(Point3::from(x));
                }
            }
        }

        // ---- Camera update: c minimizes Σ w || û × (X - c) ||^2 ----
        // û × (X - c) = û × X - û × c ⇒ Skew(û) c = Skew(û) X
        let mut cam_ata = vec![Matrix3::zeros(); n];
        let mut cam_atb = vec![Vector3::zeros(); n];
        let mut cam_wsum = vec![0.0f64; n];
        for (ti, obs) in track_obs.iter().enumerate() {
            let Some(x) = points[ti] else {
                continue;
            };
            for o in obs {
                if o.image >= n {
                    continue;
                }
                if pin_center(o.image).is_some() {
                    continue;
                }
                let u = o.bearing_w;
                let skew = Matrix3::new(0.0, -u.z, u.y, u.z, 0.0, -u.x, -u.y, u.x, 0.0);
                let residual = if let Some(c) = centers[o.image] {
                    (u.cross(&(x.coords - c.coords))).norm()
                } else {
                    1.0
                };
                let w = huber(residual, delta).max(1e-3);
                cam_ata[o.image] += skew.transpose() * skew * w;
                cam_atb[o.image] += skew.transpose() * (skew * x.coords) * w;
                cam_wsum[o.image] += w;
            }
        }
        for i in 0..n {
            if pin_center(i).is_some() {
                continue;
            }
            if rotations[i].is_none() || cam_wsum[i] < 1e-6 {
                continue;
            }
            if let Some(c) = cam_ata[i].try_inverse().map(|inv| inv * cam_atb[i]) {
                if c.iter().all(|v| v.is_finite()) {
                    centers[i] = Some(Point3::from(c));
                }
            }
        }
        // Keep seed gauge: translate so seed stays at its init (or prior).
        let Some(seed_now) = centers.get(seed).copied().flatten() else {
            continue;
        };
        let seed_target = pin_center(seed).or(seed_init).unwrap_or(Point3::origin());
        let delta_c = seed_target - seed_now;
        if delta_c.norm() > 1e-12 {
            for c in centers.iter_mut().flatten() {
                *c = Point3::from(c.coords + delta_c);
            }
            for p in points.iter_mut().flatten() {
                *p = Point3::from(p.coords + delta_c);
            }
        }
    }
    centers
}

/// Rewrite relative pose on edges whose both endpoints have absolute pose
/// priors: `R_ij` and `direction_ij` come from the prior metric frame, and the
/// edge weight is boosted so free cameras hang off a prior-consistent skeleton.
/// Returns `(rewritten, flipped)` counts (`flipped` ⊆ `rewritten`).
fn repair_edges_from_pose_priors(
    edges: &mut [GlobalSfmEdge],
    edge_weights: &mut [f64],
    priors: &[Option<Pose>],
) -> (usize, usize) {
    let mut rewritten = 0usize;
    let mut flipped = 0usize;
    for (idx, edge) in edges.iter_mut().enumerate() {
        if edge_weights.get(idx).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        let Some(Some(pi)) = priors.get(edge.image_i) else {
            continue;
        };
        let Some(Some(pj)) = priors.get(edge.image_j) else {
            continue;
        };
        let ci = pi.camera_center_world();
        let cj = pj.camera_center_world();
        let ri = pi.world_to_camera.rotation;
        let rj = pj.world_to_camera.rotation;
        let delta = cj.coords - ci.coords;
        if delta.norm() < 1e-9 {
            continue;
        }
        let Some(dir) = ri.transform_vector(&delta).try_normalize(1e-12) else {
            continue;
        };
        if dir.dot(&edge.direction_ij) < 0.0 {
            flipped += 1;
        }
        edge.direction_ij = dir;
        edge.rotation_ij = rj * ri.inverse();
        // Drop alternates — the prior frame is authoritative for this edge.
        edge.rotation_alt = None;
        edge.direction_alt = None;
        if let Some(w) = edge_weights.get_mut(idx) {
            *w = (*w).max(edge.weight) * 2.0;
        }
        rewritten += 1;
    }
    (rewritten, flipped)
}

/// Squared distance between two 3D rays (origins + unit directions).
fn ray_ray_distance_sq(
    origin_a: &Vector3<f64>,
    dir_a: &Vector3<f64>,
    origin_b: &Vector3<f64>,
    dir_b: &Vector3<f64>,
) -> f64 {
    let w0 = origin_a - origin_b;
    let a = dir_a.dot(dir_a);
    let b = dir_a.dot(dir_b);
    let c = dir_b.dot(dir_b);
    let d = dir_a.dot(&w0);
    let e = dir_b.dot(&w0);
    let denom = a * c - b * b;
    if denom.abs() < 1e-18 {
        return w0.cross(dir_a).norm_squared();
    }
    let sc = (b * e - c * d) / denom;
    let tc = (a * e - b * d) / denom;
    let p1 = origin_a + dir_a * sc;
    let p2 = origin_b + dir_b * tc;
    (p1 - p2).norm_squared()
}

/// World-space ray from a pose prior toward a free camera: origin at the
/// prior centre, direction unit-normalized in world coordinates.
fn prior_world_ray_toward_free(
    prior: &Pose,
    bearing_in_prior_frame: &Vector3<f64>,
) -> Option<(Vector3<f64>, Vector3<f64>)> {
    let origin = prior.camera_center_world().coords;
    let dir = prior
        .world_to_camera
        .rotation
        .inverse()
        .transform_vector(bearing_in_prior_frame)
        .try_normalize(1e-12)?;
    Some((origin, dir))
}

/// Build pixel correspondences for a stored pair (essential inliers when set).
fn sampson_distance(
    essential: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
    camera: &Camera,
) -> Option<f64> {
    let prev = camera.normalize_pixel(&correspondence.previous_xy)?;
    let curr = camera.normalize_pixel(&correspondence.current_xy)?;
    let prev_h = Vector3::new(prev.x, prev.y, 1.0);
    let curr_h = Vector3::new(curr.x, curr.y, 1.0);
    let e_prev = essential * prev_h;
    let et_curr = essential.transpose() * curr_h;
    let numerator = curr_h.dot(&e_prev).powi(2);
    let denominator = e_prev.x.powi(2) + e_prev.y.powi(2) + et_curr.x.powi(2) + et_curr.y.powi(2);
    if denominator < 1e-18 {
        return None;
    }
    Some((numerator / denominator).sqrt())
}

/// Mean Sampson distance (normalized coords) over essential inliers.
pub fn pair_essential_mean_sampson_error(
    pair: &PairwiseMatches,
    features: &[FeatureSet],
    camera: &Camera,
) -> Option<f64> {
    let essential = pair.essential_matrix.as_ref()?;
    let corrs = pair_correspondences(pair, features, true);
    if corrs.is_empty() {
        return None;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for corr in &corrs {
        if let Some(d) = sampson_distance(essential, corr, camera) {
            total += d;
            count += 1.0;
        }
    }
    if count > 0.0 {
        Some(total / count)
    } else {
        None
    }
}

pub fn pair_correspondences(
    pair: &PairwiseMatches,
    features: &[FeatureSet],
    prefer_essential: bool,
) -> Vec<TwoViewCorrespondence> {
    let matches = if prefer_essential {
        pair.essential_matches
            .as_ref()
            .filter(|e| !e.is_empty())
            .unwrap_or(&pair.matches)
    } else {
        &pair.matches
    };
    matches
        .iter()
        .filter_map(|&(ki, kj)| {
            Some(TwoViewCorrespondence::new(
                *features[pair.image_i].keypoints.get(ki)?,
                *features[pair.image_j].keypoints.get(kj)?,
            ))
        })
        .collect()
}

pub fn relative_pose_from_essential(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<RelativePose> {
    if correspondences.len() < 8 {
        return None;
    }
    let inliers: Vec<usize> = (0..correspondences.len()).collect();
    let cheirality = CheiralityOptions::hardened_keep_ambiguous();
    let recovered = recover_relative_pose_with_options(
        essential,
        correspondences,
        camera,
        &inliers,
        &cheirality,
    )?;
    let scale = RelativePoseEstimator::default().default_translation_scale;
    let se3 = SE3::new(recovered.rotation, recovered.translation_unit * scale);
    Some(RelativePose {
        previous_to_current: se3,
        translation_unit: recovered.translation_unit,
        translation_scale: scale,
        inliers,
        mean_sampson_error: 0.0,
        alternate: recovered.alternate,
        chirality_margin: recovered.chirality_margin(),
    })
}

/// Unit bearing from `prior_idx` toward `free_idx` under the primary/alternate
/// essential hypothesis (`pair` is ordered `image_i` / `image_j`).
fn bearing_prior_to_free(
    pair: &PairwiseMatches,
    prior_idx: usize,
    free_idx: usize,
    features: &[FeatureSet],
    camera: &Camera,
    use_alternate: bool,
) -> Option<Vector3<f64>> {
    let essential = pair.essential_matrix.as_ref()?;
    let corrs = pair_correspondences(pair, features, true);
    let rel = relative_pose_from_essential(essential, &corrs, camera)?;
    let prior_is_i = pair.image_i == prior_idx && pair.image_j == free_idx;
    let prior_is_j = pair.image_j == prior_idx && pair.image_i == free_idx;
    if !prior_is_i && !prior_is_j {
        return None;
    }
    let (r, t_unit) = if use_alternate {
        let (r_alt, t_alt) = rel.alternate.as_ref()?;
        (r_alt, t_alt)
    } else {
        (&rel.previous_to_current.rotation, &rel.translation_unit)
    };
    let t = t_unit * rel.translation_scale;
    let d_ij = (-r.inverse().transform_vector(&t)).try_normalize(1e-12)?;
    if prior_is_i {
        Some(d_ij)
    } else {
        (-r.transform_vector(&d_ij)).try_normalize(1e-12)
    }
}

fn triangulate_two_rays(
    origin_a: &Vector3<f64>,
    dir_a: &Vector3<f64>,
    origin_b: &Vector3<f64>,
    dir_b: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let w0 = origin_a - origin_b;
    let a = dir_a.dot(dir_a);
    let b = dir_a.dot(dir_b);
    let c = dir_b.dot(dir_b);
    let d = dir_a.dot(&w0);
    let e = dir_b.dot(&w0);
    let denom = a * c - b * b;
    if denom.abs() < 1e-18 {
        return None;
    }
    let sc = (b * e - c * d) / denom;
    let tc = (a * e - b * d) / denom;
    Some((origin_a + dir_a * sc + origin_b + dir_b * tc) * 0.5)
}

fn ray_point_angular_error(origin: &Vector3<f64>, dir: &Vector3<f64>, point: &Vector3<f64>) -> f64 {
    let toward = point - origin;
    let n = toward.norm();
    if n < 1e-9 {
        return std::f64::consts::PI;
    }
    dir.normalize().dot(&(toward / n)).clamp(-1.0, 1.0).acos()
}

/// Gate for free↔prior rematch E-gains: optional chirality margin floor and
/// optional triangulation anchor from two other prior↔free essentials.
#[allow(clippy::too_many_arguments)]
pub fn rematch_essential_admission_ok(
    pair: &PairwiseMatches,
    prior_idx: usize,
    free_idx: usize,
    features: &[FeatureSet],
    camera: &Camera,
    pose_priors: &[Option<Pose>],
    existing: &[PairwiseMatches],
    min_chirality_margin: f64,
    require_prior_anchor: bool,
    min_anchor_e_inliers: usize,
) -> bool {
    let Some(essential) = pair.essential_matrix.as_ref() else {
        return !require_prior_anchor;
    };
    let corrs = pair_correspondences(pair, features, true);
    let Some(rel) = relative_pose_from_essential(essential, &corrs, camera) else {
        return false;
    };
    if min_chirality_margin > 0.0 && rel.chirality_margin < min_chirality_margin {
        return false;
    }
    if !require_prior_anchor {
        return true;
    }
    let prior_pose = match pose_priors.get(prior_idx).and_then(|p| p.as_ref()) {
        Some(p) => p,
        None => return true,
    };
    let mut anchors: Vec<(usize, usize)> = Vec::new();
    for other in existing {
        if other.essential_matrix.is_none() {
            continue;
        }
        let e_count = other
            .essential_matches
            .as_ref()
            .map(|e| e.len())
            .unwrap_or(0);
        if e_count < min_anchor_e_inliers {
            continue;
        }
        let other_prior = if other.image_i == free_idx {
            let k = other.image_j;
            if pose_priors.get(k).and_then(|p| p.as_ref()).is_some() {
                k
            } else {
                continue;
            }
        } else if other.image_j == free_idx {
            let k = other.image_i;
            if pose_priors.get(k).and_then(|p| p.as_ref()).is_some() {
                k
            } else {
                continue;
            }
        } else {
            continue;
        };
        if other_prior == prior_idx {
            continue;
        }
        anchors.push((other_prior, e_count));
    }
    anchors.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    anchors.dedup_by_key(|(k, _)| *k);
    if anchors.len() < 2 {
        return true;
    }
    let (k1, _) = anchors[0];
    let (k2, _) = anchors[1];
    let Some(pose_k1) = pose_priors.get(k1).and_then(|p| p.as_ref()) else {
        return true;
    };
    let Some(pose_k2) = pose_priors.get(k2).and_then(|p| p.as_ref()) else {
        return true;
    };
    let Some(pair_k1) = existing.iter().find(|p| {
        (p.image_i == k1 && p.image_j == free_idx) || (p.image_j == k1 && p.image_i == free_idx)
    }) else {
        return true;
    };
    let Some(pair_k2) = existing.iter().find(|p| {
        (p.image_i == k2 && p.image_j == free_idx) || (p.image_j == k2 && p.image_i == free_idx)
    }) else {
        return true;
    };
    let Some(b1) = bearing_prior_to_free(pair_k1, k1, free_idx, features, camera, false) else {
        return true;
    };
    let Some(b2) = bearing_prior_to_free(pair_k2, k2, free_idx, features, camera, false) else {
        return true;
    };
    let Some((o1, d1)) = prior_world_ray_toward_free(pose_k1, &b1) else {
        return true;
    };
    let Some((o2, d2)) = prior_world_ray_toward_free(pose_k2, &b2) else {
        return true;
    };
    let Some(anchor) = triangulate_two_rays(&o1, &d1, &o2, &d2) else {
        return true;
    };
    let Some(b_primary) = bearing_prior_to_free(pair, prior_idx, free_idx, features, camera, false)
    else {
        return false;
    };
    let Some((origin_p, dir_primary)) = prior_world_ray_toward_free(prior_pose, &b_primary) else {
        return false;
    };
    let primary_err = ray_point_angular_error(&origin_p, &dir_primary, &anchor);
    let alt_err = bearing_prior_to_free(pair, prior_idx, free_idx, features, camera, true)
        .and_then(|b_alt| {
            prior_world_ray_toward_free(prior_pose, &b_alt)
                .map(|(o, d)| ray_point_angular_error(&o, &d, &anchor))
        })
        .unwrap_or(std::f64::consts::PI);
    primary_err + 1e-3 < alt_err
}

/// Bearing from camera `from` toward camera `to` in `from`'s frame, using the
/// edge's primary or alternate relative pose.
fn edge_bearing_in_frame(
    edge: &GlobalSfmEdge,
    from: usize,
    use_alternate: bool,
) -> Option<Vector3<f64>> {
    if edge.image_i == from {
        if use_alternate {
            edge.direction_alt.as_ref().copied()
        } else {
            Some(edge.direction_ij)
        }
    } else if edge.image_j == from {
        let r = if use_alternate {
            edge.rotation_alt.as_ref()?
        } else {
            &edge.rotation_ij
        };
        let dir = if use_alternate {
            edge.direction_alt.as_ref()?
        } else {
            &edge.direction_ij
        };
        (-r.transform_vector(dir)).try_normalize(1e-12)
    } else {
        None
    }
}

/// Least-squares closest point to a set of 3D rays `(origin, direction)`.
fn triangulate_point_from_rays(rays: &[(Vector3<f64>, Vector3<f64>)]) -> Option<Vector3<f64>> {
    if rays.len() < 2 {
        return None;
    }
    let mut ata = Matrix3::<f64>::zeros();
    let mut atb = Vector3::<f64>::zeros();
    for (origin, direction) in rays {
        let d = direction.normalize();
        let proj = Matrix3::identity() - d * d.transpose();
        ata += proj;
        atb += proj * origin;
    }
    ata.try_inverse().map(|inv| inv * atb)
}

/// Unit bearing in a prior camera's frame toward a world-space point.
fn expected_bearing_prior_to_world_point(
    prior: &Pose,
    world_point: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    let centre = prior.camera_center_world().coords;
    let toward = world_point - centre;
    prior
        .world_to_camera
        .rotation
        .transform_vector(&toward)
        .try_normalize(1e-12)
}

/// Triangulate free-camera centres from prior↔free essentials using incremental
/// pose priors as the metric frame (external anchor vs self-consistent rays).
#[allow(clippy::type_complexity)]
pub fn estimate_free_centres_from_prior_rays(
    pairwise: &[PairwiseMatches],
    features: &[FeatureSet],
    camera: &Camera,
    pose_priors: &[Option<Pose>],
    min_rays: usize,
    min_e_inliers: usize,
) -> HashMap<usize, Vector3<f64>> {
    let is_prior = |i: usize| pose_priors.get(i).and_then(|p| p.as_ref()).is_some();
    let mut rays_by_free: HashMap<usize, Vec<(Vector3<f64>, Vector3<f64>)>> = HashMap::new();
    for pair in pairwise {
        let pi = is_prior(pair.image_i);
        let pj = is_prior(pair.image_j);
        if pi == pj {
            continue;
        }
        let (prior_idx, free_idx) = if pi {
            (pair.image_i, pair.image_j)
        } else {
            (pair.image_j, pair.image_i)
        };
        let e_count = pair
            .essential_matches
            .as_ref()
            .map(|e| e.len())
            .unwrap_or(0);
        if e_count < min_e_inliers {
            continue;
        }
        let Some(prior_pose) = pose_priors.get(prior_idx).and_then(|p| p.as_ref()) else {
            continue;
        };
        let Some(bearing) =
            bearing_prior_to_free(pair, prior_idx, free_idx, features, camera, false)
        else {
            continue;
        };
        let Some((origin, dir)) = prior_world_ray_toward_free(prior_pose, &bearing) else {
            continue;
        };
        rays_by_free
            .entry(free_idx)
            .or_default()
            .push((origin, dir));
    }
    let mut centres = HashMap::new();
    for (free, rays) in rays_by_free {
        if rays.len() < min_rays {
            continue;
        }
        if let Some(point) = triangulate_point_from_rays(&rays) {
            centres.insert(free, point);
        }
    }
    centres
}

/// Approximate free-camera poses by scaling a prior↔free essential edge to the
/// metric baseline implied by multi-ray centre triangulation in the incremental
/// prior frame.
pub fn estimate_free_poses_from_prior_rays(
    pairwise: &[PairwiseMatches],
    features: &[FeatureSet],
    camera: &Camera,
    pose_priors: &[Option<Pose>],
    min_rays: usize,
    min_anchor_e_inliers: usize,
) -> HashMap<usize, Pose> {
    let centres = estimate_free_centres_from_prior_rays(
        pairwise,
        features,
        camera,
        pose_priors,
        min_rays,
        min_anchor_e_inliers,
    );
    let is_prior = |i: usize| pose_priors.get(i).and_then(|p| p.as_ref()).is_some();
    let mut poses = HashMap::new();
    for (free_idx, centre) in centres {
        let mut best: Option<(usize, usize)> = None;
        for pair in pairwise {
            let pi = is_prior(pair.image_i);
            let pj = is_prior(pair.image_j);
            if pi == pj {
                continue;
            }
            let (prior_idx, other) = if pi {
                (pair.image_i, pair.image_j)
            } else {
                (pair.image_j, pair.image_i)
            };
            if other != free_idx {
                continue;
            }
            let e_count = pair
                .essential_matches
                .as_ref()
                .map(|e| e.len())
                .unwrap_or(0);
            if e_count < min_anchor_e_inliers {
                continue;
            }
            if best.is_none_or(|(_, c)| e_count > c) {
                best = Some((prior_idx, e_count));
            }
        }
        let Some((anchor_prior, _)) = best else {
            continue;
        };
        let Some(prior_pose) = pose_priors.get(anchor_prior).and_then(|p| p.as_ref()) else {
            continue;
        };
        let pair = pairwise.iter().find(|p| {
            (p.image_i == anchor_prior && p.image_j == free_idx)
                || (p.image_j == anchor_prior && p.image_i == free_idx)
        });
        let Some(pair) = pair else { continue };
        let Some(essential) = pair.essential_matrix.as_ref() else {
            continue;
        };
        let corrs = pair_correspondences(pair, features, true);
        let Some(rel) = relative_pose_from_essential(essential, &corrs, camera) else {
            continue;
        };
        let c_prior = prior_pose.camera_center_world().coords;
        let dist = (centre - c_prior).norm();
        if dist < 1e-3 {
            continue;
        }
        let prior_is_i = pair.image_i == anchor_prior;
        let rel_se3 = SE3::new(
            rel.previous_to_current.rotation,
            rel.translation_unit * dist,
        );
        let w2c = if prior_is_i {
            rel_se3.compose(&prior_pose.world_to_camera)
        } else {
            rel_se3.inverse().compose(&prior_pose.world_to_camera)
        };
        poses.insert(
            free_idx,
            Pose {
                world_to_camera: w2c,
            },
        );
    }
    poses
}

/// Unit bearing from `from` camera toward `to` camera, in `from`'s frame (GT).
pub fn gt_bearing_in_prior_frame(from: &Pose, to: &Pose) -> Option<Vector3<f64>> {
    let delta = to.camera_center_world().coords - from.camera_center_world().coords;
    from.world_to_camera
        .rotation
        .transform_vector(&delta)
        .try_normalize(1e-12)
}

/// Angular error in degrees between two unit bearings (antipodal-aware).
pub fn bearing_alignment_error_deg(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let dot = a.normalize().dot(&b.normalize()).abs().clamp(-1.0, 1.0);
    dot.acos().to_degrees()
}

/// Essential primary-bearing error vs GT for a prior↔free pair (degrees).
pub fn prior_free_essential_gt_bearing_error_deg(
    pair: &PairwiseMatches,
    prior_idx: usize,
    free_idx: usize,
    features: &[FeatureSet],
    camera: &Camera,
    gt_prior: &Pose,
    gt_free: &Pose,
) -> Option<f64> {
    let gt = gt_bearing_in_prior_frame(gt_prior, gt_free)?;
    let est = bearing_prior_to_free(pair, prior_idx, free_idx, features, camera, false)?;
    Some(bearing_alignment_error_deg(&est, &gt))
}

/// Bearing from `image_i` toward `image_j` in `image_i`'s frame.
pub fn edge_bearing_i_to_j(
    _r_ij: &UnitQuaternion<f64>,
    direction_ij: &Vector3<f64>,
) -> Vector3<f64> {
    *direction_ij
}

/// Bearing from `image_j` toward `image_i` in `image_j`'s frame.
pub fn edge_bearing_j_to_i(
    r_ij: &UnitQuaternion<f64>,
    direction_ij: &Vector3<f64>,
) -> Option<Vector3<f64>> {
    (-r_ij.transform_vector(direction_ij)).try_normalize(1e-12)
}

/// For prior↔free edges with chirality alternates, swap primary/alternate
/// when the alternate bearing agrees better with other prior rays to the same
/// free camera (multi-view centre-ray consensus).
fn apply_prior_guided_free_chirality(edges: &mut [GlobalSfmEdge], priors: &[Option<Pose>]) {
    let is_prior = |i: usize| -> bool { priors.get(i).and_then(|p| p.as_ref()).is_some() };
    let mut flipped = 0usize;
    let mut candidates = 0usize;
    let mut free_nodes: HashSet<usize> = HashSet::new();
    for edge in edges.iter() {
        if edge.weight <= 0.0 {
            continue;
        }
        let pi = is_prior(edge.image_i);
        let pj = is_prior(edge.image_j);
        if pi == pj {
            continue;
        }
        let free = if pi { edge.image_j } else { edge.image_i };
        if edge.direction_alt.is_none() {
            continue;
        }
        free_nodes.insert(free);
    }
    for free in free_nodes {
        let incident: Vec<usize> = edges
            .iter()
            .enumerate()
            .filter(|(_, e)| {
                e.weight > 0.0
                    && e.direction_alt.is_some()
                    && ((e.image_i == free && is_prior(e.image_j))
                        || (e.image_j == free && is_prior(e.image_i)))
            })
            .map(|(idx, _)| idx)
            .collect();
        if incident.len() < 2 {
            continue;
        }
        for &edge_idx in &incident {
            candidates += 1;
            let edge = &edges[edge_idx];
            let prior_idx = if is_prior(edge.image_i) {
                edge.image_i
            } else {
                edge.image_j
            };
            let Some(prior_pose) = priors.get(prior_idx).and_then(|p| p.as_ref()) else {
                continue;
            };
            let score_hypothesis = |use_alt_on_edge: bool| -> f64 {
                let Some((origin, dir)) = edge_bearing_in_frame(edge, prior_idx, use_alt_on_edge)
                    .and_then(|b| prior_world_ray_toward_free(prior_pose, &b))
                else {
                    return f64::INFINITY;
                };
                let mut sum = 0.0f64;
                let mut count = 0usize;
                for &other_idx in &incident {
                    if other_idx == edge_idx {
                        continue;
                    }
                    let other = &edges[other_idx];
                    let other_prior = if other.image_i == free {
                        other.image_j
                    } else {
                        other.image_i
                    };
                    let Some(other_pose) = priors.get(other_prior).and_then(|p| p.as_ref()) else {
                        continue;
                    };
                    let Some(bearing) = edge_bearing_in_frame(other, other_prior, false) else {
                        continue;
                    };
                    let Some((o2, d2)) = prior_world_ray_toward_free(other_pose, &bearing) else {
                        continue;
                    };
                    sum += ray_ray_distance_sq(&origin, &dir, &o2, &d2);
                    count += 1;
                }
                if count == 0 {
                    f64::INFINITY
                } else {
                    sum / count as f64
                }
            };
            let primary_score = score_hypothesis(false);
            let alt_score = score_hypothesis(true);
            if alt_score + 1e-6 < primary_score {
                let edge = &mut edges[edge_idx];
                if let Some(mut r_alt) = edge.rotation_alt.take() {
                    std::mem::swap(&mut edge.rotation_ij, &mut r_alt);
                    edge.rotation_alt = Some(r_alt);
                }
                if let Some(mut d_alt) = edge.direction_alt.take() {
                    std::mem::swap(&mut edge.direction_ij, &mut d_alt);
                    edge.direction_alt = Some(d_alt);
                }
                flipped += 1;
            }
        }
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: prior-guided free chirality: {candidates} candidate edge(s), flipped {flipped}"
        );
    }
}

/// Rewrite edges incident to at least one free (non-prior) camera using a
/// merged pose field (priors where pinned, pass-1 solve elsewhere). Prior–
/// prior edges are left to [`repair_edges_from_pose_priors`]. Returns
/// `(rewritten, flipped)`.
fn repair_free_incident_edges_from_poses(
    edges: &mut [GlobalSfmEdge],
    pose_field: &[Option<Pose>],
    priors: &[Option<Pose>],
    only_flipped: bool,
    limit_indices: &HashSet<usize>,
) -> (usize, usize) {
    let is_prior = |i: usize| -> bool { priors.get(i).and_then(|p| p.as_ref()).is_some() };
    let mut rewritten = 0usize;
    let mut flipped = 0usize;
    for edge in edges.iter_mut() {
        if edge.weight <= 0.0 {
            continue;
        }
        let free_incident = !is_prior(edge.image_i) || !is_prior(edge.image_j);
        if !free_incident {
            continue;
        }
        if !limit_indices.is_empty()
            && !limit_indices.contains(&edge.image_i)
            && !limit_indices.contains(&edge.image_j)
        {
            continue;
        }
        let (Some(pi), Some(pj)) = (
            pose_field.get(edge.image_i).and_then(|p| p.as_ref()),
            pose_field.get(edge.image_j).and_then(|p| p.as_ref()),
        ) else {
            continue;
        };
        let ci = pi.camera_center_world();
        let cj = pj.camera_center_world();
        let ri = pi.world_to_camera.rotation;
        let rj = pj.world_to_camera.rotation;
        let delta = cj.coords - ci.coords;
        if delta.norm() < 1e-9 {
            continue;
        }
        let Some(dir) = ri.transform_vector(&delta).try_normalize(1e-12) else {
            continue;
        };
        let antipodal = dir.dot(&edge.direction_ij) < 0.0;
        if only_flipped && !antipodal {
            continue;
        }
        if antipodal {
            flipped += 1;
        }
        edge.direction_ij = dir;
        edge.rotation_ij = rj * ri.inverse();
        edge.rotation_alt = None;
        edge.direction_alt = None;
        edge.weight = edge.weight.max(1.0) * 1.5;
        rewritten += 1;
    }
    (rewritten, flipped)
}

/// Drop free-incident edges antipodal to a pass-1 pose field. Returns
/// `(dropped, antipodal_count)`.
fn drop_antipodal_free_incident_edges(
    edges: &mut [GlobalSfmEdge],
    pose_field: &[Option<Pose>],
    priors: &[Option<Pose>],
    limit_indices: &HashSet<usize>,
) -> (usize, usize) {
    let is_prior = |i: usize| -> bool { priors.get(i).and_then(|p| p.as_ref()).is_some() };
    let mut dropped = 0usize;
    let mut antipodal = 0usize;
    for edge in edges.iter_mut() {
        if edge.weight <= 0.0 {
            continue;
        }
        let free_incident = !is_prior(edge.image_i) || !is_prior(edge.image_j);
        if !free_incident {
            continue;
        }
        if !limit_indices.is_empty()
            && !limit_indices.contains(&edge.image_i)
            && !limit_indices.contains(&edge.image_j)
        {
            continue;
        }
        let (Some(pi), Some(pj)) = (
            pose_field.get(edge.image_i).and_then(|p| p.as_ref()),
            pose_field.get(edge.image_j).and_then(|p| p.as_ref()),
        ) else {
            continue;
        };
        let ci = pi.camera_center_world();
        let cj = pj.camera_center_world();
        let ri = pi.world_to_camera.rotation;
        let delta = cj.coords - ci.coords;
        if delta.norm() < 1e-9 {
            continue;
        }
        let Some(dir) = ri.transform_vector(&delta).try_normalize(1e-12) else {
            continue;
        };
        if dir.dot(&edge.direction_ij) < 0.0 {
            antipodal += 1;
            edge.weight = -1.0;
            dropped += 1;
        }
    }
    (dropped, antipodal)
}

/// Re-score each edge's translation direction under the globally-averaged
/// relative rotation. Candidates are ±the pairwise direction plus an optional
/// fixed-R epipolar nullspace estimate from the retained inlier sample.
/// Returns how many edges flipped relative to their incoming direction.
fn refine_edge_directions_under_rotations(
    edges: &mut [GlobalSfmEdge],
    rotations: &[Option<UnitQuaternion<f64>>],
    edge_weights: &[f64],
    camera: &Camera,
) -> usize {
    let mut flipped = 0usize;
    for (idx, edge) in edges.iter_mut().enumerate() {
        if edge_weights.get(idx).copied().unwrap_or(0.0) <= 0.0 {
            continue;
        }
        if edge.inlier_sample.len() < 8 {
            continue;
        }
        let (Some(qi), Some(qj)) = (
            rotations.get(edge.image_i).copied().flatten(),
            rotations.get(edge.image_j).copied().flatten(),
        ) else {
            continue;
        };
        // Consensus relative rotation: R_w2c_j = R_ij · R_w2c_i.
        let r_ij = qj * qi.inverse();
        let r_mat: Matrix3<f64> = r_ij.to_rotation_matrix().into_inner();
        let mut candidates = vec![edge.direction_ij, -edge.direction_ij];
        if let Some(est) = estimate_direction_fixed_rotation(camera, &r_mat, &edge.inlier_sample) {
            candidates.push(est);
            candidates.push(-est);
        }
        let mut best_dir = edge.direction_ij;
        let mut best_score: i64 = -1;
        for dir in &candidates {
            let Some(unit) = dir.try_normalize(1e-12) else {
                continue;
            };
            // Camera-j centre in cam-i is `unit`; essential translation t
            // satisfies C = −Rᵀ t ⇒ t = −R C.
            let t = -(r_mat * unit);
            let score = cheirality_count_samples(camera, &r_mat, &t, &edge.inlier_sample);
            if score > best_score {
                best_score = score;
                best_dir = unit;
            }
        }
        if best_score <= 0 {
            continue;
        }
        if best_dir.dot(&edge.direction_ij) < 0.0 {
            flipped += 1;
        }
        edge.direction_ij = best_dir;
        edge.rotation_ij = r_ij;
    }
    flipped
}

/// Fixed-R epipolar estimate of the camera-j centre direction in cam-i.
/// Solves `t · (x₂ × R x₁) = 0` over the sample, then `C = −Rᵀ t`.
fn estimate_direction_fixed_rotation(
    camera: &Camera,
    r_mat: &Matrix3<f64>,
    samples: &[(Point2<f64>, Point2<f64>)],
) -> Option<Vector3<f64>> {
    let mut ata = Matrix3::zeros();
    let mut used = 0usize;
    for (p1, p2) in samples {
        let Some(n1) = camera.normalize_pixel(p1) else {
            continue;
        };
        let Some(n2) = camera.normalize_pixel(p2) else {
            continue;
        };
        let x1 = Vector3::new(n1.x, n1.y, 1.0);
        let x2 = Vector3::new(n2.x, n2.y, 1.0);
        let rx1 = r_mat * x1;
        let cross = x2.cross(&rx1);
        ata += cross * cross.transpose();
        used += 1;
    }
    if used < 8 {
        return None;
    }
    let svd = ata.symmetric_eigen();
    // Smallest eigenvector of AᵀA.
    let mut best_i = 0usize;
    let mut best_v = f64::INFINITY;
    for i in 0..3 {
        let v = svd.eigenvalues[i];
        if v < best_v {
            best_v = v;
            best_i = i;
        }
    }
    let t = svd.eigenvectors.column(best_i).into_owned();
    let t = t.try_normalize(1e-12)?;
    (-r_mat.transpose() * t).try_normalize(1e-12)
}

fn cheirality_count_samples(
    camera: &Camera,
    r_mat: &Matrix3<f64>,
    t: &Vector3<f64>,
    samples: &[(Point2<f64>, Point2<f64>)],
) -> i64 {
    use nalgebra::{DMatrix, Matrix3x4};
    let p_prev = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p_curr = Matrix3x4::zeros();
    p_curr.fixed_view_mut::<3, 3>(0, 0).copy_from(r_mat);
    p_curr.fixed_view_mut::<3, 1>(0, 3).copy_from(t);
    let mut score = 0i64;
    for (p1, p2) in samples {
        let Some(prev) = camera.normalize_pixel(p1) else {
            continue;
        };
        let Some(curr) = camera.normalize_pixel(p2) else {
            continue;
        };
        let mut a = DMatrix::<f64>::zeros(4, 4);
        for column in 0..4 {
            a[(0, column)] = prev.x * p_prev[(2, column)] - p_prev[(0, column)];
            a[(1, column)] = prev.y * p_prev[(2, column)] - p_prev[(1, column)];
            a[(2, column)] = curr.x * p_curr[(2, column)] - p_curr[(0, column)];
            a[(3, column)] = curr.y * p_curr[(2, column)] - p_curr[(1, column)];
        }
        let svd = a.svd(true, true);
        let Some(v_t) = svd.v_t else {
            continue;
        };
        let solution = v_t.row(v_t.nrows() - 1);
        let w = solution[3];
        if w.abs() < 1e-12 {
            continue;
        }
        let world = Vector3::new(solution[0] / w, solution[1] / w, solution[2] / w);
        let camera_curr = r_mat * world + t;
        if world.z > 0.0 && camera_curr.z > 0.0 {
            score += 1;
        }
    }
    score
}

/// Drop incremental priors whose centres disagree with a free (rotation-
/// pinned, centre-free) global position solve after leave-one-out Sim(3)
/// alignment onto the other priors. Bent hubs that are locally consistent
/// with wrong edges still stick out against the free gauge. Returns filtered
/// priors and how many were kept.
pub fn filter_pose_priors_by_free_centre_residual(
    priors: &[Option<Pose>],
    free_centers: &[Option<Point3<f64>>],
    max_drops: usize,
    min_ratio_to_median: f64,
) -> (Vec<Option<Pose>>, usize) {
    let mut pairs: Vec<(usize, Point3<f64>, Point3<f64>)> = Vec::new();
    let n = priors.len().min(free_centers.len());
    for i in 0..n {
        let (Some(prior), Some(free_c)) = (priors[i].as_ref(), free_centers[i]) else {
            continue;
        };
        pairs.push((i, free_c, prior.camera_center_world()));
    }
    if pairs.len() < 4 || max_drops == 0 {
        let kept = priors.iter().filter(|p| p.is_some()).count();
        return (priors.to_vec(), kept);
    }
    let mut residuals: Vec<(usize, f64)> = Vec::with_capacity(pairs.len());
    for leave in 0..pairs.len() {
        let src: Vec<Point3<f64>> = pairs
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != leave)
            .map(|(_, p)| p.1)
            .collect();
        let dst: Vec<Point3<f64>> = pairs
            .iter()
            .enumerate()
            .filter(|&(k, _)| k != leave)
            .map(|(_, p)| p.2)
            .collect();
        let Some(fit) = umeyama_similarity_transform(&src, &dst, true) else {
            continue;
        };
        if !(fit.scale.is_finite() && fit.scale > 0.0) {
            continue;
        }
        let sim = Sim3::new(
            UnitQuaternion::from_rotation_matrix(&fit.rotation),
            fit.translation,
            fit.scale,
        );
        let (i, free_c, prior_c) = pairs[leave];
        let aligned = sim.transform_point(&free_c);
        residuals.push((i, (aligned - prior_c).norm()));
    }
    if residuals.is_empty() {
        let kept = priors.iter().filter(|p| p.is_some()).count();
        return (priors.to_vec(), kept);
    }
    residuals.sort_by(|a, b| a.1.total_cmp(&b.1));
    let median = residuals[residuals.len() / 2].1;
    let threshold = (median * min_ratio_to_median).max(1e-6);
    let mut candidates: Vec<(usize, f64)> = residuals
        .into_iter()
        .filter(|&(_, r)| r >= threshold)
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let drop_set: HashSet<usize> = candidates
        .into_iter()
        .take(max_drops)
        .map(|(i, _)| i)
        .collect();
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: free-centre LOO residual filter median={median:.4} \
             thresh={threshold:.4} drop={drop_set:?}"
        );
    }
    let mut kept = 0usize;
    let filtered: Vec<Option<Pose>> = priors
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if drop_set.contains(&i) {
                None
            } else if p.is_some() {
                kept += 1;
                p.clone()
            } else {
                None
            }
        })
        .collect();
    (filtered, kept)
}

/// Drop incremental priors whose centres systematically disagree with
/// two-view edge bearings on prior–prior pairs (GT-free). For each prior,
/// `flip_fraction` is the weight-weighted share of incident prior–prior
/// edges whose essential `direction_ij` is antipodal to the prior
/// displacement. Priors with `incident >= min_incident` and
/// `flip_fraction >= min_flip_fraction` are candidates; at most
/// `max_drops` worst (highest flip fraction) are cleared.
///
/// Returns filtered priors and how many were kept.
pub fn filter_pose_priors_by_edge_disagreement(
    priors: &[Option<Pose>],
    edges: &[GlobalSfmEdge],
    min_incident: usize,
    min_flip_fraction: f64,
    max_drops: usize,
) -> (Vec<Option<Pose>>, usize) {
    let mut flip_w = vec![0.0f64; priors.len()];
    let mut total_w = vec![0.0f64; priors.len()];
    let mut incident = vec![0usize; priors.len()];
    for edge in edges {
        if edge.weight <= 0.0 {
            continue;
        }
        let (Some(Some(pi)), Some(Some(pj))) = (priors.get(edge.image_i), priors.get(edge.image_j))
        else {
            continue;
        };
        let delta = pj.camera_center_world().coords - pi.camera_center_world().coords;
        if delta.norm() < 1e-9 {
            continue;
        }
        let Some(dir) = pi
            .world_to_camera
            .rotation
            .transform_vector(&delta)
            .try_normalize(1e-12)
        else {
            continue;
        };
        let w = edge.weight.max(1.0);
        let flipped = dir.dot(&edge.direction_ij) < 0.0;
        for &idx in &[edge.image_i, edge.image_j] {
            incident[idx] += 1;
            total_w[idx] += w;
            if flipped {
                flip_w[idx] += w;
            }
        }
    }
    let mut candidates: Vec<(usize, f64)> = (0..priors.len())
        .filter(|&i| priors[i].is_some())
        .filter(|&i| incident[i] >= min_incident && total_w[i] > 0.0)
        .map(|i| (i, flip_w[i] / total_w[i]))
        .filter(|&(_, frac)| frac >= min_flip_fraction)
        .collect();
    candidates.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let drop_set: HashSet<usize> = candidates
        .into_iter()
        .take(max_drops)
        .map(|(i, _)| i)
        .collect();
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && !drop_set.is_empty() {
        let mut detail: Vec<_> = drop_set
            .iter()
            .map(|&i| (i, flip_w[i] / total_w[i], incident[i]))
            .collect();
        detail.sort_by(|a, b| b.1.total_cmp(&a.1));
        eprintln!(
            "global-sfm debug: edge-disagreement dropped priors {detail:?} \
             (min_frac={min_flip_fraction}, max_drops={max_drops})"
        );
    }
    let mut kept = 0usize;
    let filtered: Vec<Option<Pose>> = priors
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if drop_set.contains(&i) {
                None
            } else if p.is_some() {
                kept += 1;
                p.clone()
            } else {
                None
            }
        })
        .collect();
    (filtered, kept)
}

/// Drop incremental priors whose local track support is too thin or whose
/// mean reprojection exceeds `max_mean_reprojection_px`. Returns filtered
/// priors and how many were kept.
pub fn filter_pose_priors_by_track_quality(
    camera: &Camera,
    priors: &[Option<Pose>],
    tracks: &[SfmTrack],
    min_observations: usize,
    max_mean_reprojection_px: f64,
) -> (Vec<Option<Pose>>, usize) {
    let mut sum = vec![0.0f64; priors.len()];
    let mut count = vec![0usize; priors.len()];
    for track in tracks {
        for &(image, _, pixel) in &track.observations {
            if image >= priors.len() {
                continue;
            }
            let Some(pose) = priors[image].as_ref() else {
                continue;
            };
            let Some(err) = reprojection_error_px(camera, pose, &track.position, &pixel) else {
                continue;
            };
            sum[image] += err;
            count[image] += 1;
        }
    }
    let mut kept = 0usize;
    let filtered = priors
        .iter()
        .enumerate()
        .map(|(i, prior)| {
            let Some(p) = prior else {
                return None;
            };
            if count[i] < min_observations {
                return None;
            }
            let mean = sum[i] / count[i] as f64;
            if mean > max_mean_reprojection_px {
                return None;
            }
            kept += 1;
            Some(p.clone())
        })
        .collect();
    (filtered, kept)
}

/// Tunables for [`reconstruct_global_sfm`] beyond the shared incremental-
/// mapper config (which supplies triangulation gates and the BA recipe).
#[derive(Debug, Clone, PartialEq)]
pub struct GlobalReconstructionTuning {
    /// Image whose frame becomes the reconstruction gauge.
    pub seed: usize,
    /// Rotation-consensus sweeps (8-16 is ample on connected graphs).
    pub sweeps: usize,
    /// Position least-squares conjugate-gradient budget.
    pub cg_iterations: usize,
    /// Pairs below this verified-match count are skipped when estimating
    /// relative poses.
    pub min_pair_matches: usize,
    /// Minimum essential inliers per pair for its edge to enter the graph.
    pub min_edge_inliers: usize,
    /// IRLS trim threshold for rotation averaging (degrees). Edges whose
    /// implied relative rotation disagrees with the emerging global solution
    /// by more than this are zeroed; view-graph triangles whose median loop
    /// error exceeds twice this are dropped outright.
    pub max_edge_rotation_error_deg: f64,
    /// GLOMAP-style baseline gate: an edge whose median two-view
    /// triangulation angle over its inliers falls below this (degrees)
    /// carries no usable positional information — its translation direction
    /// is noise — and is dropped from the view graph entirely. Pure-rotation
    /// / tiny-baseline pairs are exactly the ones whose bearings bend the
    /// position solve on repetitive scenes. Set to 0 to disable.
    pub min_edge_parallax_deg: f64,
    /// When true, relative-pose recovery uses
    /// [`visloc_vision::two_view::CheiralityOptions::hardened`]: min
    /// triangulation angle, ambiguity rejection, and a majority
    /// positive-depth fraction. Chirality-ambiguous essentials on repetitive
    /// façades are dropped instead of entering the view graph as bent
    /// bearings. Default false = byte-identical legacy decomposition.
    pub chirality_harden_edges: bool,
    /// Number of rotation-averaging seeds to try inside the largest
    /// connected component (highest-degree nodes preferred). The solve with
    /// the most kept edges / lowest median residual wins. `1` = legacy
    /// single-seed behaviour.
    pub rotation_seed_trials: usize,
    /// After rotation averaging, re-score / re-estimate each edge's
    /// translation direction under the consensus relative rotation using
    /// retained inlier samples. Flips chirality-ambiguous pairwise
    /// translations that survive per-pair RANSAC. Default false.
    pub refine_translations_with_global_rotations: bool,
    /// Solve camera centres together with one unknown scale per retained
    /// bearing edge instead of imposing a unit displacement on every edge.
    /// This is a default-off correction for connected global initialization;
    /// rank-deficient bearing graphs fall back to the legacy position solve.
    pub independent_edge_scales: bool,
    /// Keep chirality-ambiguous essentials as multi-hypothesis edges
    /// (primary + runner-up R/t). Rotation averaging and position bearings
    /// pick the hypothesis that agrees with the emerging global solution.
    /// When combined with [`Self::chirality_harden_edges`], uses angle /
    /// depth gates without the ambiguity *rejection* ratio. Default false.
    pub multi_hypothesis_edges: bool,
    /// Scale each edge weight by `(0.1 + chirality_margin)` so near-ambiguous
    /// façade essentials contribute less to rotation IRLS and positioning.
    /// Default false = weight = raw inlier count.
    pub weight_edges_by_chirality_margin: bool,
    /// Hybrid mapper: pin incremental orientations during rotation averaging
    /// but keep globally averaged centres for prior cameras instead of
    /// overwriting with incremental centres. Default false = full pose priors.
    pub hybrid_rotation_priors_only: bool,
    /// GLOMAP-style joint camera+point positioning after rotation averaging
    /// (ray cross-product IRLS on feature tracks) instead of pairwise
    /// translation averaging alone. Default false.
    pub joint_global_positioning: bool,
    /// When true, keep only view-graph edges whose stored
    /// [`PairwiseMatches::two_view_config`] is `Calibrated` (or `Multiple`),
    /// **or** Uncalibrated/other configs that still carry a strong
    /// [`PairwiseMatches::essential_matches`] subset (≥ `min_pair_matches`).
    /// Drops F-only / planar / panoramic admissions whose essential
    /// translation is often ill-conditioned. Requires the full verifier to
    /// have populated `two_view_config`. Default false.
    pub calibrated_view_edges_only: bool,
    /// When true, build relative-pose edges from
    /// [`PairwiseMatches::essential_matches`] when present (E RANSAC inliers)
    /// instead of the verifier's winning F/H inlier set. Tracks and
    /// incremental growth still use [`PairwiseMatches::matches`]. Default false.
    pub prefer_essential_edge_matches: bool,
    /// Like [`Self::prefer_essential_edge_matches`], but only for pairs where
    /// *at least one* endpoint lacks a pose prior (free camera). Prior–prior
    /// edges keep the winning F/H set (and may still be repaired). Default false.
    pub prefer_essential_edge_matches_free_endpoints: bool,
    /// Prefer E inliers only on edges incident to these image indices (hub
    /// surgery). Empty = unused. Checked after the global prefer flag and
    /// before the free-endpoint flag.
    pub prefer_essential_edge_image_indices: Vec<usize>,
    /// When true with a non-empty stem index list, require *both* endpoints in
    /// the set (clique edges only) instead of any incident edge.
    pub prefer_essential_edge_stem_clique: bool,
    /// Prefer E inliers only on these explicit `(image_i, image_j)` pairs
    /// (order-insensitive). Empty = unused. Checked before the stem list.
    pub prefer_essential_edge_pairs: Vec<(usize, usize)>,
    /// When a stem-list (or explicit pair list) selects a pair for E preference
    /// but `essential_matches` is missing/too short, drop the edge entirely
    /// instead of falling back to F/H. Default false.
    pub require_essential_for_selected_edges: bool,
    /// Drop edges incident to these images unless strong E inliers exist
    /// (no F/H fallback). Independent of the prefer-stem list so a bent free
    /// camera can be isolated without requiring E on the load-bearing hub.
    /// Empty = unused.
    pub require_essential_edge_image_indices: Vec<usize>,
    /// Minimum E inlier count for [`Self::require_essential_edge_image_indices`]
    /// (and selected-edge require). `0` = [`Self::min_pair_matches`]. Does not
    /// affect which E subsets are *stored* on pairs.
    pub require_essential_min_e_inliers: usize,
    /// Multiply view-graph edge weight when the edge was built from essential
    /// inliers (`prefer_essential_*`). `1.0` = no boost. Default 1.0.
    pub essential_edge_weight_boost: f64,
    /// Hybrid: overwrite relative rotation + translation direction on edges
    /// whose *both* endpoints have pose priors, using the prior metric frame.
    /// Boosts those edge weights so free cameras hang off a prior-consistent
    /// skeleton. Default false.
    pub repair_edges_from_pose_priors: bool,
    /// Hybrid: set the position-averaging scale row from a prior–prior edge's
    /// true metric length (instead of a unit seed edge). Default false.
    pub metric_scale_from_pose_priors: bool,
    /// Hybrid: drop priors that disagree with a free-centre (rotation-pinned)
    /// probe solve after Sim(3) onto the prior set. Default false.
    pub drop_inconsistent_pose_priors: bool,
    /// Unused for free-centre residual (kept for CLI/API stability).
    pub inconsistent_prior_min_incident: usize,
    /// Free-centre residual must be at least this × the median residual.
    pub inconsistent_prior_min_flip_fraction: f64,
    /// Cap on how many inconsistent priors to drop (worst first).
    pub inconsistent_prior_max_drops: usize,
    /// After hybrid BA, re-estimate free (non-prior) camera poses by PnP
    /// against tracks that are also observed by at least one prior-pinned
    /// camera — so free hubs hang off prior-anchored structure, not their
    /// own bent bearings. Default false.
    pub repnp_free_cameras_from_priors: bool,
    /// Minimum prior-anchored 2D–3D correspondences for
    /// [`Self::repnp_free_cameras_from_priors`]. `0` = mapper `min_pnp_inliers`.
    pub repnp_free_min_corrs: usize,
    /// Before the hybrid global solve, triangulate prior-only structure and
    /// PnP each free camera into it; successful poses are promoted to hard
    /// pose priors so averaging cannot bend them. Default false.
    pub repnp_seed_free_as_priors: bool,
    /// Image indices that must not be promoted by
    /// [`Self::repnp_seed_free_as_priors`] (e.g. explicitly dropped hubs).
    pub repnp_seed_exclude_image_indices: Vec<usize>,
    /// After the first global solve, rewrite edges incident to free cameras
    /// from pass-1 poses (priors stay authoritative) and re-run averaging.
    /// Targets chirality-flipped bearings on prior↔free bridges. Default false.
    pub repair_free_edges_from_solved_poses: bool,
    /// When repairing free-incident edges, only touch edges whose current
    /// bearing is antipodal (dot < 0) to the pass-1 pose field. Default false
    /// (= rewrite all free-incident edges when repair is enabled).
    pub repair_free_edges_only_flipped: bool,
    /// Limit free-incident repair to edges incident to these image indices.
    /// Empty = all free-incident edges.
    pub repair_free_edges_image_indices: Vec<usize>,
    /// After pass-1 global, drop (not rewrite) free-incident edges whose
    /// bearing is antipodal to the pass-1 pose field, then re-solve. Default
    /// false.
    pub drop_free_edges_antipodal_to_solved: bool,
    /// Before rotation averaging, flip prior↔free edges to the chirality
    /// alternate when multi-view prior rays to the same free camera agree
    /// better. Requires `--multi-hypothesis-edges`. Default false.
    pub prior_guided_free_chirality: bool,
    /// When building edges, triangulate each free camera from prior↔free
    /// essentials in the incremental metric frame and flip chirality on
    /// prior↔free pairs whose alternate bearing aligns better with that
    /// anchor. Requires `--multi-hypothesis-edges`. Default false.
    pub metric_prior_chirality_edges: bool,
    /// Min prior↔free rays to triangulate a free centre for
    /// [`Self::metric_prior_chirality_edges`].
    pub metric_prior_chirality_min_rays: usize,
    /// Pick primary vs alternate edge chirality by GT pose agreement (oracle
    /// ceiling experiment; requires per-image GT poses). Default false.
    pub gt_chirality_oracle: bool,
}

impl Default for GlobalReconstructionTuning {
    fn default() -> Self {
        Self {
            seed: 0,
            sweeps: 12,
            cg_iterations: 500,
            min_pair_matches: 10,
            min_edge_inliers: 15,
            max_edge_rotation_error_deg: 10.0,
            min_edge_parallax_deg: 2.0,
            chirality_harden_edges: false,
            rotation_seed_trials: 1,
            refine_translations_with_global_rotations: false,
            independent_edge_scales: false,
            multi_hypothesis_edges: false,
            weight_edges_by_chirality_margin: false,
            hybrid_rotation_priors_only: false,
            joint_global_positioning: false,
            calibrated_view_edges_only: false,
            prefer_essential_edge_matches: false,
            prefer_essential_edge_matches_free_endpoints: false,
            prefer_essential_edge_image_indices: Vec::new(),
            prefer_essential_edge_stem_clique: false,
            prefer_essential_edge_pairs: Vec::new(),
            require_essential_for_selected_edges: false,
            require_essential_edge_image_indices: Vec::new(),
            require_essential_min_e_inliers: 0,
            essential_edge_weight_boost: 1.0,
            repair_edges_from_pose_priors: false,
            metric_scale_from_pose_priors: false,
            drop_inconsistent_pose_priors: false,
            inconsistent_prior_min_incident: 3,
            inconsistent_prior_min_flip_fraction: 2.0,
            inconsistent_prior_max_drops: 1,
            repnp_free_cameras_from_priors: false,
            repnp_free_min_corrs: 0,
            repnp_seed_free_as_priors: false,
            repnp_seed_exclude_image_indices: Vec::new(),
            repair_free_edges_from_solved_poses: false,
            repair_free_edges_only_flipped: false,
            repair_free_edges_image_indices: Vec::new(),
            drop_free_edges_antipodal_to_solved: false,
            prior_guided_free_chirality: false,
            metric_prior_chirality_edges: false,
            metric_prior_chirality_min_rays: 3,
            gt_chirality_oracle: false,
        }
    }
}

/// Errors surfaced by [`reconstruct_global_sfm`].
#[derive(Debug, Clone, PartialEq)]
pub enum GlobalReconstructionError {
    /// The view-graph stage could not solve any component from the given
    /// edges (no usable pairs, or every pair failed estimation).
    NoUsableEdges,
    /// The final bundle adjustment failed; the pre-BA geometry is lost so
    /// only the error surfaces.
    Ba(String),
}

impl std::fmt::Display for GlobalReconstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GlobalReconstructionError::NoUsableEdges => {
                write!(f, "global SfM found no usable pairwise edges")
            }
            GlobalReconstructionError::Ba(error) => {
                write!(f, "global SfM bundle adjustment failed: {error}")
            }
        }
    }
}
impl std::error::Error for GlobalReconstructionError {}

/// Re-estimate poses of free (non-prior) cameras via PnP on tracks that are
/// also seen by at least one prior-pinned camera. Returns how many poses were
/// replaced.
#[allow(clippy::too_many_arguments)]
fn repnp_free_cameras_from_prior_structure(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    poses: &mut [Option<Pose>],
    pose_priors: &[Option<Pose>],
    mapper: &crate::incremental_sfm::IncrementalSfmConfig,
    min_corrs: usize,
) -> usize {
    let prior_count = pose_priors.iter().filter(|p| p.is_some()).count();
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: re-PnP scan ({} priors, {} registered free candidates)",
            prior_count,
            poses
                .iter()
                .enumerate()
                .filter(
                    |(i, p)| p.is_some() && pose_priors.get(*i).and_then(|q| q.as_ref()).is_none()
                )
                .count()
        );
    }
    let mut replaced = 0usize;
    for image in 0..features.len() {
        if poses[image].is_none() {
            continue;
        }
        if pose_priors.get(image).and_then(|p| p.as_ref()).is_some() {
            continue; // keep pinned prior cameras
        }
        let mut corrs: Vec<Correspondence2D3D> = Vec::new();
        for (track_id, track) in tracks.iter().enumerate() {
            let Some(position) = track_point[track_id] else {
                continue;
            };
            let Some((_, kp)) = track.iter().find(|&&(img, _)| img == image) else {
                continue;
            };
            let anchored = track
                .iter()
                .any(|&(img, _)| pose_priors.get(img).and_then(|p| p.as_ref()).is_some());
            if !anchored {
                continue;
            }
            let Some(pixel) = features[image].keypoints.get(*kp).copied() else {
                continue;
            };
            corrs.push(Correspondence2D3D {
                point2d: pixel,
                point3d: position,
                confidence: None,
            });
        }
        if corrs.len() < min_corrs {
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: re-PnP skip image {image}: only {} prior-anchored corrs (need {min_corrs})",
                    corrs.len(),
                );
            }
            continue;
        }
        let report = match mapper.pnp_solver {
            PnpSolver::P3p => PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: mapper.pnp_max_iterations.max(128),
                confidence: Some(0.999),
                reprojection_threshold: mapper.max_reprojection_error_px,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
            }
            .estimate(&corrs, camera),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: mapper.max_reprojection_error_px,
                confidence: Some(0.999),
                ..PnPRansac::default()
            }
            .estimate(&corrs, camera),
        };
        let Some(report) =
            report.filter(|r| r.inliers.len() >= min_corrs.min(mapper.min_pnp_inliers).max(4))
        else {
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: re-PnP fail image {image}: {} corrs, PnP rejected",
                    corrs.len()
                );
            }
            continue;
        };
        // Free cameras often sit in a self-consistent wrong basin with low
        // local reproj on prior-anchored tracks; still replace when PnP finds
        // a well-supported pose.
        let old_pose = poses[image].as_ref().unwrap();
        let mean = |pose: &Pose| -> f64 {
            let mut s = 0.0;
            let mut n = 0usize;
            for c in &corrs {
                if let Some(e) = reprojection_error_px(camera, pose, &c.point3d, &c.point2d) {
                    s += e;
                    n += 1;
                }
            }
            if n == 0 {
                f64::INFINITY
            } else {
                s / n as f64
            }
        };
        let old_mean = mean(old_pose);
        let new_mean = mean(&report.pose);
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            eprintln!(
                "global-sfm debug: re-PnP free image {image}: {} corrs -> {} inliers, \
                 mean reproj {old_mean:.3} -> {new_mean:.3} px (accept)",
                corrs.len(),
                report.inliers.len()
            );
        }
        poses[image] = Some(report.pose);
        replaced += 1;
    }
    replaced
}

/// Triangulate tracks using only already-pinned prior poses, then PnP each
/// free camera into that structure and write successful poses into
/// `pose_priors` (promoting them to hard pins for the subsequent global
/// solve). Returns how many free cameras were seeded.
fn pnp_seed_free_cameras_into_priors(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    pose_priors: &mut [Option<Pose>],
    mapper: &crate::incremental_sfm::IncrementalSfmConfig,
    min_corrs: usize,
    exclude: &HashSet<usize>,
) -> usize {
    let prior_count = pose_priors.iter().filter(|p| p.is_some()).count();
    if prior_count < 2 {
        return 0;
    }
    let built = build_track_output(features, pairwise, mapper, Some(camera));
    let poses_for_tri: Vec<Option<Pose>> = pose_priors.to_vec();
    let mut track_point = vec![None; built.tracks.len()];
    triangulate_pending(
        camera,
        features,
        &built.tracks,
        &poses_for_tri,
        mapper,
        &mut track_point,
    );
    let tri_count = track_point.iter().filter(|p| p.is_some()).count();
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: pre-global PnP seed: {prior_count} priors triangulated \
             {tri_count} / {} tracks",
            built.tracks.len()
        );
    }
    let mut seeded = 0usize;
    for image in 0..features.len() {
        if pose_priors.get(image).and_then(|p| p.as_ref()).is_some() {
            continue;
        }
        if exclude.contains(&image) {
            continue;
        }
        let mut corrs: Vec<Correspondence2D3D> = Vec::new();
        for (track_id, track) in built.tracks.iter().enumerate() {
            let Some(position) = track_point[track_id] else {
                continue;
            };
            let Some((_, kp)) = track.iter().find(|&&(img, _)| img == image) else {
                continue;
            };
            let anchored = track
                .iter()
                .any(|&(img, _)| pose_priors.get(img).and_then(|p| p.as_ref()).is_some());
            if !anchored {
                continue;
            }
            let Some(pixel) = features[image].keypoints.get(*kp).copied() else {
                continue;
            };
            corrs.push(Correspondence2D3D {
                point2d: pixel,
                point3d: position,
                confidence: None,
            });
        }
        if corrs.len() < min_corrs {
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: pre-global PnP seed skip image {image}: \
                     only {} prior-anchored corrs (need {min_corrs})",
                    corrs.len()
                );
            }
            continue;
        }
        let report = match mapper.pnp_solver {
            PnpSolver::P3p => PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: mapper.pnp_max_iterations.max(128),
                confidence: Some(0.999),
                reprojection_threshold: mapper.max_reprojection_error_px,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
            }
            .estimate(&corrs, camera),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: mapper.max_reprojection_error_px,
                confidence: Some(0.999),
                ..PnPRansac::default()
            }
            .estimate(&corrs, camera),
        };
        let Some(report) =
            report.filter(|r| r.inliers.len() >= min_corrs.min(mapper.min_pnp_inliers).max(4))
        else {
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: pre-global PnP seed fail image {image}: \
                     {} corrs, PnP rejected",
                    corrs.len()
                );
            }
            continue;
        };
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            eprintln!(
                "global-sfm debug: pre-global PnP seed image {image}: \
                 {} corrs -> {} inliers (promote to prior)",
                corrs.len(),
                report.inliers.len()
            );
        }
        pose_priors[image] = Some(report.pose);
        seeded += 1;
    }
    seeded
}

/// End-to-end GLOMAP-style reconstruction: estimate per-pair relative poses
/// from verified matches, average them globally, triangulate the feature
/// tracks against the averaged cameras, and finish with one bundle adjustment
/// over everything registered.
///
/// This is the global analogue of [`crate::incremental_sfm::incremental_sfm`]'s grow-from-a-
/// seed loop: instead of chaining registrations image by image, every camera
/// lands in one consistent frame at once and BA only has to polish local
/// geometry. Images outside the seed's view-graph component stay unposed.
///
/// Output: per-image poses (`None` outside the solved component), assembled
/// BA-refined tracks, and mean track reprojection in pixels.
#[allow(clippy::type_complexity)]
pub fn reconstruct_global_sfm(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tuning: &GlobalReconstructionTuning,
    mapper: &crate::incremental_sfm::IncrementalSfmConfig,
) -> Result<(Vec<Option<Pose>>, Vec<SfmTrack>, f64), GlobalReconstructionError> {
    reconstruct_global_sfm_with_priors(camera, features, pairwise, tuning, mapper, None, None)
}

/// Like [`reconstruct_global_sfm`], but pins cameras that already have absolute
/// poses (typically from a prior incremental solve) and places the remaining
/// images in that gauge.
#[allow(clippy::type_complexity)]
pub fn reconstruct_global_sfm_with_priors(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tuning: &GlobalReconstructionTuning,
    mapper: &crate::incremental_sfm::IncrementalSfmConfig,
    pose_priors: Option<&[Option<Pose>]>,
    gt_poses: Option<&[Option<Pose>]>,
) -> Result<(Vec<Option<Pose>>, Vec<SfmTrack>, f64), GlobalReconstructionError> {
    // ---- Relative poses → global edges ------------------------------------
    let estimator = RelativePoseEstimator {
        cheirality: match (tuning.chirality_harden_edges, tuning.multi_hypothesis_edges) {
            (true, true) => CheiralityOptions::hardened_keep_ambiguous(),
            (true, false) => CheiralityOptions::hardened(),
            (false, _) => CheiralityOptions::default(),
        },
        ..RelativePoseEstimator::default()
    };
    let mut edges = Vec::new();
    let mut chirality_rejected = 0usize;
    let mut multi_hyp_edges = 0usize;
    let mut pair_fail_estimate = vec![0usize; features.len()];
    let mut pair_fail_inliers = vec![0usize; features.len()];
    let mut pair_fail_parallax = vec![0usize; features.len()];
    let mut pair_kept = vec![0usize; features.len()];
    let config_allowed = |pair: &PairwiseMatches| -> bool {
        if !tuning.calibrated_view_edges_only {
            return true;
        }
        match pair.two_view_config {
            None => true,
            Some(ConfigurationType::Calibrated | ConfigurationType::Multiple) => true,
            // F-won Uncalibrated pairs can still carry a strong essential
            // subset (courtyard stem-E / rematch bridges). Keep those so
            // calibrated-only filtering does not discard true E bearings.
            Some(_) => pair
                .essential_matches
                .as_ref()
                .is_some_and(|e| e.len() >= tuning.min_pair_matches),
        }
    };
    fn edge_matches(
        pair: &PairwiseMatches,
        prefer_essential: bool,
        min_pair_matches: usize,
    ) -> &[(usize, usize)] {
        if prefer_essential {
            if let Some(ess) = pair.essential_matches.as_ref() {
                if ess.len() >= min_pair_matches {
                    return ess.as_slice();
                }
            }
        }
        pair.matches.as_slice()
    }
    /// Prefer E inliers for this pair given the free-endpoint / global flags.
    fn prefer_essential_for_pair(
        tuning: &GlobalReconstructionTuning,
        pose_priors: Option<&[Option<Pose>]>,
        image_i: usize,
        image_j: usize,
    ) -> bool {
        if tuning.prefer_essential_edge_matches {
            return true;
        }
        let mut selected = false;
        if !tuning.prefer_essential_edge_pairs.is_empty() {
            let a = image_i.min(image_j);
            let b = image_i.max(image_j);
            selected |= tuning
                .prefer_essential_edge_pairs
                .iter()
                .any(|&(i, j)| i.min(j) == a && i.max(j) == b);
        }
        if !tuning.prefer_essential_edge_image_indices.is_empty() {
            let hit_i = tuning
                .prefer_essential_edge_image_indices
                .contains(&image_i);
            let hit_j = tuning
                .prefer_essential_edge_image_indices
                .contains(&image_j);
            selected |= if tuning.prefer_essential_edge_stem_clique {
                hit_i && hit_j
            } else {
                hit_i || hit_j
            };
        }
        // Require-stems also prefer E when present (so isolation uses E bearings).
        if !tuning.require_essential_edge_image_indices.is_empty() {
            selected |= tuning
                .require_essential_edge_image_indices
                .iter()
                .any(|&idx| idx == image_i || idx == image_j);
        }
        if selected {
            return true;
        }
        if !tuning.prefer_essential_edge_matches_free_endpoints {
            return false;
        }
        match pose_priors {
            None => true,
            Some(priors) => {
                let free_i = priors.get(image_i).is_none_or(Option::is_none);
                let free_j = priors.get(image_j).is_none_or(Option::is_none);
                free_i || free_j
            }
        }
    }
    /// True when this pair is in the stem/pair selection set (regardless of
    /// whether E inliers are actually available).
    fn essential_selection_covers_pair(
        tuning: &GlobalReconstructionTuning,
        image_i: usize,
        image_j: usize,
    ) -> bool {
        // Same coverage as prefer_essential_for_pair without the free-endpoint
        // fallback (require-E only applies to explicit stem/pair selections).
        if !tuning.prefer_essential_edge_pairs.is_empty() {
            let a = image_i.min(image_j);
            let b = image_i.max(image_j);
            if tuning
                .prefer_essential_edge_pairs
                .iter()
                .any(|&(i, j)| i.min(j) == a && i.max(j) == b)
            {
                return true;
            }
        }
        if !tuning.prefer_essential_edge_image_indices.is_empty() {
            let hit_i = tuning
                .prefer_essential_edge_image_indices
                .contains(&image_i);
            let hit_j = tuning
                .prefer_essential_edge_image_indices
                .contains(&image_j);
            return if tuning.prefer_essential_edge_stem_clique {
                hit_i && hit_j
            } else {
                hit_i || hit_j
            };
        }
        false
    }
    /// True when this pair must have strong E inliers or be dropped.
    fn requires_essential_for_pair(
        tuning: &GlobalReconstructionTuning,
        image_i: usize,
        image_j: usize,
    ) -> bool {
        if tuning.require_essential_for_selected_edges
            && essential_selection_covers_pair(tuning, image_i, image_j)
        {
            return true;
        }
        if !tuning.require_essential_edge_image_indices.is_empty() {
            return tuning
                .require_essential_edge_image_indices
                .iter()
                .any(|&idx| idx == image_i || idx == image_j);
        }
        false
    }
    let mut essential_edge_uses = 0usize;
    let mut essential_required_drops = 0usize;
    let metric_chirality_flips = std::cell::Cell::new(0usize);
    let gt_oracle_flips = std::cell::Cell::new(0usize);
    let free_centres = if tuning.metric_prior_chirality_edges {
        pose_priors
            .map(|priors| {
                estimate_free_centres_from_prior_rays(
                    pairwise,
                    features,
                    camera,
                    priors,
                    tuning.metric_prior_chirality_min_rays,
                    tuning.min_pair_matches,
                )
            })
            .unwrap_or_default()
    } else {
        HashMap::new()
    };
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && tuning.metric_prior_chirality_edges {
        eprintln!(
            "global-sfm debug: metric-prior chirality anchors for {} free camera(s)",
            free_centres.len()
        );
    }
    for pair in pairwise {
        let use_essential = tuning.independent_edge_scales
            || prefer_essential_for_pair(tuning, pose_priors, pair.image_i, pair.image_j);
        if requires_essential_for_pair(tuning, pair.image_i, pair.image_j) {
            let min_e = if tuning.require_essential_min_e_inliers > 0 {
                tuning.require_essential_min_e_inliers
            } else {
                tuning.min_pair_matches
            };
            let has_ess = pair
                .essential_matches
                .as_ref()
                .is_some_and(|e| e.len() >= min_e);
            if !has_ess {
                essential_required_drops += 1;
                continue;
            }
        }
        let matches = edge_matches(pair, use_essential, tuning.min_pair_matches);
        if matches.len() < tuning.min_pair_matches {
            continue;
        }
        if !config_allowed(pair) {
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: dropped non-calibrated edge {}-{} ({:?})",
                    pair.image_i, pair.image_j, pair.two_view_config
                );
            }
            continue;
        }
        let used_essential = use_essential
            && pair
                .essential_matches
                .as_ref()
                .is_some_and(|e| e.len() >= tuning.min_pair_matches);
        if used_essential {
            essential_edge_uses += 1;
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: essential-edge {}-{} ({} E inliers)",
                    pair.image_i,
                    pair.image_j,
                    pair.essential_matches
                        .as_ref()
                        .map(|e| e.len())
                        .unwrap_or(0)
                );
            }
        }
        let correspondences: Vec<TwoViewCorrespondence> = matches
            .iter()
            .filter_map(|&(ki, kj)| {
                Some(TwoViewCorrespondence::new(
                    *features[pair.image_i].keypoints.get(ki)?,
                    *features[pair.image_j].keypoints.get(kj)?,
                ))
            })
            .collect();
        let relative = estimator.estimate(&correspondences, camera);
        let Some(relative) = relative else {
            if tuning.chirality_harden_edges {
                chirality_rejected += 1;
            }
            pair_fail_estimate[pair.image_i] += 1;
            pair_fail_estimate[pair.image_j] += 1;
            continue;
        };
        if relative.inliers.len() < tuning.min_edge_inliers {
            pair_fail_inliers[pair.image_i] += 1;
            pair_fail_inliers[pair.image_j] += 1;
            continue;
        };
        // Bearing of camera j's centre expressed in camera i's frame:
        // C_j^(i) = −R_ijᵀ t_ij.
        let mut r_ij = relative.previous_to_current.rotation;
        let t_ij = relative.previous_to_current.translation;
        let Some(mut direction_ij) = (-r_ij.inverse().transform_vector(&t_ij)).try_normalize(1e-12)
        else {
            continue;
        };
        let (mut rotation_alt, mut direction_alt) = if tuning.multi_hypothesis_edges {
            match relative.alternate {
                Some((r_alt, t_alt_unit)) => {
                    let scale = relative.translation_scale;
                    let t_alt = t_alt_unit * scale;
                    match (-r_alt.inverse().transform_vector(&t_alt)).try_normalize(1e-12) {
                        Some(d_alt) => {
                            multi_hyp_edges += 1;
                            (Some(r_alt), Some(d_alt))
                        }
                        None => (None, None),
                    }
                }
                None => (None, None),
            }
        } else {
            (None, None)
        };
        if tuning.metric_prior_chirality_edges {
            let is_prior = |i: usize| {
                pose_priors
                    .and_then(|p| p.get(i))
                    .and_then(|p| p.as_ref())
                    .is_some()
            };
            let pi = is_prior(pair.image_i);
            let pj = is_prior(pair.image_j);
            if pi != pj {
                let (prior_idx, free_idx) = if pi {
                    (pair.image_i, pair.image_j)
                } else {
                    (pair.image_j, pair.image_i)
                };
                if let (Some(priors), Some(free_centre)) =
                    (pose_priors, free_centres.get(&free_idx))
                {
                    if let Some(prior_pose) = priors.get(prior_idx).and_then(|p| p.as_ref()) {
                        if let Some(expected) =
                            expected_bearing_prior_to_world_point(prior_pose, free_centre)
                        {
                            let flip = (|| {
                                let bearing_primary = if prior_idx == pair.image_i {
                                    direction_ij
                                } else {
                                    (-r_ij.transform_vector(&direction_ij)).try_normalize(1e-12)?
                                };
                                let primary_dot = bearing_primary.dot(&expected);
                                let r_alt = rotation_alt.as_ref()?;
                                let d_alt = direction_alt.as_ref()?;
                                let bearing_alt = if prior_idx == pair.image_i {
                                    *d_alt
                                } else {
                                    (-r_alt.transform_vector(d_alt)).try_normalize(1e-12)?
                                };
                                Some(bearing_alt.dot(&expected) > primary_dot + 1e-3)
                            })();
                            if flip == Some(true) {
                                std::mem::swap(&mut r_ij, rotation_alt.as_mut().unwrap());
                                std::mem::swap(&mut direction_ij, direction_alt.as_mut().unwrap());
                                metric_chirality_flips.set(metric_chirality_flips.get() + 1);
                            }
                        }
                    }
                }
            }
        }
        if tuning.gt_chirality_oracle {
            if let Some(gt) = gt_poses {
                if let (Some(gti), Some(gtj)) = (
                    gt.get(pair.image_i).and_then(|p| p.as_ref()),
                    gt.get(pair.image_j).and_then(|p| p.as_ref()),
                ) {
                    if let Some(gt_in_i) = gt_bearing_in_prior_frame(gti, gtj) {
                        let err_pri = bearing_alignment_error_deg(&direction_ij, &gt_in_i);
                        if let Some(d_alt) = direction_alt.as_ref() {
                            let err_alt = bearing_alignment_error_deg(d_alt, &gt_in_i);
                            if err_alt + 1e-3 < err_pri {
                                if let Some(r_alt) = rotation_alt.as_mut() {
                                    std::mem::swap(&mut r_ij, r_alt);
                                    std::mem::swap(
                                        &mut direction_ij,
                                        direction_alt.as_mut().unwrap(),
                                    );
                                    gt_oracle_flips.set(gt_oracle_flips.get() + 1);
                                }
                            }
                        }
                    }
                }
            }
        }
        // GLOMAP-style baseline gate: triangulate a sample of the inlier
        // correspondences under this pair's pose and measure the median ray
        // intersection angle. Tiny-baseline pairs produce well-fit essential
        // matrices whose translation direction is pure noise — dropping them
        // protects both averaging stages from systematically bent bearings.
        if tuning.min_edge_parallax_deg > 0.0 {
            let pose_i_to_j = relative.previous_to_current;
            let step: usize = (relative.inliers.len() / 100).max(1);
            let mut angles: Vec<f64> = Vec::new();
            for &idx in relative.inliers.iter().step_by(step) {
                let Some(corr) = correspondences.get(idx) else {
                    continue;
                };
                let Some(point) = triangulate_two_view_left_frame(
                    camera,
                    camera,
                    &pose_i_to_j,
                    &corr.previous_xy,
                    &corr.current_xy,
                ) else {
                    continue;
                };
                let cam_j_centre: Vector3<f64> = r_ij.inverse() * (-t_ij);
                let ray1 = point.coords;
                let ray2 = point.coords - cam_j_centre;
                let cos = ray1.normalize().dot(&ray2.normalize()).clamp(-1.0, 1.0);
                angles.push(cos.acos());
            }
            angles.sort_by(|a, b| a.total_cmp(b));
            if let Some(angle) = angles.get(angles.len() / 2).copied() {
                if angle < tuning.min_edge_parallax_deg.to_radians() {
                    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                        eprintln!(
                            "global-sfm debug: dropped low-parallax edge {}-{} ({:.2} deg, {} inliers)",
                            pair.image_i,
                            pair.image_j,
                            angle.to_degrees(),
                            relative.inliers.len()
                        );
                    }
                    pair_fail_parallax[pair.image_i] += 1;
                    pair_fail_parallax[pair.image_j] += 1;
                    continue;
                }
            }
        }
        // Retain a subsample of inliers for post-rotation translation refine.
        let step = (relative.inliers.len() / 32).max(1);
        let inlier_sample: Vec<(Point2<f64>, Point2<f64>)> = relative
            .inliers
            .iter()
            .step_by(step)
            .filter_map(|&idx| {
                let c = correspondences.get(idx)?;
                Some((c.previous_xy, c.current_xy))
            })
            .collect();
        let mut weight = if tuning.weight_edges_by_chirality_margin {
            relative.inliers.len() as f64 * (0.1 + relative.chirality_margin.clamp(0.0, 1.0))
        } else {
            relative.inliers.len() as f64
        };
        if used_essential && tuning.essential_edge_weight_boost != 1.0 {
            weight *= tuning.essential_edge_weight_boost.max(0.0);
        }
        edges.push(GlobalSfmEdge {
            image_i: pair.image_i,
            image_j: pair.image_j,
            rotation_ij: r_ij,
            direction_ij,
            weight,
            inlier_sample,
            rotation_alt,
            direction_alt,
        });
        pair_kept[pair.image_i] += 1;
        pair_kept[pair.image_j] += 1;
    }
    if tuning.prior_guided_free_chirality {
        if let Some(priors) = pose_priors {
            apply_prior_guided_free_chirality(&mut edges, priors);
        }
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && tuning.metric_prior_chirality_edges {
        eprintln!(
            "global-sfm debug: metric-prior chirality flipped {} prior↔free edge(s)",
            metric_chirality_flips.get()
        );
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && tuning.gt_chirality_oracle {
        eprintln!(
            "global-sfm debug: GT chirality oracle flipped {} edge(s)",
            gt_oracle_flips.get()
        );
    }
    // Orphan rescue: images that lost every edge at the strict inlier gate get
    // one more chance at half the threshold (floored at 8). Courtyard's
    // DSC_0308 typically has a single verified pair whose essential inliers
    // sit just under the default 15 — connecting it beats leaving a hole.
    {
        let mut degree = vec![0usize; features.len()];
        for e in &edges {
            degree[e.image_i] += 1;
            degree[e.image_j] += 1;
        }
        let orphans: Vec<usize> = degree
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();
        let rescue_inliers = (tuning.min_edge_inliers / 2).max(8);
        if !orphans.is_empty() && rescue_inliers < tuning.min_edge_inliers {
            let orphan_set: HashSet<usize> = orphans.iter().copied().collect();
            let existing: HashSet<(usize, usize)> = edges
                .iter()
                .map(|e| (e.image_i.min(e.image_j), e.image_i.max(e.image_j)))
                .collect();
            let mut rescued = 0usize;
            for pair in pairwise {
                if !orphan_set.contains(&pair.image_i) && !orphan_set.contains(&pair.image_j) {
                    continue;
                }
                if !config_allowed(pair) {
                    continue;
                }
                let key = (
                    pair.image_i.min(pair.image_j),
                    pair.image_i.max(pair.image_j),
                );
                if existing.contains(&key) {
                    continue;
                }
                if requires_essential_for_pair(tuning, pair.image_i, pair.image_j) {
                    let min_e = if tuning.require_essential_min_e_inliers > 0 {
                        tuning.require_essential_min_e_inliers
                    } else {
                        tuning.min_pair_matches
                    };
                    if pair
                        .essential_matches
                        .as_ref()
                        .is_none_or(|e| e.len() < min_e)
                    {
                        continue;
                    }
                }
                let matches = edge_matches(
                    pair,
                    tuning.independent_edge_scales
                        || prefer_essential_for_pair(
                            tuning,
                            pose_priors,
                            pair.image_i,
                            pair.image_j,
                        ),
                    tuning.min_pair_matches,
                );
                if matches.len() < tuning.min_pair_matches {
                    continue;
                }
                let correspondences: Vec<TwoViewCorrespondence> = matches
                    .iter()
                    .filter_map(|&(ki, kj)| {
                        Some(TwoViewCorrespondence::new(
                            *features[pair.image_i].keypoints.get(ki)?,
                            *features[pair.image_j].keypoints.get(kj)?,
                        ))
                    })
                    .collect();
                let Some(relative) = estimator.estimate(&correspondences, camera) else {
                    continue;
                };
                if relative.inliers.len() < rescue_inliers {
                    continue;
                }
                let r_ij = relative.previous_to_current.rotation;
                let t_ij = relative.previous_to_current.translation;
                let Some(direction_ij) =
                    (-r_ij.inverse().transform_vector(&t_ij)).try_normalize(1e-12)
                else {
                    continue;
                };
                let (rotation_alt, direction_alt) = if tuning.multi_hypothesis_edges {
                    match relative.alternate {
                        Some((r_alt, t_alt_unit)) => {
                            let t_alt = t_alt_unit * relative.translation_scale;
                            match (-r_alt.inverse().transform_vector(&t_alt)).try_normalize(1e-12) {
                                Some(d_alt) => (Some(r_alt), Some(d_alt)),
                                None => (None, None),
                            }
                        }
                        None => (None, None),
                    }
                } else {
                    (None, None)
                };
                let step = (relative.inliers.len() / 32).max(1);
                let inlier_sample: Vec<(Point2<f64>, Point2<f64>)> = relative
                    .inliers
                    .iter()
                    .step_by(step)
                    .filter_map(|&idx| {
                        let c = correspondences.get(idx)?;
                        Some((c.previous_xy, c.current_xy))
                    })
                    .collect();
                let weight = if tuning.weight_edges_by_chirality_margin {
                    relative.inliers.len() as f64
                        * (0.1 + relative.chirality_margin.clamp(0.0, 1.0))
                } else {
                    relative.inliers.len() as f64
                };
                edges.push(GlobalSfmEdge {
                    image_i: pair.image_i,
                    image_j: pair.image_j,
                    rotation_ij: r_ij,
                    direction_ij,
                    weight,
                    inlier_sample,
                    rotation_alt,
                    direction_alt,
                });
                pair_kept[pair.image_i] += 1;
                pair_kept[pair.image_j] += 1;
                rescued += 1;
            }
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: orphan-edge rescue (min_inliers={rescue_inliers}) \
                     added {rescued} edge(s) for {} orphan image(s)",
                    orphans.len()
                );
            }
        }
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        if tuning.chirality_harden_edges {
            eprintln!(
                "global-sfm debug: chirality-harden rejected {chirality_rejected} pairs; kept {} edges",
                edges.len()
            );
        }
        if tuning.multi_hypothesis_edges {
            eprintln!(
                "global-sfm debug: multi-hypothesis edges carrying alternate: {multi_hyp_edges} / {}",
                edges.len()
            );
        }
        if tuning.prefer_essential_edge_matches
            || tuning.prefer_essential_edge_matches_free_endpoints
            || !tuning.prefer_essential_edge_image_indices.is_empty()
            || !tuning.prefer_essential_edge_pairs.is_empty()
            || !tuning.require_essential_edge_image_indices.is_empty()
        {
            eprintln!(
                "global-sfm debug: prefer-essential-edge-matches used E inliers on {essential_edge_uses} pairs \
                 (all={} free-endpoints={} stem-indices={} clique={} pairs={} require-stems={} require-drop={essential_required_drops})",
                tuning.prefer_essential_edge_matches,
                tuning.prefer_essential_edge_matches_free_endpoints,
                tuning.prefer_essential_edge_image_indices.len(),
                tuning.prefer_essential_edge_stem_clique,
                tuning.prefer_essential_edge_pairs.len(),
                tuning.require_essential_edge_image_indices.len(),
            );
        }
        for i in 0..features.len() {
            if pair_kept[i] == 0
                && (pair_fail_estimate[i] + pair_fail_inliers[i] + pair_fail_parallax[i] > 0)
            {
                eprintln!(
                    "global-sfm debug: image {i} has 0 edges after relative-pose gates \
                     (fail estimate={} inliers={} parallax={})",
                    pair_fail_estimate[i], pair_fail_inliers[i], pair_fail_parallax[i]
                );
            }
        }
    }

    // ---- Triplet loop-closure sanitisation ---------------------------------
    // Wrong-chirality / wrong-pose essential estimates survive per-pair
    // RANSAC on repetitive real scenes but fail three-edge rotation loops:
    // R_wu·R_vw·R_uv must be ≈ identity around every triangle. Score each
    // edge by the MEDIAN loop error over its common-neighbour triangles and
    // drop edges above twice the IRLS trim threshold.
    {
        let mut neighbors: HashMap<usize, HashSet<usize>> = HashMap::new();
        for e in &edges {
            neighbors.entry(e.image_i).or_default().insert(e.image_j);
            neighbors.entry(e.image_j).or_default().insert(e.image_i);
        }
        let mut edge_lookup: HashMap<(usize, usize), usize> = HashMap::new();
        for (index, e) in edges.iter().enumerate() {
            edge_lookup.insert((e.image_i, e.image_j), index);
            edge_lookup.insert((e.image_j, e.image_i), index);
        }
        let step_hyps = |from: usize, index: usize| -> Vec<UnitQuaternion<f64>> {
            let e = &edges[index];
            let mut hyps = vec![e.step_primary(from)];
            if let Some(r_alt) = e.rotation_alt {
                hyps.push(if e.image_i == from {
                    r_alt
                } else {
                    r_alt.inverse()
                });
            }
            hyps
        };
        let mut loop_errors: Vec<Vec<f64>> = vec![Vec::new(); edges.len()];
        for e_index in 0..edges.len() {
            let (u, v) = (edges[e_index].image_i, edges[e_index].image_j);
            let Some(common) = neighbors.get(&u) else {
                continue;
            };
            for &w in common {
                if w == v || !neighbors.get(&v).is_some_and(|nv| nv.contains(&w)) {
                    continue;
                }
                let Some(&vw) = edge_lookup.get(&(v, w)) else {
                    continue;
                };
                let Some(&wu) = edge_lookup.get(&(w, u)) else {
                    continue;
                };
                // Min loop error over primary/alternate combinations so an
                // edge with a wrong primary but correct alternate is not
                // dropped by triplet sanitation.
                let mut best = f64::INFINITY;
                for s_uv in step_hyps(u, e_index) {
                    for s_vw in step_hyps(v, vw) {
                        for s_wu in step_hyps(w, wu) {
                            best = best.min((s_wu * s_vw * s_uv).angle());
                        }
                    }
                }
                if best.is_finite() {
                    loop_errors[e_index].push(best);
                }
            }
        }
        let mut dropped = 0usize;
        for (index, errs) in loop_errors.iter_mut().enumerate() {
            if errs.is_empty() {
                continue;
            }
            errs.sort_by(|a, b| a.total_cmp(b));
            let median = errs[errs.len() / 2];
            if median > tuning.max_edge_rotation_error_deg.to_radians() * 2.0 {
                dropped += 1;
                edges[index].weight = -1.0; // marker: dropped
            }
        }
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            eprintln!("global-sfm debug: triplet sanitation dropped {dropped} edges");
            let mut deg = vec![0usize; features.len()];
            for e in &edges {
                deg[e.image_i] += 1;
                deg[e.image_j] += 1;
            }
            for (i, &d) in deg.iter().enumerate() {
                if d == 0 {
                    eprintln!(
                        "global-sfm debug: image {i} degree 0 after triplet sanitation (pre-sanitation kept edges involving it: {})",
                        pair_kept[i]
                    );
                }
            }
        }
        edges.retain(|e| e.weight > 0.0);
    }

    // GT-free: probe with rotation-pinned / centre-free solve, drop priors
    // whose centres disagree with the free gauge after Sim(3), then continue
    // with the thinned prior set for the real hybrid solve.
    let pose_priors_filtered;
    let pose_priors: Option<&[Option<Pose>]> = if tuning.drop_inconsistent_pose_priors {
        match pose_priors {
            Some(priors) => {
                let before = priors.iter().filter(|p| p.is_some()).count();
                let mut probe_edges = edges.clone();
                let probe_seed = {
                    // Prefer any prior as probe seed (same component later).
                    priors
                        .iter()
                        .enumerate()
                        .find_map(|(i, p)| p.as_ref().map(|_| i))
                        .unwrap_or(0)
                };
                let filtered = match solve_global_sfm_with_options(
                    features.len(),
                    &mut probe_edges,
                    probe_seed,
                    tuning.sweeps,
                    tuning.cg_iterations,
                    tuning.max_edge_rotation_error_deg,
                    tuning.refine_translations_with_global_rotations,
                    Some(camera),
                    Some(priors),
                    false, // free centres
                    None,
                    false,
                    false,
                    tuning.independent_edge_scales,
                ) {
                    Some(probe) => {
                        let free_centers: Vec<Option<Point3<f64>>> = probe
                            .poses
                            .iter()
                            .map(|p| p.as_ref().map(|pose| pose.camera_center_world()))
                            .collect();
                        let (filtered, kept) = filter_pose_priors_by_free_centre_residual(
                            priors,
                            &free_centers,
                            tuning.inconsistent_prior_max_drops,
                            tuning.inconsistent_prior_min_flip_fraction,
                        );
                        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                            eprintln!(
                                "global-sfm debug: free-centre prior filter kept {kept} / {before}"
                            );
                        }
                        filtered
                    }
                    None => {
                        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                            eprintln!(
                                    "global-sfm debug: free-centre probe failed; keeping all {before} priors"
                                );
                        }
                        priors.to_vec()
                    }
                };
                pose_priors_filtered = filtered;
                Some(pose_priors_filtered.as_slice())
            }
            None => None,
        }
    } else {
        pose_priors
    };

    // Optional: promote free cameras that PnP cleanly into prior-only
    // structure to hard pose priors before rotation/position averaging.
    let seeded_priors_storage = if tuning.repnp_seed_free_as_priors {
        pose_priors.map(|priors| {
            let mut owned = priors.to_vec();
            let min_corrs = if tuning.repnp_free_min_corrs > 0 {
                tuning.repnp_free_min_corrs
            } else {
                mapper.min_pnp_inliers
            };
            let n = pnp_seed_free_cameras_into_priors(
                camera,
                features,
                pairwise,
                &mut owned,
                mapper,
                min_corrs,
                &tuning
                    .repnp_seed_exclude_image_indices
                    .iter()
                    .copied()
                    .collect::<HashSet<_>>(),
            );
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() || n > 0 {
                eprintln!(
                    "global-sfm: pre-global PnP seeded {n} free camera(s) as pose priors \
                     (min_corrs={min_corrs})"
                );
            }
            owned
        })
    } else {
        None
    };
    let pose_priors = seeded_priors_storage.as_deref().or(pose_priors);

    // Seed selection: the caller's seed may sit in a small side component
    // (courtyard-class view graphs have them); solve the LARGEST connected
    // component instead.
    let adjacency_for_seed =
        build_adjacency(features.len(), &edges).ok_or(GlobalReconstructionError::NoUsableEdges)?;
    let seed = {
        let mut seen = vec![false; features.len()];
        let mut best = (0usize, usize::MAX); // (size, smallest member)
        let mut preferred_size = None;
        for start in 0..features.len() {
            if seen[start] || !adjacency_for_seed.contains_key(&start) {
                continue;
            }
            let mut stack = vec![start];
            seen[start] = true;
            let (mut size, mut smallest) = (0usize, usize::MAX);
            while let Some(node) = stack.pop() {
                size += 1;
                smallest = smallest.min(node);
                if let Some(nbrs) = adjacency_for_seed.get(&node) {
                    for &(n, _) in nbrs {
                        if !seen[n] {
                            seen[n] = true;
                            stack.push(n);
                        }
                    }
                }
            }
            if start == tuning.seed {
                preferred_size = Some(size);
            }
            if (size, std::cmp::Reverse(smallest)) > (best.0, std::cmp::Reverse(best.1)) {
                best = (size, smallest);
            }
        }
        // Hybrid: pin gauge to a prior camera inside the largest component.
        if let Some(priors) = pose_priors {
            let mut prior_in_best = None;
            let mut seen = vec![false; features.len()];
            // Re-walk the best component to find any prior member.
            if best.1 < features.len() && adjacency_for_seed.contains_key(&best.1) {
                let mut stack = vec![best.1];
                seen[best.1] = true;
                while let Some(node) = stack.pop() {
                    if priors.get(node).and_then(|p| p.as_ref()).is_some() {
                        prior_in_best = Some(node);
                        break;
                    }
                    if let Some(nbrs) = adjacency_for_seed.get(&node) {
                        for &(n, _) in nbrs {
                            if !seen[n] {
                                seen[n] = true;
                                stack.push(n);
                            }
                        }
                    }
                }
            }
            if let Some(p) = prior_in_best {
                p
            } else {
                match preferred_size {
                    Some(size) if size == best.0 => tuning.seed,
                    _ => best.1,
                }
            }
        } else {
            match preferred_size {
                Some(size) if size == best.0 => tuning.seed,
                _ => best.1,
            }
        }
    };

    let solved = {
        // Multi-seed rotation averaging: try the component's highest-degree
        // nodes and keep the solve that triangulates with the lowest mean
        // reprojection. Bearing residual alone prefers self-consistent wrong
        // basins on repetitive façades; image reprojection breaks the tie.
        // With absolute pose priors the gauge is fixed — a single seed suffices.
        let trials = if pose_priors.is_some() {
            1
        } else {
            tuning.rotation_seed_trials.max(1)
        };
        let mut candidate_seeds = vec![seed];
        if trials > 1 {
            let mut degrees: Vec<(usize, usize)> = adjacency_for_seed
                .iter()
                .map(|(&node, nbrs)| (nbrs.len(), node))
                .collect();
            degrees.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            for &(_, node) in degrees.iter().take(trials) {
                if !candidate_seeds.contains(&node) {
                    candidate_seeds.push(node);
                }
            }
            candidate_seeds.truncate(trials);
        }
        let built_for_score = build_track_output(features, pairwise, mapper, Some(camera));
        // (solution, registered, reproj, bearing_rad, obs_count, seed, flip_alts, trans_flips)
        let mut candidates: Vec<(
            GlobalSfmPoses,
            usize,
            f64,
            f64,
            usize,
            usize,
            bool,
            usize,
        )> = Vec::new();
        // When multi-hypothesis edges are present, also try the global
        // "all-alternates-as-primary" basin — MST seeding locks onto the
        // stored primary, so adaptive consensus alone may never leave it.
        let hyp_flips: &[bool] = if tuning.multi_hypothesis_edges {
            &[false, true]
        } else {
            &[false]
        };
        for &trial_seed in &candidate_seeds {
            for &flip_alts in hyp_flips {
                let mut trial_edges = edges.clone();
                if flip_alts {
                    for e in &mut trial_edges {
                        e.swap_primary_alternate();
                    }
                }
                let joint = if tuning.joint_global_positioning {
                    Some((
                        camera,
                        features,
                        built_for_score.tracks.as_slice(),
                    ))
                } else {
                    None
                };
                let Some(solution) = solve_global_sfm_with_options(
                    features.len(),
                    &mut trial_edges,
                    trial_seed,
                    tuning.sweeps,
                    tuning.cg_iterations,
                    tuning.max_edge_rotation_error_deg,
                    tuning.refine_translations_with_global_rotations,
                    Some(camera),
                    pose_priors,
                    !tuning.hybrid_rotation_priors_only,
                    joint,
                    tuning.repair_edges_from_pose_priors,
                    tuning.metric_scale_from_pose_priors,
                    tuning.independent_edge_scales,
                ) else {
                    continue;
                };
                let registered = solution.poses.iter().filter(|p| p.is_some()).count();
                let mut track_point = vec![None; built_for_score.tracks.len()];
                triangulate_pending(
                    camera,
                    features,
                    &built_for_score.tracks,
                    &solution.poses,
                    mapper,
                    &mut track_point,
                );
                let (mut reproj_sum, mut reproj_count) = (0.0f64, 0usize);
                for (track_id, track) in built_for_score.tracks.iter().enumerate() {
                    let Some(position) = track_point[track_id] else {
                        continue;
                    };
                    for &(image, kp) in track {
                        let Some(pose) = &solution.poses[image] else {
                            continue;
                        };
                        let Some(pixel) = features[image].keypoints.get(kp).copied() else {
                            continue;
                        };
                        if let Some(error) = reprojection_error_px(camera, pose, &position, &pixel)
                        {
                            reproj_sum += error;
                            reproj_count += 1;
                        }
                    }
                }
                let reproj = if reproj_count > 0 {
                    reproj_sum / reproj_count as f64
                } else {
                    f64::INFINITY
                };
                if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                    eprintln!(
                        "global-sfm debug: seed {trial_seed} flip_alts={flip_alts} registered={registered} \
                         pre-BA reproj={reproj:.3} px bearing={:.2} deg obs={reproj_count}",
                        solution.mean_bearing_residual_rad.to_degrees()
                    );
                }
                let bearing = solution.mean_bearing_residual_rad;
                let trans_flips = solution.translation_refine_flips;
                candidates.push((
                    solution,
                    registered,
                    reproj,
                    bearing,
                    reproj_count,
                    trial_seed,
                    flip_alts,
                    trans_flips,
                ));
            }
        }
        // Rank: most cameras, then require ≥50% of the densest triangulation
        // among that registration count (rejects collapsed self-consistent
        // basins with tiny track support), then lowest reproj*(1+bearing)
        // score. `trans_flips` is logged for diagnostics only — courtyard A/B
        // showed fewer flips does not imply better GT alignment.
        let max_reg = candidates.iter().map(|c| c.1).max().unwrap_or(0);
        let max_obs = candidates
            .iter()
            .filter(|c| c.1 == max_reg)
            .map(|c| c.4)
            .max()
            .unwrap_or(0);
        let obs_floor = max_obs / 2;
        let best = candidates
            .into_iter()
            .filter(|c| c.1 == max_reg && c.4 >= obs_floor)
            .min_by(|a, b| {
                let score = |c: &(_, _, f64, f64, _, _, _, _)| {
                    if c.2.is_finite() {
                        c.2 * (1.0 + c.3)
                    } else {
                        f64::INFINITY
                    }
                };
                score(a)
                    .partial_cmp(&score(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && (trials > 1 || hyp_flips.len() > 1)
        {
            if let Some((_, reg, reproj, bearing, obs, chosen, flip, trans_flips)) = &best {
                eprintln!(
                    "global-sfm debug: multi-seed chose seed {chosen} flip_alts={flip} \
                     (registered={reg}, reproj={reproj:.3} px, bearing={:.2} deg, obs={obs}, \
                     trans_flips={trans_flips}, floor={obs_floor}) \
                     from {} seeds × {} hyp configs",
                    bearing.to_degrees(),
                    candidate_seeds.len(),
                    hyp_flips.len()
                );
            }
        }
        best.map(|(s, _, _, _, _, _, _, _)| s)
    }
    .ok_or(GlobalReconstructionError::NoUsableEdges)?;
    let mut solved = solved;
    if tuning.repair_free_edges_from_solved_poses {
        if let Some(priors) = pose_priors {
            let pose_field: Vec<Option<Pose>> = (0..features.len())
                .map(|i| {
                    priors
                        .get(i)
                        .and_then(|p| p.clone())
                        .or_else(|| solved.poses.get(i).cloned().flatten())
                })
                .collect();
            let limit: HashSet<usize> = tuning
                .repair_free_edges_image_indices
                .iter()
                .copied()
                .collect();
            let (rewritten, flipped) = repair_free_incident_edges_from_poses(
                &mut edges,
                &pose_field,
                priors,
                tuning.repair_free_edges_only_flipped,
                &limit,
            );
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() || rewritten > 0 {
                eprintln!(
                    "global-sfm: free-incident edge repair from pass-1 poses rewrote \
                     {rewritten} edges (flipped {flipped}); re-solving"
                );
            }
            if rewritten > 0 {
                if let Some(solution2) = solve_global_sfm_with_options(
                    features.len(),
                    &mut edges,
                    seed,
                    tuning.sweeps,
                    tuning.cg_iterations,
                    tuning.max_edge_rotation_error_deg,
                    tuning.refine_translations_with_global_rotations,
                    Some(camera),
                    pose_priors,
                    !tuning.hybrid_rotation_priors_only,
                    None,
                    false,
                    tuning.metric_scale_from_pose_priors,
                    tuning.independent_edge_scales,
                ) {
                    solved = solution2;
                }
            }
        }
    }
    if tuning.drop_free_edges_antipodal_to_solved {
        if let Some(priors) = pose_priors {
            let pose_field: Vec<Option<Pose>> = (0..features.len())
                .map(|i| {
                    priors
                        .get(i)
                        .and_then(|p| p.clone())
                        .or_else(|| solved.poses.get(i).cloned().flatten())
                })
                .collect();
            let limit: HashSet<usize> = tuning
                .repair_free_edges_image_indices
                .iter()
                .copied()
                .collect();
            let (dropped, antipodal) =
                drop_antipodal_free_incident_edges(&mut edges, &pose_field, priors, &limit);
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() || dropped > 0 {
                eprintln!(
                    "global-sfm: dropped {dropped} antipodal free-incident edges \
                     ({antipodal} flipped vs pass-1); re-solving"
                );
            }
            if dropped > 0 {
                edges.retain(|e| e.weight > 0.0);
                if let Some(solution2) = solve_global_sfm_with_options(
                    features.len(),
                    &mut edges,
                    seed,
                    tuning.sweeps,
                    tuning.cg_iterations,
                    tuning.max_edge_rotation_error_deg,
                    tuning.refine_translations_with_global_rotations,
                    Some(camera),
                    pose_priors,
                    !tuning.hybrid_rotation_priors_only,
                    None,
                    false,
                    tuning.metric_scale_from_pose_priors,
                    tuning.independent_edge_scales,
                ) {
                    solved = solution2;
                }
            }
        }
    }
    let poses = solved.poses;
    if !poses.iter().any(Option::is_some) {
        return Err(GlobalReconstructionError::NoUsableEdges);
    }

    // ---- Triangulate tracks against the averaged cameras -------------------
    let built = build_track_output(features, pairwise, mapper, Some(camera));
    let tracks = built.tracks;
    let mut track_point = vec![None; tracks.len()];
    let mut poses = poses;
    triangulate_pending(camera, features, &tracks, &poses, mapper, &mut track_point);

    // ---- One joint bundle adjustment ---------------------------------------
    let _ba_result = run_bundle_adjustment(
        camera,
        features,
        &tracks,
        mapper,
        &mut poses,
        &mut track_point,
        false,
    )
    .map_err(|error| GlobalReconstructionError::Ba(error.to_string()))?;

    if tuning.repnp_free_cameras_from_priors {
        if let Some(priors) = pose_priors {
            let min_corrs = if tuning.repnp_free_min_corrs > 0 {
                tuning.repnp_free_min_corrs
            } else {
                mapper.min_pnp_inliers
            };
            // Soft-anchors: free cameras that we preferred E for (hub stems)
            // count as structure anchors so a dropped hub (DSC_0296) can still
            // support its neighbour free cameras.
            let mut anchors = priors.to_vec();
            for &idx in &tuning.prefer_essential_edge_image_indices {
                if idx < anchors.len() && anchors[idx].is_none() {
                    if let Some(pose) = poses.get(idx).and_then(|p| p.as_ref()) {
                        anchors[idx] = Some(pose.clone());
                    }
                }
            }
            let n = repnp_free_cameras_from_prior_structure(
                camera,
                features,
                &tracks,
                &track_point,
                &mut poses,
                &anchors,
                mapper,
                min_corrs,
            );
            if n > 0 {
                if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                    eprintln!(
                        "global-sfm debug: re-PnP replaced {n} free camera(s) from prior structure"
                    );
                }
                let _ = run_bundle_adjustment(
                    camera,
                    features,
                    &tracks,
                    mapper,
                    &mut poses,
                    &mut track_point,
                    false,
                )
                .map_err(|error| GlobalReconstructionError::Ba(error.to_string()))?;
            }
        }
    }

    // ---- Residual PnP for images outside the view-graph component ----------
    // Same post-BA pass the incremental mapper uses: 2D–3D correspondences
    // from tracks that already have triangulated points. Courtyard's missing
    // DSC_0308 is typically disconnected from rotation averaging but may
    // still see enough reconstructed structure to register.
    let rescued = post_refinement_registration_pass(
        camera,
        features,
        &tracks,
        mapper,
        &mut poses,
        &mut track_point,
    )
    .map_err(|error| GlobalReconstructionError::Ba(error.to_string()))?;
    if rescued > 0 {
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
            eprintln!("global-sfm debug: residual PnP registered {rescued} leftover image(s)");
        }
        let _ = run_bundle_adjustment(
            camera,
            features,
            &tracks,
            mapper,
            &mut poses,
            &mut track_point,
            false,
        )
        .map_err(|error| GlobalReconstructionError::Ba(error.to_string()))?;
    }

    // ---- Assemble output tracks --------------------------------------------
    let mut out_tracks: Vec<SfmTrack> = Vec::new();
    let (mut reproj_sum, mut reproj_count) = (0.0f64, 0usize);
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(position) = track_point[track_id] else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, kp) in track {
            let Some(pose) = &poses[image] else {
                continue;
            };
            let Some(pixel) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            observations.push((image, kp, pixel));
            if let Some(error) = reprojection_error_px(camera, pose, &position, &pixel) {
                reproj_sum += error;
                reproj_count += 1;
            }
        }
        if observations.len() >= 2.min(mapper.min_track_length) {
            out_tracks.push(SfmTrack {
                position,
                observations,
            });
        }
    }
    let mean_reprojection_px = if reproj_count > 0 {
        reproj_sum / reproj_count as f64
    } else {
        f64::NAN
    };
    Ok((poses, out_tracks, mean_reprojection_px))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point2;

    /// Project a world point into a pose; `None` if behind camera or off-image.
    fn project_point(camera: &Camera, pose: &Pose, p: &Point3<f64>) -> Option<Point2<f64>> {
        let cam = pose.transform_world_point(p);
        if cam.z <= 0.05 {
            return None;
        }
        let px = camera.project(&cam)?;
        if px.x < 0.0 || px.x >= camera.width as f64 || px.y < 0.0 || px.y >= camera.height as f64 {
            return None;
        }
        Some(Point2::new(px.x, px.y))
    }

    /// A synthetic ring scene with rendered features and ground-truth
    /// pairwise matches across every co-observing pair — the global SfM
    /// end-to-end fixture (mirrors `incremental_sfm`'s own test scene).
    struct E2eScene {
        camera: Camera,
        gt_poses: Vec<Pose>,
        features: Vec<FeatureSet>,
        pairwise: Vec<PairwiseMatches>,
    }

    fn build_e2e_scene(n: usize, radius: f64) -> E2eScene {
        use visloc_vision::features::FeatureSet as FS;
        // Point cloud around the origin.
        let mut points = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in 0..=2 {
                    points.push(Point3::new(
                        xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.3,
                    ));
                }
            }
        }
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut gt_poses = Vec::new();
        for k in 0..n {
            let angle = std::f64::consts::TAU * k as f64 / n as f64;
            let center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            let forward = (Point3::origin() - center).normalize();
            let world_up = Vector3::new(0.0, 1.0, 0.0);
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward);
            let r_c2w = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let q_w2c = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r_c2w),
            )
            .inverse();
            let t_w2c = -(q_w2c * center.coords);
            gt_poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }
        let mut features = Vec::new();
        let mut visible: Vec<std::collections::HashMap<usize, usize>> = Vec::new();
        for pose in &gt_poses {
            let (kps, descs): (Vec<_>, Vec<_>) = points
                .iter()
                .enumerate()
                .filter_map(|(pidx, p)| {
                    project_point(&camera, pose, p).map(|px| (px, vec![pidx as f32, 1.0, 0.0, 0.0]))
                })
                .unzip();
            visible.push(
                points
                    .iter()
                    .enumerate()
                    .filter_map(|(pidx, p)| project_point(&camera, pose, p).map(|px| (pidx, px)))
                    .enumerate()
                    .map(|(kidx, (pidx, _))| (pidx, kidx))
                    .collect(),
            );
            features.push(FS::new(kps, descs).unwrap());
        }
        let mut pairwise = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut matches = Vec::new();
                for (&pidx, &ki) in &visible[i] {
                    if let Some(&kj) = visible[j].get(&pidx) {
                        matches.push((ki, kj));
                    }
                }
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                        two_view_config: None,
                        essential_matches: None,
                        essential_matrix: None,
                    });
                }
            }
        }
        E2eScene {
            camera,
            gt_poses,
            features,
            pairwise,
        }
    }

    #[test]
    fn reconstruct_global_sfm_end_to_end_recovers_the_ring() {
        let n = 6usize;
        let scene = build_e2e_scene(n, 3.0);
        assert!(
            scene.pairwise.len() >= n,
            "fixture sanity: the full ring must be connected"
        );
        let tuning = GlobalReconstructionTuning::default();
        let mapper = crate::incremental_sfm::IncrementalSfmConfig {
            min_track_length: 3,
            max_reprojection_error_px: 2.0,
            ..crate::incremental_sfm::IncrementalSfmConfig::default()
        };
        let (poses, tracks, mean_reproj) = reconstruct_global_sfm(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &tuning,
            &mapper,
        )
        .expect("global reconstruction succeeds on the synthetic ring");
        assert_eq!(poses.iter().filter(|p| p.is_some()).count(), n);
        assert!(
            tracks.len() >= 20,
            "expected a dense track set, got {}",
            tracks.len()
        );
        assert!(
            mean_reproj.is_finite() && mean_reproj < 0.5,
            "mean reprojection {mean_reproj:.3} px too high"
        );
        // Gauge: the solver frame is camera 0's frame; rotations must match
        // R_i·R_0⁻¹ and centres up to one global scale.
        let gauge = scene.gt_poses[0].world_to_camera.rotation.inverse();
        for (image, pose) in poses.iter().enumerate() {
            let pose = pose.as_ref().unwrap();
            let expected = scene.gt_poses[image].world_to_camera.rotation * gauge;
            let rot_err = (pose.world_to_camera.rotation.inverse() * expected).angle();
            assert!(rot_err < 1e-2, "image {image} rotation error {rot_err} rad");
        }
        let c_est = |i: usize| poses[i].as_ref().unwrap().camera_center_world();
        let baseline_est = (c_est(0) - c_est(1)).norm();
        let r0 = scene.gt_poses[0].world_to_camera.rotation;
        let c0 = scene.gt_poses[0].camera_center_world();
        let scale = (r0 * (scene.gt_poses[1].camera_center_world() - c0)).norm() / baseline_est;
        for image in 0..n {
            let expected = r0 * (scene.gt_poses[image].camera_center_world() - c0);
            let err = (c_est(image) * scale - expected).coords.norm();
            assert!(err < 5e-2, "image {image} centre error {err} after scaling");
        }
    }

    #[test]
    fn filter_pose_priors_drops_sparse_high_reproj_cameras() {
        use visloc_core::types::Camera;
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let pose_a = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let pose_b = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::x());
        let priors = vec![Some(pose_a.clone()), Some(pose_b.clone())];
        let tracks = vec![
            SfmTrack {
                position: Point3::new(0.0, 0.0, 5.0),
                observations: (0..60).map(|_| (0, 0, Point2::new(320.0, 240.0))).collect(),
            },
            SfmTrack {
                position: Point3::new(0.1, 0.0, 5.0),
                observations: vec![(1, 0, Point2::new(330.0, 240.0)); 3],
            },
        ];
        let (filtered, kept) =
            filter_pose_priors_by_track_quality(&camera, &priors, &tracks, 50, 0.45);
        assert_eq!(kept, 1);
        assert!(filtered[0].is_some());
        assert!(filtered[1].is_none());
    }

    #[test]
    fn filter_pose_priors_by_edge_disagreement_drops_flipped_hub() {
        // Star of 3 cameras: essentials place hub 1 at (+1,0); prior puts it
        // at (−3,4) so both incident bearings flip → hub is dropped.
        let q = UnitQuaternion::identity();
        let pose = |c: Vector3<f64>| Pose::from_world_to_camera(q, -c);
        let priors = vec![
            Some(pose(Vector3::new(0.0, 0.0, 0.0))),
            Some(pose(Vector3::new(-3.0, 4.0, 0.0))),
            Some(pose(Vector3::new(0.0, 2.0, 0.0))),
        ];
        let edges = vec![
            GlobalSfmEdge {
                image_i: 0,
                image_j: 1,
                rotation_ij: q,
                direction_ij: Vector3::x(),
                weight: 100.0,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
            GlobalSfmEdge {
                image_i: 2,
                image_j: 1,
                rotation_ij: q,
                direction_ij: Vector3::new(1.0, -2.0, 0.0).normalize(),
                weight: 100.0,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
        ];
        let (filtered, kept) = filter_pose_priors_by_edge_disagreement(&priors, &edges, 2, 0.5, 1);
        assert_eq!(kept, 2);
        assert!(filtered[1].is_none(), "bent hub must be dropped");
        assert!(filtered[0].is_some());
        assert!(filtered[2].is_some());
    }

    #[test]
    fn filter_pose_priors_by_free_centre_residual_drops_outlier() {
        let q = UnitQuaternion::identity();
        let pose = |c: Vector3<f64>| Pose::from_world_to_camera(q, -c);
        let priors = vec![
            Some(pose(Vector3::new(0.0, 0.0, 0.0))),
            Some(pose(Vector3::new(1.0, 0.0, 0.0))),
            Some(pose(Vector3::new(2.0, 0.0, 0.0))),
            Some(pose(Vector3::new(3.0, 0.0, 0.0))),
        ];
        // Free centres match priors except camera 1 is far off.
        let free = vec![
            Some(Point3::new(0.0, 0.0, 0.0)),
            Some(Point3::new(1.0, 5.0, 0.0)),
            Some(Point3::new(2.0, 0.0, 0.0)),
            Some(Point3::new(3.0, 0.0, 0.0)),
        ];
        let (filtered, kept) = filter_pose_priors_by_free_centre_residual(&priors, &free, 1, 2.0);
        assert_eq!(kept, 3);
        assert!(filtered[1].is_none());
        assert!(filtered[0].is_some() && filtered[2].is_some() && filtered[3].is_some());
    }

    #[test]
    fn rotation_only_priors_keep_global_centres_not_incremental() {
        let n = 6usize;
        let (gt_poses, edges) = ring_scene(n, 3.0);
        let bad = 2usize;
        let gt_c = gt_poses[bad].camera_center_world();
        let wrong_c = gt_c + Vector3::new(5.0, 0.0, 0.0);
        let p = &gt_poses[bad];
        let wrong_prior = Pose::from_world_to_camera(
            p.world_to_camera.rotation,
            -p.world_to_camera.rotation.transform_vector(&wrong_c.coords),
        );
        let priors: Vec<Option<Pose>> = gt_poses
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                if i == bad {
                    Some(wrong_prior.clone())
                } else {
                    Some(pose.clone())
                }
            })
            .collect();
        let mut edges_full = edges.clone();
        let full = solve_global_sfm(
            n,
            &mut edges_full,
            0,
            12,
            400,
            10.0,
            false,
            None,
            Some(&priors),
            true,
            None,
            false,
            false,
        )
        .unwrap();
        let mut edges_rot = edges;
        let rot_only = solve_global_sfm(
            n,
            &mut edges_rot,
            0,
            12,
            400,
            10.0,
            false,
            None,
            Some(&priors),
            false,
            None,
            false,
            false,
        )
        .unwrap();
        let full_c = full.poses[bad].as_ref().unwrap().camera_center_world();
        let rot_c = rot_only.poses[bad].as_ref().unwrap().camera_center_world();
        assert!(
            (full_c - wrong_c).norm() < 1e-3,
            "full priors must pin wrong incremental centre"
        );
        assert!(
            (rot_c - gt_c).norm() < (rot_c - wrong_c).norm(),
            "rotation-only {rot_c:?} should be closer to GT {gt_c:?} than wrong {wrong_c:?}"
        );
        assert!(
            (rot_c - gt_c).norm() < 0.5,
            "rotation-only should recover ring geometry within 50 cm"
        );
    }

    #[test]
    fn calibrated_view_edges_only_skips_uncalibrated_pairs() {
        let n = 6usize;
        let mut scene = build_e2e_scene(n, 3.0);
        for pair in &mut scene.pairwise {
            pair.two_view_config = Some(ConfigurationType::Uncalibrated);
        }
        let mapper = crate::incremental_sfm::IncrementalSfmConfig {
            min_track_length: 3,
            max_reprojection_error_px: 2.0,
            ..crate::incremental_sfm::IncrementalSfmConfig::default()
        };
        let blocked = GlobalReconstructionTuning {
            calibrated_view_edges_only: true,
            ..GlobalReconstructionTuning::default()
        };
        let err = reconstruct_global_sfm(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &blocked,
            &mapper,
        );
        assert_eq!(err, Err(GlobalReconstructionError::NoUsableEdges));

        for pair in &mut scene.pairwise {
            pair.two_view_config = Some(ConfigurationType::Calibrated);
        }
        let (poses, _, _) = reconstruct_global_sfm(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &blocked,
            &mapper,
        )
        .expect("calibrated configs remain usable");
        assert_eq!(poses.iter().filter(|p| p.is_some()).count(), n);

        // Uncalibrated config with a strong essential subset must still pass
        // the calibrated-only gate (courtyard F-won / E-bearing bridges).
        for pair in &mut scene.pairwise {
            pair.two_view_config = Some(ConfigurationType::Uncalibrated);
            pair.essential_matches = Some(pair.matches.clone());
        }
        let (poses_e, _, _) = reconstruct_global_sfm(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &blocked,
            &mapper,
        )
        .expect("uncalibrated+essential remains usable");
        assert_eq!(poses_e.iter().filter(|p| p.is_some()).count(), n);
    }

    #[test]
    fn reconstruct_global_sfm_reports_no_usable_edges_on_disconnected_input() {
        let scene = build_e2e_scene(4, 3.0);
        // Keep only pairs among images {0, 1}; images 2 and 3 become islands.
        let pairwise: Vec<PairwiseMatches> = scene
            .pairwise
            .iter()
            .filter(|p| p.image_i <= 1 && p.image_j <= 1)
            .cloned()
            .collect();
        let tuning = GlobalReconstructionTuning::default();
        // Two-image component can only form length-2 tracks.
        let mapper = crate::incremental_sfm::IncrementalSfmConfig {
            min_track_length: 2,
            ..crate::incremental_sfm::IncrementalSfmConfig::default()
        };
        let (poses, _tracks, _mean) =
            reconstruct_global_sfm(&scene.camera, &scene.features, &pairwise, &tuning, &mapper)
                .expect("the connected pair still solves");
        assert!(poses[0].is_some() && poses[1].is_some());
        assert!(
            poses[2].is_none() && poses[3].is_none(),
            "islands stay unposed"
        );
    }

    /// Deterministic pseudo-random unit vector.
    fn rand_unit(rng: &mut u64) -> Vector3<f64> {
        let next = |rng: &mut u64| {
            *rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*rng >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        loop {
            let v = Vector3::new(next(rng), next(rng), next(rng));
            let n = v.norm();
            if n.is_finite() && n > 1e-3 {
                return v / n;
            }
        }
    }

    fn random_rotation(rng: &mut u64) -> UnitQuaternion<f64> {
        let axis = nalgebra::Unit::new_normalize(rand_unit(rng));
        let angle = next_u01(rng) * std::f64::consts::PI;
        UnitQuaternion::from_axis_angle(&axis, angle)
    }

    fn next_u01(rng: &mut u64) -> f64 {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*rng >> 11) as f64 / (1u64 << 53) as f64
    }

    /// A ring of `n` cameras at radius `radius` looking at the origin, with
    /// exact pairwise edges (rotation + bearing direction derived from GT).
    fn ring_scene(n: usize, radius: f64) -> (Vec<Pose>, Vec<GlobalSfmEdge>) {
        let mut poses = Vec::new();
        for k in 0..n {
            let angle = std::f64::consts::TAU * k as f64 / n as f64;
            let center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            let forward = (Point3::origin() - center).normalize();
            let world_up = Vector3::new(0.0, 1.0, 0.0);
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward);
            let r_c2w = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let q_w2c = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r_c2w),
            )
            .inverse();
            let t_w2c = -(q_w2c * center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }
        let mut edges = Vec::new();
        for i in 0..n {
            // Consecutive neighbour plus one skip edge per camera.
            for &skip in &[1usize, 2] {
                let j = (i + skip) % n;
                if j == i
                    || edges.iter().any(|e: &GlobalSfmEdge| {
                        (e.image_i == j && e.image_j == i) || (e.image_i == i && e.image_j == j)
                    })
                {
                    continue;
                }
                let ci = poses[i].camera_center_world();
                let cj = poses[j].camera_center_world();
                let r_i = poses[i].world_to_camera.rotation;
                let r_j = poses[j].world_to_camera.rotation;
                let rotation_ij = r_j * r_i.inverse();
                let direction_world = cj - ci;
                let direction_ij = r_i.transform_vector(&direction_world).normalize();
                edges.push(GlobalSfmEdge {
                    image_i: i,
                    image_j: j,
                    rotation_ij,
                    direction_ij,
                    weight: 100.0,
                    inlier_sample: Vec::new(),
                    rotation_alt: None,
                    direction_alt: None,
                });
            }
        }
        (poses, edges)
    }

    #[test]
    fn prior_edge_repair_unflips_prior_pair_direction() {
        let n = 6usize;
        let (gt_poses, mut edges) = ring_scene(n, 3.0);
        // Flip the 0-1 edge direction (both will be priors).
        let target = edges
            .iter_mut()
            .find(|e| (e.image_i == 0 && e.image_j == 1) || (e.image_i == 1 && e.image_j == 0))
            .expect("ring has 0-1 edge");
        let before = target.direction_ij;
        target.direction_ij = -before;
        let priors: Vec<Option<Pose>> = gt_poses.iter().cloned().map(Some).collect();
        let mut weights: Vec<f64> = edges.iter().map(|e| e.weight).collect();
        let (rewritten, flipped) = repair_edges_from_pose_priors(&mut edges, &mut weights, &priors);
        assert!(
            rewritten >= n,
            "every consecutive prior pair should rewrite"
        );
        assert!(
            flipped >= 1,
            "the deliberately flipped edge must be counted"
        );
        let fixed = edges
            .iter()
            .find(|e| (e.image_i == 0 && e.image_j == 1) || (e.image_i == 1 && e.image_j == 0))
            .unwrap();
        assert!(
            fixed.direction_ij.dot(&before) > 0.9,
            "repaired direction should match the original prior-consistent bearing"
        );
    }

    #[test]
    fn free_incident_edge_repair_uses_solved_poses() {
        let n = 6usize;
        let (gt_poses, mut edges) = ring_scene(n, 3.0);
        // Priors on 0..3; free on 3..6 — use 0,1,2 as priors
        let priors: Vec<Option<Pose>> = (0..n)
            .map(|i| {
                if i < 3 {
                    Some(gt_poses[i].clone())
                } else {
                    None
                }
            })
            .collect();
        let pose_field: Vec<Option<Pose>> = gt_poses.iter().cloned().map(Some).collect();
        // Flip a free-incident edge 2-4
        let target = edges
            .iter_mut()
            .find(|e| (e.image_i == 2 && e.image_j == 4) || (e.image_i == 4 && e.image_j == 2))
            .expect("ring has 2-4 edge");
        let before = target.direction_ij;
        target.direction_ij = -before;
        let (rewritten, flipped) = repair_free_incident_edges_from_poses(
            &mut edges,
            &pose_field,
            &priors,
            false,
            &HashSet::new(),
        );
        assert!(rewritten >= 1);
        assert!(flipped >= 1);
        let fixed = edges
            .iter()
            .find(|e| (e.image_i == 2 && e.image_j == 4) || (e.image_i == 4 && e.image_j == 2))
            .unwrap();
        assert!(fixed.direction_ij.dot(&before) > 0.9);
    }

    #[test]
    fn global_sfm_recovers_ring_rotations_and_center_geometry() {
        let n = 12usize;
        let (gt_poses, edges) = ring_scene(n, 3.0);
        let mut edges = edges;
        let result = solve_global_sfm(
            n, &mut edges, 0, 16, 400, 10.0, false, None, None, true, None, false, false,
        )
        .expect("graph is usable");
        // Every camera solved; rotations match GT directly (seed pins the
        // gauge frame), centre geometry matches up to the monocular scale.
        // The solver's world frame is camera 0's frame (seed w2c = identity),
        // so the gauge-corrected ground truth is R_i ∘ R_0⁻¹.
        let gauge = gt_poses[0].world_to_camera.rotation.inverse();
        for (image, pose) in result.poses.iter().enumerate() {
            let pose = pose
                .as_ref()
                .unwrap_or_else(|| panic!("image {image} unposed"));
            let expected = gt_poses[image].world_to_camera.rotation * gauge;
            let rot_err = (pose.world_to_camera.rotation.inverse() * expected).angle();
            assert!(rot_err < 1e-3, "image {image} rotation error {rot_err} rad");
        }
        // Centres: map GT into the solver frame (camera 0 at origin, its
        // axes as world axes) and align the monocular global scale on a
        // baseline pair before comparing.
        let r0 = gt_poses[0].world_to_camera.rotation;
        let c0 = gt_poses[0].camera_center_world();
        let mapped: Vec<Point3<f64>> = (0..n)
            .map(|i| Point3::from(r0 * (gt_poses[i].camera_center_world() - c0)))
            .collect();
        let baseline_true = (mapped[0] - mapped[1]).norm();
        let baseline_est = (result.poses[0].as_ref().unwrap().camera_center_world()
            - result.poses[1].as_ref().unwrap().camera_center_world())
        .norm();
        let scale = baseline_true / baseline_est;
        for (est, expected) in result.poses.iter().zip(mapped.iter()) {
            let est = est.as_ref().unwrap().camera_center_world() * scale;
            let err = (est - expected).norm();
            assert!(err < 5e-3, "centre error {err} after scaling");
        }
        assert!(
            result.mean_bearing_residual_rad < 1e-3,
            "bearing residual {}",
            result.mean_bearing_residual_rad
        );
    }

    #[test]
    fn global_sfm_leaves_disconnected_images_unposed() {
        let (_poses, mut edges) = ring_scene(8, 3.0);
        // Drop every edge touching image 7: the ring becomes a chain over 0..7
        // plus an isolated vertex. Rebuild edges only among 0..=6.
        edges.retain(|e| e.image_i != 7 && e.image_j != 7);
        let result = solve_global_sfm(
            9, &mut edges, 0, 12, 300, 10.0, false, None, None, true, None, false, false,
        )
        .expect("graph is usable");
        for image in 0..7 {
            assert!(result.poses[image].is_some(), "image {image} should solve");
        }
        assert!(
            result.poses[7].is_none(),
            "isolated image must stay unposed"
        );
        assert!(
            result.poses[8].is_none(),
            "out-of-range image must stay unposed"
        );
    }

    #[test]
    fn global_sfm_survives_small_edge_noise() {
        let n = 10usize;
        let (_gt_poses, edges) = ring_scene(n, 3.0);
        let mut rng = 42u64;
        let noisy: Vec<GlobalSfmEdge> = edges
            .iter()
            .map(|edge| {
                let jitter_axis = nalgebra::Unit::new_normalize(rand_unit(&mut rng));
                let jitter = UnitQuaternion::from_axis_angle(
                    &jitter_axis,
                    (next_u01(&mut rng) - 0.5) * 0.02,
                );
                GlobalSfmEdge {
                    rotation_ij: jitter * edge.rotation_ij,
                    direction_ij: (edge.direction_ij + rand_unit(&mut rng) * 0.02).normalize(),
                    ..edge.clone()
                }
            })
            .collect();
        let mut noisy = noisy;
        let result = solve_global_sfm(
            n, &mut noisy, 0, 16, 600, 10.0, false, None, None, true, None, false, false,
        )
        .unwrap();
        assert!(
            result.mean_bearing_residual_rad < 0.15,
            "residual {} too large under small noise",
            result.mean_bearing_residual_rad
        );
        for (image, pose) in result.poses.iter().enumerate() {
            assert!(pose.is_some(), "image {image} dropped under small noise");
        }
    }

    #[test]
    fn rotation_averaging_converges_from_inconsistent_tree_seed() {
        // Three cameras, fully connected with EXACT edges — but the tree seed
        // uses only two of them, so the third starts inconsistent and the
        // consensus sweeps must pull it to the exact solution (which here is
        // globally consistent by construction).
        let (poses, edges) = ring_scene(3, 3.0);
        let adjacency = build_adjacency(3, &edges).unwrap();
        let averaged = average_rotations(3, &edges, &adjacency, 0, 24, 10.0).0;
        let gauge = poses[0].world_to_camera.rotation.inverse();
        for (image, rotation) in averaged.iter().enumerate() {
            let q = rotation.expect("all three cameras reachable");
            let expected = poses[image].world_to_camera.rotation * gauge;
            let err = (q.inverse() * expected).angle();
            assert!(err < 1e-6, "image {image} rotation error {err}");
        }
    }

    #[test]
    fn position_averaging_recovers_collinear_and_offaxis_centers() {
        // Four centres forming a tetrahedron-ish configuration; directions
        // and rotations consistent with GT. The unit-displacement rows give a
        // well-posed scale.
        let gt_centers = [
            Point3::new(0.0f64, 0.0, 0.0),
            Point3::new(1.5, 0.2, -0.3),
            Point3::new(-0.4, 1.1, 0.5),
            Point3::new(0.3, -0.6, 1.4),
        ];
        // Identity world-to-camera rotations keep the frame mapping trivial:
        // bearings equal world displacement directions.
        let identity: Vec<Option<UnitQuaternion<f64>>> = vec![Some(UnitQuaternion::identity()); 4];
        let mut edges = Vec::new();
        for (i, _) in gt_centers.iter().enumerate() {
            for j in (i + 1)..4 {
                let d = gt_centers[j] - gt_centers[i];
                edges.push(GlobalSfmEdge {
                    image_i: i,
                    image_j: j,
                    rotation_ij: UnitQuaternion::identity(),
                    direction_ij: d.normalize(),
                    weight: 10.0,
                    inlier_sample: Vec::new(),
                    rotation_alt: None,
                    direction_alt: None,
                });
            }
        }
        let weights = vec![1.0; edges.len()];
        let positions = average_positions(&edges, &weights, &identity, 0, 500);
        // Gauge: seed pinned at the origin and exactly one unit-displacement
        // scale row, so the solution equals the true configuration up to one
        // uniform scale. Align on camera 1 and verify the rest.
        let s = (gt_centers[1] - gt_centers[0]).norm()
            / (positions[1].expect("member") - positions[0].expect("seed")).norm();
        for (image, expected) in gt_centers.iter().enumerate() {
            let got = positions[image].expect("tetrahedron member");
            let err =
                ((got - positions[0].expect("seed")).scale(s) - (expected - gt_centers[0])).norm();
            assert!(err < 1e-2, "centre {image}: {err:.4} off after scaling");
        }
    }

    #[test]
    fn independent_edge_scale_positions_recover_variable_baselines() {
        // Unlike the legacy unit-displacement formulation, this fixture has
        // six edges with deliberately different lengths.  A complete
        // four-camera bearing graph constrains the per-edge scales up to one
        // global gauge.
        let gt_centers = [
            Point3::new(0.0f64, 0.0, 0.0),
            Point3::new(1.5, 0.2, -0.3),
            Point3::new(-0.4, 1.1, 0.5),
            Point3::new(0.3, -0.6, 1.4),
        ];
        let identity: Vec<Option<UnitQuaternion<f64>>> =
            vec![Some(UnitQuaternion::identity()); gt_centers.len()];
        let mut edges = Vec::new();
        for i in 0..gt_centers.len() {
            for j in (i + 1)..gt_centers.len() {
                let d = gt_centers[j] - gt_centers[i];
                edges.push(GlobalSfmEdge {
                    image_i: i,
                    image_j: j,
                    rotation_ij: UnitQuaternion::identity(),
                    direction_ij: d.normalize(),
                    weight: 10.0,
                    inlier_sample: Vec::new(),
                    rotation_alt: None,
                    direction_alt: None,
                });
            }
        }
        let weights = vec![10.0; edges.len()];
        let positions =
            average_positions_with_independent_edge_scales(&edges, &weights, &identity, 0, 8)
                .expect("complete bearing graph should have full rank");
        let estimate = |i: usize| positions[i].expect("all cameras are reachable");
        let scale = (gt_centers[1] - gt_centers[0]).norm() / (estimate(1) - estimate(0)).norm();
        for (image, expected) in gt_centers.iter().enumerate() {
            let got = (estimate(image) - estimate(0)).scale(scale);
            let want = *expected - gt_centers[0];
            assert!(
                (got - want).norm() < 1e-6,
                "camera {image} differs by {:.6} m",
                (got - want).norm()
            );
        }

        // The linear system is built from physical endpoints, not traversal
        // order.  Reversing the edge stream must preserve the solution.
        let mut reversed = edges.clone();
        reversed.reverse();
        let reversed_positions =
            average_positions_with_independent_edge_scales(&reversed, &weights, &identity, 0, 8)
                .expect("reordered complete bearing graph should remain solvable");
        for (image, _) in gt_centers.iter().enumerate() {
            let a = estimate(image);
            let b = reversed_positions[image].expect("reordered camera");
            assert!((a - b).norm() < 1e-8, "camera {image} changed with order");
        }
    }

    #[test]
    fn independent_edge_scale_positions_reject_rank_deficient_tree() {
        let identity = vec![Some(UnitQuaternion::identity()); 3];
        let edges = vec![
            GlobalSfmEdge {
                image_i: 0,
                image_j: 1,
                rotation_ij: UnitQuaternion::identity(),
                direction_ij: Vector3::x(),
                weight: 1.0,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
            GlobalSfmEdge {
                image_i: 1,
                image_j: 2,
                rotation_ij: UnitQuaternion::identity(),
                direction_ij: Vector3::y(),
                weight: 1.0,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
        ];
        assert!(
            average_positions_with_independent_edge_scales(&edges, &[1.0, 1.0], &identity, 0, 8)
                .is_none(),
            "a tree has unconstrained edge scales and must use the legacy fallback"
        );
    }

    #[test]
    fn invalid_edges_are_rejected() {
        let good = GlobalSfmEdge {
            image_i: 0,
            image_j: 1,
            rotation_ij: UnitQuaternion::identity(),
            direction_ij: Vector3::x(),
            weight: 5.0,
            inlier_sample: Vec::new(),
            rotation_alt: None,
            direction_alt: None,
        };
        let degenerate = GlobalSfmEdge {
            image_i: 0,
            image_j: 0,
            ..good.clone()
        };
        assert!(build_adjacency(2, &[degenerate]).is_none());
        let zero_direction = GlobalSfmEdge {
            direction_ij: Vector3::zeros(),
            ..good.clone()
        };
        assert!(build_adjacency(2, &[zero_direction]).is_none());
        let negative_weight = GlobalSfmEdge {
            weight: -1.0,
            ..good.clone()
        };
        assert!(build_adjacency(2, &[negative_weight]).is_none());
        assert!(build_adjacency(2, &[good]).is_some());
        let _ = random_rotation(&mut 1u64); // exercised helper stays referenced
    }

    #[test]
    fn global_r_translation_refine_unflips_wrong_chirality() {
        use visloc_core::types::Camera;
        let camera = Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0);
        // Cam0 at origin looking +Z; cam1 translated along +X by 0.5.
        let r = UnitQuaternion::identity();
        let t_cam = Vector3::new(-0.5, 0.0, 0.0); // world-to-cam translation for cam1
                                                  // C_1 in cam0 = -R^T t = (0.5, 0, 0)
        let true_dir = Vector3::new(0.5, 0.0, 0.0).normalize();
        let mut samples = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                let p = Point3::new(xi as f64 * 0.3, yi as f64 * 0.3, 4.0);
                let p0 = camera.project(&p).unwrap();
                let p1_cam = r * p + t_cam;
                let p1 = camera.project(&Point3::from(p1_cam)).unwrap();
                samples.push((p0, p1));
            }
        }
        assert!(samples.len() >= 8);
        let mut edges = [GlobalSfmEdge {
            image_i: 0,
            image_j: 1,
            rotation_ij: r,
            direction_ij: -true_dir, // deliberately flipped
            weight: samples.len() as f64,
            inlier_sample: samples,
            rotation_alt: None,
            direction_alt: None,
        }];
        let rotations = vec![Some(r), Some(r)];
        let weights = vec![1.0];
        let flipped =
            refine_edge_directions_under_rotations(&mut edges, &rotations, &weights, &camera);
        assert_eq!(flipped, 1, "must flip the wrong-chirality edge");
        assert!(
            edges[0].direction_ij.dot(&true_dir) > 0.9,
            "refined direction {:?} should align with {:?}",
            edges[0].direction_ij,
            true_dir
        );
    }

    #[test]
    fn multi_hypothesis_rotation_averaging_picks_alternate() {
        // Triangle of identity cameras. Edge 0–1 and 0–2 are exact. Edge 1–2
        // stores a *wrong* primary rotation (90° about Z) but the correct
        // identity as alternate. Adaptive averaging must select the alternate
        // so all three orientations stay identity in the seed frame.
        let identity = UnitQuaternion::identity();
        let wrong =
            UnitQuaternion::from_axis_angle(&Vector3::z_axis(), std::f64::consts::FRAC_PI_2);
        let edges = vec![
            GlobalSfmEdge {
                image_i: 0,
                image_j: 1,
                rotation_ij: identity,
                direction_ij: Vector3::x(),
                weight: 100.0,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
            GlobalSfmEdge {
                image_i: 0,
                image_j: 2,
                rotation_ij: identity,
                direction_ij: Vector3::y(),
                weight: 100.0,
                inlier_sample: Vec::new(),
                rotation_alt: None,
                direction_alt: None,
            },
            GlobalSfmEdge {
                image_i: 1,
                image_j: 2,
                rotation_ij: wrong,
                direction_ij: (Vector3::y() - Vector3::x()).normalize(),
                weight: 50.0,
                inlier_sample: Vec::new(),
                rotation_alt: Some(identity),
                direction_alt: Some((Vector3::y() - Vector3::x()).normalize()),
            },
        ];
        let adjacency = build_adjacency(3, &edges).unwrap();
        let averaged = average_rotations(3, &edges, &adjacency, 0, 16, 10.0).0;
        for (image, rotation) in averaged.iter().enumerate() {
            let q = rotation.expect("reachable");
            let err = q.angle();
            assert!(
                err < 1e-3,
                "image {image} should stay near identity, err={err}"
            );
        }
        // Without the alternate, IRLS would zero the inconsistent edge or
        // bend image 2 — pin that primary-only fails the tight check.
        let primary_only: Vec<GlobalSfmEdge> = edges
            .iter()
            .map(|e| GlobalSfmEdge {
                rotation_alt: None,
                direction_alt: None,
                ..e.clone()
            })
            .collect();
        let adjacency_po = build_adjacency(3, &primary_only).unwrap();
        let averaged_po = average_rotations(3, &primary_only, &adjacency_po, 0, 16, 10.0).0;
        // With a wrong primary and no alt, at least one non-seed rotation
        // drifts OR the bad edge is trimmed — either way image 2 or the
        // consensus is worse than the multi-hyp solve. Require multi-hyp
        // strictly better on image 2.
        let err_multi = averaged[2].unwrap().angle();
        let err_primary = averaged_po[2].unwrap().angle();
        assert!(
            err_multi + 1e-6 < err_primary || err_primary > 0.1,
            "multi-hyp err={err_multi} should beat primary-only err={err_primary}"
        );
    }
}
