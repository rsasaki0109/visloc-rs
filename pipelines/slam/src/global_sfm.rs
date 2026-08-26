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

use nalgebra::{Matrix3, Point2, Point3, UnitQuaternion, Vector3};

use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_vision::two_view::{CheiralityOptions, RelativePoseEstimator, TwoViewCorrespondence};

use crate::incremental_sfm::{
    build_tracks_detailed, reprojection_error_px, run_bundle_adjustment, triangulate_pending,
    PairwiseMatches, SfmTrack,
};

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
}

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
) -> Option<GlobalSfmPoses> {
    let adjacency = build_adjacency(num_images, edges)?;
    let (rotations, edge_weights) = average_rotations(
        num_images,
        edges,
        &adjacency,
        seed,
        sweeps,
        max_edge_rotation_error_deg,
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
            let flipped =
                refine_edge_directions_under_rotations(edges, &rotations, &edge_weights, cam);
            if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
                eprintln!(
                    "global-sfm debug: global-R translation refine flipped {flipped} edge directions"
                );
            }
        }
    }
    let centers = average_positions(edges, &edge_weights, &rotations, seed, cg_iterations);
    let mut poses: Vec<Option<Pose>> = vec![None; num_images];
    for &image in &component {
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
        poses,
    })
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
    let edge = &edges[index];
    if edge.image_i == from {
        edge.rotation_ij
    } else {
        edge.rotation_ij.inverse()
    }
}

/// Maximum-spanning-tree rotation seeding (Kruskal over descending weights).
/// Returns per-image orientations plus the reached member list; images not
/// connected to the root stay `None`.
fn tree_seed_rotations(
    num_images: usize,
    edges: &[GlobalSfmEdge],
    adjacency: &Adjacency,
    root: usize,
) -> (Vec<Option<UnitQuaternion<f64>>>, Vec<usize>) {
    // Maximum-spanning-tree growth from the root, Prim-style: always extend
    // through the highest-weight edge leaving the reached set. A Kruskal
    // pass cannot be used here because it consumes tree edges inside the
    // unreached mass before the frontier ever reaches them, stranding the
    // seed (observed as posed=4 on courtyard's 37-image component).
    let mut reached = vec![false; num_images];
    let mut rotations: Vec<Option<UnitQuaternion<f64>>> = vec![None; num_images];
    reached[root] = true;
    rotations[root] = Some(UnitQuaternion::identity());
    // Frontier heap of (negated weight, tie-break index, from, edge_index).
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
    push_frontier(&mut heap, root, &reached);
    let mut members = vec![root];
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
    (rotations, members)
}

/// Iterative geodesic-consensus rotation averaging over the root's
/// component. Each sweep recomputes every non-seed orientation as the
/// weighted SO(3) mean of its neighbours' predictions.
pub fn average_rotations(
    num_images: usize,
    edges: &[GlobalSfmEdge],
    adjacency: &Adjacency,
    seed: usize,
    sweeps: usize,
    max_edge_rotation_error_deg: f64,
) -> (Vec<Option<UnitQuaternion<f64>>>, Vec<f64>) {
    // IRLS edge weights: start from the verified-inlier weight; each sweep
    // zeroes edges whose implied relative rotation disagrees with the
    // emerging global solution beyond the trim threshold.
    let mut weights: Vec<f64> = edges.iter().map(|e| e.weight.max(1.0)).collect();
    let max_error_rad = max_edge_rotation_error_deg.to_radians();
    let (mut rotations, members) = tree_seed_rotations(num_images, edges, adjacency, seed);
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() {
        eprintln!(
            "global-sfm debug: tree seed reached {} of {} images from root {}",
            members.len(),
            num_images,
            seed
        );
    }
    let mut ordered_members = members.clone();
    ordered_members.sort_unstable();
    for _ in 0..sweeps {
        // Gauss-Seidel style: consume this sweep's own updates immediately
        // (Jacobi snapshots can oscillate on noisy view graphs).
        for &image in &ordered_members {
            if image == seed {
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
                let predicted = step_from(edges, edge_index, neighbor) * neighbor_q;
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
                    let predicted = step_from(edges, edge_index, e.image_i) * qi;
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
        let world = q.inverse().transform_vector(&edge.direction_ij);
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

    // One scale-fixing unit-displacement row on the highest-weight edge
    // touching the seed: the perpendicular rows alone are homogeneous, so
    // exactly one unit row fixes the global scale without biasing any other
    // edge's length. It is handled as an affine rhs row, not part of H.
    let scale_index = bearings
        .iter()
        .enumerate()
        .filter(|(_, b)| b.i == seed || b.j == seed)
        .max_by(|(ia, a), (ib, b)| {
            a.weight
                .total_cmp(&b.weight)
                .then_with(|| a.j.cmp(&b.j).reverse())
                .then_with(|| ib.cmp(ia))
        })
        .map(|(i, _)| i);

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
    if let Some(k) = scale_index {
        let b = &bearings[k];
        *rhs.entry(b.i).or_default() += b.direction.scale(-b.weight);
        *rhs.entry(b.j).or_default() += b.direction.scale(b.weight);
    }

    let hess_vec = |bs: &[Bearing],
                    ws: &[f64],
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
        for (&node, val) in v {
            *out.entry(node).or_default() += val.scale(PRIOR_WEIGHT);
        }
        out
    };

    let solve_cg = |ws: &[f64], cg_iterations: usize| -> HashMap<usize, Vector3<f64>> {
        let zero_field: HashMap<usize, Vector3<f64>> =
            members.iter().map(|&i| (i, Vector3::zeros())).collect();
        let mut out = zero_field.clone();
        let h0 = hess_vec(bearings.as_slice(), ws, &zero_field);
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
            let hp = hess_vec(bearings.as_slice(), ws, &p);
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
            if Some(k) == scale_index {
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
                .filter(|&k| Some(k) != scale_index)
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
            .filter(|&k| Some(k) != scale_index)
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
    // ---- Relative poses → global edges ------------------------------------
    let estimator = RelativePoseEstimator {
        cheirality: if tuning.chirality_harden_edges {
            CheiralityOptions::hardened()
        } else {
            CheiralityOptions::default()
        },
        ..RelativePoseEstimator::default()
    };
    let mut edges = Vec::new();
    let mut chirality_rejected = 0usize;
    for pair in pairwise {
        if pair.matches.len() < tuning.min_pair_matches {
            continue;
        }
        let correspondences: Vec<TwoViewCorrespondence> = pair
            .matches
            .iter()
            .filter_map(|&(ki, kj)| {
                Some(TwoViewCorrespondence::new(
                    *features[pair.image_i].keypoints.get(ki)?,
                    *features[pair.image_j].keypoints.get(kj)?,
                ))
            })
            .collect();
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            if tuning.chirality_harden_edges {
                chirality_rejected += 1;
            }
            continue;
        };
        if relative.inliers.len() < tuning.min_edge_inliers {
            continue;
        }
        // Bearing of camera j's centre expressed in camera i's frame:
        // C_j^(i) = −R_ijᵀ t_ij.
        let r_ij = relative.previous_to_current.rotation;
        let t_ij = relative.previous_to_current.translation;
        let Some(direction_ij) = (-r_ij.inverse().transform_vector(&t_ij)).try_normalize(1e-12)
        else {
            continue;
        };
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
        edges.push(GlobalSfmEdge {
            image_i: pair.image_i,
            image_j: pair.image_j,
            rotation_ij: r_ij,
            direction_ij,
            weight: relative.inliers.len() as f64,
            inlier_sample,
        });
    }
    if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && tuning.chirality_harden_edges {
        eprintln!(
            "global-sfm debug: chirality-harden rejected {chirality_rejected} pairs; kept {} edges",
            edges.len()
        );
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
        let step = |from: usize, index: usize| -> UnitQuaternion<f64> {
            let e = &edges[index];
            if e.image_i == from {
                e.rotation_ij
            } else {
                e.rotation_ij.inverse()
            }
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
                let loop_r = step(w, wu) * step(v, vw) * step(u, e_index);
                loop_errors[e_index].push(loop_r.angle());
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
        }
        edges.retain(|e| e.weight > 0.0);
    }

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
        match preferred_size {
            Some(size) if size == best.0 => tuning.seed,
            _ => best.1,
        }
    };

    let solved = {
        // Multi-seed rotation averaging: try the component's highest-degree
        // nodes (plus the selected seed) and keep the solve with the most
        // surviving edges / lowest median residual. Escapes flip basins that
        // a single bad tree-root can lock into on repetitive façades.
        let trials = tuning.rotation_seed_trials.max(1);
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
        let mut best: Option<(GlobalSfmPoses, usize, f64, usize)> = None;
        for &trial_seed in &candidate_seeds {
            let mut trial_edges = edges.clone();
            let Some(solution) = solve_global_sfm(
                features.len(),
                &mut trial_edges,
                trial_seed,
                tuning.sweeps,
                tuning.cg_iterations,
                tuning.max_edge_rotation_error_deg,
                tuning.refine_translations_with_global_rotations,
                Some(camera),
            ) else {
                continue;
            };
            // Score: prefer more registered cameras, then lower mean bearing
            // residual.
            let registered = solution.poses.iter().filter(|p| p.is_some()).count();
            let residual = solution.mean_bearing_residual_rad;
            let better = match &best {
                None => true,
                Some((_, best_reg, best_res, _)) => {
                    registered > *best_reg
                        || (registered == *best_reg && residual < *best_res)
                }
            };
            if better {
                best = Some((solution, registered, residual, trial_seed));
            }
        }
        if std::env::var_os("VISLOC_GLOBAL_DEBUG").is_some() && trials > 1 {
            if let Some((_, reg, res, chosen)) = &best {
                eprintln!(
                    "global-sfm debug: multi-seed chose seed {chosen} (registered={reg}, bearing_mean={:.3} deg) from {} trials",
                    res.to_degrees(),
                    candidate_seeds.len()
                );
            }
        }
        best.map(|(s, _, _, _)| s)
    }
    .ok_or(GlobalReconstructionError::NoUsableEdges)?;
    let poses = solved.poses;
    if !poses.iter().any(Option::is_some) {
        return Err(GlobalReconstructionError::NoUsableEdges);
    }

    // ---- Triangulate tracks against the averaged cameras -------------------
    let built = build_tracks_detailed(features.len(), pairwise, mapper.min_track_length);
    let tracks = built.tracks;
    let mut track_point = vec![None; tracks.len()];
    triangulate_pending(camera, features, &tracks, &poses, mapper, &mut track_point);

    // ---- One joint bundle adjustment ---------------------------------------
    let mut ba_poses = poses.clone();
    let _ba_result = run_bundle_adjustment(
        camera,
        features,
        &tracks,
        mapper,
        &mut ba_poses,
        &mut track_point,
        false,
    )
    .map_err(|error| GlobalReconstructionError::Ba(error.to_string()))?;
    let poses = ba_poses;

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
        let mapper = crate::incremental_sfm::IncrementalSfmConfig::default();
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
                });
            }
        }
        (poses, edges)
    }

    #[test]
    fn global_sfm_recovers_ring_rotations_and_center_geometry() {
        let n = 12usize;
        let (gt_poses, edges) = ring_scene(n, 3.0);
        let mut edges = edges;
        let result = solve_global_sfm(n, &mut edges, 0, 16, 400, 10.0, false, None)
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
        let mut edges = edges;
        let result = solve_global_sfm(9, &mut edges, 0, 12, 300, 10.0, false, None)
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
        let result = solve_global_sfm(n, &mut noisy, 0, 16, 600, 10.0, false, None).unwrap();
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
    fn invalid_edges_are_rejected() {
        let good = GlobalSfmEdge {
            image_i: 0,
            image_j: 1,
            rotation_ij: UnitQuaternion::identity(),
            direction_ij: Vector3::x(),
            weight: 5.0,
            inlier_sample: Vec::new(),
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
}
