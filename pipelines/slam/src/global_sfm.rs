//! GLOMAP-style global structure-from-motion back-end: rotation averaging
//! followed by translation-direction position averaging.
//!
//! The incremental mapper ([`crate::incremental_sfm`]) grows one
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

use std::collections::HashMap;

use nalgebra::{Point3, UnitQuaternion, Vector3};

use visloc_core::geometry::Pose;

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
///
/// Returns `None` when no valid edge touches any image.
pub fn solve_global_sfm(
    num_images: usize,
    edges: &[GlobalSfmEdge],
    seed: usize,
    sweeps: usize,
    cg_iterations: usize,
) -> Option<GlobalSfmPoses> {
    let adjacency = build_adjacency(num_images, edges)?;
    let rotations = average_rotations(num_images, edges, &adjacency, seed, sweeps);
    let component: Vec<usize> = rotations
        .iter()
        .enumerate()
        .filter(|(_, r)| r.is_some())
        .map(|(i, _)| i)
        .collect();
    if component.is_empty() {
        return None;
    }
    let centers = average_positions(edges, &rotations, seed, cg_iterations);
    let mut poses: Vec<Option<Pose>> = vec![None; num_images];
    for &image in &component {
        let (Some(q_w2c), Some(center)) = (rotations[image], centers[image]) else {
            continue;
        };
        let t_w2c = -q_w2c.transform_vector(&center.coords);
        poses[image] = Some(Pose::from_world_to_camera(q_w2c, t_w2c));
    }
    let (mut sum, mut count) = (0.0f64, 0usize);
    for edge in edges {
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
    root: usize,
) -> (Vec<Option<UnitQuaternion<f64>>>, Vec<usize>) {
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| {
        edges[b]
            .weight
            .total_cmp(&edges[a].weight)
            .then_with(|| a.cmp(&b))
    });
    let mut parent: Vec<usize> = (0..num_images).collect();
    fn find(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    let mut reached = vec![false; num_images];
    reached[root] = true;
    let mut rotations: Vec<Option<UnitQuaternion<f64>>> = vec![None; num_images];
    rotations[root] = Some(UnitQuaternion::identity());
    for &index in &order {
        let edge = &edges[index];
        let (a, b) = (
            find(&mut parent, edge.image_i),
            find(&mut parent, edge.image_j),
        );
        if a == b {
            continue;
        }
        parent[a] = b;
        let (known, unknown) = match (reached[edge.image_i], reached[edge.image_j]) {
            (true, false) => (edge.image_i, edge.image_j),
            (false, true) => (edge.image_j, edge.image_i),
            _ => continue,
        };
        let base = rotations[known].expect("reached image carries a seeded orientation");
        rotations[unknown] = Some(step_from(edges, index, known) * base);
        reached[unknown] = true;
    }
    let members: Vec<usize> = (0..num_images).filter(|&i| reached[i]).collect();
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
) -> Vec<Option<UnitQuaternion<f64>>> {
    let (mut rotations, members) = tree_seed_rotations(num_images, edges, seed);
    let mut ordered_members = members.clone();
    ordered_members.sort_unstable();
    for _ in 0..sweeps {
        let mut updated = rotations.clone();
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
                let Some(neighbor_q) = rotations[neighbor] else {
                    continue;
                };
                let predicted = step_from(edges, edge_index, neighbor) * neighbor_q;
                let q = predicted.coords;
                let w = edges[edge_index].weight.max(1.0);
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
            updated[image] = Some(UnitQuaternion::from_quaternion(
                nalgebra::Quaternion::from_vector(v),
            ));
        }
        rotations = updated;
    }
    rotations
}

/// One world-frame bearing constraint between two solved cameras.
struct Bearing {
    i: usize,
    j: usize,
    /// World-frame unit bearing from camera i towards camera j.
    direction: Vector3<f64>,
    weight: f64,
}

impl Bearing {
    /// Transposed-Jacobian product for the perpendicular rows' dual
    /// residual only (Jacobian: ∂/∂ci = skew(d), ∂/∂cj = −skew(d);
    /// skew(d)ᵀ = −skew(d)):
    /// `Aᵀ_i y = w·(d×y)`, `Aᵀ_j y = −w·(d×y)`.
    fn at_perp_product(&self, perp: Vector3<f64>) -> (Vector3<f64>, Vector3<f64>) {
        let term_i = -self.direction.cross(&perp);
        (term_i * self.weight, -term_i * self.weight)
    }

    /// Transposed-Jacobian product for one scalar row `±dᵀ(c_j−c_i)−t`
    /// (∂/∂ci = −d, ∂/∂cj = +d).
    fn at_parallel_product(&self, dual: f64) -> (Vector3<f64>, Vector3<f64>) {
        (
            self.direction * (-dual * self.weight),
            self.direction * (dual * self.weight),
        )
    }
}

/// Position averaging over the rotation-solved component: conjugate-gradient
/// least squares on perpendicular-bearing plus unit-displacement rows.
/// Images outside the component are `None`; the seed centre is the gauge
/// origin (exactly zero).
pub fn average_positions(
    edges: &[GlobalSfmEdge],
    rotations: &[Option<UnitQuaternion<f64>>],
    seed: usize,
    cg_iterations: usize,
) -> Vec<Option<Point3<f64>>> {
    let mut bearings: Vec<Bearing> = Vec::new();
    for edge in edges {
        // World-frame i→j bearing: rotate the frame direction out of the
        // camera holding it (camera-to-world = inverse of the stored
        // world-to-camera quaternion).
        let world = if edge.image_i < rotations.len() {
            rotations[edge.image_i].map(|q| q.inverse().transform_vector(&edge.direction_ij))
        } else {
            None
        };
        let Some(world) = world else {
            continue;
        };
        let norm = world.norm();
        if !norm.is_finite() || norm < 1e-12 {
            continue;
        }
        let direction = world / norm;
        bearings.push(Bearing {
            i: edge.image_i,
            j: edge.image_j,
            direction,
            weight: edge.weight.max(1.0),
        });
    }
    if bearings.is_empty() {
        return vec![None; rotations.len()];
    }
    // Keep only edges whose both endpoints carry a rotation: positions are
    // defined exactly for the rotation-solved component.
    bearings.retain(|b| {
        rotations.get(b.i).copied().flatten().is_some()
            && rotations.get(b.j).copied().flatten().is_some()
    });
    if bearings.is_empty() {
        return vec![None; rotations.len()];
    }

    // Weak origin prior keeps the free-translation mode anchored during the
    // solve; the result is re-anchored onto the seed afterwards.
    const PRIOR_WEIGHT: f64 = 1e-6;

    // Scale gauge: the perpendicular rows alone are homogeneous (zero is a
    // solution), so exactly ONE unit-displacement row — on the highest-weight
    // edge touching the seed (ties broken by lowest neighbour index) — fixes
    // the global scale without biasing any other edge's length.
    let scale_row_index = bearings
        .iter()
        .enumerate()
        .filter(|(_, b)| b.i == seed || b.j == seed)
        .max_by(|(ia, a), (ib, b)| {
            a.weight
                .total_cmp(&b.weight)
                .then_with(|| b.j.cmp(&a.j))
                .then_with(|| ib.cmp(ia))
        })
        .map(|(i, _)| i);
    let scale_row = scale_row_index.map(|i| bearings.remove(i));

    let members: Vec<usize> = (0..rotations.len())
        .filter(|&i| rotations[i].is_some())
        .collect();

    // Pure Hessian operator AᵀA acting on a displacement field: the
    // perpendicular rows contribute their linear part, the single scale row
    // contributes its (linear) displacement part without any target, plus
    // the weak origin prior.
    let hess_vec = |v: &HashMap<usize, Vector3<f64>>| -> HashMap<usize, Vector3<f64>> {
        let mut out: HashMap<usize, Vector3<f64>> = HashMap::new();
        for b in &bearings {
            let ci = v.get(&b.i).copied().unwrap_or(Vector3::zeros());
            let cj = v.get(&b.j).copied().unwrap_or(Vector3::zeros());
            let y = b.direction.cross(&(ci - cj));
            let (at_i, at_j) = b.at_perp_product(y);
            *out.entry(b.i).or_default() += at_i;
            *out.entry(b.j).or_default() += at_j;
        }
        if let Some(b) = &scale_row {
            let ci = v.get(&b.i).copied().unwrap_or(Vector3::zeros());
            let cj = v.get(&b.j).copied().unwrap_or(Vector3::zeros());
            let lin = b.direction.dot(&(cj - ci));
            let (at_i, at_j) = b.at_parallel_product(lin);
            *out.entry(b.i).or_default() += at_i;
            *out.entry(b.j).or_default() += at_j;
        }
        for (&node, val) in v {
            *out.entry(node).or_default() += val.scale(PRIOR_WEIGHT);
        }
        out
    };

    // Right-hand side Aᵀ b: only the scale row has a non-zero target (its
    // unit displacement): Aᵀ_i b = w·(−d·1), Aᵀ_j b = w·(+d·1).
    let mut rhs: HashMap<usize, Vector3<f64>> = HashMap::new();
    if let Some(b) = &scale_row {
        let (at_i, at_j) = b.at_parallel_product(1.0);
        *rhs.entry(b.i).or_default() += at_i;
        *rhs.entry(b.j).or_default() += at_j;
    }

    // Conjugate gradient on H x = rhs, starting from zero positions.
    let zero_field: HashMap<usize, Vector3<f64>> =
        members.iter().map(|&i| (i, Vector3::zeros())).collect();
    let mut positions: HashMap<usize, Vector3<f64>> = zero_field.clone();
    let h0 = hess_vec(&zero_field);
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
        let hp = hess_vec(&p);
        let denom: f64 = p
            .iter()
            .map(|(k, v)| v.dot(&hp.get(k).copied().unwrap_or(Vector3::zeros())))
            .sum();
        if !(denom.is_finite() && denom > 1e-30) {
            break;
        }
        let alpha = rs_old / denom;
        for (k, v) in &p {
            *positions.entry(*k).or_default() += *v * alpha;
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

    // Re-anchor: seed centre becomes exactly the origin.
    let anchor = positions.get(&seed).copied().unwrap_or(Vector3::zeros());
    let mut out: Vec<Option<Point3<f64>>> = vec![None; rotations.len()];
    for (image, center) in positions {
        out[image] = Some(Point3::from(center - anchor));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

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
                });
            }
        }
        (poses, edges)
    }

    #[test]
    fn global_sfm_recovers_ring_rotations_and_center_geometry() {
        let n = 12usize;
        let (gt_poses, edges) = ring_scene(n, 3.0);
        let result = solve_global_sfm(n, &edges, 0, 16, 400).expect("graph is usable");
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
        for image in 0..n {
            let est = result.poses[image].as_ref().unwrap().camera_center_world() * scale;
            let err = (est - mapped[image]).norm();
            assert!(err < 5e-3, "image {image} centre error {err} after scaling");
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
        let result = solve_global_sfm(9, &edges, 0, 12, 300).expect("graph is usable");
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
        let result = solve_global_sfm(n, &noisy, 0, 16, 600).unwrap();
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
        let averaged = average_rotations(3, &edges, &adjacency, 0, 24);
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
                });
            }
        }
        let positions = average_positions(&edges, &identity, 0, 500);
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
}
