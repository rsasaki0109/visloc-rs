//! Homography estimation and pose decomposition — COLMAP's planar/panoramic
//! degeneracy path for two-view verification.
//!
//! Ported from COLMAP (BSD-3-Clause, ETH Zurich / UNC Chapel Hill):
//! - DLT homography solver:
//!   `src/colmap/estimators/solvers/homography_matrix.cc`
//!   (`HomographyMatrixEstimator::Estimate` / `::Residuals`).
//! - Analytic decomposition + cheirality-based pose selection:
//!   `src/colmap/geometry/homography_matrix.cc`
//!   (`DecomposeHomographyMatrix`, `PoseFromHomographyMatrix`), an
//!   implementation of Malis & Vargas, "Deeper understanding of the
//!   homography decomposition for vision-based control", INRIA RR-6303
//!   (2007).
//!
//! Unlike the essential/fundamental 8-point solvers, COLMAP's homography DLT
//! does **not** Hartley-normalize the input pixel coordinates (see the cited
//! `Estimate()` — it builds the 2N×9 system directly from the raw image
//! points). This module preserves that choice for fidelity: a homography is
//! linear in projective coordinates, and is less numerically sensitive to
//! un-normalized pixel magnitudes than the epipolar solvers are.

use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, Point3, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use visloc_core::types::Camera;

use super::TwoViewCorrespondence;

/// Squared reprojection error `‖point2 − H·point1‖²` in pixel space, port of
/// `HomographyMatrixEstimator::Residuals`
/// (`src/colmap/estimators/solvers/homography_matrix.cc:94-131`).
pub fn homography_squared_error(
    homography: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
) -> Option<f64> {
    let p1 = Vector3::new(
        correspondence.previous_xy.x,
        correspondence.previous_xy.y,
        1.0,
    );
    let mapped = homography * p1;
    if mapped.z.abs() < 1e-12 {
        return None;
    }
    let dx = correspondence.current_xy.x - mapped.x / mapped.z;
    let dy = correspondence.current_xy.y - mapped.y / mapped.z;
    Some(dx * dx + dy * dy)
}

/// Direct linear transform for the planar homography `x2 ~ H x1`, ported from
/// `HomographyMatrixEstimator::Estimate`
/// (`src/colmap/estimators/solvers/homography_matrix.cc:41-92`). Operates on
/// raw pixel coordinates (no Hartley normalization — see module doc). COLMAP
/// special-cases the exact 4-point system with a partial-pivot LU solve and
/// uses an SVD nullspace for N>4; this port uses the `AᵀA` SVD trick
/// throughout (see [`super::EightPointEssentialMatrixEstimator::estimate`]'s
/// doc comment) so a single code path is numerically correct for N==4 as well
/// as N>4 under nalgebra's thin-SVD semantics.
pub fn estimate_homography_dlt(correspondences: &[TwoViewCorrespondence]) -> Option<Matrix3<f64>> {
    let n = correspondences.len();
    if n < 4 {
        return None;
    }

    let mut a = DMatrix::<f64>::zeros(2 * n, 9);
    for (i, c) in correspondences.iter().enumerate() {
        let (x, y) = (c.previous_xy.x, c.previous_xy.y);
        let (xp, yp) = (c.current_xy.x, c.current_xy.y);
        // Row 2i:   [x y 1  0 0 0  -xp*x -xp*y -xp] · h = 0
        a[(2 * i, 0)] = x;
        a[(2 * i, 1)] = y;
        a[(2 * i, 2)] = 1.0;
        a[(2 * i, 6)] = -xp * x;
        a[(2 * i, 7)] = -xp * y;
        a[(2 * i, 8)] = -xp;
        // Row 2i+1: [0 0 0  x y 1  -yp*x -yp*y -yp] · h = 0
        a[(2 * i + 1, 3)] = x;
        a[(2 * i + 1, 4)] = y;
        a[(2 * i + 1, 5)] = 1.0;
        a[(2 * i + 1, 6)] = -yp * x;
        a[(2 * i + 1, 7)] = -yp * y;
        a[(2 * i + 1, 8)] = -yp;
    }

    let ata = a.transpose() * a;
    let svd = ata.svd(true, true);
    let v_t = svd.v_t?;
    let h = v_t.row(v_t.nrows() - 1);
    let homography = Matrix3::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);
    if homography.determinant().abs() < 1e-8 {
        return None;
    }
    Some(homography)
}

/// RANSAC configuration for [`homography_ransac`]. Default `max_error_px`
/// mirrors COLMAP's `TwoViewGeometryOptions::TwoViewGeometryOptions()`
/// (`ransac_options.max_error = 4.0`,
/// `src/colmap/estimators/two_view_geometry.h:124`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomographyRansacConfig {
    pub iterations: usize,
    pub max_error_px: f64,
    pub seed: u64,
}

impl Default for HomographyRansacConfig {
    fn default() -> Self {
        Self {
            iterations: 256,
            max_error_px: 4.0,
            seed: 7,
        }
    }
}

/// Output of the homography RANSAC loop.
#[derive(Debug, Clone, PartialEq)]
pub struct HomographyReport {
    pub homography: Matrix3<f64>,
    pub inliers: Vec<usize>,
}

/// LO-RANSAC-shaped (refit-on-inliers) homography estimation: uniform random
/// 4-point samples scored by pixel-space reprojection error, then a final
/// refit on the winning inlier set.
pub fn homography_ransac(
    correspondences: &[TwoViewCorrespondence],
    config: &HomographyRansacConfig,
) -> Option<HomographyReport> {
    const SAMPLE_SIZE: usize = 4;
    if correspondences.len() < SAMPLE_SIZE {
        return None;
    }

    let mut rng = SmallRng::seed_from_u64(config.seed);
    let threshold_sq = config.max_error_px * config.max_error_px;
    let indices: Vec<usize> = (0..correspondences.len()).collect();

    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_homography: Option<Matrix3<f64>> = None;

    for _ in 0..config.iterations {
        let mut subset = indices.clone();
        subset.shuffle(&mut rng);
        let sample: Vec<TwoViewCorrespondence> = subset[..SAMPLE_SIZE]
            .iter()
            .map(|&i| correspondences[i])
            .collect();
        let Some(candidate) = estimate_homography_dlt(&sample) else {
            continue;
        };
        let inliers = score_homography_inliers(&candidate, correspondences, threshold_sq);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_homography = Some(candidate);
        }
    }

    let homography = best_homography?;
    if best_inliers.len() < SAMPLE_SIZE {
        return None;
    }

    let inlier_correspondences: Vec<TwoViewCorrespondence> =
        best_inliers.iter().map(|&i| correspondences[i]).collect();
    let refined = estimate_homography_dlt(&inlier_correspondences).unwrap_or(homography);
    let final_inliers = score_homography_inliers(&refined, correspondences, threshold_sq);
    let final_inliers = if final_inliers.len() >= best_inliers.len() {
        final_inliers
    } else {
        best_inliers
    };

    Some(HomographyReport {
        homography: refined,
        inliers: final_inliers,
    })
}

fn score_homography_inliers(
    homography: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    threshold_sq: f64,
) -> Vec<usize> {
    correspondences
        .iter()
        .enumerate()
        .filter_map(|(i, c)| {
            homography_squared_error(homography, c).filter(|&e| e <= threshold_sq)?;
            Some(i)
        })
        .collect()
}

/// One candidate `(rotation, translation, plane_normal)` motion from
/// [`decompose_homography_matrix`]. `translation` and `normal` are defined up
/// to the same unknown plane-distance scale `d` as COLMAP's decomposition
/// (`H = K2 (R − t·nᵀ/d) K1⁻¹`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HomographyMotion {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
    pub normal: Vector3<f64>,
}

fn compute_opposite_of_minor(m: &Matrix3<f64>, row: usize, col: usize) -> f64 {
    let col1 = if col == 0 { 1 } else { 0 };
    let col2 = if col == 2 { 1 } else { 2 };
    let row1 = if row == 0 { 1 } else { 0 };
    let row2 = if row == 2 { 1 } else { 2 };
    m[(row1, col2)] * m[(row2, col1)] - m[(row1, col1)] * m[(row2, col2)]
}

fn sign_of_number(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn compute_homography_rotation(
    h_normalized: &Matrix3<f64>,
    t_star: &Vector3<f64>,
    n: &Vector3<f64>,
    v: f64,
) -> Matrix3<f64> {
    h_normalized * (Matrix3::identity() - (2.0 / v) * t_star * n.transpose())
}

/// Faithful port of COLMAP's `DecomposeHomographyMatrix`
/// (`src/colmap/geometry/homography_matrix.cc:67-188`). Returns the (up to)
/// four candidate motions that explain `H` up to scale, or a single
/// zero-translation candidate when `H` is (numerically) a pure rotation
/// homography (`tvg.cc:97-104` — this is the algebraic signature this module
/// exists to detect: [`super::colmap_verification`]'s PANORAMIC branch relies
/// on exactly this short-circuit reporting zero translation).
pub fn decompose_homography_matrix(
    homography: &Matrix3<f64>,
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
) -> Vec<HomographyMotion> {
    let Some(k2_inv) = k2.try_inverse() else {
        return Vec::new();
    };

    // Remove calibration from the homography.
    let mut h_normalized = k2_inv * homography * k1;

    // Remove scale: divide by the middle (median) singular value.
    let svd = h_normalized.svd(false, false);
    let sigma_mid = svd.singular_values[1];
    if sigma_mid.abs() < 1e-12 {
        return Vec::new();
    }
    h_normalized /= sigma_mid;

    // Ensure a rotation (not a reflection) comes out below; see the cited
    // source for the determinant-sign argument (Sylvester's identity).
    if h_normalized.determinant() < 0.0 {
        h_normalized = -h_normalized;
    }

    let s = h_normalized.transpose() * h_normalized - Matrix3::identity();

    const MIN_INFINITY_NORM: f64 = 1e-3;
    let infinity_norm = s.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if infinity_norm < MIN_INFINITY_NORM {
        // H is (numerically) a rotation matrix: pure-rotation / no-baseline
        // configuration (COLMAP `tvg.cc:97-104`).
        return vec![HomographyMotion {
            rotation: h_normalized,
            translation: Vector3::zeros(),
            normal: Vector3::zeros(),
        }];
    }

    let m00 = compute_opposite_of_minor(&s, 0, 0);
    let m11 = compute_opposite_of_minor(&s, 1, 1);
    let m22 = compute_opposite_of_minor(&s, 2, 2);

    let rt_m00 = m00.max(0.0).sqrt();
    let rt_m11 = m11.max(0.0).sqrt();
    let rt_m22 = m22.max(0.0).sqrt();

    let m01 = compute_opposite_of_minor(&s, 0, 1);
    let m12 = compute_opposite_of_minor(&s, 1, 2);
    let m02 = compute_opposite_of_minor(&s, 0, 2);

    let e12 = sign_of_number(m12);
    let e02 = sign_of_number(m02);
    let e01 = sign_of_number(m01);

    let n_s = [s[(0, 0)].abs(), s[(1, 1)].abs(), s[(2, 2)].abs()];
    let idx = n_s
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let mut np1 = Vector3::zeros();
    let mut np2 = Vector3::zeros();
    match idx {
        0 => {
            np1[0] = s[(0, 0)];
            np2[0] = s[(0, 0)];
            np1[1] = s[(0, 1)] + rt_m22;
            np2[1] = s[(0, 1)] - rt_m22;
            np1[2] = s[(0, 2)] + e12 * rt_m11;
            np2[2] = s[(0, 2)] - e12 * rt_m11;
        }
        1 => {
            np1[0] = s[(0, 1)] + rt_m22;
            np2[0] = s[(0, 1)] - rt_m22;
            np1[1] = s[(1, 1)];
            np2[1] = s[(1, 1)];
            np1[2] = s[(1, 2)] - e02 * rt_m00;
            np2[2] = s[(1, 2)] + e02 * rt_m00;
        }
        _ => {
            np1[0] = s[(0, 2)] + e01 * rt_m11;
            np2[0] = s[(0, 2)] - e01 * rt_m11;
            np1[1] = s[(1, 2)] + rt_m00;
            np2[1] = s[(1, 2)] - rt_m00;
            np1[2] = s[(2, 2)];
            np2[2] = s[(2, 2)];
        }
    }

    let trace_s = s.trace();
    let v = 2.0 * (1.0 + trace_s - m00 - m11 - m22).max(0.0).sqrt();

    let e_sii = sign_of_number(s[(idx, idx)]);
    let r2 = 2.0 + trace_s + v;
    let nt2 = 2.0 + trace_s - v;
    let r = r2.max(0.0).sqrt();
    let n_t = nt2.max(0.0).sqrt();

    let n1 = np1.normalize();
    let n2 = np2.normalize();

    let half_nt = 0.5 * n_t;
    let esii_t_r = e_sii * r;

    let t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
    let t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

    let r1 = compute_homography_rotation(&h_normalized, &t1_star, &n1, v);
    let t1 = r1 * t1_star;

    let r2_rot = compute_homography_rotation(&h_normalized, &t2_star, &n2, v);
    let t2 = r2_rot * t2_star;

    vec![
        HomographyMotion {
            rotation: r1,
            translation: t1,
            normal: -n1,
        },
        HomographyMotion {
            rotation: r1,
            translation: -t1,
            normal: n1,
        },
        HomographyMotion {
            rotation: r2_rot,
            translation: t2,
            normal: -n2,
        },
        HomographyMotion {
            rotation: r2_rot,
            translation: -t2,
            normal: n2,
        },
    ]
}

/// Homogeneous-DLT midpoint triangulation of one correspondence given a
/// candidate `cam2_from_cam1` motion. Reuses the same linear (4-equation,
/// SVD-nullspace) construction as [`super::cheirality_score`]'s inline solve
/// rather than COLMAP's closed-form `TriangulateMidPoint`
/// (`src/colmap/geometry/triangulation.cc`) — numerically different, but
/// classification-equivalent for the cheirality test this function exists
/// for.
fn triangulate_dlt(
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    ray1: &Point2<f64>,
    ray2: &Point2<f64>,
) -> Option<Point3<f64>> {
    let p1 = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p2 = Matrix3x4::zeros();
    p2.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
    p2.fixed_view_mut::<3, 1>(0, 3).copy_from(translation);

    let mut a = DMatrix::<f64>::zeros(4, 4);
    for column in 0..4 {
        a[(0, column)] = ray1.x * p1[(2, column)] - p1[(0, column)];
        a[(1, column)] = ray1.y * p1[(2, column)] - p1[(1, column)];
        a[(2, column)] = ray2.x * p2[(2, column)] - p2[(0, column)];
        a[(3, column)] = ray2.y * p2[(2, column)] - p2[(1, column)];
    }
    let svd = a.svd(false, true);
    let v_t = svd.v_t?;
    let solution = v_t.row(v_t.nrows() - 1);
    let w = solution[3];
    if w.abs() < 1e-12 {
        return None;
    }
    Some(Point3::new(
        solution[0] / w,
        solution[1] / w,
        solution[2] / w,
    ))
}

/// Cheirality count + summed pixel reprojection error for one candidate
/// motion, the selection criterion `PoseFromHomographyMatrix` uses
/// (`CheckCheiralityAndReprojErrorSum`,
/// `src/colmap/geometry/homography_matrix.cc:192-217`). This port sums
/// squared pixel reprojection error rather than COLMAP's angular bearing
/// residual (`1 − cos θ` per ray) — both are "how well does this motion
/// explain the triangulated point" scores that agree on which candidate wins;
/// pixel error is reused here because [`Camera::project`] is the repo's
/// existing reprojection primitive.
fn cheirality_and_reproj_sum(
    motion: &HomographyMotion,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> (usize, f64) {
    let mut count = 0usize;
    let mut sum = 0.0;
    for c in correspondences {
        let Some(ray1) = camera.normalize_pixel(&c.previous_xy) else {
            continue;
        };
        let Some(ray2) = camera.normalize_pixel(&c.current_xy) else {
            continue;
        };
        let Some(point1) = triangulate_dlt(&motion.rotation, &motion.translation, &ray1, &ray2)
        else {
            continue;
        };
        if point1.z <= 0.0 {
            continue;
        }
        let point2_coords = motion.rotation * point1.coords + motion.translation;
        if point2_coords.z <= 0.0 {
            continue;
        }
        count += 1;
        let point2 = Point3::from(point2_coords);
        if let (Some(p1_proj), Some(p2_proj)) = (camera.project(&point1), camera.project(&point2)) {
            let d1 = p1_proj - c.previous_xy;
            let d2 = p2_proj - c.current_xy;
            sum += d1.norm_squared() + d2.norm_squared();
        }
    }
    (count, sum)
}

/// Port of `PoseFromHomographyMatrix`
/// (`src/colmap/geometry/homography_matrix.cc:221-254`): decompose `H` and
/// pick the candidate motion with (a) the most cheirality-passing
/// triangulated points, tie-broken by (b) lowest summed reprojection error.
/// Returns `None` only when [`decompose_homography_matrix`] itself yields no
/// candidate (degenerate `H` or non-invertible `k2`) — see the "still commits
/// to a candidate" note below for why an all-zero-count result is not treated
/// as failure.
pub fn pose_from_homography_matrix(
    homography: &Matrix3<f64>,
    k1: &Matrix3<f64>,
    k2: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<(Matrix3<f64>, Vector3<f64>, Vector3<f64>)> {
    let motions = decompose_homography_matrix(homography, k1, k2);
    if motions.is_empty() {
        return None;
    }

    let mut best: Option<(usize, usize, f64)> = None;
    for (i, motion) in motions.iter().enumerate() {
        let (count, sum) = cheirality_and_reproj_sum(motion, correspondences, camera);
        let is_better = match best {
            None => true,
            Some((_, best_count, best_sum)) => {
                count > best_count || (count == best_count && sum < best_sum)
            }
        };
        if is_better {
            best = Some((i, count, sum));
        }
    }

    // Note: unlike a "reject if nothing triangulated" gate, COLMAP's own
    // selection loop (`PoseFromHomographyMatrix`,
    // `src/colmap/geometry/homography_matrix.cc:239-253`) still commits to a
    // candidate even when `count == 0` for every motion: its tie-break
    // compares `tentative_points3D.size() > points3D->size()` (both zero)
    // OR-else `reproj_residual_sum < best_reproj_residual_sum`, and an empty
    // sum (`0.0`) is trivially less than the initial `DBL_MAX`, so the first
    // candidate always wins that comparison. This matters specifically for
    // the pure-rotation short-circuit in [`decompose_homography_matrix`]
    // (`tvg.cc:97-104`): with zero translation, every ray pair shares a
    // camera centre and has no unique triangulation, so `count` is
    // legitimately `0` there — yet COLMAP still reports the (correct,
    // zero-translation) candidate rather than failing. Mirror that: only
    // `motions.is_empty()` (checked above) is a hard failure.
    let (idx, _count, _) = best?;
    let motion = &motions[idx];
    Some((motion.rotation, motion.translation, motion.normal))
}
