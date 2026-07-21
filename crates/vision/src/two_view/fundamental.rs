//! Fundamental-matrix estimation for COLMAP-style two-view verification.
//!
//! Ported from COLMAP's `FundamentalMatrixEightPointEstimator`
//! (`src/colmap/estimators/solvers/fundamental_matrix.cc:124-184`,
//! BSD-3-Clause, ETH Zurich / UNC Chapel Hill): a Hartley-normalized 8-point
//! solve in **pixel space**. Unlike the essential matrix (which needs known
//! intrinsics to turn pixels into normalized bearing rays, see
//! [`super::EightPointEssentialMatrixEstimator`]), the fundamental matrix is
//! defined directly on uncalibrated pixel correspondences — that is the whole
//! point of estimating both: comparing how well each explains the same
//! matches is exactly COLMAP's `E_F_inlier_ratio` / `H_F_inlier_ratio` tests in
//! [`super::colmap_verification`].
//!
//! The estimator differs from [`super::EightPointEssentialMatrixEstimator`] in
//! one crucial step: after the linear 8-point solve, COLMAP zeroes only the
//! *smallest* singular value (a generic rank-2 projection). The essential
//! matrix additionally forces its two non-zero singular values to be equal —
//! a constraint that only holds when the pixel coordinates have first been
//! calibrated into normalized bearings. Applying the essential matrix's
//! equal-singular-value constraint to raw pixel data would silently corrupt
//! the fundamental matrix, so this module keeps its own SVD-based rank-2
//! projection (`solvers/fundamental_matrix.cc:165-176`).

use nalgebra::{DMatrix, Matrix3, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use super::{hartley_normalization, TwoViewCorrespondence};

/// Hartley-normalized linear 8-point solve for the fundamental matrix, port of
/// `FundamentalMatrixEightPointEstimator::Estimate`
/// (`src/colmap/estimators/solvers/fundamental_matrix.cc:124-176`). Requires
/// at least 8 correspondences; uses the `AᵀA` trick (see
/// [`super::EightPointEssentialMatrixEstimator::estimate`]'s doc comment) so
/// the exactly-8-point case does not lose the nullspace direction to
/// nalgebra's thin SVD.
pub fn estimate_fundamental_dlt(correspondences: &[TwoViewCorrespondence]) -> Option<Matrix3<f64>> {
    let n = correspondences.len();
    if n < 8 {
        return None;
    }

    let (t1, pts1) = hartley_normalization(correspondences.iter().map(|c| c.previous_xy))?;
    let (t2, pts2) = hartley_normalization(correspondences.iter().map(|c| c.current_xy))?;

    // Homogeneous linear system x2ᵀ F x1 = 0, one row per correspondence.
    let mut a = DMatrix::<f64>::zeros(n, 9);
    for i in 0..n {
        let (x, y) = (pts1[i].x, pts1[i].y);
        let (xp, yp) = (pts2[i].x, pts2[i].y);
        a[(i, 0)] = xp * x;
        a[(i, 1)] = xp * y;
        a[(i, 2)] = xp;
        a[(i, 3)] = yp * x;
        a[(i, 4)] = yp * y;
        a[(i, 5)] = yp;
        a[(i, 6)] = x;
        a[(i, 7)] = y;
        a[(i, 8)] = 1.0;
    }

    let ata = a.transpose() * a;
    let svd = ata.svd(true, true);
    let v_t = svd.v_t?;
    let last = v_t.row(v_t.nrows() - 1);
    let f_normalized = Matrix3::new(
        last[0], last[1], last[2], last[3], last[4], last[5], last[6], last[7], last[8],
    );

    // Rank-2 projection: zero the smallest singular value only (no equal-
    // singular-value constraint — see module doc).
    let f_svd = f_normalized.svd(true, true);
    let u = f_svd.u?;
    let v_t2 = f_svd.v_t?;
    let mut singular_values = f_svd.singular_values;
    singular_values[2] = 0.0;
    let f_rank2 = u * Matrix3::from_diagonal(&singular_values) * v_t2;

    Some(t2.transpose() * f_rank2 * t1)
}

/// Squared Sampson distance for the fundamental matrix in pixel space, the
/// same closed-form error `ComputeSquaredSampsonError` computes in COLMAP
/// (used by `FundamentalMatrixEightPointEstimator::Residuals`,
/// `solvers/fundamental_matrix.cc:178-184`) — identical algebra to
/// [`super::sampson_distance_squared`] but without the camera normalization
/// step, since `F` already operates on raw pixels.
pub fn fundamental_squared_sampson_error(
    fundamental: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
) -> f64 {
    let p1 = Vector3::new(
        correspondence.previous_xy.x,
        correspondence.previous_xy.y,
        1.0,
    );
    let p2 = Vector3::new(
        correspondence.current_xy.x,
        correspondence.current_xy.y,
        1.0,
    );
    let f_p1 = fundamental * p1;
    let ft_p2 = fundamental.transpose() * p2;
    let numerator = p2.dot(&f_p1).powi(2);
    let denominator = f_p1.x.powi(2) + f_p1.y.powi(2) + ft_p2.x.powi(2) + ft_p2.y.powi(2);
    if denominator < 1e-18 {
        f64::INFINITY
    } else {
        numerator / denominator
    }
}

/// RANSAC configuration for [`fundamental_ransac`]. Defaults mirror COLMAP's
/// `TwoViewGeometryOptions::TwoViewGeometryOptions()` (`ransac_options.max_error
/// = 4.0`, `src/colmap/estimators/two_view_geometry.h:124`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FundamentalRansacConfig {
    pub iterations: usize,
    pub max_error_px: f64,
    pub seed: u64,
}

impl Default for FundamentalRansacConfig {
    fn default() -> Self {
        Self {
            iterations: 256,
            max_error_px: 4.0,
            seed: 7,
        }
    }
}

/// Output of the fundamental-matrix RANSAC loop.
#[derive(Debug, Clone, PartialEq)]
pub struct FundamentalReport {
    pub fundamental: Matrix3<f64>,
    pub inliers: Vec<usize>,
}

/// LO-RANSAC-shaped (refit-on-inliers) fundamental-matrix estimation: uniform
/// random 8-point samples scored by pixel-space Sampson distance, then a
/// final refit on the winning inlier set (mirroring
/// [`super::EssentialRansac`]'s local-optimization step, which is COLMAP's own
/// `LORANSAC` pattern — `src/colmap/optim/loransac.h`, cited in
/// `docs/colmap_port_plan.md` §1.6).
pub fn fundamental_ransac(
    correspondences: &[TwoViewCorrespondence],
    config: &FundamentalRansacConfig,
) -> Option<FundamentalReport> {
    const SAMPLE_SIZE: usize = 8;
    if correspondences.len() < SAMPLE_SIZE {
        return None;
    }

    let mut rng = SmallRng::seed_from_u64(config.seed);
    let threshold_sq = config.max_error_px * config.max_error_px;
    let indices: Vec<usize> = (0..correspondences.len()).collect();

    let mut best_inliers: Vec<usize> = Vec::new();
    let mut best_fundamental: Option<Matrix3<f64>> = None;

    for _ in 0..config.iterations {
        let mut subset = indices.clone();
        subset.shuffle(&mut rng);
        let sample: Vec<TwoViewCorrespondence> = subset[..SAMPLE_SIZE]
            .iter()
            .map(|&i| correspondences[i])
            .collect();
        let Some(candidate) = estimate_fundamental_dlt(&sample) else {
            continue;
        };
        let inliers = score_fundamental_inliers(&candidate, correspondences, threshold_sq);
        if inliers.len() > best_inliers.len() {
            best_inliers = inliers;
            best_fundamental = Some(candidate);
        }
    }

    let fundamental = best_fundamental?;
    if best_inliers.len() < SAMPLE_SIZE {
        return None;
    }

    let inlier_correspondences: Vec<TwoViewCorrespondence> =
        best_inliers.iter().map(|&i| correspondences[i]).collect();
    let refined = estimate_fundamental_dlt(&inlier_correspondences).unwrap_or(fundamental);
    let final_inliers = score_fundamental_inliers(&refined, correspondences, threshold_sq);
    let final_inliers = if final_inliers.len() >= best_inliers.len() {
        final_inliers
    } else {
        best_inliers
    };

    Some(FundamentalReport {
        fundamental: refined,
        inliers: final_inliers,
    })
}

fn score_fundamental_inliers(
    fundamental: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    threshold_sq: f64,
) -> Vec<usize> {
    correspondences
        .iter()
        .enumerate()
        .filter(|(_, c)| fundamental_squared_sampson_error(fundamental, c) <= threshold_sq)
        .map(|(i, _)| i)
        .collect()
}
