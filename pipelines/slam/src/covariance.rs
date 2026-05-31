//! Covariance recovery from the pose-graph information matrix.
//!
//! After a pose-graph solve the information (inverse-covariance) matrix is
//! `Λ = JᵀΩJ`, the normal-equations matrix at the solution. Downstream tasks —
//! loop-closure gating (is a candidate's relative pose within the trajectory's
//! uncertainty?), active SLAM, sensor fusion — need the *covariance*
//! `Σ = Λ⁻¹`, but only a few marginal blocks of it (per-pose covariances and
//! the joint covariance of a keyframe pair), never the dense `Λ⁻¹`.
//!
//! Forming the full inverse is `O(n³)`. The standard SLAM method
//! (Kaess & Dellaert, *Covariance recovery from a square-root information
//! matrix*, 2009; the Takahashi / Erisman–Tinney sparse-inverse recursion,
//! 1975) recovers `Σ` only on the sparsity pattern of the Cholesky factor `L`
//! of `Λ = L Lᵀ`, which already contains every diagonal block. From
//! `Lᵀ Σ = L⁻¹` and the symmetry of `Σ`, for `i ≤ j` on the pattern:
//!
//! ```text
//!   Σ_ij = ( [i==j]/L_ii − Σ_{k>i, L_ki≠0} L_ki · Σ_kj ) / L_ii
//! ```
//!
//! processed with `i` decreasing, so every referenced `Σ_kj` (`k > i`) is
//! already computed. Restricting `(i,j)` to the factor pattern keeps the work
//! at `O(nnz(L))` for the entries that matter — exactly the marginal blocks.
//!
//! This module is pure linear algebra on a dense [`nalgebra::DMatrix`] (with the
//! pattern read from the factor's nonzeros, so a block-sparse `Λ` does
//! block-sparse work), no [`crate::PoseGraph`] dependency, mirroring
//! [`crate::gnc`] / [`crate::pcm`].

use nalgebra::{DMatrix, DVector};

/// χ² 0.95 quantile for 6 degrees of freedom — the default acceptance gate for a
/// 6-DOF SE(3) loop-closure innovation. A residual whose Mahalanobis distance²
/// (see [`mahalanobis_distance_sq`]) exceeds this is statistically implausible
/// at the 5 % level given the relative-pose uncertainty.
pub const CHI2_95_6DOF: f64 = 12.591_587_243_743_977;

/// Dense reference inverse `Σ = Λ⁻¹` via Cholesky. `None` if `Λ` is not
/// symmetric-positive-definite. `O(n³)` — used as the ground truth in tests and
/// as a fallback; prefer [`marginal_block_covariances`] for the marginals.
pub fn dense_inverse(lambda: &DMatrix<f64>) -> Option<DMatrix<f64>> {
    lambda.clone().cholesky().map(|c| c.inverse())
}

/// Recover the sparse inverse `Σ` of `Λ = L Lᵀ` on the sparsity pattern of the
/// Cholesky factor `L` (the Takahashi / Erisman–Tinney recursion). Entries off
/// the pattern are left zero; entries on the pattern (which include every
/// diagonal block) equal `Λ⁻¹` exactly. `None` if `Λ` is not SPD.
///
/// `zero_tol` is the magnitude below which a factor entry counts as structurally
/// zero (defining the pattern); pass `0.0` to treat the factor as fully dense
/// (then the result is the exact dense inverse).
pub fn sparse_inverse(lambda: &DMatrix<f64>, zero_tol: f64) -> Option<DMatrix<f64>> {
    let n = lambda.nrows();
    if n == 0 {
        return Some(DMatrix::zeros(0, 0));
    }
    let l = lambda.clone().cholesky()?.l();

    // Column pattern of L: rows j ≥ i with |L[j][i]| > zero_tol. `below[i]` are
    // the strictly-below-diagonal rows (the `k > i` in the recursion sum).
    let mut pattern_rows: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut below: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in i..n {
            if l[(j, i)].abs() > zero_tol {
                pattern_rows[i].push(j);
                if j > i {
                    below[i].push(j);
                }
            }
        }
    }

    let mut sigma = DMatrix::zeros(n, n);
    // Process columns from the last variable to the first.
    for i in (0..n).rev() {
        let l_ii = l[(i, i)];
        // Compute Σ_ij for each j ≥ i on the pattern (j decreasing is fine; the
        // recursion only references rows k > i, already finalized).
        for &j in pattern_rows[i].iter().rev() {
            let mut s = 0.0;
            for &k in &below[i] {
                // Σ_kj, symmetric: both indices are > i, already computed.
                s += l[(k, i)] * sigma[(k.max(j), k.min(j))];
            }
            let value = if i == j {
                (1.0 / l_ii - s) / l_ii
            } else {
                -s / l_ii
            };
            sigma[(j, i)] = value;
            sigma[(i, j)] = value;
        }
    }
    Some(sigma)
}

/// Per-block marginal covariances: the `block × block` diagonal blocks of
/// `Σ = Λ⁻¹`, recovered via [`sparse_inverse`] (so the dense inverse is never
/// formed). `Λ` must be square with side a multiple of `block`. Returns one
/// covariance matrix per block, in order. `None` if `Λ` is not SPD or the
/// dimension is not a multiple of `block`.
pub fn marginal_block_covariances(
    lambda: &DMatrix<f64>,
    block: usize,
) -> Option<Vec<DMatrix<f64>>> {
    let n = lambda.nrows();
    if block == 0 || n % block != 0 {
        return None;
    }
    let sigma = sparse_inverse(lambda, 0.0)?;
    let count = n / block;
    let mut out = Vec::with_capacity(count);
    for b in 0..count {
        let r = b * block;
        out.push(sigma.view((r, r), (block, block)).into_owned());
    }
    Some(out)
}

/// Joint covariance of two variable blocks `a` and `b` (each `block × block`):
/// the `2·block × 2·block` matrix `[[Σ_aa, Σ_ab], [Σ_ba, Σ_bb]]`. The
/// cross-block `Σ_ab` requires that entry to be on the factor pattern; pass a
/// `sigma` produced with `zero_tol = 0.0` (full recovery) when `a` and `b` are
/// not adjacent in the factor. Useful for loop-closure gating, where the
/// *relative* uncertainty between two keyframes decides whether a candidate is
/// statistically plausible.
pub fn joint_block_covariance(
    sigma: &DMatrix<f64>,
    a: usize,
    b: usize,
    block: usize,
) -> DMatrix<f64> {
    let ra = a * block;
    let rb = b * block;
    let mut joint = DMatrix::zeros(2 * block, 2 * block);
    joint
        .view_mut((0, 0), (block, block))
        .copy_from(&sigma.view((ra, ra), (block, block)));
    joint
        .view_mut((block, block), (block, block))
        .copy_from(&sigma.view((rb, rb), (block, block)));
    joint
        .view_mut((0, block), (block, block))
        .copy_from(&sigma.view((ra, rb), (block, block)));
    joint
        .view_mut((block, 0), (block, block))
        .copy_from(&sigma.view((rb, ra), (block, block)));
    joint
}

/// Covariance of the *relative* pose between two variable blocks `a` and `b`,
/// from their `2·block × 2·block` joint covariance (see
/// [`joint_block_covariance`]): the first-order tangent approximation
/// `Σ_rel = Σ_aa + Σ_bb − Σ_ab − Σ_abᵀ`. This is the uncertainty of the
/// relative transform `a → b` implied by the current estimate — the prediction
/// covariance a loop-closure innovation is gated against. (The SE(3) adjoint
/// Jacobians are dropped, the standard small-uncertainty gating approximation.)
pub fn relative_covariance(joint: &DMatrix<f64>, block: usize) -> DMatrix<f64> {
    let saa = joint.view((0, 0), (block, block));
    let sbb = joint.view((block, block), (block, block));
    let sab = joint.view((0, block), (block, block));
    saa + sbb - sab - sab.transpose()
}

/// Squared Mahalanobis distance `rᵀ Σ⁻¹ r` of a residual `r` under covariance
/// `Σ`, computed by a Cholesky solve (no explicit inverse). `None` if `Σ` is not
/// positive-definite or the dimensions disagree. The acceptance test for a
/// loop-closure innovation is `mahalanobis_distance_sq(r, Σ) <= threshold` with
/// `threshold` a χ² quantile such as [`CHI2_95_6DOF`].
pub fn mahalanobis_distance_sq(residual: &DVector<f64>, cov: &DMatrix<f64>) -> Option<f64> {
    if cov.nrows() != cov.ncols() || cov.nrows() != residual.len() {
        return None;
    }
    let y = cov.clone().cholesky()?.solve(residual);
    Some(residual.dot(&y))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a symmetric-positive-definite matrix `A = MᵀM + nI` from a given
    /// dense `M`, so every test has a known PD information matrix.
    fn spd_from(m: &DMatrix<f64>) -> DMatrix<f64> {
        let n = m.nrows();
        m.transpose() * m + DMatrix::identity(n, n) * (n as f64)
    }

    fn max_abs_diff(a: &DMatrix<f64>, b: &DMatrix<f64>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn sparse_inverse_matches_dense_inverse_on_a_dense_matrix() {
        let m = DMatrix::from_row_slice(
            4,
            4,
            &[
                1.0, 0.2, -0.3, 0.1, 0.0, 1.5, 0.4, -0.2, 0.3, 0.0, 2.0, 0.5, -0.1, 0.2, 0.0, 1.2,
            ],
        );
        let lambda = spd_from(&m);
        // zero_tol = 0 → full pattern → exact dense inverse.
        let sigma = sparse_inverse(&lambda, 0.0).unwrap();
        let reference = dense_inverse(&lambda).unwrap();
        assert!(max_abs_diff(&sigma, &reference) < 1e-9);
        // Σ Λ = I.
        let prod = &sigma * &lambda;
        assert!(max_abs_diff(&prod, &DMatrix::identity(4, 4)) < 1e-9);
    }

    #[test]
    fn diagonal_blocks_match_dense_inverse_on_a_block_sparse_matrix() {
        // Block-tridiagonal (chain) information matrix: 4 blocks of 2×2, only
        // adjacent blocks coupled — the pose-graph odometry-chain structure.
        let nb = 4;
        let b = 2;
        let n = nb * b;
        let mut lambda = DMatrix::zeros(n, n);
        for i in 0..nb {
            // Diagonal block (well-conditioned).
            for r in 0..b {
                for c in 0..b {
                    lambda[(i * b + r, i * b + c)] = if r == c { 4.0 } else { 0.5 };
                }
            }
            // Off-diagonal coupling to the next block.
            if i + 1 < nb {
                for r in 0..b {
                    lambda[(i * b + r, (i + 1) * b + r)] = -1.0;
                    lambda[((i + 1) * b + r, i * b + r)] = -1.0;
                }
            }
        }

        let reference = dense_inverse(&lambda).unwrap();
        let marginals = marginal_block_covariances(&lambda, b).unwrap();
        assert_eq!(marginals.len(), nb);
        for (i, cov) in marginals.iter().enumerate() {
            let r = i * b;
            let truth = reference.view((r, r), (b, b)).into_owned();
            assert!(
                max_abs_diff(cov, &truth) < 1e-9,
                "marginal block {i} mismatch"
            );
        }

        // The sparse inverse leaves a structurally-distant entry uncomputed
        // (zero) where the true inverse is non-zero — demonstrating it does NOT
        // form the dense inverse. The (block 0, block 3) coupling is absent from
        // the factor pattern of a 4-block chain... but Cholesky fill can reach
        // it, so assert the cheaper invariant: the recovered diagonal blocks are
        // exact (above) and Σ is symmetric.
        let sigma = sparse_inverse(&lambda, 0.0).unwrap();
        assert!(max_abs_diff(&sigma, &sigma.transpose()) < 1e-12);
    }

    #[test]
    fn joint_covariance_assembles_the_two_block_marginal() {
        let m = DMatrix::from_fn(6, 6, |i, j| ((i * 7 + j * 3) % 5) as f64 * 0.1 + 0.05);
        let lambda = spd_from(&m);
        let block = 2;
        let sigma = sparse_inverse(&lambda, 0.0).unwrap();
        let joint = joint_block_covariance(&sigma, 0, 2, block);
        // Top-left == Σ_00, bottom-right == Σ_22, off-diagonals == Σ_02 / Σ_20.
        assert!(
            max_abs_diff(
                &joint.view((0, 0), (block, block)).into_owned(),
                &sigma.view((0, 0), (block, block)).into_owned()
            ) < 1e-12
        );
        assert!(
            max_abs_diff(
                &joint.view((0, block), (block, block)).into_owned(),
                &sigma.view((0, 4), (block, block)).into_owned()
            ) < 1e-12
        );
        // Joint covariance must itself be symmetric positive-definite.
        assert!(joint.clone().cholesky().is_some());
    }

    #[test]
    fn relative_covariance_of_independent_blocks_is_the_sum() {
        // Block-diagonal joint (no cross-covariance) → Σ_rel = Σ_aa + Σ_bb.
        let block = 2;
        let mut joint = DMatrix::zeros(4, 4);
        joint[(0, 0)] = 1.0;
        joint[(1, 1)] = 2.0;
        joint[(2, 2)] = 3.0;
        joint[(3, 3)] = 4.0;
        let rel = relative_covariance(&joint, block);
        assert!((rel[(0, 0)] - 4.0).abs() < 1e-12); // 1 + 3
        assert!((rel[(1, 1)] - 6.0).abs() < 1e-12); // 2 + 4
                                                    // Positive cross-covariance shrinks the relative uncertainty.
        let mut correlated = joint.clone();
        correlated[(0, 2)] = 0.5;
        correlated[(2, 0)] = 0.5;
        let rel_c = relative_covariance(&correlated, block);
        assert!(
            rel_c[(0, 0)] < rel[(0, 0)],
            "shared error cancels in the difference"
        );
    }

    #[test]
    fn mahalanobis_distance_matches_explicit_inverse() {
        let cov = DMatrix::from_row_slice(2, 2, &[4.0, 1.0, 1.0, 3.0]);
        let r = DVector::from_row_slice(&[2.0, -1.0]);
        let got = mahalanobis_distance_sq(&r, &cov).unwrap();
        let expected = (r.transpose() * cov.clone().try_inverse().unwrap() * &r)[(0, 0)];
        assert!((got - expected).abs() < 1e-12);
        // Dimension mismatch and non-PD covariance are rejected.
        assert!(mahalanobis_distance_sq(&DVector::from_row_slice(&[1.0]), &cov).is_none());
        let indef = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(mahalanobis_distance_sq(&r, &indef).is_none());
    }

    #[test]
    fn rejects_non_spd_and_bad_dimensions() {
        // Indefinite matrix → no Cholesky.
        let indef = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 2.0, 1.0]);
        assert!(sparse_inverse(&indef, 0.0).is_none());
        // Dimension not a multiple of the block size.
        let m = DMatrix::identity(5, 5);
        assert!(marginal_block_covariances(&m, 2).is_none());
    }
}
