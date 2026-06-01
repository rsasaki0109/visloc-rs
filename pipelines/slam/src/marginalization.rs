//! Schur-complement marginalization of a Gaussian information system.
//!
//! Fixed-lag smoothing and sliding-window estimation keep cost bounded by
//! *dropping* old variables — but a dropped variable still carried information
//! about the ones it touched, and discarding that information would make the
//! window over-confident and inconsistent. Marginalization preserves it: the
//! retained variables get exactly their marginal of the joint Gaussian, which in
//! *information form* is a Schur complement of the information matrix. Only the
//! marginalized block is inverted (it is small — the states leaving the window),
//! never the full system, so this is the bounded-cost operation a windowed
//! smoother runs every step.
//!
//! Given the information system `(Λ, η)` — `Λ = JᵀΩJ` the information
//! (inverse-covariance) matrix and `η = Λμ` the information vector — partitioned
//! into kept (`k`) and marginalized (`m`) variables,
//!
//! ```text
//!   Λ = [ Λ_kk  Λ_km ]      η = [ η_k ]
//!       [ Λ_mk  Λ_mm ]          [ η_m ]
//! ```
//!
//! the marginal over the kept block is the Gaussian with
//!
//! ```text
//!   Λ' = Λ_kk − Λ_km Λ_mm⁻¹ Λ_mk      (the Schur complement Λ/Λ_mm)
//!   η' = η_k  − Λ_km Λ_mm⁻¹ η_m
//! ```
//!
//! `(Λ', η')` is then re-added to the window as a dense Gaussian *prior* on the
//! Markov blanket of the marginalized states. The fundamental identity (a test
//! asserts it) is `Λ'⁻¹ == (Λ⁻¹)_kk`: marginalize-then-invert equals
//! invert-then-extract, i.e. `Λ'` carries precisely the kept block's marginal
//! covariance — no more (would be over-confident), no less (would discard
//! information).
//!
//! Pure linear algebra on a dense [`nalgebra::DMatrix`], no [`crate::PoseGraph`]
//! dependency, mirroring [`crate::covariance`] / [`crate::gnc`] / [`crate::pcm`].

use std::collections::HashSet;

use nalgebra::{DMatrix, DVector};

/// Marginalize every variable *not* in `keep` out of the information system
/// `(lambda, eta)`, returning the reduced `(Λ', η')` over the kept block, with
/// rows/columns ordered exactly as `keep`.
///
/// `keep` lists the retained dimension indices into `lambda` (need not be
/// sorted; duplicates are ignored). The complement is marginalized via the Schur
/// complement `Λ/Λ_mm`. Returns `None` if `lambda`/`eta` shapes disagree, a
/// `keep` index is out of range, or the marginalized block `Λ_mm` is not
/// positive-definite (an under-constrained variable cannot be marginalized — its
/// information is rank-deficient). Marginalizing nothing (`keep` covers every
/// index) returns the system unchanged, reordered as `keep`.
pub fn marginalize(
    lambda: &DMatrix<f64>,
    eta: &DVector<f64>,
    keep: &[usize],
) -> Option<(DMatrix<f64>, DVector<f64>)> {
    let n = lambda.nrows();
    if lambda.ncols() != n || eta.len() != n {
        return None;
    }
    // Deduplicate `keep` while preserving its order; validate ranges.
    let mut seen = HashSet::new();
    let mut keep_ix: Vec<usize> = Vec::with_capacity(keep.len());
    for &i in keep {
        if i >= n {
            return None;
        }
        if seen.insert(i) {
            keep_ix.push(i);
        }
    }
    let marg_ix: Vec<usize> = (0..n).filter(|i| !seen.contains(i)).collect();

    let lambda_kk = submatrix(lambda, &keep_ix, &keep_ix);
    let eta_k = subvector(eta, &keep_ix);

    if marg_ix.is_empty() {
        return Some((lambda_kk, eta_k));
    }

    let lambda_km = submatrix(lambda, &keep_ix, &marg_ix);
    let lambda_mm = submatrix(lambda, &marg_ix, &marg_ix);
    let eta_m = subvector(eta, &marg_ix);

    // Λ_mk = Λ_km^T (the system is symmetric). Invert Λ_mm only — the small,
    // leaving block — via Cholesky; `None` if it is not SPD.
    let chol = lambda_mm.cholesky()?;
    // W = Λ_mm⁻¹ Λ_mk = Λ_mm⁻¹ Λ_km^T  (m×k); V = Λ_mm⁻¹ η_m  (m).
    let w = chol.solve(&lambda_km.transpose());
    let v = chol.solve(&eta_m);

    let lambda_prime = &lambda_kk - &lambda_km * w;
    let eta_prime = &eta_k - &lambda_km * v;
    // Symmetrize against round-off so downstream Cholesky/consumers see an exact
    // symmetric prior (the Schur complement is symmetric in exact arithmetic).
    let lambda_prime = 0.5 * (&lambda_prime + lambda_prime.transpose());
    Some((lambda_prime, eta_prime))
}

/// Gather the `rows × cols` submatrix of `m` at the given index lists.
fn submatrix(m: &DMatrix<f64>, rows: &[usize], cols: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(rows.len(), cols.len(), |r, c| m[(rows[r], cols[c])])
}

/// Gather the subvector of `v` at the given index list.
fn subvector(v: &DVector<f64>, idx: &[usize]) -> DVector<f64> {
    DVector::from_fn(idx.len(), |r, _| v[idx[r]])
}

#[cfg(test)]
mod tests {
    use super::*;

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

    /// The defining property: the marginal information's inverse equals the kept
    /// block of the full covariance. Marginalize-then-invert == invert-then-
    /// extract — `Λ'` carries exactly the kept block's marginal covariance.
    #[test]
    fn marginal_information_inverse_equals_the_covariance_block() {
        let m = DMatrix::from_fn(6, 6, |i, j| ((i * 5 + j * 3) % 7) as f64 * 0.1 - 0.3);
        let lambda = spd_from(&m);
        let eta = DVector::from_fn(6, |i, _| (i as f64) - 2.5);
        let keep = [0usize, 2, 5];

        let (lambda_prime, _eta_prime) = marginalize(&lambda, &eta, &keep).unwrap();
        // Reference: full covariance, then pull the kept block.
        let full_cov = lambda.clone().cholesky().unwrap().inverse();
        let cov_block =
            DMatrix::from_fn(keep.len(), keep.len(), |r, c| full_cov[(keep[r], keep[c])]);
        let marg_cov = lambda_prime.clone().cholesky().unwrap().inverse();
        assert!(
            max_abs_diff(&marg_cov, &cov_block) < 1e-9,
            "Λ'⁻¹ must equal the kept block of Λ⁻¹: err {}",
            max_abs_diff(&marg_cov, &cov_block)
        );
        // The Schur complement is symmetric.
        assert!(max_abs_diff(&lambda_prime, &lambda_prime.transpose()) < 1e-12);
    }

    /// The marginal mean equals the kept components of the full mean:
    /// `Λ'⁻¹ η' == (Λ⁻¹ η)_keep`.
    #[test]
    fn marginal_mean_matches_the_full_mean_on_the_kept_block() {
        let m = DMatrix::from_fn(5, 5, |i, j| ((i * 3 + j * 7) % 5) as f64 * 0.2 - 0.4);
        let lambda = spd_from(&m);
        let eta = DVector::from_fn(5, |i, _| 0.5 * i as f64 - 1.0);
        let keep = [1usize, 3, 4];

        let (lambda_prime, eta_prime) = marginalize(&lambda, &eta, &keep).unwrap();
        let full_mean = lambda.clone().cholesky().unwrap().solve(&eta);
        let marg_mean = lambda_prime.cholesky().unwrap().solve(&eta_prime);
        for (r, &i) in keep.iter().enumerate() {
            assert!(
                (marg_mean[r] - full_mean[i]).abs() < 1e-9,
                "marginal mean[{r}] != full mean[{i}]: {} vs {}",
                marg_mean[r],
                full_mean[i]
            );
        }
    }

    /// Keeping every variable is a (reordering-only) no-op; a non-SPD
    /// marginalized block and out-of-range indices are rejected.
    #[test]
    fn keep_all_is_identity_and_bad_inputs_are_rejected() {
        let lambda = spd_from(&DMatrix::from_fn(3, 3, |i, j| (i + j) as f64 * 0.1));
        let eta = DVector::from_fn(3, |i, _| i as f64);
        let (lp, ep) = marginalize(&lambda, &eta, &[0, 1, 2]).unwrap();
        assert!(max_abs_diff(&lp, &lambda) < 1e-12);
        assert!((ep - &eta).norm() < 1e-12);

        // Out-of-range keep index.
        assert!(marginalize(&lambda, &eta, &[0, 9]).is_none());

        // A marginalized block that is not positive-definite: build a Λ whose
        // lone marginalized variable has zero information (rank-deficient).
        let mut singular = DMatrix::<f64>::identity(3, 3);
        singular[(2, 2)] = 0.0; // variable 2 unconstrained
        assert!(marginalize(&singular, &eta, &[0, 1]).is_none());
    }
}
