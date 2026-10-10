//! Nister five-point essential-matrix solver.
//!
//! This is a faithful Rust port of PoseLib's `relpose_5pt`
//! (`PoseLib/solvers/relpose_5pt.cc`) together with the Sturm-sequence real
//! root isolator it calls (`PoseLib/misc/sturm.h`). COLMAP's
//! `EssentialMatrixFivePointEstimator` delegates its minimal case to exactly
//! that PoseLib routine (`src/colmap/estimators/solvers/essential_matrix.cc`),
//! so this module is what makes the port's initial-pair classification agree
//! with COLMAP's.
//!
//! Reference: D. Nister, "An efficient solution to the five-point relative
//! pose problem", IEEE-T-PAMI 26(6), 2004.

use nalgebra::linalg::SymmetricEigen;
use nalgebra::{DMatrix, Matrix3, Matrix3x2, SMatrix, Vector3};

/// Degree of the Nister determinant polynomial and its Sturm sequence.
const N: usize = 10;

/// Solves the relative pose from `n >= 5` normalized bearing-vector
/// correspondences.
///
/// Returns up to ten essential-matrix hypotheses. The basis of the
/// four-dimensional null space is chosen from the four smallest right singular
/// vectors of the epipolar constraint matrix, which reproduces COLMAP's
/// `rightCols<4>()` null-space extraction for both the minimal (`n == 5`) and
/// over-determined (`n > 5`) cases.
#[allow(clippy::needless_range_loop)]
pub(crate) fn relpose_5pt(rays1: &[Vector3<f64>], rays2: &[Vector3<f64>]) -> Vec<Matrix3<f64>> {
    debug_assert_eq!(rays1.len(), rays2.len());
    let n_points = rays1.len();
    if n_points < 5 {
        return Vec::new();
    }

    // Epipolar constraints: for each pair the 9-vector is
    // (x1.x * x2, x1.y * x2, x1.z * x2) — the column-major flattening of
    // x1 * x2^T. We stack these as the rows of `m_transpose` (n x 9).
    let mut m_transpose = DMatrix::<f64>::zeros(n_points, 9);
    for i in 0..n_points {
        let x1 = rays1[i];
        let x2 = rays2[i];
        for j in 0..3 {
            for i2 in 0..3 {
                // Column-major flattening: index 3*j + i2 holds E(i2, j),
                // whose epipolar coefficient is x1[j] * x2[i2].
                m_transpose[(i, 3 * j + i2)] = x1[j] * x2[i2];
            }
        }
    }

    // Null space: the four eigenvectors of `A^T A` with the smallest
    // eigenvalues (A = the n x 9 constraint matrix). This spans the same
    // four-dimensional subspace as COLMAP / PoseLib's
    // `fullPivHouseholderQr().matrixQ().rightCols(4)`.
    // Fixed-size from here on (no heap per RANSAC trial); same arithmetic.
    let ata: SMatrix<f64, 9, 9> = (m_transpose.transpose() * &m_transpose)
        .fixed_view::<9, 9>(0, 0)
        .into_owned();
    // A degenerate minimal sample (coincident/collinear rays, a failed-to-
    // normalize bearing upstream) can put a NaN/Inf into `ata`.
    // `SymmetricEigen::new` calls the unbounded `try_new(.., max_niter=0)`
    // ("continues indefinitely until convergence" per its own doc), which
    // never converges on such input -- same class of hang as
    // `pnp/gp3p.rs::real_roots_of_degree_le_8`. Bound it and reject the
    // sample (no hypotheses) instead of hanging the RANSAC loop that calls
    // this on every trial.
    if !ata.iter().all(|value| value.is_finite()) {
        return Vec::new();
    }
    let Some(eig) = SymmetricEigen::try_new(ata, f64::EPSILON, 10_000) else {
        return Vec::new();
    };
    let mut order: Vec<usize> = (0..9).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[a]
            .partial_cmp(&eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut n_basis = [[0.0f64; 9]; 4];
    for r in 0..4 {
        let col = order[r];
        for row in 0..9 {
            n_basis[r][row] = eig.eigenvectors[(row, col)];
        }
    }

    let mut coeffs = [[0.0f64; 20]; 10];
    compute_trace_constraints(&n_basis, &mut coeffs);

    // Solve the 10x10 linear subsystem for the leading block.
    let mut a_block = SMatrix::<f64, 10, 10>::zeros();
    let mut b_block = SMatrix::<f64, 10, 10>::zeros();
    for i in 0..10 {
        for j in 0..10 {
            a_block[(i, j)] = coeffs[i][j];
            b_block[(i, j)] = coeffs[i][10 + j];
        }
    }
    let lu = a_block.lu();
    let Some(solution) = lu.solve(&b_block) else {
        return Vec::new();
    };
    for i in 0..10 {
        for j in 0..10 {
            coeffs[i][10 + j] = solution[(i, j)];
        }
    }

    // Eliminations using the six bottom rows (PoseLib `A` 3x13).
    let mut a = [[0.0f64; 13]; 3];
    for i in 0..3 {
        let top = 4 + 2 * i;
        let bot = 5 + 2 * i;
        a[i][0] = -coeffs[bot][10];
        a[i][1] = coeffs[top][10] - coeffs[bot][11];
        a[i][2] = coeffs[top][11] - coeffs[bot][12];
        a[i][3] = coeffs[top][12];
        a[i][4] = -coeffs[bot][13];
        a[i][5] = coeffs[top][13] - coeffs[bot][14];
        a[i][6] = coeffs[top][14] - coeffs[bot][15];
        a[i][7] = coeffs[top][15];
        a[i][8] = -coeffs[bot][16];
        a[i][9] = coeffs[top][16] - coeffs[bot][17];
        a[i][10] = coeffs[top][17] - coeffs[bot][18];
        a[i][11] = coeffs[top][18] - coeffs[bot][19];
        a[i][12] = coeffs[top][19];
    }

    // Tenth-degree determinant polynomial.
    let mut c = [0.0f64; 11];
    c[0] =
        a[0][12] * a[1][3] * a[2][7] - a[0][12] * a[1][7] * a[2][3] - a[0][3] * a[2][7] * a[1][12]
            + a[0][7] * a[2][3] * a[1][12]
            + a[0][3] * a[1][7] * a[2][12]
            - a[0][7] * a[1][3] * a[2][12];
    c[1] = a[0][11] * a[1][3] * a[2][7] - a[0][11] * a[1][7] * a[2][3]
        + a[0][12] * a[1][2] * a[2][7]
        + a[0][12] * a[1][3] * a[2][6]
        - a[0][12] * a[1][6] * a[2][3]
        - a[0][12] * a[1][7] * a[2][2]
        - a[0][2] * a[2][7] * a[1][12]
        - a[0][3] * a[2][6] * a[1][12]
        - a[0][3] * a[2][7] * a[1][11]
        + a[0][6] * a[2][3] * a[1][12]
        + a[0][7] * a[2][2] * a[1][12]
        + a[0][7] * a[2][3] * a[1][11]
        + a[0][2] * a[1][7] * a[2][12]
        + a[0][3] * a[1][6] * a[2][12]
        + a[0][3] * a[1][7] * a[2][11]
        - a[0][6] * a[1][3] * a[2][12]
        - a[0][7] * a[1][2] * a[2][12]
        - a[0][7] * a[1][3] * a[2][11];
    c[2] = a[0][10] * a[1][3] * a[2][7] - a[0][10] * a[1][7] * a[2][3]
        + a[0][11] * a[1][2] * a[2][7]
        + a[0][11] * a[1][3] * a[2][6]
        - a[0][11] * a[1][6] * a[2][3]
        - a[0][11] * a[1][7] * a[2][2]
        + a[1][1] * a[0][12] * a[2][7]
        + a[0][12] * a[1][2] * a[2][6]
        + a[0][12] * a[1][3] * a[2][5]
        - a[0][12] * a[1][5] * a[2][3]
        - a[0][12] * a[1][6] * a[2][2]
        - a[0][12] * a[1][7] * a[2][1]
        - a[0][1] * a[2][7] * a[1][12]
        - a[0][2] * a[2][6] * a[1][12]
        - a[0][2] * a[2][7] * a[1][11]
        - a[0][3] * a[2][5] * a[1][12]
        - a[0][3] * a[2][6] * a[1][11]
        - a[0][3] * a[2][7] * a[1][10]
        + a[0][5] * a[2][3] * a[1][12]
        + a[0][6] * a[2][2] * a[1][12]
        + a[0][6] * a[2][3] * a[1][11]
        + a[0][7] * a[2][1] * a[1][12]
        + a[0][7] * a[2][2] * a[1][11]
        + a[0][7] * a[2][3] * a[1][10]
        + a[0][1] * a[1][7] * a[2][12]
        + a[0][2] * a[1][6] * a[2][12]
        + a[0][2] * a[1][7] * a[2][11]
        + a[0][3] * a[1][5] * a[2][12]
        + a[0][3] * a[1][6] * a[2][11]
        + a[0][3] * a[1][7] * a[2][10]
        - a[0][5] * a[1][3] * a[2][12]
        - a[0][6] * a[1][2] * a[2][12]
        - a[0][6] * a[1][3] * a[2][11]
        - a[0][7] * a[1][1] * a[2][12]
        - a[0][7] * a[1][2] * a[2][11]
        - a[0][7] * a[1][3] * a[2][10];
    c[3] = a[0][3] * a[1][7] * a[2][9] - a[0][3] * a[1][9] * a[2][7] - a[0][7] * a[1][3] * a[2][9]
        + a[0][7] * a[1][9] * a[2][3]
        + a[0][9] * a[1][3] * a[2][7]
        - a[0][9] * a[1][7] * a[2][3]
        + a[0][10] * a[1][2] * a[2][7]
        + a[0][10] * a[1][3] * a[2][6]
        - a[0][10] * a[1][6] * a[2][3]
        - a[0][10] * a[1][7] * a[2][2]
        + a[1][0] * a[0][12] * a[2][7]
        + a[0][11] * a[1][1] * a[2][7]
        + a[0][11] * a[1][2] * a[2][6]
        + a[0][11] * a[1][3] * a[2][5]
        - a[0][11] * a[1][5] * a[2][3]
        - a[0][11] * a[1][6] * a[2][2]
        - a[0][11] * a[1][7] * a[2][1]
        + a[1][1] * a[0][12] * a[2][6]
        + a[0][12] * a[1][2] * a[2][5]
        + a[0][12] * a[1][3] * a[2][4]
        - a[0][12] * a[1][4] * a[2][3]
        - a[0][12] * a[1][5] * a[2][2]
        - a[0][12] * a[1][6] * a[2][1]
        - a[0][12] * a[1][7] * a[2][0]
        - a[0][0] * a[2][7] * a[1][12]
        - a[0][1] * a[2][6] * a[1][12]
        - a[0][1] * a[2][7] * a[1][11]
        - a[0][2] * a[2][5] * a[1][12]
        - a[0][2] * a[2][6] * a[1][11]
        - a[0][2] * a[2][7] * a[1][10]
        - a[0][3] * a[2][4] * a[1][12]
        - a[0][3] * a[2][5] * a[1][11]
        - a[0][3] * a[2][6] * a[1][10]
        + a[0][4] * a[2][3] * a[1][12]
        + a[0][5] * a[2][2] * a[1][12]
        + a[0][5] * a[2][3] * a[1][11]
        + a[0][6] * a[2][1] * a[1][12]
        + a[0][6] * a[2][2] * a[1][11]
        + a[0][6] * a[2][3] * a[1][10]
        + a[0][7] * a[2][0] * a[1][12]
        + a[0][7] * a[2][1] * a[1][11]
        + a[0][7] * a[2][2] * a[1][10]
        + a[0][0] * a[1][7] * a[2][12]
        + a[0][1] * a[1][6] * a[2][12]
        + a[0][1] * a[1][7] * a[2][11]
        + a[0][2] * a[1][5] * a[2][12]
        + a[0][2] * a[1][6] * a[2][11]
        + a[0][2] * a[1][7] * a[2][10]
        + a[0][3] * a[1][4] * a[2][12]
        + a[0][3] * a[1][5] * a[2][11]
        + a[0][3] * a[1][6] * a[2][10]
        - a[0][4] * a[1][3] * a[2][12]
        - a[0][5] * a[1][2] * a[2][12]
        - a[0][5] * a[1][3] * a[2][11]
        - a[0][6] * a[1][1] * a[2][12]
        - a[0][6] * a[1][2] * a[2][11]
        - a[0][6] * a[1][3] * a[2][10]
        - a[0][7] * a[1][0] * a[2][12]
        - a[0][7] * a[1][1] * a[2][11]
        - a[0][7] * a[1][2] * a[2][10];
    c[4] = a[0][2] * a[1][7] * a[2][9] - a[0][2] * a[1][9] * a[2][7]
        + a[0][3] * a[1][6] * a[2][9]
        + a[0][3] * a[1][7] * a[2][8]
        - a[0][3] * a[1][8] * a[2][7]
        - a[0][3] * a[1][9] * a[2][6]
        - a[0][6] * a[1][3] * a[2][9]
        + a[0][6] * a[1][9] * a[2][3]
        - a[0][7] * a[1][2] * a[2][9]
        - a[0][7] * a[1][3] * a[2][8]
        + a[0][7] * a[1][8] * a[2][3]
        + a[0][7] * a[1][9] * a[2][2]
        + a[0][8] * a[1][3] * a[2][7]
        - a[0][8] * a[1][7] * a[2][3]
        + a[0][9] * a[1][2] * a[2][7]
        + a[0][9] * a[1][3] * a[2][6]
        - a[0][9] * a[1][6] * a[2][3]
        - a[0][9] * a[1][7] * a[2][2]
        + a[0][10] * a[1][1] * a[2][7]
        + a[0][10] * a[1][2] * a[2][6]
        + a[0][10] * a[1][3] * a[2][5]
        - a[0][10] * a[1][5] * a[2][3]
        - a[0][10] * a[1][6] * a[2][2]
        - a[0][10] * a[1][7] * a[2][1]
        + a[1][0] * a[0][11] * a[2][7]
        + a[1][0] * a[0][12] * a[2][6]
        + a[0][11] * a[1][1] * a[2][6]
        + a[0][11] * a[1][2] * a[2][5]
        + a[0][11] * a[1][3] * a[2][4]
        - a[0][11] * a[1][4] * a[2][3]
        - a[0][11] * a[1][5] * a[2][2]
        - a[0][11] * a[1][6] * a[2][1]
        - a[0][11] * a[1][7] * a[2][0]
        + a[1][1] * a[0][12] * a[2][5]
        + a[0][12] * a[1][2] * a[2][4]
        - a[0][12] * a[1][4] * a[2][2]
        - a[0][12] * a[1][5] * a[2][1]
        - a[0][12] * a[1][6] * a[2][0]
        - a[0][0] * a[2][6] * a[1][12]
        - a[0][0] * a[2][7] * a[1][11]
        - a[0][1] * a[2][5] * a[1][12]
        - a[0][1] * a[2][6] * a[1][11]
        - a[0][1] * a[2][7] * a[1][10]
        - a[0][2] * a[2][4] * a[1][12]
        - a[0][2] * a[2][5] * a[1][11]
        - a[0][2] * a[2][6] * a[1][10]
        - a[0][3] * a[2][4] * a[1][11]
        - a[0][3] * a[2][5] * a[1][10]
        + a[0][4] * a[2][2] * a[1][12]
        + a[0][4] * a[2][3] * a[1][11]
        + a[0][5] * a[2][1] * a[1][12]
        + a[0][5] * a[2][2] * a[1][11]
        + a[0][5] * a[2][3] * a[1][10]
        + a[0][6] * a[2][0] * a[1][12]
        + a[0][6] * a[2][1] * a[1][11]
        + a[0][6] * a[2][2] * a[1][10]
        + a[0][7] * a[2][0] * a[1][11]
        + a[0][7] * a[2][1] * a[1][10]
        + a[0][0] * a[1][6] * a[2][12]
        + a[0][0] * a[1][7] * a[2][11]
        + a[0][1] * a[1][5] * a[2][12]
        + a[0][1] * a[1][6] * a[2][11]
        + a[0][1] * a[1][7] * a[2][10]
        + a[0][2] * a[1][4] * a[2][12]
        + a[0][2] * a[1][5] * a[2][11]
        + a[0][2] * a[1][6] * a[2][10]
        + a[0][3] * a[1][4] * a[2][11]
        + a[0][3] * a[1][5] * a[2][10]
        - a[0][4] * a[1][2] * a[2][12]
        - a[0][4] * a[1][3] * a[2][11]
        - a[0][5] * a[1][1] * a[2][12]
        - a[0][5] * a[1][2] * a[2][11]
        - a[0][5] * a[1][3] * a[2][10]
        - a[0][6] * a[1][0] * a[2][12]
        - a[0][6] * a[1][1] * a[2][11]
        - a[0][6] * a[1][2] * a[2][10]
        - a[0][7] * a[1][0] * a[2][11]
        - a[0][7] * a[1][1] * a[2][10];
    c[5] = a[0][1] * a[1][7] * a[2][9] - a[0][1] * a[1][9] * a[2][7]
        + a[0][2] * a[1][6] * a[2][9]
        + a[0][2] * a[1][7] * a[2][8]
        - a[0][2] * a[1][8] * a[2][7]
        - a[0][2] * a[1][9] * a[2][6]
        + a[0][3] * a[1][5] * a[2][9]
        + a[0][3] * a[1][6] * a[2][8]
        - a[0][3] * a[1][8] * a[2][6]
        - a[0][3] * a[1][9] * a[2][5]
        - a[0][5] * a[1][3] * a[2][9]
        + a[0][5] * a[1][9] * a[2][3]
        - a[0][6] * a[1][2] * a[2][9]
        - a[0][6] * a[1][3] * a[2][8]
        + a[0][6] * a[1][8] * a[2][3]
        + a[0][6] * a[1][9] * a[2][2]
        - a[0][7] * a[1][1] * a[2][9]
        - a[0][7] * a[1][2] * a[2][8]
        + a[0][7] * a[1][8] * a[2][2]
        + a[0][7] * a[1][9] * a[2][1]
        + a[0][8] * a[1][2] * a[2][7]
        + a[0][8] * a[1][3] * a[2][6]
        - a[0][8] * a[1][6] * a[2][3]
        - a[0][8] * a[1][7] * a[2][2]
        + a[0][9] * a[1][1] * a[2][7]
        + a[0][9] * a[1][2] * a[2][6]
        + a[0][9] * a[1][3] * a[2][5]
        - a[0][9] * a[1][5] * a[2][3]
        - a[0][9] * a[1][6] * a[2][2]
        - a[0][9] * a[1][7] * a[2][1]
        + a[0][10] * a[1][0] * a[2][7]
        + a[0][10] * a[1][1] * a[2][6]
        + a[0][10] * a[1][2] * a[2][5]
        + a[0][10] * a[1][3] * a[2][4]
        - a[0][10] * a[1][4] * a[2][3]
        - a[0][10] * a[1][5] * a[2][2]
        - a[0][10] * a[1][6] * a[2][1]
        - a[0][10] * a[1][7] * a[2][0]
        + a[1][0] * a[0][11] * a[2][6]
        + a[1][0] * a[0][12] * a[2][5]
        + a[0][11] * a[1][1] * a[2][5]
        + a[0][11] * a[1][2] * a[2][4]
        - a[0][11] * a[1][4] * a[2][2]
        - a[0][11] * a[1][5] * a[2][1]
        - a[0][11] * a[1][6] * a[2][0]
        + a[1][1] * a[0][12] * a[2][4]
        - a[0][12] * a[1][4] * a[2][1]
        - a[0][12] * a[1][5] * a[2][0]
        - a[0][0] * a[2][5] * a[1][12]
        - a[0][0] * a[2][6] * a[1][11]
        - a[0][0] * a[2][7] * a[1][10]
        - a[0][1] * a[2][4] * a[1][12]
        - a[0][1] * a[2][5] * a[1][11]
        - a[0][1] * a[2][6] * a[1][10]
        - a[0][2] * a[2][4] * a[1][11]
        - a[0][2] * a[2][5] * a[1][10]
        - a[0][3] * a[2][4] * a[1][10]
        + a[0][4] * a[2][1] * a[1][12]
        + a[0][4] * a[2][2] * a[1][11]
        + a[0][4] * a[2][3] * a[1][10]
        + a[0][5] * a[2][0] * a[1][12]
        + a[0][5] * a[2][1] * a[1][11]
        + a[0][5] * a[2][2] * a[1][10]
        + a[0][6] * a[2][0] * a[1][11]
        + a[0][6] * a[2][1] * a[1][10]
        + a[0][7] * a[2][0] * a[1][10]
        + a[0][0] * a[1][5] * a[2][12]
        + a[0][0] * a[1][6] * a[2][11]
        + a[0][0] * a[1][7] * a[2][10]
        + a[0][1] * a[1][4] * a[2][12]
        + a[0][1] * a[1][5] * a[2][11]
        + a[0][1] * a[1][6] * a[2][10]
        + a[0][2] * a[1][4] * a[2][11]
        + a[0][2] * a[1][5] * a[2][10]
        + a[0][3] * a[1][4] * a[2][10]
        - a[0][4] * a[1][1] * a[2][12]
        - a[0][4] * a[1][2] * a[2][11]
        - a[0][4] * a[1][3] * a[2][10]
        - a[0][5] * a[1][0] * a[2][12]
        - a[0][5] * a[1][1] * a[2][11]
        - a[0][5] * a[1][2] * a[2][10]
        - a[0][6] * a[1][0] * a[2][11]
        - a[0][6] * a[1][1] * a[2][10]
        - a[0][7] * a[1][0] * a[2][10];
    c[6] = a[0][0] * a[1][7] * a[2][9] - a[0][0] * a[1][9] * a[2][7]
        + a[0][1] * a[1][6] * a[2][9]
        + a[0][1] * a[1][7] * a[2][8]
        - a[0][1] * a[1][8] * a[2][7]
        - a[0][1] * a[1][9] * a[2][6]
        + a[0][2] * a[1][5] * a[2][9]
        + a[0][2] * a[1][6] * a[2][8]
        - a[0][2] * a[1][8] * a[2][6]
        - a[0][2] * a[1][9] * a[2][5]
        + a[0][3] * a[1][4] * a[2][9]
        + a[0][3] * a[1][5] * a[2][8]
        - a[0][3] * a[1][8] * a[2][5]
        - a[0][3] * a[1][9] * a[2][4]
        - a[0][4] * a[1][3] * a[2][9]
        + a[0][4] * a[1][9] * a[2][3]
        - a[0][5] * a[1][2] * a[2][9]
        - a[0][5] * a[1][3] * a[2][8]
        + a[0][5] * a[1][8] * a[2][3]
        + a[0][5] * a[1][9] * a[2][2]
        - a[0][6] * a[1][1] * a[2][9]
        - a[0][6] * a[1][2] * a[2][8]
        + a[0][6] * a[1][8] * a[2][2]
        + a[0][6] * a[1][9] * a[2][1]
        - a[0][7] * a[1][0] * a[2][9]
        - a[0][7] * a[1][1] * a[2][8]
        + a[0][7] * a[1][8] * a[2][1]
        + a[0][7] * a[1][9] * a[2][0]
        + a[0][8] * a[1][1] * a[2][7]
        + a[0][8] * a[1][2] * a[2][6]
        + a[0][8] * a[1][3] * a[2][5]
        - a[0][8] * a[1][5] * a[2][3]
        - a[0][8] * a[1][6] * a[2][2]
        - a[0][8] * a[1][7] * a[2][1]
        + a[0][9] * a[1][0] * a[2][7]
        + a[0][9] * a[1][1] * a[2][6]
        + a[0][9] * a[1][2] * a[2][5]
        + a[0][9] * a[1][3] * a[2][4]
        - a[0][9] * a[1][4] * a[2][3]
        - a[0][9] * a[1][5] * a[2][2]
        - a[0][9] * a[1][6] * a[2][1]
        - a[0][9] * a[1][7] * a[2][0]
        + a[0][10] * a[1][0] * a[2][6]
        + a[0][10] * a[1][1] * a[2][5]
        + a[0][10] * a[1][2] * a[2][4]
        - a[0][10] * a[1][4] * a[2][2]
        - a[0][10] * a[1][5] * a[2][1]
        - a[0][10] * a[1][6] * a[2][0]
        + a[1][0] * a[0][11] * a[2][5]
        + a[1][0] * a[0][12] * a[2][4]
        + a[0][11] * a[1][1] * a[2][4]
        - a[0][11] * a[1][4] * a[2][1]
        - a[0][11] * a[1][5] * a[2][0]
        - a[0][12] * a[1][4] * a[2][0]
        - a[0][0] * a[2][4] * a[1][12]
        - a[0][0] * a[2][5] * a[1][11]
        - a[0][0] * a[2][6] * a[1][10]
        - a[0][1] * a[2][4] * a[1][11]
        - a[0][1] * a[2][5] * a[1][10]
        - a[0][2] * a[2][4] * a[1][10]
        + a[0][4] * a[2][0] * a[1][12]
        + a[0][4] * a[2][1] * a[1][11]
        + a[0][4] * a[2][2] * a[1][10]
        + a[0][5] * a[2][0] * a[1][11]
        + a[0][5] * a[2][1] * a[1][10]
        + a[0][6] * a[2][0] * a[1][10]
        + a[0][0] * a[1][4] * a[2][12]
        + a[0][0] * a[1][5] * a[2][11]
        + a[0][0] * a[1][6] * a[2][10]
        + a[0][1] * a[1][4] * a[2][11]
        + a[0][1] * a[1][5] * a[2][10]
        + a[0][2] * a[1][4] * a[2][10]
        - a[0][4] * a[1][0] * a[2][12]
        - a[0][4] * a[1][1] * a[2][11]
        - a[0][4] * a[1][2] * a[2][10]
        - a[0][5] * a[1][0] * a[2][11]
        - a[0][5] * a[1][1] * a[2][10]
        - a[0][6] * a[1][0] * a[2][10];
    c[7] = a[0][0] * a[1][6] * a[2][9] + a[0][0] * a[1][7] * a[2][8]
        - a[0][0] * a[1][8] * a[2][7]
        - a[0][0] * a[1][9] * a[2][6]
        + a[0][1] * a[1][5] * a[2][9]
        + a[0][1] * a[1][6] * a[2][8]
        - a[0][1] * a[1][8] * a[2][6]
        - a[0][1] * a[1][9] * a[2][5]
        + a[0][2] * a[1][4] * a[2][9]
        + a[0][2] * a[1][5] * a[2][8]
        - a[0][2] * a[1][8] * a[2][5]
        - a[0][2] * a[1][9] * a[2][4]
        + a[0][3] * a[1][4] * a[2][8]
        - a[0][3] * a[1][8] * a[2][4]
        - a[0][4] * a[1][2] * a[2][9]
        - a[0][4] * a[1][3] * a[2][8]
        + a[0][4] * a[1][8] * a[2][3]
        + a[0][4] * a[1][9] * a[2][2]
        - a[0][5] * a[1][1] * a[2][9]
        - a[0][5] * a[1][2] * a[2][8]
        + a[0][5] * a[1][8] * a[2][2]
        + a[0][5] * a[1][9] * a[2][1]
        - a[0][6] * a[1][0] * a[2][9]
        - a[0][6] * a[1][1] * a[2][8]
        + a[0][6] * a[1][8] * a[2][1]
        + a[0][6] * a[1][9] * a[2][0]
        - a[0][7] * a[1][0] * a[2][8]
        + a[0][7] * a[1][8] * a[2][0]
        + a[0][8] * a[1][0] * a[2][7]
        + a[0][8] * a[1][1] * a[2][6]
        + a[0][8] * a[1][2] * a[2][5]
        + a[0][8] * a[1][3] * a[2][4]
        - a[0][8] * a[1][4] * a[2][3]
        - a[0][8] * a[1][5] * a[2][2]
        - a[0][8] * a[1][6] * a[2][1]
        - a[0][8] * a[1][7] * a[2][0]
        + a[0][9] * a[1][0] * a[2][6]
        + a[0][9] * a[1][1] * a[2][5]
        + a[0][9] * a[1][2] * a[2][4]
        - a[0][9] * a[1][4] * a[2][2]
        - a[0][9] * a[1][5] * a[2][1]
        - a[0][9] * a[1][6] * a[2][0]
        + a[0][10] * a[1][0] * a[2][5]
        + a[0][10] * a[1][1] * a[2][4]
        - a[0][10] * a[1][4] * a[2][1]
        - a[0][10] * a[1][5] * a[2][0]
        + a[1][0] * a[0][11] * a[2][4]
        - a[0][11] * a[1][4] * a[2][0]
        - a[0][0] * a[2][4] * a[1][11]
        - a[0][0] * a[2][5] * a[1][10]
        - a[0][1] * a[2][4] * a[1][10]
        + a[0][4] * a[2][0] * a[1][11]
        + a[0][4] * a[2][1] * a[1][10]
        + a[0][5] * a[2][0] * a[1][10]
        + a[0][0] * a[1][4] * a[2][11]
        + a[0][0] * a[1][5] * a[2][10]
        + a[0][1] * a[1][4] * a[2][10]
        - a[0][4] * a[1][0] * a[2][11]
        - a[0][4] * a[1][1] * a[2][10]
        - a[0][5] * a[1][0] * a[2][10];
    c[8] = a[0][0] * a[1][5] * a[2][9] + a[0][0] * a[1][6] * a[2][8]
        - a[0][0] * a[1][8] * a[2][6]
        - a[0][0] * a[1][9] * a[2][5]
        + a[0][1] * a[1][4] * a[2][9]
        + a[0][1] * a[1][5] * a[2][8]
        - a[0][1] * a[1][8] * a[2][5]
        - a[0][1] * a[1][9] * a[2][4]
        + a[0][2] * a[1][4] * a[2][8]
        - a[0][2] * a[1][8] * a[2][4]
        - a[0][4] * a[1][1] * a[2][9]
        - a[0][4] * a[1][2] * a[2][8]
        + a[0][4] * a[1][8] * a[2][2]
        + a[0][4] * a[1][9] * a[2][1]
        - a[0][5] * a[1][0] * a[2][9]
        - a[0][5] * a[1][1] * a[2][8]
        + a[0][5] * a[1][8] * a[2][1]
        + a[0][5] * a[1][9] * a[2][0]
        - a[0][6] * a[1][0] * a[2][8]
        + a[0][6] * a[1][8] * a[2][0]
        + a[0][8] * a[1][0] * a[2][6]
        + a[0][8] * a[1][1] * a[2][5]
        + a[0][8] * a[1][2] * a[2][4]
        - a[0][8] * a[1][4] * a[2][2]
        - a[0][8] * a[1][5] * a[2][1]
        - a[0][8] * a[1][6] * a[2][0]
        + a[0][9] * a[1][0] * a[2][5]
        + a[0][9] * a[1][1] * a[2][4]
        - a[0][9] * a[1][4] * a[2][1]
        - a[0][9] * a[1][5] * a[2][0]
        + a[0][10] * a[1][0] * a[2][4]
        - a[0][10] * a[1][4] * a[2][0]
        - a[0][0] * a[2][4] * a[1][10]
        + a[0][4] * a[2][0] * a[1][10]
        + a[0][0] * a[1][4] * a[2][10]
        - a[0][4] * a[1][0] * a[2][10];
    c[9] = a[0][0] * a[1][4] * a[2][9] + a[0][0] * a[1][5] * a[2][8]
        - a[0][0] * a[1][8] * a[2][5]
        - a[0][0] * a[1][9] * a[2][4]
        + a[0][1] * a[1][4] * a[2][8]
        - a[0][1] * a[1][8] * a[2][4]
        - a[0][4] * a[1][0] * a[2][9]
        - a[0][4] * a[1][1] * a[2][8]
        + a[0][4] * a[1][8] * a[2][1]
        + a[0][4] * a[1][9] * a[2][0]
        - a[0][5] * a[1][0] * a[2][8]
        + a[0][5] * a[1][8] * a[2][0]
        + a[0][8] * a[1][0] * a[2][5]
        + a[0][8] * a[1][1] * a[2][4]
        - a[0][8] * a[1][4] * a[2][1]
        - a[0][8] * a[1][5] * a[2][0]
        + a[0][9] * a[1][0] * a[2][4]
        - a[0][9] * a[1][4] * a[2][0];
    c[10] = a[0][0] * a[1][4] * a[2][8] - a[0][0] * a[1][8] * a[2][4] - a[0][4] * a[1][0] * a[2][8]
        + a[0][4] * a[1][8] * a[2][0]
        + a[0][8] * a[1][0] * a[2][4]
        - a[0][8] * a[1][4] * a[2][0];
    let roots = bisect_sturm(&c);

    let mut models = Vec::with_capacity(roots.len());
    for &z in &roots {
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z2 * z2;

        let mut b = Matrix3x2::zeros();
        for row in 0..3 {
            b[(row, 0)] = a[row][0] * z3 + a[row][1] * z2 + a[row][2] * z + a[row][3];
            b[(row, 1)] = a[row][4] * z3 + a[row][5] * z2 + a[row][6] * z + a[row][7];
        }
        let bb = Vector3::new(
            a[0][8] * z4 + a[0][9] * z3 + a[0][10] * z2 + a[0][11] * z + a[0][12],
            a[1][8] * z4 + a[1][9] * z3 + a[1][10] * z2 + a[1][11] * z + a[1][12],
            a[2][8] * z4 + a[2][9] * z3 + a[2][10] * z2 + a[2][11] * z + a[2][12],
        );

        // Solve with the top two rows first; fall back to a full 3x2
        // least-squares solve when the third equation is not satisfied.
        let det = b[(0, 0)] * b[(1, 1)] - b[(0, 1)] * b[(1, 0)];
        let (mut x, mut y) = if det.abs() > 1e-300 {
            let inv = 1.0 / det;
            (
                inv * (b[(1, 1)] * bb[0] - b[(0, 1)] * bb[1]),
                inv * (-b[(1, 0)] * bb[0] + b[(0, 0)] * bb[1]),
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        if !(b[(2, 0)] * x + b[(2, 1)] * y - bb[2]).abs().is_finite()
            || (b[(2, 0)] * x + b[(2, 1)] * y - bb[2]).abs() > 1e-6
        {
            let normal = b.transpose() * b;
            let Some(normal_inv) = normal.try_inverse() else {
                continue;
            };
            let solution = normal_inv * (b.transpose() * bb);
            x = solution[0];
            y = solution[1];
        }

        let (x, y) = (-x, -y);
        let inv_norm = 1.0 / (x * x + y * y + z * z + 1.0).sqrt();
        let mut e = [0.0f64; 9];
        for k in 0..9 {
            e[k] = (n_basis[0][k] * x + n_basis[1][k] * y + n_basis[2][k] * z + n_basis[3][k])
                * inv_norm;
        }
        // A degenerate minimal sample can make the back-substitution produce
        // non-finite entries; downstream decomposition would not converge on
        // such a matrix, so drop it here.
        if !e.iter().all(|value| value.is_finite()) {
            continue;
        }
        models.push(Matrix3::from_column_slice(&e));
    }

    models
}

/// First-order * first-order -> second-degree product (`[x^2, xy, xz, x,
/// y^2, yz, y, z^2, z, 1]`).
fn o1(a: &[f64; 4], b: &[f64; 4], c: &mut [f64; 10]) {
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[1] + a[1] * b[0];
    c[2] = a[0] * b[2] + a[2] * b[0];
    c[3] = a[0] * b[3] + a[3] * b[0];
    c[4] = a[1] * b[1];
    c[5] = a[1] * b[2] + a[2] * b[1];
    c[6] = a[1] * b[3] + a[3] * b[1];
    c[7] = a[2] * b[2];
    c[8] = a[2] * b[3] + a[3] * b[2];
    c[9] = a[3] * b[3];
}

fn o1p(a: &[f64; 4], b: &[f64; 4], c: &mut [f64; 10]) {
    c[0] += a[0] * b[0];
    c[1] += a[0] * b[1] + a[1] * b[0];
    c[2] += a[0] * b[2] + a[2] * b[0];
    c[3] += a[0] * b[3] + a[3] * b[0];
    c[4] += a[1] * b[1];
    c[5] += a[1] * b[2] + a[2] * b[1];
    c[6] += a[1] * b[3] + a[3] * b[1];
    c[7] += a[2] * b[2];
    c[8] += a[2] * b[3] + a[3] * b[2];
    c[9] += a[3] * b[3];
}

fn o1m(a: &[f64; 4], b: &[f64; 4], c: &mut [f64; 10]) {
    c[0] -= a[0] * b[0];
    c[1] -= a[0] * b[1] + a[1] * b[0];
    c[2] -= a[0] * b[2] + a[2] * b[0];
    c[3] -= a[0] * b[3] + a[3] * b[0];
    c[4] -= a[1] * b[1];
    c[5] -= a[1] * b[2] + a[2] * b[1];
    c[6] -= a[1] * b[3] + a[3] * b[1];
    c[7] -= a[2] * b[2];
    c[8] -= a[2] * b[3] + a[3] * b[2];
    c[9] -= a[3] * b[3];
}

fn o2(a: &[f64; 10], b: &[f64; 4], c: &mut [f64; 20]) {
    c[0] = a[0] * b[0];
    c[1] = a[4] * b[1];
    c[2] = a[0] * b[1] + a[1] * b[0];
    c[3] = a[1] * b[1] + a[4] * b[0];
    c[4] = a[0] * b[2] + a[2] * b[0];
    c[5] = a[0] * b[3] + a[3] * b[0];
    c[6] = a[4] * b[2] + a[5] * b[1];
    c[7] = a[4] * b[3] + a[6] * b[1];
    c[8] = a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
    c[9] = a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
    c[10] = a[2] * b[2] + a[7] * b[0];
    c[11] = a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
    c[12] = a[3] * b[3] + a[9] * b[0];
    c[13] = a[5] * b[2] + a[7] * b[1];
    c[14] = a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
    c[15] = a[6] * b[3] + a[9] * b[1];
    c[16] = a[7] * b[2];
    c[17] = a[7] * b[3] + a[8] * b[2];
    c[18] = a[8] * b[3] + a[9] * b[2];
    c[19] = a[9] * b[3];
}

fn o2p(a: &[f64; 10], b: &[f64; 4], c: &mut [f64; 20]) {
    c[0] += a[0] * b[0];
    c[1] += a[4] * b[1];
    c[2] += a[0] * b[1] + a[1] * b[0];
    c[3] += a[1] * b[1] + a[4] * b[0];
    c[4] += a[0] * b[2] + a[2] * b[0];
    c[5] += a[0] * b[3] + a[3] * b[0];
    c[6] += a[4] * b[2] + a[5] * b[1];
    c[7] += a[4] * b[3] + a[6] * b[1];
    c[8] += a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
    c[9] += a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
    c[10] += a[2] * b[2] + a[7] * b[0];
    c[11] += a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
    c[12] += a[3] * b[3] + a[9] * b[0];
    c[13] += a[5] * b[2] + a[7] * b[1];
    c[14] += a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
    c[15] += a[6] * b[3] + a[9] * b[1];
    c[16] += a[7] * b[2];
    c[17] += a[7] * b[3] + a[8] * b[2];
    c[18] += a[8] * b[3] + a[9] * b[2];
    c[19] += a[9] * b[3];
}

/// Builds the ten cubic trace/determinant constraints from the 4x9 null-space
/// basis. `n_basis[r][3 * j + i]` holds the coefficient of the `(i, j)` entry
/// of `E` (column-major) in the `r`-th basis vector.
#[allow(clippy::needless_range_loop)]
fn compute_trace_constraints(n_basis: &[[f64; 9]; 4], coeffs: &mut [[f64; 20]; 10]) {
    let ee = |i: usize, j: usize| -> [f64; 4] {
        let c = 3 * j + i;
        [n_basis[0][c], n_basis[1][c], n_basis[2][c], n_basis[3][c]]
    };

    // Determinant constraint (row 9).
    {
        let mut d = [0.0f64; 10];
        let mut row = [0.0f64; 20];
        o1(&ee(0, 1), &ee(1, 2), &mut d);
        o1m(&ee(0, 2), &ee(1, 1), &mut d);
        o2(&d, &ee(2, 0), &mut row);
        o1(&ee(0, 2), &ee(1, 0), &mut d);
        o1m(&ee(0, 0), &ee(1, 2), &mut d);
        o2p(&d, &ee(2, 1), &mut row);
        o1(&ee(0, 0), &ee(1, 1), &mut d);
        o1m(&ee(0, 1), &ee(1, 0), &mut d);
        o2p(&d, &ee(2, 2), &mut row);
        coeffs[9] = row;
    }

    // E E^T, then subtract the trace.
    let mut eet = [[[0.0f64; 10]; 3]; 3];
    for i in 0..3 {
        for j in i..3 {
            let mut d = [0.0f64; 10];
            o1(&ee(i, 0), &ee(j, 0), &mut d);
            o1p(&ee(i, 1), &ee(j, 1), &mut d);
            o1p(&ee(i, 2), &ee(j, 2), &mut d);
            eet[i][j] = d;
        }
    }
    for i in 0..3 {
        for j in 0..i {
            eet[i][j] = eet[j][i];
        }
    }
    for k in 0..10 {
        let t = 0.5 * (eet[0][0][k] + eet[1][1][k] + eet[2][2][k]);
        eet[0][0][k] -= t;
        eet[1][1][k] -= t;
        eet[2][2][k] -= t;
    }

    let mut cnt = 0;
    for i in 0..3 {
        for j in 0..3 {
            let mut row = [0.0f64; 20];
            o2(&eet[i][0], &ee(0, j), &mut row);
            o2p(&eet[i][1], &ee(1, j), &mut row);
            o2p(&eet[i][2], &ee(2, j), &mut row);
            coeffs[cnt] = row;
            cnt += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Sturm-sequence root isolation, ported from PoseLib `misc/sturm.h` (N = 10).
// ---------------------------------------------------------------------------

/// Horner evaluation of a degree-`deg` monic polynomial with coefficient
/// vector `f[0..=deg]` (leading coefficient `f[deg]` is assumed to be 1).
fn polyval(f: &[f64], deg: usize, x: f64) -> f64 {
    if deg == 0 {
        return 1.0;
    }
    let mut fx = x + f[deg - 1];
    let mut i = deg as isize - 2;
    while i >= 0 {
        fx = x * fx + f[i as usize];
        i -= 1;
    }
    fx
}

fn build_sturm_seq(fvec: &[f64; 2 * N + 1], svec: &mut [f64; 3 * N]) {
    let mut f = [0.0f64; 3 * N];
    f[..(2 * N + 1)].copy_from_slice(&fvec[..(2 * N + 1)]);

    let mut o1 = 0usize;
    let mut o2 = N + 1;
    let mut o3 = 2 * N + 1;

    for i in 0..(N - 1) {
        let q1 = f[o1 + (N - i)] * f[o2 + (N - 1 - i)];
        let q0 = f[o1 + (N - 1 - i)] * f[o2 + (N - 1 - i)] - f[o1 + (N - i)] * f[o2 + (N - 2 - i)];

        f[o3] = f[o1] - q0 * f[o2];
        for j in 1..(N - 1 - i) {
            f[o3 + j] = f[o1 + j] - q1 * f[o2 + j - 1] - q0 * f[o2 + j];
        }
        let c = -f[o3 + (N - 2 - i)].abs();
        let ci = 1.0 / c;
        for j in 0..(N - 1 - i) {
            f[o3 + j] *= ci;
        }

        std::mem::swap(&mut o1, &mut o2);
        std::mem::swap(&mut o2, &mut o3);

        svec[3 * i] = q0;
        svec[3 * i + 1] = q1;
        svec[3 * i + 2] = c;
    }

    svec[3 * N - 3] = f[o1];
    svec[3 * N - 2] = f[o1 + 1];
    svec[3 * N - 1] = f[o2];
}

fn signchanges(svec: &[f64; 3 * N], x: f64) -> i32 {
    let mut f = [0.0f64; N + 1];
    f[N] = svec[3 * N - 1];
    f[N - 1] = svec[3 * N - 3] + x * svec[3 * N - 2];

    for i in (0..=(N - 2)).rev() {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    let mut count = 0;
    let mut neg1 = f[0] < 0.0;
    for i in 0..N {
        let neg2 = f[i + 1] < 0.0;
        if neg1 != neg2 {
            count += 1;
        }
        neg1 = neg2;
    }
    count
}

fn get_bounds(fvec: &[f64; 2 * N + 1]) -> f64 {
    let mut max = 0.0f64;
    for value in fvec.iter().take(N) {
        max = max.max(value.abs());
    }
    1.0 + max
}

#[allow(clippy::too_many_arguments)]
fn ridders_method_newton(
    fvec: &[f64; 2 * N + 1],
    a_in: f64,
    b_in: f64,
    roots: &mut [f64; N],
    n_roots: &mut usize,
    tol: f64,
) {
    let mut a = a_in;
    let mut b = b_in;
    let mut fa = polyval(fvec, N, a);
    let mut fb = polyval(fvec, N, b);
    if !((fa < 0.0) ^ (fb < 0.0)) {
        return;
    }

    const TOL_NEWTON: f64 = 1e-3;
    for _ in 0..30 {
        if (a - b).abs() < TOL_NEWTON {
            break;
        }
        let c = (a + b) * 0.5;
        let fc = polyval(fvec, N, c);
        let s = (fc * fc - fa * fb).sqrt();
        if s == 0.0 {
            break;
        }
        let d = if fa < fb {
            c + (a - c) * fc / s
        } else {
            c + (c - a) * fc / s
        };
        let fd = polyval(fvec, N, d);

        if if fd >= 0.0 { fc < 0.0 } else { fc > 0.0 } {
            a = c;
            fa = fc;
            b = d;
            fb = fd;
        } else if if fd >= 0.0 { fa < 0.0 } else { fa > 0.0 } {
            b = d;
            fb = fd;
        } else {
            a = d;
            fa = fd;
        }
    }

    let mut x = (a + b) * 0.5;
    let fpvec = &fvec[N + 1..];
    for _ in 0..10 {
        let fx = polyval(fvec, N, x);
        if fx.abs() < tol {
            break;
        }
        let fpx = (N as f64) * polyval(fpvec, N - 1, x);
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < tol {
            break;
        }
    }

    if *n_roots < N {
        roots[*n_roots] = x;
        *n_roots += 1;
    }
}

#[allow(clippy::too_many_arguments)]
fn isolate_roots(
    fvec: &[f64; 2 * N + 1],
    svec: &[f64; 3 * N],
    a: f64,
    b: f64,
    sa: i32,
    sb: i32,
    roots: &mut [f64; N],
    n_roots: &mut usize,
    tol: f64,
    depth: usize,
    budget: &mut usize,
) {
    const MAX_DEPTH: usize = 300;
    if depth > MAX_DEPTH || *budget == 0 {
        return;
    }
    *budget -= 1;
    if b - a < tol {
        if *n_roots < N {
            roots[*n_roots] = b;
            *n_roots += 1;
        }
        return;
    }

    let n_rts = sa - sb;
    if n_rts > 1 {
        let c = (a + b) * 0.5;
        let sc = signchanges(svec, c);
        isolate_roots(
            fvec,
            svec,
            a,
            c,
            sa,
            sc,
            roots,
            n_roots,
            tol,
            depth + 1,
            budget,
        );
        isolate_roots(
            fvec,
            svec,
            c,
            b,
            sc,
            sb,
            roots,
            n_roots,
            tol,
            depth + 1,
            budget,
        );
    } else if n_rts == 1 {
        ridders_method_newton(fvec, a, b, roots, n_roots, tol);
    }
}

/// Returns all real roots of the degree-10 polynomial `coeffs[0..=10]`.
fn bisect_sturm(coeffs: &[f64; N + 1]) -> Vec<f64> {
    if coeffs[N] == 0.0 {
        return Vec::new();
    }

    let mut fvec = [0.0f64; 2 * N + 1];
    fvec[..(N + 1)].copy_from_slice(&coeffs[..(N + 1)]);

    let c_inv = 1.0 / fvec[N];
    for value in fvec.iter_mut().take(N) {
        *value *= c_inv;
    }
    fvec[N] = 1.0;

    for i in 0..(N - 1) {
        fvec[N + 1 + i] = fvec[i + 1] * ((i + 1) as f64 / N as f64);
    }
    fvec[2 * N] = 1.0;

    let mut svec = [0.0f64; 3 * N];
    build_sturm_seq(&fvec, &mut svec);

    let r0 = get_bounds(&fvec);
    let (a, b) = (-r0, r0);
    let sa = signchanges(&svec, a);
    let sb = signchanges(&svec, b);
    let expected_roots = sa - sb;
    if expected_roots <= 0 {
        return Vec::new();
    }
    // The Sturm sequence of a genuine degree-`N` polynomial has at most `N`
    // real roots. Over-determined (n > 5) inputs can yield a subspace whose
    // polynomial is not a true degree-`N` one, so guard against an
    // inconsistent count that would otherwise make `isolate_roots` recurse
    // pathologically.
    if expected_roots > N as i32 {
        return Vec::new();
    }

    let mut roots = [0.0f64; N];
    let mut n_roots = 0usize;
    let mut budget = 50_000usize;
    isolate_roots(
        &fvec,
        &svec,
        a,
        b,
        sa,
        sb,
        &mut roots,
        &mut n_roots,
        1e-10,
        0,
        &mut budget,
    );
    roots[..n_roots].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Canonicalize an essential matrix the same way the PoseLib reference
    /// program does: Frobenius-normalize, flip the sign so the first
    /// significant row-major entry is positive, and flatten row-major.
    fn canonicalize(e: &Matrix3<f64>) -> [f64; 9] {
        let n = e.norm();
        let mut c = e / n;
        let mut flip = false;
        'outer: for i in 0..3 {
            for j in 0..3 {
                if c[(i, j)].abs() > 1e-9 {
                    flip = c[(i, j)] < 0.0;
                    break 'outer;
                }
            }
        }
        if flip {
            c = -c;
        }
        let mut out = [0.0f64; 9];
        for i in 0..3 {
            for j in 0..3 {
                out[3 * i + j] = c[(i, j)];
            }
        }
        out
    }

    fn from_row_major(a: &[f64; 9]) -> Matrix3<f64> {
        let mut m = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                m[(i, j)] = a[3 * i + j];
            }
        }
        m
    }

    fn max_abs_diff(a: &[f64; 9], b: &[f64; 9]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
    }

    /// Ground-truth data for `ref5pt.cpp` (a rotation + translation applied to
    /// five 3D points, then normalized to bearing vectors). The expected
    /// canonical matrices were produced by PoseLib's `relpose_5pt` built in
    /// the pinned COLMAP container; the solver must reproduce the same
    /// solution set.
    #[test]
    #[allow(clippy::excessive_precision)]
    fn matches_poselib_relpose_5pt_reference() {
        let x1 = [
            Vector3::new(
                0.07469715670684389,
                -0.049798104471229267,
                0.99596208942458531,
            ),
            Vector3::new(
                0.20637499145878313,
                0.075045451439557501,
                0.97559086871424749,
            ),
            Vector3::new(
                -0.11280056649047149,
                0.14502929977346335,
                0.98297636513125164,
            ),
            Vector3::new(
                0.053490469515994045,
                0.13372617378998511,
                0.9895736860458898,
            ),
            Vector3::new(-0.25465730964743338, -0.1175341429142, 0.9598621671326335),
        ];
        let x2 = [
            Vector3::new(
                -0.00082194084172214691,
                -0.19635969956791816,
                0.98053158684401887,
            ),
            Vector3::new(
                0.08702790442460788,
                -0.06294079357988476,
                0.99421557036439312,
            ),
            Vector3::new(
                -0.23920034110119834,
                -0.012404764010233136,
                0.97089099215458818,
            ),
            Vector3::new(
                -0.025178915607500921,
                -0.024932322677008921,
                0.99937200355761413,
            ),
            Vector3::new(
                -0.3350682764722655,
                -0.27984803050765594,
                0.89967456889860553,
            ),
        ];
        let expected: [[f64; 9]; 4] = [
            [
                4.508292739e-02,
                6.639035305e-01,
                -3.110085583e-02,
                -6.326289247e-01,
                8.300302802e-02,
                2.958320010e-01,
                -8.619322909e-02,
                -2.285846874e-01,
                4.389424481e-02,
            ],
            [
                -5.070106574e-01,
                2.292556590e-01,
                -2.704940379e-01,
                1.914999009e-01,
                5.433845181e-01,
                3.568977315e-01,
                2.307379706e-01,
                -3.215854470e-01,
                3.525426833e-02,
            ],
            [
                -5.261017666e-01,
                1.239866598e-01,
                -3.735533346e-01,
                1.197890024e-01,
                4.464800859e-01,
                -3.203573669e-01,
                3.791714425e-01,
                3.191601823e-01,
                -7.965692740e-02,
            ],
            [
                -4.176923994e-02,
                -2.710554639e-01,
                -9.009772162e-02,
                1.149162858e-01,
                -9.965876760e-02,
                -6.783772628e-01,
                1.618812428e-01,
                6.278092759e-01,
                -1.139443273e-01,
            ],
        ];

        let models = relpose_5pt(&x1, &x2);
        let mut got: Vec<[f64; 9]> = models.iter().map(canonicalize).collect();
        assert_eq!(got.len(), expected.len(), "solution count mismatch");

        let expected: Vec<[f64; 9]> = expected
            .iter()
            .map(|e| canonicalize(&from_row_major(e)))
            .collect();
        for expected_model in &expected {
            let best = got
                .iter()
                .enumerate()
                .map(|(i, g)| (i, max_abs_diff(g, expected_model)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            assert!(
                best.1 < 1e-5,
                "no matching model (best diff {:.3e})",
                best.1
            );
            got.swap_remove(best.0);
        }
    }
}
