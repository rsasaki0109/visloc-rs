//! Simplicial *block* Cholesky for the block-structured pose-graph normal
//! equations.
//!
//! The SE(3) and rotation-relaxation normal matrices are not arbitrary sparse
//! matrices: every variable is a fixed-size block (`6×6` for an SE(3) pose,
//! `3×3` for a rotation column or a translation center), and a nonzero only ever
//! appears as a whole dense block — an edge couples two *blocks*, never two
//! stray scalars. `nalgebra_sparse::CscCholesky` ignores that and factors the
//! system one scalar column at a time, so it pays the sparse gather/scatter
//! bookkeeping `b²` times per block and never touches a dense kernel.
//!
//! This module factors at block granularity instead. The symbolic phase is the
//! standard Gilbert–Ng–Peyton elimination-tree column pattern (the same
//! machinery the [`crate::reordering`] fill counter uses, here recorded rather
//! than merely counted), and the numeric phase is a left-looking Cholesky whose
//! "scalars" are stack-allocated `B×B` matrices — so each diagonal factorization,
//! triangular solve, and trailing update is a single dense `nalgebra` kernel that
//! the compiler unrolls and vectorizes. The block size `B` is a const generic, so
//! the `B = 3` and `B = 6` instantiations are fully monomorphized.
//!
//! The caller hands in COO triplets that are already in the fill-reducing
//! variable order (see [`crate::reordering`]), so this works purely in the
//! permuted space exactly as the `CscCholesky` path does; the matrix is assumed
//! symmetric (both off-diagonal cross blocks are supplied) and positive-definite
//! by construction (a sum of `JᵀΩJ` edge outer products plus an optional `λI`).
//! A non-positive-definite diagonal block aborts the factorization with `Err`,
//! matching the scalar path's `SingularSystem` behavior.

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

const NONE: usize = usize::MAX;

/// Solve the SPD system `(A + λI) X = RHS`, where `A` is given by `triplets`
/// (scalar COO, summed on assembly, symmetric, in the caller's permuted order)
/// at block size `block_size`, and `RHS` has one or more columns. Returns the
/// solution `X` (same row order as `RHS`) or `Err(())` when a diagonal block is
/// not positive-definite. `block_size` must be 3 or 6 — the only sizes the
/// pose-graph back-end assembles.
pub(crate) fn solve_spd_block(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    rhs: &DMatrix<f64>,
    lambda: f64,
) -> Result<DMatrix<f64>, ()> {
    match block_size {
        3 => solve_dispatch::<3>(triplets, dim, rhs, lambda),
        6 => solve_dispatch::<6>(triplets, dim, rhs, lambda),
        other => panic!("block_cholesky supports block sizes 3 and 6, got {other}"),
    }
}

fn solve_dispatch<const B: usize>(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    rhs: &DMatrix<f64>,
    lambda: f64,
) -> Result<DMatrix<f64>, ()> {
    let factor = BlockCholesky::<B>::factor(triplets, dim, lambda)?;
    let mut out = DMatrix::<f64>::zeros(dim, rhs.ncols());
    for c in 0..rhs.ncols() {
        let column = DVector::from_column_slice(rhs.column(c).as_slice());
        out.set_column(c, &factor.solve_vec(&column));
    }
    Ok(out)
}

/// A computed block Cholesky factor `A = L Lᵀ` at block size `B`.
struct BlockCholesky<const B: usize> {
    /// Number of block columns (`dim / B`).
    n: usize,
    /// `col_rows[j]` is the sorted list of block rows present in column `j` of
    /// `L`, with `col_rows[j][0] == j` (the diagonal block).
    col_rows: Vec<Vec<usize>>,
    /// `col_vals[j][t]` is the `B×B` block of `L` at row `col_rows[j][t]`. Slot
    /// `0` holds the lower-triangular diagonal factor `L_jj`.
    col_vals: Vec<Vec<SMatrix<f64, B, B>>>,
    /// `diag_inv[j] = L_jj⁻¹`, cached so the forward/backward solves are plain
    /// matrix products instead of repeated triangular solves.
    diag_inv: Vec<SMatrix<f64, B, B>>,
}

impl<const B: usize> BlockCholesky<B> {
    /// Symbolically and numerically factor `(A + λI)`.
    fn factor(triplets: &[(usize, usize, f64)], dim: usize, lambda: f64) -> Result<Self, ()> {
        debug_assert!(dim % B == 0, "dim must be a multiple of the block size");
        let n = dim / B;

        // Lower-triangular original block values (block row ≥ block col), and the
        // structural pattern they induce.
        let mut a_lower: Vec<Vec<(usize, SMatrix<f64, B, B>)>> = Vec::new();
        let (block_lower, blocks) = assemble_blocks::<B>(triplets, n, lambda);
        a_lower.extend(blocks);

        let (col_rows, contributors) = symbolic(&block_lower, n);

        // Allocate the factor's value slots, zero-initialized.
        let mut col_vals: Vec<Vec<SMatrix<f64, B, B>>> = col_rows
            .iter()
            .map(|rows| vec![SMatrix::<f64, B, B>::zeros(); rows.len()])
            .collect();
        let mut diag_inv = vec![SMatrix::<f64, B, B>::zeros(); n];

        for j in 0..n {
            let rows = &col_rows[j];
            let m = rows.len();
            // Dense workspace for column j's blocks, indexed by position in
            // `rows`. Seed it with the original `A` blocks of column j.
            let mut ws = vec![SMatrix::<f64, B, B>::zeros(); m];
            for &(i, block) in &a_lower[j] {
                ws[pos(rows, i)] = block;
            }

            // Left-looking trailing updates: subtract Lᵢₖ·Lⱼₖᵀ for every prior
            // column k that fills into column j.
            for &k in &contributors[j] {
                let k_rows = &col_rows[k];
                let pj = pos(k_rows, j);
                let ljk_t = col_vals[k][pj].transpose();
                for t in pj..k_rows.len() {
                    let i = k_rows[t];
                    let update = col_vals[k][t] * ljk_t;
                    ws[pos(rows, i)] -= update;
                }
            }

            // Factor the (updated) diagonal block and record L_jj, L_jj⁻¹.
            let chol = ws[0].cholesky().ok_or(())?;
            let ljj = chol.l();
            let ljj_inv = ljj.try_inverse().ok_or(())?;
            col_vals[j][0] = ljj;
            diag_inv[j] = ljj_inv;

            // Lᵢⱼ = Yᵢ · (L_jjᵀ)⁻¹ = Yᵢ · (L_jj⁻¹)ᵀ for the below-diagonal rows.
            let ljj_inv_t = ljj_inv.transpose();
            for t in 1..m {
                col_vals[j][t] = ws[t] * ljj_inv_t;
            }
        }

        Ok(Self {
            n,
            col_rows,
            col_vals,
            diag_inv,
        })
    }

    /// Solve `A x = b` for a single right-hand side via block forward and
    /// backward substitution.
    fn solve_vec(&self, b: &DVector<f64>) -> DVector<f64> {
        // Gather the dense RHS into per-block sub-vectors.
        let mut y: Vec<SVector<f64, B>> = (0..self.n)
            .map(|j| SVector::<f64, B>::from_fn(|k, _| b[j * B + k]))
            .collect();

        // Forward substitution: solve L y = b, column by column.
        for j in 0..self.n {
            let yj = self.diag_inv[j] * y[j];
            y[j] = yj;
            // Below-diagonal rows (skip the diagonal slot 0); i > j, so the
            // update never aliases y[j].
            for (&i, block) in self.col_rows[j].iter().zip(&self.col_vals[j]).skip(1) {
                y[i] -= block * yj;
            }
        }

        // Backward substitution: solve Lᵀ x = y, columns in reverse.
        for j in (0..self.n).rev() {
            let mut acc = y[j];
            for (&i, block) in self.col_rows[j].iter().zip(&self.col_vals[j]).skip(1) {
                acc -= block.transpose() * y[i];
            }
            y[j] = self.diag_inv[j].transpose() * acc;
        }

        // Scatter back into a dense solution vector.
        let mut x = DVector::<f64>::zeros(self.n * B);
        for j in 0..self.n {
            for k in 0..B {
                x[j * B + k] = y[j][k];
            }
        }
        x
    }
}

/// Assemble the lower-triangular original block values and the off-diagonal
/// block pattern from scalar COO triplets, folding `λ` into every diagonal
/// block. Returns `(block_lower, a_lower)` where `block_lower[i]` is the sorted
/// set of block columns `< i` coupled to block row `i`, and `a_lower[j]` is the
/// list of `(block_row ≥ j, block)` originally present in block column `j`.
#[allow(clippy::type_complexity)]
fn assemble_blocks<const B: usize>(
    triplets: &[(usize, usize, f64)],
    n: usize,
    lambda: f64,
) -> (Vec<Vec<usize>>, Vec<Vec<(usize, SMatrix<f64, B, B>)>>) {
    // Sparse map per block column: block row → dense B×B block.
    let mut cols: Vec<std::collections::BTreeMap<usize, SMatrix<f64, B, B>>> =
        vec![std::collections::BTreeMap::new(); n];
    let mut pattern: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    for &(r, c, v) in triplets {
        let (br, bc) = (r / B, c / B);
        if br > bc {
            cols[bc].entry(br).or_insert_with(SMatrix::zeros)[(r % B, c % B)] += v;
            pattern[br].insert(bc);
        } else if br == bc {
            // Diagonal block: keep the full B×B (it is symmetric and is the
            // block we Cholesky-factor).
            cols[bc].entry(br).or_insert_with(SMatrix::zeros)[(r % B, c % B)] += v;
        }
        // br < bc (strict upper) is the transpose of an entry we already keep in
        // the lower triangle, so it is dropped.
    }

    if lambda != 0.0 {
        for (j, col) in cols.iter_mut().enumerate() {
            let diag = col.entry(j).or_insert_with(SMatrix::zeros);
            for d in 0..B {
                diag[(d, d)] += lambda;
            }
        }
    }

    let block_lower = pattern
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();
    let a_lower = cols
        .into_iter()
        .map(|m| m.into_iter().collect())
        .collect();
    (block_lower, a_lower)
}

/// Gilbert–Ng–Peyton symbolic factorization: from the strictly-lower block
/// pattern, build the elimination tree and the per-column nonzero row pattern of
/// `L` (each sorted, diagonal first), plus its transpose `contributors[j]` = the
/// prior columns that fill into column `j`.
fn symbolic(block_lower: &[Vec<usize>], n: usize) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    // Elimination tree via the path-compressing ancestor walk.
    let mut parent = vec![NONE; n];
    let mut ancestor = vec![NONE; n];
    for (i, lower) in block_lower.iter().enumerate() {
        for &j in lower {
            let mut node = j;
            while ancestor[node] != NONE && ancestor[node] != i {
                let next = ancestor[node];
                ancestor[node] = i;
                node = next;
            }
            if ancestor[node] == NONE {
                ancestor[node] = i;
                parent[node] = i;
            }
        }
    }

    // Column patterns: row i contributes to every column on the etree path from
    // each of its original lower neighbours up to i. Iterating i upward keeps
    // each column's row list sorted; `mark` dedupes within a row.
    let mut col_rows: Vec<Vec<usize>> = (0..n).map(|j| vec![j]).collect();
    let mut mark = vec![NONE; n];
    for (i, lower) in block_lower.iter().enumerate() {
        mark[i] = i;
        for &j in lower {
            let mut k = j;
            while mark[k] != i {
                mark[k] = i;
                col_rows[k].push(i);
                match parent[k] {
                    NONE => break,
                    p => k = p,
                }
            }
        }
    }

    // contributors[j] = { k < j : L[j][k] ≠ 0 }, the transpose of the row lists.
    let mut contributors: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (k, rows) in col_rows.iter().enumerate() {
        for &i in rows {
            if i > k {
                contributors[i].push(k);
            }
        }
    }
    (col_rows, contributors)
}

/// Position of block row `i` in a sorted column row list. The caller guarantees
/// membership (original nonzeros and fill both lie inside the column pattern).
#[inline]
fn pos(rows: &[usize], i: usize) -> usize {
    rows.binary_search(&i)
        .expect("row index lies within the column's symbolic pattern")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use nalgebra_sparse::{CooMatrix, CscMatrix};

    /// Deterministic splitmix64 stream for reproducible random test matrices.
    struct Rng(u64);
    impl Rng {
        fn next_f64(&mut self) -> f64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) * 2.0 - 1.0
        }
    }

    /// Build a random SPD matrix `M = A Aᵀ + s·I` (dense) plus the COO triplets
    /// of its symmetric pattern, optionally sparsifying off-diagonal *blocks* so
    /// the structure resembles a pose graph rather than a full matrix.
    fn random_spd(
        n: usize,
        b: usize,
        block_keep: impl Fn(usize, usize) -> bool,
        seed: u64,
    ) -> (DMatrix<f64>, Vec<(usize, usize, f64)>) {
        let dim = n * b;
        let mut rng = Rng(seed);
        let a = DMatrix::<f64>::from_fn(dim, dim, |_, _| rng.next_f64());
        let mut m = &a * a.transpose();
        for k in 0..dim {
            m[(k, k)] += dim as f64; // strengthen the diagonal for conditioning
        }

        // Zero out the off-diagonal blocks the pattern drops, keeping symmetry.
        for bi in 0..n {
            for bj in 0..n {
                if bi != bj && !block_keep(bi, bj) {
                    for r in 0..b {
                        for c in 0..b {
                            m[(bi * b + r, bj * b + c)] = 0.0;
                        }
                    }
                }
            }
        }

        let mut triplets = Vec::new();
        for r in 0..dim {
            for c in 0..dim {
                if m[(r, c)] != 0.0 {
                    triplets.push((r, c, m[(r, c)]));
                }
            }
        }
        (m, triplets)
    }

    fn assert_solves(m: &DMatrix<f64>, triplets: &[(usize, usize, f64)], b: usize, lambda: f64) {
        let dim = m.nrows();
        // Reference: dense Cholesky of (M + λI).
        let mut damped = m.clone();
        for k in 0..dim {
            damped[(k, k)] += lambda;
        }

        // Two right-hand sides exercise the multi-column path.
        let mut rng = Rng(0xabcd);
        let rhs = DMatrix::<f64>::from_fn(dim, 2, |_, _| rng.next_f64());
        let expected = damped
            .clone()
            .cholesky()
            .expect("reference SPD")
            .solve(&rhs);

        let got = solve_spd_block(triplets, dim, b, &rhs, lambda).expect("block factor SPD");
        assert!(
            (&got - &expected).norm() < 1e-9,
            "block solve disagrees with dense Cholesky (b={b}, λ={lambda}): err {}",
            (&got - &expected).norm()
        );

        // Also agree with nalgebra's scalar sparse Cholesky on the same system.
        let mut coo = CooMatrix::<f64>::new(dim, dim);
        for &(r, c, v) in triplets {
            coo.push(r, c, v);
        }
        for k in 0..dim {
            if lambda != 0.0 {
                coo.push(k, k, lambda);
            }
        }
        let csc = CscMatrix::from(&coo);
        let scalar = nalgebra_sparse::factorization::CscCholesky::factor(&csc)
            .expect("scalar SPD")
            .solve(&rhs);
        assert!(
            (&got - &scalar).norm() < 1e-9,
            "block solve disagrees with scalar CscCholesky (b={b}): err {}",
            (&got - &scalar).norm()
        );
    }

    #[test]
    fn block3_dense_matrix_matches_references() {
        let (m, t) = random_spd(5, 3, |_, _| true, 1);
        assert_solves(&m, &t, 3, 0.0);
        assert_solves(&m, &t, 3, 0.25);
    }

    #[test]
    fn block6_dense_matrix_matches_references() {
        let (m, t) = random_spd(4, 6, |_, _| true, 2);
        assert_solves(&m, &t, 6, 0.0);
        assert_solves(&m, &t, 6, 1.5);
    }

    #[test]
    fn block6_banded_with_loop_matches_references() {
        // A chain with a long-range loop edge — the pose-graph shape, where fill
        // appears beyond the original pattern and exercises the etree.
        let n = 12;
        let keep = |bi: usize, bj: usize| {
            let d = bi.abs_diff(bj);
            d == 1 || (bi == 0 && bj == n - 1) || (bi == n - 1 && bj == 0)
        };
        let (m, t) = random_spd(n, 6, keep, 7);
        assert_solves(&m, &t, 6, 0.0);
        assert_solves(&m, &t, 6, 1e-3);
    }

    #[test]
    fn block3_sparse_pattern_matches_references() {
        let n = 20;
        let keep = |bi: usize, bj: usize| bi.abs_diff(bj) <= 2;
        let (m, t) = random_spd(n, 3, keep, 11);
        assert_solves(&m, &t, 3, 0.0);
    }

    #[test]
    fn duplicate_triplets_are_summed() {
        // Splitting each entry into two halves must reproduce the single-entry
        // system, matching the COO summation contract.
        let (m, t) = random_spd(6, 3, |bi, bj| bi.abs_diff(bj) <= 1, 3);
        let split: Vec<(usize, usize, f64)> = t
            .iter()
            .flat_map(|&(r, c, v)| [(r, c, v * 0.25), (r, c, v * 0.75)])
            .collect();
        assert_solves(&m, &split, 3, 0.1);
    }

    /// Temporary A/B timing harness (block vs. scalar CscCholesky) on a
    /// pose-graph-shaped system, factored in the same fill-reducing order both
    /// ways. Run with `cargo test -p visloc-slam --release -- --ignored
    /// --nocapture bench_block_vs_scalar`.
    #[test]
    #[ignore]
    fn bench_block_vs_scalar() {
        use crate::reordering::Reordering;
        use std::time::Instant;

        let b = 6;
        let side = 50; // 2500 blocks ~ sphere2500 scale
        let n = side * side;
        let dim = n * b;

        // 2D-grid block adjacency: the canonical wide graph whose factor fills.
        let id = |r: usize, c: usize| r * side + c;
        let mut edges: Vec<(usize, usize)> = Vec::new();
        for r in 0..side {
            for c in 0..side {
                if c + 1 < side {
                    edges.push((id(r, c), id(r, c + 1)));
                }
                if r + 1 < side {
                    edges.push((id(r, c), id(r + 1, c)));
                }
            }
        }

        // Assemble a diagonally-dominant SPD system from the edges.
        let mut rng = Rng(99);
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        let mut push_block = |br: usize, bc: usize, rng: &mut Rng| {
            for r in 0..b {
                for c in 0..b {
                    triplets.push((br * b + r, bc * b + c, 0.1 * rng.next_f64()));
                }
            }
        };
        for &(u, v) in &edges {
            push_block(u, v, &mut rng);
            push_block(v, u, &mut rng);
        }
        for j in 0..n {
            for d in 0..b {
                triplets.push((j * b + d, j * b + d, 10.0));
            }
        }

        let order = Reordering::fill_reducing(dim, b, &triplets);
        let permuted = order.permute_triplets(&triplets);
        let rhs = DMatrix::<f64>::from_fn(dim, 1, |i, _| (i % 7) as f64 - 3.0);

        let reps = 5;
        // Warm up + correctness cross-check on the permuted system.
        let block0 = solve_spd_block(&permuted, dim, b, &rhs, 1e-3).expect("block SPD");

        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = solve_spd_block(&permuted, dim, b, &rhs, 1e-3).unwrap();
        }
        let block_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let mut coo = CooMatrix::<f64>::new(dim, dim);
        for &(r, c, v) in &permuted {
            coo.push(r, c, v);
        }
        for k in 0..dim {
            coo.push(k, k, 1e-3);
        }
        let csc = CscMatrix::from(&coo);
        let scalar0 = nalgebra_sparse::factorization::CscCholesky::factor(&csc)
            .unwrap()
            .solve(&rhs);
        assert!((&block0 - &scalar0).norm() < 1e-6, "bench systems disagree");

        let t1 = Instant::now();
        for _ in 0..reps {
            let csc = CscMatrix::from(&coo);
            let _ = nalgebra_sparse::factorization::CscCholesky::factor(&csc)
                .unwrap()
                .solve(&rhs);
        }
        let scalar_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;

        println!(
            "block{b} {n} blocks (dim {dim}): block {block_ms:.1} ms vs scalar {scalar_ms:.1} ms => {:.2}x",
            scalar_ms / block_ms
        );
    }

    #[test]
    fn non_spd_block_returns_err() {
        // A zero matrix is not positive-definite: the first diagonal block
        // Cholesky must fail rather than panic.
        let dim = 9;
        let rhs = DMatrix::<f64>::zeros(dim, 1);
        assert!(solve_spd_block(&[], dim, 3, &rhs, 0.0).is_err());
    }
}
