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
//!
//! # Reusing the symbolic analysis
//!
//! A Levenberg–Marquardt solve factors a *new* normal matrix every iteration,
//! but always with the *same* sparsity (and the same fill-reducing order). The
//! pattern-dependent work — the Gilbert–Ng–Peyton symbolic factorization
//! (elimination tree, per-column row patterns, levels) and the COO→block pattern
//! assembly — is therefore iteration-invariant. [`analyze`] computes it once into
//! a [`BlockSymbolic`], and [`solve_spd_block_cached`] reuses that across solves,
//! re-running only the value scatter and the numeric factorization. (Measured on
//! the g2o benchmarks the symbolic phase + the old per-iteration `BTreeMap`
//! assembly were ~20–30 % of the solve, so caching them is a clean ~1.1× with no
//! change to the result; the small chain graphs gain most relative to their tiny
//! numeric phase.) The value scatter itself is now a direct binary search into
//! the cached per-column row list rather than a fresh `BTreeMap` per column. The
//! one-shot [`solve_spd_block`] (used by callers without a repeated pattern, e.g.
//! the BA Schur solve and the chordal initializer) routes through the same
//! analyze-then-refactor path with a throwaway analysis.
//!
//! # Why not supernodal?
//!
//! Classic high-performance sparse Cholesky (CHOLMOD et al.) is *supernodal*: it
//! amalgamates columns of identical structure into dense panels and runs the
//! trailing updates as BLAS-3 panel products. That win comes from turning a
//! *scalar* factorization into dense kernels — but this factorization is already
//! dense at `B×B` block granularity, so the bulk of that benefit is captured.
//! Amalgamating the `B×B` blocks into still-wider panels was implemented and
//! measured (see the `supernode_width_distribution` test): on a `sphere2500`-
//! scale 2D grid the supernodes are mostly width-1 leaves with a handful of wide
//! separators, and routing those separators through dynamic `DMatrix` panels was
//! *slower* than the per-block path at every width threshold tried — even for the
//! single widest (≈450×450) separator. nalgebra has no tuned BLAS backend, so a
//! dynamic-panel GEMM/Cholesky does not beat the fully-unrolled, heap-free
//! `SMatrix<B, B>` kernel here, and the panel assembly/scatter is pure overhead.
//! The per-block left-looking path below is therefore the production path.
//!
//! # Parallelism
//!
//! The numeric phase is parallelized across the elimination tree. Group the
//! columns into *levels* — a column's level is one past the deepest level among
//! its contributors — so every column in a level is mutually independent (a
//! contributor is always a proper descendant, hence a strictly lower level).
//! Processing level by level, a whole level is factored on the rayon pool while
//! the lower levels (already finalized) are read; columns are written back after
//! the level completes, so the shared read and the mutation never overlap. The
//! schedule is just a topological reordering of the `0..n` sweep, so the result
//! is bit-identical to the sequential factor (a test asserts this).
//!
//! Across-level parallelism alone is bounded by the tree shape: in pose-graph /
//! BA factors the work concentrates in the *narrow* separator levels near the
//! root (where only a handful of columns are independent), while the *wide*
//! levels are the cheap leaves — so it can only feed a few cores on the bulk of
//! the work, and the width-1 separator chain stays fully serial. To keep it from
//! ever regressing the cheap-but-wide leaf levels of chain-like graphs (e.g.
//! parking-garage), a level is farmed out across its columns only when it is both
//! wide enough ([`PARALLEL_MIN_LEVEL_WIDTH`]) and heavy enough
//! ([`PARALLEL_MIN_LEVEL_WORK`]), and only for systems past
//! [`PARALLEL_MIN_BLOCKS`].
//!
//! The serial separator chain is then attacked by a *second*, orthogonal axis:
//! **intra-column** parallelism. A heavy separator column has hundreds of
//! contributors, and its trailing update `Y = A_j - Σ_k Lᵢₖ·Lⱼₖᵀ` is a sum over
//! those contributors — embarrassingly parallel, since each contributor is an
//! already-finalized descendant. A column that did not go out with its level
//! (above all the width-1 chain) but has at least [`INTRA_MIN_CONTRIB`]
//! contributors *and* enough trailing-update work ([`INTRA_MIN_WORK`], so the
//! rayon dispatch is amortized — a column with many contributors but few rows is
//! left inline) is therefore factored by reducing that sum on the rayon pool (see
//! [`factor_column_intra`]). This is pure-Rust *intra-separator* parallelism at
//! block granularity — note it is **not** the dense-GEMM/BLAS-3 split dismissed
//! under "Why not supernodal?": it parallelizes *across contributors* (the
//! left-looking updates), which needs no tuned BLAS. Because the contributor sum
//! is regrouped, the result is not bit-identical to the sequential subtraction
//! (floating-point addition is not associative) — but it agrees to rounding and
//! is deterministic across thread counts (chunking keyed to [`INTRA_CONTRIB_CHUNK`],
//! folded in order). The across-level path stays exactly bit-identical; only the
//! heavy-column path trades that for filling the pool.
//!
//! Together the two axes help most on heavy, solve-dominated graphs (≈1.4× on
//! `torus3D`, ≈1.26× on `rim` — up from ≈1.17× / ≈1.09× with across-level alone)
//! and are neutral on small or fast-converging ones, with no regression on
//! chain-like graphs. The [`INTRA_MIN_WORK`] bar is what keeps the dense 3D
//! graphs (`cubicle`, `sphere2500`) neutral: their separators clear the
//! contributor count but are too light to amortize an intra dispatch, so a
//! bare-count gate regressed them — gating on work recovers that and adds
//! ≈5 % end-to-end on `torus3D`/`rim`.

use std::collections::BTreeSet;

use nalgebra::{DMatrix, DVector, SMatrix, SVector};

const NONE: usize = usize::MAX;

/// Below this block count the elimination tree is too small for the thread
/// pool to pay for itself, so the numeric phase stays single-threaded.
const PARALLEL_MIN_BLOCKS: usize = 256;
/// A level (a set of mutually-independent columns) is farmed out to the rayon
/// pool only when it is at least this wide. The pool is persistent, so the
/// per-task cost is low and even modestly wide levels of expensive separator
/// columns pay off; narrower levels — the deep chain near the root — run
/// sequentially.
const PARALLEL_MIN_LEVEL_WIDTH: usize = 2;
/// …and only when the level's estimated work (trailing-update block multiplies,
/// `Σ_j |contributors(j)|·|rows(j)|`) clears this bar. Cheap-but-wide leaf
/// levels — common in chain-like graphs (e.g. parking-garage) — would otherwise
/// lose the rayon dispatch cost, turning the parallel path into a regression.
const PARALLEL_MIN_LEVEL_WORK: usize = 8192;

/// Intra-column parallelism: a column that does *not* go out with its level
/// (the narrow levels — above all the width-1 separator chain near the root,
/// which across-level parallelism leaves fully serial) is still parallel *over
/// its own contributors* when it has at least this many. The heavy separator
/// columns have hundreds, so their trailing-update reduction (`Σ_k Lᵢₖ·Lⱼₖᵀ`)
/// fills the pool on exactly the columns the level scheme cannot.
const INTRA_MIN_CONTRIB: usize = 64;
/// …and only when that column's trailing-update work (block multiplies,
/// `|contributors|·|rows|`) clears this bar. A high contributor count alone is
/// not enough: a column with many contributors but few rows does little
/// arithmetic per contributor, so the rayon dispatch and the per-chunk dense
/// delta dominate and the intra path runs *slower* than the inline column. The
/// bar admits only the genuinely heavy separators (where intra wins big) and
/// leaves the cheap-but-tall ones inline. Measured to-rounding-identical at any
/// threshold; `12000` is where the per-column win crosses the dispatch overhead
/// across the SE-Sync 3D graphs (it lifts the dense `cubicle`/`sphere2500`
/// factor from ~1.06× to neutral-or-better while keeping the `torus3D`/`rim`
/// gains, ~+5 % end-to-end). Mirrors [`PARALLEL_MIN_LEVEL_WORK`].
const INTRA_MIN_WORK: usize = 12000;
/// Contributors per rayon task in the intra-column reduction. Fixed (not keyed
/// to the thread count) so the fold order — hence the result — is the same on
/// any number of threads; a heavy column has enough contributors to make many
/// such chunks regardless.
const INTRA_CONTRIB_CHUNK: usize = 8;

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
    solve_spd_block_inner(
        triplets,
        dim,
        block_size,
        rhs,
        lambda,
        default_thread_count(),
        PARALLEL_MIN_LEVEL_WORK,
        INTRA_MIN_CONTRIB,
        INTRA_MIN_WORK,
    )
}

/// Like [`solve_spd_block`], but reuse a previously computed [`BlockSymbolic`]
/// (analyzing and storing it on the first call). For a system whose sparsity is
/// fixed across solves — the Levenberg–Marquardt normal equations, re-solved with
/// new values every iteration in the *same* fill-reducing order — this skips the
/// symbolic factorization and the COO→block pattern assembly on every call but
/// the first, leaving only the value scatter and the numeric factorization. The
/// result is identical to calling `solve_spd_block` with the same `triplets`.
/// The caller owns the `cache` (alongside the fill-reducing order cache) and must
/// only feed systems of the same pattern/`block_size`/`dim` into it.
pub(crate) fn solve_spd_block_cached(
    cache: &mut Option<BlockSymbolic>,
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    rhs: &DMatrix<f64>,
    lambda: f64,
) -> Result<DMatrix<f64>, ()> {
    let sym = cache.get_or_insert_with(|| analyze(triplets, dim, block_size));
    debug_assert_eq!(sym.block_size, block_size);
    debug_assert_eq!(sym.n * block_size, dim);
    let threads = default_thread_count();
    match block_size {
        3 => solve_with_symbolic::<3>(
            sym,
            triplets,
            rhs,
            lambda,
            threads,
            PARALLEL_MIN_LEVEL_WORK,
            INTRA_MIN_CONTRIB,
            INTRA_MIN_WORK,
        ),
        6 => solve_with_symbolic::<6>(
            sym,
            triplets,
            rhs,
            lambda,
            threads,
            PARALLEL_MIN_LEVEL_WORK,
            INTRA_MIN_CONTRIB,
            INTRA_MIN_WORK,
        ),
        other => panic!("block_cholesky supports block sizes 3 and 6, got {other}"),
    }
}

/// Factor the SPD system `A` (given by `triplets`, scalar COO, symmetric, in the
/// caller's order) at block size `block_size` and return the block factor `L` of
/// `A = L Lᵀ` as plain `nalgebra` data: `(col_rows, col_vals, diag_inv)`, where
/// `col_rows[j]` are the block rows present in column `j` of `L` (diagonal `== j`
/// first, then strictly-below rows ascending), `col_vals[j]` the matching
/// `block_size × block_size` blocks (`col_vals[j][0] == L_jj`), and `diag_inv[j]
/// == L_jj⁻¹`. `Err(())` when a diagonal block is not positive-definite.
///
/// This exposes the factor for covariance recovery (the block Takahashi recursion
/// in [`crate::covariance`] reads exactly this representation) without forming a
/// dense `A`. No fill-reducing permutation is applied — the factor is in the
/// caller's variable order, so block column `j` maps straight back to variable
/// `j` (more fill than the reordered solve, but covariance recovery is not the
/// per-iteration hot path and the natural order keeps the index mapping trivial).
/// `block_size` must be 3 or 6.
#[allow(clippy::type_complexity)]
pub(crate) fn factor_blocks(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<DMatrix<f64>>>, Vec<DMatrix<f64>>), ()> {
    match block_size {
        3 => factor_blocks_inner::<3>(triplets, dim),
        6 => factor_blocks_inner::<6>(triplets, dim),
        other => panic!("block_cholesky supports block sizes 3 and 6, got {other}"),
    }
}

#[allow(clippy::type_complexity)]
fn factor_blocks_inner<const B: usize>(
    triplets: &[(usize, usize, f64)],
    dim: usize,
) -> Result<(Vec<Vec<usize>>, Vec<Vec<DMatrix<f64>>>, Vec<DMatrix<f64>>), ()> {
    let sym = analyze(triplets, dim, B);
    let (col_vals, diag_inv) = refactor_numeric::<B>(
        &sym,
        triplets,
        0.0,
        default_thread_count(),
        PARALLEL_MIN_LEVEL_WORK,
        INTRA_MIN_CONTRIB,
        INTRA_MIN_WORK,
    )?;
    let to_dyn = |m: &SMatrix<f64, B, B>| DMatrix::from_column_slice(B, B, m.as_slice());
    let col_vals_dyn = col_vals
        .iter()
        .map(|col| col.iter().map(to_dyn).collect())
        .collect();
    let diag_inv_dyn = diag_inv.iter().map(to_dyn).collect();
    Ok((sym.col_rows.clone(), col_vals_dyn, diag_inv_dyn))
}

/// The block size known, dispatch to the monomorphized solver. `threads` caps
/// the worker threads used for the numeric phase (`1` forces the sequential
/// path); `min_level_work` is the per-level work bar below which a level is not
/// farmed out across its columns; `min_intra_contrib` and `min_intra_work` gate
/// the intra-column path — a column that stayed inline is factored in parallel
/// over its contributors only when it has at least `min_intra_contrib` of them
/// *and* its trailing-update work (`|contributors|·|rows|`) clears
/// `min_intra_work`. All four are knobs the tests and the A/B benchmark drive
/// directly to isolate one path — e.g. `min_level_work = 0` forces the across-
/// level (bit-identical) path, while `min_level_work = usize::MAX` with a small
/// `min_intra_contrib` and `min_intra_work = 0` forces the intra-column
/// (deterministic, to-rounding) one.
#[allow(clippy::too_many_arguments)]
fn solve_spd_block_inner(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
    rhs: &DMatrix<f64>,
    lambda: f64,
    threads: usize,
    min_level_work: usize,
    min_intra_contrib: usize,
    min_intra_work: usize,
) -> Result<DMatrix<f64>, ()> {
    match block_size {
        3 => solve_dispatch::<3>(
            triplets,
            dim,
            rhs,
            lambda,
            threads,
            min_level_work,
            min_intra_contrib,
            min_intra_work,
        ),
        6 => solve_dispatch::<6>(
            triplets,
            dim,
            rhs,
            lambda,
            threads,
            min_level_work,
            min_intra_contrib,
            min_intra_work,
        ),
        other => panic!("block_cholesky supports block sizes 3 and 6, got {other}"),
    }
}

/// Worker-thread budget for the numeric phase: the size of rayon's global pool
/// (which honors `RAYON_NUM_THREADS`, so setting it to `1` disables the
/// parallel path cleanly).
fn default_thread_count() -> usize {
    rayon::current_num_threads()
}

#[allow(clippy::too_many_arguments)]
fn solve_dispatch<const B: usize>(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    rhs: &DMatrix<f64>,
    lambda: f64,
    threads: usize,
    min_level_work: usize,
    min_intra_contrib: usize,
    min_intra_work: usize,
) -> Result<DMatrix<f64>, ()> {
    // Uncached: analyze the pattern then refactor+solve in one shot.
    let sym = analyze(triplets, dim, B);
    solve_with_symbolic::<B>(
        &sym,
        triplets,
        rhs,
        lambda,
        threads,
        min_level_work,
        min_intra_contrib,
        min_intra_work,
    )
}

/// The iteration-invariant structure of a block-Cholesky factorization: the
/// elimination-tree column patterns and the original block pattern, with no
/// numeric values. Computed once by [`analyze`] from a sparsity pattern and
/// reused to refactor every system that shares it — the Levenberg–Marquardt
/// normal equations keep one pattern across all iterations — so neither the
/// Gilbert–Ng–Peyton symbolic phase nor the COO→block assembly is redone on
/// each solve; only the block *values* are re-scattered and the numeric phase
/// re-run. (`block_size`-agnostic: every field is a combinatorial block index,
/// so one cache serves the `B = 3` and `B = 6` refactors alike.)
pub(crate) struct BlockSymbolic {
    /// Block size the pattern was analyzed for (3 or 6); a guard for reuse.
    block_size: usize,
    /// Number of block columns (`dim / block_size`).
    n: usize,
    /// `col_rows[j]`: sorted block rows present in column `j` of `L`, diagonal
    /// (`== j`) first.
    col_rows: Vec<Vec<usize>>,
    /// `contributors[j]`: the prior columns that fill into column `j`.
    contributors: Vec<Vec<usize>>,
    /// Elimination-tree levels — groups of mutually independent columns.
    levels: Vec<Vec<usize>>,
    /// `a_pattern[c]`: sorted block rows `≥ c` present in original block column
    /// `c` (diagonal included) — the slots a refactor scatters values into.
    a_pattern: Vec<Vec<usize>>,
}

/// Symbolic analysis: from the scalar COO sparsity pattern (values ignored)
/// build the [`BlockSymbolic`] shared by every refactor of a fixed-pattern
/// system. `dim` must be a multiple of `block_size` (3 or 6).
pub(crate) fn analyze(
    triplets: &[(usize, usize, f64)],
    dim: usize,
    block_size: usize,
) -> BlockSymbolic {
    assert!(
        dim % block_size == 0,
        "dim must be a multiple of the block size"
    );
    let n = dim / block_size;
    let (block_lower, a_pattern) = assemble_pattern(triplets, n, block_size);
    let (col_rows, contributors) = symbolic(&block_lower, n);
    let levels = build_levels(&contributors, n);
    BlockSymbolic {
        block_size,
        n,
        col_rows,
        contributors,
        levels,
        a_pattern,
    }
}

/// Refactor the system given by `triplets` (and `λ`) against a cached symbolic
/// structure, then solve every column of `rhs`. The thread/work knobs are the
/// same as [`solve_dispatch`].
#[allow(clippy::too_many_arguments)]
fn solve_with_symbolic<const B: usize>(
    sym: &BlockSymbolic,
    triplets: &[(usize, usize, f64)],
    rhs: &DMatrix<f64>,
    lambda: f64,
    threads: usize,
    min_level_work: usize,
    min_intra_contrib: usize,
    min_intra_work: usize,
) -> Result<DMatrix<f64>, ()> {
    debug_assert_eq!(sym.block_size, B);
    let (col_vals, diag_inv) = refactor_numeric::<B>(
        sym,
        triplets,
        lambda,
        threads,
        min_level_work,
        min_intra_contrib,
        min_intra_work,
    )?;
    let mut out = DMatrix::<f64>::zeros(sym.n * B, rhs.ncols());
    for c in 0..rhs.ncols() {
        let column = DVector::from_column_slice(rhs.column(c).as_slice());
        let x = solve_block_system::<B>(&sym.col_rows, &col_vals, &diag_inv, sym.n, &column);
        out.set_column(c, &x);
    }
    Ok(out)
}

/// Numeric phase: scatter the original block values from `triplets` into the
/// cached pattern (`sym.a_pattern`), fold in `λ`, and run the left-looking
/// block Cholesky over `sym`'s elimination-tree levels. Returns the factor's
/// `col_vals` and `diag_inv`. Bit-for-bit equivalent to factoring the same
/// `triplets` from scratch: duplicate triplets are summed in input order and the
/// diagonal `λ` is added afterwards, exactly as the one-shot path did.
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
fn refactor_numeric<const B: usize>(
    sym: &BlockSymbolic,
    triplets: &[(usize, usize, f64)],
    lambda: f64,
    threads: usize,
    min_level_work: usize,
    min_intra_contrib: usize,
    min_intra_work: usize,
) -> Result<(Vec<Vec<SMatrix<f64, B, B>>>, Vec<SMatrix<f64, B, B>>), ()> {
    let n = sym.n;
    let col_rows = &sym.col_rows;
    let contributors = &sym.contributors;

    // Original lower-triangular block values laid out on the cached pattern: a
    // direct scatter (binary search into the small per-column row list) replaces
    // the per-iteration `BTreeMap` assembly.
    let mut a_lower: Vec<Vec<(usize, SMatrix<f64, B, B>)>> = sym
        .a_pattern
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|&br| (br, SMatrix::<f64, B, B>::zeros()))
                .collect()
        })
        .collect();
    for &(r, c, v) in triplets {
        let (br, bc) = (r / B, c / B);
        if br >= bc {
            let slot = sym.a_pattern[bc]
                .binary_search(&br)
                .expect("triplet's block lies in the analyzed pattern");
            a_lower[bc][slot].1[(r % B, c % B)] += v;
        }
    }
    if lambda != 0.0 {
        for (bc, rows) in sym.a_pattern.iter().enumerate() {
            let slot = rows
                .binary_search(&bc)
                .expect("diagonal block lies in the analyzed pattern");
            for d in 0..B {
                a_lower[bc][slot].1[(d, d)] += lambda;
            }
        }
    }

    let mut col_vals: Vec<Vec<SMatrix<f64, B, B>>> = col_rows
        .iter()
        .map(|rows| vec![SMatrix::<f64, B, B>::zeros(); rows.len()])
        .collect();
    let mut diag_inv = vec![SMatrix::<f64, B, B>::zeros(); n];

    // Reusable relative-index map for the sequential path (see `factor_column`).
    let mut map = vec![0usize; n];

    for level in &sym.levels {
        let parallel = threads > 1
            && n >= PARALLEL_MIN_BLOCKS
            && level.len() >= PARALLEL_MIN_LEVEL_WIDTH
            && level
                .iter()
                .map(|&j| contributors[j].len() * col_rows[j].len())
                .sum::<usize>()
                >= min_level_work;
        if parallel {
            // Farm the level's independent columns out to the rayon pool, each
            // worker reading the (already-finalized) lower levels and writing
            // back after the parallel section.
            let computed =
                factor_level_parallel::<B>(level, n, col_rows, contributors, &a_lower, &col_vals)?;
            for (j, vals, inv) in computed {
                col_vals[j] = vals;
                diag_inv[j] = inv;
            }
        } else {
            // Narrow / light level: its heavy columns (the serial separator
            // chain) go parallel over their contributors, the rest run inline.
            // The intra path carries a fixed per-column overhead (a rayon
            // dispatch plus a per-chunk dense delta), so it only pays on columns
            // whose trailing-update work (`|contributors|·|rows|`) clears
            // `min_intra_work`; a high contributor count with few rows is cheap
            // enough to lose that race and is left inline. (Mirrors the
            // `min_level_work` bar on the across-level path.)
            let intra = threads > 1 && n >= PARALLEL_MIN_BLOCKS;
            for &j in level {
                let (vals, inv) = if intra
                    && contributors[j].len() >= min_intra_contrib
                    && contributors[j].len() * col_rows[j].len() >= min_intra_work
                {
                    factor_column_intra::<B>(j, col_rows, &contributors[j], &a_lower[j], &col_vals)?
                } else {
                    factor_column::<B>(
                        j,
                        col_rows,
                        &contributors[j],
                        &a_lower[j],
                        &col_vals,
                        &mut map,
                    )?
                };
                col_vals[j] = vals;
                diag_inv[j] = inv;
            }
        }
    }

    Ok((col_vals, diag_inv))
}

/// Solve `A x = b` for a single right-hand side via block forward and backward
/// substitution against a factor `L` given by its `col_rows` / `col_vals` and
/// the cached diagonal inverses `diag_inv`.
fn solve_block_system<const B: usize>(
    col_rows: &[Vec<usize>],
    col_vals: &[Vec<SMatrix<f64, B, B>>],
    diag_inv: &[SMatrix<f64, B, B>],
    n: usize,
    b: &DVector<f64>,
) -> DVector<f64> {
    // Gather the dense RHS into per-block sub-vectors.
    let mut y: Vec<SVector<f64, B>> = (0..n)
        .map(|j| SVector::<f64, B>::from_fn(|k, _| b[j * B + k]))
        .collect();

    // Forward substitution: solve L y = b, column by column.
    for j in 0..n {
        let yj = diag_inv[j] * y[j];
        y[j] = yj;
        // Below-diagonal rows (skip the diagonal slot 0); i > j, so the update
        // never aliases y[j].
        for (&i, block) in col_rows[j].iter().zip(&col_vals[j]).skip(1) {
            y[i] -= block * yj;
        }
    }

    // Backward substitution: solve Lᵀ x = y, columns in reverse.
    for j in (0..n).rev() {
        let mut acc = y[j];
        for (&i, block) in col_rows[j].iter().zip(&col_vals[j]).skip(1) {
            acc -= block.transpose() * y[i];
        }
        y[j] = diag_inv[j].transpose() * acc;
    }

    // Scatter back into a dense solution vector.
    let mut x = DVector::<f64>::zeros(n * B);
    for j in 0..n {
        for k in 0..B {
            x[j * B + k] = y[j][k];
        }
    }
    x
}

/// Left-looking factorization of a single block column `j`: gather column `j`'s
/// original blocks, subtract the trailing updates `Lᵢₖ·Lⱼₖᵀ` of every prior
/// contributor `k`, Cholesky the diagonal, and scale the below-diagonal rows.
/// Reads only already-finalized columns (`col_vals[k]`, `k` a descendant of
/// `j`) and writes nothing shared — it returns the column's value blocks (slot
/// `0` is `L_jj`) and `L_jj⁻¹`, so the caller can run it on a worker thread and
/// scatter the result afterwards. `map` is a reusable length-`n` scratch buffer
/// (seeded here for column `j`'s rows).
#[allow(clippy::type_complexity)]
fn factor_column<const B: usize>(
    j: usize,
    col_rows: &[Vec<usize>],
    contributors_j: &[usize],
    a_lower_j: &[(usize, SMatrix<f64, B, B>)],
    col_vals: &[Vec<SMatrix<f64, B, B>>],
    map: &mut [usize],
) -> Result<(Vec<SMatrix<f64, B, B>>, SMatrix<f64, B, B>), ()> {
    let rows = &col_rows[j];
    let m = rows.len();
    for (t, &i) in rows.iter().enumerate() {
        map[i] = t;
    }
    // Dense workspace for column j's blocks, indexed by position in `rows`,
    // seeded with the original `A` blocks of column j.
    let mut ws = vec![SMatrix::<f64, B, B>::zeros(); m];
    for &(i, block) in a_lower_j {
        ws[map[i]] = block;
    }

    for &k in contributors_j {
        let k_rows = &col_rows[k];
        let pj = pos(k_rows, j);
        let ljk_t = col_vals[k][pj].transpose();
        for t in pj..k_rows.len() {
            let i = k_rows[t];
            ws[map[i]] -= col_vals[k][t] * ljk_t;
        }
    }

    // Factor the (updated) diagonal block and record L_jj, L_jj⁻¹.
    let chol = ws[0].cholesky().ok_or(())?;
    let ljj = chol.l();
    let ljj_inv = ljj.try_inverse().ok_or(())?;

    // Lᵢⱼ = Yᵢ · (L_jjᵀ)⁻¹ = Yᵢ · (L_jj⁻¹)ᵀ for the below-diagonal rows.
    let ljj_inv_t = ljj_inv.transpose();
    let mut vals = vec![SMatrix::<f64, B, B>::zeros(); m];
    vals[0] = ljj;
    for t in 1..m {
        vals[t] = ws[t] * ljj_inv_t;
    }
    Ok((vals, ljj_inv))
}

/// Left-looking factorization of a single *heavy* block column `j`, parallel
/// over its contributors. Identical math to [`factor_column`], but the trailing
/// update `Y = A_j - Σ_k Lᵢₖ·Lⱼₖᵀ` — the dominant cost of a wide separator
/// column, with hundreds of contributors — is computed as a rayon reduction:
/// each fixed-size chunk of contributors accumulates a private dense delta over
/// the column's row pattern, and the chunk deltas are then summed in order.
/// This feeds the pool on exactly the columns the level scheme leaves serial
/// (the narrow separator chain near the root).
///
/// Because the contributor sum is regrouped into chunks, the result is *not*
/// bit-identical to the left-to-right sequential subtraction — floating-point
/// addition is not associative — but it is a valid Cholesky factor agreeing to
/// rounding (≈1e-12) and is *deterministic*: the chunking is keyed to the fixed
/// [`INTRA_CONTRIB_CHUNK`], not the thread count, and the chunk deltas are
/// folded in index order, so the same factor comes out on any number of
/// threads. (The across-level [`factor_level_parallel`] path stays exactly
/// bit-identical; only this heavy-column path trades that for filling the pool.)
#[allow(clippy::type_complexity)]
fn factor_column_intra<const B: usize>(
    j: usize,
    col_rows: &[Vec<usize>],
    contributors_j: &[usize],
    a_lower_j: &[(usize, SMatrix<f64, B, B>)],
    col_vals: &[Vec<SMatrix<f64, B, B>>],
) -> Result<(Vec<SMatrix<f64, B, B>>, SMatrix<f64, B, B>), ()> {
    use rayon::prelude::*;
    let rows = &col_rows[j];
    let m = rows.len();

    // Each chunk of contributors reduces into a private dense delta indexed by
    // position in `rows`; `collect` preserves chunk order so the fold below is
    // schedule-independent.
    let deltas: Vec<Vec<SMatrix<f64, B, B>>> = contributors_j
        .par_chunks(INTRA_CONTRIB_CHUNK)
        .map(|chunk| {
            let mut delta = vec![SMatrix::<f64, B, B>::zeros(); m];
            for &k in chunk {
                let k_rows = &col_rows[k];
                let pj = pos(k_rows, j);
                let ljk_t = col_vals[k][pj].transpose();
                for t in pj..k_rows.len() {
                    delta[pos(rows, k_rows[t])] += col_vals[k][t] * ljk_t;
                }
            }
            delta
        })
        .collect();

    // Y = A_j - Σ deltas, folded in chunk order.
    let mut ws = vec![SMatrix::<f64, B, B>::zeros(); m];
    for &(i, block) in a_lower_j {
        ws[pos(rows, i)] = block;
    }
    for delta in &deltas {
        for t in 0..m {
            ws[t] -= delta[t];
        }
    }

    let chol = ws[0].cholesky().ok_or(())?;
    let ljj = chol.l();
    let ljj_inv = ljj.try_inverse().ok_or(())?;
    let ljj_inv_t = ljj_inv.transpose();
    let mut vals = vec![SMatrix::<f64, B, B>::zeros(); m];
    vals[0] = ljj;
    for t in 1..m {
        vals[t] = ws[t] * ljj_inv_t;
    }
    Ok((vals, ljj_inv))
}

/// Factor every column of one independent level on the rayon pool. Each worker
/// reuses a private scratch `map` (via `map_init`) and emits its `(j, L column,
/// L_jj⁻¹)` triples; the shared `col_vals` is borrowed immutably for the whole
/// parallel section (only lower, finished levels are read), so no
/// synchronization is needed. The first non-SPD column short-circuits the
/// collect into `Err`.
#[allow(clippy::type_complexity)]
fn factor_level_parallel<const B: usize>(
    level: &[usize],
    n: usize,
    col_rows: &[Vec<usize>],
    contributors: &[Vec<usize>],
    a_lower: &[Vec<(usize, SMatrix<f64, B, B>)>],
    col_vals: &[Vec<SMatrix<f64, B, B>>],
) -> Result<Vec<(usize, Vec<SMatrix<f64, B, B>>, SMatrix<f64, B, B>)>, ()> {
    use rayon::prelude::*;
    level
        .par_iter()
        .map_init(
            || vec![0usize; n],
            |map, &j| {
                let (vals, inv) =
                    factor_column::<B>(j, col_rows, &contributors[j], &a_lower[j], col_vals, map)?;
                Ok((j, vals, inv))
            },
        )
        .collect()
}

/// Assign each column to an elimination-tree level — one more than the deepest
/// level among its contributors — and bucket the columns by level. Columns in
/// the same bucket are mutually independent (no contributor edge), so a bucket
/// can be factored in parallel. Contributors are strictly-earlier columns, so a
/// single forward sweep suffices and each bucket comes out sorted.
fn build_levels(contributors: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    let mut level = vec![0usize; n];
    let mut max_level = 0;
    for j in 0..n {
        let lv = contributors[j]
            .iter()
            .map(|&k| level[k] + 1)
            .max()
            .unwrap_or(0);
        level[j] = lv;
        max_level = max_level.max(lv);
    }
    let mut levels = vec![Vec::new(); max_level + 1];
    for (j, &lv) in level.iter().enumerate() {
        levels[lv].push(j);
    }
    levels
}

/// Block sparsity pattern from scalar COO triplets (values ignored). Returns
/// `(block_lower, a_pattern)` where `block_lower[i]` is the sorted set of block
/// columns `< i` coupled to block row `i` (what the symbolic phase walks), and
/// `a_pattern[c]` is the sorted set of block rows `≥ c` present in block column
/// `c`, the diagonal always included (the diagonal block is the one we
/// Cholesky-factor and the `λ`-damping target). A refactor scatters the original
/// block values into these `a_pattern` slots. The pattern is iteration-invariant,
/// so it lives in [`BlockSymbolic`] and the per-solve `BTreeMap` value assembly
/// the old `assemble_blocks` did is gone.
fn assemble_pattern(
    triplets: &[(usize, usize, f64)],
    n: usize,
    block_size: usize,
) -> (Vec<Vec<usize>>, Vec<Vec<usize>>) {
    let mut lower: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    let mut col_pat: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];

    for &(r, c, _) in triplets {
        let (br, bc) = (r / block_size, c / block_size);
        if br > bc {
            lower[br].insert(bc);
            col_pat[bc].insert(br);
        } else if br == bc {
            col_pat[bc].insert(br);
        }
        // br < bc (strict upper) is the transpose of a lower entry; dropped.
    }
    for (bc, set) in col_pat.iter_mut().enumerate() {
        set.insert(bc);
    }

    (
        lower.into_iter().map(|s| s.into_iter().collect()).collect(),
        col_pat
            .into_iter()
            .map(|s| s.into_iter().collect())
            .collect(),
    )
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

    /// `build_levels` must bucket exactly the columns with no contributor edge
    /// between them: a contributor sits at a strictly lower level, so same-level
    /// columns are safe to factor concurrently.
    #[test]
    fn build_levels_groups_independent_columns() {
        // 0,1 are leaves; 2 depends on {0,1}; 3 depends on {2}; 4 is another
        // leaf; 5 depends on {3,4}.
        let contributors = vec![vec![], vec![], vec![0, 1], vec![2], vec![], vec![3, 4]];
        let levels = build_levels(&contributors, 6);
        assert_eq!(levels, vec![vec![0, 1, 4], vec![2], vec![3], vec![5]]);
    }

    /// Build the diagonally-dominant SPD triplets of a `side × side` 2D-grid
    /// block graph at block size `b`, already in the fill-reducing order — the
    /// mesh regime whose elimination tree has wide independent levels.
    fn grid_system(side: usize, b: usize) -> (usize, Vec<(usize, usize, f64)>) {
        use crate::reordering::Reordering;
        let n = side * side;
        let dim = n * b;
        let id = |r: usize, c: usize| r * side + c;
        let mut rng = Rng(0x5eed);
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        let mut push_block = |br: usize, bc: usize, rng: &mut Rng| {
            for r in 0..b {
                for c in 0..b {
                    triplets.push((br * b + r, bc * b + c, 0.1 * rng.next_f64()));
                }
            }
        };
        for r in 0..side {
            for c in 0..side {
                if c + 1 < side {
                    push_block(id(r, c), id(r, c + 1), &mut rng);
                    push_block(id(r, c + 1), id(r, c), &mut rng);
                }
                if r + 1 < side {
                    push_block(id(r, c), id(r + 1, c), &mut rng);
                    push_block(id(r + 1, c), id(r, c), &mut rng);
                }
            }
        }
        for j in 0..n {
            for d in 0..b {
                triplets.push((j * b + d, j * b + d, 10.0));
            }
        }
        let order = Reordering::fill_reducing(dim, b, &triplets);
        (dim, order.permute_triplets(&triplets))
    }

    /// The multi-threaded numeric phase must produce a *bit-identical* factor to
    /// the sequential one: same columns, same arithmetic, only the scheduling
    /// differs. Forcing the parallel path on a cheap sparse grid
    /// (`min_level_work = 0`, so every width≥2 level is dispatched) exercises
    /// `factor_level_parallel` + the result scatter without paying the dense
    /// arithmetic it would take to clear the production work gate.
    #[test]
    fn parallel_factor_matches_sequential() {
        let side = 24; // 576 blocks > PARALLEL_MIN_BLOCKS, sparse ⇒ debug-fast
        let b = 6;
        let (dim, permuted) = grid_system(side, b);
        let n = dim / b;

        // The forced gate (work bar 0) must still find a wide level to dispatch,
        // or the parallel run would fall back to sequential and prove nothing.
        let sym = analyze(&permuted, dim, b);
        assert!(
            n >= PARALLEL_MIN_BLOCKS
                && sym
                    .levels
                    .iter()
                    .any(|l| l.len() >= PARALLEL_MIN_LEVEL_WIDTH),
            "fixture has no wide level to dispatch"
        );

        let rhs = DMatrix::<f64>::from_fn(dim, 2, |i, c| ((i + c) % 7) as f64 - 3.0);
        // Force the across-level path (`min_level_work = 0`) and disable the
        // intra-column path (`min_intra_contrib = usize::MAX`) so this isolates
        // the across-level schedule, whose only difference from sequential is
        // column order — hence bit-identical.
        let seq = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 1, 0, usize::MAX, 0)
            .expect("seq SPD");
        let par = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 8, 0, usize::MAX, 0)
            .expect("par SPD");
        assert_eq!(
            seq, par,
            "across-level parallel factor disagrees with the sequential factor (must be bit-identical)"
        );
    }

    /// The intra-column path ([`factor_column_intra`]) parallelizes a heavy
    /// column's contributor sum, regrouped into chunks, so unlike the across-
    /// level path it is *not* bit-identical to the sequential subtraction — but
    /// it must still be a valid factor (agreeing with the sequential solve to
    /// rounding) and *deterministic* (the same on any thread count). Forcing the
    /// sequential branch (`min_level_work = usize::MAX`, so no level goes out
    /// across its columns) routes every heavy column through `factor_column_intra`.
    #[test]
    fn intra_column_factor_matches_sequential() {
        let side = 32; // 1024 blocks > PARALLEL_MIN_BLOCKS, sparse ⇒ debug-fast
        let b = 6;
        let (dim, permuted) = grid_system(side, b);
        let n = dim / b;

        // The fixture must actually have a column heavy enough to take the
        // intra path, or the test would silently fall back to sequential.
        let sym = analyze(&permuted, dim, b);
        let max_contrib = sym.contributors.iter().map(|c| c.len()).max().unwrap_or(0);
        assert!(
            n >= PARALLEL_MIN_BLOCKS && max_contrib >= INTRA_MIN_CONTRIB,
            "fixture has no column heavy enough for the intra path (max contributors {max_contrib})"
        );

        let rhs = DMatrix::<f64>::from_fn(dim, 2, |i, c| ((i + c) % 7) as f64 - 3.0);
        // No level qualifies for the across path (`min_level_work = usize::MAX`),
        // so every heavy column drops into the intra-column path at the
        // production threshold.
        let force_seq = usize::MAX;
        let mic = INTRA_MIN_CONTRIB;
        let seq = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 1, force_seq, mic, 0)
            .expect("seq SPD");
        let par8 = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 8, force_seq, mic, 0)
            .expect("intra SPD");
        let par4 = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 4, force_seq, mic, 0)
            .expect("intra SPD");

        // Deterministic across thread counts (chunking is keyed to a constant,
        // not the pool size).
        assert_eq!(
            par8, par4,
            "intra factor must not depend on the thread count"
        );
        // A valid factor: the solve agrees with the sequential one to rounding,
        // even though the factor is not bit-identical.
        let rel = (&par8 - &seq).norm() / seq.norm().max(1.0);
        assert!(
            rel < 1e-9,
            "intra solve disagrees with the sequential solve: relative error {rel:e}"
        );
    }

    /// A cached symbolic factorization must, when reused to refactor a *new*
    /// value set of the same sparsity pattern, produce a bit-identical result to
    /// factoring that system from scratch — this is the contract the LM loop
    /// relies on. Two random SPD systems share a pattern (the `keep` predicate is
    /// deterministic, so the off-diagonal block pattern is identical; only the
    /// values differ); the cache is built on the first and reused on the second.
    #[test]
    fn cached_refactor_matches_one_shot() {
        let (n, b) = (20, 6);
        let dim = n * b;
        let lambda = 1e-3;
        let keep = |bi: usize, bj: usize| bi.abs_diff(bj) <= 1;
        let (_, t1) = random_spd(n, b, keep, 1);
        let (_, t2) = random_spd(n, b, keep, 2);
        let rhs = DMatrix::<f64>::from_fn(dim, 2, |i, c| ((i + c) % 5) as f64 - 2.0);

        let mut cache = None;
        // First solve analyzes and stores the symbolic structure.
        let c1 = solve_spd_block_cached(&mut cache, &t1, dim, b, &rhs, lambda).expect("c1");
        let o1 = solve_spd_block(&t1, dim, b, &rhs, lambda).expect("o1");
        assert_eq!(c1, o1, "first cached solve must match the one-shot factor");
        assert!(
            cache.is_some(),
            "symbolic structure must be cached after the first solve"
        );

        // Second solve reuses the cache with different values — the real test.
        let c2 = solve_spd_block_cached(&mut cache, &t2, dim, b, &rhs, lambda).expect("c2");
        let o2 = solve_spd_block(&t2, dim, b, &rhs, lambda).expect("o2");
        assert_eq!(
            c2, o2,
            "cached refactor with new values must match the one-shot factor bit-for-bit"
        );
    }

    /// A/B timing of the numeric phase: single-threaded vs. all-cores, on the
    /// `sphere2500`-scale 2D grid (the regime with the widest independent
    /// levels). Reports the achieved speedup. Run with `cargo test -p
    /// visloc-slam --release -- --ignored --nocapture bench_block_parallel`.
    #[test]
    #[ignore]
    fn bench_block_parallel_scaling() {
        use std::time::Instant;

        let b = 6;
        let side = 50; // 2500 blocks ~ sphere2500 scale
        let (dim, permuted) = grid_system(side, b);
        let rhs = DMatrix::<f64>::from_fn(dim, 1, |i, _| (i % 7) as f64 - 3.0);
        let threads = default_thread_count();
        let reps = 5;

        // Measure the full production parallel path: the across-level schedule
        // for the wide levels (`min_level_work = 0` forces every one onto the
        // pool) *and* the intra-column reduction for the grid's heavy width-1
        // separators (`min_intra_contrib = INTRA_MIN_CONTRIB`), which level
        // parallelism alone leaves serial. Warm up + cross-check the schedules
        // (only to rounding — the intra path is not bit-identical).
        let mic = INTRA_MIN_CONTRIB;
        let one =
            solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 1, 0, mic, 0).expect("seq SPD");
        let many = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, threads, 0, mic, 0)
            .expect("par SPD");
        assert!(
            (&one - &many).norm() / one.norm().max(1.0) < 1e-9,
            "parallel/sequential disagree"
        );

        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, 1, 0, mic, 0).unwrap();
        }
        let seq_ms = t0.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let t1 = Instant::now();
        for _ in 0..reps {
            let _ =
                solve_spd_block_inner(&permuted, dim, b, &rhs, 1e-3, threads, 0, mic, 0).unwrap();
        }
        let par_ms = t1.elapsed().as_secs_f64() * 1e3 / reps as f64;

        let sym = analyze(&permuted, dim, b);
        let col_rows = &sym.col_rows;
        let contributors = &sym.contributors;
        let levels = &sym.levels;
        let widths: Vec<usize> = levels.iter().map(|l| l.len()).collect();

        // Trailing-update block-multiply count per column ≈ the dominant factor
        // cost. Bucket the work by the *width* of the level the column sits in,
        // to show where the cost actually lives relative to where the
        // parallelism (wide levels) is.
        let col_work = |j: usize| -> u64 {
            contributors[j]
                .iter()
                .map(|&k| (col_rows[k].len() - pos(&col_rows[k], j)) as u64)
                .sum()
        };
        // Bucket the work by level width: parallelism only exists in wide
        // levels, so seeing the work concentrate in the narrow buckets is the
        // direct evidence that across-level parallelism cannot pay off here.
        let mut work_by_bucket = std::collections::BTreeMap::<usize, u64>::new();
        let mut total = 0u64;
        for level in levels {
            let w: u64 = level.iter().map(|&j| col_work(j)).sum();
            let bucket = match level.len() {
                1 => 1,
                2..=7 => 2,
                8..=31 => 8,
                32..=127 => 32,
                _ => 128,
            };
            *work_by_bucket.entry(bucket).or_default() += w;
            total += w;
        }
        let total = total.max(1);
        println!(
            "grid {side}x{side} (dim {dim}): seq {seq_ms:.1} ms vs {threads}-thread {par_ms:.1} ms => {:.2}x ({} levels, widest {})",
            seq_ms / par_ms,
            widths.len(),
            widths.iter().max().unwrap(),
        );
        let pct: Vec<String> = work_by_bucket
            .iter()
            .map(|(&b, &w)| format!("width>={b}: {:.1}%", 100.0 * w as f64 / total as f64))
            .collect();
        println!("  factor work by level width — {}", pct.join(", "));
    }

    #[test]
    fn non_spd_block_returns_err() {
        // A zero matrix is not positive-definite: the first diagonal block
        // Cholesky must fail rather than panic.
        let dim = 9;
        let rhs = DMatrix::<f64>::zeros(dim, 1);
        assert!(solve_spd_block(&[], dim, 3, &rhs, 0.0).is_err());
    }

    /// Group consecutive block columns into fundamental supernodes: column `j+1`
    /// joins `j`'s supernode iff its row pattern equals `j`'s with `j`'s own
    /// diagonal removed (so they share one dense panel below). Returns
    /// `(start, width)` per supernode. Used only by the measurement below — see
    /// the module's "Why not supernodal?" note for why amalgamation is not on
    /// the production path.
    fn detect_supernodes(col_rows: &[Vec<usize>]) -> Vec<(usize, usize)> {
        let n = col_rows.len();
        let mut out = Vec::new();
        let mut j = 0;
        while j < n {
            let start = j;
            while j + 1 < n
                && col_rows[j].len() == col_rows[j + 1].len() + 1
                && col_rows[j][1..] == col_rows[j + 1][..]
            {
                j += 1;
            }
            out.push((start, j - start + 1));
            j += 1;
        }
        out
    }

    /// Records the supernode-width distribution of the factor on a `sphere2500`-
    /// scale 2D-grid graph (the mesh-like regime that produces the *widest*
    /// panels) in the production fill-reducing order. The takeaway — most
    /// supernodes are width-1 leaves, with only a handful of wide separators — is
    /// the evidence behind the module's decision not to go supernodal. Run with
    /// `cargo test -p visloc-slam --release -- --ignored --nocapture
    /// supernode_width_distribution`.
    #[test]
    #[ignore]
    fn supernode_width_distribution() {
        use crate::reordering::Reordering;

        let b = 6;
        let side = 50;
        let n = side * side;
        let dim = n * b;
        let id = |r: usize, c: usize| r * side + c;
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        let mut push_block = |br: usize, bc: usize| {
            for r in 0..b {
                for c in 0..b {
                    triplets.push((br * b + r, bc * b + c, if r == c { 1.0 } else { 0.2 }));
                }
            }
        };
        for r in 0..side {
            for c in 0..side {
                if c + 1 < side {
                    push_block(id(r, c), id(r, c + 1));
                    push_block(id(r, c + 1), id(r, c));
                }
                if r + 1 < side {
                    push_block(id(r, c), id(r + 1, c));
                    push_block(id(r + 1, c), id(r, c));
                }
            }
        }
        for j in 0..n {
            for d in 0..b {
                triplets.push((j * b + d, j * b + d, 10.0));
            }
        }

        let order = Reordering::fill_reducing(dim, b, &triplets);
        let permuted = order.permute_triplets(&triplets);
        let sym = analyze(&permuted, dim, b);
        let col_rows = &sym.col_rows;

        let supers = detect_supernodes(col_rows);
        let widths: Vec<usize> = supers.iter().map(|&(_, w)| w).collect();
        let singletons = widths.iter().filter(|&&w| w == 1).count();
        let max_w = *widths.iter().max().unwrap();
        let total_cols: usize = widths.iter().sum();
        let in_wide: usize = widths.iter().filter(|&&w| w >= 4).copied().sum();
        let mut hist = std::collections::BTreeMap::<usize, usize>::new();
        for &w in &widths {
            let bucket = match w {
                1 => 1,
                2..=3 => 2,
                4..=7 => 4,
                8..=15 => 8,
                _ => 16,
            };
            *hist.entry(bucket).or_default() += 1;
        }
        println!(
            "grid {side}x{side} ({n} blocks): {} supernodes, {singletons} singletons, max width {max_w}, {:.1}% of columns in width>=4 panels",
            supers.len(),
            100.0 * in_wide as f64 / total_cols as f64,
        );
        println!("  width bucket (>=) -> supernode count: {hist:?}");
    }
}
