# Pose-graph optimizer internals

Engineering detail behind the pose-graph / bundle-adjustment back-end. The
README keeps the headline results and tables; this note holds the full
"how/why" prose that used to live inline there.

## Pose-graph optimization on the g2o benchmarks

visloc-rs ships a pure-Rust, deterministic SE(3) pose-graph optimizer
(`PoseGraph::optimize_se3_iterative` — iterative Gauss-Newton / Levenberg-
Marquardt on the SE(3) manifold, full 6x6 anisotropic information matrices,
robust kernels, dense or sparse Cholesky) plus `.g2o` `EDGE_SE3:QUAT` read/write
([`read_g2o`](../pipelines/slam/src/g2o.rs)), so it runs directly on the canonical
pose-graph datasets the SLAM back-end literature reports on — no C++, no Ceres,
no ROS. For graphs with **wrong** loop closures it also ships
`PoseGraph::optimize_se3_gnc`, a Graduated Non-Convexity solver (Yang et al.
2020, Geman-McClure / truncated-least-squares surrogates) that rejects outlier
constraints a plain Huber/Cauchy IRLS solve cannot. The same GNC machinery is
available for bundle adjustment as `BundleAdjustment::optimize_gnc`, rejecting
wrong feature correspondences per observation.

| Dataset | Poses | Edges | initial chi^2 | final chi^2 | reduction | solve |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `parking-garage` | 1661 | 6275 | 1.67e4 | 1.27e0 | **99.99 %** | ~0.3 s |
| `sphere2500` | 2500 | 4949 | 2.61e6 | 1.35e3 | **99.95 %** | ~1.8 s |
| `torus3D` | 5000 | 9048 | 4.80e6 | 6.00e4 | **98.75 %** | ~9.7 s |
| `cubicle` | 5750 | 16869 | 1.08e7 | 2.75e3 | **99.97 %** | ~10 s |
| `rim` | 10195 | 29743 | 1.28e8 | 8.73e7 | 31.58 % | ~15 s |
| built-in synthetic loop | 120 | 120 | 2.68e2 | ~1e-21 | **100 %** | <0.2 s |

(`solve` is the SE(3) optimization from raw odometry, no chordal seeding; the
chordal init below converges the hard 3D graphs to a far lower chi^2 in even
less time.)

The fill-reducing reordering is what makes these tractable at all — solved in the
natural variable order, the Cholesky factor fills in catastrophically. For each
graph it picks the cheaper of a Reverse Cuthill-McKee (band-minimizing) and a
nested-dissection (separator) ordering by symbolic Cholesky factor size, and —
since the sparsity pattern is identical across iterations — computes that
ordering just once. RCM wins on the near-banded `parking-garage` corridor;
nested dissection wins on the wide 3D meshes, taking `torus3D` from *no
convergence within minutes* to seconds.

The dense ICP graphs `cubicle` and `rim` defeat *both* geometric orderings — the
factor blows up past the dense-matrix size, so even *counting* it dominates the
solve. A **minimum-degree** rescue ordering (the local heuristic behind
AMD/SuiteSparse) is held in reserve for exactly this case: the symbolic count is
capped at a small multiple of the minimum-degree factor, so a blown-up geometric
ordering is abandoned cheaply and the rescue ordering is adopted, taking
`cubicle` from a >10-minute timeout to ~10 s. It is *only* used on that
catastrophic blow-up — minimum degree's factor, though it has fewer nonzeros,
factorizes more slowly than a balanced geometric ordering (its elimination tree
is deeper and less cache-friendly), so it never second-guesses a healthy
ordering (e.g. it leaves `torus3D` on nested dissection). The minimum-degree
pivot is selected with a lazy binary heap rather than an `O(n^2)` linear scan.

## Block Cholesky factorization

The pose-graph normal matrix is not an arbitrary sparse matrix: every variable
is a fixed-size block (`6x6` for an SE(3) pose, `3x3` for a rotation column or a
translation center), and an edge couples two *blocks*, never two stray scalars.
A scalar sparse Cholesky (such as `nalgebra_sparse`'s `CscCholesky`) ignores
that and factors one scalar column at a time, paying the sparse gather/scatter
bookkeeping `b^2` times per block. visloc-rs factors at block granularity
instead: a left-looking Cholesky over the block elimination tree whose "scalars"
are stack-allocated `BxB` matrices, so each diagonal factorization, triangular
solve, and trailing update is a single dense `nalgebra` kernel the compiler
unrolls and vectorizes (the block size is a const generic, fully monomorphized
for `B = 3` and `B = 6`). The numeric phase scatters each column's trailing
updates through a precomputed relative-index map rather than a `binary_search`
per touched block, so on the same fill-reducing order this is **~5.6x** faster
than scalar `CscCholesky` on a `sphere2500`-scale `6x6` system in isolation, and
**~3-4x** faster end-to-end at bit-equivalent solutions:

| Dataset | scalar `CscCholesky` | **block Cholesky** | speedup |
| --- | ---: | ---: | ---: |
| `parking-garage` | ~0.9 s | **~0.3 s** | ~2.9x |
| `sphere2500` | ~6.8 s | **~1.8 s** | ~3.7x |
| `torus3D` | ~42 s | **~9.7 s** | ~4.3x |
| `cubicle` | ~34 s | **~10 s** | ~3.4x |
| `rim` | ~55 s | **~15 s** | ~3.8x |

The same back-end factors the **bundle-adjustment** Schur complement. After the
landmarks are eliminated, the reduced camera system is itself block-structured
(`6x6` pose blocks), so visual BA routes it through the identical `B = 6` block
Cholesky rather than scalar `CscCholesky` — bit-comparable to the dense solve
(an integration test cross-checks `Sparse` against `Dense`) and ~1.4x faster
end-to-end on a covisibility-dense 120-keyframe synthetic scene (the reduced
factorization is one of several costs per iteration, so the gain tracks its
share). Visual-inertial systems interleave `3`-DOF velocity blocks that break
the uniform `6x6` tiling and keep the scalar factorization.

Across the iterations of one optimization the normal matrix changes values but
never its sparsity, so the solver splits the classic way: the **symbolic
analysis** (elimination tree, per-column fill patterns, levels) and the COO->block
pattern assembly are computed once and cached alongside the fill-reducing order,
and every subsequent iteration only re-scatters the block values and re-runs the
numeric factorization. On the g2o benchmarks that pattern work was ~20-30% of each
solve, so caching it is a clean ~1.1x (and more on the small chain graphs, whose
numeric phase is tiny next to the per-iteration assembly) at an identical result.

The numeric phase is also parallelized across the block elimination tree: columns
are grouped into independent *levels* (a column depends only on its descendants,
which sit at strictly lower levels), and each sufficiently heavy level is factored
on a `rayon` pool while the finished lower levels are read — a topological
reordering of the sequential sweep, so the factor stays bit-identical (and disabled
cleanly with `RAYON_NUM_THREADS=1`). Across-level parallelism alone is bounded by
the tree shape — the work concentrates in the narrow separator levels near the root
while the wide levels are cheap leaves, leaving the width-1 separator chain serial.
That chain is attacked by a second, orthogonal axis: a heavy separator column's
trailing update is a sum over its hundreds of (already-finished) contributors, so
when a column stays off the level path *and is heavy enough to amortize the rayon
dispatch* it is instead factored by reducing that sum across the pool. This is
pure-Rust *intra-separator* parallelism — it splits the left-looking updates across
contributors, not a dense panel across cores, so unlike the supernodal/BLAS-3 route
it needs no tuned BLAS (it trades exact bit-identity for a deterministic,
agrees-to-rounding factor). Both axes are gated on *work*, not just shape: a per-level
bar keeps cheap-but-wide leaf levels (`parking-garage`) off the parallel path, and a
per-column bar keeps the light separators of the dense 3D graphs (`cubicle`,
`sphere2500`) inline — a bare contributor-count gate regressed them, since they clear
the count but do too little arithmetic per dispatch. Together the two axes reach ~1.4x
end-to-end on `torus3D` and ~1.26x on `rim` (up from ~1.17x / ~1.09x with across-level
alone), with the work gate adding a further ~5 % on those two and keeping every other
graph neutral — never a regression.

The loader is also robust to the malformed information matrices that real
scan-matching datasets ship: `cubicle` and `rim` contain edges whose `Omega` is
not positive-semidefinite (a rotation sub-block with off-diagonal entries
dwarfing its diagonal, eigenvalues down to ~-6e6), which would make the
Gauss-Newton `H` indefinite and the Cholesky factorization fail outright.
`read_g2o` projects every information matrix onto the PSD cone (clamping
negative eigenvalues to zero) on load, which is the exact identity on a valid
matrix, so these datasets optimize instead of aborting. `cubicle` then drives
down cleanly (**99.97 %**); `rim`, started from raw odometry, is genuinely
harder — LM makes early progress then stalls (its damping saturates), reaching
only **31.58 %**. That stall is a *basin* problem, not a solver bug, and a
chordal rotation initialization fixes it (next).

## Chordal rotation initialization (`--chordal-init`)

On strongly non-convex 3D graphs the SE(3) cost surface has deep local minima in
rotation, so Levenberg-Marquardt started from odometry settles into a poor basin
and stalls. Seeding it with a **chordal rotation initialization** (Carlone et
al., *Initialization Techniques for 3D SLAM*, ICRA 2015) lands it near the global
optimum. `PoseGraph::initialize_rotations_chordal` relaxes every rotation from
`SO(3)` to an unconstrained `3x3` matrix, minimizing the Frobenius residual
`sum_e w_e * ||R_to - R_meas * R_from||^2` as one linear least-squares problem;
because the relaxation decouples by rotation column, the per-node `9`-vector
splits into three `3`-vector systems that share *one* `3n x 3n` normal matrix
(factored once, solved for three right-hand sides), and each relaxed block is
projected back onto `SO(3)` with an SVD. Translations are then re-derived by the
existing linear translation solve before the full SE(3) run.

`optimize_se3_iterative` runs this seeding **by default** (`PoseGraphSe3Config {
chordal_init: true, .. }`): the rotation optimum is a fixed point of the
relaxation, so on an already-consistent graph it leaves the estimate essentially
unchanged (a cheap extra factorization) while rescuing the hard ones, and the
step is best-effort (a singular relaxation is silently skipped) so it can never
turn a solvable problem into a failure. The `pgo_g2o_benchmark` example disables
the in-solver default and drives the step manually behind `--chordal-init`, so
its before/after chi^2 below stays a clean, independently-measured comparison.

The effect is a uniform win on the hard 3D graphs — never a worse final chi^2,
always equal-or-faster, and it flips three datasets (`sphere2500`, `cubicle`,
`rim`) from non-converged to converged (times below use the block Cholesky
throughout):

| Dataset | LM from odometry (final chi^2, time) | **+chordal init** (final chi^2, time) |
| --- | --- | --- |
| `sphere2500` | 1.35e3, ~1.8 s | 1.35e3, **~1.1 s** (converges) |
| `torus3D` | 6.00e4, ~9.7 s | **2.42e4**, **~5.5 s** |
| `cubicle` | 2.75e3, ~10 s | 2.75e3, **~4.0 s** (converges) |
| `rim` | 8.73e7, ~15 s (31.6 %) | **8.34e4**, **~10 s** (**99.9 %**, converges) |

`rim`'s final chi^2 drops by ~1050x and it finally converges; `torus3D` reaches
a ~2.5x lower optimum in ~2x less wall-clock (and converges too under the
Marquardt `--diag-damping` mode). `sphere2500` and `cubicle` already bottom out
at their optimum from odometry, so chordal init does not lower their chi^2 — it
converges them and roughly halves the wall-clock. The chordal solve itself is
cheap (~0.4 s on `torus3D`, ~0.8 s on `rim`). `parking-garage` is already trivial
from odometry, so the init leaves it unchanged — no regression.

Reproduce (the fetch script pulls the standard SE-Sync dataset suite —
`sphere2500`, `torus3D`, `parking-garage`, `cubicle`, `grid3D`, `rim`):

```sh
scripts/fetch_pgo_g2o_datasets.sh datasets/pgo_g2o
cargo run --release --example pgo_g2o_benchmark -- datasets/pgo_g2o/parking-garage.g2o
# hard 3D graphs: seed LM with a chordal rotation initialization
cargo run --release --example pgo_g2o_benchmark -- --chordal-init datasets/pgo_g2o/rim.g2o
# or, zero-setup, the built-in deterministic loop graph:
cargo run --example pgo_g2o_benchmark
```

### GTSAM parity

chi^2 is the sum of Mahalanobis edge residuals. visloc-rs stores each vertex as a
world-frame pose and each measurement inverted, so its residual is the g2o/GTSAM
body-frame residual rotated by the measurement adjoint; `read_g2o` therefore
carries every information matrix through the matching congruence
(`Ad(Z^-1)^T Omega Ad(Z^-1)`) so the weighted cost it minimizes is *identical* to
the g2o/GTSAM `Sum e^T Omega e` — not merely a monotone surrogate. (Skip that
congruence and an anisotropic graph like `sphere2500`, whose information weights
translation ~40x below rotation, minimizes an adjoint-twisted cost and lands on a
different optimum — the bug this fixes.) Seeded from the same raw odometry,
visloc-rs lands on the same optimum as GTSAM 4.x's Levenberg-Marquardt:

| Dataset | init chi^2 (both) | visloc-rs final | GTSAM final |
| --- | ---: | ---: | ---: |
| `parking-garage` | 1.6727e4 | 1.2684e0 | 1.2684e0 |
| `sphere2500` | 2.6113e6 | 1.3515e3 | 1.3515e3 |
| `cubicle` | 1.0811e7 | 2.7527e3 | 2.7493e3 |
| `torus3D` | 4.8012e6 | 6.000e4 | 5.996e4 |
| `rim` | 1.2766e8 | 8.73e7 | 6.11e5 |

The initial chi^2 matches GTSAM to every printed digit — a direct check that the
two solvers minimize the same function. From raw odometry both tie on the
easy/anisotropic graphs (`parking`, `sphere`, `cubicle` within 0.12 %); the
strongly non-convex `torus3D`/`rim` need a rotation seed. With `--chordal-init`
visloc-rs reaches **2.42e4** (`torus3D`) and **8.34e4** (`rim`) — below GTSAM's
raw-LM optimum — and since GTSAM's own `InitializePose3` chordal seed throws an
indeterminate-linear-system exception on `sphere`/`cubicle`/`rim`, visloc-rs
converges graphs from a seed the reference implementation cannot.
(`scripts/gtsam_pgo_benchmark.py` reproduces the GTSAM column.)

## Outlier-robust PGO (Graduated Non-Convexity)

Real loop closures are sometimes **wrong** (perceptual aliasing, place-recognition
false positives), and a single bad one pulls a least-squares solve — or even a
Huber/Cauchy IRLS solve, whose influence function is non-convex so it only finds
a local minimum — into a corrupted basin. `PoseGraph::optimize_se3_gnc` adds
**Graduated Non-Convexity** (Yang et al. 2020, the engine behind Kimera-RPGO and
TEASER++): it anneals a control parameter from a convex surrogate that trusts
every edge toward the true robust cost that rejects outliers, so it is shepherded
into the inlier basin *before* the cost turns non-convex. Two surrogates ship —
Geman-McClure (smooth) and truncated-least-squares (hard inlier/outlier verdict).

Injecting random wrong loop closures into a real `.g2o` graph (the standard
robust-PGO protocol) and measuring the chi^2 over the **original** edges only —
the cost a solver should recover if it rejects the outliers —
([`examples/pgo_g2o_robust_benchmark.rs`](../examples/pgo_g2o_robust_benchmark.rs)):

| Graph (+ injected outliers) | L2 | Huber | GNC-GM | GNC-TLS |
| --- | ---: | ---: | ---: | ---: |
| `sphere2500` + 30 (`c=3`) | 89.1× | 51.0× | **1.0×** (30/30, 0 FP) | **1.0×** (30/30, 0 FP) |
| `torus3D` + 40 (`c=6`) | 5.7× | 1.9× | **1.0×** (40/40, 19 FP) | **1.0×** (40/40, 0 FP) |

Numbers are the inlier-edge chi^2 as a multiple of the outlier-free baseline
(`1.0×` = full recovery); `(recall, false positives)` is the outlier
classification from the final per-edge GNC weights. L2 and Huber are badly
corrupted (5-89× the baseline); both GNC variants recover the outlier-free
solution and reject every injected outlier. `c` is the inlier residual scale and
must match the graph's noise level — `sphere2500`'s residuals are ~8× tighter
than `torus3D`'s, so a `c` that is perfect on one over-rejects on the other.

That coupling is exactly what `GncConfig::auto_scale` removes: instead of a
hand-set `c`, it estimates the inlier scale from the residual distribution at
the start of the solve — the Iglewicz-Hoaglin robust cutoff
`median(ρ) + k·1.4826·MAD(ρ)` on the residual norms `ρ` (breakdown-robust to
~50 % outliers; `--auto-c` in the benchmark). It adapts per graph with no tuning:
on `sphere2500` it picks `c ≈ 16` and recovers exactly (`30/30`, 0 FP); on
`torus3D` it picks `c ≈ 10` and matches the *best* fixed `c` available at that
seed (`torus3D` is c-insensitive across `c ∈ [6, 10]` — its residual floor is an
intrinsic hard-graph property, not a scale-tuning gap). The same `auto_scale`
field drives the bundle-adjustment GNC solver. Truncated-least-squares is the
more decisive kernel: its hard verdict drives false positives to zero where the
smooth Geman-McClure leaves a few borderline edges down-weighted.

## Outlier-robust bundle adjustment (Graduated Non-Convexity)

The same failure mode bites bundle adjustment: a cluster of **wrong feature
correspondences** (a repeated texture, a mistracked point, a wrong temporal
match during VO chaining) that the initialisation already believes can capture
the reconstruction, and a Huber/Cauchy IRLS solve only down-weights them
*locally*. `BundleAdjustment::optimize_gnc` applies the same Graduated
Non-Convexity machinery per **observation** instead of per edge: each
reprojection residual gets a Black-Rangarajan weight `w ∈ [0,1]` that is folded
into the Schur-complement normal equations, and the control parameter is annealed
from a convex (least-squares) surrogate toward the true robust cost. Only
reprojections are reweighted — structural and inertial priors (gravity /
position / pairwise-pose / IMU) are never switched off — so a wrong
correspondence is the only thing GNC can reject. `gnc.c` is the inlier
reprojection scale **in pixels** (or set `gnc.auto_scale` to estimate it from
the residual MAD, as for the pose-graph solver).

Validated on a rigid, fully-observed scene with injected outlier observations
(wrong correspondences shifted 70-90 px;
[`pipelines/slam/tests/gnc_robust_ba.rs`](../pipelines/slam/tests/gnc_robust_ba.rs)):
Geman-McClure drives **every** injected outlier to a vanishing weight (perfect
recall), separated from the lightest inlier by orders of magnitude; truncated-
least-squares additionally gives the hard 0/1 verdict — outliers at machine-zero,
inliers kept at exactly `1.0` — and recovers truth near-exactly (`< 1e-4` pose-
centre error) where a plain least-squares solve is dragged off by the same
outliers. As in the pose-graph case TLS is the kernel to reach for: keeping
inliers fully weighted preserves the weakly-observable monocular depth direction
that Geman-McClure's fractional weights loosen.
