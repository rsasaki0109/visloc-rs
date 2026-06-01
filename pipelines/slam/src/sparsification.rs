//! Chow-Liu sparsification of a dense Gaussian marginalization prior.
//!
//! Marginalizing a state out of a graph (see [`crate::marginalization`]) is exact
//! but *densifying*: the Schur complement fills in a fully-connected clique among
//! the marginalized state's Markov blanket. Re-adding that as a dense
//! `GaussianPrior` couples every blanket pose to every other, so a sliding window
//! that marginalizes repeatedly accumulates dense priors and loses the sparsity
//! the whole sparse-Cholesky stack relies on.
//!
//! **Chow-Liu** (1968) gives the information-theoretically optimal way to keep it
//! sparse: among all *tree*-structured Gaussians, the one closest to the dense
//! prior in KL divergence is the maximum-weight spanning tree whose edge weights
//! are the pairwise **mutual informations**. Concretely, for the dense prior
//! `N(μ, Σ)` with `Σ = Λ⁻¹`, block `i` (one pose) and block `j`:
//!
//! ```text
//!   I(i,j) = ½ ( ln|Σ_ii| + ln|Σ_jj| − ln|Σ_{ij}| )
//! ```
//!
//! (`Σ_{ij}` the 2-block joint covariance). Build the max-spanning-tree `T` over
//! those weights; the tree-structured Gaussian that preserves every node marginal
//! and every tree-edge pairwise marginal is, in information form,
//!
//! ```text
//!   Λ_tree = Σ_{(i,j)∈T} J_ij − Σ_i (deg_i − 1) Λ_i      (embedded into the full dim)
//! ```
//!
//! where `J_ij = Σ_{ij}⁻¹` is the pair's joint information and `Λ_i = Σ_ii⁻¹` the
//! node's marginal information — the standard junction-tree factorization
//! `∏_{edges} p(x_i,x_j) / ∏_i p(x_i)^{deg_i−1}` written in information form. The
//! mean is preserved by `η_tree = Λ_tree μ`. The result couples only tree-adjacent
//! poses, so the prior is sparse (`N−1` off-diagonal blocks instead of `N(N−1)/2`)
//! while remaining a consistent — never over-confident — approximation that is
//! *exact* whenever the dense prior was already tree-structured.
//!
//! Pure linear algebra on a dense [`nalgebra::DMatrix`], no [`crate::PoseGraph`]
//! dependency, mirroring [`crate::marginalization`] / [`crate::covariance`].

use nalgebra::{DMatrix, DVector};

/// A sparsified prior: the tree-structured information system plus the node-pair
/// edges (block indices) that carry its off-diagonal coupling.
#[derive(Debug, Clone, PartialEq)]
pub struct SparsifiedPrior {
    /// Tree-structured information matrix (same dimension as the dense input).
    pub lambda: DMatrix<f64>,
    /// Tree-structured information vector (preserves the dense prior's mean).
    pub eta: DVector<f64>,
    /// Spanning-tree edges as `(i, j)` block indices (`i < j`), `N − 1` of them.
    pub edges: Vec<(usize, usize)>,
}

/// Approximate the dense Gaussian prior `(lambda, eta)` — `N` stacked `block_dim`
/// blocks (one per pose) — with the KL-optimal **tree-structured** Gaussian
/// (Chow-Liu). Preserves every node marginal and every tree-edge pairwise
/// marginal exactly; off-tree couplings are dropped.
///
/// Returns `None` if shapes disagree, `block_dim == 0`, the dimension is not a
/// multiple of `block_dim`, or `lambda` is not invertible (a degenerate prior).
/// With `N ≤ 1` block the input is already maximally sparse and is returned
/// unchanged (no edges).
pub fn sparsify_chow_liu(
    lambda: &DMatrix<f64>,
    eta: &DVector<f64>,
    block_dim: usize,
) -> Option<SparsifiedPrior> {
    let n = lambda.nrows();
    if block_dim == 0 || lambda.ncols() != n || eta.len() != n || n % block_dim != 0 {
        return None;
    }
    let num_nodes = n / block_dim;

    // Dense covariance and mean of the prior (the prior block is small — it is the
    // Markov blanket of one marginalized state — so a dense inverse is cheap).
    let chol = lambda.clone().cholesky()?;
    let sigma = chol.inverse();
    let mu = chol.solve(eta);

    if num_nodes <= 1 {
        return Some(SparsifiedPrior {
            lambda: lambda.clone(),
            eta: eta.clone(),
            edges: Vec::new(),
        });
    }

    let block = |m: &DMatrix<f64>, i: usize, j: usize| -> DMatrix<f64> {
        m.view((i * block_dim, j * block_dim), (block_dim, block_dim))
            .into_owned()
    };

    // Per-node marginal covariance log-determinants (for the mutual informations).
    let node_logdet: Vec<f64> = (0..num_nodes)
        .map(|i| logdet_spd(&block(&sigma, i, i)))
        .collect::<Option<Vec<_>>>()?;

    // Pairwise mutual informations I(i,j) = ½(ln|Σ_ii| + ln|Σ_jj| − ln|Σ_joint|).
    // A higher value means a stronger coupling worth keeping in the tree.
    let mut weighted_edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..num_nodes {
        for j in (i + 1)..num_nodes {
            let joint = joint_block(&sigma, i, j, block_dim);
            let Some(joint_logdet) = logdet_spd(&joint) else {
                continue;
            };
            let mi = 0.5 * (node_logdet[i] + node_logdet[j] - joint_logdet);
            weighted_edges.push((mi, i, j));
        }
    }

    // Maximum-weight spanning tree (Kruskal): the Chow-Liu tree.
    weighted_edges.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut dsu = DisjointSet::new(num_nodes);
    let mut edges: Vec<(usize, usize)> = Vec::with_capacity(num_nodes - 1);
    for &(_, i, j) in &weighted_edges {
        if dsu.union(i, j) {
            edges.push((i, j));
            if edges.len() == num_nodes - 1 {
                break;
            }
        }
    }
    // The prior is connected (it is the clique of one marginalized state's
    // blanket), so a spanning tree always has exactly N−1 edges; bail otherwise.
    if edges.len() != num_nodes - 1 {
        return None;
    }

    // Assemble Λ_tree = Σ_{(i,j)} J_ij − Σ_i (deg_i − 1) Λ_i, embedded block-wise.
    let mut degree = vec![0usize; num_nodes];
    for &(i, j) in &edges {
        degree[i] += 1;
        degree[j] += 1;
    }
    let mut lambda_tree = DMatrix::<f64>::zeros(n, n);
    for &(i, j) in &edges {
        let joint = joint_block(&sigma, i, j, block_dim);
        let j_info = joint.cholesky()?.inverse(); // 2b×2b joint information
                                                  // Scatter the four sub-blocks into (i,i),(i,j),(j,i),(j,j).
        add_block(&mut lambda_tree, i, i, &sub(&j_info, 0, 0, block_dim), 1.0);
        add_block(&mut lambda_tree, i, j, &sub(&j_info, 0, 1, block_dim), 1.0);
        add_block(&mut lambda_tree, j, i, &sub(&j_info, 1, 0, block_dim), 1.0);
        add_block(&mut lambda_tree, j, j, &sub(&j_info, 1, 1, block_dim), 1.0);
    }
    for (i, &deg) in degree.iter().enumerate() {
        if deg > 1 {
            let lambda_i = block(&sigma, i, i).cholesky()?.inverse();
            add_block(&mut lambda_tree, i, i, &lambda_i, -((deg - 1) as f64));
        }
    }
    // Symmetrize against round-off.
    lambda_tree = 0.5 * (&lambda_tree + lambda_tree.transpose());
    // Preserve the mean exactly: η_tree = Λ_tree μ.
    let eta_tree = &lambda_tree * &mu;

    Some(SparsifiedPrior {
        lambda: lambda_tree,
        eta: eta_tree,
        edges,
    })
}

/// Approximate the dense prior with the maximum-entropy **block-diagonal**
/// Gaussian: keep every node's marginal covariance exactly but assume the nodes
/// are mutually **independent** — drop every cross-block correlation. This is the
/// naive "drop the correlations" sparsifier and the zero-edge contestant against
/// the Chow-Liu tree: same per-node marginal guarantee, but `0` off-diagonal
/// blocks instead of `N−1`. Being the empty-graph member of the tree family, its
/// KL to the dense prior is an **upper bound** on Chow-Liu's (which optimizes over
/// all trees, the empty graph included), so a head-to-head [`kl_divergence`]
/// quantifies exactly what the tree's `N−1` retained couplings buy.
///
/// Each block is `Λ_ii := (Σ_ii)⁻¹` (the node's *marginal* information, not the
/// dense conditional block `Λ`'s diagonal), off-diagonals zero, mean preserved via
/// `η = Λ_diag μ`. Returns `None` on the same shape / non-SPD failures as
/// [`sparsify_chow_liu`].
pub fn sparsify_diagonal(
    lambda: &DMatrix<f64>,
    eta: &DVector<f64>,
    block_dim: usize,
) -> Option<SparsifiedPrior> {
    let n = lambda.nrows();
    if block_dim == 0 || lambda.ncols() != n || eta.len() != n || n % block_dim != 0 {
        return None;
    }
    let num_nodes = n / block_dim;

    let chol = lambda.clone().cholesky()?;
    let sigma = chol.inverse();
    let mu = chol.solve(eta);

    let mut lambda_diag = DMatrix::<f64>::zeros(n, n);
    for i in 0..num_nodes {
        let sigma_ii = sigma
            .view((i * block_dim, i * block_dim), (block_dim, block_dim))
            .into_owned();
        let lambda_ii = sigma_ii.cholesky()?.inverse();
        add_block(&mut lambda_diag, i, i, &lambda_ii, 1.0);
    }
    // Symmetrize against round-off and preserve the mean exactly.
    lambda_diag = 0.5 * (&lambda_diag + lambda_diag.transpose());
    let eta_diag = &lambda_diag * &mu;

    Some(SparsifiedPrior {
        lambda: lambda_diag,
        eta: eta_diag,
        edges: Vec::new(),
    })
}

/// KL divergence `D( N(μ, Λp⁻¹) ‖ N(μ, Λq⁻¹) )` between two Gaussians given in
/// **information form** that share a mean (every sparsifier here preserves `μ`),
/// so the mean term vanishes:
///
/// ```text
///   ½ ( tr(Λq Σp) − n + ln|Λp| − ln|Λq| ),   Σp = Λp⁻¹.
/// ```
///
/// The standard sparsification-quality metric: how far the approximation `q` sits
/// from the exact dense prior `p` — `0` iff identical, larger is worse. Pass the
/// dense prior as `p` and a sparsifier's `lambda` as `q` to score it. `None` if
/// either matrix is not SPD or the dimensions disagree.
pub fn kl_divergence(p_lambda: &DMatrix<f64>, q_lambda: &DMatrix<f64>) -> Option<f64> {
    let n = p_lambda.nrows();
    if p_lambda.ncols() != n || q_lambda.nrows() != n || q_lambda.ncols() != n {
        return None;
    }
    let sigma_p = p_lambda.clone().cholesky()?.inverse();
    let tr = (q_lambda * &sigma_p).trace();
    Some(0.5 * (tr - n as f64 + logdet_spd(p_lambda)? - logdet_spd(q_lambda)?))
}

/// The `2·block_dim × 2·block_dim` joint covariance of nodes `i` and `j`.
fn joint_block(sigma: &DMatrix<f64>, i: usize, j: usize, b: usize) -> DMatrix<f64> {
    let mut joint = DMatrix::<f64>::zeros(2 * b, 2 * b);
    let put = |dst: &mut DMatrix<f64>, br: usize, bc: usize, si: usize, sj: usize| {
        for r in 0..b {
            for c in 0..b {
                dst[(br * b + r, bc * b + c)] = sigma[(si * b + r, sj * b + c)];
            }
        }
    };
    put(&mut joint, 0, 0, i, i);
    put(&mut joint, 0, 1, i, j);
    put(&mut joint, 1, 0, j, i);
    put(&mut joint, 1, 1, j, j);
    joint
}

/// Extract the `(br, bc)` `b×b` sub-block of a `2b×2b` matrix.
fn sub(m: &DMatrix<f64>, br: usize, bc: usize, b: usize) -> DMatrix<f64> {
    m.view((br * b, bc * b), (b, b)).into_owned()
}

/// Scatter-add `scale · src` into the `(bi, bj)` block of `dst` (block size from `src`).
fn add_block(dst: &mut DMatrix<f64>, bi: usize, bj: usize, src: &DMatrix<f64>, scale: f64) {
    let b = src.nrows();
    for r in 0..b {
        for c in 0..src.ncols() {
            dst[(bi * b + r, bj * b + c)] += scale * src[(r, c)];
        }
    }
}

/// Natural log of the determinant of an SPD matrix via Cholesky
/// (`2·Σ ln L_ii`). `None` if not positive-definite.
fn logdet_spd(m: &DMatrix<f64>) -> Option<f64> {
    let chol = m.clone().cholesky()?;
    let l = chol.l();
    Some(2.0 * (0..l.nrows()).map(|i| l[(i, i)].ln()).sum::<f64>())
}

/// Union-find for Kruskal's spanning tree.
struct DisjointSet {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    /// Union the sets of `a` and `b`; returns `true` if they were disjoint.
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spd(m: &DMatrix<f64>) -> DMatrix<f64> {
        let n = m.nrows();
        m.transpose() * m + DMatrix::identity(n, n) * (n as f64)
    }

    /// Thin wrapper over the public [`kl_divergence`] scorer for the tests.
    fn kl_same_mean(lambda_p: &DMatrix<f64>, lambda_q: &DMatrix<f64>) -> f64 {
        kl_divergence(lambda_p, lambda_q).unwrap()
    }

    fn node_block(m: &DMatrix<f64>, i: usize, j: usize, b: usize) -> DMatrix<f64> {
        m.view((i * b, j * b), (b, b)).into_owned()
    }

    /// On a genuinely dense (fully-coupled) 4-node prior, the Chow-Liu tree
    /// preserves every NODE marginal and every TREE-EDGE pairwise marginal exactly,
    /// has exactly N−1 edges, and zeros every off-tree block.
    #[test]
    fn chow_liu_preserves_node_and_tree_edge_marginals_and_is_sparse() {
        let b = 2;
        let n = 8; // 4 nodes × 2
        let lambda = spd(&DMatrix::from_fn(n, n, |i, j| {
            ((i * 7 + j * 3) % 11) as f64 * 0.13 - 0.4
        }));
        let eta = DVector::from_fn(n, |i, _| 0.3 * i as f64 - 1.0);

        let sp = sparsify_chow_liu(&lambda, &eta, b).unwrap();
        assert_eq!(sp.edges.len(), 3, "spanning tree of 4 nodes has 3 edges");

        let sigma = lambda.clone().cholesky().unwrap().inverse();
        let sigma_t = sp.lambda.clone().cholesky().unwrap().inverse();
        // Node marginals preserved.
        for i in 0..4 {
            let d = (node_block(&sigma_t, i, i, b) - node_block(&sigma, i, i, b))
                .abs()
                .max();
            assert!(d < 1e-9, "node {i} marginal not preserved: {d}");
        }
        // Tree-edge pairwise marginals preserved.
        for &(i, j) in &sp.edges {
            let d = (node_block(&sigma_t, i, j, b) - node_block(&sigma, i, j, b))
                .abs()
                .max();
            assert!(d < 1e-9, "tree edge ({i},{j}) cross-cov not preserved: {d}");
        }
        // Off-tree blocks of the information matrix are exactly zero.
        let on_tree =
            |a: usize, c: usize| a == c || sp.edges.contains(&(a, c)) || sp.edges.contains(&(c, a));
        for a in 0..4 {
            for c in 0..4 {
                if !on_tree(a, c) {
                    assert!(
                        node_block(&sp.lambda, a, c, b).abs().max() < 1e-12,
                        "off-tree block ({a},{c}) must be zero"
                    );
                }
            }
        }
        // The mean is preserved: Λ_tree⁻¹ η_tree == Λ⁻¹ η.
        let mu = lambda.clone().cholesky().unwrap().solve(&eta);
        let mu_t = sp.lambda.clone().cholesky().unwrap().solve(&sp.eta);
        assert!((mu - mu_t).norm() < 1e-9, "mean not preserved");
    }

    /// When the dense prior is ALREADY tree-structured (block-tridiagonal = a
    /// chain), Chow-Liu recovers it exactly — zero KL divergence — because the
    /// chain is its own optimal tree.
    #[test]
    fn exact_recovery_when_prior_is_already_a_tree() {
        let b = 2;
        let num = 4;
        let n = b * num;
        // Build a block-tridiagonal (chain 0-1-2-3) SPD information matrix.
        let mut lambda = DMatrix::<f64>::zeros(n, n);
        for i in 0..num {
            let diag = DMatrix::from_fn(b, b, |r, c| if r == c { 4.0 } else { 0.5 });
            add_block(&mut lambda, i, i, &diag, 1.0);
        }
        for i in 0..(num - 1) {
            let off = DMatrix::from_fn(b, b, |r, c| if r == c { -1.0 } else { 0.2 });
            add_block(&mut lambda, i, i + 1, &off, 1.0);
            add_block(&mut lambda, i + 1, i, &off.transpose(), 1.0);
        }
        lambda = 0.5 * (&lambda + lambda.transpose());
        let eta = DVector::from_fn(n, |i, _| 0.2 * i as f64);

        let sp = sparsify_chow_liu(&lambda, &eta, b).unwrap();
        // The recovered tree IS the chain, and reproduces the prior exactly.
        let kl = kl_same_mean(&lambda, &sp.lambda);
        assert!(
            kl < 1e-9,
            "tree-structured prior must be recovered exactly, KL={kl}"
        );
        assert!(
            (&lambda - &sp.lambda).abs().max() < 1e-8,
            "Λ_tree must equal the original chain Λ"
        );
    }

    /// The Chow-Liu tree is the KL-OPTIMAL tree: its KL to the dense prior is no
    /// worse than any other spanning tree's. Checked against all spanning trees of
    /// a 4-node prior (enumerated by their 3 edges).
    #[test]
    fn chow_liu_minimizes_kl_over_all_trees() {
        let b = 1; // scalar nodes keep the enumeration cheap
        let num = 4;
        let lambda = spd(&DMatrix::from_fn(num, num, |i, j| {
            ((i * 5 + j * 2) % 7) as f64 * 0.3 - 0.7
        }));
        let eta = DVector::from_fn(num, |i, _| i as f64 - 1.5);
        let sp = sparsify_chow_liu(&lambda, &eta, b).unwrap();
        let cl_kl = kl_same_mean(&lambda, &sp.lambda);

        // Enumerate every set of 3 edges over 4 nodes that forms a spanning tree,
        // build that tree's Gaussian the same way, and confirm none beats Chow-Liu.
        let all_edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let sigma = lambda.clone().cholesky().unwrap().inverse();
        let mu = lambda.clone().cholesky().unwrap().solve(&eta);
        for a in 0..all_edges.len() {
            for bb in (a + 1)..all_edges.len() {
                for c in (bb + 1)..all_edges.len() {
                    let edges = [all_edges[a], all_edges[bb], all_edges[c]];
                    let mut dsu = DisjointSet::new(num);
                    if !edges.iter().all(|&(i, j)| dsu.union(i, j)) {
                        continue; // not a tree (cycle)
                    }
                    let lam_t = build_tree_lambda(&sigma, &edges, num, b);
                    let kl = kl_same_mean(&lambda, &lam_t);
                    assert!(
                        cl_kl <= kl + 1e-9,
                        "Chow-Liu KL {cl_kl} must not exceed tree {edges:?} KL {kl}"
                    );
                    // each candidate also preserves the mean (sanity)
                    let mu_t = lam_t
                        .cholesky()
                        .unwrap()
                        .solve(&(&build_tree_lambda(&sigma, &edges, num, b) * &mu));
                    assert!((&mu - &mu_t).norm() < 1e-9);
                }
            }
        }
    }

    /// Helper mirroring the production tree assembly, for the optimality test.
    fn build_tree_lambda(
        sigma: &DMatrix<f64>,
        edges: &[(usize, usize)],
        num: usize,
        b: usize,
    ) -> DMatrix<f64> {
        let mut degree = vec![0usize; num];
        for &(i, j) in edges {
            degree[i] += 1;
            degree[j] += 1;
        }
        let mut lam = DMatrix::<f64>::zeros(num * b, num * b);
        for &(i, j) in edges {
            let joint = joint_block(sigma, i, j, b);
            let info = joint.cholesky().unwrap().inverse();
            add_block(&mut lam, i, i, &sub(&info, 0, 0, b), 1.0);
            add_block(&mut lam, i, j, &sub(&info, 0, 1, b), 1.0);
            add_block(&mut lam, j, i, &sub(&info, 1, 0, b), 1.0);
            add_block(&mut lam, j, j, &sub(&info, 1, 1, b), 1.0);
        }
        for (i, &deg) in degree.iter().enumerate() {
            if deg > 1 {
                let li = node_block(sigma, i, i, b).cholesky().unwrap().inverse();
                add_block(&mut lam, i, i, &li, -((deg - 1) as f64));
            }
        }
        0.5 * (&lam + lam.transpose())
    }

    /// The diagonal sparsifier preserves every NODE marginal exactly (same
    /// guarantee as Chow-Liu) but zeros every off-diagonal block and preserves the
    /// mean. The zero-edge contestant.
    #[test]
    fn diagonal_preserves_node_marginals_and_drops_all_correlations() {
        let b = 2;
        let n = 8; // 4 nodes × 2
        let lambda = spd(&DMatrix::from_fn(n, n, |i, j| {
            ((i * 7 + j * 3) % 11) as f64 * 0.13 - 0.4
        }));
        let eta = DVector::from_fn(n, |i, _| 0.3 * i as f64 - 1.0);

        let sp = sparsify_diagonal(&lambda, &eta, b).unwrap();
        assert!(sp.edges.is_empty(), "diagonal sparsifier keeps no edges");

        let sigma = lambda.clone().cholesky().unwrap().inverse();
        let sigma_d = sp.lambda.clone().cholesky().unwrap().inverse();
        for i in 0..4 {
            // node marginal preserved
            let d = (node_block(&sigma_d, i, i, b) - node_block(&sigma, i, i, b))
                .abs()
                .max();
            assert!(d < 1e-9, "node {i} marginal not preserved: {d}");
            for j in 0..4 {
                if i != j {
                    // every off-diagonal information block is exactly zero
                    assert!(
                        node_block(&sp.lambda, i, j, b).abs().max() < 1e-12,
                        "off-diagonal block ({i},{j}) must be zero"
                    );
                }
            }
        }
        // Mean preserved: Λ_diag⁻¹ η_diag == Λ⁻¹ η.
        let mu = lambda.clone().cholesky().unwrap().solve(&eta);
        let mu_d = sp.lambda.clone().cholesky().unwrap().solve(&sp.eta);
        assert!((mu - mu_d).norm() < 1e-9, "mean not preserved");
    }

    /// The head-to-head: on a genuinely correlated dense prior the Chow-Liu tree's
    /// KL to the dense prior is STRICTLY smaller than the diagonal (independence)
    /// sparsifier's — the `N−1` retained couplings strictly help — while the dense
    /// prior itself scores `0`. This is the ordering the sparsification shootout
    /// rests on (`0 = dense < Chow-Liu < diagonal`), and it follows from Chow-Liu
    /// optimizing over all trees, the edgeless one (= the diagonal) included.
    #[test]
    fn chow_liu_strictly_beats_diagonal_on_a_correlated_prior() {
        let b = 2;
        let n = 8;
        let lambda = spd(&DMatrix::from_fn(n, n, |i, j| {
            ((i * 5 + j * 2) % 9) as f64 * 0.17 - 0.5
        }));
        let eta = DVector::from_fn(n, |i, _| 0.2 * i as f64 - 0.7);

        let clt = sparsify_chow_liu(&lambda, &eta, b).unwrap();
        let diag = sparsify_diagonal(&lambda, &eta, b).unwrap();

        let kl_dense = kl_divergence(&lambda, &lambda).unwrap();
        let kl_clt = kl_divergence(&lambda, &clt.lambda).unwrap();
        let kl_diag = kl_divergence(&lambda, &diag.lambda).unwrap();

        assert!(
            kl_dense.abs() < 1e-12,
            "dense scores zero KL, got {kl_dense}"
        );
        assert!(kl_clt >= -1e-12, "KL is non-negative, got {kl_clt}");
        assert!(
            kl_clt + 1e-9 < kl_diag,
            "Chow-Liu KL {kl_clt} must strictly beat diagonal KL {kl_diag}"
        );
        // Sparsity ordering: 0 edges < N−1 tree edges < N(N−1)/2 dense.
        assert_eq!(diag.edges.len(), 0);
        assert_eq!(clt.edges.len(), 3);
    }

    /// When the dense prior is ALREADY block-diagonal (independent nodes), the
    /// diagonal sparsifier — and Chow-Liu — recover it exactly (KL 0): there is no
    /// correlation to keep, so all three contestants coincide.
    #[test]
    fn diagonal_is_exact_when_prior_is_already_independent() {
        let b = 2;
        let num = 3;
        let n = b * num;
        let mut lambda = DMatrix::<f64>::zeros(n, n);
        for i in 0..num {
            let diag = DMatrix::from_fn(b, b, |r, c| if r == c { 3.0 + i as f64 } else { 0.4 });
            add_block(&mut lambda, i, i, &diag, 1.0);
        }
        lambda = 0.5 * (&lambda + lambda.transpose());
        let eta = DVector::from_fn(n, |i, _| 0.1 * i as f64);

        let diag = sparsify_diagonal(&lambda, &eta, b).unwrap();
        let kl = kl_divergence(&lambda, &diag.lambda).unwrap();
        assert!(
            kl < 1e-9,
            "independent prior must be recovered exactly, KL={kl}"
        );
        assert!(
            (&lambda - &diag.lambda).abs().max() < 1e-9,
            "Λ_diag must equal the original block-diagonal Λ"
        );
    }

    #[test]
    fn rejects_bad_inputs_and_passes_through_single_block() {
        let lambda = spd(&DMatrix::from_fn(4, 4, |i, j| (i + j) as f64 * 0.1));
        let eta = DVector::from_fn(4, |i, _| i as f64);
        assert!(sparsify_chow_liu(&lambda, &eta, 0).is_none()); // block_dim 0
        assert!(sparsify_chow_liu(&lambda, &eta, 3).is_none()); // 4 not divisible by 3
                                                                // Single block (N=1): returned unchanged, no edges.
        let sp = sparsify_chow_liu(&lambda, &eta, 4).unwrap();
        assert!(sp.edges.is_empty());
        assert!((&sp.lambda - &lambda).abs().max() < 1e-12);
    }
}
