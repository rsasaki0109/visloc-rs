//! Fill-reducing reordering for the sparse Cholesky back-end.
//!
//! `nalgebra_sparse::CscCholesky` factors the system in its natural variable
//! order — it performs no fill-reducing permutation of its own. For pose
//! graphs that arrive in a poor order (or that are intrinsically wide, like the
//! 3D `torus`/`sphere` benchmarks), the Cholesky factor `L` then acquires far
//! more nonzeros than the original `H`, and both the factorization time and the
//! memory blow up super-linearly.
//!
//! A *symmetric* permutation `P` reorders rows and columns identically, so
//! `PᵀHP` stays symmetric positive-definite and its Cholesky factor solves the
//! same system — only the sparsity of `L` changes. [`Reordering::fill_reducing`]
//! considers two cheap geometric orderings of the *block-adjacency* graph plus a
//! minimum-degree *rescue*:
//!
//! - [Reverse Cuthill–McKee][rcm] (RCM): a band-minimizing breadth-first
//!   ordering. Near-optimal for "thin" graphs (chains, corridors) whose factor
//!   stays banded — e.g. the `parking-garage` benchmark.
//! - [Nested dissection][nd] (George's automatic, BFS-level-separator variant):
//!   recursively splits the graph by a small vertex separator and orders that
//!   separator *last*. Wins on wide, regular meshes such as `sphere2500`/`torus`.
//! - [Minimum degree][md] (MD): greedily eliminate the lowest-degree vertex,
//!   the local heuristic at the heart of AMD/SuiteSparse. It is the *rescue*
//!   ordering for dense, irregular graphs (ICP pose graphs such as
//!   `cubicle`/`rim`) where both geometric orderings explode — there MD cuts the
//!   factor from intractable to ~10⁵ blocks and the solve from a >10-minute
//!   timeout to tens of seconds.
//!
//! We pick the cheaper of RCM and nested dissection by an exact symbolic factor
//! count (`symbolic_cholesky_nnz`). A poorly-ordered factor can have
//! *catastrophic* fill — counting it in full would itself dominate the solve —
//! so the count (`symbolic_cholesky_nnz_capped`) abandons a candidate once it
//! exceeds the dense-factor cap `n²`. Only when *both* geometric orderings blow
//! past that cap do we compute minimum degree and adopt it if it is genuinely
//! sparser. MD is held back as a rescue rather than always run because its
//! factor, though it can have *fewer* nonzeros, factorizes *more slowly* than a
//! healthy geometric ordering in the scalar (non-supernodal) backend — its
//! elimination tree is deeper and less cache-friendly — so letting it
//! second-guess a healthy ordering would regress the regular benchmarks.
//!
//! Everything here is purely structural and fully deterministic (ties broken by
//! ascending node id), so it preserves the solver's bit-for-bit reproducibility
//! while leaving the numerical answer unchanged up to floating-point summation
//! order within the factorization.
//!
//! [md]: https://en.wikipedia.org/wiki/Minimum_degree_algorithm
//! [rcm]: https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
//! [nd]: https://en.wikipedia.org/wiki/Nested_dissection

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet, VecDeque};

use nalgebra::DVector;

/// Subgraphs with at most this many nodes are emitted in their natural id order
/// instead of being dissected further — their factor is small enough that a
/// separator buys nothing, and stopping early bounds the recursion overhead.
const NESTED_DISSECTION_LEAF: usize = 8;

/// A geometric ordering (RCM / nested dissection) is preferred over minimum
/// degree as long as its symbolic factor stays within this multiple of minimum
/// degree's — geometric factors are sparser-structured and factorize faster per
/// nonzero in the scalar backend, so a modestly larger one still wins. Only when
/// *both* geometric factors exceed this ratio (a catastrophic blow-up, as on
/// dense ICP graphs) is the far-sparser minimum-degree ordering used instead.
const RESCUE_FILL_RATIO: usize = 4;

/// A symmetric permutation of an `n`-dimensional linear system, computed at the
/// granularity of fixed-size variable blocks (6 for SE(3) poses, 3 for
/// translation-only centers) so that a variable's coordinates always move
/// together.
pub(crate) struct Reordering {
    /// `old_of_new[i]` is the original dimension index now sitting at the
    /// permuted position `i`.
    old_of_new: Vec<usize>,
    /// `new_of_old[j]` is the permuted position that original dimension index
    /// `j` moved to. Inverse of `old_of_new`.
    new_of_old: Vec<usize>,
}

impl Reordering {
    /// Pick the fill-reducing ordering for the system whose structural pattern
    /// is given by `triplets`, treating each contiguous run of `block_size`
    /// dimensions as one graph node (`dim` must be a multiple of `block_size`).
    ///
    /// Picks the cheaper of the nested-dissection and Reverse Cuthill–McKee
    /// orderings by symbolic Cholesky factor size, falling back to a
    /// minimum-degree ordering only when both blow past the dense-factor cap.
    /// The pattern alone determines the result, so callers compute this once and
    /// reuse it across solver iterations (the values change, the pattern does
    /// not).
    pub(crate) fn fill_reducing(
        dim: usize,
        block_size: usize,
        triplets: &[(usize, usize, f64)],
    ) -> Self {
        debug_assert!(block_size >= 1, "block_size must be positive");
        debug_assert!(
            dim % block_size == 0,
            "dim ({dim}) must be a multiple of block_size ({block_size})"
        );
        let n = dim / block_size;
        let adjacency = block_adjacency(n, block_size, triplets);

        // Minimum degree is cheap to compute and reliably keeps its factor
        // sparse, so cost it first and use it both as a baseline and to bound
        // the costing of the others: a catastrophically bad geometric factor is
        // then abandoned after only a few × MD's worth of counting instead of
        // running to its full (e.g. `cubicle`-sized) blow-up.
        let md = minimum_degree_order(&adjacency);
        let md_nnz = symbolic_cholesky_nnz(&adjacency, &md);

        // Prefer the two cheap, BFS-based geometric orderings: their balanced
        // elimination trees *factorize* faster per nonzero in the scalar (non-
        // supernodal) backend than minimum degree's deeper, scattered tree, so
        // a geometric ordering within a few × MD's fill is the better choice.
        // Cap their counts at `RESCUE_FILL_RATIO × md_nnz`; a blown-up factor
        // (dense ICP graphs such as `cubicle`/`rim`) trips the cap cheaply.
        let cap = md_nnz.saturating_mul(RESCUE_FILL_RATIO);
        let nested = nested_dissection_order(&adjacency);
        let rcm = reverse_cuthill_mckee_order(&adjacency);
        let nested_nnz = symbolic_cholesky_nnz_capped(&adjacency, &nested, cap);
        let rcm_nnz = symbolic_cholesky_nnz_capped(&adjacency, &rcm, cap);

        // Use the cheaper geometric ordering when it stays within the rescue
        // ratio; otherwise both blew past it and minimum degree is the rescue.
        let best_geometric = nested_nnz.min(rcm_nnz);
        let chosen = if best_geometric <= cap {
            if nested_nnz <= rcm_nnz {
                nested
            } else {
                rcm
            }
        } else {
            md
        };
        Self::from_block_order(&chosen, dim, block_size)
    }

    /// Expand a block elimination order (`block_order[i]` is the original block
    /// eliminated `i`-th) into a dimension-level permutation, keeping each
    /// block's coordinates contiguous and in their original sub-order.
    fn from_block_order(block_order: &[usize], dim: usize, block_size: usize) -> Self {
        let mut old_of_new = Vec::with_capacity(dim);
        for &old_block in block_order {
            for k in 0..block_size {
                old_of_new.push(old_block * block_size + k);
            }
        }
        let mut new_of_old = vec![0usize; dim];
        for (new_index, &old_index) in old_of_new.iter().enumerate() {
            new_of_old[old_index] = new_index;
        }
        Self {
            old_of_new,
            new_of_old,
        }
    }

    /// Remap COO triplets into the permuted ordering: an entry at original
    /// `(r, c)` moves to `(new_of_old[r], new_of_old[c])`, value unchanged.
    pub(crate) fn permute_triplets(
        &self,
        triplets: &[(usize, usize, f64)],
    ) -> Vec<(usize, usize, f64)> {
        triplets
            .iter()
            .map(|&(r, c, v)| (self.new_of_old[r], self.new_of_old[c], v))
            .collect()
    }

    /// Permute a right-hand side into the reordered system: `out[i]` is the
    /// original entry that moved to position `i`.
    pub(crate) fn permute_rhs(&self, b: &DVector<f64>) -> DVector<f64> {
        DVector::from_iterator(b.len(), self.old_of_new.iter().map(|&old| b[old]))
    }

    /// Scatter a solution of the reordered system back into the original
    /// variable order: the value at permuted position `i` belongs to original
    /// index `old_of_new[i]`.
    pub(crate) fn restore_solution(&self, permuted: &DVector<f64>) -> DVector<f64> {
        let mut out = DVector::zeros(permuted.len());
        for (i, &old) in self.old_of_new.iter().enumerate() {
            out[old] = permuted[i];
        }
        out
    }
}

/// Build the symmetric block-adjacency graph: an undirected edge between two
/// distinct blocks for every off-diagonal nonzero coupling them. Neighbour
/// lists are sorted and deduplicated (via `BTreeSet`) for deterministic
/// traversal.
fn block_adjacency(
    n: usize,
    block_size: usize,
    triplets: &[(usize, usize, f64)],
) -> Vec<Vec<usize>> {
    let mut sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for &(row, col, _) in triplets {
        let (br, bc) = (row / block_size, col / block_size);
        if br != bc {
            sets[br].insert(bc);
            sets[bc].insert(br);
        }
    }
    sets.into_iter().map(|s| s.into_iter().collect()).collect()
}

/// Reverse Cuthill–McKee elimination order: breadth-first over every connected
/// component, rooting each at the lowest-degree unvisited node and expanding
/// neighbours in ascending `(degree, id)` order, then reversed.
fn reverse_cuthill_mckee_order(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let degree: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut visited = vec![false; n];
    let mut order: Vec<usize> = Vec::with_capacity(n);
    loop {
        let root = (0..n)
            .filter(|&i| !visited[i])
            .min_by_key(|&i| (degree[i], i));
        let Some(root) = root else { break };

        visited[root] = true;
        let mut queue = VecDeque::new();
        queue.push_back(root);
        while let Some(node) = queue.pop_front() {
            order.push(node);
            let mut neighbours: Vec<usize> = adjacency[node]
                .iter()
                .copied()
                .filter(|&m| !visited[m])
                .collect();
            neighbours.sort_by_key(|&m| (degree[m], m));
            for m in neighbours {
                if !visited[m] {
                    visited[m] = true;
                    queue.push_back(m);
                }
            }
        }
    }
    // Reversing the Cuthill–McKee order is what makes it *Reverse* CM, which
    // empirically yields less fill than plain CM at no extra cost.
    order.reverse();
    order
}

/// Minimum-degree elimination order: repeatedly eliminate the live vertex of
/// smallest current degree, adding a fill clique among its remaining neighbours,
/// until the graph is empty. This is the greedy local heuristic at the heart of
/// AMD/SuiteSparse and is typically the strongest of the three on the *irregular*
/// graphs that band (RCM) and balanced-separator (nested dissection) orderings
/// handle poorly — e.g. ICP pose graphs such as `cubicle`/`rim`. Ties break by
/// ascending id, keeping the result deterministic.
///
/// The degree-minimum is found by a linear scan (`O(n)` per step), and neighbour
/// sets are updated exactly rather than via AMD's approximate quotient-graph
/// bound — simpler, and adequate because minimum degree keeps the working graph
/// sparse on exactly the graphs where it is the right ordering.
fn minimum_degree_order(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut neighbours: Vec<HashSet<usize>> = adjacency
        .iter()
        .map(|row| row.iter().copied().collect())
        .collect();
    let mut eliminated = vec![false; n];
    let mut order = Vec::with_capacity(n);

    for _ in 0..n {
        let pivot = (0..n)
            .filter(|&i| !eliminated[i])
            .min_by_key(|&i| (neighbours[i].len(), i))
            .expect("a non-eliminated vertex remains");
        eliminated[pivot] = true;
        order.push(pivot);

        let live: Vec<usize> = neighbours[pivot]
            .iter()
            .copied()
            .filter(|&u| !eliminated[u])
            .collect();
        for &u in &live {
            neighbours[u].remove(&pivot);
        }
        // The eliminated pivot couples all its live neighbours: add the fill
        // clique among them.
        for a in 0..live.len() {
            for b in (a + 1)..live.len() {
                neighbours[live[a]].insert(live[b]);
                neighbours[live[b]].insert(live[a]);
            }
        }
    }
    order
}

/// Nested-dissection elimination order using George's automatic algorithm: find
/// a vertex separator from a pseudo-peripheral breadth-first level structure,
/// recursively order the two sides it disconnects, then order the separator
/// last. Disconnected pieces are ordered independently.
fn nested_dissection_order(adjacency: &[Vec<usize>]) -> Vec<usize> {
    let n = adjacency.len();
    let mut order = Vec::with_capacity(n);
    dissect((0..n).collect(), adjacency, &mut order);
    order
}

/// Recursive core of [`nested_dissection_order`]. Appends the elimination order
/// of `nodes` to `out`: variables eliminated first land at the front, so the
/// two halves are appended before the separator that decouples them.
fn dissect(nodes: Vec<usize>, adjacency: &[Vec<usize>], out: &mut Vec<usize>) {
    if nodes.len() <= NESTED_DISSECTION_LEAF {
        let mut leaf = nodes;
        leaf.sort_unstable();
        out.extend(leaf);
        return;
    }

    let in_set: HashSet<usize> = nodes.iter().copied().collect();
    let components = connected_components(&nodes, adjacency, &in_set);
    if components.len() > 1 {
        // Independent components share no fill, so order each on its own.
        for component in components {
            dissect(component, adjacency, out);
        }
        return;
    }

    let levels = pseudo_peripheral_levels(&nodes, adjacency, &in_set);
    if levels.len() <= 1 {
        // No level separator exists (e.g. a near-complete subgraph); give up.
        let mut leaf = nodes;
        leaf.sort_unstable();
        out.extend(leaf);
        return;
    }

    // Removing one whole BFS level disconnects the levels before it from the
    // levels after it (BFS edges only span adjacent levels). Use the middle
    // level as the separator to keep the two halves balanced.
    let mid = levels.len() / 2;
    let (mut left, mut right, mut separator) = (Vec::new(), Vec::new(), Vec::new());
    for (level_index, level) in levels.into_iter().enumerate() {
        match level_index.cmp(&mid) {
            Ordering::Less => left.extend(level),
            Ordering::Equal => separator.extend(level),
            Ordering::Greater => right.extend(level),
        }
    }

    dissect(left, adjacency, out);
    dissect(right, adjacency, out);
    separator.sort_unstable();
    out.extend(separator);
}

/// Connected components of the subgraph induced by `nodes`, each sorted and the
/// list ordered by ascending lowest member, for determinism.
fn connected_components(
    nodes: &[usize],
    adjacency: &[Vec<usize>],
    in_set: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    let mut sorted_nodes = nodes.to_vec();
    sorted_nodes.sort_unstable();
    let mut visited: HashSet<usize> = HashSet::with_capacity(nodes.len());
    let mut components = Vec::new();
    for &start in &sorted_nodes {
        if visited.contains(&start) {
            continue;
        }
        let mut component = Vec::new();
        let mut stack = vec![start];
        visited.insert(start);
        while let Some(node) = stack.pop() {
            component.push(node);
            for &neighbour in &adjacency[node] {
                if in_set.contains(&neighbour) && visited.insert(neighbour) {
                    stack.push(neighbour);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// Rooted BFS level structure of the (assumed connected) subgraph `in_set`,
/// started from a pseudo-peripheral node so the structure is long and its
/// levels — the separator candidates — are thin. Each returned level is sorted.
fn pseudo_peripheral_levels(
    nodes: &[usize],
    adjacency: &[Vec<usize>],
    in_set: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    let subset_degree = |v: usize| {
        adjacency[v]
            .iter()
            .filter(|&&u| in_set.contains(&u))
            .count()
    };

    let mut start = *nodes
        .iter()
        .min_by_key(|&&v| (subset_degree(v), v))
        .expect("dissect only recurses on non-empty node sets");
    let mut levels = bfs_levels(start, adjacency, in_set);
    loop {
        let candidate = *levels
            .last()
            .expect("a BFS level structure has at least one level")
            .iter()
            .min_by_key(|&&v| (subset_degree(v), v))
            .expect("levels are non-empty");
        if candidate == start {
            break;
        }
        let candidate_levels = bfs_levels(candidate, adjacency, in_set);
        if candidate_levels.len() > levels.len() {
            start = candidate;
            levels = candidate_levels;
        } else {
            break;
        }
    }
    levels
}

/// BFS distance layers from `start` within `in_set`; layer `d` holds the nodes
/// at graph distance `d`. Each layer is sorted for deterministic separators.
fn bfs_levels(start: usize, adjacency: &[Vec<usize>], in_set: &HashSet<usize>) -> Vec<Vec<usize>> {
    let mut visited: HashSet<usize> = HashSet::new();
    visited.insert(start);
    let mut levels: Vec<Vec<usize>> = Vec::new();
    let mut current = vec![start];
    while !current.is_empty() {
        let mut next = Vec::new();
        for &node in &current {
            for &neighbour in &adjacency[node] {
                if in_set.contains(&neighbour) && visited.insert(neighbour) {
                    next.push(neighbour);
                }
            }
        }
        current.sort_unstable();
        levels.push(current);
        current = next;
    }
    levels
}

/// Exact nonzero count of the Cholesky factor `L` for `adjacency` eliminated in
/// `block_order`, via the elimination tree and per-column counts (the standard
/// Gilbert–Ng–Peyton symbolic factorization). Used only to rank candidate
/// orderings, so it counts blocks, not scalars — block sizes are uniform, so the
/// ranking is the same. This uncapped form costs the minimum-degree baseline
/// (whose fill is reliably small); the others are costed with the capped form.
fn symbolic_cholesky_nnz(adjacency: &[Vec<usize>], block_order: &[usize]) -> usize {
    symbolic_cholesky_nnz_capped(adjacency, block_order, usize::MAX)
}

/// Like [`symbolic_cholesky_nnz`] but abandons counting as soon as the running
/// total exceeds `cap`, returning some value `> cap`. A bad ordering's factor
/// can have *catastrophic* fill (hundreds of millions of blocks on dense ICP
/// graphs like `cubicle`), so an uncapped count of it would itself be the
/// bottleneck. Costing the cheap minimum-degree ordering first and capping the
/// others by it bounds the selection's total work to roughly the best fill.
fn symbolic_cholesky_nnz_capped(
    adjacency: &[Vec<usize>],
    block_order: &[usize],
    cap: usize,
) -> usize {
    let n = adjacency.len();

    // Relabel the pattern into elimination-index space, where eliminating in
    // increasing index order is exactly `block_order`.
    let mut new_of_old = vec![0usize; n];
    for (new_index, &old) in block_order.iter().enumerate() {
        new_of_old[old] = new_index;
    }
    let mut pattern: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (new_index, &old) in block_order.iter().enumerate() {
        let mut row: Vec<usize> = adjacency[old].iter().map(|&u| new_of_old[u]).collect();
        row.sort_unstable();
        pattern[new_index] = row;
    }

    // Elimination tree: parent[j] is the first row below the diagonal in column
    // j of L. Built with the classic path-compressing ancestor walk.
    let mut parent = vec![usize::MAX; n];
    let mut ancestor = vec![usize::MAX; n];
    for (j, row) in pattern.iter().enumerate() {
        for &i in row {
            if i >= j {
                continue;
            }
            let mut node = i;
            while ancestor[node] != usize::MAX && ancestor[node] != j {
                let next = ancestor[node];
                ancestor[node] = j;
                node = next;
            }
            if ancestor[node] == usize::MAX {
                ancestor[node] = j;
                parent[node] = j;
            }
        }
    }

    // Column counts: for each row i, the nonzeros of L below the diagonal are
    // the union of the etree paths from each original lower neighbour up to i.
    // Marking each path once per row counts every L entry exactly once. We
    // accumulate the running total (`n` diagonal entries plus one per marked
    // path step) so we can bail the moment it exceeds `cap`.
    let mut total = n;
    if total > cap {
        return total;
    }
    let mut mark = vec![usize::MAX; n];
    for (i, row) in pattern.iter().enumerate() {
        mark[i] = i;
        for &j in row {
            if j >= i {
                continue;
            }
            let mut k = j;
            while mark[k] != i {
                total += 1;
                if total > cap {
                    return total;
                }
                mark[k] = i;
                match parent[k] {
                    usize::MAX => break,
                    next => k = next,
                }
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill_reducing_permutation_is_a_bijection() {
        let triplets = grid_triplets(5, 5, 2);
        let order = Reordering::fill_reducing(50, 2, &triplets);

        let mut seen = vec![false; 50];
        for &old in &order.old_of_new {
            assert!(!seen[old], "index {old} appears twice");
            seen[old] = true;
        }
        assert!(
            seen.into_iter().all(|s| s),
            "permutation must cover all dims"
        );
        for (j, &old) in order.old_of_new.iter().enumerate() {
            assert_eq!(
                order.new_of_old[old], j,
                "old_of_new and new_of_old disagree"
            );
        }
    }

    #[test]
    fn blocks_stay_contiguous() {
        let triplets = grid_triplets(4, 4, 3);
        let order = Reordering::fill_reducing(48, 3, &triplets);
        for chunk in order.old_of_new.chunks(3) {
            let block = chunk[0] / 3;
            assert_eq!(chunk, &[block * 3, block * 3 + 1, block * 3 + 2]);
        }
    }

    #[test]
    fn fill_reducing_matches_the_original_solution() {
        // Build a small SPD system, solve it through the reordering, and confirm
        // the restored solution equals the directly-solved one.
        use nalgebra::DMatrix;

        let dim = 6;
        let block_size = 2;
        let mut h = DMatrix::<f64>::zeros(dim, dim);
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for k in 0..dim {
            h[(k, k)] = 4.0;
            triplets.push((k, k, 4.0));
        }
        for block in 0..2 {
            for k in 0..block_size {
                let (a, b) = (block * block_size + k, (block + 1) * block_size + k);
                h[(a, b)] = -1.0;
                h[(b, a)] = -1.0;
                triplets.push((a, b, -1.0));
                triplets.push((b, a, -1.0));
            }
        }
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x_direct = h.clone().lu().solve(&b).expect("SPD system is solvable");

        let order = Reordering::fill_reducing(dim, block_size, &triplets);
        let permuted = order.permute_triplets(&triplets);
        let mut h_perm = DMatrix::<f64>::zeros(dim, dim);
        for (r, c, v) in permuted {
            h_perm[(r, c)] += v;
        }
        let b_perm = order.permute_rhs(&b);
        let x_perm = h_perm
            .lu()
            .solve(&b_perm)
            .expect("permuted system is solvable");
        let x_restored = order.restore_solution(&x_perm);

        assert!(
            (x_restored - x_direct).norm() < 1e-10,
            "reordered solve must match the original solution"
        );
    }

    #[test]
    fn symbolic_count_matches_brute_force_simulation() {
        // The etree-based count must equal a direct symbolic elimination on
        // several structures and orderings.
        for (rows, cols) in [(1, 6), (3, 4), (5, 5), (2, 9)] {
            let adjacency = grid_adjacency(rows, cols);
            let n = adjacency.len();
            let natural: Vec<usize> = (0..n).collect();
            let rcm = reverse_cuthill_mckee_order(&adjacency);
            let nested = nested_dissection_order(&adjacency);
            let md = minimum_degree_order(&adjacency);
            for order in [&natural, &rcm, &nested, &md] {
                assert_eq!(
                    symbolic_cholesky_nnz(&adjacency, order),
                    brute_force_fill(&adjacency, order),
                    "grid {rows}x{cols}"
                );
            }
        }
    }

    #[test]
    fn minimum_degree_order_is_a_valid_permutation() {
        let adjacency = grid_adjacency(7, 9);
        let order = minimum_degree_order(&adjacency);
        let mut seen = vec![false; adjacency.len()];
        for &node in &order {
            assert!(!seen[node], "node {node} appears twice");
            seen[node] = true;
        }
        assert!(seen.into_iter().all(|s| s), "order must cover every node");
    }

    #[test]
    fn minimum_degree_beats_the_natural_order_on_an_arrow() {
        // An "arrow": a hub connected to every spoke. Eliminating the hub first
        // (natural order) fills in the entire spoke clique; minimum degree
        // eliminates the degree-1 spokes first and leaves the factor sparse.
        let n = 40;
        let mut sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for spoke in 1..n {
            sets[0].insert(spoke);
            sets[spoke].insert(0);
        }
        let adjacency: Vec<Vec<usize>> =
            sets.into_iter().map(|s| s.into_iter().collect()).collect();
        let natural: Vec<usize> = (0..n).collect();
        let md = minimum_degree_order(&adjacency);
        assert!(
            symbolic_cholesky_nnz(&adjacency, &md) < symbolic_cholesky_nnz(&adjacency, &natural),
            "minimum degree should avoid the arrow's hub-first fill"
        );
    }

    #[test]
    fn symbolic_count_cap_aborts_above_the_budget() {
        // The hub-first arrow order has a large factor; capping below it returns
        // a sentinel above the cap, while the uncapped count is exact.
        let n = 30;
        let mut sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for spoke in 1..n {
            sets[0].insert(spoke);
            sets[spoke].insert(0);
        }
        let adjacency: Vec<Vec<usize>> =
            sets.into_iter().map(|s| s.into_iter().collect()).collect();
        let natural: Vec<usize> = (0..n).collect();
        let exact = symbolic_cholesky_nnz(&adjacency, &natural);
        let cap = exact / 2;
        let capped = symbolic_cholesky_nnz_capped(&adjacency, &natural, cap);
        assert!(
            capped > cap,
            "capped count must signal exceeding the budget"
        );
        assert_eq!(
            symbolic_cholesky_nnz_capped(&adjacency, &natural, exact),
            exact,
            "a cap at the exact total must return the exact total"
        );
    }

    #[test]
    fn nested_dissection_beats_the_natural_order_on_a_wide_grid() {
        // A square 2D grid is the canonical case where a band ordering cannot
        // avoid fill but a separator ordering can.
        let adjacency = grid_adjacency(12, 12);
        let natural: Vec<usize> = (0..adjacency.len()).collect();
        let nested = nested_dissection_order(&adjacency);
        assert!(
            symbolic_cholesky_nnz(&adjacency, &nested)
                < symbolic_cholesky_nnz(&adjacency, &natural),
            "nested dissection should reduce fill on a wide grid"
        );
    }

    #[test]
    fn fill_reducing_prefers_the_cheaper_geometric_ordering_on_a_healthy_graph() {
        // A regular grid is "healthy": neither geometric factor blows past the
        // rescue ratio, so `fill_reducing` keeps the cheaper of RCM / nested
        // dissection rather than reaching for minimum degree.
        let triplets = grid_triplets(10, 10, 1);
        let adjacency = block_adjacency(100, 1, &triplets);
        let rcm_nnz = symbolic_cholesky_nnz(&adjacency, &reverse_cuthill_mckee_order(&adjacency));
        let nd_nnz = symbolic_cholesky_nnz(&adjacency, &nested_dissection_order(&adjacency));
        let md_nnz = symbolic_cholesky_nnz(&adjacency, &minimum_degree_order(&adjacency));
        let best_geometric = rcm_nnz.min(nd_nnz);
        assert!(
            best_geometric <= md_nnz * RESCUE_FILL_RATIO,
            "the grid should not trip the minimum-degree rescue"
        );

        let order = Reordering::fill_reducing(100, 1, &triplets);
        let chosen_blocks: Vec<usize> = order.old_of_new.clone();
        assert_eq!(
            symbolic_cholesky_nnz(&adjacency, &chosen_blocks),
            best_geometric,
            "fill_reducing must keep the cheaper geometric ordering on a healthy graph"
        );
    }

    /// Direct symbolic elimination: eliminate blocks in `block_order`, adding a
    /// clique among each pivot's higher-numbered neighbours, and total the
    /// resulting column nonzero counts. O(fill²), so test-only.
    fn brute_force_fill(adjacency: &[Vec<usize>], block_order: &[usize]) -> usize {
        let n = adjacency.len();
        let mut new_of_old = vec![0usize; n];
        for (new_index, &old) in block_order.iter().enumerate() {
            new_of_old[old] = new_index;
        }
        let mut sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for (new_index, &old) in block_order.iter().enumerate() {
            for &neighbour in &adjacency[old] {
                sets[new_index].insert(new_of_old[neighbour]);
            }
        }
        let mut total = 0;
        for j in 0..n {
            let higher: Vec<usize> = sets[j].iter().copied().filter(|&x| x > j).collect();
            total += 1 + higher.len();
            for a in 0..higher.len() {
                for b in (a + 1)..higher.len() {
                    sets[higher[a]].insert(higher[b]);
                    sets[higher[b]].insert(higher[a]);
                }
            }
        }
        total
    }

    /// Adjacency of a `rows x cols` 4-neighbour 2D grid (block ids row-major).
    fn grid_adjacency(rows: usize, cols: usize) -> Vec<Vec<usize>> {
        let n = rows * cols;
        let mut sets: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        let id = |r: usize, c: usize| r * cols + c;
        for r in 0..rows {
            for c in 0..cols {
                if c + 1 < cols {
                    sets[id(r, c)].insert(id(r, c + 1));
                    sets[id(r, c + 1)].insert(id(r, c));
                }
                if r + 1 < rows {
                    sets[id(r, c)].insert(id(r + 1, c));
                    sets[id(r + 1, c)].insert(id(r, c));
                }
            }
        }
        sets.into_iter().map(|s| s.into_iter().collect()).collect()
    }

    /// COO triplets (with diagonal) of a `rows x cols` grid expanded to
    /// `block_size`-dimensional variable blocks.
    fn grid_triplets(rows: usize, cols: usize, block_size: usize) -> Vec<(usize, usize, f64)> {
        let adjacency = grid_adjacency(rows, cols);
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for (block, neighbours) in adjacency.iter().enumerate() {
            for k in 0..block_size {
                triplets.push((block * block_size + k, block * block_size + k, 4.0));
            }
            for &neighbour in neighbours {
                for k in 0..block_size {
                    triplets.push((block * block_size + k, neighbour * block_size + k, -1.0));
                }
            }
        }
        triplets
    }
}
