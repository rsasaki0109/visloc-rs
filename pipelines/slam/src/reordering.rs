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
//! same system — only the sparsity of `L` changes. We compute two candidate
//! orderings over the *block-adjacency* graph of the variables and keep the one
//! whose symbolic Cholesky has fewer nonzeros:
//!
//! - [Reverse Cuthill–McKee][rcm] (RCM): a band-minimizing breadth-first
//!   ordering. Near-optimal for "thin" graphs (chains, corridors) whose factor
//!   stays banded — e.g. the `parking-garage` benchmark.
//! - [Nested dissection][nd] (George's automatic, BFS-level-separator variant):
//!   recursively splits the graph by a small vertex separator and orders that
//!   separator *last*. This is what tames intrinsically wide graphs (2D/3D
//!   meshes such as `torus3D`), where a band ordering cannot avoid large fill.
//!
//! The two are complementary, so [`Reordering::fill_reducing`] builds both and
//! selects the cheaper by an exact symbolic factorization count (see
//! [`symbolic_cholesky_nnz`]). Everything here is purely structural and fully
//! deterministic (ties broken by ascending node id), so it preserves the
//! solver's bit-for-bit reproducibility while leaving the numerical answer
//! unchanged up to floating-point summation order within the factorization.
//!
//! [rcm]: https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm
//! [nd]: https://en.wikipedia.org/wiki/Nested_dissection

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashSet, VecDeque};

use nalgebra::DVector;

/// Subgraphs with at most this many nodes are emitted in their natural id order
/// instead of being dissected further — their factor is small enough that a
/// separator buys nothing, and stopping early bounds the recursion overhead.
const NESTED_DISSECTION_LEAF: usize = 8;

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
    /// Computes both a Reverse Cuthill–McKee and a nested-dissection ordering
    /// and returns whichever yields the smaller symbolic Cholesky factor. The
    /// pattern alone determines the result, so callers compute this once and
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

        let rcm = reverse_cuthill_mckee_order(&adjacency);
        let nested = nested_dissection_order(&adjacency);
        let chosen = if symbolic_cholesky_nnz(&adjacency, &nested)
            <= symbolic_cholesky_nnz(&adjacency, &rcm)
        {
            nested
        } else {
            rcm
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
/// Gilbert–Ng–Peyton symbolic factorization). Used only to pick between the two
/// candidate orderings, so it counts blocks, not scalars — block sizes are
/// uniform, so the ranking is the same.
fn symbolic_cholesky_nnz(adjacency: &[Vec<usize>], block_order: &[usize]) -> usize {
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
    // Marking each path once per row counts every L entry exactly once.
    let mut column_count = vec![1usize; n]; // diagonal entries
    let mut mark = vec![usize::MAX; n];
    for (i, row) in pattern.iter().enumerate() {
        mark[i] = i;
        for &j in row {
            if j >= i {
                continue;
            }
            let mut k = j;
            while mark[k] != i {
                column_count[k] += 1;
                mark[k] = i;
                match parent[k] {
                    usize::MAX => break,
                    next => k = next,
                }
            }
        }
    }
    column_count.iter().sum()
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
            for order in [&natural, &rcm, &nested] {
                assert_eq!(
                    symbolic_cholesky_nnz(&adjacency, order),
                    brute_force_fill(&adjacency, order),
                    "grid {rows}x{cols}"
                );
            }
        }
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
    fn fill_reducing_keeps_the_cheaper_candidate() {
        let triplets = grid_triplets(10, 10, 1);
        let adjacency = block_adjacency(100, 1, &triplets);
        let rcm = reverse_cuthill_mckee_order(&adjacency);
        let nested = nested_dissection_order(&adjacency);
        let best =
            symbolic_cholesky_nnz(&adjacency, &rcm).min(symbolic_cholesky_nnz(&adjacency, &nested));

        // Reconstruct the chosen ordering's block sequence and score it.
        let order = Reordering::fill_reducing(100, 1, &triplets);
        let chosen_blocks: Vec<usize> = order.old_of_new.clone();
        assert_eq!(
            symbolic_cholesky_nnz(&adjacency, &chosen_blocks),
            best,
            "fill_reducing must keep whichever candidate has the smaller factor"
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
