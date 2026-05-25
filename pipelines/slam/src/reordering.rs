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
//! same system — only the sparsity of `L` changes. We pick `P` with
//! [Reverse Cuthill–McKee][rcm] (RCM): a band-minimizing breadth-first ordering
//! over the *block-adjacency* graph of the variables. Clustering each
//! variable's neighbours near it on the diagonal keeps `L` close to banded,
//! which is exactly the structure Cholesky preserves without fill.
//!
//! The reordering is purely structural and fully deterministic (ties broken by
//! ascending node id), so it preserves the solver's bit-for-bit reproducibility
//! while leaving the numerical answer unchanged up to floating-point summation
//! order within the factorization.
//!
//! [rcm]: https://en.wikipedia.org/wiki/Cuthill%E2%80%93McKee_algorithm

use std::collections::{BTreeSet, VecDeque};

use nalgebra::DVector;

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
    /// Reverse Cuthill–McKee ordering derived from the structural (symmetric)
    /// nonzero pattern of `triplets`, treating each contiguous run of
    /// `block_size` dimensions as a single graph node. `dim` must be a multiple
    /// of `block_size`. Triplet values are ignored; only `(row, col)` positions
    /// matter, and the pattern is symmetrized (an entry at `(r, c)` induces an
    /// edge in both directions).
    pub(crate) fn reverse_cuthill_mckee(
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

        // Block-adjacency graph: an edge between two distinct blocks whenever
        // the system couples them through any off-diagonal nonzero.
        let mut adjacency: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
        for &(row, col, _) in triplets {
            let (br, bc) = (row / block_size, col / block_size);
            if br != bc {
                adjacency[br].insert(bc);
                adjacency[bc].insert(br);
            }
        }
        let degree: Vec<usize> = adjacency.iter().map(BTreeSet::len).collect();

        // Cuthill–McKee: breadth-first over every connected component, rooting
        // each at the lowest-degree unvisited node and expanding neighbours in
        // ascending (degree, id) order.
        let mut visited = vec![false; n];
        let mut block_order: Vec<usize> = Vec::with_capacity(n);
        loop {
            let root = (0..n)
                .filter(|&i| !visited[i])
                .min_by_key(|&i| (degree[i], i));
            let Some(root) = root else { break };

            visited[root] = true;
            let mut queue = VecDeque::new();
            queue.push_back(root);
            while let Some(node) = queue.pop_front() {
                block_order.push(node);
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
        block_order.reverse();

        // Expand the block ordering to a dimension-level permutation, keeping
        // each block's coordinates contiguous and in their original sub-order.
        let mut old_of_new = Vec::with_capacity(dim);
        for &old_block in &block_order {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permutation_is_a_bijection() {
        // A 4-block chain (block_size 2): 0-1-2-3.
        let triplets = chain_triplets(4, 2);
        let order = Reordering::reverse_cuthill_mckee(8, 2, &triplets);

        let mut seen = vec![false; 8];
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
        let triplets = chain_triplets(3, 3);
        let order = Reordering::reverse_cuthill_mckee(9, 3, &triplets);
        // Every group of 3 consecutive permuted dims must come from one block.
        for chunk in order.old_of_new.chunks(3) {
            let block = chunk[0] / 3;
            assert_eq!(chunk, &[block * 3, block * 3 + 1, block * 3 + 2]);
        }
    }

    #[test]
    fn permuting_and_solving_matches_the_original_system() {
        // Build a small SPD system H x = b, solve it permuted, and confirm the
        // restored solution equals the directly-solved one.
        use nalgebra::DMatrix;

        // 3 blocks of size 2; H is a banded SPD matrix (chain coupling).
        let dim = 6;
        let block_size = 2;
        let mut h = DMatrix::<f64>::zeros(dim, dim);
        for k in 0..dim {
            h[(k, k)] = 4.0;
        }
        // Couple neighbouring blocks with -1 on matching coordinates.
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for block in 0..2 {
            for k in 0..block_size {
                let (a, b) = (block * block_size + k, (block + 1) * block_size + k);
                h[(a, b)] = -1.0;
                h[(b, a)] = -1.0;
                triplets.push((a, b, -1.0));
                triplets.push((b, a, -1.0));
            }
        }
        for k in 0..dim {
            triplets.push((k, k, 4.0));
        }
        let b = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let x_direct = h.clone().lu().solve(&b).expect("SPD system is solvable");

        let order = Reordering::reverse_cuthill_mckee(dim, block_size, &triplets);
        // Assemble the permuted dense matrix from the permuted triplets.
        let permuted_triplets = order.permute_triplets(&triplets);
        let mut h_perm = DMatrix::<f64>::zeros(dim, dim);
        for (r, c, v) in permuted_triplets {
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
    fn reverses_a_pathologically_ordered_arrow() {
        // An "arrow" graph where node 0 connects to all others is a classic
        // fill-in trap when eliminated first; RCM should push the high-degree
        // hub to the *end* of the elimination order (i.e. early in old_of_new).
        let n = 6;
        let block_size = 1;
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for k in 0..n {
            triplets.push((k, k, 2.0));
        }
        for spoke in 1..n {
            triplets.push((0, spoke, -1.0));
            triplets.push((spoke, 0, -1.0));
        }
        let order = Reordering::reverse_cuthill_mckee(n, block_size, &triplets);
        // Eliminating the hub early (in the new order, position 0 is eliminated
        // first) would fill in every spoke pair. Cuthill–McKee roots at a
        // low-degree spoke and reaches the hub only after it, so reversing puts
        // the hub among the last variables eliminated — never near the front.
        let hub_position = order.new_of_old[0];
        assert!(
            hub_position >= n - 2,
            "hub should be eliminated late, got position {hub_position} of {n}"
        );
    }

    /// A linear chain of `blocks` nodes coupled on every coordinate, returned as
    /// COO triplets including the diagonal.
    fn chain_triplets(blocks: usize, block_size: usize) -> Vec<(usize, usize, f64)> {
        let dim = blocks * block_size;
        let mut triplets: Vec<(usize, usize, f64)> = Vec::new();
        for k in 0..dim {
            triplets.push((k, k, 2.0));
        }
        for block in 0..blocks.saturating_sub(1) {
            for k in 0..block_size {
                let (a, b) = (block * block_size + k, (block + 1) * block_size + k);
                triplets.push((a, b, -1.0));
                triplets.push((b, a, -1.0));
            }
        }
        triplets
    }
}
