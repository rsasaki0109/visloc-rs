//! Hierarchical k-means vocabulary — the leaf-word quantizer underneath
//! COLMAP's vocab-tree retrieval.
//!
//! **What COLMAP's `main` branch actually does today, fetched and read
//! directly (`src/colmap/retrieval/visual_index.cc`, `Quantize`,
//! `VisualIndex::BuildOptions`, 2026-07-17):** COLMAP switched from a
//! FLANN hierarchical-clustering index to a single flat `faiss::Clustering`
//! call in May 2025 — `Quantize()` runs one k-means over the *entire*
//! training set straight to `options.num_visual_words` leaf centroids
//! (default `256 * 256 = 65536`, `num_iterations = 100`, `num_rounds = 3`
//! restarts keeping the best objective), then wraps those centroids in a
//! `faiss::IndexIVF` (`"IVF{2√k},ITQ{d/2},SH"`) purely so that *assigning* a
//! query descriptor to its nearest word(s) is sub-linear in the leaf count.
//! There is **no recursive branching tree at query time in current COLMAP**
//! — "vocab tree" is now a historical name (`vocab_tree_path`,
//! `VocabTreePairGenerator`) for what is architecturally a flat quantizer
//! plus an approximate-nearest-neighbor index over the words.
//!
//! **This module's deliberate deviation**, per the milestone's own
//! "training in-tree is acceptable, document the deviation" allowance: two
//! honest substitutions, made because this repo may not add a new
//! ANN-index dependency (no faiss/FLANN) and targets corpora many orders of
//! magnitude smaller than COLMAP's default 65536-word design point:
//!
//! 1. **Build**: a genuine recursive hierarchical k-means — `branching_factor`
//!    children per node, `depth` levels, exactly the Nister & Stewenius
//!    (CVPR 2006) construction COLMAP used before its faiss migration, and
//!    literally what the milestone brief asked for ("branching factor +
//!    depth"). Reuses this crate's existing deterministic k-means++
//!    (`place_recognition::Vocabulary::build`) at every node instead of a
//!    new clustering implementation — no new linear-algebra code, per the
//!    task's "check what already exists" instruction.
//! 2. **Assignment**: an exact linear scan over the flattened leaf
//!    centroids, rather than an approximate index (faiss/FLANN). COLMAP's
//!    own leaf-assignment step (`FindWordIds`, delegated entirely to
//!    faiss's `IndexIVF::search`) is *itself* approximate at web scale for
//!    speed; at this milestone's scale (tens of images, hundreds-to-low-
//!    thousands of leaf words) an exact scan is fast enough and can only
//!    help ranking quality relative to an approximate one, never hurt it.
//!
//! Neither substitution touches the TF-IDF / Hamming-embedding *scoring*
//! semantics ported faithfully in [`super::index`] — those are unchanged by
//! how a descriptor finds its word.

use crate::place_recognition::Vocabulary;

/// Hierarchical-k-means build configuration.
///
/// `branching_factor`/`depth` are this module's substitute for COLMAP's
/// `VisualIndex::BuildOptions::num_visual_words` (default `256*256`, cited
/// above) — the *desired* leaf count is `branching_factor.pow(depth)`, sized
/// down from COLMAP's web-scale default because a vocabulary that large
/// would leave most words with zero-to-one training samples on a
/// tens-of-images corpus, defeating both IDF weighting (needs several images
/// per word to be informative) and the Hamming-embedding threshold learning
/// (`min_he_entries`, ported from `inverted_index.h`'s `kMinEntries = 5`
/// below). The actual leaf count returned by [`HierarchicalVocabulary::build`]
/// may be less, exactly as COLMAP's own `BuildOptions::num_visual_words` doc
/// comment warns ("the actual number of visual words might be less") —
/// here because a node's descriptor bucket was too small to split further.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HkmBuildOptions {
    /// Children per internal node (COLMAP: implicit in the flat
    /// `num_visual_words` target; no direct default to cite — chosen to
    /// keep per-node k-means calls cheap and each level's cluster count
    /// modest).
    pub branching_factor: usize,
    /// Number of splitting levels. `branching_factor.pow(depth)` is the
    /// requested (upper-bound) leaf count.
    pub depth: usize,
    /// Lloyd iterations per node, passed straight through to
    /// [`Vocabulary::build`].
    pub iterations: usize,
    /// Deterministic seed; every node derives its own sub-seed from this
    /// one plus its position in the tree, so a rebuild with the same seed
    /// and inputs is byte-identical (the milestone's own determinism test).
    pub seed: u64,
}

impl Default for HkmBuildOptions {
    fn default() -> Self {
        Self {
            branching_factor: 10,
            depth: 3,
            iterations: 20,
            seed: 0,
        }
    }
}

/// A hierarchical-k-means vocabulary: the flattened set of leaf-node
/// centroids (see module doc for why assignment is a flat scan over these
/// rather than a tree descent) plus the tree shape it was built with, for
/// reporting.
#[derive(Debug, Clone, PartialEq)]
pub struct HierarchicalVocabulary {
    leaves: Vec<Vec<f32>>,
    dim: usize,
    branching_factor: usize,
    depth: usize,
}

/// Squared Euclidean distance between two equal-length descriptors.
fn sq_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Mean descriptor of a bucket (the base-case leaf centroid when a node
/// cannot be split further).
fn mean_descriptor(descriptors: &[&[f32]], dim: usize) -> Vec<f32> {
    if descriptors.is_empty() {
        return vec![0.0; dim];
    }
    let mut sum = vec![0.0f32; dim];
    for d in descriptors {
        for (s, &x) in sum.iter_mut().zip(d.iter()) {
            *s += x;
        }
    }
    let n = descriptors.len() as f32;
    for s in sum.iter_mut() {
        *s /= n;
    }
    sum
}

impl HierarchicalVocabulary {
    /// Build a hierarchical vocabulary over `descriptors` (all must share one
    /// positive dimension). Returns `None` for empty input, dimension
    /// mismatches, or zero dimension — same degenerate-input contract as
    /// [`Vocabulary::build`].
    pub fn build(descriptors: &[&[f32]], options: &HkmBuildOptions) -> Option<HierarchicalVocabulary> {
        if descriptors.is_empty() {
            return None;
        }
        let dim = descriptors[0].len();
        if dim == 0 || descriptors.iter().any(|d| d.len() != dim) {
            return None;
        }
        let leaves = build_node(
            descriptors,
            options.branching_factor,
            options.depth,
            options.iterations,
            options.seed,
            dim,
        );
        if leaves.is_empty() {
            return None;
        }
        Some(HierarchicalVocabulary {
            leaves,
            dim,
            branching_factor: options.branching_factor,
            depth: options.depth,
        })
    }

    /// Number of leaf visual words (may be less than `branching_factor.pow(depth)`
    /// — see [`HkmBuildOptions`]'s doc comment).
    pub fn num_words(&self) -> usize {
        self.leaves.len()
    }

    /// Local-descriptor dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The requested tree shape this vocabulary was built with (for
    /// reporting; the realized leaf count is [`Self::num_words`]).
    pub fn shape(&self) -> (usize, usize) {
        (self.branching_factor, self.depth)
    }

    /// The `num_neighbors` nearest leaf word ids for `descriptor`, nearest
    /// first (ties broken by ascending word id for determinism). COLMAP's
    /// `IndexOptions::num_neighbors` (index-time, default 1) and
    /// `QueryOptions::num_neighbors` (query-time, default 5) both resolve to
    /// a call like this one, backed there by an approximate `faiss::IndexIVF`
    /// search — here an exact linear scan (module doc, deviation 2).
    pub fn nearest_words(&self, descriptor: &[f32], num_neighbors: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f32)> = self
            .leaves
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_distance(c, descriptor)))
            .collect();
        scored.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(num_neighbors.max(1));
        scored.into_iter().map(|(i, _)| i).collect()
    }
}

/// Recursive tree build: cluster `descriptors` into `branching` groups (via
/// the crate's existing k-means++), then recurse into each non-empty group
/// with one less `depth`. A group that cannot be split further (too few
/// members, `depth` exhausted, or `Vocabulary::build` itself declining
/// degenerate input) becomes exactly one leaf.
fn build_node(
    descriptors: &[&[f32]],
    branching: usize,
    depth: usize,
    iterations: usize,
    seed: u64,
    dim: usize,
) -> Vec<Vec<f32>> {
    if depth == 0 || branching < 2 || descriptors.len() < 2 {
        return vec![mean_descriptor(descriptors, dim)];
    }
    let target_k = branching.min(descriptors.len());
    if target_k < 2 {
        return vec![mean_descriptor(descriptors, dim)];
    }
    let Some(vocab) = Vocabulary::build(descriptors, target_k, iterations, seed) else {
        return vec![mean_descriptor(descriptors, dim)];
    };

    let mut buckets: Vec<Vec<&[f32]>> = vec![Vec::new(); vocab.k()];
    for &d in descriptors {
        let idx = vocab
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, sq_distance(c, d)))
            .fold((0usize, f32::INFINITY), |best, cur| if cur.1 < best.1 { cur } else { best })
            .0;
        buckets[idx].push(d);
    }

    let mut leaves = Vec::new();
    for (i, bucket) in buckets.into_iter().enumerate() {
        if bucket.is_empty() {
            continue;
        }
        // Each bucket gets its own deterministic sub-seed so a rebuild with
        // the same top-level seed reproduces byte-identical leaves, and
        // siblings don't accidentally share a k-means++ trajectory.
        let sub_seed = seed
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .wrapping_add((i as u64).wrapping_mul(0x1000_0000_01B3))
            .wrapping_add(depth as u64);
        if depth == 1 || bucket.len() < 2 {
            // Leaf level (or a bucket too small to split further): this
            // node's own k-means centroid, already the mean of `bucket`
            // after Lloyd's last iteration, is the leaf.
            leaves.push(vocab.centroids[i].clone());
        } else {
            leaves.extend(build_node(&bucket, branching, depth - 1, iterations, sub_seed, dim));
        }
    }
    leaves
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(dim: usize, i: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; dim];
        w[i] = 1.0;
        w
    }

    /// Deterministic jittered cluster around `center`.
    fn cluster(center: &[f32], n: usize, jitter: f32, seed: u64) -> Vec<Vec<f32>> {
        // Tiny local LCG, independent of place_recognition's private one —
        // determinism is all this needs, not cryptographic quality.
        struct Lcg(u64);
        impl Lcg {
            fn next(&mut self) -> f32 {
                self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
            }
        }
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| center.iter().map(|&c| c + (rng.next() - 0.5) * 2.0 * jitter).collect())
            .collect()
    }

    #[test]
    fn build_is_deterministic_given_the_same_seed() {
        let dim = 8;
        let mut pool: Vec<Vec<f32>> = Vec::new();
        for i in 0..8 {
            pool.extend(cluster(&word(dim, i), 40, 0.05, 100 + i as u64));
        }
        let refs: Vec<&[f32]> = pool.iter().map(|v| v.as_slice()).collect();
        let opts = HkmBuildOptions {
            branching_factor: 4,
            depth: 2,
            iterations: 15,
            seed: 42,
        };
        let v1 = HierarchicalVocabulary::build(&refs, &opts).unwrap();
        let v2 = HierarchicalVocabulary::build(&refs, &opts).unwrap();
        assert_eq!(v1, v2, "same seed + same input must reproduce byte-identical leaves");
        assert!(v1.num_words() >= 2, "expected a genuine multi-leaf split, got {}", v1.num_words());
    }

    #[test]
    fn build_recovers_well_separated_clusters_as_distinct_leaves() {
        let dim = 6;
        let centers: Vec<Vec<f32>> = (0..6).map(|i| word(dim, i)).collect();
        let mut pool: Vec<Vec<f32>> = Vec::new();
        for (i, c) in centers.iter().enumerate() {
            pool.extend(cluster(c, 50, 0.03, 500 + i as u64));
        }
        let refs: Vec<&[f32]> = pool.iter().map(|v| v.as_slice()).collect();
        let opts = HkmBuildOptions {
            branching_factor: 3,
            depth: 2,
            iterations: 25,
            seed: 7,
        };
        let vocab = HierarchicalVocabulary::build(&refs, &opts).unwrap();
        // Every true cluster centre should be well matched by some leaf.
        for c in &centers {
            let nearest = (0..vocab.num_words())
                .map(|i| sq_distance(c, &vocab.leaves[i]))
                .fold(f32::INFINITY, f32::min);
            assert!(nearest < 0.05, "centre {c:?} unmatched by any leaf (d^2={nearest})");
        }
    }

    #[test]
    fn nearest_words_returns_requested_count_nearest_first() {
        let dim = 4;
        let mut pool: Vec<Vec<f32>> = Vec::new();
        for i in 0..4 {
            pool.extend(cluster(&word(dim, i), 30, 0.02, 200 + i as u64));
        }
        let refs: Vec<&[f32]> = pool.iter().map(|v| v.as_slice()).collect();
        let opts = HkmBuildOptions {
            branching_factor: 2,
            depth: 2,
            iterations: 20,
            seed: 3,
        };
        let vocab = HierarchicalVocabulary::build(&refs, &opts).unwrap();
        let probe = word(dim, 0);
        let top2 = vocab.nearest_words(&probe, 2);
        assert_eq!(top2.len(), 2);
        let d0 = sq_distance(&vocab.leaves[top2[0]], &probe);
        let d1 = sq_distance(&vocab.leaves[top2[1]], &probe);
        assert!(d0 <= d1, "results must be nearest-first (d0={d0} d1={d1})");
    }

    #[test]
    fn degenerate_inputs_return_none() {
        let opts = HkmBuildOptions::default();
        assert!(HierarchicalVocabulary::build(&[], &opts).is_none());
        let d = [vec![1.0f32, 2.0], vec![1.0, 2.0]];
        let mismatched: Vec<&[f32]> = vec![&d[0][..], &[1.0f32][..]];
        assert!(HierarchicalVocabulary::build(&mismatched, &opts).is_none());
    }
}
