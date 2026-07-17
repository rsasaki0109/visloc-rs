//! `VocabTreePairGenerator`-equivalent: turn a finalized [`VocabTree`] into a
//! deduplicated, symmetric candidate-pair stream, the alternative pair
//! source this milestone adds alongside flat-VLAD top-K
//! (`place_recognition::retrieve_mutual`, used by
//! `examples/unordered_sfm_demo.rs`'s existing `candidate_pairs`).
//!
//! Ported from `src/colmap/controllers/pairing.h`/`pairing.cc`
//! (`VocabTreePairingOptions`, `VocabTreePairGenerator`), BSD-3-Clause, ETH
//! Zurich / UNC Chapel Hill, fetched 2026-07-17. COLMAP's own class streams
//! pairs through a thread pool against a `FeatureMatcherCache`/`Database`
//! (`query_idx_`/`result_idx_`, `Next()` batches) because it targets
//! collections too large to hold every image's descriptors in memory at
//! once; this repo's unordered-SfM demo already loads every image's full
//! `FeatureSet` up front (`load_images`), so [`generate_pairs`] is the
//! synchronous, in-memory equivalent of what `VocabTreePairGenerator::Next()`
//! produces, without the streaming/threading machinery COLMAP needs at its
//! scale and this repo's ETH3D-sized acceptance benchmark does not.

use std::collections::BTreeSet;

use super::index::VocabTree;

/// `VocabTreePairingOptions` — only the fields this port's in-memory,
/// non-streaming pair generator needs. `num_images`'s COLMAP default (100,
/// `controllers/pairing.h`) is kept as this struct's own default so a caller
/// who doesn't override it gets COLMAP-parity behavior; note that on
/// small collections (COLMAP's own docs assume thousands-of-images corpora)
/// `num_images = 100` may exceed the collection size, in which case every
/// other image becomes a candidate — effectively exhaustive, and worth
/// overriding to something like the VLAD path's `--retrieval-topk` for an
/// apples-to-apples pair-budget comparison on small benchmarks (see this
/// milestone's ETH3D acceptance experiment in `docs/colmap_port_plan.md`).
///
/// Not ported: `num_nearest_neighbors`/`num_checks` (COLMAP's per-query-word
/// assignment tuning — this port exposes the same knob directly as
/// [`super::index::VocabTreeOptions::query_num_neighbors`], not duplicated
/// here), `vocab_tree_path`/`match_list_path` (file-based COLMAP CLI
/// plumbing, not applicable to an in-memory Rust API), `max_num_features`
/// (COLMAP's per-image feature cap for very dense images — this repo's
/// features are already capped by the SuperPoint extractor's own
/// `max_keypoints`, upstream of this module).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VocabTreePairGeneratorOptions {
    /// Number of top-scoring images retrieved per query image. COLMAP
    /// default 100 (`VocabTreePairingOptions::num_images`,
    /// `controllers/pairing.h`).
    pub num_images: usize,
}

impl Default for VocabTreePairGeneratorOptions {
    fn default() -> Self {
        Self { num_images: 100 }
    }
}

/// Generate symmetric, deduplicated candidate pairs `(i, j)` with `i < j`
/// from a fully-populated, [`VocabTree::finalize`]d tree: for each image,
/// query the tree with its own descriptors and keep the top
/// `options.num_images` most-similar *other* images — COLMAP's
/// `VocabTreePairGenerator::Next()`, minus the batching/thread-pool
/// machinery (module doc). `image_descriptors[i]` must be the exact
/// descriptor set `image_id == i` was added to `vt` with (same indexing
/// convention the rest of this milestone's demo integration uses).
///
/// A pair from `i`'s own top-N is kept even if `j` does not reciprocally
/// retrieve `i` in its own top-N — COLMAP's own generator is one-directional
/// per query image, not a mutual-nearest-neighbor filter like
/// `place_recognition::retrieve_mutual`; the symmetric `BTreeSet` here only
/// dedups the `(i, j)`/`(j, i)` pair identity, it does not require mutuality.
pub fn generate_pairs(vt: &VocabTree, image_descriptors: &[Vec<Vec<f32>>], options: &VocabTreePairGeneratorOptions) -> Vec<(usize, usize)> {
    let mut set: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (i, descs) in image_descriptors.iter().enumerate() {
        if descs.is_empty() {
            continue;
        }
        let scores = vt.query(descs, None);
        let mut kept = 0usize;
        for s in scores {
            if s.image_id == i {
                continue; // never propose an image paired with itself
            }
            set.insert((i.min(s.image_id), i.max(s.image_id)));
            kept += 1;
            if kept >= options.num_images {
                break;
            }
        }
    }
    set.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab_tree::hkm::HkmBuildOptions;
    use crate::vocab_tree::index::VocabTreeOptions;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
        }
    }
    fn cluster(center: &[f32], n: usize, jitter: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| center.iter().map(|&c| c + (rng.next() - 0.5) * 2.0 * jitter).collect())
            .collect()
    }
    fn word(dim: usize, i: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; dim];
        w[i] = 1.0;
        w
    }

    /// Six images: three near-duplicate pairs of "places" (0<->1, 2<->3,
    /// 4<->5), each place pair sharing almost all its visual content. The
    /// generator should propose every same-place pair (from at least one
    /// query direction) with no duplicate `(i, j)`/`(j, i)` entries and no
    /// `(i, i)` self-pair.
    #[test]
    fn generate_pairs_is_deduplicated_and_symmetric() {
        let dim = 12;
        let places: Vec<Vec<Vec<f32>>> = (0..3)
            .flat_map(|p| {
                let a = {
                    let mut img = cluster(&word(dim, p * 2), 20, 0.02, 100 + p as u64 * 10);
                    img.extend(cluster(&word(dim, p * 2 + 1), 20, 0.02, 200 + p as u64 * 10));
                    img
                };
                let b = {
                    let mut img = cluster(&word(dim, p * 2), 20, 0.02, 300 + p as u64 * 10);
                    img.extend(cluster(&word(dim, p * 2 + 1), 20, 0.02, 400 + p as u64 * 10));
                    img
                };
                vec![a, b]
            })
            .collect();

        let training: Vec<&[f32]> = places.iter().flatten().map(|v| v.as_slice()).collect();
        let hkm = HkmBuildOptions {
            branching_factor: 3,
            depth: 2,
            iterations: 15,
            seed: 5,
        };
        let opts = VocabTreeOptions {
            embedding_dim: 8,
            min_he_entries: 3,
            ..VocabTreeOptions::default()
        };
        let mut tree = VocabTree::build(&training, &hkm, &opts).unwrap();
        for (i, img) in places.iter().enumerate() {
            tree.add_image(i, img);
        }
        tree.finalize();

        let pairs = generate_pairs(&tree, &places, &VocabTreePairGeneratorOptions { num_images: 5 });

        // No self-pairs, no duplicates (BTreeSet already guarantees dedup;
        // this also checks i < j ordering held for every entry).
        assert!(pairs.iter().all(|&(i, j)| i < j), "every pair must be normalized i<j: {pairs:?}");

        // Each planted same-place pair should appear.
        for p in 0..3 {
            let (a, b) = (p * 2, p * 2 + 1);
            assert!(
                pairs.contains(&(a, b)),
                "same-place pair ({a},{b}) should be proposed; got {pairs:?}"
            );
        }
    }

    #[test]
    fn generate_pairs_respects_num_images_budget() {
        let dim = 8;
        let images: Vec<Vec<Vec<f32>>> = (0..6)
            .map(|i| cluster(&word(dim, i % dim), 15, 0.05, 700 + i as u64))
            .collect();
        let training: Vec<&[f32]> = images.iter().flatten().map(|v| v.as_slice()).collect();
        let hkm = HkmBuildOptions {
            branching_factor: 2,
            depth: 2,
            iterations: 10,
            seed: 9,
        };
        let opts = VocabTreeOptions {
            embedding_dim: 8,
            min_he_entries: 2,
            ..VocabTreeOptions::default()
        };
        let mut tree = VocabTree::build(&training, &hkm, &opts).unwrap();
        for (i, img) in images.iter().enumerate() {
            tree.add_image(i, img);
        }
        tree.finalize();

        // With num_images = 1, each of the 6 images contributes at most one
        // pair, so the union can have at most 6 pairs.
        let pairs = generate_pairs(&tree, &images, &VocabTreePairGeneratorOptions { num_images: 1 });
        assert!(pairs.len() <= 6, "expected <=6 pairs with a 1-per-query budget, got {}", pairs.len());
    }
}
