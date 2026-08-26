//! Inverted-file TF-IDF + Hamming-embedding scoring over a
//! [`super::hkm::HierarchicalVocabulary`] — a faithful port of the scoring
//! COLMAP's retrieval module actually implements.
//!
//! Ported, with citations per feature, from (BSD-3-Clause, ETH Zurich / UNC
//! Chapel Hill, `github.com/colmap/colmap`, `main`, fetched 2026-07-17):
//!
//! - `src/colmap/retrieval/inverted_file.h` — `InvertedFile<kEmbeddingDim>`:
//!   `ComputeIDFWeight` (`idf = ln(N / n_docs_with_word)`, squared and
//!   cached as `squared_idf_weight_`), `ComputeHammingEmbedding` (per-word
//!   per-dimension **median** threshold learned from the training sample
//!   assigned to that word, skipped — "the closest thing to a stop-word
//!   mechanism in this codebase" per the task's own ask — for any word with
//!   fewer than `kMinEntries = 5` training members), `ScoreFeature` (the
//!   exact per-word voting loop this module's [`InvertedFile::score_feature`]
//!   reproduces, including the **burstiness normalization**
//!   `score /= sqrt(num_image_votes)` before applying the word's squared IDF
//!   weight — Eqn. 2, Arandjelović & Zisserman, "Scalable descriptor
//!   distinctiveness for location recognition", ACCV 2014), and
//!   `ComputeImageSelfSimilarities` (self-similarity accumulates
//!   `squared_idf_weight_` once **per entry**, i.e. per feature-occurrence in
//!   that word, not deduplicated per unique word — ported exactly as-is in
//!   [`VocabTree::finalize`]).
//! - `src/colmap/retrieval/inverted_index.h` — `InvertedIndex`: `Query`
//!   (per-descriptor per-assigned-word scoring, accumulated across words
//!   into one score per image, then normalized by
//!   `1/sqrt(query_self_similarity) * 1/sqrt(db_image_self_similarity)` — the
//!   two-sided cosine-similarity-style normalization this module's
//!   [`VocabTree::query`] reproduces) and `GenerateHammingEmbeddingProjection`
//!   (a random matrix's `Q` factor from a full `QR` decomposition, taking
//!   `kEmbeddingDim` of its rows as an orthonormal projection basis — this
//!   module's [`generate_orthonormal_projection`] reaches the same
//!   *property* — an orthonormal row set — via incremental Gram-Schmidt
//!   instead of one square QR, needing no new linear-algebra dependency).
//! - `src/colmap/retrieval/utils.h` — `HammingDistWeightFunctor<N, kSigma>`:
//!   `weight(h) = exp(-h²/σ²)` for `h ≤ ⌊1.5σ⌋`, else `0`; `σ = 16` default —
//!   ported verbatim as [`HammingWeightLut`].
//! - `src/colmap/retrieval/visual_index.h` — `VisualIndex::Create`'s default
//!   `embedding_dim = 64` (COLMAP's SIFT path also defaults `desc_dim = 128`;
//!   this repo's SuperPoint descriptors are 256-dim float, not 128-dim
//!   `uint8` SIFT — the embedding dimension is independent of the input
//!   descriptor dimension in COLMAP's own design, `kEmbeddingDim` is a
//!   separate template parameter from `kDescDim`, so keeping COLMAP's literal
//!   default of 64 rather than deriving a new ratio is the direct, cited
//!   choice, not a new one).
//!
//! **Not ported** (the milestone's own "or document as follow-up"
//! allowance): `vote_and_verify.h/.cc` (RANSAC-free spatial re-ranking) is
//! gated behind `QueryOptions::num_images_after_verification`, whose COLMAP
//! default is `0` (disabled) — this module simply never implements the
//! disabled-by-default path, so no behavior is lost relative to a
//! default-configured COLMAP. Binary serialization (`Read`/`Write`) is not
//! ported either: this milestone builds the tree fresh from the target
//! collection's own descriptors every run (see the milestone's own
//! "training in-tree is acceptable" allowance), so there is nothing to
//! persist across runs yet.

use std::collections::HashMap;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use super::hkm::{HierarchicalVocabulary, HkmBuildOptions};

/// One scored image from a [`VocabTree::query`] — COLMAP's
/// `retrieval::ImageScore` (`utils.h`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageScore {
    pub image_id: usize,
    pub score: f32,
}

/// Deterministic work counters from one [`VocabTree::query_with_work`]
/// call. They exist so retrieval-cost scaling can be asserted without
/// depending on wall-clock noise: `leaf_distance_computations` is the
/// corpus-size-independent word-assignment cost, while `entries_visited`
/// is the corpus-size-dependent inverted-file cost.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct QueryWorkStats {
    /// Leaf-centroid distance computations performed while assigning query
    /// descriptors to visual words (well-formed input: `descriptors ×
    /// num_words`).
    pub leaf_distance_computations: usize,
    /// Inverted-file entries scored across every assigned word.
    pub entries_visited: usize,
}

/// Configuration for [`VocabTree`]. Field-for-field citations in the module
/// doc comment above; defaults reproduce COLMAP's own where one exists.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VocabTreeOptions {
    /// Hamming-embedding binary-signature length (`kEmbeddingDim`).
    /// COLMAP default 64 (`VisualIndex::Create(desc_dim=128, embedding_dim=64)`).
    pub embedding_dim: usize,
    /// Hamming-distance-to-weight Gaussian width (`kSigma` in
    /// `HammingDistWeightFunctor`). COLMAP default 16.
    pub hamming_sigma: f32,
    /// Visual words assigned per descriptor when **indexing** an image
    /// (COLMAP `IndexOptions::num_neighbors`, default 1 — hard assignment).
    pub index_num_neighbors: usize,
    /// Visual words assigned per descriptor when **querying** (COLMAP
    /// `QueryOptions::num_neighbors`, default 5 — soft assignment: a query
    /// descriptor near a word boundary still finds db features indexed under
    /// the neighboring word).
    pub query_num_neighbors: usize,
    /// Minimum training samples a word needs before its Hamming-embedding
    /// threshold is learned (COLMAP `inverted_index.h`'s `kMinEntries = 5`).
    /// Words below this keep an all-zero threshold vector (COLMAP's own
    /// default-constructed `thresholds_`), same as COLMAP.
    pub min_he_entries: usize,
    /// Deterministic seed for the random Hamming-embedding projection basis.
    pub projection_seed: u64,
}

impl Default for VocabTreeOptions {
    fn default() -> Self {
        Self {
            embedding_dim: 64,
            hamming_sigma: 16.0,
            index_num_neighbors: 1,
            query_num_neighbors: 5,
            min_he_entries: 5,
            projection_seed: 1,
        }
    }
}

/// `weight(h) = exp(-h²/σ²)` for `h ≤ ⌊1.5σ⌋`, else `0` — verbatim port of
/// `HammingDistWeightFunctor` (`retrieval/utils.h`), precomputed as a
/// look-up table exactly as COLMAP does.
#[derive(Debug, Clone)]
struct HammingWeightLut {
    lut: Vec<f32>,
    max_distance: usize,
}

impl HammingWeightLut {
    fn new(embedding_dim: usize, sigma: f32) -> Self {
        let max_distance = (1.5 * sigma) as usize;
        let sigma_sq = sigma * sigma;
        let lut = (0..=embedding_dim)
            .map(|h| {
                if h <= max_distance {
                    (-(h as f32) * (h as f32) / sigma_sq).exp()
                } else {
                    0.0
                }
            })
            .collect();
        Self { lut, max_distance }
    }

    fn weight(&self, h: usize) -> f32 {
        self.lut.get(h).copied().unwrap_or(0.0)
    }
}

/// One inverted-file entry: a database feature's image/feature index plus
/// its binary (Hamming-embedded) signature — COLMAP's `InvertedFileEntry`
/// (`retrieval/inverted_file_entry.h`), minus the `geometry` (keypoint x/y/
/// scale/orientation) field, which only feeds `vote_and_verify`'s spatial
/// re-ranking — not ported, see the module doc.
#[derive(Debug, Clone)]
struct InvertedFileEntry {
    image_id: usize,
    binary: Vec<u64>,
}

fn binarize(projected: &[f32], thresholds: &[f32]) -> Vec<u64> {
    let words = thresholds.len().div_ceil(64);
    let mut bits = vec![0u64; words.max(1)];
    for (i, (&p, &t)) in projected.iter().zip(thresholds).enumerate() {
        if p > t {
            bits[i / 64] |= 1u64 << (i % 64);
        }
    }
    bits
}

fn hamming_distance(a: &[u64], b: &[u64]) -> usize {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

fn median(sorted: &[f32]) -> f32 {
    let n = sorted.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

/// One visual word's inverted file — COLMAP's `InvertedFile<kEmbeddingDim>`.
#[derive(Debug, Clone)]
struct InvertedFile {
    entries: Vec<InvertedFileEntry>,
    thresholds: Vec<f32>,
    squared_idf_weight: f32,
    sorted: bool,
}

impl InvertedFile {
    fn new(embedding_dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            thresholds: vec![0.0; embedding_dim],
            squared_idf_weight: 0.0,
            sorted: false,
        }
    }

    /// `ComputeHammingEmbedding`: per-dimension median threshold over the
    /// training sample assigned to this word (`inverted_file.h:276-293`).
    fn compute_hamming_embedding(&mut self, projected_samples: &[Vec<f32>]) {
        let dim = self.thresholds.len();
        for d in 0..dim {
            let mut vals: Vec<f32> = projected_samples.iter().map(|p| p[d]).collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            self.thresholds[d] = median(&vals);
        }
    }

    fn add_entry(&mut self, image_id: usize, projected: &[f32]) {
        let binary = binarize(projected, &self.thresholds);
        self.entries.push(InvertedFileEntry { image_id, binary });
        self.sorted = false;
    }

    fn sort_entries(&mut self) {
        self.entries.sort_by_key(|e| e.image_id);
        self.sorted = true;
    }

    /// `ComputeIDFWeight`: `idf = ln(N / n_docs_with_word)`, squared and
    /// cached (`inverted_file.h:256-269`).
    fn compute_idf_weight(&mut self, num_total_images: usize) {
        if self.entries.is_empty() || num_total_images == 0 {
            self.squared_idf_weight = 0.0;
            return;
        }
        let unique: std::collections::HashSet<usize> =
            self.entries.iter().map(|e| e.image_id).collect();
        let idf = (num_total_images as f64 / unique.len() as f64).ln();
        self.squared_idf_weight = (idf * idf) as f32;
    }

    /// `ComputeImageSelfSimilarities`: adds this word's squared IDF weight
    /// once per **entry** (per feature-occurrence), not deduplicated per
    /// unique image (`inverted_file.h:364-370`).
    fn accumulate_self_similarities(&self, acc: &mut HashMap<usize, f64>) {
        for e in &self.entries {
            *acc.entry(e.image_id).or_insert(0.0) += self.squared_idf_weight as f64;
        }
    }

    /// `ScoreFeature`: burstiness-normalized, IDF-weighted per-image voting
    /// for one query feature already assigned to this word
    /// (`inverted_file.h:296-355`). Requires [`Self::sort_entries`] to have
    /// run (COLMAP: `IsUsable`/`kEntriesSorted`), asserted via `debug_assert`.
    fn score_feature(&self, query_projected: &[f32], lut: &HammingWeightLut) -> Vec<(usize, f32)> {
        debug_assert!(
            self.sorted,
            "score_feature requires sorted entries (call finalize() first)"
        );
        if self.entries.is_empty() {
            return Vec::new();
        }
        let query_bits = binarize(query_projected, &self.thresholds);

        let mut out = Vec::new();
        let mut cur_image = self.entries[0].image_id;
        let mut cur_score = 0.0f32;
        let mut cur_votes = 0usize;
        let flush = |image_id: usize, score: f32, votes: usize, out: &mut Vec<(usize, f32)>| {
            if votes > 0 {
                let s = (score / (votes as f32).sqrt()) * self.squared_idf_weight;
                out.push((image_id, s));
            }
        };
        for e in &self.entries {
            if e.image_id != cur_image {
                flush(cur_image, cur_score, cur_votes, &mut out);
                cur_image = e.image_id;
                cur_score = 0.0;
                cur_votes = 0;
            }
            let hd = hamming_distance(&query_bits, &e.binary);
            if hd <= lut.max_distance {
                cur_score += lut.weight(hd);
                cur_votes += 1;
            }
        }
        flush(cur_image, cur_score, cur_votes, &mut out);
        out
    }
}

/// A random projection basis with orthonormal rows (`embedding_dim × dim`),
/// built incrementally via modified Gram-Schmidt over Box-Muller Gaussian
/// vectors. Reaches the same property COLMAP's `GenerateHammingEmbedding
/// Projection` uses a square-matrix `QR` factorization for (any subset of
/// full rows of an orthogonal matrix is itself an orthonormal set — Eigen's
/// `colPivHouseholderQr().matrixQ()` followed by `topRows<kEmbeddingDim>()`),
/// without a new dependency on this crate's linear-algebra stack for a
/// generic QR.
fn generate_orthonormal_projection(dim: usize, embedding_dim: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(
        embedding_dim <= dim,
        "embedding_dim ({embedding_dim}) must not exceed the descriptor dimension ({dim})"
    );
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(embedding_dim);
    while rows.len() < embedding_dim {
        let mut v = gaussian_vector(dim, &mut rng);
        for r in &rows {
            let dot: f32 = v.iter().zip(r.iter()).map(|(a, b)| a * b).sum();
            for (vi, ri) in v.iter_mut().zip(r.iter()) {
                *vi -= dot * ri;
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-6 {
            for x in v.iter_mut() {
                *x /= norm;
            }
            rows.push(v);
        }
        // else: degenerate draw (near-linear-dependence); redraw silently.
    }
    rows
}

/// Standard-normal sample via the Box-Muller transform, from this crate's
/// existing `rand`/`SmallRng` dependency (already used by
/// `two_view::mod`'s RANSAC sampling) — no new dependency (e.g. `rand_distr`)
/// needed for a Gaussian.
fn gaussian_vector(dim: usize, rng: &mut SmallRng) -> Vec<f32> {
    (0..dim)
        .map(|_| {
            let u1: f64 = rng.gen_range(1e-12..1.0);
            let u2: f64 = rng.gen_range(0.0..1.0);
            ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
        })
        .collect()
}

fn project(proj_matrix: &[Vec<f32>], descriptor: &[f32]) -> Vec<f32> {
    proj_matrix
        .iter()
        .map(|row| row.iter().zip(descriptor.iter()).map(|(a, b)| a * b).sum())
        .collect()
}

/// A vocab-tree-style retrieval index: [`HierarchicalVocabulary`] word
/// assignment feeding a per-word TF-IDF + Hamming-embedding inverted file —
/// COLMAP's `VisualIndex`/`InvertedIndex` combined into one type (this port
/// has no GPU/CPU backend split to preserve, unlike COLMAP's
/// `FaissVisualIndex` vs. legacy-flann distinction).
pub struct VocabTree {
    vocab: HierarchicalVocabulary,
    proj_matrix: Vec<Vec<f32>>,
    files: Vec<InvertedFile>,
    normalization_constants: HashMap<usize, f32>,
    weight_lut: HammingWeightLut,
    options: VocabTreeOptions,
    image_ids: std::collections::HashSet<usize>,
    finalized: bool,
}

impl VocabTree {
    /// `Build`: train the hierarchical vocabulary, generate the Hamming-
    /// embedding projection, and learn per-word thresholds from
    /// `training_descriptors` — mirrors `VisualIndex::Build`'s own sequence
    /// (`visual_index.cc:433-462`): quantize, `GenerateHammingEmbedding
    /// Projection`, assign the *same* training descriptors to words with
    /// `num_neighbors = 1` (COLMAP hardcodes this at build time,
    /// `kNumNeighbors = 1` local to `Build`, independent of
    /// `IndexOptions`/`QueryOptions`), then `ComputeHammingEmbedding`.
    pub fn build(
        training_descriptors: &[&[f32]],
        hkm_options: &HkmBuildOptions,
        options: &VocabTreeOptions,
    ) -> Option<VocabTree> {
        let vocab = HierarchicalVocabulary::build(training_descriptors, hkm_options)?;
        if options.embedding_dim == 0 || options.embedding_dim > vocab.dim() {
            return None;
        }
        let proj_matrix = generate_orthonormal_projection(
            vocab.dim(),
            options.embedding_dim,
            options.projection_seed,
        );

        let mut files: Vec<InvertedFile> = (0..vocab.num_words())
            .map(|_| InvertedFile::new(options.embedding_dim))
            .collect();

        // Build-time word assignment always uses num_neighbors = 1
        // (`visual_index.cc`'s local `kNumNeighbors`), independent of
        // `options.index_num_neighbors` (which governs `add_image`, i.e.
        // COLMAP's separate `IndexOptions::num_neighbors`).
        let mut per_word_projected: Vec<Vec<Vec<f32>>> = vec![Vec::new(); vocab.num_words()];
        for &d in training_descriptors {
            let word = vocab.nearest_words(d, 1)[0];
            per_word_projected[word].push(project(&proj_matrix, d));
        }
        for (i, samples) in per_word_projected.into_iter().enumerate() {
            if samples.len() >= options.min_he_entries {
                files[i].compute_hamming_embedding(&samples);
            }
        }

        Some(VocabTree {
            vocab,
            proj_matrix,
            files,
            normalization_constants: HashMap::new(),
            weight_lut: HammingWeightLut::new(options.embedding_dim, options.hamming_sigma),
            options: *options,
            image_ids: std::collections::HashSet::new(),
            finalized: false,
        })
    }

    pub fn num_words(&self) -> usize {
        self.vocab.num_words()
    }

    pub fn num_images(&self) -> usize {
        self.image_ids.len()
    }

    pub fn is_image_indexed(&self, image_id: usize) -> bool {
        self.image_ids.contains(&image_id)
    }

    /// `Add`: assign each of `descriptors` to its nearest
    /// `options.index_num_neighbors` word(s) and append an entry to each.
    /// A no-op if `image_id` is already indexed (COLMAP: same guard,
    /// `visual_index.cc:122-125`). Invalidates [`Self::finalize`] until
    /// called again.
    pub fn add_image(&mut self, image_id: usize, descriptors: &[Vec<f32>]) {
        if self.is_image_indexed(image_id) {
            return;
        }
        self.image_ids.insert(image_id);
        self.finalized = false;
        for d in descriptors {
            if d.len() != self.vocab.dim() {
                continue;
            }
            let proj = project(&self.proj_matrix, d);
            for word in self
                .vocab
                .nearest_words(d, self.options.index_num_neighbors)
            {
                self.files[word].add_entry(image_id, &proj);
            }
        }
    }

    /// `Prepare`: sort each word's entries, then compute IDF weights and
    /// per-image normalization constants over the now-complete corpus
    /// (`InvertedIndex::Finalize`/`ComputeWeightsAndNormalizationConstants`,
    /// `inverted_index.h:166-174, 423-449`).
    pub fn finalize(&mut self) {
        for f in &mut self.files {
            f.sort_entries();
        }
        let num_total_images = self.image_ids.len();
        for f in &mut self.files {
            f.compute_idf_weight(num_total_images);
        }
        let mut self_sim: HashMap<usize, f64> = HashMap::new();
        for f in &self.files {
            f.accumulate_self_similarities(&mut self_sim);
        }
        self.normalization_constants = self_sim
            .into_iter()
            .map(|(id, s)| {
                (
                    id,
                    if s > 0.0 {
                        (1.0 / s.sqrt()) as f32
                    } else {
                        0.0
                    },
                )
            })
            .collect();
        self.finalized = true;
    }

    /// `Query`: score every indexed image against `descriptors` — COLMAP's
    /// `InvertedIndex::Query` (`inverted_index.h:248-298`). Panics in debug
    /// builds if [`Self::finalize`] hasn't run since the last `add_image`.
    /// Results are sorted by descending score (ties broken by ascending
    /// `image_id`), truncated to `max_num_images` if given (COLMAP
    /// `QueryOptions::max_num_images`, default unlimited).
    pub fn query(
        &self,
        descriptors: &[Vec<f32>],
        max_num_images: Option<usize>,
    ) -> Vec<ImageScore> {
        self.query_with_work(descriptors, max_num_images).0
    }

    /// Same scoring as [`Self::query`], additionally reporting deterministic
    /// work counters so callers can measure retrieval-cost scaling without
    /// depending on wall-clock noise.
    pub fn query_with_work(
        &self,
        descriptors: &[Vec<f32>],
        max_num_images: Option<usize>,
    ) -> (Vec<ImageScore>, QueryWorkStats) {
        let mut stats = QueryWorkStats::default();
        debug_assert!(self.finalized, "query() requires finalize() to have run");
        if descriptors.is_empty() {
            return (Vec::new(), stats);
        }

        // Self-similarity of the query, summed across every (descriptor,
        // assigned-word) pair — including duplicate words across the
        // `query_num_neighbors` assignment list, matching COLMAP's
        // `ComputeSelfSimilarity` iterating the full `word_ids` matrix.
        let mut self_similarity = 0.0f64;
        let mut assignments: Vec<Vec<usize>> = Vec::with_capacity(descriptors.len());
        for d in descriptors {
            let words = if d.len() == self.vocab.dim() {
                self.vocab
                    .nearest_words(d, self.options.query_num_neighbors)
            } else {
                Vec::new()
            };
            stats.leaf_distance_computations += self.vocab.num_words();
            for &w in &words {
                self_similarity += self.files[w].squared_idf_weight as f64;
            }
            assignments.push(words);
        }
        let normalization_weight = if self_similarity > 0.0 {
            (1.0 / self_similarity.sqrt()) as f32
        } else {
            1.0
        };

        let mut scores: HashMap<usize, f32> = HashMap::new();
        for (d, words) in descriptors.iter().zip(assignments.iter()) {
            if words.is_empty() {
                continue;
            }
            let proj = project(&self.proj_matrix, d);
            for &w in words {
                stats.entries_visited += self.files[w].entries.len();
                for (image_id, s) in self.files[w].score_feature(&proj, &self.weight_lut) {
                    *scores.entry(image_id).or_insert(0.0) += s;
                }
            }
        }

        let mut out: Vec<ImageScore> = scores
            .into_iter()
            .map(|(image_id, score)| {
                let norm_const = self
                    .normalization_constants
                    .get(&image_id)
                    .copied()
                    .unwrap_or(0.0);
                ImageScore {
                    image_id,
                    score: score * normalization_weight * norm_const,
                }
            })
            .collect();
        out.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.image_id.cmp(&b.image_id))
        });
        if let Some(n) = max_num_images {
            out.truncate(n);
        }
        (out, stats)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word_desc(dim: usize, i: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; dim];
        w[i] = 1.0;
        w
    }

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
        }
    }

    fn cluster(center: &[f32], n: usize, jitter: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                center
                    .iter()
                    .map(|&c| c + (rng.next() - 0.5) * 2.0 * jitter)
                    .collect()
            })
            .collect()
    }

    fn build_test_tree(training: &[&[f32]], embedding_dim: usize) -> VocabTree {
        let hkm = HkmBuildOptions {
            branching_factor: 3,
            depth: 2,
            iterations: 15,
            seed: 11,
        };
        let opts = VocabTreeOptions {
            embedding_dim,
            min_he_entries: 3,
            ..VocabTreeOptions::default()
        };
        VocabTree::build(training, &hkm, &opts).expect("tree should build on well-formed input")
    }

    /// An image retrieves itself first: three visually distinct "places",
    /// each queried with its own descriptors, must rank its own image_id at
    /// the top of the returned scores.
    #[test]
    fn image_retrieves_itself_first() {
        let dim = 12;
        let places: Vec<Vec<Vec<f32>>> = (0..3)
            .map(|p| {
                let mut img = cluster(&word_desc(dim, p), 20, 0.03, 1000 + p as u64);
                img.extend(cluster(&word_desc(dim, p + 3), 20, 0.03, 2000 + p as u64));
                img
            })
            .collect();

        let training: Vec<&[f32]> = places.iter().flatten().map(|v| v.as_slice()).collect();
        let mut tree = build_test_tree(&training, 8);
        for (i, img) in places.iter().enumerate() {
            tree.add_image(i, img);
        }
        tree.finalize();

        for (i, img) in places.iter().enumerate() {
            let scores = tree.query(img, None);
            let top = scores.first().expect("at least one scored image");
            assert_eq!(
                top.image_id, i,
                "image {i} should retrieve itself first; got {scores:?}"
            );
        }
    }

    /// A near-duplicate of an indexed image (same content, extra jitter)
    /// must rank above an unrelated third image.
    #[test]
    fn near_duplicate_ranks_above_unrelated_image() {
        let dim = 12;
        let content_a = {
            let mut img = cluster(&word_desc(dim, 0), 25, 0.02, 10);
            img.extend(cluster(&word_desc(dim, 1), 25, 0.02, 20));
            img
        };
        let content_a_dup = {
            let mut img = cluster(&word_desc(dim, 0), 25, 0.02, 30);
            img.extend(cluster(&word_desc(dim, 1), 25, 0.02, 40));
            img
        };
        let content_b = {
            let mut img = cluster(&word_desc(dim, 4), 25, 0.02, 50);
            img.extend(cluster(&word_desc(dim, 5), 25, 0.02, 60));
            img
        };

        let all: Vec<Vec<Vec<f32>>> = vec![content_a.clone(), content_a_dup, content_b];
        let training: Vec<&[f32]> = all.iter().flatten().map(|v| v.as_slice()).collect();
        let mut tree = build_test_tree(&training, 8);
        for (i, img) in all.iter().enumerate() {
            tree.add_image(i, img);
        }
        tree.finalize();

        let scores = tree.query(&content_a, None);
        let score_of = |id: usize| {
            scores
                .iter()
                .find(|s| s.image_id == id)
                .map(|s| s.score)
                .unwrap_or(0.0)
        };
        let (s_dup, s_b) = (score_of(1), score_of(2));
        assert!(
            s_dup > s_b,
            "near-duplicate (image 1, score {s_dup}) should outrank the unrelated image (image 2, score {s_b})"
        );
    }

    #[test]
    fn hamming_weight_lut_matches_colmap_formula() {
        let lut = HammingWeightLut::new(64, 16.0);
        assert_eq!(lut.max_distance, 24); // floor(1.5 * 16)
        assert!((lut.weight(0) - 1.0).abs() < 1e-6);
        let expected_8 = (-64.0f32 / 256.0).exp(); // exp(-h^2/sigma^2), h=8, sigma=16
        assert!((lut.weight(8) - expected_8).abs() < 1e-6);
        assert_eq!(lut.weight(25), 0.0); // beyond max_distance
    }

    #[test]
    fn projection_rows_are_orthonormal() {
        let rows = generate_orthonormal_projection(20, 6, 123);
        assert_eq!(rows.len(), 6);
        for r in &rows {
            let norm = r.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "row not unit-norm: {norm}");
        }
        for i in 0..rows.len() {
            for j in (i + 1)..rows.len() {
                let dot: f32 = rows[i].iter().zip(rows[j].iter()).map(|(a, b)| a * b).sum();
                assert!(dot.abs() < 1e-3, "rows {i},{j} not orthogonal (dot={dot})");
            }
        }
    }

    /// M4 acceptance, deterministic form: the corpus-size-dependent part of
    /// a vocab-tree query must grow (near-)linearly with the number of
    /// indexed images, so that replacing the flat-VLAD quadratic pairwise
    /// scan is a genuine sub-linear-vs-quadratic win at retrieval scale.
    /// Work counters (`QueryWorkStats`) make this machine-independent.
    #[test]
    fn query_work_grows_linearly_while_flat_pairwise_is_quadratic() {
        // Synthetic "places" corpus: `places` clusters in descriptor space;
        // every image draws its descriptors from one place. Dimension 256
        // keeps the crate-default Hamming embedding (64) valid.
        let dim = 256;
        let per_image = 8;
        let places = 24usize;
        let mut rng = Lcg(4242);
        let centroids: Vec<Vec<f32>> = (0..places)
            .map(|_| (0..dim).map(|_| rng.next() * 2.0 - 1.0).collect())
            .collect();
        let mut descriptors_for = |image: usize| -> Vec<Vec<f32>> {
            let c = &centroids[image % places];
            (0..per_image)
                .map(|_| {
                    c.iter()
                        .map(|&v| v + (rng.next() - 0.5) * 0.2)
                        .collect::<Vec<f32>>()
                })
                .collect()
        };

        let mut build_and_query_work = |num_images: usize| -> usize {
            let mut training: Vec<Vec<Vec<f32>>> = Vec::new();
            let mut corpus: Vec<Vec<Vec<f32>>> = Vec::new();
            for image in 0..num_images {
                let d = descriptors_for(image);
                if image % 32 == 0 {
                    training.push(d.clone());
                }
                corpus.push(d);
            }
            let flat_training: Vec<&[f32]> = training
                .iter()
                .flat_map(|d| d.iter().map(|v| v.as_slice()))
                .collect();
            let hkm = HkmBuildOptions {
                branching_factor: 4,
                depth: 2,
                iterations: 3,
                seed: 7,
                ..HkmBuildOptions::default()
            };
            let options = VocabTreeOptions::default();
            let mut tree = VocabTree::build(&flat_training, &hkm, &options).expect("tree builds");
            for (image, d) in corpus.iter().enumerate() {
                tree.add_image(image, d);
            }
            tree.finalize();
            let queries: Vec<Vec<f32>> = corpus[num_images / 2].clone();
            let (_, stats) = tree.query_with_work(&queries, None);
            stats.entries_visited
        };

        let work_small = build_and_query_work(256);
        let work_large = build_and_query_work(2048);
        let ratio = work_large as f64 / work_small as f64;
        // 8× more images: linear scaling predicts ~8; allow slack for IDF /
        // word-distribution drift but far below the quadratic 64×.
        assert!(
            ratio < 10.0,
            "entries_visited grew {ratio:.2}x for an 8x corpus — not near-linear"
        );
        // The flat-VLAD mutual-NN scan this index replaces is exactly
        // quadratic in corpus size (every query against every global).
        let flat_ratio = ((2048f64) * 2047.0) / ((256f64) * 255.0);
        assert!(flat_ratio > 60.0, "sanity: flat scan is ~64x for 8x corpus");
    }
}
