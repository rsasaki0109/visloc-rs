//! Appearance-based place recognition: aggregate the per-keypoint *local*
//! descriptors of an image into a single fixed-length *global* descriptor (VLAD),
//! and retrieve the most similar images across two sets.
//!
//! This is the front-end that PROPOSES loop / cross-session-bridge candidates by
//! appearance, before any geometry is computed. The rest of the stack already
//! exists: each proposed image pair is geometrically verified (two-view essential
//! or PnP → a relative pose), the resulting bridge candidates are screened for
//! mutual consistency (`visloc-slam`'s PCM `consistent_session_bridges`), and the
//! surviving bridge welds two sessions (`PoseGraph::merge_session`). The missing
//! piece was the retrieval itself — there is no global image descriptor or
//! vocabulary anywhere else in the crate.
//!
//! **VLAD** (Vector of Locally Aggregated Descriptors, Jégou et al. 2010): given a
//! visual vocabulary of `k` centroids over the local-descriptor space, each local
//! descriptor is assigned to its nearest centroid and its *residual* `(x − cₖ)` is
//! accumulated into that centroid's block. The `k × dim` concatenation is then
//! intra-normalized (per block) and globally L2-normalized — the robust variant of
//! Arandjelović & Zisserman, "All about VLAD" (2013) — so cosine similarity of two
//! VLADs is a meaningful appearance similarity.
//!
//! Everything here is deterministic (a fixed-seed LCG for k-means++), pure
//! `f32`/`Vec`, no image or external-model dependency, so it is unit-testable on
//! controlled descriptors.

/// A visual vocabulary: `k` cluster centroids over the local-descriptor space,
/// used to assign each local descriptor to its nearest visual word for VLAD.
#[derive(Debug, Clone, PartialEq)]
pub struct Vocabulary {
    pub centroids: Vec<Vec<f32>>,
}

/// Small deterministic linear-congruential RNG, so vocabulary construction is
/// reproducible across runs and platforms (no `rand` dependency).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Squared Euclidean distance between two equal-length descriptors.
fn sq_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

impl Vocabulary {
    /// Build a vocabulary by k-means (Lloyd's algorithm, k-means++ seeding) over a
    /// pooled set of local descriptors. Deterministic given `seed`. Returns `None`
    /// if there are fewer than `k` descriptors, `k == 0`, or the descriptors are
    /// not all of one positive dimension.
    pub fn build(
        descriptors: &[&[f32]],
        k: usize,
        iterations: usize,
        seed: u64,
    ) -> Option<Vocabulary> {
        if k == 0 || descriptors.len() < k {
            return None;
        }
        let dim = descriptors[0].len();
        if dim == 0 || descriptors.iter().any(|d| d.len() != dim) {
            return None;
        }

        // k-means++ seeding: first centroid uniformly at random, each subsequent
        // one sampled with probability proportional to its squared distance to the
        // nearest chosen centroid.
        let mut rng = Lcg(seed.wrapping_mul(2) | 1);
        let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
        let first = (rng.next_u64() as usize) % descriptors.len();
        centroids.push(descriptors[first].to_vec());
        while centroids.len() < k {
            let d2: Vec<f32> = descriptors
                .iter()
                .map(|d| {
                    centroids
                        .iter()
                        .map(|c| sq_distance(d, c))
                        .fold(f32::INFINITY, f32::min)
                })
                .collect();
            let total: f64 = d2.iter().map(|&v| v as f64).sum();
            if total <= 0.0 {
                // All remaining descriptors coincide with a centroid; pad with a
                // distinct descriptor if one exists, else stop.
                if let Some(d) = descriptors
                    .iter()
                    .find(|d| !centroids.contains(&d.to_vec()))
                {
                    centroids.push(d.to_vec());
                    continue;
                }
                break;
            }
            let mut target = rng.unit() * total;
            let mut chosen = descriptors.len() - 1;
            for (i, &w) in d2.iter().enumerate() {
                target -= w as f64;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            centroids.push(descriptors[chosen].to_vec());
        }
        let k = centroids.len();

        // Lloyd iterations.
        for _ in 0..iterations {
            let mut sums = vec![vec![0.0f32; dim]; k];
            let mut counts = vec![0usize; k];
            for d in descriptors {
                let a = nearest_centroid(&centroids, d);
                for (s, &x) in sums[a].iter_mut().zip(d.iter()) {
                    *s += x;
                }
                counts[a] += 1;
            }
            for (c, (sum, &n)) in centroids.iter_mut().zip(sums.iter().zip(counts.iter())) {
                if n > 0 {
                    for (ci, &si) in c.iter_mut().zip(sum.iter()) {
                        *ci = si / n as f32;
                    }
                }
                // Empty clusters keep their previous centroid (stable, no-op).
            }
        }
        Some(Vocabulary { centroids })
    }

    /// Number of visual words.
    pub fn k(&self) -> usize {
        self.centroids.len()
    }

    /// Local-descriptor dimension (0 if empty).
    pub fn dim(&self) -> usize {
        self.centroids.first().map_or(0, |c| c.len())
    }
}

/// Index of the nearest centroid to `d` (by squared Euclidean distance).
fn nearest_centroid(centroids: &[Vec<f32>], d: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let dist = sq_distance(c, d);
        if dist < best_d {
            best_d = dist;
            best = i;
        }
    }
    best
}

/// L2-normalize a slice in place (no-op if the norm is ~0).
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-12 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute the intra- and globally-normalized VLAD global descriptor for an
/// image's local descriptors against `vocab`. Length is `vocab.k() * vocab.dim()`.
/// Returns an all-zero vector of that length if there are no local descriptors
/// (so every image yields a comparable fixed-length descriptor).
pub fn vlad(local_descriptors: &[Vec<f32>], vocab: &Vocabulary) -> Vec<f32> {
    let (k, dim) = (vocab.k(), vocab.dim());
    let mut blocks = vec![vec![0.0f32; dim]; k];
    for d in local_descriptors {
        if d.len() != dim {
            continue;
        }
        let a = nearest_centroid(&vocab.centroids, d);
        for (b, (&x, &c)) in blocks[a]
            .iter_mut()
            .zip(d.iter().zip(vocab.centroids[a].iter()))
        {
            *b += x - c;
        }
    }
    // Intra-normalization: L2-normalize each per-centroid block independently, so
    // no single visual word can dominate the descriptor (the "All about VLAD"
    // robust variant).
    for block in blocks.iter_mut() {
        l2_normalize(block);
    }
    let mut out: Vec<f32> = blocks.into_iter().flatten().collect();
    l2_normalize(&mut out); // global L2 normalization
    out
}

/// Cosine similarity of two descriptors. For L2-normalized inputs (as [`vlad`]
/// returns) this is just the dot product. Returns 0 if the lengths differ or
/// either is empty.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(&x, &y)| x * y).sum();
    let na = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na <= 1e-12 || nb <= 1e-12 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// A retrieved cross-set candidate pair: the `query`-set index, the `db`-set
/// index, and their appearance `similarity`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetrievedPair {
    pub query: usize,
    pub db: usize,
    pub similarity: f32,
}

/// Propose cross-set place-recognition matches: for each query global descriptor,
/// its most similar db descriptor; keep only the **mutual** nearest neighbours
/// (the query is also the db descriptor's best query) whose cosine similarity is
/// at least `min_similarity`. Mutual-NN is the standard robust retrieval filter —
/// it discards the many one-sided spurious matches that a fixed top-1 would keep.
///
/// Results are sorted by descending similarity (ties broken by `query` then `db`),
/// so a caller can take the strongest proposals first.
pub fn retrieve_mutual(
    query_globals: &[Vec<f32>],
    db_globals: &[Vec<f32>],
    min_similarity: f32,
) -> Vec<RetrievedPair> {
    if query_globals.is_empty() || db_globals.is_empty() {
        return Vec::new();
    }
    let best_in = |from: &[Vec<f32>], v: &[f32]| -> Option<usize> {
        let mut best = None;
        let mut best_s = f32::NEG_INFINITY;
        for (i, u) in from.iter().enumerate() {
            let s = cosine_similarity(v, u);
            if s > best_s {
                best_s = s;
                best = Some(i);
            }
        }
        best
    };
    // db-side best query, precomputed once.
    let db_best: Vec<Option<usize>> = db_globals
        .iter()
        .map(|d| best_in(query_globals, d))
        .collect();

    let mut pairs: Vec<RetrievedPair> = Vec::new();
    for (qi, q) in query_globals.iter().enumerate() {
        if let Some(di) = best_in(db_globals, q) {
            if db_best[di] == Some(qi) {
                let s = cosine_similarity(q, &db_globals[di]);
                if s >= min_similarity {
                    pairs.push(RetrievedPair {
                        query: qi,
                        db: di,
                        similarity: s,
                    });
                }
            }
        }
    }
    pairs.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.query.cmp(&b.query))
            .then(a.db.cmp(&b.db))
    });
    pairs
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A local descriptor cluster around `center` with `n` jittered members. The
    /// jitter is a deterministic function of the seed so tests are reproducible.
    fn cluster(center: &[f32], n: usize, jitter: f32, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = Lcg(seed);
        (0..n)
            .map(|_| {
                center
                    .iter()
                    .map(|&c| c + ((rng.unit() as f32) - 0.5) * 2.0 * jitter)
                    .collect()
            })
            .collect()
    }

    /// Standard-basis visual words, with a vocabulary whose centroids sit at a
    /// fraction of each word — so a descriptor near word `i` lands on centroid `i`
    /// with a large, structured residual `(1−f)·eᵢ` (not the near-zero noise that
    /// would arise if the centroids coincided with the descriptors). `k = dim`.
    fn basis_vocab(dim: usize, fraction: f32) -> Vocabulary {
        let centroids = (0..dim)
            .map(|i| {
                let mut c = vec![0.0f32; dim];
                c[i] = fraction;
                c
            })
            .collect();
        Vocabulary { centroids }
    }

    fn word(dim: usize, i: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; dim];
        w[i] = 1.0;
        w
    }

    /// Two "images" made of the same visual-word mix have similar VLADs; an image
    /// made of different words is dissimilar.
    #[test]
    fn vlad_similarity_reflects_shared_visual_words() {
        let dim = 4;
        let vocab = basis_vocab(dim, 0.3);

        // Image A and A' both mix words 0 and 1; image B mixes words 2 and 3.
        let mut a = cluster(&word(dim, 0), 15, 0.05, 1);
        a.extend(cluster(&word(dim, 1), 15, 0.05, 2));
        let mut a2 = cluster(&word(dim, 0), 15, 0.05, 3);
        a2.extend(cluster(&word(dim, 1), 15, 0.05, 4));
        let mut b = cluster(&word(dim, 2), 15, 0.05, 5);
        b.extend(cluster(&word(dim, 3), 15, 0.05, 6));

        let va = vlad(&a, &vocab);
        let va2 = vlad(&a2, &vocab);
        let vb = vlad(&b, &vocab);
        assert_eq!(va.len(), 16);

        let same = cosine_similarity(&va, &va2);
        let diff = cosine_similarity(&va, &vb);
        assert!(
            same > 0.9,
            "same-content images should have high VLAD similarity, got {same}"
        );
        assert!(
            same > diff + 0.5,
            "same-content ({same}) must clearly exceed different-content ({diff})"
        );
    }

    /// k-means recovers well-separated clusters: every true cluster centre is
    /// close to some learned centroid.
    #[test]
    fn kmeans_recovers_separated_clusters() {
        let dim = 4;
        let centers: Vec<Vec<f32>> = (0..4).map(|i| word(dim, i)).collect();
        let mut pool: Vec<Vec<f32>> = Vec::new();
        for (i, c) in centers.iter().enumerate() {
            pool.extend(cluster(c, 30, 0.05, 300 + i as u64));
        }
        let refs: Vec<&[f32]> = pool.iter().map(|v| v.as_slice()).collect();
        let vocab = Vocabulary::build(&refs, 4, 25, 7).unwrap();
        assert_eq!(vocab.k(), 4);
        for c in &centers {
            let nearest = vocab
                .centroids
                .iter()
                .map(|v| sq_distance(c, v))
                .fold(f32::INFINITY, f32::min);
            assert!(
                nearest < 0.05,
                "true centre {c:?} should be matched by a learned centroid (d²={nearest})"
            );
        }
    }

    /// Cross-set retrieval recovers the planted correspondences (place p in set A
    /// is the same place as p in set B) and rejects unrelated images via the
    /// mutual-nearest-neighbour filter.
    #[test]
    fn retrieve_mutual_recovers_planted_cross_session_matches() {
        let dim = 6;
        let vocab = basis_vocab(dim, 0.3);
        // Each "place" p mixes words p and p+1. Sessions A and B both traverse
        // places 0..4 (so A[p] and B[p] are the same place, different jitter).
        let make_place = |p: usize, seed: u64| {
            let mut img = cluster(&word(dim, p), 20, 0.05, seed);
            img.extend(cluster(&word(dim, p + 1), 20, 0.05, seed + 1000));
            img
        };
        let a_globals: Vec<Vec<f32>> = (0..5)
            .map(|p| vlad(&make_place(p, p as u64), &vocab))
            .collect();
        let b_globals: Vec<Vec<f32>> = (0..5)
            .map(|p| vlad(&make_place(p, 500 + p as u64), &vocab))
            .collect();

        let pairs = retrieve_mutual(&a_globals, &b_globals, 0.5);
        // Every place should be retrieved to its own counterpart.
        for p in 0..5 {
            assert!(
                pairs.iter().any(|r| r.query == p && r.db == p),
                "place {p} should retrieve its cross-session counterpart; got {pairs:?}"
            );
        }
        // No mismatched pair should survive (mutual-NN + threshold).
        assert!(
            pairs.iter().all(|r| r.query == r.db),
            "only same-place pairs should survive, got {pairs:?}"
        );
    }

    #[test]
    fn build_rejects_degenerate_inputs() {
        let d = [vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let refs: Vec<&[f32]> = d.iter().map(|v| v.as_slice()).collect();
        assert!(Vocabulary::build(&refs, 0, 5, 1).is_none()); // k = 0
        assert!(Vocabulary::build(&refs, 5, 5, 1).is_none()); // k > n
        assert!(Vocabulary::build(&[], 1, 5, 1).is_none()); // empty
    }

    #[test]
    fn empty_image_yields_a_zero_global_of_the_right_length() {
        let d = [vec![1.0f32, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let refs: Vec<&[f32]> = d.iter().map(|v| v.as_slice()).collect();
        let vocab = Vocabulary::build(&refs, 2, 5, 1).unwrap();
        let v = vlad(&[], &vocab);
        assert_eq!(v.len(), vocab.k() * vocab.dim());
        assert!(v.iter().all(|&x| x == 0.0));
    }
}
