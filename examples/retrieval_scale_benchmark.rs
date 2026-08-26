//! Retrieval-scale benchmark: vocab-tree inverted index vs flat VLAD
//! nearest-neighbour pair generation on a synthetic corpus.
//!
//! `docs/colmap_port_plan.md`'s M4 milestone predicted that the vocab-tree's
//! real payoff is sub-linear retrieval at thousands-of-images scale — flat
//! global-descriptor pair generation compares every query against every db
//! image (quadratic), while an inverted index scores only the entries of each
//! query descriptor's assigned words (near-linear in corpus size at fixed
//! vocabulary width). This example measures both arms on a deterministic
//! synthetic "places" corpus, so the scaling claim is reproducible without
//! any external dataset.
//!
//! For every corpus size the harness reports wall-clock seconds, the exact
//! comparison counts (analytic for the flat scan, measured work counters for
//! the tree), and recall@K of same-place proposals, so quality must hold
//! while cost scales.
//!
//! Run (release build recommended):
//!
//! ```bash
//! cargo run --release --example retrieval_scale_benchmark -- \
//!     --sizes 500,1000,2000,4000
//! ```
//!

use std::time::Instant;

use visloc_rs::vision::place_recognition::{cosine_similarity, vlad, Vocabulary};
use visloc_rs::vision::vocab_tree::{HkmBuildOptions, QueryWorkStats, VocabTree, VocabTreeOptions};

/// Deterministic LCG so every run of the benchmark sees the same corpus.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) as f32
    }

    /// Irwin–Hall approximation of a standard normal (sum of 4 uniforms).
    fn normal(&mut self) -> f32 {
        let s: f32 = (0..4).map(|_| self.next_f32()).sum();
        (s - 2.0) * 1.732
    }
}

struct Corpus {
    /// Per-image local descriptors (256-dim), `per_image` per image.
    descriptors: Vec<Vec<Vec<f32>>>,
    /// Ground-truth place label per image.
    labels: Vec<usize>,
}

/// `num_places` clusters in descriptor space; every image belongs to one
/// place and samples noisy descriptors around its centroid.
fn build_corpus(
    num_images: usize,
    num_places: usize,
    per_image: usize,
    dim: usize,
    seed: u64,
) -> Corpus {
    let mut rng = Lcg(seed);
    let centroids: Vec<Vec<f32>> = (0..num_places)
        .map(|_| (0..dim).map(|_| rng.normal() * 2.0).collect())
        .collect();
    let mut descriptors = Vec::with_capacity(num_images);
    let mut labels = Vec::with_capacity(num_images);
    for image in 0..num_images {
        let place = image % num_places;
        let centroid = &centroids[place];
        descriptors.push(
            (0..per_image)
                .map(|_| {
                    centroid
                        .iter()
                        .map(|&v| v + rng.normal() * 0.08)
                        .collect::<Vec<f32>>()
                })
                .collect(),
        );
        labels.push(place);
    }
    Corpus {
        descriptors,
        labels,
    }
}

/// Mean per-query fraction of top-K proposals that are same-place,
/// normalized by how many same-place images actually exist per query (the
/// synthetic analogue of "retrieved pairs are geometrically valid").
fn recall_at_topk(pairs: &[(usize, usize)], labels: &[usize], k: usize) -> f64 {
    let mut by_query: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for &(q, d) in pairs {
        by_query.entry(q).or_default().push(d);
    }
    let mut per_query_recall: Vec<f64> = Vec::new();
    let same_place_counts: std::collections::HashMap<usize, usize> = labels
        .iter()
        .enumerate()
        .map(|(i, l)| {
            (
                i,
                labels
                    .iter()
                    .filter(|o| **o == *l)
                    .count()
                    .saturating_sub(1),
            )
        })
        .collect();
    for (&query, dbs) in &by_query {
        let relevant = same_place_counts.get(&query).copied().unwrap_or(0);
        if relevant == 0 {
            continue;
        }
        let take = k.min(relevant);
        let hits = dbs
            .iter()
            .take(take)
            .filter(|&&d| labels[d] == labels[query])
            .count();
        per_query_recall.push(hits as f64 / take as f64);
    }
    if per_query_recall.is_empty() {
        return f64::NAN;
    }
    per_query_recall.iter().sum::<f64>() / per_query_recall.len() as f64
}

fn tree_hkm(depth: usize) -> HkmBuildOptions {
    // branching^depth leaf words: small enough to build fast on a laptop,
    // deep enough that word separation keeps same-place images ahead of
    // cross-place leakage (the regime the benchmark studies).
    HkmBuildOptions {
        branching_factor: 16,
        depth,
        iterations: 8,
        seed: 7,
    }
}

fn main() -> Result<(), String> {
    let mut args = std::env::args().skip(1).peekable();
    let mut sizes: Vec<usize> = vec![250, 500, 1000, 2000];
    let mut per_image = 64usize;
    let mut num_places = 200usize;
    let mut train_per_image = 8usize;
    let mut vocab_words = 256usize;
    let mut topk = 10usize;
    let mut seed = 7u64;
    let mut tree_depth = 3usize;
    while let Some(flag) = args.next() {
        let mut value = |name: &str| -> Result<String, String> {
            args.next()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match flag.as_str() {
            "--sizes" => {
                sizes = value("--sizes")?
                    .split(',')
                    .map(|s| s.parse())
                    .collect::<Result<_, _>>()
                    .map_err(|e| format!("--sizes: {e}"))?;
            }
            "--per-image" => {
                per_image = value("--per-image")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--places" => {
                num_places = value("--places")?.parse().map_err(|e| format!("{e}"))?;
            }
            "--train-per-image" => {
                train_per_image = value("--train-per-image")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--vocab-words" => {
                vocab_words = value("--vocab-words")?
                    .parse()
                    .map_err(|e| format!("{e}"))?;
            }
            "--topk" => topk = value("--topk")?.parse().map_err(|e| format!("{e}"))?,
            "--seed" => seed = value("--seed")?.parse().map_err(|e| format!("{e}"))?,
            "--tree-depth" => {
                tree_depth = value("--tree-depth")?.parse().map_err(|e| format!("{e}"))?;
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }

    println!(
        "| images | flat scan (s) | flat comparisons | flat recall@{topk} | tree query (s) | tree entries visited | tree recall@{topk} |"
    );
    println!("|---|---|---|---|---|---|---|");

    let dim = 256;
    // Fixed training pool, independent of every sweep size: both vocabularies
    // (k-means for VLAD and the HKM tree) train once on the same fixed
    // descriptors, so per-size differences measure corpus-size scaling, not
    // vocabulary drift. `VocabTree::build` is deterministic in its seed and
    // inputs, so rebuilding it per size reproduces the identical tree.
    let training_corpus = build_corpus(512, num_places, per_image, dim, seed);
    let training: Vec<&[f32]> = training_corpus
        .descriptors
        .iter()
        .flat_map(|d| d.iter().take(train_per_image).map(|v| v.as_slice()))
        .collect();
    let vocab = Vocabulary::build(&training, vocab_words, 8, seed).expect("vocabulary builds");
    let options = VocabTreeOptions::default();

    for &n in &sizes {
        let corpus = build_corpus(n, num_places, per_image, dim, seed);

        // ---- flat VLAD arm -------------------------------------------------
        let flat_started = Instant::now();
        let globals: Vec<Vec<f32>> = corpus.descriptors.iter().map(|d| vlad(d, &vocab)).collect();
        let mut flat_pairs: Vec<(usize, usize)> = Vec::new();
        let mut flat_comparisons = 0u64;
        for (qi, q) in globals.iter().enumerate() {
            let mut sims: Vec<(usize, f32)> = globals
                .iter()
                .enumerate()
                .map(|(i, g)| {
                    flat_comparisons += 1;
                    (i, cosine_similarity(q, g))
                })
                .collect();
            sims.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            flat_pairs.extend(sims.into_iter().take(topk).map(|(di, _)| (qi, di)));
        }
        let flat_seconds = flat_started.elapsed().as_secs_f64();
        let flat_recall = recall_at_topk(&flat_pairs, &corpus.labels, topk);

        // ---- vocab-tree arm -------------------------------------------------
        let tree_started = Instant::now();
        let mut tree = VocabTree::build(&training, &tree_hkm(tree_depth), &options)
            .expect("vocab tree builds");
        for (image, d) in corpus.descriptors.iter().enumerate() {
            tree.add_image(image, d);
        }
        tree.finalize();
        let mut tree_pairs: Vec<(usize, usize)> = Vec::new();
        let mut entries_total = 0u64;
        let mut leaf_scans_total = 0u64;
        for (image, d) in corpus.descriptors.iter().enumerate() {
            let (scores, stats): (_, QueryWorkStats) = tree.query_with_work(d, Some(topk));
            entries_total += stats.entries_visited as u64;
            leaf_scans_total += stats.leaf_distance_computations as u64;
            for s in scores {
                if s.image_id != image {
                    tree_pairs.push((image, s.image_id));
                }
            }
        }
        let tree_seconds = tree_started.elapsed().as_secs_f64();
        let tree_recall = recall_at_topk(&tree_pairs, &corpus.labels, topk);

        println!(
            "| {n} | {flat_seconds:.3} | {flat_comparisons} | {flat_recall:.3} | {tree_seconds:.3} | {entries_total} (+{leaf_scans_total} leaf scans) | {tree_recall:.3} |"
        );
    }
    Ok(())
}
