//! Deterministic candidate-pair generation for ordered image sequences.
//!
//! Matching and geometric verification intentionally stay outside this module:
//! this layer only combines cheap sequence-order candidates with optional
//! higher-value skip, appearance, and transitive candidates. Keeping the
//! sources attached to a deduplicated pair lets callers choose a more expensive
//! matcher for ambiguous/high-value edges without rebuilding pair policy in a
//! demo binary.

use std::collections::{BTreeMap, BTreeSet};

/// Why an ordered view-graph pair was proposed.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum OrderedPairSource {
    /// A neighbour inside the per-image short temporal window.
    Temporal,
    /// A configured fixed offset intended to preserve a stronger baseline.
    Skip,
    /// A non-local pair supplied by an appearance-retrieval stage.
    Appearance,
    /// A pair inferred from the persistent correspondence graph.
    Transitive,
}

/// One unique, normalized image pair and all policies that proposed it.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrderedPairCandidate {
    pub image_i: usize,
    pub image_j: usize,
    pub sources: Vec<OrderedPairSource>,
}

/// Pair-generation policy shared by sequential SfM frontends.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct OrderedPairGeneratorConfig {
    /// Lower bound applied to an adaptive temporal-window hint.
    pub min_temporal_window: usize,
    /// Upper bound applied to an adaptive temporal-window hint.
    pub max_temporal_window: usize,
    /// Additional offsets, normally wider than `max_temporal_window`.
    pub skip_offsets: Vec<usize>,
    /// Propose skip offsets only from every Nth source image. `1` preserves
    /// the dense policy; larger values provide a deterministic edge budget
    /// without weakening the always-on short temporal backbone.
    pub skip_source_stride: usize,
}

impl Default for OrderedPairGeneratorConfig {
    fn default() -> Self {
        Self {
            min_temporal_window: 5,
            max_temporal_window: 5,
            skip_offsets: Vec::new(),
            skip_source_stride: 1,
        }
    }
}

/// Optional dynamic inputs produced by tracking, retrieval, or track mining.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct OrderedPairHints {
    /// Desired successor count for each source image. Missing entries use the
    /// configured maximum; present entries are clamped to the configured range.
    pub temporal_window_by_image: Vec<usize>,
    pub appearance_pairs: Vec<(usize, usize)>,
    pub transitive_pairs: Vec<(usize, usize)>,
}

fn insert_pair(
    pairs: &mut BTreeMap<(usize, usize), BTreeSet<OrderedPairSource>>,
    n_images: usize,
    a: usize,
    b: usize,
    source: OrderedPairSource,
) {
    if a == b || a >= n_images || b >= n_images {
        return;
    }
    let key = (a.min(b), a.max(b));
    pairs.entry(key).or_default().insert(source);
}

/// Generate a sorted, duplicate-free ordered view graph.
pub fn generate_ordered_pairs(
    n_images: usize,
    config: &OrderedPairGeneratorConfig,
    hints: &OrderedPairHints,
) -> Vec<OrderedPairCandidate> {
    if n_images < 2 {
        return Vec::new();
    }

    let min_window = config.min_temporal_window.min(config.max_temporal_window);
    let max_window = config.max_temporal_window.max(config.min_temporal_window);
    let mut pairs = BTreeMap::<(usize, usize), BTreeSet<OrderedPairSource>>::new();

    for image_i in 0..n_images {
        let requested = hints
            .temporal_window_by_image
            .get(image_i)
            .copied()
            .unwrap_or(max_window);
        let window = requested.clamp(min_window, max_window);
        for image_j in (image_i + 1)..=(image_i + window).min(n_images - 1) {
            insert_pair(
                &mut pairs,
                n_images,
                image_i,
                image_j,
                OrderedPairSource::Temporal,
            );
        }
        if image_i % config.skip_source_stride.max(1) == 0 {
            for &offset in &config.skip_offsets {
                if offset > 0 {
                    if let Some(image_j) = image_i.checked_add(offset) {
                        insert_pair(
                            &mut pairs,
                            n_images,
                            image_i,
                            image_j,
                            OrderedPairSource::Skip,
                        );
                    }
                }
            }
        }
    }

    for &(a, b) in &hints.appearance_pairs {
        insert_pair(&mut pairs, n_images, a, b, OrderedPairSource::Appearance);
    }
    for &(a, b) in &hints.transitive_pairs {
        insert_pair(&mut pairs, n_images, a, b, OrderedPairSource::Transitive);
    }

    pairs
        .into_iter()
        .map(|((image_i, image_j), sources)| OrderedPairCandidate {
            image_i,
            image_j,
            sources: sources.into_iter().collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_window_matches_the_legacy_sequential_policy() {
        let config = OrderedPairGeneratorConfig {
            min_temporal_window: 2,
            max_temporal_window: 2,
            skip_offsets: Vec::new(),
            skip_source_stride: 1,
        };
        let pairs = generate_ordered_pairs(5, &config, &OrderedPairHints::default());
        let indices: Vec<_> = pairs.iter().map(|p| (p.image_i, p.image_j)).collect();
        assert_eq!(
            indices,
            vec![(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        );
        assert!(pairs
            .iter()
            .all(|p| p.sources == [OrderedPairSource::Temporal]));
    }

    #[test]
    fn adaptive_windows_are_clamped_per_source_image() {
        let config = OrderedPairGeneratorConfig {
            min_temporal_window: 1,
            max_temporal_window: 3,
            skip_offsets: Vec::new(),
            skip_source_stride: 1,
        };
        let hints = OrderedPairHints {
            temporal_window_by_image: vec![0, 2, 99],
            ..OrderedPairHints::default()
        };
        let pairs = generate_ordered_pairs(5, &config, &hints);
        let indices: Vec<_> = pairs.iter().map(|p| (p.image_i, p.image_j)).collect();
        assert_eq!(
            indices,
            vec![(0, 1), (1, 2), (1, 3), (2, 3), (2, 4), (3, 4)]
        );
    }

    #[test]
    fn sources_merge_on_one_normalized_pair() {
        let config = OrderedPairGeneratorConfig {
            min_temporal_window: 1,
            max_temporal_window: 1,
            skip_offsets: vec![2, 0, 2],
            skip_source_stride: 1,
        };
        let hints = OrderedPairHints {
            appearance_pairs: vec![(2, 0), (9, 0), (1, 1)],
            transitive_pairs: vec![(0, 2)],
            ..OrderedPairHints::default()
        };
        let pairs = generate_ordered_pairs(4, &config, &hints);
        let pair = pairs
            .iter()
            .find(|p| (p.image_i, p.image_j) == (0, 2))
            .unwrap();
        assert_eq!(
            pair.sources,
            vec![
                OrderedPairSource::Skip,
                OrderedPairSource::Appearance,
                OrderedPairSource::Transitive,
            ]
        );
        assert_eq!(
            pairs
                .iter()
                .filter(|p| (p.image_i, p.image_j) == (0, 2))
                .count(),
            1
        );
    }

    #[test]
    fn skip_stride_budgets_only_skip_sources() {
        let config = OrderedPairGeneratorConfig {
            min_temporal_window: 1,
            max_temporal_window: 1,
            skip_offsets: vec![3],
            skip_source_stride: 2,
        };
        let pairs = generate_ordered_pairs(8, &config, &OrderedPairHints::default());
        let skip_pairs: Vec<_> = pairs
            .iter()
            .filter(|p| p.sources.contains(&OrderedPairSource::Skip))
            .map(|p| (p.image_i, p.image_j))
            .collect();
        assert_eq!(skip_pairs, vec![(0, 3), (2, 5), (4, 7)]);
        let temporal_count = pairs
            .iter()
            .filter(|p| p.sources.contains(&OrderedPairSource::Temporal))
            .count();
        assert_eq!(temporal_count, 7);
    }

    #[test]
    fn output_is_lexically_sorted_and_handles_tiny_inputs() {
        assert!(generate_ordered_pairs(
            1,
            &OrderedPairGeneratorConfig::default(),
            &OrderedPairHints::default()
        )
        .is_empty());
        let hints = OrderedPairHints {
            appearance_pairs: vec![(4, 0), (3, 1)],
            ..OrderedPairHints::default()
        };
        let pairs = generate_ordered_pairs(5, &OrderedPairGeneratorConfig::default(), &hints);
        assert!(pairs
            .windows(2)
            .all(|w| { (w[0].image_i, w[0].image_j) < (w[1].image_i, w[1].image_j) }));
    }
}
