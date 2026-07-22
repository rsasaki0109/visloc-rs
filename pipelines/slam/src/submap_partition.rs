//! Deterministic overlapping partitions for hierarchical ordered-image SfM.
//!
//! Boundaries are selected near a target window length, but may move within a
//! bounded search interval to a seam with stronger verified cross-boundary
//! support. An optional per-cut quality hint lets a frontend incorporate motion,
//! blur, or dynamic-region evidence without changing geometric acceptance gates.

use std::error::Error;
use std::fmt;
use std::ops::Range;

use crate::PairwiseMatches;

#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSubmapPartitionConfig {
    /// Smallest permitted window, including its overlap with the previous one.
    pub min_images: usize,
    /// Preferred window length before seam-quality adjustment.
    pub target_images: usize,
    /// Hard upper bound on a window's image count.
    pub max_images: usize,
    /// Images reconstructed independently in both adjacent submaps.
    pub overlap_images: usize,
    /// Maximum displacement of a boundary from `target_images`.
    pub boundary_search_radius: usize,
}

impl Default for AdaptiveSubmapPartitionConfig {
    fn default() -> Self {
        Self {
            min_images: 24,
            target_images: 64,
            max_images: 96,
            overlap_images: 16,
            boundary_search_radius: 16,
        }
    }
}

/// Optional frontend evidence indexed by cut position. Entry `c` describes the
/// seam between images `c - 1` and `c`; larger finite values favor that seam.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AdaptiveSubmapPartitionHints {
    pub boundary_quality_by_cut: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubmapWindow {
    pub image_range: Range<usize>,
    /// Sum of verified correspondences on pair edges crossing the chosen end.
    /// The final window has no outgoing seam and therefore reports zero.
    pub outgoing_seam_support: usize,
}

impl SubmapWindow {
    pub fn image_count(&self) -> usize {
        self.image_range.len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubmapPartitionError {
    ZeroMinimum,
    InvalidSizeOrder,
    OverlapTooLarge,
    PairImageOutOfRange {
        pair_index: usize,
        image_index: usize,
        image_count: usize,
    },
}

impl fmt::Display for SubmapPartitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMinimum => write!(f, "submap min_images must be positive"),
            Self::InvalidSizeOrder => write!(
                f,
                "submap sizes must satisfy min_images <= target_images <= max_images"
            ),
            Self::OverlapTooLarge => {
                write!(f, "submap overlap_images must be smaller than min_images")
            }
            Self::PairImageOutOfRange {
                pair_index,
                image_index,
                image_count,
            } => write!(
                f,
                "pair {pair_index} references image {image_index}; image count is {image_count}"
            ),
        }
    }
}

impl Error for SubmapPartitionError {}

/// Partition an ordered sequence into contiguous overlapping windows.
pub fn partition_ordered_submaps(
    image_count: usize,
    pairwise: &[PairwiseMatches],
    config: &AdaptiveSubmapPartitionConfig,
    hints: &AdaptiveSubmapPartitionHints,
) -> Result<Vec<SubmapWindow>, SubmapPartitionError> {
    validate_config(config)?;
    for (pair_index, pair) in pairwise.iter().enumerate() {
        for image_index in [pair.image_i, pair.image_j] {
            if image_index >= image_count {
                return Err(SubmapPartitionError::PairImageOutOfRange {
                    pair_index,
                    image_index,
                    image_count,
                });
            }
        }
    }
    if image_count == 0 {
        return Ok(Vec::new());
    }
    if image_count <= config.max_images {
        return Ok(vec![SubmapWindow {
            image_range: 0..image_count,
            outgoing_seam_support: 0,
        }]);
    }

    let seam_support = seam_support_by_cut(image_count, pairwise);
    let mut windows = Vec::new();
    let mut start = 0;
    while image_count - start > config.max_images {
        let target = (start + config.target_images).min(image_count);
        let lower =
            (target.saturating_sub(config.boundary_search_radius)).max(start + config.min_images);
        let upper = (target + config.boundary_search_radius)
            .min(start + config.max_images)
            .min(image_count);
        let end = (lower..=upper)
            .filter(|&cut| image_count - (cut - config.overlap_images) >= config.min_images)
            .max_by(|&left, &right| {
                seam_score(left, &seam_support, hints)
                    .total_cmp(&seam_score(right, &seam_support, hints))
                    .then_with(|| right.abs_diff(target).cmp(&left.abs_diff(target)))
                    .then_with(|| right.cmp(&left))
            })
            .unwrap_or_else(|| (start + config.max_images).min(image_count));
        windows.push(SubmapWindow {
            image_range: start..end,
            outgoing_seam_support: seam_support[end],
        });
        let next_start = end - config.overlap_images;
        debug_assert!(next_start > start, "validated overlap guarantees progress");
        start = next_start;
    }
    windows.push(SubmapWindow {
        image_range: start..image_count,
        outgoing_seam_support: 0,
    });
    Ok(windows)
}

/// Select and remap verified pairs into one window's local image indices.
pub fn remap_pairs_to_submap(
    pairwise: &[PairwiseMatches],
    image_range: Range<usize>,
) -> Vec<PairwiseMatches> {
    pairwise
        .iter()
        .filter(|pair| image_range.contains(&pair.image_i) && image_range.contains(&pair.image_j))
        .map(|pair| PairwiseMatches {
            image_i: pair.image_i - image_range.start,
            image_j: pair.image_j - image_range.start,
            matches: pair.matches.clone(),
        })
        .collect()
}

fn validate_config(config: &AdaptiveSubmapPartitionConfig) -> Result<(), SubmapPartitionError> {
    if config.min_images == 0 {
        return Err(SubmapPartitionError::ZeroMinimum);
    }
    if config.min_images > config.target_images || config.target_images > config.max_images {
        return Err(SubmapPartitionError::InvalidSizeOrder);
    }
    if config.overlap_images >= config.min_images {
        return Err(SubmapPartitionError::OverlapTooLarge);
    }
    Ok(())
}

fn seam_support_by_cut(image_count: usize, pairwise: &[PairwiseMatches]) -> Vec<usize> {
    let mut difference = vec![0_i64; image_count + 1];
    for pair in pairwise {
        let left = pair.image_i.min(pair.image_j);
        let right = pair.image_i.max(pair.image_j);
        let support = pair.matches.len() as i64;
        difference[left + 1] += support;
        difference[right + 1] -= support;
    }
    let mut active = 0_i64;
    difference
        .into_iter()
        .map(|delta| {
            active += delta;
            active.max(0) as usize
        })
        .collect()
}

fn seam_score(cut: usize, seam_support: &[usize], hints: &AdaptiveSubmapPartitionHints) -> f64 {
    let quality = hints
        .boundary_quality_by_cut
        .get(cut)
        .copied()
        .filter(|value| value.is_finite() && *value >= 0.0)
        .unwrap_or(1.0);
    seam_support[cut] as f64 * quality
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(i: usize, j: usize, count: usize) -> PairwiseMatches {
        PairwiseMatches {
            image_i: i,
            image_j: j,
            matches: (0..count).map(|k| (k, k)).collect(),
        }
    }

    fn config() -> AdaptiveSubmapPartitionConfig {
        AdaptiveSubmapPartitionConfig {
            min_images: 4,
            target_images: 6,
            max_images: 8,
            overlap_images: 2,
            boundary_search_radius: 2,
        }
    }

    #[test]
    fn partitions_cover_sequence_with_exact_overlap_and_no_short_tail() {
        let windows =
            partition_ordered_submaps(17, &[], &config(), &AdaptiveSubmapPartitionHints::default())
                .unwrap();
        assert_eq!(
            windows
                .iter()
                .map(|window| window.image_range.clone())
                .collect::<Vec<_>>(),
            vec![0..6, 4..10, 8..14, 12..17]
        );
        assert!(windows.iter().all(|window| window.image_count() >= 4));
        for adjacent in windows.windows(2) {
            assert_eq!(
                adjacent[0].image_range.end - adjacent[1].image_range.start,
                2
            );
        }
    }

    #[test]
    fn boundary_moves_to_the_best_supported_seam() {
        let pairs = vec![
            pair(3, 7, 100),
            pair(4, 7, 80),
            pair(6, 7, 100),
            pair(0, 1, 500),
        ];
        let windows = partition_ordered_submaps(
            14,
            &pairs,
            &config(),
            &AdaptiveSubmapPartitionHints::default(),
        )
        .unwrap();
        assert_eq!(windows[0].image_range, 0..7);
        assert_eq!(windows[0].outgoing_seam_support, 280);
    }

    #[test]
    fn motion_quality_hint_can_avoid_an_unsafe_seam() {
        let pairs = vec![pair(2, 6, 50), pair(3, 7, 50)];
        let mut hints = AdaptiveSubmapPartitionHints::default();
        hints.boundary_quality_by_cut = vec![1.0; 15];
        hints.boundary_quality_by_cut[6] = 0.0;
        hints.boundary_quality_by_cut[7] = 3.0;
        let windows = partition_ordered_submaps(14, &pairs, &config(), &hints).unwrap();
        assert_eq!(windows[0].image_range, 0..7);
    }

    #[test]
    fn remaps_only_internal_pairs_to_local_indices() {
        let pairs = vec![pair(1, 3, 2), pair(3, 5, 3), pair(5, 7, 4)];
        let local = remap_pairs_to_submap(&pairs, 3..7);
        assert_eq!(local, vec![pair(0, 2, 3)]);
    }

    #[test]
    fn rejects_invalid_configuration_and_pair_index() {
        let mut invalid = config();
        invalid.overlap_images = invalid.min_images;
        assert_eq!(
            partition_ordered_submaps(10, &[], &invalid, &Default::default()),
            Err(SubmapPartitionError::OverlapTooLarge)
        );
        assert!(matches!(
            partition_ordered_submaps(5, &[pair(0, 5, 1)], &config(), &Default::default()),
            Err(SubmapPartitionError::PairImageOutOfRange { .. })
        ));
    }
}
