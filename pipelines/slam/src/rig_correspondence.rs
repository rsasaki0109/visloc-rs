//! Compact correspondence storage for a calibrated rig's local view graph.
//!
//! [`visloc_vision::two_view::CorrespondenceGraph`] is the general COLMAP
//! compatibility graph used by the existing mapper.  It deliberately keeps a
//! hash map and a vector for every image/keypoint while it is being assembled.
//! A rig replay can have millions of observations, so this module provides a
//! separate read-only preview representation: observation ids are flattened to
//! `u32`, endpoint pairs are canonicalized once, and the final rows are filled
//! with a two-pass CSR construction.  Nothing in the mapper consumes this
//! representation yet; adding it therefore cannot change mapper output.
//!
//! The builder accepts the existing [`PairwiseMatches`] verified-match type.
//! It intentionally does not accept raw descriptor matches.  A pair is an
//! undirected relation even when its stored image order is reversed, and each
//! retained relation is represented by one neighbour in each direction.

use std::mem::size_of;

use thiserror::Error;

use crate::incremental_sfm::PairwiseMatches;

/// Flattened id of an image/keypoint observation in a rig-local graph.
pub type RigObservationId = u32;

const MAX_FLATTENED_OBSERVATIONS: u64 = u32::MAX as u64 + 1;
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x00000100000001b3;

/// Errors found before a rig-local CSR can be materialized.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum RigCorrespondenceBuildError {
    /// One image has more observations than can be addressed by a flattened
    /// `u32` id, including the id zero.
    #[error(
        "image {image} has {count} observations; flattened observation ids support at most {max}"
    )]
    FeatureCountOverflow {
        image: usize,
        count: usize,
        max: u64,
    },
    /// The cumulative observation count cannot be represented by a flattened
    /// `u32` id.  The bound is checked before any graph allocation.
    #[error(
        "flattened observation count overflows u32 ids at image {image}: cumulative count {total} exceeds {max}"
    )]
    ObservationCountOverflow { image: usize, total: u64, max: u64 },
    /// A valid u32-id graph is too large for the current target's `usize`
    /// indexing model.  This is a representation check, not a policy cap.
    #[error("flattened observation count {total} does not fit usize")]
    ObservationCountNotRepresentable { total: u64 },
    /// The image-offset sentinel would overflow the host vector length.
    #[error("image count overflows the CSR image-offset vector")]
    ImageCountOverflow,
    /// A pair references an image that is absent from the supplied feature
    /// count vector.
    #[error(
        "pair {pair_index} references image {image}, outside feature image range 0..{image_count}"
    )]
    ImageOutOfRange {
        pair_index: usize,
        image: usize,
        image_count: usize,
    },
    /// A pair joins two observations from the same image.  Such a relation is
    /// invalid for this image-to-image correspondence graph.
    #[error("pair {pair_index} is a self-correspondence for image {image}")]
    SelfCorrespondence { pair_index: usize, image: usize },
    /// A verified match references a keypoint outside the corresponding image
    /// feature set.
    #[error(
        "pair {pair_index} match {match_index} references image {image} keypoint {keypoint}, outside 0..{observation_count}"
    )]
    ObservationOutOfRange {
        pair_index: usize,
        match_index: usize,
        image: usize,
        keypoint: usize,
        observation_count: usize,
    },
    /// The input match count or its directed expansion overflowed `usize`.
    #[error("verified match count overflows the CSR builder")]
    MatchCountOverflow,
    /// A CSR offset or edge count cannot be represented by its storage type.
    #[error("CSR edge count cannot be represented by the target")]
    EdgeCountOverflow,
    /// The persistent-byte estimate overflowed `usize` while accounting for
    /// the representation.  It is checked rather than wrapped.
    #[error("estimated persistent CSR byte count overflows usize")]
    PersistentByteCountOverflow,
}

/// Preview measurements for a canonical rig-local correspondence graph.
///
/// `duplicate_drops` counts duplicate undirected input matches, so a repeated
/// match contributes one drop even though the retained CSR stores two
/// directed neighbours.  `estimated_persistent_bytes` accounts for the three
/// compact arrays (`image_offsets`, `row_offsets`, and `neighbors`) and does
/// not claim to model allocator slack or a process-specific `Vec` header.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RigCorrespondencePreviewStats {
    /// Total feature observations across all rig images, including isolated
    /// observations with an empty CSR row.
    pub total_observations: usize,
    /// Number of unique directed neighbour entries in the final CSR.
    pub directed_unique_edges: usize,
    /// Number of unique undirected observation relations.
    pub undirected_unique_edges: usize,
    /// Number of duplicate undirected matches removed from the verified input.
    pub duplicate_drops: usize,
    /// Bytes occupied by the compact CSR arrays, as an estimate of persistent
    /// storage (allocation slack and object headers are excluded).
    pub estimated_persistent_bytes: usize,
    /// Largest number of neighbours in any one observation row.
    pub max_row_degree: usize,
    /// Stable FNV-1a digest of the canonical CSR representation.
    pub digest: u64,
}

impl RigCorrespondencePreviewStats {
    /// Alias with an explicit name for callers that keep multiple digests in a
    /// diagnostic record.
    pub fn deterministic_digest(self) -> u64 {
        self.digest
    }
}

/// Read-only compact CSR for rig-local observation correspondences.
///
/// `image_offsets[image]` is the first flattened observation id for that
/// image, and the final sentinel is the total observation count.  Image
/// offsets are `u64` so the sentinel `2^32` remains representable even though
/// every actual observation id is a `u32`.  `row_offsets` has
/// `total_observations + 1` entries and indexes the `neighbors` array.  The
/// neighbour rows are sorted in ascending flattened observation-id order and
/// contain no duplicates.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigCorrespondenceCsr {
    /// Flattened observation base id for each input image.
    pub image_offsets: Vec<u64>,
    /// CSR row offsets. `u64` keeps the persisted layout independent of the
    /// host pointer width while still permitting checked `usize` indexing.
    pub row_offsets: Vec<u64>,
    /// Canonical, sorted, duplicate-free neighbour ids.
    pub neighbors: Vec<RigObservationId>,
}

impl RigCorrespondenceCsr {
    /// Number of images represented by the graph, including empty images.
    pub fn image_count(&self) -> usize {
        self.image_offsets.len().saturating_sub(1)
    }

    /// Number of observations represented by the graph.
    pub fn total_observations(&self) -> usize {
        self.row_offsets.len().saturating_sub(1)
    }

    /// Number of directed neighbour entries in this CSR.
    pub fn directed_unique_edges(&self) -> usize {
        self.neighbors.len()
    }

    /// Number of undirected relations, which is half the symmetric directed
    /// count.  The builder guarantees symmetry, so this conversion is exact.
    pub fn undirected_unique_edges(&self) -> usize {
        self.neighbors.len() / 2
    }

    /// Return one sorted neighbour row, or `None` for an out-of-range id.
    pub fn neighbors_for(&self, observation: RigObservationId) -> Option<&[RigObservationId]> {
        let row = usize::try_from(observation).ok()?;
        let start = usize::try_from(*self.row_offsets.get(row)?).ok()?;
        let end = usize::try_from(*self.row_offsets.get(row.checked_add(1)?)?).ok()?;
        self.neighbors.get(start..end)
    }

    /// Return a row's degree, or `None` for an out-of-range id.
    pub fn row_degree(&self, observation: RigObservationId) -> Option<usize> {
        self.neighbors_for(observation).map(<[_]>::len)
    }

    /// Convert an image/keypoint pair to its flattened observation id.
    pub fn observation_id(&self, image: usize, keypoint: usize) -> Option<RigObservationId> {
        let base = *self.image_offsets.get(image)?;
        let end = *self.image_offsets.get(image + 1)?;
        let keypoint = u64::try_from(keypoint).ok()?;
        let id = base.checked_add(keypoint)?;
        (keypoint < end.checked_sub(base)?).then_some(id as RigObservationId)
    }

    /// Convert a flattened id back to its image and image-local keypoint.
    ///
    /// Empty-image duplicate offsets are skipped by `partition_point`, so the
    /// returned image always owns the observation. The lookup is logarithmic
    /// in image count and avoids a second observation-sized owner array.
    pub fn observation(&self, observation: RigObservationId) -> Option<(usize, usize)> {
        let observation = u64::from(observation);
        if observation >= *self.image_offsets.last()? {
            return None;
        }
        let image = self
            .image_offsets
            .partition_point(|&offset| offset <= observation)
            .checked_sub(1)?;
        let keypoint = usize::try_from(observation.checked_sub(self.image_offsets[image])?).ok()?;
        Some((image, keypoint))
    }

    /// Estimate the bytes occupied by the compact arrays.
    pub fn estimated_persistent_bytes(&self) -> Option<usize> {
        let image_bytes = self.image_offsets.len().checked_mul(size_of::<u64>())?;
        let row_bytes = self.row_offsets.len().checked_mul(size_of::<u64>())?;
        let neighbor_bytes = self.neighbors.len().checked_mul(size_of::<u32>())?;
        image_bytes
            .checked_add(row_bytes)?
            .checked_add(neighbor_bytes)
    }

    /// Stable digest of the exact canonical CSR arrays.
    pub fn digest(&self) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        hash_bytes(&mut hash, b"visloc-rig-correspondence-csr-v1\0");
        hash_u64(&mut hash, self.image_offsets.len() as u64);
        for &offset in &self.image_offsets {
            hash_u64(&mut hash, offset);
        }
        hash_u64(&mut hash, self.row_offsets.len() as u64);
        for &offset in &self.row_offsets {
            hash_u64(&mut hash, offset);
        }
        hash_u64(&mut hash, self.neighbors.len() as u64);
        for &neighbor in &self.neighbors {
            hash_u32(&mut hash, neighbor);
        }
        hash
    }
}

/// CSR plus the measurements collected while building it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigCorrespondenceBuild {
    pub csr: RigCorrespondenceCsr,
    pub stats: RigCorrespondencePreviewStats,
}

/// Deterministic builder for one rig-local observation graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RigCorrespondenceCsrBuilder {
    feature_counts: Vec<usize>,
    image_offsets: Vec<u64>,
    total_observations: usize,
}

impl RigCorrespondenceCsrBuilder {
    /// Validate feature counts and prepare flattened image offsets.
    pub fn new(feature_counts: &[usize]) -> Result<Self, RigCorrespondenceBuildError> {
        let image_offset_capacity = feature_counts
            .len()
            .checked_add(1)
            .ok_or(RigCorrespondenceBuildError::ImageCountOverflow)?;
        let mut image_offsets = Vec::with_capacity(image_offset_capacity);
        let mut total = 0u64;
        for (image, &count) in feature_counts.iter().enumerate() {
            let count_u64 = u64::try_from(count).map_err(|_| {
                RigCorrespondenceBuildError::FeatureCountOverflow {
                    image,
                    count,
                    max: MAX_FLATTENED_OBSERVATIONS,
                }
            })?;
            if count_u64 > MAX_FLATTENED_OBSERVATIONS {
                return Err(RigCorrespondenceBuildError::FeatureCountOverflow {
                    image,
                    count,
                    max: MAX_FLATTENED_OBSERVATIONS,
                });
            }
            if total > MAX_FLATTENED_OBSERVATIONS - count_u64 {
                return Err(RigCorrespondenceBuildError::ObservationCountOverflow {
                    image,
                    total: total.saturating_add(count_u64),
                    max: MAX_FLATTENED_OBSERVATIONS,
                });
            }
            image_offsets.push(total);
            total += count_u64;
        }
        image_offsets.push(total);
        let total_observations = usize::try_from(total)
            .map_err(|_| RigCorrespondenceBuildError::ObservationCountNotRepresentable { total })?;
        Ok(Self {
            feature_counts: feature_counts.to_vec(),
            image_offsets,
            total_observations,
        })
    }

    /// Construct directly from feature sets, using each set's keypoint count.
    pub fn from_features(
        features: &[visloc_vision::features::FeatureSet],
    ) -> Result<Self, RigCorrespondenceBuildError> {
        let counts = features
            .iter()
            .map(|features| features.keypoints.len())
            .collect::<Vec<_>>();
        Self::new(&counts)
    }

    /// Borrow the feature-count vector used to validate pair endpoints.
    pub fn feature_counts(&self) -> &[usize] {
        &self.feature_counts
    }

    /// Number of flattened observations in this rig-local domain.
    pub fn total_observations(&self) -> usize {
        self.total_observations
    }

    /// Flattened image bases, including the trailing total-observation
    /// sentinel.
    pub fn image_offsets(&self) -> &[u64] {
        &self.image_offsets
    }

    /// Build a canonical CSR and collect preview measurements.
    pub fn build(
        &self,
        pairwise: &[PairwiseMatches],
    ) -> Result<RigCorrespondenceBuild, RigCorrespondenceBuildError> {
        let mut input_match_count = 0usize;
        for pair in pairwise {
            input_match_count = input_match_count
                .checked_add(pair.matches.len())
                .ok_or(RigCorrespondenceBuildError::MatchCountOverflow)?;
        }

        // First pass: flatten and canonicalize every verified relation.  The
        // temporary vector is one scalar pair per input match, never a Vec per
        // observation; its size is therefore proportional to the input.
        input_match_count
            .checked_mul(size_of::<(RigObservationId, RigObservationId)>())
            .ok_or(RigCorrespondenceBuildError::MatchCountOverflow)?;
        let mut undirected = Vec::with_capacity(input_match_count);
        for (pair_index, pair) in pairwise.iter().enumerate() {
            self.validate_pair_images(pair_index, pair)?;
            for (match_index, &(left, right)) in pair.matches.iter().enumerate() {
                let left_id =
                    self.flatten_observation(pair_index, match_index, pair.image_i, left)?;
                let right_id =
                    self.flatten_observation(pair_index, match_index, pair.image_j, right)?;
                undirected.push(if left_id < right_id {
                    (left_id, right_id)
                } else {
                    (right_id, left_id)
                });
            }
        }

        // Canonical edge order makes the result independent of pair and match
        // stream order.  Deduplicate before expanding to both directions so
        // duplicate_drops has the intuitive one-per-input-match meaning.
        undirected.sort_unstable();
        let mut unique_len = 0usize;
        let mut duplicate_drops = 0usize;
        for read in 0..undirected.len() {
            if unique_len != 0 && undirected[read] == undirected[unique_len - 1] {
                duplicate_drops = duplicate_drops
                    .checked_add(1)
                    .ok_or(RigCorrespondenceBuildError::MatchCountOverflow)?;
            } else {
                undirected[unique_len] = undirected[read];
                unique_len += 1;
            }
        }
        undirected.truncate(unique_len);
        let undirected_unique_edges = undirected.len();
        let directed_unique_edges = undirected_unique_edges
            .checked_mul(2)
            .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;

        // Second pass/count: row_offsets initially hold row degrees, then are
        // converted in place to prefix offsets.  This avoids a Vec<Vec<_>> and
        // keeps temporary state O(observations + unique matches).
        let row_offset_len = self
            .total_observations
            .checked_add(1)
            .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
        let mut row_offsets = vec![0u64; row_offset_len];
        for &(left, right) in &undirected {
            let left = usize::try_from(left)
                .map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)?;
            let right = usize::try_from(right)
                .map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)?;
            row_offsets[left] = row_offsets[left]
                .checked_add(1)
                .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
            row_offsets[right] = row_offsets[right]
                .checked_add(1)
                .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
        }
        let max_row_degree = row_offsets
            .iter()
            .take(self.total_observations)
            .map(|&degree| usize::try_from(degree).unwrap_or(usize::MAX))
            .max()
            .unwrap_or(0);
        let mut running = 0u64;
        for row_offset in row_offsets.iter_mut().take(self.total_observations) {
            let degree = *row_offset;
            *row_offset = running;
            running = running
                .checked_add(degree)
                .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
        }
        row_offsets[self.total_observations] = running;
        if usize::try_from(running).ok() != Some(directed_unique_edges) {
            return Err(RigCorrespondenceBuildError::EdgeCountOverflow);
        }

        // Fill each row from the canonical edge list, then sort each row.  A
        // second small cursor array is bounded by the observation count.
        directed_unique_edges
            .checked_mul(size_of::<RigObservationId>())
            .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
        let mut neighbors = vec![0u32; directed_unique_edges];
        let mut cursors = row_offsets[..self.total_observations].to_vec();
        for &(left, right) in &undirected {
            fill_neighbor(&mut neighbors, &mut cursors, left, right)?;
            fill_neighbor(&mut neighbors, &mut cursors, right, left)?;
        }
        for offsets in row_offsets.windows(2) {
            let start = usize::try_from(offsets[0])
                .map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)?;
            let end = usize::try_from(offsets[1])
                .map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)?;
            neighbors[start..end].sort_unstable();
        }

        let csr = RigCorrespondenceCsr {
            image_offsets: self.image_offsets.clone(),
            row_offsets,
            neighbors,
        };
        let estimated_persistent_bytes = csr
            .estimated_persistent_bytes()
            .ok_or(RigCorrespondenceBuildError::PersistentByteCountOverflow)?;
        let stats = RigCorrespondencePreviewStats {
            total_observations: self.total_observations,
            directed_unique_edges,
            undirected_unique_edges,
            duplicate_drops,
            estimated_persistent_bytes,
            max_row_degree,
            digest: csr.digest(),
        };
        Ok(RigCorrespondenceBuild { csr, stats })
    }

    /// Build only the compact graph, discarding the preview measurements.
    pub fn build_csr(
        &self,
        pairwise: &[PairwiseMatches],
    ) -> Result<RigCorrespondenceCsr, RigCorrespondenceBuildError> {
        self.build(pairwise).map(|output| output.csr)
    }

    /// Build and return only the preview measurements.
    pub fn preview_stats(
        &self,
        pairwise: &[PairwiseMatches],
    ) -> Result<RigCorrespondencePreviewStats, RigCorrespondenceBuildError> {
        self.build(pairwise).map(|output| output.stats)
    }

    fn validate_pair_images(
        &self,
        pair_index: usize,
        pair: &PairwiseMatches,
    ) -> Result<(), RigCorrespondenceBuildError> {
        if pair.image_i >= self.feature_counts.len() {
            return Err(RigCorrespondenceBuildError::ImageOutOfRange {
                pair_index,
                image: pair.image_i,
                image_count: self.feature_counts.len(),
            });
        }
        if pair.image_j >= self.feature_counts.len() {
            return Err(RigCorrespondenceBuildError::ImageOutOfRange {
                pair_index,
                image: pair.image_j,
                image_count: self.feature_counts.len(),
            });
        }
        if pair.image_i == pair.image_j {
            return Err(RigCorrespondenceBuildError::SelfCorrespondence {
                pair_index,
                image: pair.image_i,
            });
        }
        Ok(())
    }

    fn flatten_observation(
        &self,
        pair_index: usize,
        match_index: usize,
        image: usize,
        keypoint: usize,
    ) -> Result<RigObservationId, RigCorrespondenceBuildError> {
        let observation_count = self.feature_counts[image];
        if keypoint >= observation_count {
            return Err(RigCorrespondenceBuildError::ObservationOutOfRange {
                pair_index,
                match_index,
                image,
                keypoint,
                observation_count,
            });
        }
        let offset = self.image_offsets[image];
        let id = offset
            .checked_add(u64::try_from(keypoint).map_err(|_| {
                RigCorrespondenceBuildError::ObservationOutOfRange {
                    pair_index,
                    match_index,
                    image,
                    keypoint,
                    observation_count,
                }
            })?)
            .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
        RigObservationId::try_from(id).map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)
    }
}

fn fill_neighbor(
    neighbors: &mut [RigObservationId],
    cursors: &mut [u64],
    source: RigObservationId,
    target: RigObservationId,
) -> Result<(), RigCorrespondenceBuildError> {
    let source =
        usize::try_from(source).map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)?;
    let cursor = cursors
        .get_mut(source)
        .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
    let index =
        usize::try_from(*cursor).map_err(|_| RigCorrespondenceBuildError::EdgeCountOverflow)?;
    let slot = neighbors
        .get_mut(index)
        .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
    *slot = target;
    *cursor = cursor
        .checked_add(1)
        .ok_or(RigCorrespondenceBuildError::EdgeCountOverflow)?;
    Ok(())
}

/// Build a compact rig-local CSR from feature counts and verified pairs.
pub fn build_rig_correspondence_csr(
    feature_counts: &[usize],
    pairwise: &[PairwiseMatches],
) -> Result<RigCorrespondenceCsr, RigCorrespondenceBuildError> {
    RigCorrespondenceCsrBuilder::new(feature_counts)?.build_csr(pairwise)
}

/// Build a compact rig-local CSR and return its preview statistics.
pub fn build_rig_correspondence(
    feature_counts: &[usize],
    pairwise: &[PairwiseMatches],
) -> Result<RigCorrespondenceBuild, RigCorrespondenceBuildError> {
    RigCorrespondenceCsrBuilder::new(feature_counts)?.build(pairwise)
}

/// Collect bounded correspondence preview statistics without exposing the
/// temporary builder state to callers.
pub fn preview_rig_correspondence_stats(
    feature_counts: &[usize],
    pairwise: &[PairwiseMatches],
) -> Result<RigCorrespondencePreviewStats, RigCorrespondenceBuildError> {
    RigCorrespondenceCsrBuilder::new(feature_counts)?.preview_stats(pairwise)
}

/// Convenience wrapper using the existing feature-set representation.
pub fn build_rig_correspondence_csr_from_features(
    features: &[visloc_vision::features::FeatureSet],
    pairwise: &[PairwiseMatches],
) -> Result<RigCorrespondenceCsr, RigCorrespondenceBuildError> {
    RigCorrespondenceCsrBuilder::from_features(features)?.build_csr(pairwise)
}

/// Convenience preview wrapper using the existing feature-set representation.
pub fn preview_rig_correspondence_stats_from_features(
    features: &[visloc_vision::features::FeatureSet],
    pairwise: &[PairwiseMatches],
) -> Result<RigCorrespondencePreviewStats, RigCorrespondenceBuildError> {
    RigCorrespondenceCsrBuilder::from_features(features)?.preview_stats(pairwise)
}

fn hash_bytes(hash: &mut u64, bytes: &[u8]) {
    for &byte in bytes {
        *hash ^= u64::from(byte);
        *hash = hash.wrapping_mul(FNV_PRIME);
    }
}

fn hash_u32(hash: &mut u64, value: u32) {
    hash_bytes(hash, &value.to_le_bytes());
}

fn hash_u64(hash: &mut u64, value: u64) {
    hash_bytes(hash, &value.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(image_i: usize, image_j: usize, matches: &[(usize, usize)]) -> PairwiseMatches {
        PairwiseMatches::new(image_i, image_j, matches.to_vec())
    }

    #[test]
    fn input_permutation_and_duplicate_rows_produce_one_canonical_graph() {
        let counts = [3, 3, 2];
        let first = vec![
            pair(0, 1, &[(2, 1), (0, 0), (2, 1)]),
            pair(1, 2, &[(0, 1), (2, 0)]),
            pair(1, 0, &[(0, 0)]),
        ];
        let second = vec![
            pair(2, 1, &[(0, 2), (1, 0)]),
            pair(0, 1, &[(0, 0), (2, 1)]),
            pair(0, 1, &[(2, 1)]),
        ];
        let output_a = build_rig_correspondence(&counts, &first).unwrap();
        let output_b = build_rig_correspondence(&counts, &second).unwrap();
        assert_eq!(output_a.csr, output_b.csr);
        assert_eq!(output_a.stats.digest, output_b.stats.digest);
        assert_eq!(output_a.stats.duplicate_drops, 2);
        assert_eq!(output_a.stats.undirected_unique_edges, 4);
        assert_eq!(output_a.stats.directed_unique_edges, 8);
        assert_eq!(output_a.csr.neighbors_for(0), Some([3u32].as_slice()));
        assert_eq!(output_a.csr.neighbors_for(2), Some([4u32].as_slice()));
        // Row 3 has two neighbours and must be sorted regardless of input
        // pair/match order.
        assert_eq!(output_a.csr.neighbors_for(3), Some([0u32, 7u32].as_slice()));
    }

    #[test]
    fn invalid_image_keypoint_and_self_relations_are_rejected() {
        let invalid_image = build_rig_correspondence_csr(&[1, 1], &[pair(0, 2, &[(0, 0)])]);
        assert!(matches!(
            invalid_image,
            Err(RigCorrespondenceBuildError::ImageOutOfRange { .. })
        ));

        let invalid_keypoint = build_rig_correspondence_csr(&[1, 1], &[pair(0, 1, &[(1, 0)])]);
        assert!(matches!(
            invalid_keypoint,
            Err(RigCorrespondenceBuildError::ObservationOutOfRange { .. })
        ));

        let self_pair = build_rig_correspondence_csr(&[2], &[pair(0, 0, &[(0, 0)])]);
        assert!(matches!(
            self_pair,
            Err(RigCorrespondenceBuildError::SelfCorrespondence { .. })
        ));
    }

    #[test]
    fn observation_ids_use_each_image_boundary_and_a_u64_sentinel() {
        let output = build_rig_correspondence(&[1, 1, 0], &[]).unwrap();
        assert_eq!(output.csr.image_offsets, vec![0, 1, 2, 2]);
        assert_eq!(output.csr.observation_id(0, 0), Some(0));
        assert_eq!(output.csr.observation_id(1, 0), Some(1));
        assert_eq!(output.csr.observation_id(0, 1), None);
        assert_eq!(output.csr.observation_id(1, 1), None);
        assert_eq!(output.csr.observation_id(2, 0), None);
        assert_eq!(output.csr.image_count(), 3);
        assert_eq!(output.csr.observation(0), Some((0, 0)));
        assert_eq!(output.csr.observation(1), Some((1, 0)));
        assert_eq!(output.csr.observation(2), None);

        let leading_empty = build_rig_correspondence(&[0, 2, 0, 1], &[]).unwrap();
        assert_eq!(leading_empty.csr.observation(0), Some((1, 0)));
        assert_eq!(leading_empty.csr.observation(1), Some((1, 1)));
        assert_eq!(leading_empty.csr.observation(2), Some((3, 0)));

        // No CSR rows are allocated here: this only exercises the
        // representational boundary where the final image's empty sentinel is
        // exactly 2^32 and therefore cannot be a u32 observation id.
        if usize::BITS > 32 {
            let boundary = RigCorrespondenceCsrBuilder::new(&[u32::MAX as usize, 1, 0]).unwrap();
            assert_eq!(
                boundary.image_offsets(),
                &[
                    0,
                    u32::MAX as u64,
                    MAX_FLATTENED_OBSERVATIONS,
                    MAX_FLATTENED_OBSERVATIONS,
                ]
            );
        }
    }

    #[test]
    fn digest_is_stable_for_stream_permutations_and_changes_for_edges() {
        let counts = [2, 2, 2];
        let a = preview_rig_correspondence_stats(
            &counts,
            &[pair(0, 1, &[(1, 0)]), pair(1, 2, &[(1, 1)])],
        )
        .unwrap();
        let b = preview_rig_correspondence_stats(
            &counts,
            &[pair(2, 1, &[(1, 1)]), pair(1, 0, &[(0, 1)])],
        )
        .unwrap();
        let c = preview_rig_correspondence_stats(
            &counts,
            &[pair(0, 1, &[(0, 0)]), pair(1, 2, &[(1, 1)])],
        )
        .unwrap();
        assert_eq!(a.digest, b.digest);
        assert_eq!(a.deterministic_digest(), b.deterministic_digest());
        assert_ne!(a.digest, c.digest);
    }

    #[test]
    fn persistent_memory_accounting_grows_with_directed_edges() {
        let counts = [4, 4];
        let one = build_rig_correspondence(&counts, &[pair(0, 1, &[(0, 0)])]).unwrap();
        let two = build_rig_correspondence(&counts, &[pair(0, 1, &[(0, 0), (1, 1)])]).unwrap();
        assert_eq!(
            one.stats.estimated_persistent_bytes,
            one.csr.estimated_persistent_bytes().unwrap()
        );
        assert_eq!(
            two.stats.estimated_persistent_bytes,
            two.csr.estimated_persistent_bytes().unwrap()
        );
        assert_eq!(
            two.stats.estimated_persistent_bytes - one.stats.estimated_persistent_bytes,
            2 * size_of::<u32>()
        );
        assert!(two.stats.estimated_persistent_bytes > one.stats.estimated_persistent_bytes);
    }

    #[test]
    fn flattened_observation_overflow_is_checked_without_allocating_rows() {
        let error = RigCorrespondenceCsrBuilder::new(&[usize::MAX, 1]).unwrap_err();
        assert!(matches!(
            error,
            RigCorrespondenceBuildError::ObservationCountOverflow { .. }
                | RigCorrespondenceBuildError::FeatureCountOverflow { .. }
        ));
    }
}
