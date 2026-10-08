//! Mapper input preparation: union traversal order, match caps, edge hashes and locus canonicalization.

use super::*;

/// Reorder only the already-verified traversal stream.  This deliberately
/// leaves image/keypoint indices and every correspondence value untouched so
/// an A/B isolates legacy union traversal rather than matching or geometric
/// verification.  The two-argument wrapper is retained for the old index-only
/// controls and for unit tests; the executable uses the feature-aware helper
/// below for physical hashing.
#[cfg(test)]
pub(super) fn apply_union_traversal_order(
    pairwise: &mut [PairwiseMatches],
    order: UnionTraversalOrder,
) {
    apply_union_traversal_order_with_features(pairwise, order, &[]);
}

fn quantized_physical_coordinate(value: f64) -> i64 {
    let scaled = value * 1_000_000.0;
    if scaled.is_finite() {
        scaled.round().clamp(i64::MIN as f64, i64::MAX as f64) as i64
    } else {
        i64::MIN
    }
}

pub(super) fn physical_hash_mix(mut hash: u64, value: u64) -> u64 {
    for byte in value.to_le_bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

/// Return a deterministic ordering key for one verified physical edge.  The
/// coordinate quantization makes the primary key independent of feature-row
/// order while retaining feature indices only as a final collision breaker.
fn physical_edge_order_key(
    pair: &PairwiseMatches,
    keypoint_i: usize,
    keypoint_j: usize,
    seed: u64,
    features: &[FeatureSet],
) -> (u64, usize, usize, usize, usize) {
    let (image_i, image_j, keypoint_i, keypoint_j) = if pair.image_i <= pair.image_j {
        (pair.image_i, pair.image_j, keypoint_i, keypoint_j)
    } else {
        (pair.image_j, pair.image_i, keypoint_j, keypoint_i)
    };
    let point_i = features
        .get(image_i)
        .and_then(|set| set.keypoints.get(keypoint_i));
    let point_j = features
        .get(image_j)
        .and_then(|set| set.keypoints.get(keypoint_j));
    let (x_i, y_i) = point_i.map_or((i64::MIN, i64::MIN), |point| {
        (
            quantized_physical_coordinate(point.x),
            quantized_physical_coordinate(point.y),
        )
    });
    let (x_j, y_j) = point_j.map_or((i64::MIN, i64::MIN), |point| {
        (
            quantized_physical_coordinate(point.x),
            quantized_physical_coordinate(point.y),
        )
    });
    let mut hash = 0xcbf29ce484222325u64 ^ seed;
    for value in [
        image_i as u64,
        image_j as u64,
        x_i as u64,
        y_i as u64,
        x_j as u64,
        y_j as u64,
    ] {
        hash = physical_hash_mix(hash, value);
    }
    (hash, image_i, image_j, keypoint_i, keypoint_j)
}

fn physical_pair_order_key(
    pair: &PairwiseMatches,
    seed: u64,
    features: &[FeatureSet],
) -> (u64, usize, usize) {
    pair.matches
        .iter()
        .map(|&(keypoint_i, keypoint_j)| {
            let edge = physical_edge_order_key(pair, keypoint_i, keypoint_j, seed, features);
            (edge.0, edge.1, edge.2)
        })
        .min()
        .unwrap_or_else(|| {
            let mut hash = 0xcbf29ce484222325u64 ^ seed;
            hash = physical_hash_mix(hash, pair.image_i as u64);
            hash = physical_hash_mix(hash, pair.image_j as u64);
            (
                hash,
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            )
        })
}

/// Feature-aware variant of [`apply_union_traversal_order`].  Physical-hash
/// modes sort matches inside each pair and then pairs by their first physical
/// edge.  No correspondence is added, removed, or rewritten.
pub(super) fn apply_union_traversal_order_with_features(
    pairwise: &mut [PairwiseMatches],
    order: UnionTraversalOrder,
    features: &[FeatureSet],
) {
    if matches!(
        order,
        UnionTraversalOrder::ReverseMatches | UnionTraversalOrder::ReverseBoth
    ) {
        for pair in pairwise.iter_mut() {
            pair.matches.reverse();
        }
    }
    if matches!(
        order,
        UnionTraversalOrder::ReversePairs | UnionTraversalOrder::ReverseBoth
    ) {
        pairwise.reverse();
    }
    let (seed, descending) = match order {
        UnionTraversalOrder::PhysicalHash(seed) => (seed, false),
        UnionTraversalOrder::PhysicalHashReverse(seed) => (seed, true),
        _ => return,
    };
    for pair in pairwise.iter_mut() {
        let mut keyed_matches: Vec<_> = pair
            .matches
            .iter()
            .copied()
            .map(|(keypoint_i, keypoint_j)| {
                (
                    physical_edge_order_key(pair, keypoint_i, keypoint_j, seed, features),
                    (keypoint_i, keypoint_j),
                )
            })
            .collect();
        keyed_matches.sort_unstable_by(|lhs, rhs| lhs.0.cmp(&rhs.0));
        pair.matches = keyed_matches.into_iter().map(|(_, edge)| edge).collect();
    }
    pairwise.sort_unstable_by_key(|pair| physical_pair_order_key(pair, seed, features));
    if descending {
        for pair in pairwise.iter_mut() {
            pair.matches.reverse();
        }
        pairwise.reverse();
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct MapperMatchCapStats {
    pub(super) pairs_capped: usize,
    pub(super) matches_before: usize,
    pub(super) matches_after: usize,
    pub(super) essential_before: usize,
    pub(super) essential_after: usize,
}

/// Bound the correspondence stream consumed by the mapper while preserving
/// the complete verified stream for diagnostics and snapshot export.  The
/// verifier's deterministic inlier order is retained (rather than choosing
/// by a post-hoc score), so this option cannot introduce a second geometry
/// ranking policy.  `None`/the caller's omission is the historical path.
pub(super) fn cap_mapper_pair_matches(
    pairwise: &mut [PairwiseMatches],
    limit: usize,
) -> MapperMatchCapStats {
    let mut stats = MapperMatchCapStats::default();
    for pair in pairwise {
        stats.matches_before += pair.matches.len();
        stats.essential_before += pair.essential_matches.as_ref().map_or(0, Vec::len);
        let pair_matches_capped = pair.matches.len() > limit;
        let essential_capped = pair
            .essential_matches
            .as_ref()
            .is_some_and(|matches| matches.len() > limit);
        if pair_matches_capped || essential_capped {
            stats.pairs_capped += 1;
        }
        pair.matches.truncate(limit);
        if let Some(matches) = pair.essential_matches.as_mut() {
            matches.truncate(limit);
        }
        stats.matches_after += pair.matches.len();
        stats.essential_after += pair.essential_matches.as_ref().map_or(0, Vec::len);
    }
    stats
}

/// Stable FNV-1a hash of the multiset of verified `(image, keypoint)` edges.
/// Pair direction and traversal order are normalized, while duplicate edges
/// remain duplicated in the sorted stream.  It is a diagnostic integrity label
/// rather than a cryptographic digest.
pub(super) fn unordered_pairwise_edge_hash(pairwise: &[PairwiseMatches]) -> u64 {
    let mut pair_order = pairwise
        .iter()
        .enumerate()
        .map(|(index, pair)| {
            (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
                index,
            )
        })
        .collect::<Vec<_>>();
    pair_order.sort_unstable_by_key(|&(image_i, image_j, _)| (image_i, image_j));
    let mut hash = 0xcbf29ce484222325u64;
    let mut group_start = 0;
    while group_start < pair_order.len() {
        let (image_i, image_j, _) = pair_order[group_start];
        let mut group_end = group_start + 1;
        while group_end < pair_order.len()
            && pair_order[group_end].0 == image_i
            && pair_order[group_end].1 == image_j
        {
            group_end += 1;
        }
        let group_match_count = pair_order[group_start..group_end]
            .iter()
            .map(|&(_, _, index)| pairwise[index].matches.len())
            .sum();
        let mut matches = Vec::with_capacity(group_match_count);
        for &(_, _, index) in &pair_order[group_start..group_end] {
            let pair = &pairwise[index];
            let swapped = pair.image_i > pair.image_j;
            matches.extend(pair.matches.iter().map(|&(keypoint_i, keypoint_j)| {
                if swapped {
                    (keypoint_j, keypoint_i)
                } else {
                    (keypoint_i, keypoint_j)
                }
            }));
        }
        matches.sort_unstable();
        for (keypoint_i, keypoint_j) in matches {
            for value in [image_i, image_j, keypoint_i, keypoint_j] {
                for byte in (value as u64).to_le_bytes() {
                    hash ^= u64::from(byte);
                    hash = hash.wrapping_mul(0x100000001b3u64);
                }
            }
        }
        group_start = group_end;
    }
    hash
}

/// Parse `--track-source`'s value into the M2 [`TrackSource`] A/B switch.
/// `TrackSource` lives in `visloc-slam` and has no `FromStr` of its own (it's
/// a plain engine config knob, not a CLI type), so this demo owns the string
/// mapping.
pub(super) fn parse_track_source(s: &str) -> Result<TrackSource, String> {
    match s {
        "union-find" => Ok(TrackSource::UnionFind),
        "graph" => Ok(TrackSource::CorrespondenceGraph),
        other => Err(format!(
            "unknown --track-source {other:?} (expected union-find|graph)"
        )),
    }
}

pub(super) fn parse_diagnose_stems(raw: &str) -> Result<Vec<String>, String> {
    let stems: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|stem| !stem.is_empty())
        .map(str::to_owned)
        .collect();
    if stems.is_empty() {
        return Err("--diagnose-pair-stems requires at least one non-empty stem".into());
    }
    Ok(stems)
}

pub(super) fn image_stem(name: &str) -> &str {
    Path::new(name)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(name)
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct LocusCanonicalizationStats {
    pub(super) metadata_images: usize,
    pub(super) metadata_rows: usize,
    pub(super) physical_loci: usize,
    pub(super) collapsed_rows: usize,
    pub(super) input_matches: usize,
    pub(super) output_matches: usize,
    pub(super) deduplicated_matches: usize,
    pub(super) changed_pairs: usize,
}

const fn finite_order(value: f64) -> (u8, f64) {
    if value.is_finite() {
        (0, value)
    } else if value.is_nan() {
        (2, 0.0)
    } else if value.is_sign_negative() {
        (1, 0.0)
    } else {
        (3, 0.0)
    }
}

fn finite_order_cmp(lhs: f64, rhs: f64) -> CmpOrdering {
    let (lhs_class, lhs_value) = finite_order(lhs);
    let (rhs_class, rhs_value) = finite_order(rhs);
    lhs_class
        .cmp(&rhs_class)
        .then_with(|| lhs_value.total_cmp(&rhs_value))
}

fn locus_row_cmp(
    features: &[FeatureSet],
    metadata: &[Option<Vec<FeatureLocusMetadata>>],
    image: usize,
    lhs: usize,
    rhs: usize,
) -> CmpOrdering {
    let lhs_metadata = metadata
        .get(image)
        .and_then(|rows| rows.as_ref())
        .and_then(|rows| rows.get(lhs));
    let rhs_metadata = metadata
        .get(image)
        .and_then(|rows| rows.as_ref())
        .and_then(|rows| rows.get(rhs));
    let metadata_order = match (lhs_metadata, rhs_metadata) {
        (Some(lhs), Some(rhs)) => feature_locus_key(*lhs)
            .cmp(&feature_locus_key(*rhs))
            .then_with(|| finite_order_cmp(lhs.orientation, rhs.orientation)),
        (Some(_), None) => CmpOrdering::Less,
        (None, Some(_)) => CmpOrdering::Greater,
        (None, None) => CmpOrdering::Equal,
    };
    metadata_order
        .then_with(|| {
            let lhs = features.get(image).and_then(|set| set.keypoints.get(lhs));
            let rhs = features.get(image).and_then(|set| set.keypoints.get(rhs));
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => {
                    finite_order_cmp(lhs.x, rhs.x).then_with(|| finite_order_cmp(lhs.y, rhs.y))
                }
                (Some(_), None) => CmpOrdering::Less,
                (None, Some(_)) => CmpOrdering::Greater,
                (None, None) => CmpOrdering::Equal,
            }
        })
        .then_with(|| {
            let lhs = features.get(image).and_then(|set| set.descriptors.get(lhs));
            let rhs = features.get(image).and_then(|set| set.descriptors.get(rhs));
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => canonical_descriptor_cmp(lhs, rhs),
                (Some(_), None) => CmpOrdering::Less,
                (None, Some(_)) => CmpOrdering::Greater,
                (None, None) => CmpOrdering::Equal,
            }
        })
        .then_with(|| lhs.cmp(&rhs))
}

/// Build an old-row → representative-row map for every image.  Only rows
/// with complete finite metadata participate in physical grouping; absent or
/// malformed metadata intentionally falls back to identity for that row.
fn build_locus_representatives(
    features: &[FeatureSet],
    metadata: &[Option<Vec<FeatureLocusMetadata>>],
) -> Result<(Vec<Vec<usize>>, LocusCanonicalizationStats), String> {
    if features.len() != metadata.len() {
        return Err(format!(
            "orientation locus canonicalization: {} feature sets but {} metadata sets",
            features.len(),
            metadata.len()
        ));
    }
    let mut maps = Vec::with_capacity(features.len());
    let mut stats = LocusCanonicalizationStats::default();
    for (image, set) in features.iter().enumerate() {
        let mut representatives: Vec<usize> = (0..set.len()).collect();
        let Some(rows) = metadata[image].as_ref() else {
            maps.push(representatives);
            continue;
        };
        if rows.len() != set.len() {
            return Err(format!(
                "orientation locus canonicalization: image {image} has {} metadata rows but {} features",
                rows.len(),
                set.len()
            ));
        }
        stats.metadata_images += 1;
        stats.metadata_rows += rows.len();
        let mut groups: HashMap<FeatureLocusKey, Vec<usize>> = HashMap::new();
        for (row, &row_metadata) in rows.iter().enumerate() {
            if let Some(key) = feature_locus_key(row_metadata) {
                groups.entry(key).or_default().push(row);
            }
        }
        stats.physical_loci += groups.len();
        for rows in groups.values() {
            let representative = rows
                .iter()
                .copied()
                .min_by(|lhs, rhs| locus_row_cmp(features, metadata, image, *lhs, *rhs))
                .expect("non-empty locus group");
            for &row in rows {
                representatives[row] = representative;
            }
            stats.collapsed_rows += rows.len().saturating_sub(1);
        }
        maps.push(representatives);
    }
    Ok((maps, stats))
}

pub(super) fn descriptor_squared_distance(lhs: &[f32], rhs: &[f32]) -> f64 {
    if lhs.len() != rhs.len() {
        return f64::INFINITY;
    }
    let mut sum = 0.0;
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        let delta = f64::from(lhs) - f64::from(rhs);
        if !delta.is_finite() {
            return f64::INFINITY;
        }
        sum += delta * delta;
    }
    if sum.is_finite() {
        sum
    } else {
        f64::INFINITY
    }
}

fn match_geometry_residual(
    features: &[FeatureSet],
    pair: &PairwiseMatches,
    keypoint_i: usize,
    keypoint_j: usize,
    camera: Option<&Camera>,
) -> Option<f64> {
    let essential = pair.essential_matrix.as_ref()?;
    let camera = camera?;
    let point_i = features.get(pair.image_i)?.keypoints.get(keypoint_i)?;
    let point_j = features.get(pair.image_j)?.keypoints.get(keypoint_j)?;
    normalized_essential_squared_sampson_error(
        essential,
        &TwoViewCorrespondence::new(*point_i, *point_j),
        camera,
    )
}

pub(super) fn match_candidate_cmp(
    lhs: &(usize, usize, f64, Option<f64>),
    rhs: &(usize, usize, f64, Option<f64>),
) -> CmpOrdering {
    let distance_order = match (lhs.2.is_finite(), rhs.2.is_finite()) {
        (true, true) => lhs.2.total_cmp(&rhs.2),
        (true, false) => CmpOrdering::Less,
        (false, true) => CmpOrdering::Greater,
        (false, false) => CmpOrdering::Equal,
    };
    distance_order
        .then_with(|| match (lhs.3, rhs.3) {
            (Some(lhs), Some(rhs)) => match (lhs.is_finite(), rhs.is_finite()) {
                (true, true) => lhs.total_cmp(&rhs),
                (true, false) => CmpOrdering::Less,
                (false, true) => CmpOrdering::Greater,
                (false, false) => CmpOrdering::Equal,
            },
            (Some(_), None) => CmpOrdering::Less,
            (None, Some(_)) => CmpOrdering::Greater,
            (None, None) => CmpOrdering::Equal,
        })
        .then_with(|| lhs.0.cmp(&rhs.0))
        .then_with(|| lhs.1.cmp(&rhs.1))
}

/// Remap accepted correspondences from orientation rows to one representative
/// row per physical locus.  Matching still ran on every descriptor variant;
/// this post-verification step only prevents two orientations of one locus
/// from entering one multi-view track as separate same-image observations.
/// Duplicate locus-pairs choose the lowest primary descriptor distance, then
/// calibrated geometric residual when available, then stable source indices.
pub(super) fn canonicalize_pairwise_loci(
    features: &[FeatureSet],
    metadata: &[Option<Vec<FeatureLocusMetadata>>],
    pairwise: &mut [PairwiseMatches],
    camera: Option<&Camera>,
) -> Result<LocusCanonicalizationStats, String> {
    let (representatives, mut stats) = build_locus_representatives(features, metadata)?;
    // A legacy feature dump has no physical-locus metadata.  In that case the
    // opt-in flag must be a strict no-op: even sorting an identity-mapped
    // match stream would change the legacy UnionFind traversal basin.
    if stats.metadata_images == 0 {
        for pair in pairwise {
            stats.input_matches += pair.matches.len();
            stats.output_matches += pair.matches.len();
        }
        return Ok(stats);
    }
    for pair in pairwise {
        let map_i = representatives.get(pair.image_i).ok_or_else(|| {
            format!(
                "orientation locus canonicalization: invalid image index {}",
                pair.image_i
            )
        })?;
        let map_j = representatives.get(pair.image_j).ok_or_else(|| {
            format!(
                "orientation locus canonicalization: invalid image index {}",
                pair.image_j
            )
        })?;
        let before = pair.matches.len();
        stats.input_matches += before;
        let mut selected: HashMap<(usize, usize), (usize, usize, f64, Option<f64>)> =
            HashMap::new();
        for &(raw_i, raw_j) in &pair.matches {
            let Some(&canonical_i) = map_i.get(raw_i) else {
                continue;
            };
            let Some(&canonical_j) = map_j.get(raw_j) else {
                continue;
            };
            let distance = features
                .get(pair.image_i)
                .and_then(|set| set.descriptors.get(raw_i))
                .zip(
                    features
                        .get(pair.image_j)
                        .and_then(|set| set.descriptors.get(raw_j)),
                )
                .map_or(f64::INFINITY, |(lhs, rhs)| {
                    descriptor_squared_distance(lhs, rhs)
                });
            let candidate = (
                raw_i,
                raw_j,
                distance,
                match_geometry_residual(features, pair, raw_i, raw_j, camera),
            );
            let key = (canonical_i, canonical_j);
            if selected
                .get(&key)
                .is_none_or(|current| match_candidate_cmp(&candidate, current) == CmpOrdering::Less)
            {
                selected.insert(key, candidate);
            }
        }
        let mut selected_entries: Vec<_> = selected.into_iter().collect();
        selected_entries.sort_by(|lhs, rhs| {
            locus_row_cmp(features, metadata, pair.image_i, lhs.0 .0, rhs.0 .0)
                .then_with(|| locus_row_cmp(features, metadata, pair.image_j, lhs.0 .1, rhs.0 .1))
        });
        pair.matches = selected_entries
            .iter()
            .map(|&((canonical_i, canonical_j), _)| (canonical_i, canonical_j))
            .collect();

        // The essential subset is an independently consumed endpoint list in
        // the mapper.  Apply the same physical representative map and retain
        // one deterministic endpoint pair per locus, without assuming that
        // every imported E row also occurs in the winning `matches` vector.
        if let Some(essential_matches) = pair.essential_matches.as_mut() {
            let mut essential = HashMap::<(usize, usize), (usize, usize)>::new();
            for &(raw_i, raw_j) in essential_matches.iter() {
                let (Some(&canonical_i), Some(&canonical_j)) = (map_i.get(raw_i), map_j.get(raw_j))
                else {
                    continue;
                };
                essential
                    .entry((canonical_i, canonical_j))
                    .or_insert((canonical_i, canonical_j));
            }
            let mut essential_entries: Vec<_> = essential.into_values().collect();
            essential_entries.sort_by(|lhs, rhs| {
                locus_row_cmp(features, metadata, pair.image_i, lhs.0, rhs.0)
                    .then_with(|| locus_row_cmp(features, metadata, pair.image_j, lhs.1, rhs.1))
            });
            *essential_matches = essential_entries;
        }
        let after = pair.matches.len();
        stats.output_matches += after;
        stats.deduplicated_matches += before.saturating_sub(after);
        if before != after {
            stats.changed_pairs += 1;
        }
    }
    Ok(stats)
}
