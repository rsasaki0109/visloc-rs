//! Verified-pair snapshot export / import, validation and snapshot-backed image loading.

use super::*;

pub(super) struct ImportedVerifiedPair {
    pub(super) image_i: usize,
    pub(super) image_j: usize,
    pub(super) matches: Vec<(usize, usize)>,
    pub(super) config: ConfigurationType,
    pub(super) essential_matrix: Option<Matrix3<f64>>,
}

/// Verifier fields which are not currently consumed by PairwiseMatches but
/// are required to make a verified stream auditable and losslessly replayable.
#[derive(Debug, Clone, Default)]
pub(super) struct SnapshotPairMetadata {
    pub(super) raw_match_count: usize,
    pub(super) raw_matches: Vec<(usize, usize)>,
    pub(super) accepted_inlier_indices: Vec<usize>,
    pub(super) essential_inlier_indices: Vec<usize>,
    pub(super) e_inlier_count: usize,
    pub(super) f_inlier_count: usize,
    pub(super) h_inlier_count: usize,
    pub(super) fundamental: Option<Matrix3<f64>>,
    pub(super) homography: Option<Matrix3<f64>>,
    pub(super) relative_pose: Option<(Matrix3<f64>, Vector3<f64>)>,
}

const fn configuration_code(config: Option<ConfigurationType>) -> u8 {
    match config {
        None => 255,
        Some(ConfigurationType::Undefined) => 0,
        Some(ConfigurationType::Degenerate) => 1,
        Some(ConfigurationType::Uncalibrated) => 2,
        Some(ConfigurationType::Calibrated) => 3,
        Some(ConfigurationType::Planar) => 4,
        Some(ConfigurationType::Panoramic) => 5,
        Some(ConfigurationType::PlanarOrPanoramic) => 6,
        Some(ConfigurationType::Watermark) => 7,
        Some(ConfigurationType::Multiple) => 8,
    }
}

fn configuration_from_code(code: u8) -> Result<Option<ConfigurationType>, String> {
    Ok(match code {
        0 => Some(ConfigurationType::Undefined),
        1 => Some(ConfigurationType::Degenerate),
        2 => Some(ConfigurationType::Uncalibrated),
        3 => Some(ConfigurationType::Calibrated),
        4 => Some(ConfigurationType::Planar),
        5 => Some(ConfigurationType::Panoramic),
        6 => Some(ConfigurationType::PlanarOrPanoramic),
        7 => Some(ConfigurationType::Watermark),
        8 => Some(ConfigurationType::Multiple),
        255 => None,
        other => {
            return Err(format!(
                "verified-pair snapshot has unknown configuration code {other}"
            ))
        }
    })
}

fn matrix_bits(matrix: Option<&Matrix3<f64>>) -> Option<[u64; 9]> {
    matrix.map(|matrix| std::array::from_fn(|index| matrix.as_slice()[index].to_bits()))
}

fn matrix_from_bits(bits: Option<[u64; 9]>) -> Option<Matrix3<f64>> {
    bits.map(|bits| Matrix3::from_column_slice(&bits.map(f64::from_bits)))
}

fn vector_bits(vector: Option<&Vector3<f64>>) -> Option<[u64; 3]> {
    vector.map(|vector| std::array::from_fn(|index| vector[index].to_bits()))
}

pub(super) fn snapshot_metadata_from_report(
    raw_matches: &[DescriptorMatch],
    report: &TwoViewGeometryReport,
) -> SnapshotPairMetadata {
    SnapshotPairMetadata {
        raw_match_count: raw_matches.len(),
        raw_matches: raw_matches
            .iter()
            .map(|m| (m.query_index, m.train_index))
            .collect(),
        accepted_inlier_indices: report.inliers.clone(),
        essential_inlier_indices: report.essential_inliers.clone(),
        e_inlier_count: report.e_inlier_count,
        f_inlier_count: report.f_inlier_count,
        h_inlier_count: report.h_inlier_count,
        fundamental: report.fundamental,
        homography: report.homography,
        relative_pose: report.relative_pose,
    }
}

pub(super) fn snapshot_pair_record(
    pair: &PairwiseMatches,
    metadata: Option<&SnapshotPairMetadata>,
) -> SnapshotPairRecord {
    let fallback_accepted: Vec<u64> = (0..pair.matches.len() as u64).collect();
    let fallback_essential: Vec<u64> = pair
        .essential_matches
        .as_ref()
        .map(|matches| (0..matches.len() as u64).collect())
        .unwrap_or_default();
    let metadata = metadata.cloned().unwrap_or_default();
    SnapshotPairRecord {
        image_i: pair.image_i as u64,
        image_j: pair.image_j as u64,
        raw_match_count: if metadata.raw_match_count == 0 {
            pair.matches.len() as u64
        } else {
            metadata.raw_match_count as u64
        },
        raw_matches: if metadata.raw_matches.is_empty() {
            pair.matches
                .iter()
                .map(|&(left, right)| (left as u64, right as u64))
                .collect()
        } else {
            metadata
                .raw_matches
                .iter()
                .map(|&(left, right)| (left as u64, right as u64))
                .collect()
        },
        accepted_inlier_indices: if metadata.accepted_inlier_indices.is_empty() {
            fallback_accepted
        } else {
            metadata
                .accepted_inlier_indices
                .iter()
                .map(|&value| value as u64)
                .collect()
        },
        essential_inlier_indices: if metadata.essential_inlier_indices.is_empty() {
            fallback_essential
        } else {
            metadata
                .essential_inlier_indices
                .iter()
                .map(|&value| value as u64)
                .collect()
        },
        matches: pair
            .matches
            .iter()
            .map(|&(left, right)| (left as u64, right as u64))
            .collect(),
        essential_matches: pair.essential_matches.as_ref().map(|matches| {
            matches
                .iter()
                .map(|&(left, right)| (left as u64, right as u64))
                .collect()
        }),
        config: configuration_code(pair.two_view_config),
        calibrated: pair.two_view_config == Some(ConfigurationType::Calibrated),
        e_inlier_count: if metadata.e_inlier_count == 0 {
            pair.essential_matches.as_ref().map_or(0, Vec::len) as u64
        } else {
            metadata.e_inlier_count as u64
        },
        f_inlier_count: metadata.f_inlier_count as u64,
        h_inlier_count: metadata.h_inlier_count as u64,
        essential_matrix_bits: matrix_bits(pair.essential_matrix.as_ref()),
        fundamental_matrix_bits: matrix_bits(metadata.fundamental.as_ref()),
        homography_matrix_bits: matrix_bits(metadata.homography.as_ref()),
        relative_rotation_bits: metadata
            .relative_pose
            .as_ref()
            .and_then(|(rotation, _)| matrix_bits(Some(rotation))),
        relative_translation_bits: metadata
            .relative_pose
            .as_ref()
            .and_then(|(_, translation)| vector_bits(Some(translation))),
    }
}

pub(super) fn snapshot_metadata_map_from_pairs(
    pairwise: &[PairwiseMatches],
) -> HashMap<(usize, usize), SnapshotPairMetadata> {
    pairwise
        .iter()
        .map(|pair| {
            (
                (
                    pair.image_i.min(pair.image_j),
                    pair.image_i.max(pair.image_j),
                ),
                SnapshotPairMetadata::default(),
            )
        })
        .collect()
}

fn snapshot_metadata_from_record(record: &SnapshotPairRecord) -> SnapshotPairMetadata {
    SnapshotPairMetadata {
        raw_match_count: record.raw_match_count as usize,
        raw_matches: record
            .raw_matches
            .iter()
            .map(|&(left, right)| (left as usize, right as usize))
            .collect(),
        accepted_inlier_indices: record
            .accepted_inlier_indices
            .iter()
            .map(|&value| value as usize)
            .collect(),
        essential_inlier_indices: record
            .essential_inlier_indices
            .iter()
            .map(|&value| value as usize)
            .collect(),
        e_inlier_count: record.e_inlier_count as usize,
        f_inlier_count: record.f_inlier_count as usize,
        h_inlier_count: record.h_inlier_count as usize,
        fundamental: matrix_from_bits(record.fundamental_matrix_bits),
        homography: matrix_from_bits(record.homography_matrix_bits),
        relative_pose: matrix_from_bits(record.relative_rotation_bits).and_then(|rotation| {
            vector_from_bits(record.relative_translation_bits)
                .map(|translation| (rotation, translation))
        }),
    }
}

pub(super) fn snapshot_metadata_map_from_snapshot(
    snapshot: &VerifiedPairSnapshot,
) -> HashMap<(usize, usize), SnapshotPairMetadata> {
    snapshot
        .pairs
        .iter()
        .map(|record| {
            (
                (
                    record.image_i.min(record.image_j) as usize,
                    record.image_i.max(record.image_j) as usize,
                ),
                snapshot_metadata_from_record(record),
            )
        })
        .collect()
}

/// Promote only stable F-winning edges to a calibrated E for the opt-in
/// sequence fallback.  The normal pair stream remains F-winning and keeps its
/// original matrix; this pass is deliberately performed after verification,
/// using the lossless raw-match metadata already captured for diagnostics.
/// Thus a sequence fallback can use `Kᵀ F K` when the direct E estimate is a
/// façade-biased outlier without changing ordinary track construction.
#[derive(Debug, Clone, Default)]
pub(super) struct SequenceFToEPromotionStats {
    pub(super) promoted: usize,
    pub(super) high_support_overrides: usize,
    pub(super) high_support_override_pair_indices: Vec<usize>,
}

pub(super) fn promote_sequence_fundamentals_to_essentials(
    pairwise: &mut [PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    features: &[FeatureSet],
    camera: &Camera,
) -> SequenceFToEPromotionStats {
    let mut stats = SequenceFToEPromotionStats::default();
    for (pair_index, pair) in pairwise.iter_mut().enumerate() {
        if pair.two_view_config != Some(ConfigurationType::Uncalibrated)
            || pair.essential_matches.is_some()
        {
            continue;
        }
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        let Some(metadata) = metadata_by_pair.get(&key) else {
            continue;
        };
        let Some(fundamental) = metadata.fundamental else {
            continue;
        };
        if metadata.raw_matches.len() < 8
            || pair.image_i >= features.len()
            || pair.image_j >= features.len()
        {
            continue;
        }
        let correspondences = metadata
            .raw_matches
            .iter()
            .filter_map(|&(keypoint_i, keypoint_j)| {
                Some(TwoViewCorrespondence::new(
                    *features[pair.image_i].keypoints.get(keypoint_i)?,
                    *features[pair.image_j].keypoints.get(keypoint_j)?,
                ))
            })
            .collect::<Vec<_>>();
        if correspondences.len() < 8 {
            continue;
        }
        let report = TwoViewGeometryReport {
            config: ConfigurationType::Uncalibrated,
            inliers: metadata.accepted_inlier_indices.clone(),
            essential: pair.essential_matrix,
            fundamental: Some(fundamental),
            homography: metadata.homography,
            relative_pose: metadata.relative_pose,
            essential_inliers: metadata.essential_inlier_indices.clone(),
            e_inlier_count: metadata.e_inlier_count,
            f_inlier_count: metadata.f_inlier_count,
            h_inlier_count: metadata.h_inlier_count,
        };
        let Some(diagnostics) = f_to_e_candidate_diagnostics(&report, &correspondences, camera)
        else {
            continue;
        };
        let strict_sequence_gate = sequence_f_to_e_stability_gate(&diagnostics);
        let high_support_override =
            !strict_sequence_gate && sequence_f_to_e_high_support_override_gate(&diagnostics);
        if !strict_sequence_gate && !high_support_override {
            continue;
        }
        let Some(essential) = project_fundamental_to_essential(&fundamental, camera) else {
            continue;
        };
        pair.essential_matrix = Some(essential);
        stats.promoted += 1;
        if high_support_override {
            stats.high_support_overrides += 1;
            stats.high_support_override_pair_indices.push(pair_index);
        }
        if std::env::var_os("VISLOC_SFM_DEBUG").is_some() {
            eprintln!(
                "sfm-debug-sequence-f2e: {} {} gate={} f_inliers={} ef_inliers={} overlap={:.6} cheirality={:.6} margin={:.6} angle_p25_deg={:.6} rotation_spread_deg={:.6} translation_spread_deg={:.6}",
                pair.image_i,
                pair.image_j,
                if high_support_override {
                    "high_support_translation_spread_override"
                } else {
                    "strict"
                },
                diagnostics.f_inliers,
                diagnostics.ef_inliers,
                diagnostics.ef_overlap_on_f,
                diagnostics.cheirality_ratio,
                diagnostics.cheirality_margin,
                diagnostics.ef_angle_p25_deg,
                diagnostics.pose_rotation_spread_deg,
                diagnostics.pose_translation_spread_deg,
            );
        }
    }
    stats
}

pub(super) fn snapshot_indices_for_matches(
    raw_matches: &[(usize, usize)],
    matches: &[(usize, usize)],
) -> Option<Vec<usize>> {
    let mut used = vec![false; raw_matches.len()];
    let mut indices = Vec::with_capacity(matches.len());
    for &needle in matches {
        let index = raw_matches
            .iter()
            .enumerate()
            .find(|(index, value)| **value == needle && !used[*index])
            .map(|(index, _)| index)?;
        used[index] = true;
        indices.push(index);
    }
    Some(indices)
}

pub(super) fn snapshot_metadata_matches_pair(
    pair: &PairwiseMatches,
    metadata: &SnapshotPairMetadata,
) -> bool {
    if metadata.raw_match_count != metadata.raw_matches.len()
        || metadata.accepted_inlier_indices.len() != pair.matches.len()
    {
        return false;
    }
    let Some(accepted) = snapshot_indices_for_matches(&metadata.raw_matches, &pair.matches) else {
        return false;
    };
    if accepted != metadata.accepted_inlier_indices {
        return false;
    }
    match (
        &pair.essential_matches,
        metadata.essential_inlier_indices.as_slice(),
    ) {
        (Some(matches), indices) => {
            snapshot_indices_for_matches(&metadata.raw_matches, matches).as_deref() == Some(indices)
        }
        (None, []) => true,
        (None, _) => false,
    }
}

pub(super) fn write_verified_pair_snapshot(
    path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
) -> Result<(), String> {
    write_verified_pair_snapshot_with_validation(
        path,
        image_names,
        features,
        camera,
        pairwise,
        metadata_by_pair,
        args,
        None,
        false,
    )
}

pub(super) fn write_verified_pair_snapshot_atomic(
    path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
    feature_validation: &SnapshotFeatureValidation,
) -> Result<(), String> {
    write_verified_pair_snapshot_with_validation(
        path,
        image_names,
        features,
        camera,
        pairwise,
        metadata_by_pair,
        args,
        Some(feature_validation),
        true,
    )
}

fn write_verified_pair_snapshot_with_validation(
    path: &Path,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
    feature_validation: Option<&SnapshotFeatureValidation>,
    atomic: bool,
) -> Result<(), String> {
    let snapshot = snapshot_for_export(
        image_names,
        features,
        camera,
        pairwise,
        metadata_by_pair,
        args,
        feature_validation,
    )?;
    if args.shared_snapshot_envelope {
        verified_pair_snapshot::write_shared_atomic(path, &snapshot)
    } else if atomic {
        verified_pair_snapshot::write_atomic(path, &snapshot)
    } else {
        verified_pair_snapshot::write(path, &snapshot)
    }
}

pub(super) fn snapshot_for_export(
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    metadata_by_pair: &HashMap<(usize, usize), SnapshotPairMetadata>,
    args: &Args,
    feature_validation: Option<&SnapshotFeatureValidation>,
) -> Result<VerifiedPairSnapshot, String> {
    let feature_counts: Vec<u64> = feature_validation.map_or_else(
        || {
            features
                .iter()
                .map(|features| features.keypoints.len() as u64)
                .collect()
        },
        |validation| {
            validation
                .feature_counts
                .iter()
                .map(|&count| count as u64)
                .collect()
        },
    );
    let records: Vec<SnapshotPairRecord> = pairwise
        .iter()
        .map(|pair| {
            let key = (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            );
            let metadata = metadata_by_pair
                .get(&key)
                .filter(|metadata| snapshot_metadata_matches_pair(pair, metadata));
            snapshot_pair_record(pair, metadata)
        })
        .collect();
    let verifier_config = snapshot_verifier_config(args);
    let snapshot_config = snapshot_export_config(args);
    let snapshot = VerifiedPairSnapshot {
        schema_version: verified_pair_snapshot::SCHEMA_VERSION,
        image_names: image_names.to_vec(),
        image_manifest_hash: snapshot_image_manifest_hash(image_names),
        feature_manifest_hash: feature_validation.map_or_else(
            || snapshot_feature_manifest_hash(features),
            |validation| validation.feature_manifest_hash,
        ),
        feature_counts,
        width: u64::from(camera.width),
        height: u64::from(camera.height),
        intrinsics_bits: snapshot_intrinsics_bits(camera)?,
        // The full Args debug snapshot is already in the phase log. It
        // contains candidate/output paths, so storing it here made otherwise
        // identical snapshots differ across resumable run roots. Keep the
        // binary envelope path-independent; pair hashes and the runner index
        // retain the data/provenance bindings.
        effective_config_hash: effective_config_hash(&snapshot_config),
        effective_config: snapshot_config,
        verifier_config_hash: effective_config_hash(&verifier_config),
        verifier_config,
        pair_order_hash: ordered_pairwise_edge_hash(pairwise),
        unordered_edge_hash: unordered_pairwise_edge_hash(pairwise),
        accepted_match_count: pairwise
            .iter()
            .map(|pair| pair.matches.len())
            .sum::<usize>() as u64,
        pairs: records,
    };
    Ok(snapshot)
}

fn vector_from_bits(bits: Option<[u64; 3]>) -> Option<Vector3<f64>> {
    bits.map(|bits| Vector3::from_column_slice(&bits.map(f64::from_bits)))
}

fn snapshot_image_manifest_hash(image_names: &[String]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, image_names.len() as u64);
    for name in image_names {
        hash = physical_hash_mix(hash, name.len() as u64);
        for byte in name.as_bytes() {
            hash = physical_hash_mix(hash, u64::from(*byte));
        }
    }
    hash
}

pub(super) fn snapshot_feature_manifest_hash(features: &[FeatureSet]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, features.len() as u64);
    for feature_set in features {
        hash = physical_hash_mix(hash, feature_set.keypoints.len() as u64);
        hash = physical_hash_mix(hash, feature_set.descriptors.len() as u64);
        for keypoint in &feature_set.keypoints {
            hash = physical_hash_mix(hash, keypoint.x.to_bits());
            hash = physical_hash_mix(hash, keypoint.y.to_bits());
        }
        for descriptor in &feature_set.descriptors {
            hash = physical_hash_mix(hash, descriptor.len() as u64);
            for value in descriptor {
                hash = physical_hash_mix(hash, u64::from(value.to_bits()));
            }
        }
    }
    hash
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct SnapshotFeatureValidation {
    pub(super) feature_counts: Vec<usize>,
    pub(super) feature_manifest_hash: u64,
}

/// Reconstruct the exact v1 feature-manifest hash without retaining the
/// descriptor bank.  The first pass records a compact bitwise fingerprint for
/// every source file before dropping its descriptors.  This pass re-parses one
/// file at a time, verifies that the source is unchanged, and feeds the
/// calibrated in-memory keypoints plus the original descriptor bits through
/// the same hash stream as [`snapshot_feature_manifest_hash`].
pub(super) fn snapshot_feature_validation_from_files(
    paths: &[PathBuf],
    features: &[FeatureSet],
    fingerprints: &[SnapshotFeatureFileFingerprint],
) -> Result<SnapshotFeatureValidation, String> {
    if paths.len() != features.len() || paths.len() != fingerprints.len() {
        return Err(format!(
            "snapshot feature replay manifest mismatch: {} source paths, {} feature sets, {} fingerprints",
            paths.len(),
            features.len(),
            fingerprints.len()
        ));
    }
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, features.len() as u64);
    for (image, ((path, feature_set), expected)) in
        paths.iter().zip(features).zip(fingerprints).enumerate()
    {
        if feature_set.keypoints.len() != feature_set.descriptors.len() {
            return Err(format!(
                "snapshot keypoint-only image {image} has {} keypoints but {} placeholder descriptor rows",
                feature_set.keypoints.len(),
                feature_set.descriptors.len()
            ));
        }
        let source = read_feature_set(path).map_err(|error| {
            format!("cannot re-read snapshot feature source image {image} ({path:?}): {error}")
        })?;
        let observed = snapshot_feature_file_fingerprint(&source);
        if observed != *expected {
            return Err(format!(
                "snapshot feature source image {image} ({path:?}) changed between loads"
            ));
        }
        if observed.keypoint_count != feature_set.keypoints.len()
            || observed.descriptor_count != feature_set.descriptors.len()
        {
            return Err(format!(
                "snapshot feature source image {image} ({path:?}) has {} keypoints / {} descriptor rows, loaded {} / {}",
                observed.keypoint_count,
                observed.descriptor_count,
                feature_set.keypoints.len(),
                feature_set.descriptors.len(),
            ));
        }
        hash = physical_hash_mix(hash, feature_set.keypoints.len() as u64);
        hash = physical_hash_mix(hash, source.descriptors.len() as u64);
        for keypoint in &feature_set.keypoints {
            hash = physical_hash_mix(hash, keypoint.x.to_bits());
            hash = physical_hash_mix(hash, keypoint.y.to_bits());
        }
        for descriptor in &source.descriptors {
            hash = physical_hash_mix(hash, descriptor.len() as u64);
            for value in descriptor {
                hash = physical_hash_mix(hash, u64::from(value.to_bits()));
            }
        }
    }
    Ok(SnapshotFeatureValidation {
        feature_counts: features.iter().map(FeatureSet::len).collect(),
        feature_manifest_hash: hash,
    })
}

fn snapshot_intrinsics_bits(camera: &Camera) -> Result<[u64; 4], String> {
    let Some((fx, fy, cx, cy)) = camera.intrinsics() else {
        return Err("verified-pair snapshot requires a camera with intrinsics".into());
    };
    let values = [fx, fy, cx, cy];
    if values.iter().any(|value| !value.is_finite()) {
        return Err("camera intrinsics contain a non-finite value".into());
    }
    Ok(values.map(f64::to_bits))
}

/// The verifier knobs which affect the initial pair stream.  Paths and mapper
/// knobs are intentionally absent: importing a snapshot must not require the
/// original matcher input files or rerun any of those decisions.
fn snapshot_verifier_config(args: &Args) -> String {
    format!(
        "mode={:?};ratio_bits={:08x};min_matches={};cross_check=1;guided={};multiple_models={};min_e_f={:?};calibrated_prefer_essential={};refine_f2e={};strict_f2e={};calibrated_essential_primary={};force_essential={};force_essential_ratio_bits={:016x};force_essential_min={};force_essential_uncalibrated_only={};colmap_guided={}",
        args.verification_mode,
        args.match_ratio.to_bits(),
        args.min_matches,
        args.guided_matching,
        args.multiple_models,
        args.min_e_f_inlier_ratio,
        args.calibrated_prefer_essential,
        args.refine_uncalibrated_f_to_essential,
        args.strict_uncalibrated_f_to_essential,
        args.calibrated_essential_primary,
        args.force_essential_matches,
        args.force_essential_min_ef_ratio.to_bits(),
        args.force_essential_min_e_inliers,
        args.force_essential_uncalibrated_only,
        args.colmap_guided_matching,
    )
}

pub(super) fn snapshot_export_config(args: &Args) -> String {
    format!("verified-pair-export-v1;{}", snapshot_verifier_config(args))
}

/// Hash the exact pair and correspondence order consumed by track building.
/// This is deliberately stronger than [`unordered_pairwise_edge_hash`]: pair
/// order, direction, accepted order, essential subset order, configuration,
/// and the stored essential matrix all contribute.
pub(super) fn ordered_pairwise_edge_hash(pairwise: &[PairwiseMatches]) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    hash = physical_hash_mix(hash, pairwise.len() as u64);
    for pair in pairwise {
        hash = physical_hash_mix(hash, pair.image_i as u64);
        hash = physical_hash_mix(hash, pair.image_j as u64);
        hash = physical_hash_mix(hash, configuration_code(pair.two_view_config) as u64);
        hash = physical_hash_mix(hash, pair.matches.len() as u64);
        for &(left, right) in &pair.matches {
            hash = physical_hash_mix(hash, left as u64);
            hash = physical_hash_mix(hash, right as u64);
        }
        match &pair.essential_matches {
            Some(matches) => {
                hash = physical_hash_mix(hash, 1);
                hash = physical_hash_mix(hash, matches.len() as u64);
                for &(left, right) in matches {
                    hash = physical_hash_mix(hash, left as u64);
                    hash = physical_hash_mix(hash, right as u64);
                }
            }
            None => hash = physical_hash_mix(hash, 0),
        }
        match &pair.essential_matrix {
            Some(matrix) => {
                hash = physical_hash_mix(hash, 1);
                for value in matrix.as_slice() {
                    hash = physical_hash_mix(hash, value.to_bits());
                }
            }
            None => hash = physical_hash_mix(hash, 0),
        }
    }
    hash
}

pub(super) fn candidate_incident_images(candidates: &[(usize, usize)]) -> Vec<usize> {
    let mut images = candidates
        .iter()
        .flat_map(|&(left, right)| [left, right])
        .collect::<Vec<_>>();
    images.sort_unstable();
    images.dedup();
    images
}

pub(super) fn hydrate_match_images(
    features: &mut [FeatureSet],
    paths: &[PathBuf],
    fingerprints: &[SnapshotFeatureFileFingerprint],
    images: &[usize],
) -> Result<(), String> {
    if paths.len() != features.len() || fingerprints.len() != features.len() {
        return Err(format!(
            "streamed match feature manifest mismatch: {} feature sets, {} paths, {} fingerprints",
            features.len(),
            paths.len(),
            fingerprints.len(),
        ));
    }
    for &image in images {
        let path = paths
            .get(image)
            .ok_or_else(|| format!("candidate references missing feature image {image}"))?;
        let source = read_feature_set(path)
            .map_err(|error| format!("cannot stream match feature {path:?}: {error}"))?;
        let observed = snapshot_feature_file_fingerprint(&source);
        if observed != fingerprints[image] {
            return Err(format!(
                "streamed match feature image {image} ({path:?}) changed after preflight"
            ));
        }
        if source.keypoints.len() != features[image].keypoints.len() {
            return Err(format!(
                "streamed match feature image {image} has {} source and {} calibrated keypoints",
                source.keypoints.len(),
                features[image].keypoints.len(),
            ));
        }
        features[image].descriptors = source.descriptors;
    }
    Ok(())
}

pub(super) fn verification_stats_from_snapshot(
    snapshot: &VerifiedPairSnapshot,
    compact_mapper_replay: bool,
) -> Result<VerificationStats, String> {
    let mut stats = VerificationStats::default();
    for (index, pair) in snapshot.pairs.iter().enumerate() {
        let config = configuration_from_code(pair.config)?;
        if let Some(config) = config {
            stats.record(config);
        } else if !compact_mapper_replay
            && pair.accepted_inlier_indices.is_empty()
            && !pair.matches.is_empty()
        {
            return Err(format!(
                "snapshot pair {index} has accepted matches but no configuration"
            ));
        }
    }
    Ok(stats)
}

pub(super) fn validate_snapshot_for_run(
    snapshot: &VerifiedPairSnapshot,
    image_names: &[String],
    features: &[FeatureSet],
    camera: &Camera,
    precomputed_feature_validation: Option<&SnapshotFeatureValidation>,
    compact_mapper_replay: bool,
) -> Result<Vec<PairwiseMatches>, String> {
    if snapshot.schema_version != verified_pair_snapshot::SCHEMA_VERSION {
        return Err(format!(
            "unsupported verified-pair snapshot schema {}",
            snapshot.schema_version
        ));
    }
    if snapshot.image_names != image_names {
        return Err(
            "verified-pair snapshot image manifest names do not match loaded images".into(),
        );
    }
    let computed_feature_counts: Vec<usize> = features.iter().map(|f| f.keypoints.len()).collect();
    let feature_counts = precomputed_feature_validation
        .map_or(computed_feature_counts.as_slice(), |validation| {
            validation.feature_counts.as_slice()
        });
    let snapshot_counts: Vec<usize> = snapshot
        .feature_counts
        .iter()
        .map(|&value| {
            usize::try_from(value)
                .map_err(|_| format!("snapshot feature count {value} does not fit usize"))
        })
        .collect::<Result<_, _>>()?;
    if snapshot_counts != feature_counts {
        return Err(format!(
            "verified-pair snapshot feature counts do not match loaded features ({snapshot_counts:?} vs {feature_counts:?})"
        ));
    }
    let image_hash = snapshot_image_manifest_hash(image_names);
    if snapshot.image_manifest_hash != image_hash {
        return Err(format!(
            "verified-pair snapshot image manifest hash mismatch: stored {:016x}, loaded {image_hash:016x}",
            snapshot.image_manifest_hash
        ));
    }
    let feature_hash = precomputed_feature_validation.map_or_else(
        || snapshot_feature_manifest_hash(features),
        |validation| validation.feature_manifest_hash,
    );
    if snapshot.feature_manifest_hash != feature_hash {
        return Err(format!(
            "verified-pair snapshot feature manifest hash mismatch: stored {:016x}, loaded {feature_hash:016x}",
            snapshot.feature_manifest_hash
        ));
    }
    if snapshot.width != u64::from(camera.width) || snapshot.height != u64::from(camera.height) {
        return Err(format!(
            "verified-pair snapshot camera dimensions {}x{} do not match loaded {}x{}",
            snapshot.width, snapshot.height, camera.width, camera.height
        ));
    }
    if snapshot.intrinsics_bits != snapshot_intrinsics_bits(camera)? {
        return Err("verified-pair snapshot camera intrinsics do not match loaded camera".into());
    }
    if effective_config_hash(&snapshot.effective_config) != snapshot.effective_config_hash {
        return Err("verified-pair snapshot effective-config checksum is invalid".into());
    }
    if effective_config_hash(&snapshot.verifier_config) != snapshot.verifier_config_hash {
        return Err("verified-pair snapshot verifier-config checksum is invalid".into());
    }
    let pairwise = pairwise_from_snapshot(snapshot, feature_counts, compact_mapper_replay)?;
    let ordered_hash = ordered_pairwise_edge_hash(&pairwise);
    if snapshot.pair_order_hash != ordered_hash {
        return Err(format!(
            "verified-pair snapshot pair-order hash mismatch: stored {:016x}, loaded {ordered_hash:016x}",
            snapshot.pair_order_hash
        ));
    }
    let unordered_hash = unordered_pairwise_edge_hash(&pairwise);
    if snapshot.unordered_edge_hash != unordered_hash {
        return Err(format!(
            "verified-pair snapshot unordered-edge hash mismatch: stored {:016x}, loaded {unordered_hash:016x}",
            snapshot.unordered_edge_hash
        ));
    }
    let accepted_match_count: usize = pairwise.iter().map(|pair| pair.matches.len()).sum();
    if snapshot.accepted_match_count != accepted_match_count as u64 {
        return Err(format!(
            "verified-pair snapshot accepted-match count {} does not match loaded {accepted_match_count}",
            snapshot.accepted_match_count
        ));
    }
    Ok(pairwise)
}

fn pairwise_from_snapshot(
    snapshot: &VerifiedPairSnapshot,
    feature_counts: &[usize],
    compact_mapper_replay: bool,
) -> Result<Vec<PairwiseMatches>, String> {
    let mut pairs = Vec::with_capacity(snapshot.pairs.len());
    let mut seen = HashSet::new();
    for (pair_number, record) in snapshot.pairs.iter().enumerate() {
        let image_i = usize::try_from(record.image_i)
            .map_err(|_| format!("snapshot pair {pair_number} image_i does not fit usize"))?;
        let image_j = usize::try_from(record.image_j)
            .map_err(|_| format!("snapshot pair {pair_number} image_j does not fit usize"))?;
        if image_i == image_j || image_i >= feature_counts.len() || image_j >= feature_counts.len()
        {
            return Err(format!(
                "snapshot pair {pair_number} has invalid image indices ({image_i},{image_j})"
            ));
        }
        let key = (image_i.min(image_j), image_i.max(image_j));
        if !seen.insert(key) {
            return Err(format!(
                "snapshot contains duplicate image pair ({},{})",
                key.0, key.1
            ));
        }
        let config = configuration_from_code(record.config)?;
        if record.calibrated != (config == Some(ConfigurationType::Calibrated)) {
            return Err(format!(
                "snapshot pair {pair_number} calibrated flag disagrees with configuration code"
            ));
        }
        let matches = record
            .matches
            .iter()
            .map(|&(left, right)| {
                let left = usize::try_from(left).map_err(|_| {
                    format!("snapshot pair {pair_number} query index does not fit usize")
                })?;
                let right = usize::try_from(right).map_err(|_| {
                    format!("snapshot pair {pair_number} train index does not fit usize")
                })?;
                if left >= feature_counts[image_i] || right >= feature_counts[image_j] {
                    return Err(format!(
                        "snapshot pair {pair_number} accepted match ({left},{right}) is outside feature counts ({},{})",
                        feature_counts[image_i], feature_counts[image_j]
                    ));
                }
                Ok((left, right))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let essential_matches = record
            .essential_matches
            .as_ref()
            .map(|values| {
                values
                    .iter()
                    .map(|&(left, right)| {
                        let left = usize::try_from(left).map_err(|_| {
                            format!("snapshot pair {pair_number} essential query index does not fit usize")
                        })?;
                        let right = usize::try_from(right).map_err(|_| {
                            format!("snapshot pair {pair_number} essential train index does not fit usize")
                        })?;
                        if left >= feature_counts[image_i] || right >= feature_counts[image_j] {
                            return Err(format!(
                                "snapshot pair {pair_number} essential match ({left},{right}) is outside feature counts"
                            ));
                        }
                        Ok((left, right))
                    })
                    .collect::<Result<Vec<_>, String>>()
            })
            .transpose()?;
        let validate_indices = |label: &str, values: &[u64], limit: usize| -> Result<(), String> {
            for &value in values {
                let index = usize::try_from(value).map_err(|_| {
                    format!("snapshot pair {pair_number} {label} index does not fit usize")
                })?;
                if index >= limit {
                    return Err(format!(
                        "snapshot pair {pair_number} {label} index {index} is outside 0..{limit}"
                    ));
                }
            }
            Ok(())
        };
        if !compact_mapper_replay {
            if let Some(max_index) = record.accepted_inlier_indices.iter().max() {
                if *max_index >= record.raw_match_count {
                    return Err(format!(
                    "snapshot pair {pair_number} accepted inlier index {max_index} >= raw match count {}",
                    record.raw_match_count
                ));
                }
            }
            if let Some(max_index) = record.essential_inlier_indices.iter().max() {
                if *max_index >= record.raw_match_count {
                    return Err(format!(
                    "snapshot pair {pair_number} essential inlier index {max_index} >= raw match count {}",
                    record.raw_match_count
                ));
                }
            }
            let raw_match_count = usize::try_from(record.raw_match_count).map_err(|_| {
                format!(
                    "snapshot pair {pair_number} raw match count {} does not fit usize",
                    record.raw_match_count
                )
            })?;
            if raw_match_count != record.raw_matches.len() {
                return Err(format!(
                "snapshot pair {pair_number} raw match count {} does not match stream length {}",
                raw_match_count,
                record.raw_matches.len()
            ));
            }
            let raw_matches = record
            .raw_matches
            .iter()
            .map(|&(left, right)| {
                let left = usize::try_from(left).map_err(|_| {
                    format!("snapshot pair {pair_number} raw query index does not fit usize")
                })?;
                let right = usize::try_from(right).map_err(|_| {
                    format!("snapshot pair {pair_number} raw train index does not fit usize")
                })?;
                if left >= feature_counts[image_i] || right >= feature_counts[image_j] {
                    return Err(format!(
                        "snapshot pair {pair_number} raw match ({left},{right}) is outside feature counts ({},{})",
                        feature_counts[image_i], feature_counts[image_j]
                    ));
                }
                Ok((left, right))
            })
            .collect::<Result<Vec<_>, String>>()?;
            validate_indices(
                "accepted inlier",
                &record.accepted_inlier_indices,
                raw_match_count,
            )?;
            validate_indices(
                "essential inlier",
                &record.essential_inlier_indices,
                raw_match_count,
            )?;
            if record.accepted_inlier_indices.len() != matches.len() {
                return Err(format!(
                    "snapshot pair {pair_number} has {} accepted indices but {} accepted matches",
                    record.accepted_inlier_indices.len(),
                    matches.len()
                ));
            }
            for (position, &raw_index) in record.accepted_inlier_indices.iter().enumerate() {
                let raw_index = usize::try_from(raw_index).map_err(|_| {
                    format!("snapshot pair {pair_number} accepted index does not fit usize")
                })?;
                if raw_matches[raw_index] != matches[position] {
                    return Err(format!(
                    "snapshot pair {pair_number} accepted match at position {position} disagrees with raw index {raw_index}"
                ));
                }
            }
            if let Some(essential_matches) = &essential_matches {
                if record.essential_inlier_indices.len() != essential_matches.len() {
                    return Err(format!(
                    "snapshot pair {pair_number} has {} essential indices but {} essential matches",
                    record.essential_inlier_indices.len(),
                    essential_matches.len()
                ));
                }
                for (position, &raw_index) in record.essential_inlier_indices.iter().enumerate() {
                    let raw_index = usize::try_from(raw_index).map_err(|_| {
                        format!("snapshot pair {pair_number} essential index does not fit usize")
                    })?;
                    if raw_matches[raw_index] != essential_matches[position] {
                        return Err(format!(
                        "snapshot pair {pair_number} essential match at position {position} disagrees with raw index {raw_index}"
                    ));
                    }
                }
            } else if !record.essential_inlier_indices.is_empty() {
                return Err(format!(
                    "snapshot pair {pair_number} has essential indices but no essential matches"
                ));
            }
        }
        pairs.push(PairwiseMatches {
            image_i,
            image_j,
            matches,
            two_view_config: config,
            essential_matches,
            essential_matrix: matrix_from_bits(record.essential_matrix_bits),
        });
    }
    Ok(pairs)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SnapshotFeatureFileFingerprint {
    keypoint_hash: u64,
    descriptor_hash: u64,
    keypoint_count: usize,
    descriptor_count: usize,
}

fn snapshot_feature_file_fingerprint(feature_set: &FeatureSet) -> SnapshotFeatureFileFingerprint {
    let mut keypoint_hash = 0xcbf29ce484222325u64;
    keypoint_hash = physical_hash_mix(keypoint_hash, feature_set.keypoints.len() as u64);
    for keypoint in &feature_set.keypoints {
        keypoint_hash = physical_hash_mix(keypoint_hash, keypoint.x.to_bits());
        keypoint_hash = physical_hash_mix(keypoint_hash, keypoint.y.to_bits());
    }
    let mut descriptor_hash = 0xcbf29ce484222325u64;
    descriptor_hash = physical_hash_mix(descriptor_hash, feature_set.descriptors.len() as u64);
    for descriptor in &feature_set.descriptors {
        descriptor_hash = physical_hash_mix(descriptor_hash, descriptor.len() as u64);
        for value in descriptor {
            descriptor_hash = physical_hash_mix(descriptor_hash, u64::from(value.to_bits()));
        }
    }
    SnapshotFeatureFileFingerprint {
        keypoint_hash,
        descriptor_hash,
        keypoint_count: feature_set.keypoints.len(),
        descriptor_count: feature_set.descriptors.len(),
    }
}

/// Read every `*<feature_suffix>` file in `dir`, sorted lexically, returning the
/// per-image feature sets and their COLMAP image names.
pub(super) fn load_images(
    dir: &Path,
    feature_suffix: &str,
    image_suffix: &str,
) -> Result<
    (
        Vec<FeatureSet>,
        Vec<String>,
        Vec<Option<Vec<FeatureLocusMetadata>>>,
    ),
    Box<dyn std::error::Error>,
> {
    let files = list_feature_files(dir, feature_suffix)?;
    let mut features = Vec::new();
    let mut names = Vec::new();
    let mut locus_metadata = Vec::new();
    for f in &files {
        let feature_path = dir.join(f);
        let (feature_set, metadata) = if let Some(parsed) =
            read_six_column_locus_features(&feature_path)?
        {
            parsed
        } else {
            let feature_set = read_external_deep_features_txt(&feature_path)?.into_feature_set()?;
            let stem = f.strip_suffix(feature_suffix).unwrap_or(f);
            let metadata_path = dir.join(format!("{stem}_loci.txt"));
            let metadata = read_locus_sidecar(&metadata_path, feature_set.len())?;
            (feature_set, metadata.unwrap_or_default())
        };
        features.push(feature_set);
        locus_metadata.push((!metadata.is_empty()).then_some(metadata));
        names.push(image_name_for(f, feature_suffix, image_suffix));
    }
    Ok((features, names, locus_metadata))
}

/// The memory-bounded feature representation used by explicit snapshot
/// replay.  `paths` preserves the exact lexical source order so the original
/// descriptor-bound feature manifest can be recomputed after calibration.
#[derive(Debug)]
pub(super) struct SnapshotKeypointsOnlyLoad {
    pub(super) features: Vec<FeatureSet>,
    pub(super) image_names: Vec<String>,
    pub(super) locus_metadata: Vec<Option<Vec<FeatureLocusMetadata>>>,
    pub(super) paths: Vec<PathBuf>,
    pub(super) fingerprints: Vec<SnapshotFeatureFileFingerprint>,
}

/// Load file-backed features one image at a time while retaining only pixels,
/// locus metadata, and one empty descriptor row per keypoint.  Keeping the
/// outer descriptor row count is intentional: downstream row-index validation
/// treats it as part of the feature shape, while ordinary incremental mapping
/// never reads descriptor values after an imported snapshot.
pub(super) fn load_images_keypoints_only(
    dir: &Path,
    feature_suffix: &str,
    image_suffix: &str,
) -> Result<SnapshotKeypointsOnlyLoad, Box<dyn std::error::Error>> {
    let files = list_feature_files(dir, feature_suffix)?;
    let mut features = Vec::with_capacity(files.len());
    let mut image_names = Vec::with_capacity(files.len());
    let mut locus_metadata = Vec::with_capacity(files.len());
    let mut paths = Vec::with_capacity(files.len());
    let mut fingerprints = Vec::with_capacity(files.len());
    for file_name in files {
        let feature_path = dir.join(&file_name);
        let (feature_set, metadata) = if let Some(parsed) =
            read_six_column_locus_features(&feature_path)?
        {
            parsed
        } else {
            let feature_set = read_external_deep_features_txt(&feature_path)?.into_feature_set()?;
            let stem = file_name.strip_suffix(feature_suffix).unwrap_or(&file_name);
            let metadata_path = dir.join(format!("{stem}_loci.txt"));
            let metadata = read_locus_sidecar(&metadata_path, feature_set.len())?;
            (feature_set, metadata.unwrap_or_default())
        };
        let fingerprint = snapshot_feature_file_fingerprint(&feature_set);
        let FeatureSet {
            keypoints,
            descriptors,
        } = feature_set;
        let descriptor_rows = descriptors.len();
        if descriptor_rows != keypoints.len() {
            return Err(format!(
                "{}: parser returned {} descriptors for {} keypoints",
                feature_path.display(),
                descriptor_rows,
                keypoints.len()
            )
            .into());
        }
        // Moving `keypoints` out and replacing the descriptor rows drops the
        // parsed payload before the next loop iteration.
        let row_count = keypoints.len();
        features.push(FeatureSet {
            keypoints,
            descriptors: (0..row_count).map(|_| Vec::new()).collect(),
        });
        locus_metadata.push((!metadata.is_empty()).then_some(metadata));
        image_names.push(image_name_for(&file_name, feature_suffix, image_suffix));
        paths.push(feature_path);
        fingerprints.push(fingerprint);
    }
    Ok(SnapshotKeypointsOnlyLoad {
        features,
        image_names,
        locus_metadata,
        paths,
        fingerprints,
    })
}

/// Replace only keypoint coordinates after a verified-pair snapshot has been
/// validated against the base feature directory.
///
/// The snapshot's correspondence indices refer to rows, so this diagnostic
/// path must never silently reorder rows or accept a descriptor mismatch.  A
/// bitwise descriptor comparison (rather than an approximate float comparison)
/// makes that contract explicit, including signed zero and NaN payloads.  The
/// caller loads the replacement directory with [`load_images`], which applies
/// the same lexical image ordering and feature parser as the base directory.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct SnapshotCoordinateOverrideStats {
    pub(super) images: usize,
    pub(super) rows: usize,
    pub(super) changed_rows: usize,
}

pub(super) fn apply_snapshot_coordinate_override(
    base_features: &mut [FeatureSet],
    base_image_names: &[String],
    override_features: &[FeatureSet],
    override_image_names: &[String],
) -> Result<SnapshotCoordinateOverrideStats, String> {
    if base_features.len() != base_image_names.len() {
        return Err(format!(
            "base feature/image manifest mismatch: {} feature sets vs {} names",
            base_features.len(),
            base_image_names.len()
        ));
    }
    if override_features.len() != override_image_names.len() {
        return Err(format!(
            "coordinate override feature/image manifest mismatch: {} feature sets vs {} names",
            override_features.len(),
            override_image_names.len()
        ));
    }
    if base_image_names != override_image_names {
        let first_difference = base_image_names
            .iter()
            .zip(override_image_names)
            .position(|(base, replacement)| base != replacement)
            .unwrap_or(base_image_names.len().min(override_image_names.len()));
        return Err(format!(
            "coordinate override image names/order do not match at row {first_difference}: base={:?}, override={:?}",
            base_image_names.get(first_difference),
            override_image_names.get(first_difference),
        ));
    }
    let mut stats = SnapshotCoordinateOverrideStats {
        images: base_features.len(),
        ..SnapshotCoordinateOverrideStats::default()
    };
    for (image_index, (base, replacement)) in
        base_features.iter_mut().zip(override_features).enumerate()
    {
        if base.keypoints.len() != replacement.keypoints.len()
            || base.descriptors.len() != replacement.descriptors.len()
        {
            return Err(format!(
                "coordinate override row count mismatch for {}: base keypoints/descriptors={}/{}, override={}/{}",
                base_image_names[image_index],
                base.keypoints.len(),
                base.descriptors.len(),
                replacement.keypoints.len(),
                replacement.descriptors.len(),
            ));
        }
        for (row, ((base_descriptor, replacement_descriptor), replacement_keypoint)) in base
            .descriptors
            .iter()
            .zip(&replacement.descriptors)
            .zip(&replacement.keypoints)
            .enumerate()
        {
            if base_descriptor.len() != replacement_descriptor.len()
                || base_descriptor.iter().zip(replacement_descriptor).any(
                    |(base_value, replacement_value)| {
                        base_value.to_bits() != replacement_value.to_bits()
                    },
                )
            {
                return Err(format!(
                    "coordinate override descriptor/index mismatch at {} row {}",
                    base_image_names[image_index], row
                ));
            }
            if !replacement_keypoint.x.is_finite() || !replacement_keypoint.y.is_finite() {
                return Err(format!(
                    "coordinate override has non-finite keypoint at {} row {}",
                    base_image_names[image_index], row
                ));
            }
            if base.keypoints[row] != *replacement_keypoint {
                stats.changed_rows += 1;
            }
            stats.rows += 1;
        }
        // The descriptor vectors are deliberately left untouched.  Replacing
        // the keypoint vector only is what keeps every snapshot feature index
        // and descriptor byte identical.
        base.keypoints.clone_from(&replacement.keypoints);
    }
    Ok(stats)
}
