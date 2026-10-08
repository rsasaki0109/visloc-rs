//! Essential-pair quality, model cross-validation and fundamental-to-essential promotion.

use super::*;

/// Deterministic, GT-independent diagnostics for an essential-matrix report.
///
/// The verifier already exposes the four-hypothesis cheirality scores through
/// `recover_relative_pose_with_options`.  The remaining values are computed
/// from a bounded, deterministic prefix of the E inliers so a full courtyard
/// graph can be audited without making the normal mapper pay for another
/// triangulation pass.  `depth_ratio` is `min(z1,z2)/max(z1,z2)` for positive
/// depths; it is a scale-free conditioning proxy, not a metric depth claim.
#[derive(Debug, Clone, Copy)]
pub(super) struct EssentialPairQuality {
    pub(super) best_cheirality: i64,
    pub(super) second_cheirality: i64,
    pub(super) cheirality_ratio: f64,
    pub(super) mean_sampson: f64,
    /// Winning `R` as `(w,x,y,z)` and camera-2 centre direction in camera-1
    /// coordinates.  These are emitted only as diagnostic fields so the
    /// optional GT audit can compare direct-E and F→E poses without changing
    /// the mapper data path.
    pub(super) rotation_quaternion: [f64; 4],
    pub(super) center_direction: [f64; 3],
    pub(super) angle_samples: usize,
    pub(super) angle_ge_1deg: usize,
    pub(super) angle_p10_deg: f64,
    pub(super) angle_p25_deg: f64,
    pub(super) angle_median_deg: f64,
    pub(super) depth_ratio_p10: f64,
    pub(super) depth_ratio_p25: f64,
    pub(super) depth_ratio_median: f64,
}

fn quantile_nearest_rank(values: &mut [f64], fraction: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_by(f64::total_cmp);
    let rank = (fraction.clamp(0.0, 1.0) * values.len() as f64).ceil() as usize;
    values[rank.saturating_sub(1).min(values.len() - 1)]
}

pub(super) fn essential_pair_quality(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<EssentialPairQuality> {
    essential_pair_quality_for_inliers_with_options(
        report.essential.as_ref()?,
        &report.essential_inliers,
        correspondences,
        camera,
        &CheiralityOptions::default(),
    )
}

fn essential_pair_quality_for_inliers(
    essential: &Matrix3<f64>,
    e_inliers: &[usize],
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<EssentialPairQuality> {
    essential_pair_quality_for_inliers_with_options(
        essential,
        e_inliers,
        correspondences,
        camera,
        &CheiralityOptions::default(),
    )
}

fn essential_pair_quality_for_inliers_with_options(
    essential: &Matrix3<f64>,
    e_inliers: &[usize],
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    cheirality_options: &CheiralityOptions,
) -> Option<EssentialPairQuality> {
    if e_inliers.len() < 8 {
        return None;
    }
    let recovery = recover_relative_pose_with_options(
        essential,
        correspondences,
        camera,
        e_inliers,
        cheirality_options,
    )?;
    let cheirality_ratio = if e_inliers.is_empty() {
        f64::NAN
    } else {
        recovery.best_score as f64 / e_inliers.len() as f64
    };
    let mean_sampson =
        mean_normalized_essential_sampson_error(essential, correspondences, camera, e_inliers);
    let q = recovery.rotation.quaternion();
    let center = -recovery
        .rotation
        .inverse()
        .transform_vector(&recovery.translation_unit);
    let center_direction = center
        .try_normalize(1.0e-12)
        .map_or([f64::NAN; 3], |value| [value.x, value.y, value.z]);
    let rotation_quaternion = [q.w, q.i, q.j, q.k];

    const MAX_TRIANGULATION_SAMPLES: usize = 256;
    let stride = e_inliers.len().div_ceil(MAX_TRIANGULATION_SAMPLES);
    let left_to_right = SE3::new(recovery.rotation, recovery.translation_unit);
    let camera_2_center = -recovery
        .rotation
        .inverse()
        .transform_vector(&recovery.translation_unit);
    let mut angles = Vec::new();
    let mut depth_ratios = Vec::new();
    for &inlier_index in e_inliers.iter().step_by(stride.max(1)) {
        let Some(corr) = correspondences.get(inlier_index) else {
            continue;
        };
        let Some(point) = triangulate_two_view_left_frame(
            camera,
            camera,
            &left_to_right,
            &corr.previous_xy,
            &corr.current_xy,
        ) else {
            continue;
        };
        let point_2 = left_to_right.transform_point(&point);
        if !point.coords.iter().all(|v| v.is_finite())
            || !point_2.coords.iter().all(|v| v.is_finite())
            || point.z <= 0.0
            || point_2.z <= 0.0
        {
            continue;
        }
        let Some(ray_1) = point.coords.try_normalize(1.0e-12) else {
            continue;
        };
        let Some(ray_2) = (point.coords - camera_2_center).try_normalize(1.0e-12) else {
            continue;
        };
        let angle = ray_1.dot(&ray_2).clamp(-1.0, 1.0).acos().to_degrees();
        if !angle.is_finite() {
            continue;
        }
        angles.push(angle);
        let z1 = point.z;
        let z2 = point_2.z;
        let depth_ratio = z1.min(z2) / z1.max(z2);
        if depth_ratio.is_finite() {
            depth_ratios.push(depth_ratio);
        }
    }
    if angles.is_empty() || depth_ratios.is_empty() {
        return Some(EssentialPairQuality {
            best_cheirality: recovery.best_score,
            second_cheirality: recovery.second_score,
            cheirality_ratio,
            mean_sampson,
            rotation_quaternion,
            center_direction,
            angle_samples: 0,
            angle_ge_1deg: 0,
            angle_p10_deg: f64::NAN,
            angle_p25_deg: f64::NAN,
            angle_median_deg: f64::NAN,
            depth_ratio_p10: f64::NAN,
            depth_ratio_p25: f64::NAN,
            depth_ratio_median: f64::NAN,
        });
    }
    let angle_ge_1deg = angles.iter().filter(|&&angle| angle >= 1.0).count();
    let angle_p10_deg = quantile_nearest_rank(&mut angles, 0.10);
    let angle_p25_deg = quantile_nearest_rank(&mut angles, 0.25);
    let angle_median_deg = quantile_nearest_rank(&mut angles, 0.50);
    let depth_ratio_p10 = quantile_nearest_rank(&mut depth_ratios, 0.10);
    let depth_ratio_p25 = quantile_nearest_rank(&mut depth_ratios, 0.25);
    let depth_ratio_median = quantile_nearest_rank(&mut depth_ratios, 0.50);
    Some(EssentialPairQuality {
        best_cheirality: recovery.best_score,
        second_cheirality: recovery.second_score,
        cheirality_ratio,
        mean_sampson,
        rotation_quaternion,
        center_direction,
        angle_samples: angles.len(),
        angle_ge_1deg,
        angle_p10_deg,
        angle_p25_deg,
        angle_median_deg,
        depth_ratio_p10,
        depth_ratio_p25,
        depth_ratio_median,
    })
}

/// Normalized-coordinate Sampson distance for an essential matrix.  This is
/// intentionally kept local to the read-only diagnostic: the production
/// verifier already uses the same expression in `two_view::sampson_distance`
/// and its threshold units are unchanged here.
pub(super) fn normalized_essential_squared_sampson_error(
    essential: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
    camera: &Camera,
) -> Option<f64> {
    let previous = camera.normalize_pixel(&correspondence.previous_xy)?;
    let current = camera.normalize_pixel(&correspondence.current_xy)?;
    let previous_h = Vector3::new(previous.x, previous.y, 1.0);
    let current_h = Vector3::new(current.x, current.y, 1.0);
    let e_previous = essential * previous_h;
    let et_current = essential.transpose() * current_h;
    let numerator = current_h.dot(&e_previous).powi(2);
    let denominator =
        e_previous.x.powi(2) + e_previous.y.powi(2) + et_current.x.powi(2) + et_current.y.powi(2);
    if denominator < 1.0e-18 {
        return None;
    }
    let error = numerator / denominator;
    error.is_finite().then_some(error)
}

fn mean_normalized_essential_sampson_error(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    indices: &[usize],
) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for &index in indices {
        let Some(correspondence) = correspondences.get(index) else {
            continue;
        };
        let Some(error_sq) =
            normalized_essential_squared_sampson_error(essential, correspondence, camera)
        else {
            continue;
        };
        total += error_sq.sqrt();
        count += 1;
    }
    if count == 0 {
        f64::NAN
    } else {
        total / count as f64
    }
}

/// Deterministic held-out partition for the completed-model cross-validation
/// probe.  The hash depends only on the imported pair and feature indices, so
/// it is independent of mapper traversal, track conflicts, and the candidate
/// order used to produce the model.
#[cfg(test)]
pub(super) fn model_cross_validation_is_held_out(
    image_i: usize,
    image_j: usize,
    keypoint_i: usize,
    keypoint_j: usize,
) -> bool {
    let mut hash = 0xcbf29ce484222325u64;
    for value in [image_i, image_j, keypoint_i, keypoint_j] {
        for byte in (value as u64).to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3u64);
        }
    }
    hash % 5 == 0
}

/// Order-independent form of the held-out hash.  Feature permutations preserve
/// the physical endpoint coordinates, while keypoint indices need not remain
/// stable; quantizing at 1/1000 pixel keeps decimal feature-file round trips
/// in the same partition without making co-located duplicate rows diverge.
pub(super) fn model_cross_validation_is_held_out_for_pixels(
    image_i: usize,
    image_j: usize,
    pixel_i: &Point2<f64>,
    pixel_j: &Point2<f64>,
) -> bool {
    let quantize = |value: f64| {
        if value.is_finite() {
            (value * 1_000.0).round() as i64 as u64
        } else {
            u64::MAX
        }
    };
    let mut hash = 0xcbf29ce484222325u64;
    for value in [
        image_i as u64,
        image_j as u64,
        quantize(pixel_i.x),
        quantize(pixel_i.y),
        quantize(pixel_j.x),
        quantize(pixel_j.y),
    ] {
        for byte in value.to_le_bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100000001b3u64);
        }
    }
    hash.is_multiple_of(5)
}

#[derive(Debug, Default)]
pub(super) struct ModelCrossValidationBucket {
    observations: usize,
    residuals: Vec<f64>,
    under_threshold: usize,
    triangulated: usize,
    positive_depth: usize,
    angle_ge_one_degree: usize,
}

impl ModelCrossValidationBucket {
    pub(super) fn record(
        &mut self,
        residual: Option<f64>,
        threshold: f64,
        triangulated: bool,
        positive_depth: bool,
        angle_ge_one_degree: bool,
    ) {
        self.observations += 1;
        if let Some(residual) = residual.filter(|value| value.is_finite()) {
            self.residuals.push(residual);
            if residual <= threshold {
                self.under_threshold += 1;
            }
        }
        if triangulated {
            self.triangulated += 1;
        }
        if positive_depth {
            self.positive_depth += 1;
        }
        if angle_ge_one_degree {
            self.angle_ge_one_degree += 1;
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(super) struct ModelCrossValidationBucketSummary {
    pub(super) observations: usize,
    pub(super) residual_samples: usize,
    pub(super) under_threshold: usize,
    pub(super) triangulated: usize,
    pub(super) positive_depth: usize,
    pub(super) angle_ge_one_degree: usize,
    mean_sampson_root: f64,
    median_sampson_root: f64,
    p90_sampson_root: f64,
    pub(super) under_fraction: f64,
    pub(super) positive_fraction: f64,
    pub(super) angle_fraction: f64,
}

fn fraction_or_nan(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        f64::NAN
    } else {
        numerator as f64 / denominator as f64
    }
}

pub(super) fn summarize_model_cross_validation_bucket(
    bucket: &mut ModelCrossValidationBucket,
) -> ModelCrossValidationBucketSummary {
    bucket.residuals.sort_by(f64::total_cmp);
    let residual_samples = bucket.residuals.len();
    let mean_sampson_root = if residual_samples == 0 {
        f64::NAN
    } else {
        bucket.residuals.iter().sum::<f64>() / residual_samples as f64
    };
    let quantile = |fraction: f64| {
        if residual_samples == 0 {
            f64::NAN
        } else {
            let index = ((residual_samples as f64 * fraction).ceil() as usize)
                .saturating_sub(1)
                .min(residual_samples - 1);
            bucket.residuals[index]
        }
    };
    ModelCrossValidationBucketSummary {
        observations: bucket.observations,
        residual_samples,
        under_threshold: bucket.under_threshold,
        triangulated: bucket.triangulated,
        positive_depth: bucket.positive_depth,
        angle_ge_one_degree: bucket.angle_ge_one_degree,
        mean_sampson_root,
        median_sampson_root: quantile(0.5),
        p90_sampson_root: quantile(0.9),
        under_fraction: fraction_or_nan(bucket.under_threshold, residual_samples),
        positive_fraction: fraction_or_nan(bucket.positive_depth, bucket.triangulated),
        angle_fraction: fraction_or_nan(bucket.angle_ge_one_degree, bucket.triangulated),
    }
}

/// Huber location estimate for a small set of already pair-balanced angular
/// errors.  The scale comes from the median absolute deviation and the
/// conventional 1.345 Huber tuning constant; no scene/GT-specific threshold
/// is used.  A zero-MAD set is already robustly constant, so its median is
/// returned directly.
pub(super) fn robust_huber_mean(values: &[f64]) -> f64 {
    let mut finite = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.sort_by(f64::total_cmp);
    let median = finite[finite.len() / 2];
    let mut deviations = finite
        .iter()
        .map(|value| (value - median).abs())
        .collect::<Vec<_>>();
    deviations.sort_by(f64::total_cmp);
    let mad = deviations[deviations.len() / 2];
    if mad <= 1.0e-12 {
        return median;
    }
    let scale = 1.4826 * mad;
    let delta = 1.345 * scale;
    let mut location = median;
    for _ in 0..8 {
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for value in &finite {
            let residual = *value - location;
            let weight = if residual.abs() <= delta {
                1.0
            } else {
                delta / residual.abs()
            };
            weighted_sum += weight * *value;
            weight_sum += weight;
        }
        if !(weight_sum.is_finite() && weight_sum > 0.0) {
            return f64::NAN;
        }
        let next = weighted_sum / weight_sum;
        if !next.is_finite() {
            return f64::NAN;
        }
        if (next - location).abs() <= 1.0e-12 {
            location = next;
            break;
        }
        location = next;
    }
    location
}

#[derive(Debug, Clone)]
struct ModelCrossValidationPairScore {
    image_i: usize,
    image_j: usize,
    config: ConfigurationType,
    verified_inliers: usize,
    /// Rotation disagreement to the imported pair E, when the file carries
    /// one. This is a diagnostic reference for the rotation-only alternative;
    /// it is not used by the mapper or the calibrated residual score.
    rotation_disagreement_deg: f64,
    /// Rotation disagreement on the subset of imported-E references that
    /// passed the strong cheirality/parallax/stability gate below.
    stable_rotation_disagreement_deg: f64,
    /// Signed (cheirality-selected) camera-centre direction disagreement to
    /// the imported calibrated-E reference, in degrees.
    translation_disagreement_deg: f64,
    /// Quality of the imported-E reference used for the two direction fields.
    reference_cheirality_margin: f64,
    reference_angle_p25_deg: f64,
    reference_stable_refits: usize,
    reference_rotation_spread_deg: f64,
    reference_translation_spread_deg: f64,
    all: ModelCrossValidationBucketSummary,
    held_out: ModelCrossValidationBucketSummary,
}

#[derive(Debug, Clone, Copy)]
struct ImportedEssentialReferenceQuality {
    rotation: UnitQuaternion<f64>,
    center_direction: Vector3<f64>,
    cheirality_margin: f64,
    angle_p25_deg: f64,
    stable_refits: usize,
    rotation_spread_deg: f64,
    translation_spread_deg: f64,
}

#[derive(Debug, Clone, Default)]
pub(super) struct ModelCrossValidationSummary {
    imported_pairs: usize,
    registered_images: usize,
    registered_pairs: usize,
    invalid_correspondences: usize,
    normalized_threshold: f64,
    all: ModelCrossValidationBucketSummary,
    held_out: ModelCrossValidationBucketSummary,
    pair_balanced_mean_sampson_root: f64,
    pair_balanced_median_sampson_root: f64,
    pair_balanced_p90_sampson_root: f64,
    pair_balanced_under_fraction: f64,
    pair_balanced_positive_fraction: f64,
    pair_balanced_angle_fraction: f64,
    pub(super) pair_balanced_rotation_disagreement_deg: f64,
    pub(super) rotation_reference_pairs: usize,
    pair_balanced_stable_rotation_disagreement_deg: f64,
    pair_balanced_translation_disagreement_deg: f64,
    pair_balanced_translation_median_deg: f64,
    pair_balanced_translation_p90_deg: f64,
    pair_balanced_translation_huber_deg: f64,
    stable_rotation_reference_pairs: usize,
    translation_reference_pairs: usize,
    translation_reference_coverage: f64,
    image_balanced_under_fraction: f64,
    image_balanced_positive_fraction: f64,
    image_balanced_angle_fraction: f64,
    image_balanced_median_sampson_root: f64,
    image_balanced_translation_disagreement_deg: f64,
    image_translation_reference_coverage: f64,
    pairs: Vec<ModelCrossValidationPairScore>,
}

/// GT-independent ranking score for multi-hypothesis diagnostics.  This is
/// intentionally only available when at least three calibrated imported-E
/// pair references are present; callers must compare models scored against the
/// same verified multiset and camera.  Lower is better.  It is reported only
/// and never selects or mutates the normal reconstruction path.
pub(super) fn model_cross_validation_selection_score(
    summary: &ModelCrossValidationSummary,
) -> Option<f64> {
    (summary.rotation_reference_pairs >= 3
        && summary.pair_balanced_rotation_disagreement_deg.is_finite())
    .then_some(summary.pair_balanced_rotation_disagreement_deg)
}

#[derive(Debug, Default)]
struct ModelCrossValidationImageAccumulator {
    median_sampson_sum: f64,
    under_sum: f64,
    positive_sum: f64,
    angle_sum: f64,
    translation_sum: f64,
    translation_count: usize,
    reference_attempted: usize,
    reference_valid: usize,
    count: usize,
}

/// Score a completed pose model against the complete imported verified set.
///
/// This is intentionally a post-hoc diagnostic: it does not inspect the
/// reconstruction's retained tracks, so pair edges and observations rejected
/// by union-find conflicts still contribute.  Residuals use the pose-induced
/// calibrated essential matrix and the same normalized threshold family as
/// the full verifier.  Positive depth and a one-degree triangulation-angle
/// gate are reported separately rather than silently folded into the residual
/// score.  Pair and image summaries are balanced so a dense pair cannot
/// dominate the model comparison.
pub(super) fn score_model_against_verified_pairs(
    model_images_path: &Path,
    imported: &[ImportedVerifiedPair],
    features: &[FeatureSet],
    image_names: &[String],
    camera: &Camera,
) -> Result<ModelCrossValidationSummary, Box<dyn std::error::Error>> {
    let poses_by_stem = poses_from_colmap_images_txt(model_images_path)
        .map_err(|error| format!("model cross-validation: {error}"))?;
    let registered_images = image_names
        .iter()
        .filter(|name| poses_by_stem.contains_key(image_stem(name)))
        .count();
    let normalized_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0)
        .essential_sampson_threshold
        .sqrt();
    let mut all_bucket = ModelCrossValidationBucket::default();
    let mut held_out_bucket = ModelCrossValidationBucket::default();
    let mut pair_scores = Vec::new();
    let mut invalid_correspondences = 0usize;
    let mut image_accumulators: HashMap<usize, ModelCrossValidationImageAccumulator> =
        HashMap::new();
    // References are counted only after both candidate cameras are present;
    // this makes coverage describe the diagnostic population that could
    // actually be compared, rather than all rows in the replay file.
    let mut reference_attempted_pairs = 0usize;

    for pair in imported {
        let Some(pose_i) = image_names
            .get(pair.image_i)
            .and_then(|name| poses_by_stem.get(image_stem(name)))
        else {
            continue;
        };
        let Some(pose_j) = image_names
            .get(pair.image_j)
            .and_then(|name| poses_by_stem.get(image_stem(name)))
        else {
            continue;
        };
        let Some(essential) = essential_from_absolute_poses(pose_i, pose_j) else {
            continue;
        };
        let left_to_right = pose_j
            .world_to_camera
            .compose(&pose_i.world_to_camera.inverse());
        let camera_2_center = -left_to_right
            .rotation
            .inverse()
            .transform_vector(&left_to_right.translation);
        let mut pair_all = ModelCrossValidationBucket::default();
        let mut pair_held_out = ModelCrossValidationBucket::default();
        let mut valid_correspondences = Vec::with_capacity(pair.matches.len());

        for &(keypoint_i, keypoint_j) in &pair.matches {
            let Some(pixel_i) = features
                .get(pair.image_i)
                .and_then(|feature_set| feature_set.keypoints.get(keypoint_i))
            else {
                invalid_correspondences += 1;
                continue;
            };
            let Some(pixel_j) = features
                .get(pair.image_j)
                .and_then(|feature_set| feature_set.keypoints.get(keypoint_j))
            else {
                invalid_correspondences += 1;
                continue;
            };
            let correspondence = TwoViewCorrespondence::new(*pixel_i, *pixel_j);
            valid_correspondences.push(correspondence);
            let residual =
                normalized_essential_squared_sampson_error(&essential, &correspondence, camera)
                    .map(f64::sqrt);
            let triangulated_point =
                triangulate_two_view_left_frame(camera, camera, &left_to_right, pixel_i, pixel_j);
            let triangulated = triangulated_point.as_ref().is_some_and(|point| {
                point.coords.iter().all(|value| value.is_finite())
                    && left_to_right
                        .transform_point(point)
                        .coords
                        .iter()
                        .all(|value| value.is_finite())
            });
            let positive_depth = triangulated
                && triangulated_point.as_ref().is_some_and(|point| {
                    let point_j = left_to_right.transform_point(point);
                    point.z > 0.0 && point_j.z > 0.0
                });
            let angle_ge_one_degree = triangulated_point.as_ref().is_some_and(|point| {
                if !positive_depth {
                    return false;
                }
                let Some(ray_i) = point.coords.try_normalize(1.0e-12) else {
                    return false;
                };
                let Some(ray_j) = (point.coords - camera_2_center).try_normalize(1.0e-12) else {
                    return false;
                };
                let angle = ray_i.dot(&ray_j).clamp(-1.0, 1.0).acos().to_degrees();
                angle.is_finite() && angle >= 1.0
            });
            pair_all.record(
                residual,
                normalized_threshold,
                triangulated,
                positive_depth,
                angle_ge_one_degree,
            );
            all_bucket.record(
                residual,
                normalized_threshold,
                triangulated,
                positive_depth,
                angle_ge_one_degree,
            );
            if model_cross_validation_is_held_out_for_pixels(
                pair.image_i,
                pair.image_j,
                pixel_i,
                pixel_j,
            ) {
                pair_held_out.record(
                    residual,
                    normalized_threshold,
                    triangulated,
                    positive_depth,
                    angle_ge_one_degree,
                );
                held_out_bucket.record(
                    residual,
                    normalized_threshold,
                    triangulated,
                    positive_depth,
                    angle_ge_one_degree,
                );
            }
        }

        let all = summarize_model_cross_validation_bucket(&mut pair_all);
        let held_out = summarize_model_cross_validation_bucket(&mut pair_held_out);
        // The imported E is a calibrated reference only for configurations in
        // which the verifier actually selected a calibrated model.  F-winning
        // and planar rows may carry an auxiliary E diagnostic, but treating it
        // as a pose reference would mix incomparable model hypotheses.
        let has_calibrated_reference = matches!(
            pair.config,
            ConfigurationType::Calibrated | ConfigurationType::Multiple
        ) && pair.essential_matrix.is_some();
        if has_calibrated_reference {
            reference_attempted_pairs += 1;
        }
        let reference_quality = if has_calibrated_reference {
            pair.essential_matrix
                .as_ref()
                .and_then(|imported_essential| {
                    imported_essential_reference_quality(
                        imported_essential,
                        &valid_correspondences,
                        camera,
                    )
                })
        } else {
            None
        };
        let rotation_disagreement_deg = if matches!(
            pair.config,
            ConfigurationType::Calibrated | ConfigurationType::Multiple
        ) {
            pair.essential_matrix
                .as_ref()
                .and_then(|imported_essential| {
                    relative_pose_from_essential(imported_essential, &valid_correspondences, camera)
                })
                .map(|imported_pose| {
                    (left_to_right.rotation.inverse() * imported_pose.previous_to_current.rotation)
                        .angle()
                        .to_degrees()
                })
                .filter(|value| value.is_finite())
                .unwrap_or(f64::NAN)
        } else {
            f64::NAN
        };
        let stable_rotation_disagreement_deg = reference_quality
            .as_ref()
            .map(|reference| {
                (left_to_right.rotation.inverse() * reference.rotation)
                    .angle()
                    .to_degrees()
            })
            .filter(|value| value.is_finite())
            .unwrap_or(f64::NAN);
        let candidate_center_direction = camera_2_center.try_normalize(1.0e-12);
        let translation_disagreement_deg = reference_quality
            .as_ref()
            .zip(candidate_center_direction.as_ref())
            .map(|(reference, candidate)| {
                translation_direction_delta_deg(candidate, &reference.center_direction)
            })
            .filter(|value| value.is_finite())
            .unwrap_or(f64::NAN);
        for image_index in [pair.image_i, pair.image_j] {
            let accumulator = image_accumulators.entry(image_index).or_default();
            if has_calibrated_reference {
                accumulator.reference_attempted += 1;
            }
            if reference_quality.is_some() {
                accumulator.reference_valid += 1;
            }
            if translation_disagreement_deg.is_finite() {
                accumulator.translation_sum += translation_disagreement_deg;
                accumulator.translation_count += 1;
            }
            if all.residual_samples > 0 {
                accumulator.median_sampson_sum += all.median_sampson_root;
                accumulator.under_sum += all.under_fraction;
                accumulator.positive_sum += all.positive_fraction;
                accumulator.angle_sum += all.angle_fraction;
                accumulator.count += 1;
            }
        }
        pair_scores.push(ModelCrossValidationPairScore {
            image_i: pair.image_i,
            image_j: pair.image_j,
            config: pair.config,
            verified_inliers: pair.matches.len(),
            rotation_disagreement_deg,
            stable_rotation_disagreement_deg,
            translation_disagreement_deg,
            reference_cheirality_margin: reference_quality
                .as_ref()
                .map_or(f64::NAN, |reference| reference.cheirality_margin),
            reference_angle_p25_deg: reference_quality
                .as_ref()
                .map_or(f64::NAN, |reference| reference.angle_p25_deg),
            reference_stable_refits: reference_quality
                .as_ref()
                .map_or(0, |reference| reference.stable_refits),
            reference_rotation_spread_deg: reference_quality
                .as_ref()
                .map_or(f64::NAN, |reference| reference.rotation_spread_deg),
            reference_translation_spread_deg: reference_quality
                .as_ref()
                .map_or(f64::NAN, |reference| reference.translation_spread_deg),
            all,
            held_out,
        });
    }

    let all = summarize_model_cross_validation_bucket(&mut all_bucket);
    let held_out = summarize_model_cross_validation_bucket(&mut held_out_bucket);
    let finite_pair_values = |select: fn(&ModelCrossValidationBucketSummary) -> f64| {
        let mut values: Vec<f64> = pair_scores
            .iter()
            .map(|pair| select(&pair.all))
            .filter(|value| value.is_finite())
            .collect();
        if values.is_empty() {
            return (f64::NAN, f64::NAN, f64::NAN);
        }
        values.sort_by(f64::total_cmp);
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let median = values[values.len() / 2];
        let p90 = values[((values.len() * 9).saturating_sub(1) / 10).min(values.len() - 1)];
        (mean, median, p90)
    };
    let (
        pair_balanced_mean_sampson_root,
        pair_balanced_median_sampson_root,
        pair_balanced_p90_sampson_root,
    ) = finite_pair_values(|summary| summary.mean_sampson_root);
    let (pair_balanced_under_fraction, _, _) = finite_pair_values(|summary| summary.under_fraction);
    let (pair_balanced_positive_fraction, _, _) =
        finite_pair_values(|summary| summary.positive_fraction);
    let (pair_balanced_angle_fraction, _, _) = finite_pair_values(|summary| summary.angle_fraction);
    let (rotation_sum, rotation_count) = pair_scores
        .iter()
        .map(|pair| pair.rotation_disagreement_deg)
        .filter(|value| value.is_finite())
        .fold((0.0, 0usize), |(sum, count), value| {
            (sum + value, count + 1)
        });
    let pair_balanced_rotation_disagreement_deg = if rotation_count == 0 {
        f64::NAN
    } else {
        rotation_sum / rotation_count as f64
    };
    let stable_rotation_values = pair_scores
        .iter()
        .map(|pair| pair.stable_rotation_disagreement_deg)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let translation_values = pair_scores
        .iter()
        .map(|pair| pair.translation_disagreement_deg)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    let pair_balanced_stable_rotation_disagreement_deg = robust_huber_mean(&stable_rotation_values);
    let pair_balanced_translation_disagreement_deg = if translation_values.is_empty() {
        f64::NAN
    } else {
        translation_values.iter().sum::<f64>() / translation_values.len() as f64
    };
    let pair_balanced_translation_median_deg = if translation_values.is_empty() {
        f64::NAN
    } else {
        let mut sorted = translation_values.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[sorted.len() / 2]
    };
    let pair_balanced_translation_p90_deg = if translation_values.is_empty() {
        f64::NAN
    } else {
        let mut sorted = translation_values.clone();
        sorted.sort_by(f64::total_cmp);
        sorted[((sorted.len() * 9).saturating_sub(1) / 10).min(sorted.len() - 1)]
    };
    let pair_balanced_translation_huber_deg = robust_huber_mean(&translation_values);
    let mut image_under = Vec::new();
    let mut image_positive = Vec::new();
    let mut image_angle = Vec::new();
    let mut image_median_sampson = Vec::new();
    let mut image_translation = Vec::new();
    let mut image_translation_coverage = Vec::new();
    for accumulator in image_accumulators.values() {
        if accumulator.count > 0 {
            let count = accumulator.count as f64;
            image_median_sampson.push(accumulator.median_sampson_sum / count);
            image_under.push(accumulator.under_sum / count);
            image_positive.push(accumulator.positive_sum / count);
            image_angle.push(accumulator.angle_sum / count);
        }
        if accumulator.translation_count > 0 {
            image_translation
                .push(accumulator.translation_sum / accumulator.translation_count as f64);
        }
        if accumulator.reference_attempted > 0 {
            image_translation_coverage
                .push(accumulator.reference_valid as f64 / accumulator.reference_attempted as f64);
        }
    }
    let mean = |values: &[f64]| {
        if values.is_empty() {
            f64::NAN
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    };
    Ok(ModelCrossValidationSummary {
        imported_pairs: imported.len(),
        registered_images,
        registered_pairs: pair_scores.len(),
        invalid_correspondences,
        normalized_threshold,
        all,
        held_out,
        pair_balanced_mean_sampson_root,
        pair_balanced_median_sampson_root,
        pair_balanced_p90_sampson_root,
        pair_balanced_under_fraction,
        pair_balanced_positive_fraction,
        pair_balanced_angle_fraction,
        pair_balanced_rotation_disagreement_deg,
        rotation_reference_pairs: rotation_count,
        pair_balanced_stable_rotation_disagreement_deg,
        pair_balanced_translation_disagreement_deg,
        pair_balanced_translation_median_deg,
        pair_balanced_translation_p90_deg,
        pair_balanced_translation_huber_deg,
        stable_rotation_reference_pairs: stable_rotation_values.len(),
        translation_reference_pairs: translation_values.len(),
        translation_reference_coverage: if reference_attempted_pairs == 0 {
            f64::NAN
        } else {
            translation_values.len() as f64 / reference_attempted_pairs as f64
        },
        image_balanced_under_fraction: mean(&image_under),
        image_balanced_positive_fraction: mean(&image_positive),
        image_balanced_angle_fraction: mean(&image_angle),
        image_balanced_median_sampson_root: mean(&image_median_sampson),
        image_balanced_translation_disagreement_deg: robust_huber_mean(&image_translation),
        image_translation_reference_coverage: mean(&image_translation_coverage),
        pairs: pair_scores,
    })
}

pub(super) fn print_model_cross_validation_summary(
    summary: &ModelCrossValidationSummary,
    model_images_path: &Path,
    verified_pairs_path: &Path,
    image_names: &[String],
) {
    for pair in &summary.pairs {
        println!(
            "model-xval-pair: i={} j={} image_i_name={} image_j_name={} config={:?} verified={} rotation_ref_deg={:.6} stable_rotation_ref_deg={:.6} translation_ref_deg={:.6} ref_cheirality_margin={:.6} ref_angle_p25_deg={:.6} ref_stable_refits={} ref_rotation_spread_deg={:.6} ref_translation_spread_deg={:.6} observations={} residual_n={} mean={:.9e} median={:.9e} p90={:.9e} under={:.6} under_n={} triangulated={} positive={:.6} positive_n={} angle_ge_1deg={:.6} angle_n={} heldout_n={} heldout_under={:.6} heldout_positive={:.6} heldout_angle_ge_1deg={:.6}",
            pair.image_i,
            pair.image_j,
            image_names
                .get(pair.image_i)
                .map_or("<unknown>", String::as_str),
            image_names
                .get(pair.image_j)
                .map_or("<unknown>", String::as_str),
            pair.config,
            pair.verified_inliers,
            pair.rotation_disagreement_deg,
            pair.stable_rotation_disagreement_deg,
            pair.translation_disagreement_deg,
            pair.reference_cheirality_margin,
            pair.reference_angle_p25_deg,
            pair.reference_stable_refits,
            pair.reference_rotation_spread_deg,
            pair.reference_translation_spread_deg,
            pair.all.observations,
            pair.all.residual_samples,
            pair.all.mean_sampson_root,
            pair.all.median_sampson_root,
            pair.all.p90_sampson_root,
            pair.all.under_fraction,
            pair.all.under_threshold,
            pair.all.triangulated,
            pair.all.positive_fraction,
            pair.all.positive_depth,
            pair.all.angle_fraction,
            pair.all.angle_ge_one_degree,
            pair.held_out.residual_samples,
            pair.held_out.under_fraction,
            pair.held_out.positive_fraction,
            pair.held_out.angle_fraction,
        );
    }
    println!(
        "model-xval-summary: model={} verified_file={} imported_pairs={} registered_images={} registered_pairs={} invalid_correspondences={} normalized_threshold={:.9e} all_observations={} all_residual_n={} all_mean={:.9e} all_median={:.9e} all_p90={:.9e} all_under={:.6} all_positive={:.6} all_angle_ge_1deg={:.6} pair_mean={:.9e} pair_median={:.9e} pair_p90={:.9e} pair_under={:.6} pair_positive={:.6} pair_angle_ge_1deg={:.6} pair_rotation_ref_deg={:.6} rotation_ref_pairs={} pair_stable_rotation_ref_deg={:.6} stable_rotation_ref_pairs={} pair_translation_ref_deg={:.6} pair_translation_median_deg={:.6} pair_translation_p90_deg={:.6} pair_translation_huber_deg={:.6} translation_ref_pairs={} translation_ref_coverage={:.6} selection_score_deg={:.6} image_median={:.9e} image_under={:.6} image_positive={:.6} image_angle_ge_1deg={:.6} image_translation_ref_deg={:.6} image_translation_ref_coverage={:.6} heldout_observations={} heldout_residual_n={} heldout_under={:.6} heldout_positive={:.6} heldout_angle_ge_1deg={:.6}",
        model_images_path.display(),
        verified_pairs_path.display(),
        summary.imported_pairs,
        summary.registered_images,
        summary.registered_pairs,
        summary.invalid_correspondences,
        summary.normalized_threshold,
        summary.all.observations,
        summary.all.residual_samples,
        summary.all.mean_sampson_root,
        summary.all.median_sampson_root,
        summary.all.p90_sampson_root,
        summary.all.under_fraction,
        summary.all.positive_fraction,
        summary.all.angle_fraction,
        summary.pair_balanced_mean_sampson_root,
        summary.pair_balanced_median_sampson_root,
        summary.pair_balanced_p90_sampson_root,
        summary.pair_balanced_under_fraction,
        summary.pair_balanced_positive_fraction,
        summary.pair_balanced_angle_fraction,
        summary.pair_balanced_rotation_disagreement_deg,
        summary.rotation_reference_pairs,
        summary.pair_balanced_stable_rotation_disagreement_deg,
        summary.stable_rotation_reference_pairs,
        summary.pair_balanced_translation_disagreement_deg,
        summary.pair_balanced_translation_median_deg,
        summary.pair_balanced_translation_p90_deg,
        summary.pair_balanced_translation_huber_deg,
        summary.translation_reference_pairs,
        summary.translation_reference_coverage,
        model_cross_validation_selection_score(summary).unwrap_or(f64::NAN),
        summary.image_balanced_median_sampson_root,
        summary.image_balanced_under_fraction,
        summary.image_balanced_positive_fraction,
        summary.image_balanced_angle_fraction,
        summary.image_balanced_translation_disagreement_deg,
        summary.image_translation_reference_coverage,
        summary.held_out.observations,
        summary.held_out.residual_samples,
        summary.held_out.under_fraction,
        summary.held_out.positive_fraction,
        summary.held_out.angle_fraction,
    );
}

/// Express a pixel-space fundamental matrix in normalized coordinates.  For
/// `x_jᵀ F x_i = 0` and `x_norm = K⁻¹x`, the corresponding relation is
/// `x_norm,jᵀ (K_jᵀ F K_i) x_norm,i = 0`.
fn calibrated_fundamental(fundamental: &Matrix3<f64>, camera: &Camera) -> Option<Matrix3<f64>> {
    let (fx, fy, cx, cy) = camera.intrinsics()?;
    if ![fx, fy, cx, cy].iter().all(|value| value.is_finite())
        || fx.abs() < 1.0e-12
        || fy.abs() < 1.0e-12
    {
        return None;
    }
    let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    let calibrated = k.transpose() * fundamental * k;
    calibrated
        .iter()
        .all(|value| value.is_finite())
        .then_some(calibrated)
}

/// Project a pixel-space fundamental matrix into the calibrated essential
/// manifold by equalizing its two largest singular values and zeroing the
/// third.  This is the closest essential-manifold projection in Frobenius
/// norm for the fixed singular vectors.
pub(super) fn project_fundamental_to_essential(
    fundamental: &Matrix3<f64>,
    camera: &Camera,
) -> Option<Matrix3<f64>> {
    let calibrated = calibrated_fundamental(fundamental, camera)?;
    let svd = calibrated.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let singular_values = svd.singular_values;
    let scale = 0.5 * (singular_values[0] + singular_values[1]);
    if !scale.is_finite() || scale < 1.0e-12 {
        return None;
    }
    let essential = u * Matrix3::from_diagonal(&Vector3::new(scale, scale, 0.0)) * v_t;
    essential
        .iter()
        .all(|value| value.is_finite())
        .then_some(essential)
}

/// Conservative, opt-in repair for the specific COLMAP case where the full
/// verifier selected a fundamental matrix (`UNCALIBRATED`) even though this
/// caller has valid shared intrinsics.  The verifier's F inliers are
/// recomputed from *all* candidate correspondences, then the projected
/// `E_F = Kᵀ F K` is rescored in normalized coordinates. The repair is
/// accepted only when it retains at least half of the F support, clears the
/// caller's minimum support, has an unambiguous positive-depth solution, and
/// passes the strict manifold, residual-agreement, and deterministic
/// subset-refit stability gate below.
///
/// This is deliberately separate from the diagnostics-only
/// [`fundamental_to_essential_quality`]: callers can replace their accepted
/// match set only through the explicit CLI gate, while the default path never
/// invokes this function.
#[derive(Debug, Clone)]
pub(super) struct UncalibratedFToEssentialRefinement {
    pub(super) essential: Matrix3<f64>,
    pub(super) inlier_indices: Vec<usize>,
    pub(super) f_inlier_count: usize,
    pub(super) quality: EssentialPairQuality,
}

/// Direct calibrated-essential candidate used by the opt-in primary-model
/// policy. The full verifier already performs robust E RANSAC and an inlier
/// refit; this helper performs one deterministic refit on that E support and
/// rescored all candidate correspondences before admitting the result.
#[derive(Debug, Clone)]
pub(super) struct CalibratedEssentialPrimarySelection {
    pub(super) essential: Matrix3<f64>,
    pub(super) inlier_indices: Vec<usize>,
    pub(super) initial_inlier_count: usize,
    pub(super) quality: EssentialPairQuality,
}

/// Select a direct-E model for a known-intrinsics F-winning pair.
///
/// The support floor follows COLMAP's minimum-inlier gate. The `0.5` E/F
/// support floor is the same conservative floor used by the existing guarded
/// F→E path, while the pose check uses the source-derived hardened cheirality
/// policy (≥1° triangulation angle, ≤0.85 ambiguity, ≥50% positive-depth
/// support). Thus an F model may win on raw support, but a weak/planar/pure-
/// rotation E candidate cannot displace it merely because calibration exists.
pub(super) fn select_calibrated_essential_primary(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    min_matches: usize,
) -> Option<CalibratedEssentialPrimarySelection> {
    if report.config != ConfigurationType::Uncalibrated || camera.intrinsics().is_none() {
        return None;
    }
    let initial_essential = *report.essential.as_ref()?;
    let initial_inlier_count = report.essential_inliers.len();
    let required_support = min_matches.max(8);
    if initial_inlier_count < required_support {
        return None;
    }
    if report.f_inlier_count == 0
        || initial_inlier_count as f64 / (report.f_inlier_count as f64) < 0.5
    {
        return None;
    }
    // A report labelled Uncalibrated already excludes a homography that is
    // close to F, but keep the degeneracy guard explicit at this policy
    // boundary so a future classifier change cannot promote a planar edge.
    if report.h_inlier_count as f64 / report.f_inlier_count as f64 >= 0.8 {
        return None;
    }

    let initial_correspondences: Vec<TwoViewCorrespondence> = report
        .essential_inliers
        .iter()
        .filter_map(|&index| correspondences.get(index).copied())
        .collect();
    if initial_correspondences.len() != initial_inlier_count {
        return None;
    }
    let refit = EightPointEssentialMatrixEstimator::default()
        .estimate(&initial_correspondences, camera)
        .unwrap_or(initial_essential);
    if !refit.iter().all(|value| value.is_finite()) {
        return None;
    }
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let threshold_sq = normalized_threshold * normalized_threshold;
    let inlier_indices: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            normalized_essential_squared_sampson_error(&refit, correspondence, camera)
                .is_some_and(|error| error <= threshold_sq)
                .then_some(index)
        })
        .collect();
    let retained_floor = ((initial_inlier_count as f64) * 0.8).ceil() as usize;
    if inlier_indices.len() < required_support
        || inlier_indices.len() < retained_floor
        || inlier_indices.len() as f64 / (report.f_inlier_count as f64) < 0.5
    {
        return None;
    }

    let hardened = recover_relative_pose_with_options(
        &refit,
        correspondences,
        camera,
        &inlier_indices,
        &CheiralityOptions::hardened(),
    )?;
    let quality =
        essential_pair_quality_for_inliers(&refit, &inlier_indices, correspondences, camera)?;
    // `hardened` is the acceptance authority. The explicit finite checks keep
    // diagnostics and future callers from accepting an invalid quality row.
    if hardened.best_score <= 0 || !quality.mean_sampson.is_finite() || quality.angle_samples == 0 {
        return None;
    }
    Some(CalibratedEssentialPrimarySelection {
        essential: refit,
        inlier_indices,
        initial_inlier_count,
        quality,
    })
}

/// Decide whether the opt-in strict strategy must omit an F-winning pair.
///
/// A camera without usable intrinsics is deliberately left on the historical
/// F path: strict F→E is meaningful only when the caller supplied calibration.
/// Keeping this predicate separate makes the default-off and strict-pass/fail
/// behavior directly testable without invoking the parallel verifier.
pub(super) fn should_exclude_strict_uncalibrated_f_winner(
    strict: bool,
    camera: &Camera,
    report: &TwoViewGeometryReport,
    refinement: Option<&UncalibratedFToEssentialRefinement>,
) -> bool {
    strict
        && camera.intrinsics().is_some()
        && report.config == ConfigurationType::Uncalibrated
        && refinement.is_none()
}

pub(super) fn refine_uncalibrated_f_winner(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    min_matches: usize,
) -> Option<UncalibratedFToEssentialRefinement> {
    if report.config != ConfigurationType::Uncalibrated {
        return None;
    }
    let fundamental = report.fundamental.as_ref()?;
    let required_support = min_matches.max(8);
    let pixel_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0).max_error_px;
    let pixel_threshold_sq = pixel_threshold * pixel_threshold;
    let f_inliers: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (fundamental_squared_sampson_error(fundamental, correspondence) <= pixel_threshold_sq)
                .then_some(index)
        })
        .collect();
    if f_inliers.len() < required_support {
        return None;
    }

    let essential = project_fundamental_to_essential(fundamental, camera)?;
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let normalized_threshold_sq = normalized_threshold * normalized_threshold;
    let inlier_indices: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (normalized_essential_squared_sampson_error(&essential, correspondence, camera)
                .is_some_and(|error| error <= normalized_threshold_sq))
            .then_some(index)
        })
        .collect();
    if inlier_indices.len() < required_support
        || (inlier_indices.len() as f64 / f_inliers.len() as f64) < 0.5
    {
        return None;
    }

    let quality =
        essential_pair_quality_for_inliers(&essential, &inlier_indices, correspondences, camera)?;
    let cheirality_ratio = quality.best_cheirality as f64 / inlier_indices.len() as f64;
    let second_over_best = if quality.best_cheirality > 0 {
        quality.second_cheirality as f64 / quality.best_cheirality as f64
    } else {
        f64::INFINITY
    };
    // A positive-depth count alone can be misleading for an almost-pure
    // rotation. Require both a strong winner and at least one valid
    // triangulation sample; the thresholds are fixed structural guards, not
    // scene/GT-derived tuning knobs.
    const MIN_CHEIRALITY_RATIO: f64 = 0.75;
    const MAX_SECOND_OVER_BEST: f64 = 0.25;
    if quality.angle_samples == 0
        || !cheirality_ratio.is_finite()
        || cheirality_ratio < MIN_CHEIRALITY_RATIO
        || !second_over_best.is_finite()
        || second_over_best > MAX_SECOND_OVER_BEST
    {
        return None;
    }

    // A numerically plausible E_F is not necessarily a calibrated F.  Keep
    // the behavioral switch deliberately strict: the conversion must stay
    // close to the essential manifold, retain nearly all of the F support,
    // preserve its normalized residuals, and give the same pose under the
    // deterministic subset refits below.  These are geometry-consistency
    // checks, not scene-specific image/stem rules.
    let diagnostics = f_to_e_candidate_diagnostics(report, correspondences, camera)?;
    if !f_to_e_stability_gate(&diagnostics) {
        return None;
    }

    Some(UncalibratedFToEssentialRefinement {
        essential,
        inlier_indices,
        f_inlier_count: f_inliers.len(),
        quality,
    })
}

/// Diagnostics used to decide whether a calibrated F→E conversion is stable
/// enough to feed into tracks.  These values are all computed without GT: the
/// singular-value fields describe the calibration-induced manifold projection,
/// the residual fields compare the two algebraic models on the same pixels,
/// and the pose-spread fields compare deterministic F refits on F inlier
/// subsets.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct FToECandidateDiagnostics {
    pub(super) calibrated_s1: f64,
    pub(super) calibrated_s2: f64,
    pub(super) calibrated_s3: f64,
    pub(super) projection_distortion: f64,
    pub(super) s1_s2_mismatch: f64,
    pub(super) s3_s2_ratio: f64,
    pub(super) f_inliers: usize,
    pub(super) ef_inliers: usize,
    pub(super) ef_overlap_on_f: f64,
    pub(super) f_normalized_residual: f64,
    pub(super) ef_normalized_residual_on_f: f64,
    pub(super) ef_to_f_residual_ratio: f64,
    pub(super) cheirality_ratio: f64,
    pub(super) cheirality_margin: f64,
    pub(super) ef_angle_p25_deg: f64,
    pub(super) stable_refits: usize,
    pub(super) pose_rotation_spread_deg: f64,
    pub(super) pose_translation_spread_deg: f64,
}

/// Strict, GT-independent eligibility gate for replacing an F winner with
/// `E_F`.  An F that is genuinely explained by the known calibration should
/// already be close to rank-2/equal-singular essential geometry, and a stable
/// refit should not move its relative pose by several degrees.  The thresholds
/// are intentionally conservative so the opt-in path cannot broadly rewrite
/// a COLMAP-quality F graph.
pub(super) fn f_to_e_stability_gate(diagnostics: &FToECandidateDiagnostics) -> bool {
    f_to_e_stability_gate_with_max_pose_spread(diagnostics, 5.0)
}

/// Sequence registration has an independent consecutive-stem constraint and
/// only uses this conversion to recover a missing pose; it does not rewrite
/// the ordinary F-winning graph.  The existing sequential-SfM quality gate
/// treats ten degrees as the upper bound for a PnP/E pose disagreement, so
/// allow that same bound here while retaining every other strict F→E check.
pub(super) fn sequence_f_to_e_stability_gate(diagnostics: &FToECandidateDiagnostics) -> bool {
    f_to_e_stability_gate_with_max_pose_spread(diagnostics, 10.0)
}

/// High-support sequence-only exception for an otherwise strict F→E candidate.
///
/// A large translation spread is the sole field allowed to exceed the normal
/// sequence bound.  The exception is intentionally narrower than the ordinary
/// F→E gate: it needs at least 100 F and E-support rows, near-complete overlap,
/// an unambiguous positive-depth solution, and at least one degree of robust
/// fourth-view-free parallax.  The call with an infinite translation limit
/// still applies the ordinary manifold, residual, refit-count, finite-value,
/// and five-degree rotation-spread limits.
pub(super) fn sequence_f_to_e_high_support_override_gate(
    diagnostics: &FToECandidateDiagnostics,
) -> bool {
    const MIN_SUPPORT: usize = 100;
    const MIN_EF_OVERLAP_ON_F: f64 = 0.95;
    const MIN_CHEIRALITY_RATIO: f64 = 0.95;
    // This is the same second-solution exclusion used by the guarded F→E
    // refiner (`second / best <= 0.25`), expressed as a winner margin.
    const MIN_CHEIRALITY_MARGIN: f64 = 0.75;
    const MIN_ANGLE_P25_DEG: f64 = 1.0;
    diagnostics.pose_translation_spread_deg.is_finite()
        && diagnostics.pose_translation_spread_deg > 10.0
        && diagnostics.f_inliers >= MIN_SUPPORT
        && diagnostics.ef_inliers >= MIN_SUPPORT
        && diagnostics.ef_overlap_on_f.is_finite()
        && diagnostics.ef_overlap_on_f >= MIN_EF_OVERLAP_ON_F
        && diagnostics.cheirality_ratio.is_finite()
        && diagnostics.cheirality_ratio >= MIN_CHEIRALITY_RATIO
        && diagnostics.cheirality_margin.is_finite()
        && diagnostics.cheirality_margin >= MIN_CHEIRALITY_MARGIN
        && diagnostics.ef_angle_p25_deg.is_finite()
        && diagnostics.ef_angle_p25_deg >= MIN_ANGLE_P25_DEG
        && f_to_e_stability_gate_with_max_pose_spread(diagnostics, f64::INFINITY)
}

fn f_to_e_stability_gate_with_max_pose_spread(
    diagnostics: &FToECandidateDiagnostics,
    max_pose_spread_deg: f64,
) -> bool {
    const MAX_MANIFOLD_DISTORTION: f64 = 0.01;
    const MAX_S1_S2_MISMATCH: f64 = 0.02;
    const MAX_S3_S2_RATIO: f64 = 0.05;
    const MIN_EF_OVERLAP_ON_F: f64 = 0.90;
    const MAX_EF_TO_F_RESIDUAL_RATIO: f64 = 3.0;
    const MAX_POSE_SPREAD_DEG: f64 = 5.0;
    const MIN_STABLE_REFITS: usize = 2;

    let residual_agrees = (diagnostics.ef_to_f_residual_ratio.is_finite()
        && diagnostics.ef_to_f_residual_ratio <= MAX_EF_TO_F_RESIDUAL_RATIO)
        || (diagnostics.f_normalized_residual.is_finite()
            && diagnostics.ef_normalized_residual_on_f.is_finite()
            && diagnostics.f_normalized_residual <= 1.0e-8
            && diagnostics.ef_normalized_residual_on_f <= 1.0e-8);
    diagnostics.calibrated_s1.is_finite()
        && diagnostics.calibrated_s1 > 1.0e-12
        && diagnostics.calibrated_s2.is_finite()
        && diagnostics.calibrated_s2 > 1.0e-12
        && diagnostics.calibrated_s3.is_finite()
        && diagnostics.calibrated_s3 >= 0.0
        && diagnostics.projection_distortion.is_finite()
        && diagnostics.projection_distortion <= MAX_MANIFOLD_DISTORTION
        && diagnostics.s1_s2_mismatch.is_finite()
        && diagnostics.s1_s2_mismatch <= MAX_S1_S2_MISMATCH
        && diagnostics.s3_s2_ratio.is_finite()
        && diagnostics.s3_s2_ratio <= MAX_S3_S2_RATIO
        && diagnostics.f_inliers >= 8
        && diagnostics.ef_inliers >= 8
        && diagnostics.ef_overlap_on_f.is_finite()
        && diagnostics.ef_overlap_on_f >= MIN_EF_OVERLAP_ON_F
        && residual_agrees
        && diagnostics.cheirality_ratio.is_finite()
        && diagnostics.cheirality_margin.is_finite()
        && diagnostics.stable_refits >= MIN_STABLE_REFITS
        && diagnostics.pose_rotation_spread_deg.is_finite()
        && diagnostics.pose_rotation_spread_deg <= max_pose_spread_deg.min(MAX_POSE_SPREAD_DEG)
        && diagnostics.pose_translation_spread_deg.is_finite()
        && diagnostics.pose_translation_spread_deg <= max_pose_spread_deg
}

fn quaternion_delta_deg(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> f64 {
    (a.inverse() * *b).angle().abs().to_degrees()
}

pub(super) fn translation_direction_delta_deg(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let Some(a) = a.try_normalize(1.0e-12) else {
        return f64::NAN;
    };
    let Some(b) = b.try_normalize(1.0e-12) else {
        return f64::NAN;
    };
    a.dot(&b).clamp(-1.0, 1.0).acos().to_degrees()
}

/// Return whether an imported calibrated-E reference is sufficiently
/// constrained to act as a diagnostic pose reference.  The thresholds mirror
/// the existing hardened cheirality policy (50% positive, one-degree
/// parallax, 15% winner margin) and add a conservative cap on deterministic
/// subset/refit translation spread.  This is report-only; it never gates the
/// normal mapper.
pub(super) fn imported_reference_quality_is_strong(
    quality: &EssentialPairQuality,
    stable_refits: usize,
    translation_spread_deg: f64,
) -> bool {
    const MIN_CHEIRALITY_RATIO: f64 = 0.5;
    const MIN_CHEIRALITY_MARGIN: f64 = 0.15;
    const MIN_ANGLE_P25_DEG: f64 = 1.0;
    const MIN_STABLE_REFITS: usize = 2;
    const MAX_TRANSLATION_SPREAD_DEG: f64 = 20.0;
    quality.cheirality_ratio.is_finite()
        && quality.cheirality_ratio >= MIN_CHEIRALITY_RATIO
        && quality.cheirality_margin().is_finite()
        && quality.cheirality_margin() >= MIN_CHEIRALITY_MARGIN
        && quality.angle_p25_deg.is_finite()
        && quality.angle_p25_deg >= MIN_ANGLE_P25_DEG
        && stable_refits >= MIN_STABLE_REFITS
        && translation_spread_deg.is_finite()
        && translation_spread_deg <= MAX_TRANSLATION_SPREAD_DEG
}

impl EssentialPairQuality {
    fn cheirality_margin(&self) -> f64 {
        if self.best_cheirality <= 0 {
            0.0
        } else {
            (self.best_cheirality - self.second_cheirality) as f64 / self.best_cheirality as f64
        }
    }
}

/// Re-estimate an imported calibrated-E reference on three deterministic
/// subsets.  Every refit uses the same hardened cheirality selector, so the
/// sign of the camera-centre direction is resolved geometrically rather than
/// by taking an absolute dot product.  The result is a diagnostic stability
/// measure, not a replacement for the imported E.
fn imported_reference_pose_stability(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> (usize, f64, f64) {
    if correspondences.len() < 8 {
        return (0, f64::NAN, f64::NAN);
    }
    let full_indices: Vec<usize> = (0..correspondences.len()).collect();
    let Some(full_quality) = essential_pair_quality_for_inliers_with_options(
        essential,
        &full_indices,
        correspondences,
        camera,
        &CheiralityOptions::hardened(),
    ) else {
        return (0, f64::NAN, f64::NAN);
    };
    let full_rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        full_quality.rotation_quaternion[0],
        full_quality.rotation_quaternion[1],
        full_quality.rotation_quaternion[2],
        full_quality.rotation_quaternion[3],
    ));
    let full_center = Vector3::from_row_slice(&full_quality.center_direction);
    let subset_size = if correspondences.len() > 8 {
        correspondences
            .len()
            .min(64)
            .min(correspondences.len() - 1)
            .max(8)
    } else {
        correspondences.len()
    };
    let prefix = (0..subset_size).collect::<Vec<_>>();
    let suffix = (correspondences.len() - subset_size..correspondences.len()).collect::<Vec<_>>();
    let stride = correspondences.len().div_ceil(subset_size);
    let evenly_spaced = (0..correspondences.len())
        .step_by(stride.max(1))
        .take(subset_size)
        .collect::<Vec<_>>();

    let mut valid = 0usize;
    let mut max_rotation = 0.0f64;
    let mut max_translation = 0.0f64;
    for subset in [prefix, suffix, evenly_spaced] {
        if subset.len() < 8 {
            continue;
        }
        let subset_correspondences = subset
            .iter()
            .filter_map(|&index| correspondences.get(index).copied())
            .collect::<Vec<_>>();
        if subset_correspondences.len() < 8 {
            continue;
        }
        let Some(refit_essential) =
            EightPointEssentialMatrixEstimator::default().estimate(&subset_correspondences, camera)
        else {
            continue;
        };
        let refit_indices: Vec<usize> = (0..subset_correspondences.len()).collect();
        let Some(refit_recovery) = recover_relative_pose_with_options(
            &refit_essential,
            &subset_correspondences,
            camera,
            &refit_indices,
            &CheiralityOptions::hardened(),
        ) else {
            continue;
        };
        let refit_center = -refit_recovery
            .rotation
            .inverse()
            .transform_vector(&refit_recovery.translation_unit);
        let rotation_delta = quaternion_delta_deg(&full_rotation, &refit_recovery.rotation);
        let translation_delta = translation_direction_delta_deg(&full_center, &refit_center);
        if !rotation_delta.is_finite() || !translation_delta.is_finite() {
            continue;
        }
        valid += 1;
        max_rotation = max_rotation.max(rotation_delta);
        max_translation = max_translation.max(translation_delta);
    }
    (valid, max_rotation, max_translation)
}

/// Extract a strong, calibrated imported-E reference and its deterministic
/// stability diagnostics.  The accepted match order is the replay-file order
/// and is independent of the candidate mapper traversal.
fn imported_essential_reference_quality(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<ImportedEssentialReferenceQuality> {
    let indices: Vec<usize> = (0..correspondences.len()).collect();
    let quality = essential_pair_quality_for_inliers_with_options(
        essential,
        &indices,
        correspondences,
        camera,
        &CheiralityOptions::hardened(),
    )?;
    let (stable_refits, rotation_spread_deg, translation_spread_deg) =
        imported_reference_pose_stability(essential, correspondences, camera);
    if !imported_reference_quality_is_strong(&quality, stable_refits, translation_spread_deg) {
        return None;
    }
    let center_direction = Vector3::from_row_slice(&quality.center_direction);
    if !center_direction.iter().all(|value| value.is_finite()) {
        return None;
    }
    let rotation = UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
        quality.rotation_quaternion[0],
        quality.rotation_quaternion[1],
        quality.rotation_quaternion[2],
        quality.rotation_quaternion[3],
    ));
    Some(ImportedEssentialReferenceQuality {
        rotation,
        center_direction,
        cheirality_margin: quality.cheirality_margin(),
        angle_p25_deg: quality.angle_p25_deg,
        stable_refits,
        rotation_spread_deg,
        translation_spread_deg,
    })
}

/// Refit F on three deterministic subsets of its inliers and compare the
/// resulting calibrated poses to the full F→E pose.  The subset construction
/// is deliberately fixed (prefix, suffix, evenly-spaced) so a run can be
/// reproduced byte-for-byte and does not consume random state.
fn f_to_e_pose_stability(
    essential: &Matrix3<f64>,
    f_inliers: &[usize],
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> (usize, f64, f64) {
    if f_inliers.len() < 8 {
        return (0, f64::NAN, f64::NAN);
    }
    let Some(full_recovery) = recover_relative_pose_with_options(
        essential,
        correspondences,
        camera,
        f_inliers,
        &CheiralityOptions::default(),
    ) else {
        return (0, f64::NAN, f64::NAN);
    };
    let full_center = -full_recovery
        .rotation
        .inverse()
        .transform_vector(&full_recovery.translation_unit);
    let subset_size = if f_inliers.len() > 8 {
        f_inliers.len().min(64).min(f_inliers.len() - 1).max(8)
    } else {
        f_inliers.len()
    };
    let prefix = f_inliers
        .iter()
        .take(subset_size)
        .copied()
        .collect::<Vec<_>>();
    let suffix = f_inliers
        .iter()
        .rev()
        .take(subset_size)
        .copied()
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect::<Vec<_>>();
    let stride = f_inliers.len().div_ceil(subset_size);
    let evenly_spaced = f_inliers
        .iter()
        .step_by(stride.max(1))
        .take(subset_size)
        .copied()
        .collect::<Vec<_>>();
    let subsets = [prefix, suffix, evenly_spaced];

    let mut valid = 0usize;
    let mut max_rotation = 0.0f64;
    let mut max_translation = 0.0f64;
    for subset in subsets {
        if subset.len() < 8 {
            continue;
        }
        let subset_corrs = subset
            .iter()
            .filter_map(|&index| correspondences.get(index).copied())
            .collect::<Vec<_>>();
        if subset_corrs.len() < 8 {
            continue;
        }
        let Some(refit_f) = estimate_fundamental_dlt(&subset_corrs) else {
            continue;
        };
        let Some(refit_e) = project_fundamental_to_essential(&refit_f, camera) else {
            continue;
        };
        let Some(refit_recovery) = recover_relative_pose_with_options(
            &refit_e,
            correspondences,
            camera,
            &subset,
            &CheiralityOptions::default(),
        ) else {
            continue;
        };
        let rotation_delta =
            quaternion_delta_deg(&full_recovery.rotation, &refit_recovery.rotation);
        let refit_center = -refit_recovery
            .rotation
            .inverse()
            .transform_vector(&refit_recovery.translation_unit);
        let translation_delta = translation_direction_delta_deg(&full_center, &refit_center);
        if !rotation_delta.is_finite() || !translation_delta.is_finite() {
            continue;
        }
        valid += 1;
        max_rotation = max_rotation.max(rotation_delta);
        max_translation = max_translation.max(translation_delta);
    }
    (valid, max_rotation, max_translation)
}

/// Compute all GT-independent F→E candidate diagnostics.  Returning `None`
/// means that the pixel F or its calibrated representation was invalid; the
/// caller logs that case separately and keeps the legacy F result.
pub(super) fn f_to_e_candidate_diagnostics(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<FToECandidateDiagnostics> {
    if report.config != ConfigurationType::Uncalibrated {
        return None;
    }
    let fundamental = report.fundamental.as_ref()?;
    let calibrated = calibrated_fundamental(fundamental, camera)?;
    let singular_values = calibrated.svd(false, false).singular_values;
    let calibrated_norm = calibrated.norm();
    let essential = project_fundamental_to_essential(fundamental, camera)?;
    let projection_distortion = if calibrated_norm > 1.0e-12 {
        (calibrated - essential).norm() / calibrated_norm
    } else {
        f64::NAN
    };
    let s1_s2_mismatch = if singular_values[0] + singular_values[1] > 1.0e-12 {
        (singular_values[0] - singular_values[1]).abs()
            / (0.5 * (singular_values[0] + singular_values[1]))
    } else {
        f64::NAN
    };
    let s3_s2_ratio = if singular_values[1] > 1.0e-12 {
        singular_values[2] / singular_values[1]
    } else {
        f64::NAN
    };
    let pixel_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0).max_error_px;
    let f_threshold_sq = pixel_threshold * pixel_threshold;
    let f_inliers = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (fundamental_squared_sampson_error(fundamental, correspondence) <= f_threshold_sq)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let normalized_threshold_sq = normalized_threshold * normalized_threshold;
    let ef_inliers = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            normalized_essential_squared_sampson_error(&essential, correspondence, camera)
                .is_some_and(|error| error <= normalized_threshold_sq)
                .then_some(index)
        })
        .collect::<Vec<_>>();
    let ef_set = ef_inliers.iter().copied().collect::<HashSet<_>>();
    let ef_overlap_on_f = if f_inliers.is_empty() {
        f64::NAN
    } else {
        f_inliers
            .iter()
            .filter(|index| ef_set.contains(index))
            .count() as f64
            / f_inliers.len() as f64
    };
    let f_normalized_residual =
        mean_normalized_essential_sampson_error(&calibrated, correspondences, camera, &f_inliers);
    let ef_normalized_residual_on_f =
        mean_normalized_essential_sampson_error(&essential, correspondences, camera, &f_inliers);
    let ef_to_f_residual_ratio = if f_normalized_residual > 1.0e-12 {
        ef_normalized_residual_on_f / f_normalized_residual
    } else {
        f64::NAN
    };
    let quality =
        essential_pair_quality_for_inliers(&essential, &ef_inliers, correspondences, camera);
    let ef_angle_p25_deg = quality
        .as_ref()
        .map_or(f64::NAN, |quality| quality.angle_p25_deg);
    let (cheirality_ratio, cheirality_margin) = if let Some(quality) = quality.as_ref() {
        let ratio = if ef_inliers.is_empty() {
            f64::NAN
        } else {
            quality.best_cheirality as f64 / ef_inliers.len() as f64
        };
        let margin = if quality.best_cheirality > 0 {
            (quality.best_cheirality - quality.second_cheirality) as f64
                / quality.best_cheirality as f64
        } else {
            f64::NAN
        };
        (ratio, margin)
    } else {
        (f64::NAN, f64::NAN)
    };
    let (stable_refits, pose_rotation_spread_deg, pose_translation_spread_deg) =
        f_to_e_pose_stability(&essential, &f_inliers, correspondences, camera);
    Some(FToECandidateDiagnostics {
        calibrated_s1: singular_values[0],
        calibrated_s2: singular_values[1],
        calibrated_s3: singular_values[2],
        projection_distortion,
        s1_s2_mismatch,
        s3_s2_ratio,
        f_inliers: f_inliers.len(),
        ef_inliers: ef_inliers.len(),
        ef_overlap_on_f,
        f_normalized_residual,
        ef_normalized_residual_on_f,
        ef_to_f_residual_ratio,
        cheirality_ratio,
        cheirality_margin,
        ef_angle_p25_deg,
        stable_refits,
        pose_rotation_spread_deg,
        pose_translation_spread_deg,
    })
}

#[derive(Debug, Clone)]
pub(super) struct FundamentalToEssentialQuality {
    f_inliers: usize,
    ef_inliers: usize,
    f_mean_sampson_px: f64,
    direct_mean_sampson_on_f: f64,
    ef_mean_sampson_on_f: f64,
    ef_mean_sampson_on_direct: f64,
    ef_mean_sampson: f64,
    ef_quality: Option<EssentialPairQuality>,
}

/// Recompute the F inlier set from the report's refined F, then evaluate the
/// calibrated E obtained from that F.  The report does not expose F indices,
/// so recomputation is necessary and is deterministic.  This never feeds the
/// derived model back into verification or mapping; it is a diagnostics-only
/// comparison of the two calibrated pose hypotheses.
pub(super) fn fundamental_to_essential_quality(
    report: &TwoViewGeometryReport,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<FundamentalToEssentialQuality> {
    let fundamental = report.fundamental.as_ref()?;
    let pixel_threshold = TwoViewGeometryOptions::for_camera(camera, 4.0).max_error_px;
    let pixel_threshold_sq = pixel_threshold * pixel_threshold;
    let f_inliers: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (fundamental_squared_sampson_error(fundamental, correspondence) <= pixel_threshold_sq)
                .then_some(index)
        })
        .collect();
    if f_inliers.len() < 8 {
        return Some(FundamentalToEssentialQuality {
            f_inliers: f_inliers.len(),
            ef_inliers: 0,
            f_mean_sampson_px: f64::NAN,
            direct_mean_sampson_on_f: f64::NAN,
            ef_mean_sampson_on_f: f64::NAN,
            ef_mean_sampson_on_direct: f64::NAN,
            ef_mean_sampson: f64::NAN,
            ef_quality: None,
        });
    }
    let essential = project_fundamental_to_essential(fundamental, camera)?;
    let normalized_threshold =
        TwoViewGeometryOptions::for_camera(camera, 4.0).essential_sampson_threshold;
    let normalized_threshold_sq = normalized_threshold * normalized_threshold;
    let ef_inliers: Vec<usize> = correspondences
        .iter()
        .enumerate()
        .filter_map(|(index, correspondence)| {
            (normalized_essential_squared_sampson_error(&essential, correspondence, camera)
                .is_some_and(|error| error <= normalized_threshold_sq))
            .then_some(index)
        })
        .collect();
    let f_mean_sampson_px = {
        let mut total = 0.0;
        let mut count = 0usize;
        for &index in &f_inliers {
            let error = fundamental_squared_sampson_error(fundamental, &correspondences[index]);
            if error.is_finite() {
                total += error.sqrt();
                count += 1;
            }
        }
        if count == 0 {
            f64::NAN
        } else {
            total / count as f64
        }
    };
    Some(FundamentalToEssentialQuality {
        f_inliers: f_inliers.len(),
        ef_inliers: ef_inliers.len(),
        f_mean_sampson_px,
        direct_mean_sampson_on_f: report.essential.as_ref().map_or(f64::NAN, |essential| {
            mean_normalized_essential_sampson_error(essential, correspondences, camera, &f_inliers)
        }),
        ef_mean_sampson_on_f: mean_normalized_essential_sampson_error(
            &essential,
            correspondences,
            camera,
            &f_inliers,
        ),
        ef_mean_sampson_on_direct: mean_normalized_essential_sampson_error(
            &essential,
            correspondences,
            camera,
            &report.essential_inliers,
        ),
        ef_mean_sampson: mean_normalized_essential_sampson_error(
            &essential,
            correspondences,
            camera,
            &ef_inliers,
        ),
        ef_quality: essential_pair_quality_for_inliers(
            &essential,
            &ef_inliers,
            correspondences,
            camera,
        ),
    })
}

pub(super) fn format_essential_pair_quality(quality: Option<EssentialPairQuality>) -> String {
    let Some(q) = quality else {
        return " cheirality_best=NA cheirality_second=NA cheirality_ratio=NA cheirality_second_over_best=NA sampson_mean=NA pose_q=NA center_dir=NA angle_samples=0 angle_ge_1deg=0 angle_p10_deg=NA angle_p25_deg=NA angle_median_deg=NA depth_ratio_p10=NA depth_ratio_p25=NA depth_ratio_median=NA".to_owned();
    };
    let second_over_best = if q.best_cheirality > 0 {
        q.second_cheirality as f64 / q.best_cheirality as f64
    } else {
        f64::NAN
    };
    format!(
        " cheirality_best={} cheirality_second={} cheirality_ratio={:.6} cheirality_second_over_best={:.6} sampson_mean={:.8} pose_q={:.9},{:.9},{:.9},{:.9} center_dir={:.9},{:.9},{:.9} angle_samples={} angle_ge_1deg={} angle_p10_deg={:.6} angle_p25_deg={:.6} angle_median_deg={:.6} depth_ratio_p10={:.6} depth_ratio_p25={:.6} depth_ratio_median={:.6}",
        q.best_cheirality,
        q.second_cheirality,
        q.cheirality_ratio,
        second_over_best,
        q.mean_sampson,
        q.rotation_quaternion[0],
        q.rotation_quaternion[1],
        q.rotation_quaternion[2],
        q.rotation_quaternion[3],
        q.center_direction[0],
        q.center_direction[1],
        q.center_direction[2],
        q.angle_samples,
        q.angle_ge_1deg,
        q.angle_p10_deg,
        q.angle_p25_deg,
        q.angle_median_deg,
        q.depth_ratio_p10,
        q.depth_ratio_p25,
        q.depth_ratio_median,
    )
}

pub(super) fn format_fundamental_to_essential_quality(
    quality: Option<FundamentalToEssentialQuality>,
) -> String {
    let Some(q) = quality else {
        return " f2e_f_inliers=NA f2e_ef_inliers=NA f2e_f_mean_sampson_px=NA f2e_direct_mean_sampson_on_f=NA f2e_ef_mean_sampson_on_f=NA f2e_ef_mean_sampson_on_direct=NA f2e_ef_mean_sampson=NA f2e_ef_cheirality_best=NA f2e_ef_cheirality_second=NA f2e_ef_cheirality_ratio=NA f2e_ef_pose_q=NA f2e_ef_center_dir=NA f2e_ef_angle_p10_deg=NA f2e_ef_angle_p25_deg=NA f2e_ef_angle_median_deg=NA f2e_ef_depth_ratio_p10=NA f2e_ef_sampson_quality=NA".to_owned();
    };
    let (best, second, ratio) = q.ef_quality.as_ref().map_or_else(
        || ("NA".to_owned(), "NA".to_owned(), "NA".to_owned()),
        |value| {
            (
                value.best_cheirality.to_string(),
                value.second_cheirality.to_string(),
                format!("{:.6}", value.cheirality_ratio),
            )
        },
    );
    format!(
        " f2e_f_inliers={} f2e_ef_inliers={} f2e_f_mean_sampson_px={:.6} f2e_direct_mean_sampson_on_f={:.8} f2e_ef_mean_sampson_on_f={:.8} f2e_ef_mean_sampson_on_direct={:.8} f2e_ef_mean_sampson={:.8} f2e_ef_cheirality_best={} f2e_ef_cheirality_second={} f2e_ef_cheirality_ratio={} f2e_ef_pose_q={:.9},{:.9},{:.9},{:.9} f2e_ef_center_dir={:.9},{:.9},{:.9} f2e_ef_angle_p10_deg={:.6} f2e_ef_angle_p25_deg={:.6} f2e_ef_angle_median_deg={:.6} f2e_ef_depth_ratio_p10={:.6} f2e_ef_sampson_quality={}",
        q.f_inliers,
        q.ef_inliers,
        q.f_mean_sampson_px,
        q.direct_mean_sampson_on_f,
        q.ef_mean_sampson_on_f,
        q.ef_mean_sampson_on_direct,
        q.ef_mean_sampson,
        best,
        second,
        ratio,
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.rotation_quaternion[0]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.rotation_quaternion[1]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.rotation_quaternion[2]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.rotation_quaternion[3]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.center_direction[0]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.center_direction[1]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.center_direction[2]),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.angle_p10_deg),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.angle_p25_deg),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.angle_median_deg),
        q.ef_quality
            .as_ref()
            .map_or(f64::NAN, |value| value.depth_ratio_p10),
        q.ef_quality.is_some(),
    )
}
