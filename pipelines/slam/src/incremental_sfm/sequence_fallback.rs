//! Sequence relative-pose registration fallback for images PnP cannot register.

use super::*;

/// Register at most one image with sequence-relative fallback after an
/// ordinary post-refinement sweep has stalled.  Each accepted pose
/// immediately retriangulates existing tracks; the after-post scheduler can
/// then resume ordinary PnP before asking for another provisional pose.
#[allow(clippy::too_many_arguments)]
pub(super) fn sequence_relative_pose_registration_once_with_overrides_and_carry(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    sequence_override_pair_indices: Option<&[usize]>,
    carried_sequence_fallback: Option<(usize, f64)>,
) -> Result<Option<(usize, f64)>, BaError> {
    let Some(mut proposal) = sequence_relative_pose_fallback_with_overrides(
        camera,
        features,
        pairwise,
        poses,
        config,
        sequence_override_pair_indices,
    ) else {
        return Ok(None);
    };
    if config.sequence_fallback_carry_scale {
        if let Some((carried_previous_image, carried_scale)) = carried_sequence_fallback {
            if proposal.previous_image == carried_previous_image {
                let (selected_scale, carry_applied) = carried_sequence_scale_or_projection(
                    Some(carried_scale),
                    proposal.translation_scale,
                    proposal.translation_scale_median,
                );
                if carry_applied {
                    if let Some(rescaled_pose) = rescale_sequence_pose_translation(
                        poses[proposal.previous_image]
                            .as_ref()
                            .expect("fallback proposal predecessor is registered"),
                        &proposal.pose,
                        selected_scale,
                    ) {
                        proposal.pose = rescaled_pose;
                        proposal.translation_scale = selected_scale;
                        proposal.translation_scale_carried = true;
                    } else if sfm_debug_enabled() {
                        eprintln!(
                            "sfm-debug: sequence fallback carry_scale_invalid image={} previous={} carried_scale={:.6e} reason=pose_rescale_failed; using fresh proposal scale={:.6e}",
                            proposal.next_image,
                            proposal.previous_image,
                            carried_scale,
                            proposal.translation_scale,
                        );
                    }
                } else if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: sequence fallback carry_scale_invalid image={} previous={} carried_scale={:.6e} recent_median={:.6e} bounds=({:.6e},{:.6e}); using fresh proposal scale={:.6e}",
                        proposal.next_image,
                        proposal.previous_image,
                        carried_scale,
                        proposal.translation_scale_median,
                        0.25 * proposal.translation_scale_median,
                        4.0 * proposal.translation_scale_median,
                        proposal.translation_scale,
                    );
                }
            } else if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: sequence fallback carry_scale_stale image={} previous={} carried_previous={} carried_scale={:.6e}; using fresh proposal scale={:.6e}",
                    proposal.next_image,
                    proposal.previous_image,
                    carried_previous_image,
                    carried_scale,
                    proposal.translation_scale,
                );
            }
        }
    }
    let image = proposal.next_image;
    let accepted_scale = proposal.translation_scale;
    poses[image] = Some(proposal.pose);
    triangulate_pending_with_config(camera, features, tracks, poses, config, track_point);
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: sequence fallback registered image {} from previous {} \
             pair={} inliers={} triangulated={}/{} ratio={:.6} scale_mode={} median_scale={:.6e} projected_scale={:?} scale={:.6e} chirality_margin={:.3}",
            image,
            proposal.previous_image,
            proposal.pair_index,
            proposal.pair_inliers,
            proposal.triangulated_points,
            proposal.triangulation_candidates,
                proposal.triangulated_points as f64
                / proposal.triangulation_candidates.max(1) as f64,
            if proposal.translation_scale_carried {
                "carried_provisional"
            } else if proposal.translation_scale_projection.is_some() {
                if config.sequence_relaxed_constant_velocity_scale {
                    "constant_velocity_projected_relaxed"
                } else {
                    "constant_velocity_projected"
                }
            } else {
                "median_magnitude"
            },
            proposal.translation_scale_median,
            proposal.translation_scale_projection,
            proposal.translation_scale,
            proposal.chirality_margin,
        );
    }
    Ok(Some((image, accepted_scale)))
}

/// Compatibility wrapper used by the eager sequence path.  It deliberately
/// supplies no carry state, preserving the historical eager behavior even
/// when callers use the new after-post-only policy.
#[allow(clippy::too_many_arguments)]
fn sequence_relative_pose_registration_once_with_overrides(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    sequence_override_pair_indices: Option<&[usize]>,
) -> Result<bool, BaError> {
    Ok(
        sequence_relative_pose_registration_once_with_overrides_and_carry(
            camera,
            features,
            pairwise,
            tracks,
            config,
            poses,
            track_point,
            sequence_override_pair_indices,
            None,
        )?
        .is_some(),
    )
}

/// Complete the eager sequence-relative post-refinement pass.  The legacy
/// eager mode intentionally keeps chaining accepted provisional poses without
/// an intervening ordinary PnP sweep; the separate after-post scheduler uses
/// the one-shot helper above instead.
#[allow(clippy::too_many_arguments)]
pub(super) fn sequence_relative_pose_registration_pass_with_overrides(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    sequence_override_pair_indices: Option<&[usize]>,
) -> Result<usize, BaError> {
    let mut registered = 0usize;
    while sequence_relative_pose_registration_once_with_overrides(
        camera,
        features,
        pairwise,
        tracks,
        config,
        poses,
        track_point,
        sequence_override_pair_indices,
    )? {
        registered += 1;
    }
    Ok(registered)
}

/// A provisional pose proposed by the opt-in sequence fallback.  The pose is
/// deliberately kept separate from the ordinary PnP report: it is admitted
/// only after its consecutive essential edge has passed the same triangulation
/// and reprojection checks used by normal growth.
#[derive(Debug, Clone)]
pub(super) struct SequenceRelativePoseProposal {
    next_image: usize,
    previous_image: usize,
    pair_index: usize,
    pair_inliers: usize,
    triangulated_points: usize,
    triangulation_candidates: usize,
    translation_scale: f64,
    translation_scale_median: f64,
    translation_scale_projection: Option<f64>,
    translation_scale_carried: bool,
    chirality_margin: f64,
    pose: Pose,
}

/// Return the median of a finite, non-empty sample.  The helper is kept
/// private and deterministic so the sequence fallback does not depend on
/// floating-point sorting or traversal order elsewhere in the mapper.
fn finite_median(values: &mut [f64]) -> Option<f64> {
    if values.is_empty() || !values.iter().all(|value| value.is_finite()) {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

/// Estimate the metric scale of a new consecutive relative pose from the
/// latest registered consecutive camera steps.  Only numeric-stem neighbours
/// count; a missing camera breaks neither the rest of the graph nor the
/// ordinary unordered mapper, but it does prevent this opt-in fallback from
/// fabricating a scale bridge.  At least two steps are required.  A median is
/// used as the robust estimator; when a non-zero MAD is available, samples
/// farther than the standard three-MAD fence are omitted before taking the
/// final median.  No dataset-specific absolute clamp is applied.
pub(super) fn robust_recent_consecutive_step_scale(
    poses: &[Option<Pose>],
    stem_values: &[u64],
) -> Option<(f64, f64, usize)> {
    if poses.len() != stem_values.len() {
        return None;
    }
    let mut by_stem: Vec<(u64, usize)> = stem_values
        .iter()
        .copied()
        .enumerate()
        .map(|(image, stem)| (stem, image))
        .collect();
    by_stem.sort_unstable_by_key(|&(stem, image)| (stem, image));

    let mut steps = Vec::new();
    for pair in by_stem.windows(2) {
        let [(left_stem, left_image), (right_stem, right_image)] = pair else {
            unreachable!("windows(2) always has two entries");
        };
        if right_stem.saturating_sub(*left_stem) != 1 {
            continue;
        }
        let (Some(left), Some(right)) = (&poses[*left_image], &poses[*right_image]) else {
            continue;
        };
        let step = (right.camera_center_world() - left.camera_center_world()).norm();
        if step.is_finite() && step > 0.0 {
            steps.push(step);
        }
    }
    if steps.len() < 2 {
        return None;
    }

    // Keep only the latest three successful consecutive steps.  `steps` is
    // already in ascending stem order, independent of feature-row order.
    if steps.len() > 3 {
        let first = steps.len() - 3;
        steps.drain(..first);
    }
    let mut center_sample = steps.clone();
    let center = finite_median(&mut center_sample)?;
    let mut deviations: Vec<f64> = steps.iter().map(|step| (step - center).abs()).collect();
    let mad = finite_median(&mut deviations)?;

    let filtered = if mad.is_finite() && mad > 1.0e-12 {
        let fence = 3.0 * mad;
        steps
            .iter()
            .copied()
            .filter(|step| (*step - center).abs() <= fence)
            .collect::<Vec<_>>()
    } else {
        steps.clone()
    };
    let mut final_sample = if filtered.len() >= 2 { filtered } else { steps };
    let scale = finite_median(&mut final_sample)?;
    (scale.is_finite() && scale > 0.0).then_some((scale, mad, final_sample.len()))
}

#[derive(Debug, Clone, Copy)]
pub(super) struct SequenceProjectedScaleDiagnostic {
    pub(super) projected_scale: f64,
    pub(super) recent_median: f64,
    mad: f64,
    sample_count: usize,
    predicted_velocity: Vector3<f64>,
}

/// Apply the intentionally broad safety bounds for the relaxed projected
/// scale policy.  The local MAD fence belongs to the strict policy; this
/// helper only prevents a non-finite, reversed, or catastrophic scale while
/// allowing a genuine turn to change the projected step length.
pub(super) fn relaxed_projected_scale_is_valid(projected_scale: f64, recent_median: f64) -> bool {
    if !projected_scale.is_finite()
        || projected_scale <= 0.0
        || !recent_median.is_finite()
        || recent_median <= 0.0
    {
        return false;
    }
    let lower = 0.25 * recent_median;
    let upper = 4.0 * recent_median;
    lower.is_finite() && upper.is_finite() && projected_scale >= lower && projected_scale <= upper
}

/// Choose the baseline magnitude for a sequence fallback.  A carried scale
/// is deliberately subject to the same broad finite/positive and
/// 0.25x..4x-recent-median safety fence as relaxed projection.  Invalid or
/// absent carry state falls back to the freshly proposed scale, so enabling
/// the policy cannot turn a recoverable candidate into an unconditional
/// rejection.
pub(super) fn carried_sequence_scale_or_projection(
    carried_scale: Option<f64>,
    proposed_scale: f64,
    recent_median: f64,
) -> (f64, bool) {
    if let Some(scale) = carried_scale {
        if relaxed_projected_scale_is_valid(scale, recent_median) {
            return (scale, true);
        }
    }
    (proposed_scale, false)
}

/// Rescale only the camera-centre displacement of a provisional pose while
/// preserving its recovered rotation.  The proposed pose already encodes the
/// relative-pose convention; rebuilding its world-to-camera translation from
/// the new centre avoids accidentally scaling the translation in the wrong
/// frame when a reversed pair supplied the proposal.
pub(super) fn rescale_sequence_pose_translation(
    previous: &Pose,
    proposed: &Pose,
    translation_scale: f64,
) -> Option<Pose> {
    if !translation_scale.is_finite()
        || translation_scale <= 0.0
        || !proposed
            .world_to_camera
            .rotation
            .coords
            .iter()
            .all(|value| value.is_finite())
    {
        return None;
    }
    let previous_center = previous.camera_center_world();
    let proposed_center = proposed.camera_center_world();
    let displacement = proposed_center - previous_center;
    let displacement_norm = displacement.norm();
    if !displacement.iter().all(|value| value.is_finite())
        || !displacement_norm.is_finite()
        || displacement_norm <= 1.0e-12
    {
        return None;
    }
    let center = previous_center + displacement * (translation_scale / displacement_norm);
    let rotation = proposed.world_to_camera.rotation;
    let translation = -(rotation * center.coords);
    if !translation.iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(Pose::from_world_to_camera(rotation, translation))
}

/// Update the after-post carry state after one provisional registration.  A
/// normal post/PnP insertion breaks the chain; otherwise the newly accepted
/// fallback becomes the only state eligible for the next consecutive image.
pub(super) fn next_sequence_fallback_carry_state(
    fallback_image: usize,
    fallback_scale: f64,
    resumed_post_registered: usize,
) -> Option<(usize, f64)> {
    (resumed_post_registered == 0).then_some((fallback_image, fallback_scale))
}

/// Compute the un-gated diagnostics for a sequence constant-velocity
/// projection.  Keeping the raw positive/negative and out-of-fence result
/// available lets the opt-in fallback explain why a candidate was rejected.
pub(super) fn projected_recent_consecutive_step_scale_diagnostic(
    poses: &[Option<Pose>],
    stem_values: &[u64],
    latest_stem: u64,
    candidate_direction: Vector3<f64>,
) -> Option<SequenceProjectedScaleDiagnostic> {
    if poses.len() != stem_values.len() {
        return None;
    }
    let direction_norm = candidate_direction.norm();
    if !direction_norm.is_finite() || direction_norm <= 1.0e-12 {
        return None;
    }
    let direction = candidate_direction / direction_norm;
    let mut by_stem: Vec<(u64, usize)> = stem_values
        .iter()
        .copied()
        .enumerate()
        .map(|(image, stem)| (stem, image))
        .collect();
    by_stem.sort_unstable_by_key(|&(stem, image)| (stem, image));

    let mut samples: Vec<(Vector3<f64>, f64)> = Vec::new();
    for pair in by_stem.windows(2) {
        let [(left_stem, left_image), (right_stem, right_image)] = pair else {
            unreachable!("windows(2) always has two entries");
        };
        if *right_stem > latest_stem || right_stem.saturating_sub(*left_stem) != 1 {
            continue;
        }
        let (Some(left), Some(right)) = (&poses[*left_image], &poses[*right_image]) else {
            continue;
        };
        let velocity = right.camera_center_world() - left.camera_center_world();
        let magnitude = velocity.norm();
        if velocity.iter().all(|value| value.is_finite())
            && magnitude.is_finite()
            && magnitude > 0.0
        {
            samples.push((velocity, magnitude));
        }
    }
    if samples.len() < 2 {
        return None;
    }
    if samples.len() > 3 {
        let first = samples.len() - 3;
        samples.drain(..first);
    }

    let mut magnitudes: Vec<f64> = samples.iter().map(|(_, magnitude)| *magnitude).collect();
    let recent_median = finite_median(&mut magnitudes)?;
    let mut deviations: Vec<f64> = samples
        .iter()
        .map(|(_, magnitude)| (magnitude - recent_median).abs())
        .collect();
    let mad = finite_median(&mut deviations)?;
    let filtered: Vec<Vector3<f64>> = if mad.is_finite() && mad > 1.0e-12 {
        let fence = 3.0 * mad;
        samples
            .iter()
            .filter(|(_, magnitude)| (*magnitude - recent_median).abs() <= fence)
            .map(|(velocity, _)| *velocity)
            .collect()
    } else {
        samples.iter().map(|(velocity, _)| *velocity).collect()
    };
    let velocities = if filtered.len() >= 2 {
        filtered
    } else {
        samples.iter().map(|(velocity, _)| *velocity).collect()
    };
    let mut x = velocities
        .iter()
        .map(|velocity| velocity.x)
        .collect::<Vec<_>>();
    let mut y = velocities
        .iter()
        .map(|velocity| velocity.y)
        .collect::<Vec<_>>();
    let mut z = velocities
        .iter()
        .map(|velocity| velocity.z)
        .collect::<Vec<_>>();
    let predicted_velocity = Vector3::new(
        finite_median(&mut x)?,
        finite_median(&mut y)?,
        finite_median(&mut z)?,
    );
    if !predicted_velocity.iter().all(|value| value.is_finite()) {
        return None;
    }
    let projected_scale = predicted_velocity.dot(&direction);
    Some(SequenceProjectedScaleDiagnostic {
        projected_scale,
        recent_median,
        mad,
        sample_count: velocities.len(),
        predicted_velocity,
    })
}

/// Estimate a sequence fallback step from a robust constant-velocity
/// prediction.  The velocity samples are camera-centre displacements for the
/// latest one-to-three registered consecutive stem pairs ending no later than
/// `latest_stem`; a component-wise median is used so one turn or scale outlier
/// cannot dominate the prediction.  The candidate direction is supplied in
/// the same world frame and is normalized before projection.
///
/// The returned tuple is `(projected_scale, recent_median, mad, sample_count,
/// predicted_velocity)`.  A projection is valid only when it is positive,
/// finite, and within the same three-MAD fence used by
/// [`robust_recent_consecutive_step_scale`].  With zero MAD (constant recent
/// steps), a small relative floating-point tolerance is used around the
/// median rather than admitting an arbitrary turn.
#[cfg(test)]
pub(super) fn projected_recent_consecutive_step_scale(
    poses: &[Option<Pose>],
    stem_values: &[u64],
    latest_stem: u64,
    candidate_direction: Vector3<f64>,
) -> Option<(f64, f64, f64, usize, Vector3<f64>)> {
    let diagnostic = projected_recent_consecutive_step_scale_diagnostic(
        poses,
        stem_values,
        latest_stem,
        candidate_direction,
    )?;
    if !diagnostic.projected_scale.is_finite() || diagnostic.projected_scale <= 0.0 {
        return None;
    }
    let allowed_deviation = if diagnostic.mad.is_finite() && diagnostic.mad > 1.0e-12 {
        3.0 * diagnostic.mad
    } else {
        1.0e-9 * diagnostic.recent_median.max(1.0)
    };
    if !allowed_deviation.is_finite()
        || (diagnostic.projected_scale - diagnostic.recent_median).abs() > allowed_deviation
    {
        return None;
    }
    Some((
        diagnostic.projected_scale,
        diagnostic.recent_median,
        diagnostic.mad,
        diagnostic.sample_count,
        diagnostic.predicted_velocity,
    ))
}

/// Compose a recovered previous-to-current unit-translation pose with an
/// existing world-to-camera pose.  Keeping this convention in one helper
/// makes the fallback's direction/scale operation explicit and gives the
/// synthetic tests a small, pure target.
pub(super) fn compose_sequence_relative_pose(
    previous: &Pose,
    rotation: UnitQuaternion<f64>,
    translation_unit: Vector3<f64>,
    translation_scale: f64,
) -> Option<Pose> {
    if !translation_scale.is_finite()
        || translation_scale <= 0.0
        || !rotation.coords.iter().all(|value| value.is_finite())
        || !translation_unit.iter().all(|value| value.is_finite())
        || translation_unit.norm_squared() <= 1.0e-24
    {
        return None;
    }
    let relative =
        visloc_core::geometry::SE3::new(rotation, translation_unit.normalize() * translation_scale);
    let world_to_camera = relative.compose(&previous.world_to_camera);
    Some(Pose::from_world_to_camera(
        world_to_camera.rotation,
        world_to_camera.translation,
    ))
}

/// Convert a recovered two-view translation direction into a world-frame
/// camera-centre direction without introducing a metric scale.  The reverse
/// pair orientation is handled by inverting the relative transform, matching
/// the composition used by the fallback itself.
fn sequence_relative_world_translation_direction(
    previous: &Pose,
    rotation: UnitQuaternion<f64>,
    translation_unit: Vector3<f64>,
    pair_image_i_is_previous: bool,
) -> Option<Vector3<f64>> {
    let translation_norm = translation_unit.norm();
    if !translation_unit.iter().all(|value| value.is_finite())
        || !translation_norm.is_finite()
        || translation_norm <= 1.0e-12
    {
        return None;
    }
    let relative = visloc_core::geometry::SE3::new(rotation, translation_unit / translation_norm);
    let next_pose = if pair_image_i_is_previous {
        let world_to_camera = relative.compose(&previous.world_to_camera);
        Pose::from_world_to_camera(world_to_camera.rotation, world_to_camera.translation)
    } else {
        let previous_to_next = relative.inverse();
        let world_to_camera = previous_to_next.compose(&previous.world_to_camera);
        Pose::from_world_to_camera(world_to_camera.rotation, world_to_camera.translation)
    };
    let displacement = next_pose.camera_center_world() - previous.camera_center_world();
    let norm = displacement.norm();
    if !displacement.iter().all(|value| value.is_finite()) || !norm.is_finite() || norm <= 1.0e-12 {
        None
    } else {
        Some(displacement / norm)
    }
}

/// Recover the E-supported subset for a verified pair.  Full COLMAP-style
/// verification may retain an F-winning pair as `matches` while omitting its
/// E inlier list from [`PairwiseMatches`].  The sequence fallback must not
/// feed those F-only rows to an E decomposition, so reconstruct the missing
/// subset with the same normalized Sampson gate used by the verifier.  This
/// helper is only called by the opt-in fallback; ordinary track construction
/// keeps the verified winner untouched.
fn sequence_essential_matches(
    pair: &PairwiseMatches,
    camera: &Camera,
    features: &[FeatureSet],
    max_reprojection_error_px: f64,
) -> Vec<(usize, usize)> {
    if let Some(matches) = pair.essential_matches.as_ref() {
        return matches.clone();
    }
    let Some(essential) = pair.essential_matrix.as_ref() else {
        return Vec::new();
    };
    let focal = camera
        .intrinsics()
        .map(|(fx, fy, _, _)| 0.5 * (fx + fy))
        .filter(|value| value.is_finite() && *value > 0.0)
        .unwrap_or(1.0);
    let threshold = (max_reprojection_error_px / focal).abs();
    if !threshold.is_finite() || threshold <= 0.0 {
        return Vec::new();
    }
    pair.matches
        .iter()
        .copied()
        .filter(|&(keypoint_i, keypoint_j)| {
            let Some(point_i) = features
                .get(pair.image_i)
                .and_then(|set| set.keypoints.get(keypoint_i))
            else {
                return false;
            };
            let Some(point_j) = features
                .get(pair.image_j)
                .and_then(|set| set.keypoints.get(keypoint_j))
            else {
                return false;
            };
            normalized_sampson_residual(camera, essential, point_i, point_j)
                .is_some_and(|residual| residual <= threshold)
        })
        .collect()
}

/// Check the final triangulation admission for a sequence-relative pose.
/// Ordinary sequence edges retain the historical half-support requirement.
/// Only a pair explicitly marked by the caller as having passed the narrow
/// high-support F→E override may use the evidence-backed 100-point / 30%
/// floor.  The minimum seed support is still enforced in both modes.
pub(super) const fn sequence_triangulation_admission_ok(
    triangulated_points: usize,
    selected_matches: usize,
    min_seed_matches: usize,
    high_support_override: bool,
) -> bool {
    if triangulated_points < min_seed_matches {
        return false;
    }
    if high_support_override {
        // Use integer arithmetic so the boundary is deterministic and does
        // not depend on a platform's floating-point division/rounding.
        triangulated_points >= 100
            && (triangulated_points as u128) * 10 >= (selected_matches as u128) * 3
    } else {
        (triangulated_points as u128) * 2 >= selected_matches as u128
    }
}

macro_rules! log_sequence_fallback {
    ($($arg:tt)*) => {
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug-sequence-fallback: {}",
                format_args!($($arg)*)
            );
        }
    };
}

/// Find and validate a relative-pose registration for a numerically
/// consecutive image whose immediate predecessor is already registered.  The
/// function is deliberately independent of the PnP ranking path: it is called
/// only after normal selection/PnP cannot make progress, and it never changes
/// a pose itself.  Pair records are ranked by essential support and then by
/// their stable input index; a candidate must have a finite E, hardened
/// cheirality/parallax, and enough individually triangulatable correspondences
/// under the ordinary reprojection gate.
#[allow(clippy::too_many_arguments)]
pub(super) fn sequence_relative_pose_fallback_with_overrides(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    sequence_override_pair_indices: Option<&[usize]>,
) -> Option<SequenceRelativePoseProposal> {
    if !config.sequence_relative_pose_fallback {
        return None;
    }
    let Some(stem_values) = config.sequence_stem_values.as_deref() else {
        log_sequence_fallback!("rejected reason=stem_values_missing");
        return None;
    };
    if stem_values.len() != features.len() || poses.len() != features.len() {
        log_sequence_fallback!(
            "rejected reason=stem_values_length_mismatch stems={} features={} poses={}",
            stem_values.len(),
            features.len(),
            poses.len()
        );
        return None;
    }
    let mut unique_stems = HashSet::with_capacity(stem_values.len());
    if !stem_values
        .iter()
        .copied()
        .all(|stem| unique_stems.insert(stem))
    {
        log_sequence_fallback!("rejected reason=duplicate_stem_values");
        return None;
    }
    let Some((median_translation_scale, scale_mad, scale_samples)) =
        robust_recent_consecutive_step_scale(poses, stem_values)
    else {
        log_sequence_fallback!(
            "rejected reason=scale_history_insufficient_or_invalid registered={} ",
            poses.iter().filter(|pose| pose.is_some()).count()
        );
        return None;
    };
    log_sequence_fallback!(
        "scale_history scale={:.6e} mad={:.6e} samples={}",
        median_translation_scale,
        scale_mad,
        scale_samples
    );

    let mut image_order: Vec<(u64, usize)> = stem_values
        .iter()
        .copied()
        .enumerate()
        .map(|(image, stem)| (stem, image))
        .collect();
    image_order.sort_unstable_by_key(|&(stem, image)| (stem, image));

    for &(next_stem, next_image) in &image_order {
        if poses[next_image].is_some() {
            continue;
        }
        let Some(previous_stem) = next_stem.checked_sub(1) else {
            log_sequence_fallback!(
                "image={} stem={} rejected reason=no_predecessor_stem",
                next_image,
                next_stem
            );
            continue;
        };
        let Some(&(_, previous_image)) =
            image_order.iter().find(|&&(stem, _)| stem == previous_stem)
        else {
            log_sequence_fallback!(
                "image={} stem={} previous_stem={} rejected reason=predecessor_image_missing",
                next_image,
                next_stem,
                previous_stem
            );
            continue;
        };
        let Some(previous_pose) = poses[previous_image].as_ref() else {
            log_sequence_fallback!(
                "image={} stem={} previous_image={} previous_stem={} rejected reason=predecessor_unregistered",
                next_image,
                next_stem,
                previous_image,
                previous_stem
            );
            continue;
        };

        let mut pair_lookup_count = 0usize;
        let mut pair_missing_model_count = 0usize;
        let mut pair_bad_config_count = 0usize;
        let mut pair_low_support_count = 0usize;
        let mut candidates: Vec<(usize, usize)> = pairwise
            .iter()
            .enumerate()
            .filter_map(|(pair_index, pair)| {
                let joins_requested_images = (pair.image_i == previous_image
                    && pair.image_j == next_image)
                    || (pair.image_i == next_image && pair.image_j == previous_image);
                if !joins_requested_images || pair.essential_matrix.is_none() {
                    if joins_requested_images {
                        pair_lookup_count += 1;
                        if pair.essential_matrix.is_none() {
                            pair_missing_model_count += 1;
                        }
                    }
                    return None;
                }
                pair_lookup_count += 1;
                // A homography-only/degenerate record is not a stable E edge,
                // even if a stale matrix field was retained by a diagnostic
                // import. `Uncalibrated` and `Calibrated` both represent
                // non-planar epipolar configurations in this crate's enum.
                if matches!(
                    pair.two_view_config,
                    Some(
                        ConfigurationType::Undefined
                            | ConfigurationType::Degenerate
                            | ConfigurationType::Planar
                            | ConfigurationType::Panoramic
                            | ConfigurationType::PlanarOrPanoramic
                            | ConfigurationType::Watermark
                    )
                ) {
                    pair_bad_config_count += 1;
                    return None;
                }
                let support = if let Some(matches) = pair.essential_matches.as_ref() {
                    matches.len()
                } else {
                    // The actual E-supported rows are recovered below after the
                    // candidate has been selected.  Use the winning verified
                    // count here only to retain a cheap candidate prefilter.
                    pair.matches.len()
                };
                if support < config.min_seed_matches {
                    pair_low_support_count += 1;
                    return None;
                }
                Some((pair_index, support))
            })
            .collect();
        if candidates.is_empty() {
            log_sequence_fallback!(
                "image={} stem={} previous_image={} rejected reason=no_candidate_pair lookup={} missing_model={} bad_config={} low_support={} min_support={}",
                next_image,
                next_stem,
                previous_image,
                pair_lookup_count,
                pair_missing_model_count,
                pair_bad_config_count,
                pair_low_support_count,
                config.min_seed_matches
            );
            continue;
        }
        log_sequence_fallback!(
            "image={} stem={} previous_image={} candidate_pairs={} lookup={} missing_model={} bad_config={} low_support={}",
            next_image,
            next_stem,
            previous_image,
            candidates.len(),
            pair_lookup_count,
            pair_missing_model_count,
            pair_bad_config_count,
            pair_low_support_count
        );
        candidates.sort_by(|(left_index, left_support), (right_index, right_support)| {
            right_support
                .cmp(left_support)
                .then_with(|| left_index.cmp(right_index))
        });

        for (pair_index, pair_support) in candidates {
            let pair = &pairwise[pair_index];
            let high_support_override =
                sequence_override_pair_indices.is_some_and(|indices| indices.contains(&pair_index));
            let selected_matches = sequence_essential_matches(
                pair,
                camera,
                features,
                config.max_reprojection_error_px,
            );
            let mut correspondences = Vec::with_capacity(selected_matches.len());
            let mut pixels = Vec::with_capacity(selected_matches.len());
            for (ki, kj) in selected_matches {
                let (Some(pi), Some(pj)) = (
                    features[pair.image_i].keypoints.get(ki).copied(),
                    features[pair.image_j].keypoints.get(kj).copied(),
                ) else {
                    continue;
                };
                correspondences.push(TwoViewCorrespondence::new(pi, pj));
                pixels.push((pi, pj));
            }
            if correspondences.len() < config.min_seed_matches {
                log_sequence_fallback!(
                    "image={} stem={} pair={} support_hint={} rejected reason=essential_matches_below_min selected={} min_support={}",
                    next_image,
                    next_stem,
                    pair_index,
                    pair_support,
                    correspondences.len(),
                    config.min_seed_matches
                );
                continue;
            }
            let Some(essential) = pair.essential_matrix.as_ref() else {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=essential_matrix_not_stored_after_candidate",
                    next_image,
                    next_stem,
                    pair_index
                );
                continue;
            };
            if !essential.iter().all(|value| value.is_finite()) {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=essential_matrix_nonfinite",
                    next_image,
                    next_stem,
                    pair_index
                );
                continue;
            }
            log_sequence_fallback!(
                "image={} stem={} pair={} model=stored_essential direction={}-{} selected_matches={}",
                next_image,
                next_stem,
                pair_index,
                pair.image_i,
                pair.image_j,
                correspondences.len()
            );
            let inlier_indices: Vec<usize> = (0..correspondences.len()).collect();
            let Some(recovered) = recover_relative_pose_with_options(
                essential,
                &correspondences,
                camera,
                &inlier_indices,
                &CheiralityOptions::hardened(),
            ) else {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=relative_pose_recovery_failed selected={}",
                    next_image,
                    next_stem,
                    pair_index,
                    correspondences.len()
                );
                continue;
            };
            let required_support = config.min_seed_matches.max(8) as i64;
            if recovered.best_score < required_support
                || recovered.best_score * 2 < correspondences.len() as i64
            {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=cheirality_support best={} second={} selected={} required={} margin={:.6}",
                    next_image,
                    next_stem,
                    pair_index,
                    recovered.best_score,
                    recovered.second_score,
                    correspondences.len(),
                    required_support,
                    recovered.chirality_margin()
                );
                continue;
            }

            let mut projected_scale_diagnostic = None;
            let translation_scale = if config.sequence_constant_velocity_scale
                || config.sequence_relaxed_constant_velocity_scale
            {
                let Some(candidate_direction) = sequence_relative_world_translation_direction(
                    previous_pose,
                    recovered.rotation,
                    recovered.translation_unit,
                    pair.image_i == previous_image,
                ) else {
                    log_sequence_fallback!(
                        "image={} stem={} pair={} rejected reason=projected_scale_direction_invalid median_scale={:.6e}",
                        next_image,
                        next_stem,
                        pair_index,
                        median_translation_scale
                    );
                    continue;
                };
                let Some(diagnostic) = projected_recent_consecutive_step_scale_diagnostic(
                    poses,
                    stem_values,
                    previous_stem,
                    candidate_direction,
                ) else {
                    log_sequence_fallback!(
                        "image={} stem={} pair={} rejected reason=projected_scale_history_invalid median_scale={:.6e} direction=({:.6e},{:.6e},{:.6e})",
                        next_image,
                        next_stem,
                        pair_index,
                        median_translation_scale,
                        candidate_direction.x,
                        candidate_direction.y,
                        candidate_direction.z
                    );
                    continue;
                };
                let strict_projection_valid = {
                    let allowed_deviation =
                        if diagnostic.mad.is_finite() && diagnostic.mad > 1.0e-12 {
                            3.0 * diagnostic.mad
                        } else {
                            1.0e-9 * diagnostic.recent_median.max(1.0)
                        };
                    diagnostic.projected_scale.is_finite()
                        && diagnostic.projected_scale > 0.0
                        && allowed_deviation.is_finite()
                        && (diagnostic.projected_scale - diagnostic.recent_median).abs()
                            <= allowed_deviation
                };
                let relaxed_projection_valid = relaxed_projected_scale_is_valid(
                    diagnostic.projected_scale,
                    diagnostic.recent_median,
                );
                let projection_valid = if config.sequence_relaxed_constant_velocity_scale {
                    relaxed_projection_valid
                } else {
                    strict_projection_valid
                };
                if !projection_valid {
                    log_sequence_fallback!(
                        "image={} stem={} pair={} rejected reason=projected_scale_invalid policy={} median_scale={:.6e} anchored_median={:.6e} mad={:.6e} projected_scale={:.6e} broad_bounds=({:.6e},{:.6e}) velocity=({:.6e},{:.6e},{:.6e}) samples={} direction=({:.6e},{:.6e},{:.6e})",
                        next_image,
                        next_stem,
                        pair_index,
                        if config.sequence_relaxed_constant_velocity_scale {
                            "relaxed"
                        } else {
                            "strict"
                        },
                        median_translation_scale,
                        diagnostic.recent_median,
                        diagnostic.mad,
                        diagnostic.projected_scale,
                        0.25 * diagnostic.recent_median,
                        4.0 * diagnostic.recent_median,
                        diagnostic.predicted_velocity.x,
                        diagnostic.predicted_velocity.y,
                        diagnostic.predicted_velocity.z,
                        diagnostic.sample_count,
                        candidate_direction.x,
                        candidate_direction.y,
                        candidate_direction.z
                    );
                    continue;
                }
                log_sequence_fallback!(
                    "image={} stem={} pair={} scale_projection policy={} median_scale={:.6e} anchored_median={:.6e} mad={:.6e} projected_scale={:.6e} broad_bounds=({:.6e},{:.6e}) velocity=({:.6e},{:.6e},{:.6e}) samples={}",
                    next_image,
                    next_stem,
                    pair_index,
                    if config.sequence_relaxed_constant_velocity_scale {
                        "relaxed"
                    } else {
                        "strict"
                    },
                    median_translation_scale,
                    diagnostic.recent_median,
                    diagnostic.mad,
                    diagnostic.projected_scale,
                    0.25 * diagnostic.recent_median,
                    4.0 * diagnostic.recent_median,
                    diagnostic.predicted_velocity.x,
                    diagnostic.predicted_velocity.y,
                    diagnostic.predicted_velocity.z,
                    diagnostic.sample_count
                );
                projected_scale_diagnostic = Some(diagnostic.projected_scale);
                diagnostic.projected_scale
            } else {
                median_translation_scale
            };

            let Some(next_pose) = (if pair.image_i == previous_image {
                compose_sequence_relative_pose(
                    previous_pose,
                    recovered.rotation,
                    recovered.translation_unit,
                    translation_scale,
                )
            } else {
                let relative = visloc_core::geometry::SE3::new(
                    recovered.rotation,
                    recovered.translation_unit.normalize() * translation_scale,
                );
                let previous_to_next = relative.inverse();
                let world_to_camera = previous_to_next.compose(&previous_pose.world_to_camera);
                Some(Pose::from_world_to_camera(
                    world_to_camera.rotation,
                    world_to_camera.translation,
                ))
            }) else {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=composed_pose_nonfinite_or_invalid",
                    next_image,
                    next_stem,
                    pair_index
                );
                continue;
            };
            if !next_pose
                .world_to_camera
                .translation
                .iter()
                .all(|v| v.is_finite())
                || !next_pose
                    .world_to_camera
                    .rotation
                    .coords
                    .iter()
                    .all(|v| v.is_finite())
            {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=composed_pose_nonfinite",
                    next_image,
                    next_stem,
                    pair_index
                );
                continue;
            }
            let mut candidate_poses = poses.to_vec();
            candidate_poses[next_image] = Some(next_pose.clone());

            let mut triangulated_points = 0usize;
            for &inlier in &inlier_indices {
                let (pi, pj) = pixels[inlier];
                let (previous_px, next_px) = if pair.image_i == previous_image {
                    (pi, pj)
                } else {
                    (pj, pi)
                };
                if triangulate_track(
                    camera,
                    &candidate_poses,
                    &[(previous_image, previous_px), (next_image, next_px)],
                    config,
                )
                .is_some()
                {
                    triangulated_points += 1;
                }
            }
            let valid_ratio = if correspondences.is_empty() {
                0.0
            } else {
                triangulated_points as f64 / correspondences.len() as f64
            };
            let admission_required = if high_support_override {
                config.min_seed_matches.max(100)
            } else {
                config.min_seed_matches
            };
            if !sequence_triangulation_admission_ok(
                triangulated_points,
                correspondences.len(),
                config.min_seed_matches,
                high_support_override,
            ) {
                log_sequence_fallback!(
                    "image={} stem={} pair={} rejected reason=triangulation_admission mode={} selected={} cheirality_best={} triangulated={} valid_ratio={:.6} required={} min_ratio={:.2}",
                    next_image,
                    next_stem,
                    pair_index,
                    if high_support_override {
                        "high_support_override"
                    } else {
                        "standard"
                    },
                    correspondences.len(),
                    recovered.best_score,
                    triangulated_points,
                    valid_ratio,
                    admission_required,
                    if high_support_override { 0.30 } else { 0.50 },
                );
                continue;
            }
            log_sequence_fallback!(
                "image={} stem={} pair={} admitted mode={} scale_mode={} selected={} cheirality_best={} triangulated={} valid_ratio={:.6} required={} median_scale={:.6e} projected_scale={:?} scale={:.6e}",
                next_image,
                next_stem,
                pair_index,
                if high_support_override {
                    "high_support_override"
                } else {
                    "standard"
                },
                if projected_scale_diagnostic.is_some() {
                    if config.sequence_relaxed_constant_velocity_scale {
                        "constant_velocity_projected_relaxed"
                    } else {
                        "constant_velocity_projected"
                    }
                } else {
                    "median_magnitude"
                },
                correspondences.len(),
                recovered.best_score,
                triangulated_points,
                valid_ratio,
                admission_required,
                median_translation_scale,
                projected_scale_diagnostic,
                translation_scale
            );
            return Some(SequenceRelativePoseProposal {
                next_image,
                previous_image,
                pair_index,
                pair_inliers: pair_support,
                triangulated_points,
                triangulation_candidates: correspondences.len(),
                translation_scale,
                translation_scale_median: median_translation_scale,
                translation_scale_projection: projected_scale_diagnostic,
                translation_scale_carried: false,
                chirality_margin: recovered.chirality_margin(),
                pose: next_pose,
            });
        }
    }
    None
}

#[allow(clippy::too_many_arguments)]
pub(super) fn commit_sequence_relative_pose(
    proposal: SequenceRelativePoseProposal,
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    correspondence_state: &mut Option<CorrespondencePointState>,
    triangulation_seconds: &mut f64,
    registrations_since_ba: &mut usize,
) {
    let image = proposal.next_image;
    poses[image] = Some(proposal.pose);
    let started = std::time::Instant::now();
    triangulate_pending_with_config_and_state(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        correspondence_state.as_mut(),
    );
    *triangulation_seconds += started.elapsed().as_secs_f64();
    *registrations_since_ba += 1;
    if sfm_debug_enabled() {
        eprintln!(
                "sfm-debug: sequence fallback registered image {} from previous {} \
                 pair={} inliers={} triangulated={}/{} ratio={:.6} scale_mode={} median_scale={:.6e} projected_scale={:?} scale={:.6e} chirality_margin={:.3}",
                image,
                proposal.previous_image,
                proposal.pair_index,
                proposal.pair_inliers,
                proposal.triangulated_points,
                proposal.triangulation_candidates,
                proposal.triangulated_points as f64
                    / proposal.triangulation_candidates.max(1) as f64,
                if proposal.translation_scale_projection.is_some() {
                    if config.sequence_relaxed_constant_velocity_scale {
                        "constant_velocity_projected_relaxed"
                    } else {
                        "constant_velocity_projected"
                    }
                } else {
                    "median_magnitude"
                },
                proposal.translation_scale_median,
                proposal.translation_scale_projection,
                proposal.translation_scale,
                proposal.chirality_margin,
        );
    }
}

/// Emit the track-level inputs to one selected PnP attempt. This is deliberately
/// diagnostic-only: the same retained tracks and `track_point` values used to
/// build `corrs` are summarized without changing ranking or registration.
#[allow(clippy::too_many_arguments)]
pub(super) fn log_registration_track_provenance(
    image: usize,
    corr_count: usize,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    conflicting_components: &[Vec<(usize, usize)>],
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    debug_image_filter: Option<&HashSet<usize>>,
) {
    if !sfm_debug_image_enabled(image, debug_image_filter) {
        return;
    }
    let mut triangulated_track_ids = HashSet::new();
    let mut track_lengths = BTreeMap::<usize, usize>::new();
    let mut registered_support = BTreeMap::<usize, usize>::new();
    for &(keypoint, track_id) in &obs_by_image[image] {
        if track_point.get(track_id).is_none_or(Option::is_none)
            || features[image].keypoints.get(keypoint).is_none()
            || !triangulated_track_ids.insert(track_id)
        {
            continue;
        }
        let Some(track) = tracks.get(track_id) else {
            continue;
        };
        *track_lengths.entry(track.len()).or_default() += 1;
        for &(support_image, _) in track {
            if poses.get(support_image).is_some_and(Option::is_some) {
                *registered_support.entry(support_image).or_default() += 1;
            }
        }
    }

    let mut conflict_components = 0usize;
    let mut conflict_observations = 0usize;
    for component in conflicting_components {
        let target_observations = component
            .iter()
            .filter(|&&(component_image, _)| component_image == image)
            .count();
        if target_observations > 0 {
            conflict_components += 1;
            conflict_observations += target_observations;
        }
    }

    eprintln!(
        "sfm-debug: PnP provenance image={image} triangulated_tracks={} corrs={corr_count} \
         track_len={track_lengths:?} registered_support={registered_support:?} \
         conflict_components={conflict_components} conflict_observations={conflict_observations}",
        triangulated_track_ids.len(),
    );
}

/// Recreate the deterministic track-id order used by [`select_next_image`].
/// `select_next_image` intentionally returns the public PnP correspondence
/// shape, so this small debug-only join keeps track provenance out of the
/// normal PnP API while allowing the registration diagnostic to inspect the
/// exact same rows.
pub(super) fn pnp_track_ids(
    image: usize,
    corrs: &[Correspondence2D3D],
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
) -> Vec<usize> {
    let ids: Vec<usize> = obs_by_image
        .get(image)
        .into_iter()
        .flatten()
        .filter_map(|&(keypoint, track_id)| {
            track_point.get(track_id).and_then(|point| *point)?;
            features
                .get(image)
                .and_then(|feature_set| feature_set.keypoints.get(keypoint))
                .map(|_| track_id)
        })
        .collect();
    debug_assert_eq!(ids.len(), corrs.len());
    ids
}

fn pnp_geometry_median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values
        .get(values.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::NAN)
}

/// Summarise the exact 2D--3D rows offered to one PnP solve.  This is gated by
/// `VISLOC_SFM_DEBUG_IMAGES` and the optional oracle vector, so it is an
/// offline diagnostic rather than a registration policy.  The final block
/// also refines a deterministic, high-information subset (long tracks and at
/// least the inlier-angle median) solely to compare its pose basin with the
/// all-inlier result; that pose is never written back.
#[allow(clippy::too_many_arguments)]
pub(super) fn log_pnp_geometry_diagnostic(
    image: usize,
    corrs: &[Correspondence2D3D],
    inliers: &[usize],
    track_ids: &[usize],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    camera: &Camera,
    candidate_pose: &Pose,
    oracle: Option<&[Option<Pose>]>,
    debug_image_filter: Option<&HashSet<usize>>,
) {
    if !sfm_debug_image_enabled(image, debug_image_filter)
        || track_ids.len() != corrs.len()
        || corrs.is_empty()
    {
        return;
    }
    let inlier_set: HashSet<usize> = inliers.iter().copied().collect();
    let mut rows = Vec::with_capacity(corrs.len());
    for (index, corr) in corrs.iter().enumerate() {
        let Some(track) = track_ids
            .get(index)
            .and_then(|track_id| tracks.get(*track_id))
        else {
            continue;
        };
        let angle = track_max_parallax(poses, track, &corr.point3d);
        let condition = if angle.is_finite() {
            1.0 / angle.sin().abs().max(1.0e-9)
        } else {
            f64::NAN
        };
        let reprojection =
            reprojection_error_px(camera, candidate_pose, &corr.point3d, &corr.point2d)
                .unwrap_or(f64::NAN);
        rows.push((
            index,
            track.len(),
            angle.to_degrees(),
            condition,
            reprojection,
            inlier_set.contains(&index),
        ));
    }
    let summarize = |only_inliers: bool| {
        let selected: Vec<_> = rows.iter().filter(|row| !only_inliers || row.5).collect();
        let mut lengths: Vec<f64> = selected.iter().map(|row| row.1 as f64).collect();
        let mut angles: Vec<f64> = selected
            .iter()
            .map(|row| row.2)
            .filter(|value| value.is_finite())
            .collect();
        let mut conditions: Vec<f64> = selected
            .iter()
            .map(|row| row.3)
            .filter(|value| value.is_finite())
            .collect();
        let mut reprojections: Vec<f64> = selected
            .iter()
            .map(|row| row.4)
            .filter(|value| value.is_finite())
            .collect();
        (
            selected.len(),
            pnp_geometry_median(&mut lengths),
            pnp_geometry_median(&mut angles),
            pnp_geometry_median(&mut conditions),
            pnp_geometry_median(&mut reprojections),
        )
    };
    let all = summarize(false);
    let accepted = summarize(true);
    let mut inlier_angles: Vec<f64> = rows
        .iter()
        .filter(|row| row.5 && row.2.is_finite())
        .map(|row| row.2)
        .collect();
    let angle_median = pnp_geometry_median(&mut inlier_angles);
    let mut subset_indices: Vec<usize> = rows
        .iter()
        .filter(|row| row.5 && row.1 >= 3 && row.2.is_finite() && row.2 >= angle_median)
        .map(|row| row.0)
        .collect();
    if subset_indices.len() < 6 {
        let mut ranked: Vec<_> = rows.iter().filter(|row| row.5).collect();
        ranked.sort_by(|a, b| {
            b.1.cmp(&a.1)
                .then_with(|| b.2.total_cmp(&a.2))
                .then_with(|| a.0.cmp(&b.0))
        });
        subset_indices = ranked
            .into_iter()
            .take(6.max(subset_indices.len()))
            .map(|row| row.0)
            .collect();
    }
    subset_indices.sort_unstable();
    eprintln!(
        concat!(
            "sfm-debug-pnp-geometry: image={} rows={} inliers={} ",
            "all_len_med={:.2} inlier_len_med={:.2} ",
            "all_angle_med={:.3}deg inlier_angle_med={:.3}deg ",
            "all_condition_med={:.2} inlier_condition_med={:.2} ",
            "all_reproj_med={:.3}px inlier_reproj_med={:.3}px ",
            "high_info_subset={}"
        ),
        image,
        all.0,
        accepted.0,
        all.1,
        accepted.1,
        all.2,
        accepted.2,
        all.3,
        accepted.3,
        all.4,
        accepted.4,
        subset_indices.len(),
    );

    let Some(oracle) = oracle else { return };
    if subset_indices.len() < 6 {
        eprintln!(
            "sfm-debug-pnp-geometry: image={} high_info_subset<6; pose comparison skipped",
            image
        );
        return;
    }
    let subset_corrs: Vec<Correspondence2D3D> = subset_indices
        .iter()
        .filter_map(|&index| corrs.get(index).cloned())
        .collect();
    if subset_corrs.len() < 6 {
        return;
    }
    let Some(subset_pose) =
        GaussNewtonPoseRefiner::default().refine_pose(candidate_pose, &subset_corrs, camera)
    else {
        eprintln!(
            "sfm-debug-pnp-geometry: image={} high_info_subset refinement failed",
            image
        );
        return;
    };
    let all_metrics = sfm_oracle_metrics(poses, oracle);
    let mut subset_poses = poses.to_vec();
    subset_poses[image] = Some(subset_pose);
    let subset_metrics = sfm_oracle_metrics(&subset_poses, oracle);
    let (Some(all_metrics), Some(subset_metrics)) = (all_metrics, subset_metrics) else {
        eprintln!(
            "sfm-debug-pnp-geometry: image={} high_info_subset oracle comparison unavailable",
            image
        );
        return;
    };
    let all_error = all_metrics.center_errors[image].map(|value| value * 100.0);
    let subset_error = subset_metrics.center_errors[image].map(|value| value * 100.0);
    let all_rotation = all_metrics.rotation_errors[image];
    let subset_rotation = subset_metrics.rotation_errors[image];
    eprintln!(
        "sfm-debug-pnp-geometry: image={} high_info_subset={} target_center_cm={:?}->{:?} delta_cm={:?} target_rotation_deg={:?}->{:?}",
        image,
        subset_corrs.len(),
        all_error,
        subset_error,
        all_error.zip(subset_error).map(|(before, after)| after - before),
        all_rotation,
        subset_rotation,
    );
}
