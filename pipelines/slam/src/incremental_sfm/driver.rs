//! Public entry points and the top-level seed-search / growth / refinement driver.

use super::*;

/// A grown reconstruction the seed search compares: how many images it
/// registered, the per-image poses and the per-track points.
type SeedGrowth = (usize, Vec<Option<Pose>>, Vec<Option<Point3<f64>>>, Camera);

/// Run incremental SfM over an unordered image set.
///
/// `features[k]` are the keypoints + descriptors of image `k`; `pairwise` are
/// the geometrically verified matches between image pairs. Returns the refined
/// poses and merged tracks, or an [`IncrementalSfmError`] if no reconstruction
/// could be bootstrapped.
pub fn incremental_sfm(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    incremental_sfm_with_initial_poses_and_track_membership(
        camera, features, pairwise, config, None, None,
    )
}

/// Run incremental SfM with a sequence-fallback admission exception for a
/// caller-provided set of pair entries.  The pair indices must refer to the
/// supplied `pairwise` slice.  Only entries selected by the caller's
/// conservative high-support F→E promotion are eligible for the relaxed
/// triangulation fraction; every other sequence edge keeps the ordinary gate.
///
/// This is deliberately a separate opt-in entry point.  The ordinary
/// [`incremental_sfm`] and [`incremental_sfm_with_initial_poses`] paths do not
/// carry this metadata and therefore retain their existing behavior exactly.
pub fn incremental_sfm_with_sequence_fallback_overrides(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    high_support_override_pair_indices: &[usize],
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    incremental_sfm_with_initial_poses_and_track_membership_and_sequence_overrides(
        camera,
        features,
        pairwise,
        config,
        None,
        None,
        Some(high_support_override_pair_indices),
    )
}

/// Run incremental SfM from an externally supplied, partial pose model.
///
/// `initial_poses` is indexed like `features`; `None` entries are the images
/// that the ordinary PnP growth loop must register.  The supplied poses are
/// copied into the initial reconstruction and are held fixed while tracks are
/// triangulated and missing images are grown.  All poses become ordinary BA
/// variables once the initial growth phase returns, subject only to the
/// existing gauge anchors.  This is deliberately a separate opt-in entry
/// point so [`incremental_sfm`] remains byte-for-byte equivalent when no seed
/// model is supplied.
pub fn incremental_sfm_with_initial_poses(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    initial_poses: Option<&[Option<Pose>]>,
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    incremental_sfm_with_initial_poses_and_track_membership(
        camera,
        features,
        pairwise,
        config,
        initial_poses,
        None,
    )
}

/// Run the plain incremental mapper with an externally supplied set of
/// observation partitions instead of constructing tracks from pairwise
/// correspondences.
///
/// This is an explicit diagnostic/oracle entry point.  Each input track is a
/// list of `(image_index, keypoint_index)` observations; the mapper ignores
/// any oracle point coordinates, colors, errors, or camera poses and
/// re-triangulates the supplied membership from the current feature pixels
/// and intrinsics.  The ordinary [`incremental_sfm`] path remains unchanged
/// when this function is not called.
pub fn incremental_sfm_with_track_membership(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    track_membership: &[Vec<(usize, usize)>],
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    incremental_sfm_with_initial_poses_and_track_membership(
        camera,
        features,
        pairwise,
        config,
        None,
        Some(track_membership),
    )
}

/// Internal implementation shared by the ordinary, initial-pose, and
/// track-membership diagnostic entry points.
fn incremental_sfm_with_initial_poses_and_track_membership(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    initial_poses: Option<&[Option<Pose>]>,
    track_membership: Option<&[Vec<(usize, usize)>]>,
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    incremental_sfm_with_initial_poses_and_track_membership_and_sequence_overrides(
        camera,
        features,
        pairwise,
        config,
        initial_poses,
        track_membership,
        None,
    )
}

/// Internal implementation shared by the ordinary and opt-in sequence-aware
/// entry points.  `sequence_override_pair_indices` is intentionally kept out
/// of [`IncrementalSfmConfig`]: it is ephemeral metadata produced by the
/// example's post-verification F→E promotion and must not become a persisted
/// mapper setting or alter any other caller's struct literals.
fn incremental_sfm_with_initial_poses_and_track_membership_and_sequence_overrides(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    initial_poses: Option<&[Option<Pose>]>,
    track_membership: Option<&[Vec<(usize, usize)>]>,
    sequence_override_pair_indices: Option<&[usize]>,
) -> Result<IncrementalSfmResult, IncrementalSfmError> {
    // `Auto` deliberately reruns only the mapper state.  `features` and
    // `pairwise` stay borrowed and immutable, while each candidate gets its
    // own small configuration/state allocations.  This gives both policies
    // the same seed, track input, and initial poses without cloning the large
    // feature/descriptor banks.
    if config.next_image_policy == NextImagePolicy::Auto {
        let auto_started = std::time::Instant::now();
        let mut visibility_config = config.clone();
        visibility_config.next_image_policy = NextImagePolicy::VisibilityPyramid;
        let visibility_started = std::time::Instant::now();
        let visibility =
            incremental_sfm_with_initial_poses_and_track_membership_and_sequence_overrides(
                camera,
                features,
                pairwise,
                &visibility_config,
                initial_poses,
                track_membership,
                sequence_override_pair_indices,
            );
        let visibility_elapsed = visibility_started.elapsed().as_secs_f64();
        let visibility_complete = visibility.as_ref().is_ok_and(|result| {
            !next_image_auto_count_candidate_is_needed(result.registered_images, features.len())
        });

        // Visibility is intentionally the primary policy when it is complete.
        // For an incomplete result, run the count candidate as well even when
        // the missing fraction is small: a complete count candidate must win
        // before the post-refinement completion pass is considered.  This
        // avoids turning a numerically fragile visibility candidate into a
        // complete but inaccurate model merely because it was only one image
        // short.
        let (selected, selected_policy, count_elapsed) = if visibility_complete {
            if sfm_debug_enabled() {
                if let Ok(result) = &visibility {
                    let metrics = next_image_auto_metrics(result);
                    eprintln!(
                        "sfm-auto: visibility primary registered={}/{} observations={} tracks={} reproj={:.6} elapsed={:.3}s count=skipped total={:.3}s",
                        metrics.registered_images,
                        features.len(),
                        metrics.valid_observations,
                        metrics.tracks,
                        metrics.mean_reprojection_px,
                        visibility_elapsed,
                        auto_started.elapsed().as_secs_f64(),
                    );
                }
            }
            (visibility, NextImagePolicy::VisibilityPyramid, 0.0)
        } else {
            let mut count_config = config.clone();
            count_config.next_image_policy = NextImagePolicy::CorrespondenceCount;
            let count_started = std::time::Instant::now();
            let count =
                incremental_sfm_with_initial_poses_and_track_membership_and_sequence_overrides(
                    camera,
                    features,
                    pairwise,
                    &count_config,
                    initial_poses,
                    track_membership,
                    sequence_override_pair_indices,
                );
            let count_elapsed = count_started.elapsed().as_secs_f64();

            if sfm_debug_enabled() {
                let describe =
                    |label: &str, result: &Result<IncrementalSfmResult, IncrementalSfmError>| {
                        match result {
                            Ok(result) => {
                                let metrics = next_image_auto_metrics(result);
                                eprintln!(
                            "sfm-auto: {} registered={}/{} observations={} tracks={} reproj={:.6}",
                            label,
                            metrics.registered_images,
                            features.len(),
                            metrics.valid_observations,
                            metrics.tracks,
                            metrics.mean_reprojection_px,
                            );
                            }
                            Err(error) => eprintln!("sfm-auto: {label} failed={error}"),
                        }
                    };
                describe("visibility", &visibility);
                describe("count", &count);
                eprintln!(
                "sfm-auto: elapsed visibility={visibility_elapsed:.3}s count={count_elapsed:.3}s total={:.3}s",
                auto_started.elapsed().as_secs_f64(),
                );
            }

            let selected = match (visibility, count) {
                (Ok(visibility), Ok(count)) => {
                    if next_image_auto_candidate_is_better(&count, &visibility) {
                        (Ok(count), NextImagePolicy::CorrespondenceCount)
                    } else {
                        // Exact support/reprojection ties intentionally retain
                        // the visibility-first result for stable semantics.
                        (Ok(visibility), NextImagePolicy::VisibilityPyramid)
                    }
                }
                (Ok(visibility), Err(_count_error)) => {
                    (Ok(visibility), NextImagePolicy::VisibilityPyramid)
                }
                (Err(_visibility_error), Ok(count)) => {
                    (Ok(count), NextImagePolicy::CorrespondenceCount)
                }
                (Err(visibility_error), Err(_count_error)) => return Err(visibility_error),
            };
            (selected.0, selected.1, count_elapsed)
        };

        let (selected, selected_policy) = match selected {
            Ok(selected) => (selected, selected_policy),
            Err(error) => return Err(error),
        };
        if next_image_auto_post_candidate_is_needed(selected.registered_images, features.len())
            && !config.post_refinement_registration
        {
            // Run post-refinement from the same clean selected-policy inputs,
            // rather than mutating the already-completed candidate.  This
            // keeps the fallback transactional and makes a tie byte-for-byte
            // equivalent to the pre-post candidate (not merely metric-equal).
            let mut post_config = config.clone();
            post_config.next_image_policy = selected_policy;
            post_config.post_refinement_registration = true;
            let post_started = std::time::Instant::now();
            let post =
                incremental_sfm_with_initial_poses_and_track_membership_and_sequence_overrides(
                    camera,
                    features,
                    pairwise,
                    &post_config,
                    initial_poses,
                    track_membership,
                    sequence_override_pair_indices,
                );
            match post {
                Ok(post) if next_image_auto_post_candidate_is_better(&post, &selected) => {
                    if sfm_debug_enabled() {
                        eprintln!(
                            "sfm-auto: post completion adopted policy={selected_policy:?} registered={}/{} -> {}/{} elapsed={:.3}s count_elapsed={count_elapsed:.3}s total={:.3}s",
                            selected.registered_images,
                            features.len(),
                            post.registered_images,
                            features.len(),
                            post_started.elapsed().as_secs_f64(),
                            auto_started.elapsed().as_secs_f64(),
                        );
                    }
                    return Ok(post);
                }
                Ok(post) => {
                    if sfm_debug_enabled() {
                        eprintln!(
                            "sfm-auto: post completion rejected policy={selected_policy:?} registered={}/{} -> {}/{} (strict increase and non-increasing finite reprojection required) elapsed={:.3}s total={:.3}s",
                            selected.registered_images,
                            features.len(),
                            post.registered_images,
                            features.len(),
                            post_started.elapsed().as_secs_f64(),
                            auto_started.elapsed().as_secs_f64(),
                        );
                    }
                }
                Err(error) => {
                    if sfm_debug_enabled() {
                        eprintln!(
                            "sfm-auto: post completion failed policy={selected_policy:?} error={error} (pre-post candidate retained)"
                        );
                    }
                }
            }
        }
        return Ok(selected);
    }
    let sfm_started = std::time::Instant::now();
    let n_images = features.len();
    if let Some(track_membership) = track_membership {
        for (track_id, track) in track_membership.iter().enumerate() {
            let mut images = HashSet::new();
            for &(image, keypoint) in track {
                if image >= n_images {
                    return Err(IncrementalSfmError::InvalidTrackMembership(format!(
                        "track {track_id} references image {image}, but only {n_images} images are loaded"
                    )));
                }
                if keypoint >= features[image].keypoints.len()
                    || keypoint >= features[image].descriptors.len()
                {
                    return Err(IncrementalSfmError::InvalidTrackMembership(format!(
                        "track {track_id} references image {image} keypoint {keypoint}, but the loaded feature set has {} keypoints / {} descriptors",
                        features[image].keypoints.len(),
                        features[image].descriptors.len(),
                    )));
                }
                if !images.insert(image) {
                    return Err(IncrementalSfmError::InvalidTrackMembership(format!(
                        "track {track_id} contains more than one observation from image {image}"
                    )));
                }
            }
        }
    }
    if let Some(initial_poses) = initial_poses {
        if initial_poses.len() != n_images {
            return Err(IncrementalSfmError::InvalidInitialPoses(format!(
                "expected {} pose slots, got {}",
                n_images,
                initial_poses.len()
            )));
        }
        let seeded = initial_poses.iter().filter(|pose| pose.is_some()).count();
        if seeded < 2 {
            return Err(IncrementalSfmError::InvalidInitialPoses(format!(
                "at least two finite seed poses are required, got {seeded}"
            )));
        }
        for (image, pose) in initial_poses.iter().enumerate() {
            if let Some(pose) = pose {
                let rotation = pose.world_to_camera.rotation;
                let translation = pose.world_to_camera.translation;
                if !rotation.coords.iter().all(|value| value.is_finite())
                    || !translation.iter().all(|value| value.is_finite())
                {
                    return Err(IncrementalSfmError::InvalidInitialPoses(format!(
                        "pose for image {image} contains non-finite rotation or translation"
                    )));
                }
            }
        }
    }
    let debug_image_filter = if sfm_debug_enabled() {
        sfm_debug_image_filter()
    } else {
        None
    };

    // ---- 1. Build feature tracks (M2: union-find or CorrespondenceGraph) ----
    let started = std::time::Instant::now();
    let track_build = if let Some(track_membership) = track_membership {
        build_track_output_from_membership(
            features,
            pairwise,
            config.min_track_length,
            track_membership,
        )
        .map_err(IncrementalSfmError::InvalidTrackMembership)?
    } else {
        build_track_output(features, pairwise, config, Some(camera))
    };
    let TrackBuildOutput {
        mut tracks,
        mut conflicting_components,
        stats: track_build_stats,
    } = track_build;
    let track_build_seconds = started.elapsed().as_secs_f64();
    process_memory::log("mapper-after-track-build");
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: track build source={:?} input={} components={} \
             conflicts={} conflict_obs={} retained_tracks={} retained_obs={}",
            config.track_source,
            track_build_stats.input_correspondences,
            track_build_stats.connected_components,
            track_build_stats.conflicting_components,
            track_build_stats.conflicting_observations,
            track_build_stats.retained_tracks,
            track_build_stats.retained_observations,
        );
    }

    // For each image, which (keypoint, track) pairs it observes — drives both
    // triangulation and next-image selection.
    let mut obs_by_image: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n_images];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, kp) in track {
            obs_by_image[image].push((kp, track_id));
        }
    }
    process_memory::log("mapper-after-observation-index");

    // ---- 2. Seed selection: try several candidate seeds, keep the largest ----
    // The highest-match pair is not always a good seed. On repetitive structure
    // (a building photographed around near-identical façades) the most-overlapping
    // verified pair can be a handful of adjacent frames that triangulate fine but
    // form an isolated local cluster the reconstruction cannot grow out of. So
    // walk verified pairs in descending match order and keep the reconstruction
    // that registers the most images, committing as soon as one is *not trapped*
    // — reaches at least half of its connected component. A well-connected scene
    // (the strongest pair is already central) commits on the first candidate that
    // places, growing exactly one reconstruction, just as the old
    // first-qualifying-seed path did; only a repetitive scene whose strongest
    // pairs are isolated clusters keeps searching, and then takes the
    // farthest-reaching seed found. Each grow runs its periodic BA, so reach is
    // measured on the real (bundle-adjusted) trajectory, not a drifting proxy.
    //
    // `seed_trials` caps how many pairs actually *grow* a reconstruction; pairs
    // that fail the two-view baseline gate placed nothing and are skipped for
    // free, so an orbit whose highest-overlap pairs are all low-parallax adjacent
    // frames still reaches the first wide-baseline pair beyond them.
    let seed_growth_started = std::time::Instant::now();
    let (mut poses, mut track_point, grown_cam, seed_image_i, seed_image_j, seed_match_count) =
        if let Some(initial_poses) = initial_poses {
            let (poses, track_point, reach, grown_cam) = grow_from_seed_with_sequence_overrides(
                camera,
                features,
                pairwise,
                &tracks,
                &conflicting_components,
                &obs_by_image,
                config,
                debug_image_filter.as_ref(),
                None,
                Some(initial_poses),
                sequence_override_pair_indices,
            )?;
            if reach == 0 {
                return Err(IncrementalSfmError::NoSeedPair);
            }
            let seeded_images: Vec<usize> = initial_poses
                .iter()
                .enumerate()
                .filter_map(|(image, pose)| pose.as_ref().map(|_| image))
                .collect();
            let seed_image_i = seeded_images[0];
            let seed_image_j = seeded_images[1];
            let seed_match_count = pairwise
                .iter()
                .find(|pair| {
                    let key = (
                        pair.image_i.min(pair.image_j),
                        pair.image_i.max(pair.image_j),
                    );
                    key == (
                        seed_image_i.min(seed_image_j),
                        seed_image_i.max(seed_image_j),
                    )
                })
                .map_or(0, |pair| pair.matches.len());
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: initial pose growth fixed {} seed pose(s), reach={reach}",
                    seeded_images.len()
                );
            }
            (
                poses,
                track_point,
                grown_cam,
                seed_image_i,
                seed_image_j,
                seed_match_count,
            )
        } else {
            let seed_order = seed_candidate_order(pairwise, config);
            let trials = config.seed_trials.max(1);
            let not_trapped = largest_connected_component(pairwise, n_images)
                .div_ceil(2)
                .max(1);
            // A successful grow whose reach is far below the connected component
            // is a weak seed (a temporally adjacent, low-baseline pair that only
            // bootstraps a handful of frames). Keep searching past such seeds
            // instead of stopping once `seed_trials` grows have succeeded: on a
            // long connected sequence the strongest-match pairs are exactly the
            // adjacent ones, so a success-count cap exhausts the budget before
            // the wide-baseline seeds later in the order are ever tried. The
            // search still stops early as soon as a seed reaches
            // `not_trapped`, and is bounded by `seed_attempts` growth attempts.
            let weak_reach = (not_trapped / 4).max(1);
            let seed_attempts = if config.seed_attempts == 0 {
                trials
            } else {
                config.seed_attempts.max(trials)
            };
            let mut best: Option<SeedGrowth> = None;
            // Tracks which `pairwise` entry produced `best`, purely for observability
            // (the per-submap build summary log wants to report which image pair was
            // actually chosen as the seed). Always `Some` exactly when `best` is,
            // updated in lockstep below.
            let mut best_pi: Option<usize> = None;
            let mut grows = 0usize;
            let mut seed_attempted = 0usize;
            let mut seed_zero_reach = 0usize;
            for &pi in &seed_order {
                seed_attempted += 1;
                let trial_started = std::time::Instant::now();
                let (trial_poses, trial_points, reach, trial_cam) =
                    grow_from_seed_with_sequence_overrides(
                        camera,
                        features,
                        pairwise,
                        &tracks,
                        &conflicting_components,
                        &obs_by_image,
                        config,
                        debug_image_filter.as_ref(),
                        Some(&pairwise[pi]),
                        None,
                        sequence_override_pair_indices,
                    )?;
                if reach == 0 {
                    seed_zero_reach += 1;
                }
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: seed trial {pi} pair=({}, {}) matches={} -> reach={reach}",
                        pairwise[pi].image_i,
                        pairwise[pi].image_j,
                        pairwise[pi].matches.len(),
                    );
                }
                // Failed baseline-gated candidates are common on the ETH3D
                // orbit and are deliberately summarized rather than emitted
                // one-by-one when timing is enabled.  A successful trial is
                // still useful as a bounded checkpoint because it is the
                // expensive part of seed selection.
                if sfm_timing_enabled() && reach > 0 {
                    eprintln!(
                        "sfm-timing-seed-trial: index={pi} pair=({}, {}) reach={reach} \
                         elapsed={:.3}s",
                        pairwise[pi].image_i,
                        pairwise[pi].image_j,
                        trial_started.elapsed().as_secs_f64(),
                    );
                }
                if reach == 0 {
                    continue; // pair failed the seed gate — nothing placed, no grow ran
                }
                grows += 1;
                if best
                    .as_ref()
                    .is_none_or(|(best_reach, _, _, _)| reach > *best_reach)
                {
                    best = Some((reach, trial_poses, trial_points, trial_cam));
                    best_pi = Some(pi);
                }
                // Commit early on a strong seed; otherwise keep searching past
                // weak successes until the growth-attempt budget is spent.
                if reach >= not_trapped || grows >= seed_attempts {
                    break;
                }
                // A success weaker than a quarter of the connected component is
                // recorded but does not stop the search (see `weak_reach`).
                debug_assert!(weak_reach <= not_trapped);
            }
            if sfm_timing_enabled() {
                let winner_reach = best.as_ref().map_or(0, |(reach, _, _, _)| *reach);
                eprintln!(
                    "sfm-timing-seed-summary: candidates={} attempted={} zero_reach={} \
                     successful={} winner_reach={} elapsed={:.3}s",
                    seed_order.len(),
                    seed_attempted,
                    seed_zero_reach,
                    grows,
                    winner_reach,
                    seed_growth_started.elapsed().as_secs_f64(),
                );
            }
            let (_, poses, track_point, grown_cam) = best.ok_or(IncrementalSfmError::NoSeedPair)?;
            let winning_pi = best_pi.expect("set together with `best` on every assignment above");
            let seed_image_i = pairwise[winning_pi].image_i;
            let seed_image_j = pairwise[winning_pi].image_j;
            let seed_match_count = pairwise[winning_pi].matches.len();
            (
                poses,
                track_point,
                grown_cam,
                seed_image_i,
                seed_image_j,
                seed_match_count,
            )
        };
    let seed_growth_seconds = seed_growth_started.elapsed().as_secs_f64();
    process_memory::log("mapper-after-seed-growth");

    // ---- 4 + 5. Final refinement ----
    // When intrinsics refinement is on, growth already co-evolved them into
    // `grown_cam` (COLMAP keeps the camera moving with the structure so a wrong
    // focal cannot be silently absorbed). The final solve continues refining from
    // there; `cam` expresses the output poses/tracks/reprojection and is returned
    // to the caller for export.
    let final_refinement_started = std::time::Instant::now();
    process_memory::log("mapper-before-final-refinement");
    let mut cam = grown_cam;
    let mut ba_result = if config.colmap_style_mapper {
        // COLMAP's final pass IS an iterative global refinement (global BA →
        // complete/re-triangulate → filter, to convergence). The grow loop has
        // already run local BAs + growth-triggered refinements throughout.
        Some(
            iterative_global_refinement(
                &mut cam,
                features,
                &mut tracks,
                config,
                &mut poses,
                &mut track_point,
            )
            .map_err(IncrementalSfmError::Ba)?,
        )
    } else if config.final_iterative_global_refinement && config.final_global_ba {
        Some(
            iterative_global_refinement(
                &mut cam,
                features,
                &mut tracks,
                config,
                &mut poses,
                &mut track_point,
            )
            .map_err(IncrementalSfmError::Ba)?,
        )
    } else {
        // Simple schedule: one final global BA, then a few filter (+ optional
        // re-triangulate) rounds. With re-triangulation on, run at least one
        // round even when the filter budget is zero — the completion/re-seed pass
        // is the point of the round.
        let mut ba_result = if config.final_global_ba {
            let (res, refined) = run_bundle_adjustment(
                &cam,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
                config.refine_intrinsics,
            )
            .map_err(IncrementalSfmError::Ba)?;
            if let Some(c) = refined {
                cam = c;
            }
            Some(res)
        } else {
            None
        };
        // `--no-final-ba` is used for a growth-only timing/control run.  The
        // historical loop below could still enter a post-filter BA when the
        // filter removed an observation, even though `final_global_ba` was
        // disabled.  That made the flag misleading and, on large models,
        // paid for expensive solves after the caller explicitly requested no
        // final solve.  Post-BA filtering/retriangulation is part of the
        // final refinement contract, so keep it together with the initial
        // final BA.  The default (`final_global_ba=true`) is unchanged.
        let refine_rounds = simple_final_refinement_rounds(config);
        for round in 0..refine_rounds {
            let support_before = sfm_ba_debug_enabled().then(|| {
                (
                    poses.iter().filter(|pose| pose.is_some()).count(),
                    track_point.iter().filter(|point| point.is_some()).count(),
                    count_observations(&tracks, &poses, &track_point),
                )
            });
            let removed = filter_outlier_observations(
                &cam,
                features,
                &mut tracks,
                config,
                &poses,
                &mut track_point,
            );
            let support_after_filter = sfm_ba_debug_enabled().then(|| {
                (
                    poses.iter().filter(|pose| pose.is_some()).count(),
                    track_point.iter().filter(|point| point.is_some()).count(),
                    count_observations(&tracks, &poses, &track_point),
                )
            });
            let retriangulated = if config.retriangulate {
                retriangulate_tracks(&cam, features, &tracks, config, &poses, &mut track_point)
            } else {
                0
            };
            let support_after_retriangulation = sfm_ba_debug_enabled().then(|| {
                (
                    poses.iter().filter(|pose| pose.is_some()).count(),
                    track_point.iter().filter(|point| point.is_some()).count(),
                    count_observations(&tracks, &poses, &track_point),
                )
            });
            if let (Some(before), Some(after_filter), Some(after_retriangulation)) = (
                support_before,
                support_after_filter,
                support_after_retriangulation,
            ) {
                eprintln!(
                    "sfm-debug-ba-support: stage=simple_final round={} \
                     before=({},{},{}) after_filter=({},{},{}) \
                     after_retriangulation=({},{},{}) removed={} retriangulated={}",
                    round,
                    before.0,
                    before.1,
                    before.2,
                    after_filter.0,
                    after_filter.1,
                    after_filter.2,
                    after_retriangulation.0,
                    after_retriangulation.1,
                    after_retriangulation.2,
                    removed,
                    retriangulated,
                );
            }
            if removed == 0 && retriangulated == 0 {
                break;
            }
            let (res, refined) = run_bundle_adjustment(
                &cam,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
                config.refine_intrinsics,
            )
            .map_err(IncrementalSfmError::Ba)?;
            if let Some(c) = refined {
                cam = c;
            }
            ba_result = Some(res);
        }
        ba_result
    };

    // Optional pose-guided multi-model track split.  This deliberately runs
    // after the initial growth/final refinement so classification sees a
    // complete, fixed pose model.  The candidate topology is built from both
    // clean union components and the components that legacy union-find
    // discarded for same-image conflicts; one guarded BA is then used to
    // validate the rebuilt landmarks.  A partial model or a failed support /
    // cost gate leaves the ordinary result untouched.  Optional outer passes
    // always rebuild from the source components captured below, never from a
    // previously split output, so a later pass cannot recursively fragment an
    // already accepted partition.  When geometry recovery is composed with
    // this diagnostic, the source snapshot is captured before recovery and
    // the macro is invoked only after recovery/post/final stages.
    let pose_split_source = capture_pose_guided_split_source(
        config.pose_guided_track_splitting,
        track_membership,
        &tracks,
        &conflicting_components,
        &track_point,
    );
    macro_rules! apply_pose_guided_split {
        ($run:expr) => {{
            if $run {
                let (source_tracks, source_conflicting_components, source_track_point) =
                    pose_split_source
                        .as_ref()
                        .expect("pose split source captured when enabled");
        let max_iterations = config.pose_guided_track_splitting_iterations.clamp(1, 8);
        let split_max_reprojection_error = config
            .pose_guided_split_max_reprojection_error_px
            .unwrap_or(config.max_reprojection_error_px);
        let mut accepted_any = false;
        for iteration in 0..max_iterations {
            let tracks_before = tracks.clone();
            let track_point_before = track_point.clone();
            let poses_before = poses.clone();
            let cam_before = cam.clone();
            let support_before = count_observations(&tracks, &poses, &track_point);
            let registered_before = poses.iter().filter(|pose| pose.is_some()).count();
            let mean_before = mean_reprojection_for_track_range(
                &cam,
                features,
                &tracks,
                &poses,
                &track_point,
                0,
                tracks.len(),
            );
            let result = pose_guided_split_tracks(
                &cam,
                features,
                pairwise,
                &source_tracks,
                &source_conflicting_components,
                &source_track_point,
                &poses,
                config,
            );
            let Some(split) = result else {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: pose-guided track split iteration={} unavailable; stopping",
                        iteration + 1
                    );
                }
                break;
            };
            let merge_restorations = split.merge_restorations.clone();
            let stats = split.stats;
            // A bounded, explicit debug hook makes the rebuilt partition
            // inspectable without adding an oracle dependency to the mapper.
            // It is intentionally environment-only and never runs unless the
            // caller names an output path.
            if let Some(path) = std::env::var_os("VISLOC_SFM_DEBUG_POSE_SPLIT_DUMP") {
                match dump_pose_guided_track_split(
                    std::path::Path::new(&path),
                    &split.tracks,
                ) {
                    Ok(observations) if sfm_debug_enabled() => eprintln!(
                        "sfm-debug: pose-guided track split dump iteration={} path={:?} tracks={} observations={}",
                        iteration + 1,
                        path,
                        split.tracks.len(),
                        observations,
                    ),
                    Ok(_) => {}
                    Err(error) if sfm_debug_enabled() => eprintln!(
                        "sfm-debug: pose-guided track split dump failed iteration={} path={:?}: {error}",
                        iteration + 1,
                        path,
                    ),
                    Err(_) => {}
                }
            }
            let candidate_support = count_observations(&split.tracks, &poses, &split.points);
            let candidate_mean = mean_reprojection_for_track_range(
                &cam,
                features,
                &split.tracks,
                &poses,
                &split.points,
                0,
                split.tracks.len(),
            );
            // A split is allowed to trade a small amount of aggregate pixel
            // error for substantially better observation support, but only if
            // it does not discard current support and the validation BA lowers
            // the candidate's own objective.  For the first pass, this is the
            // original single-pass guard exactly; later passes additionally
            // require a strict improvement over the already accepted model.
            let support_floor = support_before.max(config.min_track_length.max(2));
            let candidate_gate = pose_guided_split_candidate_gate(
                candidate_support,
                support_floor,
                candidate_mean,
                split_max_reprojection_error,
            );
            let mut accepted = false;
            let mut candidate_ba_result = None;
            let mut after_mean = f64::INFINITY;
            let mut after_support = 0usize;
            let mut registered_after = registered_before;
            let mut merge_hard_gate = true;
            let mut merge_proposed = 0usize;
            let mut merge_good = 0usize;
            let mut merge_restored = 0usize;
            if candidate_gate {
                tracks = split.tracks;
                track_point = split.points;
                merge_proposed = merge_restorations.len();
                if let Ok((mut ba, refined)) = run_bundle_adjustment(
                    &cam,
                    features,
                    &tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                    config.refine_intrinsics,
                ) {
                    if let Some(refined) = refined {
                        cam = refined;
                    }
                    if !merge_restorations.is_empty() {
                        let (_, restored) = pose_guided_restore_invalid_merges(
                            &cam,
                            features,
                            &poses,
                            &mut tracks,
                            &mut track_point,
                            &merge_restorations,
                            config.max_reprojection_error_px,
                        );
                        merge_restored = restored;
                        merge_good = merge_proposed.saturating_sub(merge_restored);
                        if merge_restored > 0 {
                            match run_bundle_adjustment(
                                &cam,
                                features,
                                &tracks,
                                config,
                                &mut poses,
                                &mut track_point,
                                config.refine_intrinsics,
                            ) {
                                Ok((rerun_ba, rerun_cam)) => {
                                    ba = rerun_ba;
                                    if let Some(rerun_cam) = rerun_cam {
                                        cam = rerun_cam;
                                    }
                                }
                                Err(_) => {
                                    // The outer candidate snapshot handles
                                    // this failure as a whole-model rollback.
                                    merge_hard_gate = false;
                                    merge_good = 0;
                                }
                            }
                        }
                    }
                    let ba_succeeded = merge_hard_gate;
                    if !ba_succeeded {
                        after_mean = f64::INFINITY;
                        after_support = 0;
                        registered_after = registered_before;
                    } else {
                    after_mean = mean_reprojection_for_track_range(
                        &cam,
                        features,
                        &tracks,
                        &poses,
                        &track_point,
                        0,
                        tracks.len(),
                    );
                    after_support = count_observations(&tracks, &poses, &track_point);
                    registered_after = poses.iter().filter(|pose| pose.is_some()).count();
                    merge_hard_gate = stats.merged_tracks == 0
                        || pose_guided_merge_restorations_reprojection_valid(
                            &cam,
                            features,
                            &poses,
                            &tracks,
                            &track_point,
                            &merge_restorations,
                            merge_restored,
                            config.max_reprojection_error_px,
                        );
                    if !merge_hard_gate {
                        merge_good = 0;
                    }
                    accepted = pose_guided_split_candidate_accepts(
                        iteration,
                        registered_before,
                        registered_after,
                        support_floor,
                        candidate_support,
                        after_support,
                        mean_before,
                        candidate_mean,
                        after_mean,
                        split_max_reprojection_error,
                    ) && merge_hard_gate;
                    if accepted {
                        candidate_ba_result = Some(ba);
                    }
                    }
                }
            }
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: pose-guided track split iteration={} accepted={} bridge_cuts={} bridge_components={} bridge_sizes={:?} graph_support={} merge={} merge_gate={:.3} merge_candidates={} merged_tracks={} merge_proposed={} merge_good={} merge_restored={} merge_hard_gate={} candidate_gate={} support {}=>{}=>{} floor={} registered {}=>{} mean {:.6}=>{:.6}=>{:.6} components={} preserved={} split={} hypotheses={} discarded_obs={} graph_tracks={} graph_len2={} graph_hist={:?}",
                    iteration + 1,
                    accepted,
                    stats.bridge_cuts,
                    stats.bridge_cut_components,
                    stats.bridge_cut_sizes,
                    config.pose_guided_graph_support,
                    config.pose_guided_track_merging,
                    config
                        .pose_guided_merge_max_reprojection_error_px
                        .unwrap_or(split_max_reprojection_error),
                    stats.merge_candidates_tested,
                    stats.merged_tracks,
                    merge_proposed,
                    merge_good,
                    merge_restored,
                    merge_hard_gate,
                    candidate_gate,
                    support_before,
                    candidate_support,
                    after_support,
                    support_floor,
                    registered_before,
                    registered_after,
                    mean_before,
                    candidate_mean,
                    after_mean,
                    stats.input_components,
                    stats.preserved_components,
                    stats.split_components,
                    stats.hypotheses_tested,
                    stats.discarded_observations,
                    stats.graph_supported_tracks,
                    stats.graph_length_two_tracks,
                    stats.graph_support_histogram,
                );
            }
            if accepted {
                accepted_any = true;
                ba_result = candidate_ba_result;
            } else {
                tracks = tracks_before;
                track_point = track_point_before;
                poses = poses_before;
                cam = cam_before;
                break;
            }
        }
        if accepted_any {
            // Keep the original conflicts out of the later geometry-recovery
            // stage, but only after all bounded split passes have completed.
            conflicting_components.clear();
        }
            }
        }};
    }
    apply_pose_guided_split!(
        config.pose_guided_track_splitting
            && !config.geometry_guided_conflict_recovery
            && track_membership.is_none()
    );

    let mut geometry_recovered_tracks = 0usize;
    let mut geometry_recovered_observations = 0usize;
    let mut geometry_recovery_pose_ba_applied = false;
    let geometry_recovery_started = std::time::Instant::now();
    if config.geometry_guided_conflict_recovery && !conflicting_components.is_empty() {
        let recovered = recover_conflict_tracks_geometry(
            &cam,
            features,
            pairwise,
            &conflicting_components,
            &poses,
            config,
        );
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: geometry conflict recovery proposed {} tracks / {} observations",
                recovered.len(),
                recovered
                    .iter()
                    .map(|track| track.observations.len())
                    .sum::<usize>(),
            );
        }
        if !recovered.is_empty() {
            let clean_track_count = tracks.len();
            let clean_mean_before = mean_reprojection_for_track_range(
                &cam,
                features,
                &tracks,
                &poses,
                &track_point,
                0,
                clean_track_count,
            );
            let poses_before = poses.clone();
            let track_point_before = track_point.clone();
            for candidate in &recovered {
                tracks.push(candidate.observations.clone());
                track_point.push(Some(candidate.point));
            }

            // Once every image is already registered, conflict recovery is a
            // structure-density operation. The held-out MH_01 A/B showed that
            // a residual-improving extra pose BA can still worsen independent
            // GT ATE, so a complete trajectory is immutable here. Incomplete
            // models may use one guarded BA because recovered structure can
            // unlock missing-image PnP and improve the development trajectory.
            let model_complete = poses.iter().all(Option::is_some);
            let mut accepted = model_complete;
            if model_complete {
                geometry_recovered_tracks = recovered.len();
                geometry_recovered_observations =
                    recovered.iter().map(|track| track.observations.len()).sum();
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: geometry conflict recovery accepted structure-only; \
                         complete {}/{} pose model remains byte-identical",
                        poses.len(),
                        poses.len(),
                    );
                }
            } else if let Ok((result, _)) = run_bundle_adjustment(
                &cam,
                features,
                &tracks,
                config,
                &mut poses,
                &mut track_point,
                false,
            ) {
                let clean_mean_after = mean_reprojection_for_track_range(
                    &cam,
                    features,
                    &tracks,
                    &poses,
                    &track_point,
                    0,
                    clean_track_count,
                );
                let recovered_mean_after = mean_reprojection_for_track_range(
                    &cam,
                    features,
                    &tracks,
                    &poses,
                    &track_point,
                    clean_track_count,
                    tracks.len(),
                );
                let allowed_clean_mean = clean_mean_before
                    * (1.0
                        + config
                            .conflict_recovery_max_clean_error_increase_ratio
                            .max(0.0));
                accepted = clean_mean_before.is_finite()
                    && clean_mean_after.is_finite()
                    && recovered_mean_after.is_finite()
                    && clean_mean_after <= allowed_clean_mean + 1e-12
                    && recovered_mean_after <= config.conflict_recovery_max_mean_reprojection_px;
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: geometry conflict recovery guard accepted={accepted} \
                         clean_mean={clean_mean_before:.6}->{clean_mean_after:.6} \
                         (allowed {allowed_clean_mean:.6}) recovered_mean={recovered_mean_after:.6}",
                    );
                }
                if accepted {
                    geometry_recovered_tracks = recovered.len();
                    geometry_recovered_observations =
                        recovered.iter().map(|track| track.observations.len()).sum();
                    geometry_recovery_pose_ba_applied = true;
                    ba_result = Some(result);
                }
            } else if sfm_debug_enabled() {
                eprintln!("sfm-debug: geometry conflict recovery BA failed; rolling back");
            }
            if !accepted {
                tracks.truncate(clean_track_count);
                poses = poses_before;
                track_point = track_point_before;
            }
        }
    }
    let geometry_recovery_seconds = geometry_recovery_started.elapsed().as_secs_f64();

    let mut post_refinement_registered_images = 0usize;
    if config.post_refinement_registration {
        let ordinary_post_registered = post_refinement_registration_pass(
            &cam,
            features,
            &tracks,
            config,
            &mut poses,
            &mut track_point,
        )
        .map_err(IncrementalSfmError::Ba)?;
        post_refinement_registered_images = ordinary_post_registered;
        if config.sequence_relative_pose_fallback && initial_poses.is_none() {
            if config.sequence_fallback_after_post {
                // Let the ordinary post-refinement sweep (and its BA below)
                // exhaust every currently PnP-solvable image first.  Then
                // admit exactly one consecutive relative-pose fallback and
                // immediately resume ordinary PnP, so newly triangulated
                // structure can unlock images without eagerly chaining
                // provisional poses.
                if ordinary_post_registered > 0 {
                    ba_result = Some(
                        iterative_global_refinement(
                            &mut cam,
                            features,
                            &mut tracks,
                            config,
                            &mut poses,
                            &mut track_point,
                        )
                        .map_err(IncrementalSfmError::Ba)?,
                    );
                }
                let mut carried_sequence_fallback: Option<(usize, f64)> = None;
                loop {
                    let fallback_registered =
                        sequence_relative_pose_registration_once_with_overrides_and_carry(
                            &cam,
                            features,
                            pairwise,
                            tracks.as_slice(),
                            config,
                            &mut poses,
                            &mut track_point,
                            sequence_override_pair_indices,
                            carried_sequence_fallback,
                        )
                        .map_err(IncrementalSfmError::Ba)?;
                    let Some((fallback_image, fallback_scale)) = fallback_registered else {
                        break;
                    };
                    post_refinement_registered_images += 1;
                    let resumed_post_registered = post_refinement_registration_pass(
                        &cam,
                        features,
                        &tracks,
                        config,
                        &mut poses,
                        &mut track_point,
                    )
                    .map_err(IncrementalSfmError::Ba)?;
                    post_refinement_registered_images += resumed_post_registered;
                    // A normal PnP/post insertion invalidates the
                    // consecutive-provisional chain.  If no ordinary image
                    // was added, the next fallback may reuse this accepted
                    // baseline magnitude.
                    carried_sequence_fallback = next_sequence_fallback_carry_state(
                        fallback_image,
                        fallback_scale,
                        resumed_post_registered,
                    );
                    if sfm_debug_enabled() {
                        eprintln!(
                            "sfm-debug: sequence fallback after-post stage fallback=1 resumed_pnp={} carry_next={} total_post={}",
                            resumed_post_registered,
                            carried_sequence_fallback.is_some(),
                            post_refinement_registered_images,
                        );
                    }
                    ba_result = Some(
                        iterative_global_refinement(
                            &mut cam,
                            features,
                            &mut tracks,
                            config,
                            &mut poses,
                            &mut track_point,
                        )
                        .map_err(IncrementalSfmError::Ba)?,
                    );
                }
            } else {
                post_refinement_registered_images +=
                    sequence_relative_pose_registration_pass_with_overrides(
                        &cam,
                        features,
                        pairwise,
                        tracks.as_slice(),
                        config,
                        &mut poses,
                        &mut track_point,
                        sequence_override_pair_indices,
                    )
                    .map_err(IncrementalSfmError::Ba)?;
            }
        }
        if !config.sequence_fallback_after_post && post_refinement_registered_images > 0 {
            ba_result = Some(
                iterative_global_refinement(
                    &mut cam,
                    features,
                    &mut tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?,
            );
        }
    }
    let structureless_started = std::time::Instant::now();
    let structureless_registered_images = if config.colmap_style_mapper
        && config.structureless_registration
        && poses.iter().any(Option::is_none)
    {
        structureless_registration_rounds(
            &cam,
            features,
            pairwise,
            &mut tracks,
            config,
            &mut poses,
            &mut track_point,
        )
    } else {
        0
    };
    let structureless_seconds = structureless_started.elapsed().as_secs_f64();
    if config.final_ba_polish_iterations > 0 || config.geometry_weighted_ba {
        let (polish_stats, polished_result) = final_fixed_support_ba_polish(
            &cam,
            features,
            &tracks,
            config,
            &mut poses,
            &mut track_point,
        )
        .map_err(IncrementalSfmError::Ba)?;
        if let Some(result) = polished_result {
            ba_result = Some(result);
        }
        if sfm_debug_enabled() {
            eprintln!(
                concat!(
                    "sfm-debug: final BA polish accepted={} SSE {:.9e}->{:.9e} ",
                    "support tracks {}=>{} observations {}=>{}"
                ),
                polish_stats.accepted,
                polish_stats.initial_sse,
                polish_stats.final_sse,
                polish_stats.tracks_before,
                polish_stats.tracks_after,
                polish_stats.observations_before,
                polish_stats.observations_after,
            );
        }
    }
    apply_pose_guided_split!(
        config.pose_guided_track_splitting
            && config.geometry_guided_conflict_recovery
            && track_membership.is_none()
    );
    let final_track_length_gate_stats = apply_final_track_length_gate(
        &mut cam,
        features,
        &mut tracks,
        config,
        &mut poses,
        &mut track_point,
        &mut ba_result,
    );
    if sfm_debug_enabled() && config.final_min_track_length.is_some() {
        eprintln!(
            concat!(
                "sfm-debug: final track-length gate min={} attempted={} accepted={} ",
                "tracks {}-{}=>{} observations {}-{}=>{} retriangulated={} ",
                "registered {}=>{} mean {:.6}=>{:.6} finite={} support={} objective={}"
            ),
            final_track_length_gate_stats.requested_min_length,
            final_track_length_gate_stats.attempted,
            final_track_length_gate_stats.accepted,
            final_track_length_gate_stats.tracks_before,
            final_track_length_gate_stats.tracks_removed,
            final_track_length_gate_stats.tracks_after,
            final_track_length_gate_stats.observations_before,
            final_track_length_gate_stats.observations_removed,
            final_track_length_gate_stats.observations_after,
            final_track_length_gate_stats.retriangulated_tracks,
            final_track_length_gate_stats.registered_before,
            final_track_length_gate_stats.registered_after,
            final_track_length_gate_stats.mean_before_ba,
            final_track_length_gate_stats.mean_after_ba,
            final_track_length_gate_stats.finite_state,
            final_track_length_gate_stats.support_valid,
            final_track_length_gate_stats.objective_valid,
        );
    }
    let final_refinement_seconds = final_refinement_started.elapsed().as_secs_f64();
    process_memory::log("mapper-after-final-refinement");

    // ---- Assemble output tracks (only triangulated, registered observations) ----
    let assembly_started = std::time::Instant::now();
    process_memory::log("mapper-before-output-assembly");
    let mut out_tracks = Vec::new();
    let mut reproj_sum = 0.0;
    let mut reproj_count = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(position) = track_point[track_id] else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, kp) in track {
            let Some(pose) = &poses[image] else { continue };
            let Some(pixel) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            observations.push((image, kp, pixel));
            if let Some(err) = reprojection_error_px(&cam, pose, &position, &pixel) {
                reproj_sum += err;
                reproj_count += 1;
            }
        }
        if observations.len() >= config.min_track_length {
            out_tracks.push(SfmTrack {
                position,
                observations,
            });
        }
    }

    let registered_images = poses.iter().filter(|p| p.is_some()).count();
    let mean_reprojection_px = if reproj_count > 0 {
        reproj_sum / reproj_count as f64
    } else {
        f64::NAN
    };
    if sfm_timing_or_debug_enabled() {
        eprintln!(
            "sfm-timing: total={:.3}s track_build={track_build_seconds:.3}s \
             seed_growth={seed_growth_seconds:.3}s final_refinement={final_refinement_seconds:.3}s \
             geometry_recovery={geometry_recovery_seconds:.3}s \
             structureless={structureless_seconds:.3}s assembly={:.3}s",
            sfm_started.elapsed().as_secs_f64(),
            assembly_started.elapsed().as_secs_f64(),
        );
    }

    Ok(IncrementalSfmResult {
        poses,
        tracks: out_tracks,
        track_build_stats,
        registered_images,
        post_refinement_registered_images,
        structureless_registered_images,
        geometry_recovered_tracks,
        geometry_recovered_observations,
        geometry_recovery_pose_ba_applied,
        mean_reprojection_px,
        ba_result,
        refined_camera: config.refine_intrinsics.then_some(cam),
        seed_image_i,
        seed_image_j,
        seed_match_count,
    })
}
