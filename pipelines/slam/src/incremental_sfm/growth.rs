//! Seed placement, image registration growth and next-image selection.

use super::*;

/// Size of the largest connected component of the view graph — images joined by
/// a verified pair. This bounds how many images any single seed can ever reach,
/// so a seed that reaches a large fraction of it is well-connected rather than an
/// isolated local cluster of a few near-identical frames.
pub(super) fn largest_connected_component(pairwise: &[PairwiseMatches], n_images: usize) -> usize {
    if n_images == 0 {
        return 0;
    }
    let mut parent: Vec<usize> = (0..n_images).collect();
    for p in pairwise {
        union(&mut parent, p.image_i, p.image_j);
    }
    let mut count = vec![0usize; n_images];
    for i in 0..n_images {
        let r = find(&mut parent, i);
        count[r] += 1;
    }
    count.into_iter().max().unwrap_or(0)
}

/// Indices of verified pairs in descending match-count order, restricted to
/// those that clear `min_seed_matches`. These are the candidate seeds, strongest
/// first; [`grow_from_seed`] decides which one actually bootstraps the largest
/// reconstruction.
pub(super) fn seed_candidate_order(
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pairwise.len())
        .filter(|&i| pairwise[i].matches.len() >= config.min_seed_matches)
        .filter(|&i| {
            let pair = &pairwise[i];
            let key = (
                pair.image_i.min(pair.image_j),
                pair.image_i.max(pair.image_j),
            );
            !config.excluded_seed_pairs.contains(&key)
                && config.seed_pair.is_none_or(|requested| requested == key)
        })
        .collect();
    order.sort_by_key(|&i| std::cmp::Reverse(pairwise[i].matches.len()));
    order
}

/// Recover one verified pair's two-view relative pose and place both images
/// (seed `i` at the world origin, `j` at the relative pose). Returns `true` only
/// if the pair bootstraps a well-conditioned baseline: enough of its inlier
/// correspondences triangulate under the shared parallax / cheirality /
/// reprojection gate. A low-parallax pair (e.g. two adjacent frames) is rejected
/// and `poses` is left untouched for `i` and `j`.
pub(super) fn place_seed_pair(
    camera: &Camera,
    features: &[FeatureSet],
    pair: &PairwiseMatches,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
) -> bool {
    let estimator = RelativePoseEstimator::default();

    // Build correspondences, keeping the (kp_i, kp_j) map aligned so a
    // relative-pose inlier index maps back to the right keypoints.
    let mut corrs = Vec::with_capacity(pair.matches.len());
    let mut corr_kp = Vec::with_capacity(pair.matches.len());
    for &(ki, kj) in &pair.matches {
        let (Some(pi_xy), Some(pj_xy)) = (
            features[pair.image_i].keypoints.get(ki),
            features[pair.image_j].keypoints.get(kj),
        ) else {
            continue;
        };
        corrs.push(TwoViewCorrespondence::new(*pi_xy, *pj_xy));
        corr_kp.push((*pi_xy, *pj_xy));
    }
    let Some(relative) = estimator.estimate(&corrs, camera) else {
        if std::env::var_os("VISLOC_SFM_SEED_DEBUG").is_some() {
            eprintln!(
                "sfm-seed-debug: pair=({},{}) corrs={} estimate=FAIL",
                pair.image_i,
                pair.image_j,
                corrs.len()
            );
        }
        return false;
    };
    if relative.inliers.len() < config.min_seed_matches {
        if std::env::var_os("VISLOC_SFM_SEED_DEBUG").is_some() {
            eprintln!(
                "sfm-seed-debug: pair=({},{}) corrs={} inliers={} < min_seed_matches={}",
                pair.image_i,
                pair.image_j,
                corrs.len(),
                relative.inliers.len(),
                config.min_seed_matches
            );
        }
        return false;
    }
    // Tentatively place: image i at the origin, image j at the relative.
    poses[pair.image_i] = Some(Pose::from_world_to_camera(
        nalgebra::UnitQuaternion::identity(),
        Vector3::zeros(),
    ));
    poses[pair.image_j] = Some(Pose::from_world_to_camera(
        relative.previous_to_current.rotation,
        relative.previous_to_current.translation,
    ));
    // Count inlier correspondences that triangulate to well-conditioned points.
    let mut well_triangulated = 0usize;
    let centres = [
        poses[pair.image_i]
            .as_ref()
            .map(|p| p.camera_to_world().translation),
        poses[pair.image_j]
            .as_ref()
            .map(|p| p.camera_to_world().translation),
    ];
    let mut angles: Vec<f64> = Vec::new();
    for &inl in &relative.inliers {
        let (px_i, px_j) = corr_kp[inl];
        let obs = [(pair.image_i, px_i), (pair.image_j, px_j)];
        if let Some(x) = triangulate_track(camera, poses, &obs, config) {
            well_triangulated += 1;
            if let [Some(ci), Some(cj)] = centres {
                angles.push((x.coords - ci).angle(&(x.coords - cj)).to_degrees());
            }
        }
    }
    if let Some(min_angle) = config.seed_min_median_tri_angle_deg {
        angles.sort_by(f64::total_cmp);
        let median = angles.get(angles.len() / 2).copied().unwrap_or(0.0);
        if std::env::var_os("VISLOC_SFM_SEED_DEBUG").is_some() {
            eprintln!(
                "sfm-seed-debug: pair=({},{}) median_tri_angle={median:.2} min={min_angle}",
                pair.image_i, pair.image_j
            );
        }
        if median < min_angle {
            poses[pair.image_i] = None;
            poses[pair.image_j] = None;
            return false;
        }
    }
    if std::env::var_os("VISLOC_SFM_SEED_DEBUG").is_some() {
        eprintln!(
            "sfm-seed-debug: pair=({},{}) corrs={} inliers={} well_triangulated={} min_seed_matches={}",
            pair.image_i, pair.image_j, corrs.len(), relative.inliers.len(), well_triangulated, config.min_seed_matches
        );
    }
    if well_triangulated >= config.min_seed_matches {
        return true; // good baseline — keep these poses
    }
    // Low parallax: undo.
    poses[pair.image_i] = None;
    poses[pair.image_j] = None;
    false
}

/// Whether a sequence-relative proposal may be admitted from the ordinary
/// growth loop.  The after-post policy deliberately suppresses this path and
/// invokes the same proposal logic only after the ordinary post-refinement
/// sweep has stalled.
pub(super) const fn sequence_fallback_enabled_during_growth(config: &IncrementalSfmConfig) -> bool {
    config.sequence_relative_pose_fallback && !config.sequence_fallback_after_post
}

/// Whether the support-preserving targeted growth fast path is valid. It is
/// intentionally restricted to the plain mapper: correspondence-mode points
/// can be replaced, COLMAP-style local/global BA can complete tracks outside
/// the newly registered image, and sequence fallback has its own full-scan
/// commit helper. All of those modes therefore retain the historical full
/// pending-track scan.
pub(super) const fn targeted_plain_growth_enabled(
    config: &IncrementalSfmConfig,
    has_initial_poses: bool,
) -> bool {
    !has_initial_poses
        && !config.colmap_style_mapper
        && !config.incremental_correspondence_triangulation
        && !config.sequence_relative_pose_fallback
}

/// Bootstrap from `seed_pair` and grow the reconstruction by repeatedly
/// registering the best next image, running the periodic global bundle
/// adjustment every `ba_every` registrations. Returns the per-image poses,
/// per-track points and the number of registered images — the reach the seed
/// selection compares across candidates. A seed that fails the baseline gate
/// yields zero registered images.
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub(super) fn grow_from_seed_with_sequence_overrides(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    conflicting_components: &[Vec<(usize, usize)>],
    obs_by_image: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    debug_image_filter: Option<&HashSet<usize>>,
    seed_pair: Option<&PairwiseMatches>,
    initial_poses: Option<&[Option<Pose>]>,
    sequence_override_pair_indices: Option<&[usize]>,
) -> Result<(Vec<Option<Pose>>, Vec<Option<Point3<f64>>>, usize, Camera), IncrementalSfmError> {
    let grow_started = std::time::Instant::now();
    let mut select_seconds = 0.0;
    let mut pnp_seconds = 0.0;
    let mut triangulation_seconds = 0.0;
    let mut local_ba_seconds = 0.0;
    let mut global_refinement_seconds = 0.0;
    let mut pnp_attempts = 0usize;
    let mut local_ba_calls = 0usize;
    let mut global_refinement_calls = 0usize;
    let mut triangulation_full_calls = 0usize;
    let mut triangulation_targeted_calls = 0usize;
    let mut triangulation_targeted_tracks = 0usize;
    let timing_enabled = sfm_timing_enabled();
    let mut last_progress_elapsed = 0.0;
    let mut last_progress_select = 0.0;
    let mut last_progress_pnp = 0.0;
    let mut last_progress_triangulation = 0.0;
    let mut last_progress_ba = 0.0;
    let n_images = features.len();
    let mut poses: Vec<Option<Pose>> =
        initial_poses.map_or_else(|| vec![None; n_images], |poses| poses.to_vec());
    let mut track_point: Vec<Option<Point3<f64>>> = vec![None; tracks.len()];
    let mut correspondence_state = config
        .incremental_correspondence_triangulation
        .then(|| CorrespondencePointState::from_tracks(tracks, &track_point));

    // Per-trial camera clone: the seed search grows several reconstructions over
    // the same shared `tracks`, so each trial co-evolves intrinsics on its own
    // copy (no cross-trial contamination). The winning trial's camera is returned.
    let mut cam = camera.clone();
    if let Some(seed_pair) = seed_pair {
        if !place_seed_pair(&cam, features, seed_pair, config, &mut poses) {
            return Ok((poses, track_point, 0, cam));
        }
    } else if initial_poses.is_none() {
        return Err(IncrementalSfmError::InvalidInitialPoses(
            "internal growth call omitted both a seed pair and initial poses".into(),
        ));
    }
    let started = std::time::Instant::now();
    if !targeted_plain_growth_enabled(config, initial_poses.is_some()) {
        triangulate_pending_with_config_and_state(
            &cam,
            features,
            tracks,
            &poses,
            config,
            &mut track_point,
            correspondence_state.as_mut(),
        );
        triangulation_full_calls += 1;
    } else if let Some(seed_pair) = seed_pair {
        let (visited, _) = triangulate_pending_for_images_with_new_tracks(
            &cam,
            features,
            tracks,
            &poses,
            config,
            &mut track_point,
            obs_by_image,
            &[seed_pair.image_i, seed_pair.image_j],
        );
        triangulation_targeted_tracks += visited;
        triangulation_targeted_calls += 1;
    }
    triangulation_seconds += started.elapsed().as_secs_f64();

    // The cache is deliberately limited to the plain, non-COLMAP-style
    // count-policy growth path. Correspondence-mode triangulation can replace existing points,
    // and sequence fallback can add a point through a separate commit helper;
    // those modes retain their historical fresh-scan semantics. In the
    // ordinary path a point's Some/None state only changes from None to Some,
    // which is exactly what the targeted update reports below.
    let mut correspondence_count_cache = (config.next_image_policy
        == NextImagePolicy::CorrespondenceCount
        && targeted_plain_growth_enabled(config, initial_poses.is_some()))
    .then(|| build_correspondence_count_cache(features, obs_by_image, &track_point));

    // `trials[i]` counts PnP attempts on image `i`. In the simple schedule one
    // failed attempt is permanent (the cap is 1); the COLMAP schedule retries up
    // to `max_registration_trials` across global-refinement boundaries.
    let max_trials = if config.colmap_style_mapper {
        config.max_registration_trials.max(1)
    } else {
        1
    };
    let mut trials: Vec<usize> = vec![0; n_images];
    let mut registrations_since_ba = 0usize;
    // A BA can move every camera, so the next successful registration must
    // perform one historical full pending scan before targeted updates
    // resume. Keeping this deferred until after selection preserves the
    // original registration ordering exactly.
    let mut needs_full_triangulation_after_ba = false;
    // COLMAP triggers a global refinement once the registered-image count has
    // grown by `global_ba_images_ratio` since the last one.
    let mut reg_at_last_global = poses.iter().filter(|p| p.is_some()).count();
    // COLMAP `IncrementalPipeline::ReconstructSubModel`'s do-while loop
    // (`controllers/incremental_pipeline.cc:519-629`) never gives up the first
    // time no image can be registered: when a full round finds nothing
    // (`!reg_next_success`), it runs one more `IterativeGlobalRefinement` and
    // tries again, only stopping once *two consecutive* rounds both find
    // nothing (`while (reg_next_success || prev_reg_next_success)`, line 629).
    // `stalled_once` is that same one-shot recovery. It matters because
    // `select_next_image` returning `None` is not always "structurally done" —
    // a track that lacked the 6th correspondence [`select_next_image`] needs
    // can gain one once [`growth_global_refinement`]'s retriangulation
    // completes a track that had ≥2 registered observers all along, just not
    // at a pair the on-the-fly [`triangulate_pending`] happened to accept
    // (BA can tighten those same views' poses enough, between one
    // registration and the next stall, to flip a marginal parallax/
    // reprojection gate that failed moments before). This is the M4 fix for
    // the path-dependence diagnosed in `docs/colmap_port_plan.md`'s "M3
    // results" (courtyard stuck at 13-14/38 even under exhaustive pair
    // coverage): the growth-ratio-triggered refinement above only fires while
    // registrations keep succeeding, so once growth truly stalls the ratio
    // can never trigger again and this loop broke immediately, leaving
    // whatever a completing refinement might have unlocked untried.
    //
    // Deliberately **not** ported: resetting `trials` on the stall, even
    // though it would let an already-trial-exhausted image be re-offered.
    // COLMAP's own `num_reg_trials` never resets either
    // (`incremental_mapper.cc:229`, incremented unconditionally on *every*
    // `RegisterNextImage` call, success or failure, for the reconstruction's
    // whole lifetime) — and here that persistence is load-bearing, not just
    // an unported nicety: with `filter_images` on, a resetting version can
    // livelock (register a weakly-supported image → `filter_images` demotes
    // it next stall → the reset makes it eligible again → it re-registers
    // identically → demoted again → …, forever, since each re-registration
    // looks like "progress" and would keep re-arming the recovery). Never
    // resetting bounds every image, demoted or not, to
    // `max_registration_trials` total lifetime attempts, so this cannot
    // cycle more than that many times before the image is excluded for good
    // — the same guarantee COLMAP's design gets from never resetting.
    let mut stalled_once = false;
    loop {
        let started = std::time::Instant::now();
        let selection = select_next_image(
            &cam,
            config.next_image_policy,
            features,
            obs_by_image,
            &poses,
            &trials,
            max_trials,
            &track_point,
            correspondence_count_cache.as_deref(),
        );
        select_seconds += started.elapsed().as_secs_f64();
        if let Some((next_image, corrs)) = selection.as_ref() {
            log_registration_track_provenance(
                *next_image,
                corrs.len(),
                features,
                tracks,
                conflicting_components,
                obs_by_image,
                &poses,
                &track_point,
                debug_image_filter,
            );
        }
        let Some((next_image, corrs)) = selection else {
            let n_reg = poses.iter().filter(|p| p.is_some()).count();
            if initial_poses.is_none()
                && !config.colmap_style_mapper
                && sequence_fallback_enabled_during_growth(config)
            {
                if let Some(proposal) = sequence_relative_pose_fallback_with_overrides(
                    &cam,
                    features,
                    pairwise,
                    &poses,
                    config,
                    sequence_override_pair_indices,
                ) {
                    commit_sequence_relative_pose(
                        proposal,
                        &cam,
                        features,
                        tracks,
                        config,
                        &mut poses,
                        &mut track_point,
                        &mut correspondence_state,
                        &mut triangulation_seconds,
                        &mut registrations_since_ba,
                    );
                    stalled_once = false;
                    continue;
                }
            }
            // With image filtering disabled, a recovery refinement can only
            // unlock an unregistered image. Once reconstruction is complete it
            // duplicates the final iterative refinement below. Filtering is the
            // exception: even a complete model may need this round to demote a
            // weak pose, so preserve the recovery whenever `filter_images` is on.
            if n_reg == n_images && !config.filter_images {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: growth complete at {n_reg}/{n_images}; \
                         skipping redundant stall-recovery refinement",
                    );
                }
                break;
            }
            if initial_poses.is_none() && config.colmap_style_mapper && !stalled_once {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: growth stalled at {n_reg}/{n_images} registered — \
                         forcing one stall-recovery refinement and retrying",
                    );
                }
                let started = std::time::Instant::now();
                growth_global_refinement(
                    &mut cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?;
                global_refinement_seconds += started.elapsed().as_secs_f64();
                global_refinement_calls += 1;
                reg_at_last_global = poses.iter().filter(|p| p.is_some()).count();
                if config.filter_images {
                    filter_images(&cam, features, tracks, config, &mut poses, &track_point);
                }
                stalled_once = true;
                continue;
            }
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: growth exhausted at {n_reg}/{n_images} registered \
                     (colmap_style_mapper={}, stalled_once={stalled_once})",
                    config.colmap_style_mapper,
                );
                for line in diagnose_unregistered_images(
                    obs_by_image,
                    &poses,
                    &trials,
                    max_trials,
                    &track_point,
                ) {
                    eprintln!("sfm-debug: {line}");
                }
            }
            break;
        };
        trials[next_image] += 1;

        // P3P (Grunert) is the default minimal solver — well-posed on coplanar
        // façades where the linear DLT degenerates. Both share the Gauss-Newton
        // refiner and the config reprojection gate.
        let started = std::time::Instant::now();
        let report = match config.pnp_solver {
            PnpSolver::P3p => PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                // COLMAP-style dynamic budget: `iterations` is a fail-safe
                // cap; the search exits once the best model's inlier ratio
                // implies 99.9% registration confidence. Large
                // correspondence sets (repetitive-texture scenes where the
                // inlier ratio can be tiny) need samples proportional to
                // their size; small clean sets keep the historical budget.
                iterations: if corrs.len() >= 64 {
                    config.pnp_max_iterations
                } else {
                    128
                },
                confidence: (config.pnp_max_iterations > 128).then_some(0.999),
                reprojection_threshold: config.max_reprojection_error_px,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
            }
            .estimate(&corrs, &cam),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: config.max_reprojection_error_px,
                confidence: Some(0.999),
                ..PnPRansac::default()
            }
            .estimate(&corrs, &cam),
        };
        pnp_seconds += started.elapsed().as_secs_f64();
        pnp_attempts += 1;
        let Some(report) = report else {
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: PnP attempt #{} on image {next_image} failed \
                     ({} corrs -> no valid pose, need >={})",
                    trials[next_image],
                    corrs.len(),
                    config.min_pnp_inliers,
                );
            }
            if initial_poses.is_none()
                && !config.colmap_style_mapper
                && sequence_fallback_enabled_during_growth(config)
            {
                if let Some(proposal) = sequence_relative_pose_fallback_with_overrides(
                    &cam,
                    features,
                    pairwise,
                    &poses,
                    config,
                    sequence_override_pair_indices,
                ) {
                    commit_sequence_relative_pose(
                        proposal,
                        &cam,
                        features,
                        tracks,
                        config,
                        &mut poses,
                        &mut track_point,
                        &mut correspondence_state,
                        &mut triangulation_seconds,
                        &mut registrations_since_ba,
                    );
                    stalled_once = false;
                }
            }
            continue; // registration failed this attempt (may be retried)
        };
        let attempt_inliers = report.inliers.len();
        if attempt_inliers < config.min_pnp_inliers {
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: PnP attempt #{} on image {next_image} failed \
                     ({} corrs -> {} inliers, need >={})",
                    trials[next_image],
                    corrs.len(),
                    attempt_inliers,
                    config.min_pnp_inliers,
                );
            }
            if config.debug_oracle_poses.is_some()
                && sfm_debug_image_enabled(next_image, debug_image_filter)
            {
                let pnp_ids =
                    pnp_track_ids(next_image, &corrs, features, obs_by_image, &track_point);
                log_pnp_geometry_diagnostic(
                    next_image,
                    &corrs,
                    &report.inliers,
                    &pnp_ids,
                    tracks,
                    &poses,
                    &cam,
                    &report.pose,
                    config.debug_oracle_poses.as_deref(),
                    debug_image_filter,
                );
            }
            if initial_poses.is_none()
                && !config.colmap_style_mapper
                && sequence_fallback_enabled_during_growth(config)
            {
                if let Some(proposal) = sequence_relative_pose_fallback_with_overrides(
                    &cam,
                    features,
                    pairwise,
                    &poses,
                    config,
                    sequence_override_pair_indices,
                ) {
                    commit_sequence_relative_pose(
                        proposal,
                        &cam,
                        features,
                        tracks,
                        config,
                        &mut poses,
                        &mut track_point,
                        &mut correspondence_state,
                        &mut triangulation_seconds,
                        &mut registrations_since_ba,
                    );
                    stalled_once = false;
                }
            }
            continue; // registration failed this attempt (may be retried)
        }
        if config.verify_registration_two_view
            && !pose_agrees_with_two_view_neighbors(
                &cam,
                features,
                pairwise,
                &poses,
                next_image,
                &report.pose,
                config.verify_registration_min_neighbors,
                config.verify_registration_min_agree_fraction,
            )
        {
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: PnP on image {next_image} rejected by two-view consistency \
                     ({} inliers)",
                    report.inliers.len()
                );
            }
            if initial_poses.is_none()
                && !config.colmap_style_mapper
                && sequence_fallback_enabled_during_growth(config)
            {
                if let Some(proposal) = sequence_relative_pose_fallback_with_overrides(
                    &cam,
                    features,
                    pairwise,
                    &poses,
                    config,
                    sequence_override_pair_indices,
                ) {
                    commit_sequence_relative_pose(
                        proposal,
                        &cam,
                        features,
                        tracks,
                        config,
                        &mut poses,
                        &mut track_point,
                        &mut correspondence_state,
                        &mut triangulation_seconds,
                        &mut registrations_since_ba,
                    );
                    stalled_once = false;
                }
            }
            continue;
        }
        if sfm_debug_enabled() {
            let inliers = report.inliers.len();
            let ratio = inliers as f64 / corrs.len() as f64;
            eprintln!(
                "sfm-debug: PnP attempt #{} on image {next_image} succeeded \
                 ({} corrs -> {} inliers, ratio={ratio:.3})",
                trials[next_image],
                corrs.len(),
                inliers,
            );
        }
        let pnp_poses_before = config.debug_oracle_poses.is_some().then(|| poses.clone());
        // Genuine progress — a future stall earns its own one-shot recovery
        // (see `stalled_once`'s module-level doc above).
        stalled_once = false;
        let report_pose = report.pose;
        poses[next_image] = Some(report_pose.clone());
        if config.debug_oracle_poses.is_some()
            && sfm_debug_image_enabled(next_image, debug_image_filter)
        {
            let pnp_ids = pnp_track_ids(next_image, &corrs, features, obs_by_image, &track_point);
            log_pnp_geometry_diagnostic(
                next_image,
                &corrs,
                &report.inliers,
                &pnp_ids,
                tracks,
                &poses,
                &cam,
                &report_pose,
                config.debug_oracle_poses.as_deref(),
                debug_image_filter,
            );
        }
        sfm_debug_oracle_transition(
            &format!(
                "pnp image={next_image} trial={} corrs={} inliers={}",
                trials[next_image],
                corrs.len(),
                attempt_inliers,
            ),
            pnp_poses_before.as_deref(),
            &poses,
            config.debug_oracle_poses.as_deref(),
        );
        let started = std::time::Instant::now();
        if !targeted_plain_growth_enabled(config, initial_poses.is_some()) {
            triangulate_pending_with_config_and_state(
                &cam,
                features,
                tracks,
                &poses,
                config,
                &mut track_point,
                correspondence_state.as_mut(),
            );
            triangulation_full_calls += 1;
        } else if needs_full_triangulation_after_ba {
            let newly_triangulated = triangulate_pending_track_ids(
                &cam,
                features,
                tracks,
                &poses,
                config,
                &mut track_point,
                0..tracks.len(),
            );
            triangulation_full_calls += 1;
            if let Some(counts) = correspondence_count_cache.as_mut() {
                update_correspondence_count_cache(
                    features,
                    tracks,
                    &track_point,
                    &newly_triangulated,
                    counts,
                );
            }
            needs_full_triangulation_after_ba = false;
        } else {
            let (visited, newly_triangulated) = triangulate_pending_for_image_with_new_tracks(
                &cam,
                features,
                tracks,
                &poses,
                config,
                &mut track_point,
                obs_by_image,
                next_image,
            );
            triangulation_targeted_tracks += visited;
            triangulation_targeted_calls += 1;
            if let Some(counts) = correspondence_count_cache.as_mut() {
                update_correspondence_count_cache(
                    features,
                    tracks,
                    &track_point,
                    &newly_triangulated,
                    counts,
                );
            }
        }
        triangulation_seconds += started.elapsed().as_secs_f64();

        if initial_poses.is_none() && config.colmap_style_mapper {
            // COLMAP `AdjustLocalBundle`: tighten the new image + its covisible
            // neighbourhood after every registration.
            let local_poses_before = config.debug_oracle_poses.is_some().then(|| poses.clone());
            let started = std::time::Instant::now();
            adjust_local_bundle(
                &cam,
                features,
                tracks,
                config,
                &mut poses,
                &mut track_point,
                next_image,
            )
            .map_err(IncrementalSfmError::Ba)?;
            local_ba_seconds += started.elapsed().as_secs_f64();
            local_ba_calls += 1;
            sfm_debug_oracle_transition(
                &format!("local_ba image={next_image}"),
                local_poses_before.as_deref(),
                &poses,
                config.debug_oracle_poses.as_deref(),
            );

            // Growth-ratio global refinement (COLMAP `IterativeGlobalRefinement`).
            // During the seed search `tracks` is shared read-only across trials,
            // so the in-growth refinement only re-triangulates + re-BAs (touching
            // this trial's own poses/points); the track-membership *filter* that
            // would mutate the shared tracks is deferred to the final refinement,
            // after a seed has been committed. The BA's Huber kernel keeps
            // outliers down-weighted in the meantime.
            let n_reg = poses.iter().filter(|p| p.is_some()).count();
            if n_reg as f64 >= reg_at_last_global as f64 * config.global_ba_images_ratio {
                let started = std::time::Instant::now();
                growth_global_refinement(
                    &mut cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                )
                .map_err(IncrementalSfmError::Ba)?;
                global_refinement_seconds += started.elapsed().as_secs_f64();
                global_refinement_calls += 1;
                reg_at_last_global = n_reg;
                // Structure changed — give previously-failed images a fresh shot
                // by resetting their trial counters (COLMAP retries on change).
                for (i, t) in trials.iter_mut().enumerate() {
                    if poses[i].is_none() {
                        *t = 0;
                    }
                }
                // COLMAP `FilterImages`: de-register images whose pose lost support
                // after the global solve. Done AFTER the retry reset so a filtered
                // image keeps its accumulated trial count (it is re-registered at
                // most `max_registration_trials` times, not indefinitely).
                if config.filter_images {
                    filter_images(&cam, features, tracks, config, &mut poses, &track_point);
                }
            }
        } else if initial_poses.is_none() {
            registrations_since_ba += 1;
            let n_reg = poses.iter().filter(|pose| pose.is_some()).count();
            let periodic_due = config.ba_every > 0 && registrations_since_ba >= config.ba_every;
            if periodic_due
                && !periodic_ba_due(
                    config.ba_every,
                    config.periodic_ba_min_registered_images,
                    registrations_since_ba,
                    n_reg,
                )
                && sfm_debug_enabled()
            {
                eprintln!(
                    "sfm-debug: periodic BA deferred at registered={n_reg} \
                     since_last={} threshold={} (minimum_registered={})",
                    registrations_since_ba,
                    config.ba_every,
                    config.periodic_ba_min_registered_images,
                );
            }
            if periodic_ba_due(
                config.ba_every,
                config.periodic_ba_min_registered_images,
                registrations_since_ba,
                n_reg,
            ) {
                // The simple schedule keeps intrinsics fixed during growth (refine
                // is a colmap-style / final-solve concern); refined slot is None.
                let support_before = sfm_ba_debug_enabled().then(|| {
                    (
                        poses.iter().filter(|pose| pose.is_some()).count(),
                        track_point.iter().filter(|point| point.is_some()).count(),
                        count_observations(tracks, &poses, &track_point),
                    )
                });
                run_bundle_adjustment(
                    &cam,
                    features,
                    tracks,
                    config,
                    &mut poses,
                    &mut track_point,
                    false,
                )
                .map_err(IncrementalSfmError::Ba)?;
                if let Some((poses_before, tracks_before, observations_before)) = support_before {
                    let poses_after = poses.iter().filter(|pose| pose.is_some()).count();
                    let tracks_after = track_point.iter().filter(|point| point.is_some()).count();
                    let observations_after = count_observations(tracks, &poses, &track_point);
                    eprintln!(
                        "sfm-debug-ba-support: stage=periodic registered {}=>{} \
                         tracks {}=>{} observations {}=>{} pruning=none",
                        poses_before,
                        poses_after,
                        tracks_before,
                        tracks_after,
                        observations_before,
                        observations_after,
                    );
                }
                registrations_since_ba = 0;
                needs_full_triangulation_after_ba = true;
            }
        }

        // Emit a bounded checkpoint stream for long runs.  The phase deltas
        // cover the registrations since the previous checkpoint, which makes
        // the dominant interval visible without enabling per-PnP provenance.
        let registered_now = poses.iter().filter(|pose| pose.is_some()).count();
        if timing_enabled && (registered_now <= 4 || registered_now % 64 == 0) {
            let elapsed = grow_started.elapsed().as_secs_f64();
            let ba_seconds = local_ba_seconds + global_refinement_seconds;
            eprintln!(
                "sfm-timing-progress: registered={registered_now}/{n_images} \
                 image={next_image} interval={:.3}s select={:.3}s pnp={:.3}s \
                 triangulate={:.3}s ba={:.3}s",
                elapsed - last_progress_elapsed,
                select_seconds - last_progress_select,
                pnp_seconds - last_progress_pnp,
                triangulation_seconds - last_progress_triangulation,
                ba_seconds - last_progress_ba,
            );
            last_progress_elapsed = elapsed;
            last_progress_select = select_seconds;
            last_progress_pnp = pnp_seconds;
            last_progress_triangulation = triangulation_seconds;
            last_progress_ba = ba_seconds;
        }
    }

    let registered = poses.iter().filter(|p| p.is_some()).count();
    if sfm_timing_or_debug_enabled() {
        eprintln!(
            "sfm-timing: grow total={:.3}s select={select_seconds:.3}s \
             pnp={pnp_seconds:.3}s/{pnp_attempts} triangulate={triangulation_seconds:.3}s \
             triangulation_scans={triangulation_full_calls} targeted_calls={triangulation_targeted_calls} \
             targeted_tracks={triangulation_targeted_tracks} \
             local_ba={local_ba_seconds:.3}s/{local_ba_calls} \
             global_refinement={global_refinement_seconds:.3}s/{global_refinement_calls}",
            grow_started.elapsed().as_secs_f64(),
        );
    }
    Ok((poses, track_point, registered, cam))
}

/// One bounded registration sweep after final global refinement. Unlike the
/// growth loop, this cannot cycle: every missing image receives at most one
/// attempt, and the caller invokes the function at most once.
pub(crate) fn post_refinement_registration_pass(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<usize, BaError> {
    let debug_image_filter = if sfm_debug_enabled() {
        sfm_debug_image_filter()
    } else {
        None
    };
    let mut obs_by_image: Vec<Vec<(usize, usize)>> = vec![Vec::new(); features.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, kp) in track {
            obs_by_image[image].push((kp, track_id));
        }
    }

    let mut trials = vec![0usize; features.len()];
    let mut registered = 0usize;
    while let Some((image, corrs)) = select_next_image(
        camera,
        config.next_image_policy,
        features,
        &obs_by_image,
        poses,
        &trials,
        1,
        track_point,
        None,
    ) {
        trials[image] = 1;
        let report = match config.pnp_solver {
            PnpSolver::P3p => PnPRansac {
                pose_estimator: P3PGrunert,
                pose_refiner: Some(GaussNewtonPoseRefiner::default()),
                iterations: config.pnp_max_iterations,
                confidence: Some(0.999),
                reprojection_threshold: config.max_reprojection_error_px,
                seed: 7,
                early_stop_min_iterations: 0,
                early_stop_inlier_ratio: None,
            }
            .estimate(&corrs, camera),
            PnpSolver::Dlt => PnPRansac {
                reprojection_threshold: config.max_reprojection_error_px,
                ..PnPRansac::default()
            }
            .estimate(&corrs, camera),
        };
        let attempt_inliers = report.as_ref().map(|r| r.inliers.len());
        let Some(report) = report.filter(|r| r.inliers.len() >= config.min_pnp_inliers) else {
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: post-refinement PnP on image {image} failed \
                     ({} corrs -> {} inliers, need >={})",
                    corrs.len(),
                    attempt_inliers.map_or("none".to_string(), |n| n.to_string()),
                    config.min_pnp_inliers,
                );
            }
            continue;
        };

        let pnp_poses_before = config.debug_oracle_poses.is_some().then(|| poses.to_vec());
        let report_pose = report.pose;
        poses[image] = Some(report_pose.clone());
        if config.debug_oracle_poses.is_some()
            && sfm_debug_image_enabled(image, debug_image_filter.as_ref())
        {
            let pnp_ids = pnp_track_ids(image, &corrs, features, &obs_by_image, track_point);
            log_pnp_geometry_diagnostic(
                image,
                &corrs,
                &report.inliers,
                &pnp_ids,
                tracks,
                poses,
                camera,
                &report_pose,
                config.debug_oracle_poses.as_deref(),
                debug_image_filter.as_ref(),
            );
        }
        sfm_debug_oracle_transition(
            &format!(
                "post_pnp image={image} corrs={} inliers={}",
                corrs.len(),
                attempt_inliers.unwrap_or_default(),
            ),
            pnp_poses_before.as_deref(),
            poses,
            config.debug_oracle_poses.as_deref(),
        );
        triangulate_pending_with_config(camera, features, tracks, poses, config, track_point);
        adjust_local_bundle(camera, features, tracks, config, poses, track_point, image)?;
        registered += 1;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: post-refinement registered image {image} \
                 ({} corrs, {} inliers)",
                corrs.len(),
                report.inliers.len(),
            );
        }
    }
    Ok(registered)
}

#[allow(clippy::too_many_arguments)]
/// Among unregistered images still under the per-image trial cap, choose the one
/// observing the most triangulated tracks, returning it with its 2D-3D
/// correspondences.
pub(super) fn select_next_image(
    camera: &Camera,
    policy: NextImagePolicy,
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    trials: &[usize],
    max_trials: usize,
    track_point: &[Option<Point3<f64>>],
    cached_counts: Option<&[usize]>,
) -> Option<(usize, Vec<Correspondence2D3D>)> {
    if policy == NextImagePolicy::CorrespondenceCount {
        // The historical count policy ranks only by the number of valid
        // triangulated observations.  Avoid allocating a correspondence
        // vector for every unregistered image on every iteration; build the
        // winning image's rows once after the rank scan.  This is exactly the
        // same key/tie order as the general path below (image order is the
        // stable tie breaker because equal keys are not replaced).
        let mut best: Option<(usize, usize)> = None;
        for (image, observations) in obs_by_image.iter().enumerate() {
            if poses[image].is_some() || trials[image] >= max_trials {
                continue;
            }
            let count = cached_counts
                .and_then(|counts| counts.get(image).copied())
                .unwrap_or_else(|| {
                    observations
                        .iter()
                        .filter(|&&(kp, track_id)| {
                            track_point[track_id].is_some()
                                && features[image].keypoints.get(kp).is_some()
                        })
                        .count()
                });
            if count < 6 {
                continue;
            }
            if best.is_none_or(|(_, best_count)| count > best_count) {
                best = Some((image, count));
            }
        }
        let (image, _) = best?;
        let corrs = obs_by_image[image]
            .iter()
            .filter_map(|&(kp, track_id)| {
                let point3d = track_point[track_id]?;
                let point2d = features[image].keypoints.get(kp).copied()?;
                Some(Correspondence2D3D {
                    point2d,
                    point3d,
                    confidence: None,
                })
            })
            .collect();
        return Some((image, corrs));
    }

    // COLMAP's `IncrementalMapper::RankNextImages`: rank candidate images not by
    // the raw *count* of 2D–3D correspondences but by a multi-resolution
    // **visibility-pyramid score** that rewards correspondences *well distributed*
    // across the image (better-conditioned PnP), with the count as a tiebreak. An
    // image with many points clustered in one corner is a worse next view than one
    // with fewer points spread over the frame, and this score prefers the latter.
    let mut best: Option<(usize, (usize, usize), Vec<Correspondence2D3D>)> = None;
    for (image, observations) in obs_by_image.iter().enumerate() {
        if poses[image].is_some() || trials[image] >= max_trials {
            continue;
        }
        let mut corrs = Vec::new();
        for &(kp, track_id) in observations {
            let Some(point3d) = track_point[track_id] else {
                continue;
            };
            let Some(point2d) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            corrs.push(Correspondence2D3D {
                point2d,
                point3d,
                confidence: None,
            });
        }
        if corrs.len() < 6 {
            continue; // DLT PnP needs ≥6
        }
        let key = next_image_rank(camera, policy, &corrs);
        if best.as_ref().is_none_or(|(_, b, _)| key > *b) {
            best = Some((image, key, corrs));
        }
    }
    best.map(|(image, _, corrs)| (image, corrs))
}

pub(super) fn next_image_rank(
    camera: &Camera,
    policy: NextImagePolicy,
    corrs: &[Correspondence2D3D],
) -> (usize, usize) {
    match policy {
        NextImagePolicy::Auto => unreachable!("Auto is resolved before the growth loop"),
        NextImagePolicy::VisibilityPyramid => (
            visibility_pyramid_score(
                camera.width,
                camera.height,
                corrs.iter().map(|corr| corr.point2d),
            ),
            corrs.len(),
        ),
        NextImagePolicy::CorrespondenceCount => (corrs.len(), 0),
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct NextImageAutoMetrics {
    pub(super) registered_images: usize,
    pub(super) valid_observations: usize,
    pub(super) tracks: usize,
    pub(super) mean_reprojection_px: f64,
}

pub(super) fn next_image_auto_metrics(result: &IncrementalSfmResult) -> NextImageAutoMetrics {
    NextImageAutoMetrics {
        registered_images: result.registered_images,
        valid_observations: result
            .tracks
            .iter()
            .map(|track| track.observations.len())
            .sum(),
        tracks: result.tracks.len(),
        mean_reprojection_px: result.mean_reprojection_px,
    }
}

/// Return whether an incomplete visibility candidate must be compared against
/// the raw correspondence-count candidate before post-refinement.  Running
/// the comparison for every incomplete result is intentional: a candidate
/// that misses only one image can still be less accurate than a complete
/// count-policy reconstruction.
pub(super) const fn next_image_auto_count_candidate_is_needed(
    registered_images: usize,
    total_images: usize,
) -> bool {
    registered_images < total_images
}

/// Auto's completion pass is considered only for a genuinely incomplete
/// model.  A complete primary candidate is returned without a second mapper
/// run, preserving both its bytes and its runtime.
pub(super) const fn next_image_auto_post_candidate_is_needed(
    registered_images: usize,
    total_images: usize,
) -> bool {
    next_image_auto_count_candidate_is_needed(registered_images, total_images)
}

/// Post-refinement completion is a registration-only fallback.  A candidate
/// is adopted only when it strictly adds registered images without worsening
/// the finite mean reprojection error.  Equal-support results retain the
/// untouched pre-post candidate, including its tracks, poses, and BA state.
pub(super) fn next_image_auto_post_candidate_is_better(
    candidate: &IncrementalSfmResult,
    incumbent: &IncrementalSfmResult,
) -> bool {
    if candidate.registered_images <= incumbent.registered_images {
        return false;
    }

    let candidate_error = candidate.mean_reprojection_px;
    if !candidate_error.is_finite() {
        return false;
    }

    let incumbent_error = incumbent.mean_reprojection_px;
    !incumbent_error.is_finite() || candidate_error <= incumbent_error
}

/// Compare two completed Auto candidates in the documented lexicographic
/// order.  Non-finite reprojection is treated as +∞.  Equality returns false
/// so the visibility-first candidate remains the deterministic tie winner.
pub(super) fn next_image_auto_candidate_is_better(
    candidate: &IncrementalSfmResult,
    incumbent: &IncrementalSfmResult,
) -> bool {
    next_image_auto_metrics_are_better(
        next_image_auto_metrics(candidate),
        next_image_auto_metrics(incumbent),
    )
}

pub(super) fn next_image_auto_metrics_are_better(
    candidate: NextImageAutoMetrics,
    incumbent: NextImageAutoMetrics,
) -> bool {
    let support_order = candidate
        .registered_images
        .cmp(&incumbent.registered_images)
        .then_with(|| {
            candidate
                .valid_observations
                .cmp(&incumbent.valid_observations)
        })
        .then_with(|| candidate.tracks.cmp(&incumbent.tracks));
    if support_order != Ordering::Equal {
        return support_order == Ordering::Greater;
    }

    let candidate_error = if candidate.mean_reprojection_px.is_finite() {
        candidate.mean_reprojection_px
    } else {
        f64::INFINITY
    };
    let incumbent_error = if incumbent.mean_reprojection_px.is_finite() {
        incumbent.mean_reprojection_px
    } else {
        f64::INFINITY
    };
    // `total_cmp` gives deterministic ordering even for signed zero; the
    // explicit `<` keeps exact metric ties with the visibility candidate.
    candidate_error.total_cmp(&incumbent_error) == Ordering::Less
}

/// M4 diagnosis helper (`docs/colmap_port_plan.md`'s "M4 results"): classify,
/// for every still-unregistered image, *why* [`select_next_image`] will not
/// offer it — genuinely insufficient 2D-3D correspondences to a triangulated
/// track (`< 6`, the DLT/P3P minimal-sample floor), or a sufficient count but
/// an exhausted `max_registration_trials` budget. Debug-only (gated by
/// [`sfm_debug_enabled`] at the call site); this does no RANSAC of its own —
/// it only counts correspondences, so it is cheap enough to call at every
/// growth stall without affecting the release path's behaviour or perf.
fn diagnose_unregistered_images(
    obs_by_image: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    trials: &[usize],
    max_trials: usize,
    track_point: &[Option<Point3<f64>>],
) -> Vec<String> {
    let mut lines = Vec::new();
    for (image, observations) in obs_by_image.iter().enumerate() {
        if poses[image].is_some() {
            continue;
        }
        let corr_count = observations
            .iter()
            .filter(|&&(_, track_id)| track_point[track_id].is_some())
            .count();
        let reason = if corr_count < 6 {
            format!("insufficient correspondences ({corr_count} < 6)")
        } else if trials[image] >= max_trials {
            format!(
                "trials exhausted ({}/{max_trials}, {corr_count} corrs available)",
                trials[image]
            )
        } else {
            format!(
                "eligible but not selected this round ({corr_count} corrs, {}/{max_trials} trials)",
                trials[image]
            )
        };
        lines.push(format!("  image {image}: {reason}"));
    }
    lines
}

/// COLMAP visibility-pyramid score (`Image::Point3DVisibilityScore`): occupancy of
/// a stack of grids at increasing resolution (`2×2`, `4×4`, … up to `64×64`), each
/// cell counted **once** regardless of how many points land in it. Spreading
/// observations across the frame lights up more cells at every level, so the score
/// rewards spatial distribution and saturates on clusters — unlike a raw point
/// count. Returns the number of occupied cells summed over all pyramid levels.
pub(super) fn visibility_pyramid_score(
    width: u32,
    height: u32,
    points: impl Iterator<Item = Point2<f64>>,
) -> usize {
    const NUM_LEVELS: u32 = 6;
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let mut occupied: Vec<HashSet<(u32, u32)>> = vec![HashSet::new(); NUM_LEVELS as usize];
    for p in points {
        // Clamp into the image so an out-of-frame keypoint cannot index past a grid.
        let fx = (p.x / w).clamp(0.0, 0.999_999);
        let fy = (p.y / h).clamp(0.0, 0.999_999);
        for level in 0..NUM_LEVELS {
            let dim = 1u32 << (level + 1); // 2, 4, 8, 16, 32, 64
            let cx = (fx * dim as f64) as u32;
            let cy = (fy * dim as f64) as u32;
            occupied[level as usize].insert((cx, cy));
        }
    }
    occupied.iter().map(|cells| cells.len()).sum()
}

/// Approximate the translational information in each track from its widest
/// calibrated baseline.  For two unit bearing rays, `sin²(theta) =
/// 1 - (r_i·r_j)²` is invariant to the E decomposition's acute/obtuse choice
/// and is the usual first-order baseline observability factor.  This is a
/// deliberately conservative proxy, not a covariance estimate: it is only
/// used by the opt-in final BA weighting mode below.
fn track_sin2_parallax(
    poses: &[Option<Pose>],
    track: &[(usize, usize)],
    point: Option<Point3<f64>>,
) -> Option<f64> {
    let point = point?;
    let mut rays = Vec::new();
    for &(image, _) in track {
        let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
            continue;
        };
        let delta = point - pose.camera_center_world();
        let norm = delta.norm();
        if norm.is_finite() && norm > 1e-12 {
            rays.push(delta / norm);
        }
    }
    if rays.len() < 2 {
        return None;
    }
    let mut best = 0.0f64;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            let dot = rays[i].dot(&rays[j]);
            if !dot.is_finite() {
                continue;
            }
            let sin2 = (1.0 - dot.clamp(-1.0, 1.0).powi(2)).clamp(0.0, 1.0);
            best = best.max(sin2);
        }
    }
    best.is_finite().then_some(best)
}

/// Build one deterministic weight for every monocular BA observation from the
/// current, pre-solve track geometry.  The median normalization keeps the
/// experiment numerically conservative; the `[0.25, 4]` clamp prevents a
/// single unusually wide/poor baseline from dominating or disappearing.  A
/// track with no usable pose/point geometry gets unit weight, so the mode never
/// silently removes an observation.  Track length is intentionally not a
/// second multiplier because each observation already contributes its own
/// residual and multiplying by length would double-count long tracks.
pub(super) fn track_geometry_observation_weights(
    poses: &[Option<Pose>],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    observations: &[BaObservation],
) -> Vec<f64> {
    let qualities: Vec<Option<f64>> = tracks
        .iter()
        .enumerate()
        .map(|(track_id, track)| {
            track_sin2_parallax(poses, track, track_point.get(track_id).copied().flatten())
        })
        .collect();
    let mut finite: Vec<f64> = qualities
        .iter()
        .flatten()
        .copied()
        .filter(|quality| quality.is_finite() && *quality > 0.0)
        .collect();
    finite.sort_by(f64::total_cmp);
    let median = finite
        .get(finite.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(1.0);
    if !median.is_finite() || median <= f64::EPSILON {
        return vec![1.0; observations.len()];
    }

    observations
        .iter()
        .map(|observation| {
            let quality = qualities
                .get(observation.landmark_id as usize)
                .and_then(|quality| *quality)
                .filter(|quality| quality.is_finite())
                .unwrap_or(median);
            (quality / median).clamp(0.25, 4.0)
        })
        .collect()
}
