//! Environment-gated diagnostics and oracle-pose metrics for the incremental mapper.

use super::*;

/// Gate for the mapper's diagnostic `eprintln!`s (seed-sweep reach, growth
/// stalls/recoveries). Off by default (checking an env var per print site is
/// cheap; this is not a hot inner loop). Added for the M4 path-dependence
/// diagnosis in `docs/colmap_port_plan.md` — set `VISLOC_SFM_DEBUG=1` to see,
/// per seed trial, how far it grew, and, per growth stall, whether it was a
/// genuine correspondence shortfall or a trial-budget exhaustion, and whether
/// the stall-recovery refinement ([`grow_from_seed`]'s `stalled_once`) helped.
/// Set `VISLOC_SFM_DEBUG_IMAGES=20,21` to restrict the per-PnP track-provenance
/// lines to those image indices while keeping the summary diagnostics enabled.
pub(super) fn sfm_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG").is_some()
}

/// Enable the bounded phase/progress timing stream without enabling the very
/// verbose per-image debug/provenance stream.  This is intentionally a
/// separate opt-in so large mapper runs can expose their growth intervals
/// without producing one record for every failed PnP attempt.
pub(super) fn sfm_timing_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_TIMING").is_some()
}

pub(super) fn sfm_timing_or_debug_enabled() -> bool {
    sfm_debug_enabled() || sfm_timing_enabled()
}

/// Opt into the matrix-free implicit-Schur PCG backend for the incremental
/// mapper's pure-visual bundle adjustments.
///
/// [`BaConfig::matrix_free_ba`] always wins; the environment variable is an
/// escape hatch for callers that cannot set the field directly. Ineligible
/// problems fall back to the ordinary solve inside
/// [`BundleAdjustment::optimize_honoring_matrix_free`].
pub(super) fn sfm_matrix_free_ba_enabled(config: &BaConfig) -> bool {
    config.matrix_free_ba || std::env::var_os("VISLOC_SFM_BA_MATRIX_FREE").is_some()
}

/// Emit an opt-in process-memory sample for benchmark phase boundaries.
///
/// The sampler is inactive unless `VISLOC_SFM_MEMORY=1` is present, and is
/// exposed so the example runner can mark feature/snapshot ownership stages
/// before entering the mapper. It never affects reconstruction state.
pub fn log_process_memory(stage: &str) {
    process_memory::log(stage);
}

/// Emit one compact diagnostic record for every BA invocation when explicitly
/// requested.  The ordinary SFM debug stream intentionally does not expose
/// solver internals; this opt-in record makes the iteration cap, LM damping,
/// accepted/rejected steps, and robust-vs-L2 objective observable without
/// changing the solve or its default output.
pub(super) fn sfm_ba_debug_enabled() -> bool {
    sfm_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA").is_some()
}

/// Emit one record per LM/Gauss--Newton trial in addition to the compact BA
/// summary.  This is deliberately a second opt-in because a reconstruction
/// can invoke many solves during growth and the per-trial stream is otherwise
/// unnecessarily noisy.
pub(super) fn sfm_ba_step_debug_enabled() -> bool {
    sfm_ba_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_STEPS").is_some()
}

/// Compare a small, deterministic sample of the live BA visual Jacobians to
/// central differences.  This is deliberately separate from the ordinary BA
/// and per-step diagnostics because even a bounded sample performs several
/// extra projections per observation.  It is off unless
/// `VISLOC_SFM_DEBUG_BA_JACOBIANS` is explicitly set together with the BA
/// debug flags.
pub(super) fn sfm_ba_jacobian_audit_enabled() -> bool {
    sfm_ba_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_JACOBIANS").is_some()
}

/// Emit the fixed-support landmark conditioning report in addition to the
/// compact BA summary.  This is intentionally separate from
/// `VISLOC_SFM_DEBUG_BA`: a full reconstruction can contain tens of thousands
/// of landmarks, while the report is useful only for a focused basin audit.
pub(super) fn sfm_ba_landmark_debug_enabled() -> bool {
    sfm_ba_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_LANDMARKS").is_some()
}

pub(super) fn parse_sfm_debug_images(raw: &str) -> Result<HashSet<usize>, String> {
    let mut images = HashSet::new();
    for token in raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let image = token
            .parse::<usize>()
            .map_err(|_| format!("invalid image index {token:?}"))?;
        images.insert(image);
    }
    if images.is_empty() {
        return Err("at least one image index is required".into());
    }
    Ok(images)
}

pub(super) fn sfm_debug_image_filter() -> Option<HashSet<usize>> {
    let raw = std::env::var("VISLOC_SFM_DEBUG_IMAGES").ok()?;
    match parse_sfm_debug_images(&raw) {
        Ok(images) => Some(images),
        Err(error) => {
            eprintln!("sfm-debug: ignoring invalid VISLOC_SFM_DEBUG_IMAGES={raw:?}: {error}");
            None
        }
    }
}

pub(super) fn sfm_debug_image_enabled(image: usize, filter: Option<&HashSet<usize>>) -> bool {
    sfm_debug_enabled() && filter.is_none_or(|images| images.contains(&image))
}

/// Capture the immutable source used by the pose-guided splitter.  In the
/// recovery+split composition this snapshot is taken before recovered tracks
/// are appended, so a later split cannot recursively consume recovery output.
/// Track-membership diagnostics deliberately return `None`: their supplied
/// partitions are already authoritative and are not eligible for this path.
type PoseGuidedSplitSource = (
    Vec<Vec<(usize, usize)>>,
    Vec<Vec<(usize, usize)>>,
    Vec<Option<Point3<f64>>>,
);

pub(super) fn capture_pose_guided_split_source(
    enabled: bool,
    track_membership: Option<&[Vec<(usize, usize)>]>,
    tracks: &[Vec<(usize, usize)>],
    conflicting_components: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
) -> Option<PoseGuidedSplitSource> {
    (enabled && track_membership.is_none()).then(|| {
        (
            tracks.to_vec(),
            conflicting_components.to_vec(),
            track_point.to_vec(),
        )
    })
}

/// Optional registration-time oracle diagnostics.  The vector is indexed like
/// the caller's `features`/`poses` slices; a missing entry means that the
/// corresponding image has no oracle pose.  This is deliberately kept as a
/// private, allocation-light report type: enabling the diagnostic must never
/// alter a pose, track, or BA decision.
#[derive(Debug, Clone)]
pub(super) struct SfmOracleMetrics {
    registered: usize,
    common: usize,
    pub(super) center_errors: Vec<Option<f64>>,
    pub(super) center_rmse: f64,
    center_median: f64,
    center_max: f64,
    pub(super) rotation_errors: Vec<Option<f64>>,
    pub(super) rotation_mean: f64,
    rotation_median: f64,
    rotation_max: f64,
}

pub(super) fn sfm_oracle_median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    values
        .get(values.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::NAN)
}

/// Compute a Sim(3)-aligned centre/rotation report for the currently posed
/// images.  This mirrors the example's oracle score, but lives here so every
/// incremental registration/BA transition can use exactly the same alignment.
/// Fewer than three common centres intentionally yields no metric: a two-view
/// pair does not constrain a meaningful diagnostic Sim(3).
pub(super) fn sfm_oracle_metrics(
    poses: &[Option<Pose>],
    oracle: &[Option<Pose>],
) -> Option<SfmOracleMetrics> {
    let registered = poses.iter().filter(|pose| pose.is_some()).count();
    let common_indices: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| {
            (pose.is_some() && oracle.get(image).and_then(Option::as_ref).is_some())
                .then_some(image)
        })
        .collect();
    if common_indices.len() < 3 {
        return None;
    }

    let source: Vec<Vector3<f64>> = common_indices
        .iter()
        .map(|&image| poses[image].as_ref().unwrap().camera_center_world().coords)
        .collect();
    let target: Vec<Vector3<f64>> = common_indices
        .iter()
        .map(|&image| oracle[image].as_ref().unwrap().camera_center_world().coords)
        .collect();
    let n = source.len() as f64;
    let source_mean = source.iter().copied().sum::<Vector3<f64>>() / n;
    let target_mean = target.iter().copied().sum::<Vector3<f64>>() / n;
    let mut covariance = Matrix3::zeros();
    let mut source_variance = 0.0;
    for (src, dst) in source.iter().zip(&target) {
        let src_zero = *src - source_mean;
        let dst_zero = *dst - target_mean;
        covariance += dst_zero * src_zero.transpose();
        source_variance += src_zero.norm_squared();
    }
    source_variance /= n;
    if !source_variance.is_finite() || source_variance <= f64::EPSILON {
        return None;
    }
    covariance /= n;
    let svd = covariance.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let mut correction = Matrix3::identity();
    if u.determinant() * v_t.determinant() < 0.0 {
        correction[(2, 2)] = -1.0;
    }
    let rotation = u * correction * v_t;
    let numerator = svd.singular_values[0] * correction[(0, 0)]
        + svd.singular_values[1] * correction[(1, 1)]
        + svd.singular_values[2] * correction[(2, 2)];
    let scale = numerator / source_variance;
    if !scale.is_finite() || scale <= 0.0 || !rotation.iter().all(|value| value.is_finite()) {
        return None;
    }
    let translation = target_mean - scale * (rotation * source_mean);
    if !translation.iter().all(|value| value.is_finite()) {
        return None;
    }
    let align_rotation = UnitQuaternion::from_matrix(&rotation);

    let mut center_errors = vec![None; poses.len()];
    let mut rotation_errors = vec![None; poses.len()];
    for &image in &common_indices {
        let pose = poses[image].as_ref().unwrap();
        let oracle_pose = oracle[image].as_ref().unwrap();
        let aligned_center = scale * (rotation * pose.camera_center_world().coords) + translation;
        let center_error = (aligned_center - oracle_pose.camera_center_world().coords).norm();
        let aligned_orientation = align_rotation * pose.camera_to_world().rotation;
        let rotation_error = (oracle_pose.camera_to_world().rotation.inverse()
            * aligned_orientation)
            .angle()
            .to_degrees();
        if center_error.is_finite() {
            center_errors[image] = Some(center_error);
        }
        if rotation_error.is_finite() {
            rotation_errors[image] = Some(rotation_error);
        }
    }
    let finite_centres: Vec<f64> = center_errors.iter().flatten().copied().collect();
    let finite_rotations: Vec<f64> = rotation_errors.iter().flatten().copied().collect();
    if finite_centres.is_empty() || finite_rotations.is_empty() {
        return None;
    }
    let center_rmse = (finite_centres
        .iter()
        .map(|error| error * error)
        .sum::<f64>()
        / finite_centres.len() as f64)
        .sqrt();
    let center_median = {
        let mut values = finite_centres.clone();
        sfm_oracle_median(&mut values)
    };
    let center_max = finite_centres.iter().copied().fold(0.0, f64::max);
    let rotation_mean = finite_rotations.iter().sum::<f64>() / finite_rotations.len() as f64;
    let rotation_median = {
        let mut values = finite_rotations.clone();
        sfm_oracle_median(&mut values)
    };
    let rotation_max = finite_rotations.iter().copied().fold(0.0, f64::max);
    Some(SfmOracleMetrics {
        registered,
        common: common_indices.len(),
        center_errors,
        center_rmse,
        center_median,
        center_max,
        rotation_errors,
        rotation_mean,
        rotation_median,
        rotation_max,
    })
}

/// Log one before/after transition.  This is the only consumer of the new
/// oracle field; when it is `None`, even with `VISLOC_SFM_DEBUG=1`, this helper
/// returns immediately and the normal mapper has no extra work or output.
pub(super) fn sfm_debug_oracle_transition(
    label: &str,
    before: Option<&[Option<Pose>]>,
    after: &[Option<Pose>],
    oracle: Option<&[Option<Pose>]>,
) {
    if !sfm_debug_enabled() {
        return;
    }
    let Some(oracle) = oracle else { return };
    let after_metrics = sfm_oracle_metrics(after, oracle);
    let before_metrics = before.and_then(|poses| sfm_oracle_metrics(poses, oracle));
    let Some(after_metrics) = after_metrics else {
        eprintln!(
            "sfm-debug-oracle: step={label} registered={} common<3 (alignment unavailable)",
            after.iter().filter(|pose| pose.is_some()).count(),
        );
        return;
    };
    let delta = before_metrics
        .as_ref()
        .map(|metrics| (after_metrics.center_rmse - metrics.center_rmse) * 100.0);
    let delta_rotation = before_metrics
        .as_ref()
        .map(|metrics| after_metrics.rotation_mean - metrics.rotation_mean);
    let delta_text = delta.map_or_else(|| "n/a".to_string(), |value| format!("{value:+.4}"));
    let delta_rotation_text =
        delta_rotation.map_or_else(|| "n/a".to_string(), |value| format!("{value:+.4}"));
    let mut worst: Vec<(usize, f64, f64)> = after_metrics
        .center_errors
        .iter()
        .enumerate()
        .filter_map(|(image, center)| {
            let center = (*center)?;
            let rotation = after_metrics.rotation_errors[image].unwrap_or(f64::NAN);
            Some((image, center, rotation))
        })
        .collect();
    worst.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let worst_text = worst
        .iter()
        .take(3)
        .map(|(image, center, rotation)| {
            format!("{image}:{:.3}cm/{rotation:.2}deg", center * 100.0)
        })
        .collect::<Vec<_>>()
        .join(",");
    let mut changed: Vec<(usize, f64, f64)> = before_metrics
        .as_ref()
        .into_iter()
        .flat_map(|metrics| {
            after_metrics
                .center_errors
                .iter()
                .enumerate()
                .filter_map(move |(image, after)| {
                    let (Some(before), Some(after)) = (*metrics.center_errors.get(image)?, *after)
                    else {
                        return None;
                    };
                    Some((image, (after - before) * 100.0, after * 100.0))
                })
        })
        .collect();
    changed.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()).then(a.0.cmp(&b.0)));
    let changed_text = changed
        .iter()
        .take(3)
        .map(|(image, delta, after)| format!("{image}:{delta:+.3}->{after:.3}cm"))
        .collect::<Vec<_>>()
        .join(",");
    eprintln!(
        concat!(
            "sfm-debug-oracle: step={} registered={} common={} ",
            "center_rmse={:.4}cm delta={} median={:.4}cm max={:.4}cm ",
            "rotation_mean={:.3}deg delta={} median={:.3}deg max={:.3}deg ",
            "worst=[{}] changed=[{}]"
        ),
        label,
        after_metrics.registered,
        after_metrics.common,
        after_metrics.center_rmse * 100.0,
        delta_text,
        after_metrics.center_median * 100.0,
        after_metrics.center_max * 100.0,
        after_metrics.rotation_mean,
        delta_rotation_text,
        after_metrics.rotation_median,
        after_metrics.rotation_max,
        worst_text,
        changed_text,
    );
}
