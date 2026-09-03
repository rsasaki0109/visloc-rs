//! Incremental structure-from-motion from an **unordered** image set.
//!
//! The stereo-VO SfM path ([`crate::stereo_vo_ba`]) assumes an *ordered* video
//! stream: temporal frame→frame matches give forward feature tracks, and stereo
//! gives metric scale for free. That is the wrong shape for a photo collection,
//! where the images have no temporal order, no known overlap graph, and (in the
//! monocular case) no metric scale. This module is the COLMAP-style answer: it
//! takes per-image features plus a set of **geometrically verified pairwise
//! matches** (any source — VLAD-retrieved candidate pairs filtered by an
//! essential-matrix RANSAC) and grows one consistent reconstruction.
//!
//! Pipeline:
//! 1. **Tracks.** Union-find over every `(image, keypoint)` node joined by a
//!    pairwise match. Each connected component is a feature track — one 3D
//!    point seen by many images. Tracks with two keypoints in the *same* image
//!    are inconsistent and dropped.
//! 2. **Seed.** Candidate pairs (most matches first, enough parallax) bootstrap
//!    the reconstruction via two-view relative pose ([`visloc_vision::two_view`]);
//!    the candidate that grows the most images is kept, so a repetitive scene
//!    whose strongest pair is an isolated cluster of adjacent frames is not
//!    trapped. This fixes the gauge (seed image at the origin) and the arbitrary
//!    monocular scale.
//! 3. **Grow.** Repeatedly register the unregistered image that observes the
//!    most already-triangulated tracks, by PnP RANSAC
//!    ([`visloc_vision::ransac`]); then triangulate every track that two
//!    registered views now share with sufficient parallax.
//! 4. **Bundle-adjust.** Periodically and at the end, refine all registered
//!    poses and triangulated points jointly with the Schur-complement BA
//!    ([`crate::bundle`]). Monocular has a 7-DoF gauge (6 rigid + scale), so two
//!    poses are fixed — the anchor and the longest-baseline pose — to pin scale
//!    as well as the frame.
//! 5. **Filter (+ optional re-triangulate).** Post-BA, strip observations that
//!    reproject past the gate (a contaminated union-find track) and drop tracks
//!    whose re-measured parallax is below the gate (depth-ambiguous far-flung
//!    points); optionally also **re-triangulate** against the BA-refined poses
//!    (`retriangulate`, off by default) — completing tracks the narrow seed-time
//!    baseline could not triangulate and re-seeding noisy points (guarded so an
//!    already-better point is never regressed), a density lever for downstream
//!    3DGS/NeRF — then re-optimise, a few rounds. No image is ever un-posed, so
//!    registration is invariant.
//!
//! The output ([`IncrementalSfmResult`]) carries per-image poses (`None` for
//! images that never registered) and merged multi-view tracks, ready for a
//! COLMAP `points3D.txt` export and downstream 3DGS / NeRF training.

use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet};
use std::io::Write;

use nalgebra::{Matrix2x3, Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::pnp::{Correspondence2D3D, GaussNewtonPoseRefiner, P3PGrunert, PoseRefiner};
use visloc_vision::ransac::{PnPRansac, RobustPoseEstimator};
use visloc_vision::stereo_bootstrap::triangulate_two_view_left_frame;
use visloc_vision::two_view::{
    recover_relative_pose_with_options, CheiralityOptions, ConfigurationType, CorrespondenceGraph,
    RelativePoseEstimator, TwoViewCorrespondence,
};

use crate::process_memory;
use crate::{BaConfig, BaError, BaObservation, BaResult, BundleAdjustment, RobustKernel};

/// Gate for the mapper's diagnostic `eprintln!`s (seed-sweep reach, growth
/// stalls/recoveries). Off by default (checking an env var per print site is
/// cheap; this is not a hot inner loop). Added for the M4 path-dependence
/// diagnosis in `docs/colmap_port_plan.md` — set `VISLOC_SFM_DEBUG=1` to see,
/// per seed trial, how far it grew, and, per growth stall, whether it was a
/// genuine correspondence shortfall or a trial-budget exhaustion, and whether
/// the stall-recovery refinement ([`grow_from_seed`]'s `stalled_once`) helped.
/// Set `VISLOC_SFM_DEBUG_IMAGES=20,21` to restrict the per-PnP track-provenance
/// lines to those image indices while keeping the summary diagnostics enabled.
fn sfm_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG").is_some()
}

/// Enable the bounded phase/progress timing stream without enabling the very
/// verbose per-image debug/provenance stream.  This is intentionally a
/// separate opt-in so large mapper runs can expose their growth intervals
/// without producing one record for every failed PnP attempt.
fn sfm_timing_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_TIMING").is_some()
}

fn sfm_timing_or_debug_enabled() -> bool {
    sfm_debug_enabled() || sfm_timing_enabled()
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
fn sfm_ba_debug_enabled() -> bool {
    sfm_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA").is_some()
}

/// Emit one record per LM/Gauss--Newton trial in addition to the compact BA
/// summary.  This is deliberately a second opt-in because a reconstruction
/// can invoke many solves during growth and the per-trial stream is otherwise
/// unnecessarily noisy.
fn sfm_ba_step_debug_enabled() -> bool {
    sfm_ba_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_STEPS").is_some()
}

/// Compare a small, deterministic sample of the live BA visual Jacobians to
/// central differences.  This is deliberately separate from the ordinary BA
/// and per-step diagnostics because even a bounded sample performs several
/// extra projections per observation.  It is off unless
/// `VISLOC_SFM_DEBUG_BA_JACOBIANS` is explicitly set together with the BA
/// debug flags.
fn sfm_ba_jacobian_audit_enabled() -> bool {
    sfm_ba_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_JACOBIANS").is_some()
}

/// Emit the fixed-support landmark conditioning report in addition to the
/// compact BA summary.  This is intentionally separate from
/// `VISLOC_SFM_DEBUG_BA`: a full reconstruction can contain tens of thousands
/// of landmarks, while the report is useful only for a focused basin audit.
fn sfm_ba_landmark_debug_enabled() -> bool {
    sfm_ba_debug_enabled() && std::env::var_os("VISLOC_SFM_DEBUG_BA_LANDMARKS").is_some()
}

fn parse_sfm_debug_images(raw: &str) -> Result<HashSet<usize>, String> {
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

fn sfm_debug_image_filter() -> Option<HashSet<usize>> {
    let raw = std::env::var("VISLOC_SFM_DEBUG_IMAGES").ok()?;
    match parse_sfm_debug_images(&raw) {
        Ok(images) => Some(images),
        Err(error) => {
            eprintln!("sfm-debug: ignoring invalid VISLOC_SFM_DEBUG_IMAGES={raw:?}: {error}");
            None
        }
    }
}

fn sfm_debug_image_enabled(image: usize, filter: Option<&HashSet<usize>>) -> bool {
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

fn capture_pose_guided_split_source(
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
struct SfmOracleMetrics {
    registered: usize,
    common: usize,
    center_errors: Vec<Option<f64>>,
    center_rmse: f64,
    center_median: f64,
    center_max: f64,
    rotation_errors: Vec<Option<f64>>,
    rotation_mean: f64,
    rotation_median: f64,
    rotation_max: f64,
}

fn sfm_oracle_median(values: &mut [f64]) -> f64 {
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
fn sfm_oracle_metrics(poses: &[Option<Pose>], oracle: &[Option<Pose>]) -> Option<SfmOracleMetrics> {
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
fn sfm_debug_oracle_transition(
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

/// Geometrically verified matches between two images of the set. The match
/// indices are keypoint indices into `features[image_i]` / `features[image_j]`,
/// and are assumed to have already survived an essential-matrix RANSAC (i.e.
/// they are inliers, not raw descriptor nearest neighbours).
#[derive(Debug, Clone, PartialEq)]
pub struct PairwiseMatches {
    /// Index of the first image into the `features` slice.
    pub image_i: usize,
    /// Index of the second image into the `features` slice.
    pub image_j: usize,
    /// Verified `(keypoint_in_i, keypoint_in_j)` correspondences.
    pub matches: Vec<(usize, usize)>,
    /// COLMAP two-view configuration when known (full E/F/H verifier). Used by
    /// global SfM to drop planar/panoramic pairs whose essential translation is
    /// ill-conditioned. `None` preserves legacy behaviour for essential-only
    /// verification paths.
    pub two_view_config: Option<ConfigurationType>,
    /// Essential-matrix inliers when the full verifier estimated E (may differ
    /// from [`Self::matches`] when F/H won the COLMAP inlier selection). Used
    /// by opt-in global edge construction so tracks stay dense while bearings
    /// come from E.
    pub essential_matches: Option<Vec<(usize, usize)>>,
    /// Essential matrix from full two-view verification (when estimated).
    /// Global SfM can decompose this directly for prefer-E edges instead of
    /// re-running E RANSAC on the inlier subset (which can flip chirality).
    pub essential_matrix: Option<Matrix3<f64>>,
}

impl PairwiseMatches {
    /// Construct matches without a known two-view configuration.
    pub fn new(image_i: usize, image_j: usize, matches: Vec<(usize, usize)>) -> Self {
        Self {
            image_i,
            image_j,
            matches,
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        }
    }
}

/// Tunable knobs for [`incremental_sfm`].
#[derive(Debug, Clone, PartialEq)]
pub struct IncrementalSfmConfig {
    /// A pair must contribute at least this many verified matches to be a
    /// candidate seed pair. (Track building still uses *all* pairs.)
    pub min_seed_matches: usize,
    /// How many candidate seeds to grow before committing. The highest-match
    /// pair is not always a good seed: on repetitive structure (a building with
    /// near-identical façades) the most-overlapping pair can be an isolated local
    /// cluster of a few adjacent frames that the reconstruction cannot grow out
    /// of. So up to `seed_trials` candidate pairs are each grown and the one that
    /// registers the most images is kept — the COLMAP-style robust-initialisation
    /// pattern — committing early as soon as a seed reaches most of its connected
    /// component (so a well-connected scene still grows exactly one). Pairs that
    /// fail the two-view baseline gate place nothing and don't count against the
    /// budget. `1` restores the old first-qualifying-seed behaviour.
    pub seed_trials: usize,
    /// Optional diagnostic restriction to one normalized `(image_i, image_j)`
    /// seed pair. `None` preserves the normal descending-match candidate list;
    /// this is intentionally opt-in so controlled seed replays do not alter
    /// ordinary reconstruction behavior.
    pub seed_pair: Option<(usize, usize)>,
    /// Minimum triangulation (parallax) angle in degrees for a point to be
    /// accepted. Small-angle triangulations are depth-unstable and dropped.
    pub min_triangulation_angle_deg: f64,
    /// Maximum reprojection error (px) for a triangulated point in each of the
    /// two views used to triangulate it, and the PnP inlier threshold.
    pub max_reprojection_error_px: f64,
    /// A track must span at least this many distinct images to be kept.
    pub min_track_length: usize,
    /// Optional final-only minimum track length.  When set, tracks shorter
    /// than this value are removed only after registration and all configured
    /// pose-guided splitting/recovery passes have completed, then the
    /// remaining support is re-triangulated and bundle-adjusted.  `None`
    /// preserves the historical growth/PnP/final-support behavior exactly.
    /// The example CLI currently exposes only `Some(3)` as its first guarded
    /// experiment; keeping this separate from `min_track_length` is what makes
    /// the diagnostic unable to change registration history.
    pub final_min_track_length: Option<usize>,
    /// Minimum PnP inliers to accept a new image registration.
    pub min_pnp_inliers: usize,
    /// Run a global bundle adjustment after every `ba_every` registrations.
    /// `0` disables the periodic BA (only the final BA runs).
    pub ba_every: usize,
    /// Defer the plain-growth periodic BA until at least this many cameras are
    /// registered. `0` preserves the historical `ba_every` schedule. This is
    /// intentionally scoped to the simple periodic path; COLMAP-style growth
    /// uses its own local/global schedule, and this knob never suppresses the
    /// configured final BA.
    pub periodic_ba_min_registered_images: usize,
    /// Run a final global bundle adjustment over the whole reconstruction.
    pub final_global_ba: bool,
    /// Bundle-adjustment configuration shared by the periodic and final solves.
    pub ba_config: BaConfig,
    /// Optional final fixed-support least-squares polish. When non-zero, one
    /// additional BA solve runs after all registration/refinement passes with
    /// the exact existing pose/track/observation support, no retriangulation or
    /// filtering, fixed intrinsics, and a pure L2 objective. A failed or
    /// cost-increasing solve is rolled back. `0` preserves the historical
    /// schedule exactly.
    pub final_ba_polish_iterations: usize,
    /// Minimal solver the PnP RANSAC uses to register each new image.
    pub pnp_solver: PnpSolver,
    /// Maximum absolute-pose RANSAC iterations. The dynamic termination
    /// (confidence 0.999) usually exits far earlier; this cap governs
    /// heavily contaminated correspondence sets.
    pub pnp_max_iterations: usize,
    /// Post-BA track-refinement rounds. Each round removes observations that
    /// reproject worse than `max_reprojection_error_px` after the global BA —
    /// the symptom of a contaminated union-find track whose merged 3D point
    /// fits none of its observations — and re-optimises. Registration is
    /// **invariant** (no image is ever un-posed), so this only cleans structure
    /// and can never drop a registered camera; on a clean reconstruction it is
    /// a near-no-op. `0` disables it.
    pub track_filter_iterations: usize,
    /// Re-triangulate tracks in each post-BA refinement round (COLMAP's
    /// completeness/refinement step the single-pass growth lacks). Once a global
    /// BA has moved the poses, two things change: a track that failed the
    /// parallax gate at growth time (a narrow baseline *then*) can now triangulate
    /// against the BA-refined wide-baseline views, and a point first triangulated
    /// from a narrow seed-time baseline can be re-seeded from the current widest
    /// pair. Completion is unconditional; the re-seed of an existing point is a
    /// **guarded swap** — kept only if it lowers that track's mean reprojection —
    /// so a multi-view point BA already placed better is never regressed. When
    /// enabled, at least one post-BA refinement round always runs (even if
    /// `track_filter_iterations` is `0`).
    ///
    /// **`false` by default.** Growth already triangulates greedily (every
    /// un-triangulated track is retried after *every* registration against all
    /// registered views), so by the end the structure is near-complete and the
    /// post-BA pass only mops up the marginal, gate-grazing tracks. Measured on a
    /// 300-frame EuRoC MH_03 monocular subset it adds ~3 % more tracks / ~1.5 %
    /// more observations — useful **density** for a downstream 3DGS/NeRF model —
    /// but is **ATE-neutral-to-slightly-negative** (Sim(3) 2.13 → 2.27 cm), since
    /// the extra tracks are the weakly-constrained ones. Enable it when you want
    /// the densest possible structure and can spend the extra BA rounds; leave it
    /// off when trajectory accuracy is the goal. See
    /// `docs/sfm_vs_colmap_benchmark.md`.
    pub retriangulate: bool,
    /// Build conflict-free tracks from the verified correspondence stream and
    /// re-triangulate every live point whenever registration adds an
    /// observation.  This is the incremental correspondence/point-map path:
    /// the ordinary union-find track builder remains the default, while this
    /// opt-in mode keeps one observation-to-point owner and refuses a merge
    /// that would create a same-image conflict.  It deliberately uses the
    /// plain seed/growth/PnP schedule; `--colmap-style` is rejected by the
    /// example CLI when this mode is selected.
    pub incremental_correspondence_triangulation: bool,
    /// Use COLMAP's `IncrementalMapper` bundle-adjustment **schedule** instead of
    /// the simple "global BA every `ba_every` registrations + final BA" path.
    /// This is a faithful port of COLMAP's defaults — the lever that closes the
    /// small-scene monocular accuracy gap on COLMAP's home turf:
    ///
    ///  - **Local BA after every registration.** Optimise only the new image and
    ///    its most-covisible neighbours (`local_ba_num_images`) plus the points
    ///    they see, holding the rest of the reconstruction fixed — cheap, and it
    ///    keeps the freshly added geometry tight before drift can compound.
    ///  - **Growth-triggered global refinement.** When the registered-image count
    ///    has grown by `global_ba_images_ratio` since the last global solve, run
    ///    an iterative global refinement: global BA → re-triangulate/complete →
    ///    filter, looped until the changed-observation fraction falls below
    ///    `global_ba_change_rate` (≤ `global_ba_max_refinements` rounds).
    ///  - **Registration retries.** A PnP failure is not permanent; after a
    ///    global refinement adds structure, failed images are retried, up to
    ///    `max_registration_trials` attempts each — COLMAP registers every frame
    ///    where the simple single-attempt path leaves a tail unregistered.
    ///
    /// The final refinement is always the iterative global form when this is on.
    /// `false` by default (preserves the simple schedule and every existing test).
    pub colmap_style_mapper: bool,
    /// After plain (non-`colmap_style_mapper`) growth, run COLMAP's iterative
    /// global refinement (multi-round BA + filter + re-triangulate) instead of
    /// the simple one-shot final BA. Keeps the simple growth schedule (no
    /// per-registration local BA) while borrowing only the final polish pass.
    /// `false` by default.
    pub final_iterative_global_refinement: bool,
    /// COLMAP `Mapper.ba_local_num_images`: how many most-covisible registered
    /// images (besides the newly registered one) the per-registration local BA
    /// optimises. Only used when `colmap_style_mapper` is set.
    pub local_ba_num_images: usize,
    /// COLMAP `Mapper.ba_global_images_ratio`: trigger a global refinement once
    /// the registered-image count has grown by this factor since the last one.
    /// Only used when `colmap_style_mapper` is set.
    pub global_ba_images_ratio: f64,
    /// COLMAP `Mapper.ba_global_max_refinements`: max global BA → complete →
    /// filter rounds per global refinement. Only used when `colmap_style_mapper`.
    pub global_ba_max_refinements: usize,
    /// COLMAP `Mapper.ba_global_max_refinement_change_rate`: stop the global
    /// refinement loop once `changed_observations / total_observations` drops
    /// below this. Only used when `colmap_style_mapper` is set.
    pub global_ba_change_rate: f64,
    /// COLMAP `Mapper.max_reg_trials`: how many times a single image may be
    /// retried for registration (across global-refinement boundaries) before it
    /// is given up on. Only used when `colmap_style_mapper` is set.
    pub max_registration_trials: usize,
    /// After the final global refinement has tightened/re-triangulated the
    /// committed model, give every still-unregistered image one fresh PnP
    /// attempt against that updated structure. This is a bounded completion
    /// pass: counters are reset exactly once, no retry cycle is possible, and a
    /// second final refinement runs only when at least one image registers.
    /// Experimental and off by default.
    pub post_refinement_registration: bool,
    /// After ordinary 2D-3D PnP completion, try to place still-missing images
    /// from three or more registered neighbours' independently recovered
    /// relative poses. Translation scale is recovered by intersecting the
    /// neighbour-to-missing camera-centre direction lines in the existing
    /// reconstruction frame; a single essential pair is never sufficient.
    /// Experimental and off by default.
    pub structureless_registration: bool,
    /// After a successful PnP, require the absolute pose to agree (same
    /// translation hemisphere) with independent two-view essentials against
    /// already-registered neighbours. Rejects chirality-flipped façade
    /// registrations that still have low local reprojection. Default false.
    pub verify_registration_two_view: bool,
    /// Minimum neighbours with a usable two-view check before the gate may
    /// reject. Only used when `verify_registration_two_view` is set.
    pub verify_registration_min_neighbors: usize,
    /// Minimum fraction of checked neighbours that must agree (dot > 0).
    pub verify_registration_min_agree_fraction: f64,
    /// Maximum ascending-scan rounds of the structure-less completion pass.
    /// A single scan registers an image only when its consensus neighbours are
    /// *already* registered at the moment the scan reaches it, so a chain whose
    /// bridge image has a higher index than its dependent images (an island's
    /// entry point numbered above the images it unlocks — the courtyard-class
    /// second-component failure) is left behind by one pass. Each round feeds
    /// the images it registered back in as neighbours for the next round; the
    /// loop stops as soon as a round registers nothing. One round therefore
    /// reproduces the historical single-pass behaviour exactly.
    pub structureless_max_rounds: usize,
    /// Minimum registered relative-pose neighbours required to propose one
    /// structure-less camera pose.
    pub structureless_min_neighbors: usize,
    /// Minimum independently re-estimated essential inliers per neighbour.
    pub structureless_min_pair_inliers: usize,
    /// Maximum angular disagreement between neighbour-implied missing-camera
    /// rotations.
    pub structureless_max_rotation_disagreement_deg: f64,
    /// Minimum acute angle between any two camera-centre direction lines.
    pub structureless_min_intersection_angle_deg: f64,
    /// Maximum RMS line-intersection residual divided by the registered
    /// neighbour-centre spread.
    pub structureless_max_center_line_error_ratio: f64,
    /// Minimum signed neighbour-line parameter divided by neighbour spread.
    /// A small negative tolerance absorbs noisy intersections at an almost
    /// coincident adjacent frame without accepting a materially reversed
    /// essential translation direction.
    pub structureless_min_forward_ratio: f64,
    /// Minimum triangulated/reprojecting tracks required after tentative pose
    /// insertion and local refinement.
    pub structureless_min_support_tracks: usize,
    /// Maximum independent local-submap tracks synthesized from verified
    /// pairwise edges for one tentative structure-less insertion.
    pub structureless_max_local_tracks: usize,
    /// Minimum views per synthesized local-submap landmark. Two-view points
    /// are allowed because the camera pose itself already requires a separate
    /// multi-neighbour consensus.
    pub structureless_min_local_track_views: usize,
    /// Maximum mean reprojection error over the tentative image's supported
    /// tracks after local refinement.
    pub structureless_max_reprojection_error_px: f64,
    /// Maximum relative increase in the pre-existing model's mean reprojection
    /// error allowed when admitting one structure-less pose.
    pub structureless_max_clean_error_increase_ratio: f64,
    /// Revisit same-image-conflicted union-find components only after the normal
    /// reconstruction has produced trustworthy poses. Candidate landmarks are
    /// triangulated from verified edges, must agree in at least three registered
    /// views with cycle support, and enter one guarded global BA. The recovery is
    /// rolled back if it worsens the clean model's reprojection objective.
    /// Experimental and off by default.
    pub geometry_guided_conflict_recovery: bool,
    /// When enabled, allow a bounded sequence-aware registration fallback
    /// after ordinary PnP cannot place an image.  The example supplies unique
    /// numeric image-stem values through [`Self::sequence_stem_values`]; only
    /// an image whose stem is exactly one greater than an already registered
    /// predecessor is eligible.  The fallback uses a stable essential pose,
    /// the robust recent consecutive-step scale, and the normal triangulation
    /// gates before admitting a provisional pose.  `false` preserves the
    /// ordinary unordered PnP schedule exactly.
    pub sequence_relative_pose_fallback: bool,
    /// Defer sequence-relative fallback until the ordinary growth, conflict
    /// recovery, and post-refinement PnP stage has stalled.  After one
    /// provisional sequence pose is admitted, ordinary post-refinement PnP is
    /// resumed before another fallback is attempted.  This is experimental
    /// and off by default; `false` preserves eager fallback timing.
    pub sequence_fallback_after_post: bool,
    /// Use a constant-velocity projection of recent world-frame consecutive
    /// steps for the sequence fallback scale.  The projection is accepted
    /// only when positive, finite, and inside the existing robust median/MAD
    /// fence.  `false` preserves the historical median-magnitude estimator.
    pub sequence_constant_velocity_scale: bool,
    /// Use the constant-velocity projection without its strict local
    /// median/MAD fence.  Positive finite projections are still constrained
    /// to a broad 0.25x..4x recent-median scale range.  This is a separate
    /// experimental policy; `false` preserves both the strict projected mode
    /// and the historical median-magnitude mode.
    pub sequence_relaxed_constant_velocity_scale: bool,
    /// In the after-post sequence fallback, carry the accepted baseline
    /// magnitude from one consecutive provisional pose to the next.  The
    /// first fallback still uses the selected constant-velocity projection;
    /// a carried value is admitted only inside the broad 0.25x..4x
    /// recent-median sanity bounds.  A normal PnP/post registration clears
    /// the carry chain.  Experimental and off by default.
    pub sequence_fallback_carry_scale: bool,
    /// Trailing numeric stem values indexed like the feature/pose slices.
    /// `None` disables sequence lookup.  This metadata is intentionally kept
    /// outside `FeatureSet`, so library callers can opt in without changing
    /// feature files or the default unordered API.
    pub sequence_stem_values: Option<Vec<u64>>,
    /// Rebuild all legacy union-find components after a complete posed model
    /// exists.  The opt-in pass splits conflicting (and, when necessary,
    /// poorly fitting clean) components into deterministic 3-D hypotheses
    /// using fixed camera poses, then runs the ordinary final BA.  The legacy
    /// track builder and default schedule are unchanged when this is false.
    pub pose_guided_track_splitting: bool,
    /// Number of bounded outer pose-guided split attempts.  The value is
    /// ignored while `pose_guided_track_splitting` is false; its default of
    /// one preserves the original single-pass diagnostic.
    pub pose_guided_track_splitting_iterations: usize,
    /// Optional reprojection gate used only by pose-guided splitting.  `None`
    /// reuses `max_reprojection_error_px`, preserving the prior split exactly;
    /// the ordinary mapper's triangulation/PnP/filter gates are unaffected.
    pub pose_guided_split_max_reprojection_error_px: Option<f64>,
    /// When pose-guided splitting is enabled, require every observation added
    /// beyond its two-view anchor to have direct verified support from at
    /// least two distinct observations/images already in that hypothesis.
    /// Tracks with only the two-view anchor remain valid.  This is a separate
    /// experimental admission rule and is off by default so the original
    /// pose-guided partition remains reproducible.
    pub pose_guided_graph_support: bool,
    /// Before pose-guided splitting, opt into deterministic bridge-cut
    /// refinement of original correspondence components.  Only bridges whose
    /// two sides independently fit posed 3-D points while the combined side
    /// does not fit one point are cut; the legacy/default path is unchanged.
    pub pose_guided_bridge_cuts: bool,
    /// After pose-guided splitting, iteratively merge complementary tracks
    /// only when a verified cross-track edge exists and their union fits one
    /// posed 3-D point under the split reprojection gate.  The image sets of
    /// the two tracks must be disjoint, and one observation per image is
    /// enforced throughout.  `false` preserves the split-only partition.
    pub pose_guided_track_merging: bool,
    /// Optional reprojection gate used only while fitting post-split unions.
    /// `None` inherits `pose_guided_split_max_reprojection_error_px` (or the
    /// ordinary gate when the split override is absent).  The post-BA hard
    /// validation still uses `max_reprojection_error_px`.
    pub pose_guided_merge_max_reprojection_error_px: Option<f64>,
    /// Minimum registered views supporting a geometry-recovered conflict track.
    /// Values below three are clamped to three.
    pub conflict_recovery_min_views: usize,
    /// Maximum verified anchor edges tested per conflicted component, ranked by
    /// descending posed-view parallax. This bounds recovery work on large chains.
    pub conflict_recovery_max_hypotheses: usize,
    /// Per-observation reprojection gate for a geometry-recovered track.
    pub conflict_recovery_max_reprojection_error_px: f64,
    /// Maximum mean reprojection error of a recovered track before guarded BA.
    pub conflict_recovery_max_mean_reprojection_px: f64,
    /// Maximum relative increase allowed in the original clean tracks' mean
    /// reprojection after the single guarded recovery BA.
    pub conflict_recovery_max_clean_error_increase_ratio: f64,
    /// Multi-view exemption to the `min_triangulation_angle_deg` gate. A point on
    /// a forward-flying trajectory often subtends a parallax angle below the gate
    /// yet is **well-constrained** when many views observe it (each view adds a
    /// reprojection constraint on its 3 DoF). `None` keeps the strict angle gate
    /// for every track (the simple path). `Some(n)` keeps — and triangulates — a
    /// track whose widest parallax is between `low_parallax_min_angle_deg` and
    /// `min_triangulation_angle_deg` **if it has ≥ n registered observations**, so
    /// long low-parallax tracks survive while 2-view depth-ambiguous ones (which
    /// would slide freely along their ray and corrupt the poses) are still
    /// rejected. This is the lever that recovers COLMAP-grade structure density on
    /// forward-motion video without the accuracy collapse a blanket low gate
    /// causes. Used by both the simple and COLMAP-style paths when set.
    pub low_parallax_min_observations: Option<usize>,
    /// Lower parallax floor (degrees) for the multi-view exemption above: a track
    /// below this angle is dropped regardless of how many views see it (truly
    /// degenerate). Only consulted when `low_parallax_min_observations` is `Some`.
    pub low_parallax_min_angle_deg: f64,
    /// Refine the shared pinhole intrinsics `(fx, fy, cx, cy)` in the **final**
    /// global refinement (alternating BA ↔ intrinsics, see
    /// [`crate::BaConfig::refine_intrinsics`]). A slightly-off fixed calibration
    /// forces a residual onto the poses; letting the camera absorb it is COLMAP's
    /// lever for the last of the small-scene accuracy gap. Growth keeps the input
    /// intrinsics fixed; the refined camera emerges from the final solve and is
    /// returned in [`IncrementalSfmResult::refined_camera`]. `false` by default.
    pub refine_intrinsics: bool,
    /// COLMAP `Reconstruction::FilterImages`: after each growth global refinement,
    /// **de-register** any image whose count of well-supported 3D-point
    /// observations (triangulated, within `max_reprojection_error_px`) has fallen
    /// below `filter_min_image_observations`. A pose that BA + point filtering
    /// stripped of support is unreliable; dropping it (its trial counter resets, so
    /// it can re-register once the structure around it improves) keeps a bad pose
    /// from dragging the global solve. The two seed images are never filtered (they
    /// anchor the gauge), and the registered count is never taken below 3. Only
    /// used when `colmap_style_mapper` is set. `false` by default.
    pub filter_images: bool,
    /// Minimum well-supported observations an image must keep to stay registered
    /// under `filter_images`. Only consulted when `filter_images` is set.
    pub filter_min_image_observations: usize,
    /// Which algorithm builds step 1's feature tracks from `pairwise`. See
    /// [`TrackSource`]'s doc for the M2 background; `UnionFind` by default.
    pub track_source: TrackSource,
    /// Build tracks by processing verified correspondences in descending
    /// pair-level confidence and rejecting a merge that would introduce two
    /// keypoints from one image into a track. The confidence is deliberately
    /// limited to metadata retained by [`PairwiseMatches`]: verified inlier
    /// count, then essential-inlier count, with deterministic image/keypoint
    /// tie-breaks. When enabled it takes precedence over `track_source`.
    /// `false` preserves the legacy union-find/graph path exactly.
    pub confidence_ordered_tracks: bool,
    /// Build tracks with an opt-in per-correspondence geometric order. For
    /// finite E-supported matches from a `Calibrated` two-view model, the
    /// normalized Sampson residual is ordered first; F/H, degenerate, missing
    /// or invalid models deliberately fall back to the pair-level confidence
    /// order above rather than mixing incomparable residuals. Takes precedence
    /// over [`Self::confidence_ordered_tracks`]. `false` preserves the legacy
    /// path exactly.
    pub geometric_confidence_tracks: bool,
    /// Canonicalize track and observation iteration by stable physical
    /// feature keys (image id, quantized pixel coordinates, then descriptor
    /// contents) instead of input feature indices. This is useful when a
    /// caller has permuted feature rows but kept their coordinates and
    /// descriptors paired; `false` preserves the legacy index order exactly.
    pub stable_track_order: bool,
    /// Build tracks by prioritising accepted correspondences with stronger
    /// multi-view cycle support. For an edge `(i,a)-(j,b)`, a third image
    /// contributes support when both endpoints connect to the same feature in
    /// that image; distinct supporting images are ordered before the exact
    /// number of matching third-image features. Pair-level and, when safely
    /// available, calibrated-E residual confidence break ties. The legacy
    /// union-find path remains unchanged when this is `false`.
    pub cycle_supported_tracks: bool,
    /// Run one opt-in final fixed-support BA with observation weights derived
    /// from the pre-BA triangulation geometry. The proxy is a clamped,
    /// median-normalized `sin²(parallax)` information score; it changes no
    /// track or observation membership and is disabled by default.
    pub geometry_weighted_ba: bool,
    /// Apply a conditioning safeguard to landmark variables whose pre-BA point
    /// block is numerically ill-conditioned and whose current residual is
    /// already outside the reprojection gate. Such landmarks are excluded
    /// from this BA's residual rows (a fixed, high-residual point would exert
    /// a misleading camera pull). The deterministic geometry gate is disabled
    /// by default; well-fitting weak points remain ordinary BA variables.
    pub freeze_ill_conditioned_landmarks: bool,
    /// Run a bounded point-only BA with all currently registered camera poses
    /// and intrinsics fixed immediately before each global/periodic joint BA.
    /// This can absorb a large landmark correction before it enters the joint
    /// camera Schur system. `0` preserves the historical schedule exactly.
    pub landmark_ba_warm_start_iterations: usize,
    /// Minimum registered-camera count at which the landmark warm start is
    /// enabled. `0` applies it to every global/periodic BA; a positive value
    /// permits evidence-scoped experiments such as the first 27-camera BA.
    pub landmark_ba_warm_start_min_registered_images: usize,
    /// Optional COLMAP sparse-model poses used only by the registration-time
    /// `sfm-debug-oracle` transition log. The entries are indexed like
    /// `features` and `poses`; `None` disables the diagnostic completely.
    /// Supplying this field never changes registration, triangulation, or BA.
    pub debug_oracle_poses: Option<Vec<Option<Pose>>>,
    /// How unregistered images are ranked for the next PnP attempt. Raw
    /// correspondence count is the historical default; the visibility pyramid
    /// is an explicit opt-in for experiments that prefer spatial coverage.
    pub next_image_policy: NextImagePolicy,
    /// Seed pairs (`image_i`, `image_j`, normalized to `(min, max)`) that
    /// [`seed_candidate_order`] must skip. Empty by default. This is the
    /// mechanism `LocalSubmapBuilder::build`'s scale-pathology retry (see
    /// `crate::local_submap`, `NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(b)) uses
    /// to force a rebuild onto the *next*-ranked seed candidate after a
    /// previous seed pair (which reached `88/88` registration with
    /// unremarkable per-observation gates) produced an internally
    /// scale-exploded reconstruction: excluding the offending pair and
    /// re-running `incremental_sfm` deterministically walks to the next
    /// candidate in the same descending-match-count order, without
    /// perturbing any other seed-selection behaviour.
    pub excluded_seed_pairs: HashSet<(usize, usize)>,
}

/// Ranking policy for the next image offered to incremental PnP registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum NextImagePolicy {
    /// Try [`VisibilityPyramid`] first and, when it leaves any input image
    /// unregistered, rerun from the same immutable inputs with
    /// [`CorrespondenceCount`].  The better reconstruction is chosen
    /// deterministically by registered images, valid observations, tracks,
    /// then lower reprojection error.  Ties retain the visibility result.
    /// This policy is explicit and does not change the library default.
    Auto,
    /// Prefer spatially distributed 2D-3D support, then raw support count.
    VisibilityPyramid,
    /// Prefer the largest raw 2D-3D support count (the historical policy).
    #[default]
    CorrespondenceCount,
}

/// Which algorithm builds step 1's feature tracks from `pairwise` — the M2
/// port in `docs/colmap_port_plan.md` ("Persistent `CorrespondenceGraph`").
/// [`Self::UnionFind`] is the original ad hoc union-find
/// ([`build_tracks`]), kept as the default (see the M2 results section in
/// that doc for the ETH3D A/B that motivated staying opt-in rather than
/// flipping the default). [`Self::CorrespondenceGraph`] instead builds the
/// same tracks by routing through
/// `visloc_vision::two_view::correspondence_graph::CorrespondenceGraph`
/// ([`build_tracks_via_graph`]) — COLMAP's persistent view-graph object,
/// which also exposes `NumObservationsForImage`/`NumCorrespondencesForImage`/
/// `ExtractTransitiveCorrespondences`-style queries the union-find has no way
/// to answer, for future milestones (M4's transitive pairing, in particular).
/// Both paths are proven to produce byte-identical tracks on this crate's
/// existing fixtures (see the `graph_tracks_match_union_find_tracks_*` tests).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrackSource {
    /// The original ad hoc union-find over `(image, keypoint)` nodes.
    #[default]
    UnionFind,
    /// COLMAP-style persistent [`CorrespondenceGraph`] (M2 port).
    CorrespondenceGraph,
}

/// Minimal PnP solver used to register a new image against the reconstruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PnpSolver {
    /// 6-point Direct Linear Transform. Linear and fast, but **degenerate on
    /// coplanar points** — a flat building façade or planar patch yields a
    /// garbage pose. Kept for parity with the classic path.
    Dlt,
    /// Grunert's Perspective-Three-Point minimal solver. Geometrically
    /// well-posed for any three non-collinear points whether or not the scene
    /// is planar, so it registers planar façades the DLT cannot. The default.
    #[default]
    P3p,
}

impl Default for IncrementalSfmConfig {
    fn default() -> Self {
        Self {
            min_seed_matches: 30,
            seed_trials: 12,
            seed_pair: None,
            min_triangulation_angle_deg: 2.0,
            max_reprojection_error_px: 4.0,
            min_track_length: 2,
            final_min_track_length: None,
            min_pnp_inliers: 12,
            ba_every: 5,
            periodic_ba_min_registered_images: 0,
            final_global_ba: true,
            ba_config: BaConfig {
                robust_kernel: RobustKernel::Huber { delta: 3.0 },
                ..BaConfig::default()
            },
            final_ba_polish_iterations: 0,
            pnp_solver: PnpSolver::default(),
            // Legacy default: fixed 128-sample PnP with no dynamic
            // termination. Raising this above 128 opts into the COLMAP-style
            // confidence-based adaptive budget for large correspondence
            // sets.
            pnp_max_iterations: 128,
            track_filter_iterations: 2,
            retriangulate: false,
            incremental_correspondence_triangulation: false,
            // COLMAP IncrementalMapper defaults (off unless colmap_style_mapper).
            colmap_style_mapper: false,
            final_iterative_global_refinement: false,
            local_ba_num_images: 8,
            global_ba_images_ratio: 1.1,
            global_ba_max_refinements: 5,
            global_ba_change_rate: 0.0005,
            max_registration_trials: 3,
            post_refinement_registration: false,
            structureless_registration: false,
            verify_registration_two_view: false,
            verify_registration_min_neighbors: 2,
            verify_registration_min_agree_fraction: 0.5,
            structureless_max_rounds: 4,
            structureless_min_neighbors: 3,
            structureless_min_pair_inliers: 30,
            structureless_max_rotation_disagreement_deg: 3.0,
            structureless_min_intersection_angle_deg: 2.0,
            structureless_max_center_line_error_ratio: 0.25,
            structureless_min_forward_ratio: -0.005,
            structureless_min_support_tracks: 20,
            structureless_max_local_tracks: 512,
            structureless_min_local_track_views: 2,
            structureless_max_reprojection_error_px: 2.0,
            structureless_max_clean_error_increase_ratio: 0.001,
            geometry_guided_conflict_recovery: false,
            sequence_relative_pose_fallback: false,
            sequence_fallback_after_post: false,
            sequence_constant_velocity_scale: false,
            sequence_relaxed_constant_velocity_scale: false,
            sequence_fallback_carry_scale: false,
            sequence_stem_values: None,
            pose_guided_track_splitting: false,
            pose_guided_track_splitting_iterations: 1,
            pose_guided_split_max_reprojection_error_px: None,
            pose_guided_graph_support: false,
            pose_guided_bridge_cuts: false,
            pose_guided_track_merging: false,
            pose_guided_merge_max_reprojection_error_px: None,
            conflict_recovery_min_views: 3,
            conflict_recovery_max_hypotheses: 32,
            conflict_recovery_max_reprojection_error_px: 2.0,
            conflict_recovery_max_mean_reprojection_px: 1.0,
            conflict_recovery_max_clean_error_increase_ratio: 0.001,
            low_parallax_min_observations: None,
            low_parallax_min_angle_deg: 1.0,
            refine_intrinsics: false,
            filter_images: false,
            filter_min_image_observations: 15,
            track_source: TrackSource::default(),
            confidence_ordered_tracks: false,
            geometric_confidence_tracks: false,
            stable_track_order: false,
            cycle_supported_tracks: false,
            geometry_weighted_ba: false,
            freeze_ill_conditioned_landmarks: false,
            landmark_ba_warm_start_iterations: 0,
            landmark_ba_warm_start_min_registered_images: 0,
            debug_oracle_poses: None,
            next_image_policy: NextImagePolicy::default(),
            excluded_seed_pairs: HashSet::new(),
        }
    }
}

/// One reconstructed 3D point and the image observations that support it.
#[derive(Debug, Clone, PartialEq)]
pub struct SfmTrack {
    /// World-frame position (metres up to the monocular gauge scale).
    pub position: Point3<f64>,
    /// `(image_index, keypoint_index, pixel)` for every registered image that
    /// observes this point.
    pub observations: Vec<(usize, usize, Point2<f64>)>,
}

/// Diagnostics from feature-track construction before triangulation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TrackBuildStats {
    /// Verified pairwise correspondences offered to the track builder.
    pub input_correspondences: usize,
    /// Connected components formed before the minimum-length gate.
    pub connected_components: usize,
    /// Legacy components discarded because they contain one image twice.
    pub conflicting_components: usize,
    /// Observations contained in those discarded legacy components.
    pub conflicting_observations: usize,
    /// Tracks retained after conflict and minimum-length gates.
    pub retained_tracks: usize,
    /// Observations in retained tracks.
    pub retained_observations: usize,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct TrackBuildOutput {
    pub(crate) tracks: Vec<Vec<(usize, usize)>>,
    pub(crate) conflicting_components: Vec<Vec<(usize, usize)>>,
    pub(crate) stats: TrackBuildStats,
}

pub(crate) fn build_track_output(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
    camera: Option<&Camera>,
) -> TrackBuildOutput {
    let mut output = if config.incremental_correspondence_triangulation {
        build_tracks_incremental_correspondence(features, pairwise, config.min_track_length)
    } else if config.cycle_supported_tracks {
        build_tracks_cycle_supported(features, camera, pairwise, config.min_track_length)
    } else if config.geometric_confidence_tracks {
        if let Some(camera) = camera {
            build_tracks_geometric_confidence(features, camera, pairwise, config.min_track_length)
        } else {
            // The preflight API predates camera-aware residuals. Keep it useful
            // (and deterministic) when called without a camera by applying the
            // same explicit pair-level fallback used for non-E/H models.
            build_tracks_confidence_ordered(features.len(), pairwise, config.min_track_length)
        }
    } else if config.confidence_ordered_tracks {
        build_tracks_confidence_ordered(features.len(), pairwise, config.min_track_length)
    } else {
        match config.track_source {
            TrackSource::UnionFind => {
                build_tracks_detailed(features.len(), pairwise, config.min_track_length)
            }
            TrackSource::CorrespondenceGraph => {
                let tracks = build_tracks_via_graph(features, pairwise, config.min_track_length);
                let stats = TrackBuildStats {
                    input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
                    connected_components: tracks.len(),
                    retained_tracks: tracks.len(),
                    retained_observations: tracks.iter().map(Vec::len).sum(),
                    ..TrackBuildStats::default()
                };
                TrackBuildOutput {
                    tracks,
                    conflicting_components: Vec::new(),
                    stats,
                }
            }
        }
    };
    if config.stable_track_order {
        canonicalize_track_order(features, &mut output);
    }
    output
}

/// Number of coordinate units retained by the opt-in physical ordering key.
/// A micron in the image-coordinate units used by the feature files is far
/// below the precision that can affect a track decision, while the fixed grid
/// keeps the ordering independent of the source row number for normal feature
/// data.
const STABLE_TRACK_COORD_SCALE: f64 = 1_000_000.0;

fn stable_coordinate_key(value: f64) -> (u8, i64) {
    if value.is_finite() {
        (0, (value * STABLE_TRACK_COORD_SCALE).round() as i64)
    } else if value.is_nan() {
        (2, 0)
    } else if value.is_sign_negative() {
        (1, 0)
    } else {
        (3, 0)
    }
}

fn stable_descriptor_cmp(lhs: &[f32], rhs: &[f32]) -> Ordering {
    lhs.iter()
        .zip(rhs)
        .map(|(a, b)| a.total_cmp(b))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| lhs.len().cmp(&rhs.len()))
}

/// Compare two observations by a physical key rather than by their feature
/// row indices. Descriptor contents are only a deterministic tie-break for
/// co-located rows (FeatureSet currently retains no SIFT scale/orientation
/// metadata); the final index tie-break affects only byte-identical duplicate
/// rows that have no observable physical distinction.
fn stable_observation_cmp(
    features: &[FeatureSet],
    lhs: &(usize, usize),
    rhs: &(usize, usize),
) -> Ordering {
    let image_order = lhs.0.cmp(&rhs.0);
    if image_order != Ordering::Equal {
        return image_order;
    }
    let lhs_point = features.get(lhs.0).and_then(|set| set.keypoints.get(lhs.1));
    let rhs_point = features.get(rhs.0).and_then(|set| set.keypoints.get(rhs.1));
    let coordinate_order = match (lhs_point, rhs_point) {
        (Some(lhs), Some(rhs)) => stable_coordinate_key(lhs.x)
            .cmp(&stable_coordinate_key(rhs.x))
            .then_with(|| stable_coordinate_key(lhs.y).cmp(&stable_coordinate_key(rhs.y))),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    if coordinate_order != Ordering::Equal {
        return coordinate_order;
    }
    let descriptor_order = match (
        features
            .get(lhs.0)
            .and_then(|set| set.descriptors.get(lhs.1)),
        features
            .get(rhs.0)
            .and_then(|set| set.descriptors.get(rhs.1)),
    ) {
        (Some(lhs), Some(rhs)) => stable_descriptor_cmp(lhs, rhs),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    };
    descriptor_order.then_with(|| lhs.1.cmp(&rhs.1))
}

fn stable_track_cmp(
    features: &[FeatureSet],
    lhs: &[(usize, usize)],
    rhs: &[(usize, usize)],
) -> Ordering {
    lhs.iter()
        .zip(rhs)
        .map(|(lhs, rhs)| stable_observation_cmp(features, lhs, rhs))
        .find(|ordering| *ordering != Ordering::Equal)
        .unwrap_or_else(|| lhs.len().cmp(&rhs.len()))
}

/// Apply the physical ordering to every sequence consumed by the incremental
/// mapper. In particular, `tracks` drives both initial triangulation order and
/// the PnP correspondence order; sorting only the final exported points would
/// leave the order-sensitive growth path unchanged.
fn canonicalize_track_order(features: &[FeatureSet], output: &mut TrackBuildOutput) {
    for track in &mut output.tracks {
        track.sort_by(|lhs, rhs| stable_observation_cmp(features, lhs, rhs));
    }
    output
        .tracks
        .sort_by(|lhs, rhs| stable_track_cmp(features, lhs, rhs));
    for component in &mut output.conflicting_components {
        component.sort_by(|lhs, rhs| stable_observation_cmp(features, lhs, rhs));
    }
    output
        .conflicting_components
        .sort_by(|lhs, rhs| stable_track_cmp(features, lhs, rhs));
}

/// Build only the feature-track topology and return its diagnostics, without
/// seed selection, triangulation, registration, or bundle adjustment. This is
/// intended for cheap preflight rejection of a candidate view graph before an
/// expensive independent mapper arm is launched.
pub fn preview_track_build_stats(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    config: &IncrementalSfmConfig,
) -> TrackBuildStats {
    build_track_output(features, pairwise, config, None).stats
}

/// Output of [`incremental_sfm`].
#[derive(Debug, Clone)]
pub struct IncrementalSfmResult {
    /// Refined pose per input image; `None` for images that never registered.
    pub poses: Vec<Option<Pose>>,
    /// Reconstructed multi-view tracks (after the final BA, if enabled).
    pub tracks: Vec<SfmTrack>,
    /// Track-construction diagnostics measured before triangulation and BA.
    pub track_build_stats: TrackBuildStats,
    /// Number of images that registered into the reconstruction.
    pub registered_images: usize,
    /// Images added by the optional one-shot post-refinement completion pass.
    pub post_refinement_registered_images: usize,
    /// Images placed by the optional multi-neighbour relative-pose recovery
    /// after the ordinary post-refinement PnP pass.
    pub structureless_registered_images: usize,
    /// Conflict tracks admitted by the optional geometry-guided recovery gate.
    pub geometry_recovered_tracks: usize,
    /// Observations contained in admitted geometry-recovered tracks.
    pub geometry_recovered_observations: usize,
    /// Whether recovery was allowed to update poses through its guarded BA.
    /// Complete models use structure-only recovery and report `false`.
    pub geometry_recovery_pose_ba_applied: bool,
    /// Mean reprojection error (px) over every observation of every track.
    pub mean_reprojection_px: f64,
    /// Result of the final BA solve, if one ran.
    pub ba_result: Option<BaResult>,
    /// Refined camera intrinsics, when `config.refine_intrinsics` was set. The
    /// poses, tracks, and `mean_reprojection_px` are all expressed against *this*
    /// camera, so a COLMAP / 3DGS export must use it rather than the input camera.
    /// `None` when intrinsics refinement was off.
    pub refined_camera: Option<Camera>,
    /// Local index (into this call's `features`/`pairwise`) of the first
    /// image of the seed pair the winning growth trial started from.
    /// Purely observational — does not influence poses/tracks/gates.
    pub seed_image_i: usize,
    /// Local index of the second image of the winning seed pair.
    pub seed_image_j: usize,
    /// Number of verified matches in the winning seed pair (`pairwise[..].matches.len()`).
    pub seed_match_count: usize,
}

/// Why [`incremental_sfm`] could not build a reconstruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IncrementalSfmError {
    /// No verified pair met `min_seed_matches` / parallax to bootstrap from.
    NoSeedPair,
    /// The chosen seed pair's relative pose / initial triangulation failed.
    SeedInitFailed,
    /// An externally supplied diagnostic track partition violated the mapper's
    /// one-observation-per-image/index contract.
    InvalidTrackMembership(String),
    /// An opt-in initial-pose model did not satisfy the mapper's input
    /// contract (one pose per image, at least two finite poses).
    InvalidInitialPoses(String),
    /// A bundle-adjustment solve failed.
    Ba(BaError),
}

impl std::fmt::Display for IncrementalSfmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IncrementalSfmError::NoSeedPair => {
                write!(f, "no verified pair met the seed criteria")
            }
            IncrementalSfmError::SeedInitFailed => {
                write!(f, "seed pair relative-pose / triangulation failed")
            }
            IncrementalSfmError::InvalidTrackMembership(message) => {
                write!(f, "invalid track membership: {message}")
            }
            IncrementalSfmError::InvalidInitialPoses(message) => {
                write!(f, "invalid initial poses: {message}")
            }
            IncrementalSfmError::Ba(e) => write!(f, "bundle adjustment failed: {e:?}"),
        }
    }
}

impl std::error::Error for IncrementalSfmError {}

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
                if reach >= not_trapped || grows >= trials {
                    break;
                }
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

/// Union-find over `(image, keypoint)` nodes joined by pairwise matches. Returns
/// the consistent tracks (no two keypoints from the same image) spanning at
/// least `min_track_length` distinct images.
#[cfg(test)]
fn build_tracks(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    build_tracks_with_stats(n_images, pairwise, min_track_length).0
}

#[cfg(test)]
fn build_tracks_with_stats(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> (Vec<Vec<(usize, usize)>>, TrackBuildStats) {
    let output = build_tracks_detailed(n_images, pairwise, min_track_length);
    (output.tracks, output.stats)
}

pub(crate) fn build_tracks_detailed(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    let _ = n_images;
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        ..TrackBuildStats::default()
    };
    // Map each observed (image, keypoint) to a dense node id.
    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let node_of = |image: usize,
                   kp: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>|
     -> usize {
        *node_id.entry((image, kp)).or_insert_with(|| {
            nodes.push((image, kp));
            nodes.len() - 1
        })
    };

    let mut parent: Vec<usize> = Vec::new();
    let ensure = |id: usize, parent: &mut Vec<usize>| {
        while parent.len() <= id {
            let next = parent.len();
            parent.push(next);
        }
    };

    for pair in pairwise {
        for &(ki, kj) in &pair.matches {
            let a = node_of(pair.image_i, ki, &mut node_id, &mut nodes);
            let b = node_of(pair.image_j, kj, &mut node_id, &mut nodes);
            ensure(a, &mut parent);
            ensure(b, &mut parent);
            union(&mut parent, a, b);
        }
    }

    // Group nodes by representative root.
    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, kp)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, kp));
    }
    stats.connected_components = groups.len();

    let mut tracks = Vec::new();
    let mut conflicting_components = Vec::new();
    for (_root, mut obs) in groups {
        // Reject tracks with conflicting observations (same image twice): such
        // a component merged two distinct points through a bad match chain.
        let mut images_seen: HashMap<usize, usize> = HashMap::new();
        let mut conflict = false;
        for &(image, _kp) in &obs {
            let count = images_seen.entry(image).or_insert(0);
            *count += 1;
            if *count > 1 {
                conflict = true;
                break;
            }
        }
        if conflict {
            stats.conflicting_components += 1;
            stats.conflicting_observations += obs.len();
            obs.sort_unstable();
            conflicting_components.push(obs);
            continue;
        }
        if images_seen.len() >= min_track_length {
            obs.sort_unstable();
            tracks.push(obs);
        }
    }
    // Deterministic track order (the grouping `HashMap` iterates in a random
    // order per run): a stable order makes landmark ids — and therefore the
    // whole incremental reconstruction — reproducible.
    tracks.sort_unstable();
    conflicting_components.sort_unstable();
    stats.retained_tracks = tracks.len();
    stats.retained_observations = tracks.iter().map(Vec::len).sum();
    TrackBuildOutput {
        tracks,
        conflicting_components,
        stats,
    }
}

/// Accept a validated external observation partition as the mapper's track
/// topology.  The partition is intentionally copied and sorted so later
/// triangulation/PnP traversal is deterministic, while all point coordinates
/// are recomputed from the current camera and poses.  Validation of indices,
/// one observation per image, and cross-track ownership is performed by the
/// public incremental entry point before this helper is called.
fn build_track_output_from_membership(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    membership: &[Vec<(usize, usize)>],
) -> Result<TrackBuildOutput, String> {
    let mut seen = HashMap::<(usize, usize), usize>::new();
    let mut tracks = Vec::with_capacity(membership.len());
    for (track_id, source_track) in membership.iter().enumerate() {
        let mut images = HashSet::new();
        let mut track = source_track.clone();
        track.sort_unstable();
        for &(image, keypoint) in &track {
            if image >= features.len() {
                return Err(format!(
                    "track {track_id} references image {image}, but only {} images are loaded",
                    features.len()
                ));
            }
            if keypoint >= features[image].keypoints.len()
                || keypoint >= features[image].descriptors.len()
            {
                return Err(format!(
                    "track {track_id} references image {image} keypoint {keypoint}, outside the loaded feature set"
                ));
            }
            if !images.insert(image) {
                return Err(format!(
                    "track {track_id} contains more than one observation from image {image}"
                ));
            }
            if let Some(previous_track) = seen.insert((image, keypoint), track_id) {
                return Err(format!(
                    "observation ({image},{keypoint}) belongs to tracks {previous_track} and {track_id}"
                ));
            }
        }
        if images.len() >= min_track_length {
            tracks.push(track);
        }
    }
    tracks.sort_unstable();
    let stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: membership.len(),
        retained_tracks: tracks.len(),
        retained_observations: tracks.iter().map(Vec::len).sum(),
        ..TrackBuildStats::default()
    };
    Ok(TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    })
}

/// Build tracks incrementally from individual verified correspondences.
///
/// The legacy builder first takes an unrestricted transitive closure and then
/// discards the whole component when one image occurs twice.  That is a useful
/// compatibility baseline, but one bad edge can consequently hide otherwise
/// valid observations from the mapper.  This builder keeps an explicit
/// observation-to-track map while adding edges: a free observation extends a
/// track, two disjoint tracks are merged, and an edge that would introduce a
/// second observation from one image is rejected in isolation.  Edges are
/// sorted by their physical integer key, so the result does not depend on the
/// order in which a snapshot or matcher happened to emit them.
pub(crate) fn build_tracks_incremental_correspondence(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_tracks_incremental_correspondence_impl(features, pairwise, min_track_length, true)
}

/// Build conflict-preserving tracks in the verified input stream order.
///
/// This is intentionally separate from the physical-key-order policy above.
/// A caller can first supply a frozen, trusted correspondence prefix and then
/// append lower-priority bridge pairs.  Conflicting bridge edges are therefore
/// rejected without allowing their numeric image/keypoint ids to displace the
/// trusted prefix.  The caller is responsible for binding and checksumming the
/// input order when reproducibility matters.
pub(crate) fn build_tracks_incremental_correspondence_in_order(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_tracks_incremental_correspondence_impl(features, pairwise, min_track_length, false)
}

fn build_tracks_incremental_correspondence_impl(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    sort_edges: bool,
) -> TrackBuildOutput {
    #[derive(Debug, Default)]
    struct WorkingTrack {
        observations: Vec<(usize, usize)>,
        images: BTreeSet<usize>,
        active: bool,
    }

    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        ..TrackBuildStats::default()
    };
    let mut edges = Vec::new();
    for pair in pairwise {
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (keypoint_i, keypoint_j) = if swapped {
                (keypoint_j, keypoint_i)
            } else {
                (keypoint_i, keypoint_j)
            };
            // Unlike the historical union-find, the opt-in path does not
            // create unusable nodes for malformed imported rows.  The source
            // stream remains counted in `input_correspondences` above.
            if image_i == image_j
                || image_i >= features.len()
                || image_j >= features.len()
                || keypoint_i >= features[image_i].keypoints.len()
                || keypoint_j >= features[image_j].keypoints.len()
            {
                continue;
            }
            edges.push((image_i, image_j, keypoint_i, keypoint_j));
        }
    }
    if sort_edges {
        edges.sort_unstable();
    }

    let mut tracks = Vec::<WorkingTrack>::new();
    let mut observation_to_track = HashMap::<(usize, usize), usize>::new();
    let mut conflicting_components = Vec::new();

    let mut reject_conflict = |left: (usize, usize), right: (usize, usize)| {
        stats.conflicting_components += 1;
        stats.conflicting_observations += 2;
        conflicting_components.push(vec![left, right]);
    };

    for (image_i, image_j, keypoint_i, keypoint_j) in edges {
        let left = (image_i, keypoint_i);
        let right = (image_j, keypoint_j);
        let left_track = observation_to_track.get(&left).copied();
        let right_track = observation_to_track.get(&right).copied();
        match (left_track, right_track) {
            (None, None) => {
                let track_id = tracks.len();
                let mut images = BTreeSet::new();
                images.insert(image_i);
                images.insert(image_j);
                tracks.push(WorkingTrack {
                    observations: vec![left, right],
                    images,
                    active: true,
                });
                observation_to_track.insert(left, track_id);
                observation_to_track.insert(right, track_id);
            }
            (Some(track_id), None) | (None, Some(track_id)) => {
                let (observation, image) = if left_track.is_some() {
                    (right, image_j)
                } else {
                    (left, image_i)
                };
                let track = &mut tracks[track_id];
                if track.active && !track.images.contains(&image) {
                    track.observations.push(observation);
                    track.images.insert(image);
                    observation_to_track.insert(observation, track_id);
                } else {
                    reject_conflict(left, right);
                }
            }
            (Some(left_id), Some(right_id)) if left_id == right_id => {}
            (Some(left_id), Some(right_id)) => {
                let left_images = tracks[left_id].images.clone();
                let right_images = tracks[right_id].images.clone();
                if left_images.intersection(&right_images).next().is_some() {
                    reject_conflict(left, right);
                    continue;
                }
                // Union-by-size bounds map updates for highly connected view
                // graphs.  The id tie-break keeps equal-size cases stable.
                let (keep, drop) = if tracks[left_id].observations.len()
                    > tracks[right_id].observations.len()
                    || (tracks[left_id].observations.len() == tracks[right_id].observations.len()
                        && left_id < right_id)
                {
                    (left_id, right_id)
                } else {
                    (right_id, left_id)
                };
                let dropped = std::mem::take(&mut tracks[drop].observations);
                for observation in dropped {
                    observation_to_track.insert(observation, keep);
                    tracks[keep].observations.push(observation);
                }
                let dropped_images = std::mem::take(&mut tracks[drop].images);
                tracks[keep].images.extend(dropped_images);
                tracks[drop].active = false;
            }
        }
    }

    stats.connected_components = tracks.iter().filter(|track| track.active).count();
    let mut retained = Vec::new();
    for track in tracks.into_iter().filter(|track| track.active) {
        if track.observations.len() < min_track_length {
            continue;
        }
        let mut observations = track.observations;
        observations.sort_unstable();
        retained.push(observations);
    }
    retained.sort_unstable();
    conflicting_components.sort_unstable();
    stats.retained_tracks = retained.len();
    stats.retained_observations = retained.iter().map(Vec::len).sum();
    TrackBuildOutput {
        tracks: retained,
        conflicting_components,
        stats,
    }
}

/// Small, explicit observation-to-point state used by the incremental
/// correspondence triangulator.  Keeping this state separate from the
/// mapper's exported `SfmTrack` makes create/continue/merge conflict rules
/// unit-testable without a camera or a solver.
#[derive(Debug, Clone, Default)]
struct CorrespondencePointState {
    observation_to_point: HashMap<(usize, usize), usize>,
    observations: Vec<Vec<(usize, usize)>>,
    points: Vec<Option<Point3<f64>>>,
}

impl CorrespondencePointState {
    fn from_tracks(tracks: &[Vec<(usize, usize)>], points: &[Option<Point3<f64>>]) -> Self {
        let mut state = Self {
            observations: tracks.to_vec(),
            points: points.to_vec(),
            ..Self::default()
        };
        state.points.resize(tracks.len(), None);
        for (point_id, track) in tracks.iter().enumerate() {
            for &observation in track {
                state
                    .observation_to_point
                    .entry(observation)
                    .or_insert(point_id);
            }
        }
        state
    }

    #[cfg(test)]
    fn has_image(&self, point_id: usize, image: usize) -> bool {
        self.observations
            .get(point_id)
            .is_some_and(|track| track.iter().any(|&(track_image, _)| track_image == image))
    }

    #[cfg(test)]
    fn create_point(
        &mut self,
        observations: &[(usize, usize)],
        point: Point3<f64>,
    ) -> Option<usize> {
        if !point.coords.iter().all(|value| value.is_finite())
            || observations.is_empty()
            || observations.iter().enumerate().any(|(index, &(image, _))| {
                observations[..index]
                    .iter()
                    .any(|&(other_image, other_kp)| {
                        other_image == image
                            || self
                                .observation_to_point
                                .contains_key(&(other_image, other_kp))
                    })
            })
        {
            return None;
        }
        let point_id = self.observations.len();
        self.observations.push(observations.to_vec());
        self.points.push(Some(point));
        for &observation in observations {
            self.observation_to_point.insert(observation, point_id);
        }
        Some(point_id)
    }

    #[cfg(test)]
    fn continue_point(&mut self, point_id: usize, observation: (usize, usize)) -> bool {
        if point_id >= self.observations.len()
            || self.observation_to_point.contains_key(&observation)
            || self.has_image(point_id, observation.0)
        {
            return false;
        }
        self.observations[point_id].push(observation);
        self.observation_to_point.insert(observation, point_id);
        true
    }

    #[cfg(test)]
    fn merge_points(&mut self, left: usize, right: usize, point: Point3<f64>) -> bool {
        if left >= self.observations.len()
            || right >= self.observations.len()
            || left == right
            || !point.coords.iter().all(|value| value.is_finite())
        {
            return false;
        }
        if self.observations[left]
            .iter()
            .any(|&(image, _)| self.has_image(right, image))
        {
            return false;
        }
        let right_observations = std::mem::take(&mut self.observations[right]);
        for observation in right_observations {
            self.observation_to_point.insert(observation, left);
            self.observations[left].push(observation);
        }
        self.points[left] = Some(point);
        self.points[right] = None;
        true
    }

    fn retriangulate_point(&mut self, point_id: usize, point: Point3<f64>) -> bool {
        if point_id >= self.points.len() || !point.coords.iter().all(|value| value.is_finite()) {
            return false;
        }
        let changed = self.points[point_id] != Some(point);
        self.points[point_id] = Some(point);
        changed
    }
}

/// Build tracks in a deterministic confidence order while refusing a merge
/// that would put two observations from the same image in one component.
///
/// `PairwiseMatches` retains pair-level verified inlier counts (and, when the
/// full verifier ran, an essential-inlier count), but not per-match residuals
/// or descriptor distances. Those retained geometric counts are therefore the
/// complete confidence signal used here; no synthetic per-match score is
/// invented. The legacy builder remains untouched and is still the default.
pub(crate) fn build_tracks_confidence_ordered(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    build_tracks_confidence_ordered_impl(n_images, pairwise, min_track_length, 0)
}

/// Build confidence-ordered tracks while processing a trusted pair prefix
/// before every remaining pair.
///
/// Confidence ordering is retained independently inside both tiers. A frozen
/// base snapshot can therefore establish its proven tracks before newly
/// verified component bridges compete for observations.
pub(crate) fn build_tracks_confidence_ordered_with_trusted_prefix(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    trusted_pair_prefix: usize,
) -> TrackBuildOutput {
    build_tracks_confidence_ordered_impl(
        n_images,
        pairwise,
        min_track_length,
        trusted_pair_prefix.min(pairwise.len()),
    )
}

fn build_tracks_confidence_ordered_impl(
    n_images: usize,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
    trusted_pair_prefix: usize,
) -> TrackBuildOutput {
    let _ = n_images;
    #[derive(Clone, Copy)]
    struct Candidate {
        image_i: usize,
        image_j: usize,
        keypoint_i: usize,
        keypoint_j: usize,
        verified_inliers: usize,
        essential_inliers: usize,
        trusted: bool,
    }

    let mut candidates = Vec::new();
    for (pair_index, pair) in pairwise.iter().enumerate() {
        let essential_inliers = pair.essential_matches.as_ref().map_or(0, Vec::len);
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (keypoint_i, keypoint_j) = if swapped {
                (keypoint_j, keypoint_i)
            } else {
                (keypoint_i, keypoint_j)
            };
            candidates.push(Candidate {
                image_i,
                image_j,
                keypoint_i,
                keypoint_j,
                verified_inliers: pair.matches.len(),
                essential_inliers,
                trusted: pair_index < trusted_pair_prefix,
            });
        }
    }
    // Stronger verified pairs first. Every remaining field makes ties
    // independent of the input pair/vector order; duplicate candidates are
    // harmless because the second one finds the same component.
    candidates.sort_unstable_by(|a, b| {
        b.trusted
            .cmp(&a.trusted)
            .then_with(|| b.verified_inliers.cmp(&a.verified_inliers))
            .then_with(|| b.essential_inliers.cmp(&a.essential_inliers))
            .then_with(|| a.image_i.cmp(&b.image_i))
            .then_with(|| a.image_j.cmp(&b.image_j))
            .then_with(|| a.keypoint_i.cmp(&b.keypoint_i))
            .then_with(|| a.keypoint_j.cmp(&b.keypoint_j))
    });

    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images: Vec<HashSet<usize>> = Vec::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };
    let mut rejected_conflicts = 0usize;
    for candidate in candidates {
        let a = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let b = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            continue;
        }
        if component_images[ra]
            .iter()
            .any(|image| component_images[rb].contains(image))
        {
            rejected_conflicts += 1;
            continue;
        }
        // Union-by-size bounds the set movement; the root-id tie-break keeps
        // the topology independent of HashMap/set iteration details.
        let (root, child) = if component_size[ra] > component_size[rb]
            || (component_size[ra] == component_size[rb] && ra < rb)
        {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, keypoint)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, keypoint));
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for (_root, mut observations) in groups {
        observations.sort_unstable();
        let distinct_images = observations
            .iter()
            .map(|&(image, _)| image)
            .collect::<HashSet<_>>()
            .len();
        if distinct_images >= min_track_length {
            stats.retained_observations += observations.len();
            tracks.push(observations);
        }
    }
    tracks.sort_unstable();
    stats.retained_tracks = tracks.len();
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: confidence-ordered tracks rejected_conflicts={} retained_tracks={} retained_obs={}",
            rejected_conflicts, stats.retained_tracks, stats.retained_observations
        );
    }
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

/// Build tracks by preferring correspondences that are independently
/// supported by a third view.  An edge `(i,a)-(j,b)` has one exact cycle for
/// every feature `c` in a distinct image `k` for which both `(i,a)-(k,c)` and
/// `(j,b)-(k,c)` are accepted edges.  The number of distinct supporting
/// images is the primary score; the exact number of matching third-view
/// features is the secondary score.  This prevents duplicate matches in one
/// view from masquerading as broad multi-view support.
///
/// The edge list is deduplicated and sorted before the conflict-aware
/// union-find pass.  Consequently pair/vector input order cannot affect the
/// result except for physically indistinguishable, byte-identical feature
/// rows (where the existing stable feature-index fallback is unavoidable).
/// Pair-level verified/essential support and a calibrated-E Sampson residual
/// are used only after cycle support.  Descriptor distances are not retained
/// by `PairwiseMatches`, so no synthetic descriptor score is introduced.
#[allow(clippy::type_complexity)]
pub(crate) fn build_tracks_cycle_supported(
    features: &[FeatureSet],
    camera: Option<&Camera>,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    #[derive(Clone, Copy)]
    struct Candidate {
        image_i: usize,
        image_j: usize,
        keypoint_i: usize,
        keypoint_j: usize,
        distinct_third_images: usize,
        exact_cycles: usize,
        verified_inliers: usize,
        essential_inliers: usize,
        residual: Option<f64>,
    }

    // Keep both directions so a cycle lookup never depends on the orientation
    // in which a PairwiseMatches record happened to be supplied.
    type DirectedLookup = HashMap<(usize, usize), HashMap<usize, HashSet<usize>>>;
    let mut adjacency: DirectedLookup = HashMap::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        let forward = adjacency.entry((pair.image_i, pair.image_j)).or_default();
        for &(keypoint_i, keypoint_j) in &pair.matches {
            forward.entry(keypoint_i).or_default().insert(keypoint_j);
        }
        let reverse = adjacency.entry((pair.image_j, pair.image_i)).or_default();
        for &(keypoint_i, keypoint_j) in &pair.matches {
            reverse.entry(keypoint_j).or_default().insert(keypoint_i);
        }
    }

    // Keep the strongest metadata for duplicate endpoint rows.  The cycle
    // score is global and therefore computed once below from the deduplicated
    // physical edge rather than once per duplicate PairwiseMatches record.
    let mut endpoint_metadata: HashMap<(usize, usize, usize, usize), (usize, usize, Option<f64>)> =
        HashMap::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        let essential_set = if pair.two_view_config == Some(ConfigurationType::Calibrated) {
            pair.essential_matches
                .as_ref()
                .map(|matches| matches.iter().copied().collect::<HashSet<_>>())
        } else {
            None
        };
        for &(raw_keypoint_i, raw_keypoint_j) in &pair.matches {
            let residual = if essential_set
                .as_ref()
                .is_some_and(|set| set.contains(&(raw_keypoint_i, raw_keypoint_j)))
            {
                match (camera, pair.essential_matrix.as_ref()) {
                    (Some(camera), Some(essential)) => {
                        let point_i = features
                            .get(pair.image_i)
                            .and_then(|set| set.keypoints.get(raw_keypoint_i));
                        let point_j = features
                            .get(pair.image_j)
                            .and_then(|set| set.keypoints.get(raw_keypoint_j));
                        point_i.and_then(|point_i| {
                            point_j.and_then(|point_j| {
                                normalized_sampson_residual(camera, essential, point_i, point_j)
                            })
                        })
                    }
                    _ => None,
                }
            } else {
                None
            };
            let (image_i, image_j, keypoint_i, keypoint_j) = if pair.image_i < pair.image_j {
                (pair.image_i, pair.image_j, raw_keypoint_i, raw_keypoint_j)
            } else {
                (pair.image_j, pair.image_i, raw_keypoint_j, raw_keypoint_i)
            };
            let key = (image_i, image_j, keypoint_i, keypoint_j);
            let metadata = (
                pair.matches.len(),
                pair.essential_matches.as_ref().map_or(0, Vec::len),
                residual,
            );
            endpoint_metadata
                .entry(key)
                .and_modify(|existing| {
                    let residual_order = match (existing.2, metadata.2) {
                        (Some(lhs), Some(rhs)) => rhs.total_cmp(&lhs),
                        (None, Some(_)) => Ordering::Less,
                        (Some(_), None) => Ordering::Greater,
                        (None, None) => Ordering::Equal,
                    };
                    if metadata.0 > existing.0
                        || (metadata.0 == existing.0
                            && (metadata.1 > existing.1
                                || (metadata.1 == existing.1 && residual_order == Ordering::Less)))
                    {
                        *existing = metadata;
                    }
                })
                .or_insert(metadata);
        }
    }

    let mut candidates = Vec::with_capacity(endpoint_metadata.len());
    for (
        (image_i, image_j, keypoint_i, keypoint_j),
        (verified_inliers, essential_inliers, residual),
    ) in endpoint_metadata
    {
        let (distinct_third_images, exact_cycles) = cycle_support_for_edge(
            features.len(),
            image_i,
            keypoint_i,
            image_j,
            keypoint_j,
            &adjacency,
        );
        candidates.push(Candidate {
            image_i,
            image_j,
            keypoint_i,
            keypoint_j,
            distinct_third_images,
            exact_cycles,
            verified_inliers,
            essential_inliers,
            residual,
        });
    }

    candidates.sort_unstable_by(|a, b| {
        b.distinct_third_images
            .cmp(&a.distinct_third_images)
            .then_with(|| b.exact_cycles.cmp(&a.exact_cycles))
            .then_with(|| match (a.residual, b.residual) {
                (Some(lhs), Some(rhs)) => lhs.total_cmp(&rhs),
                (Some(_), None) => Ordering::Less,
                (None, Some(_)) => Ordering::Greater,
                (None, None) => Ordering::Equal,
            })
            .then_with(|| b.verified_inliers.cmp(&a.verified_inliers))
            .then_with(|| b.essential_inliers.cmp(&a.essential_inliers))
            .then_with(|| {
                stable_observation_cmp(
                    features,
                    &(a.image_i, a.keypoint_i),
                    &(b.image_i, b.keypoint_i),
                )
            })
            .then_with(|| {
                stable_observation_cmp(
                    features,
                    &(a.image_j, a.keypoint_j),
                    &(b.image_j, b.keypoint_j),
                )
            })
    });

    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images: Vec<HashSet<usize>> = Vec::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };

    for candidate in candidates {
        let a = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let b = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb
            || component_images[ra]
                .iter()
                .any(|image| component_images[rb].contains(image))
        {
            continue;
        }
        let (root, child) = if component_size[ra] > component_size[rb]
            || (component_size[ra] == component_size[rb] && ra < rb)
        {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, keypoint)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, keypoint));
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for (_root, mut observations) in groups {
        let distinct_images = observations
            .iter()
            .map(|&(image, _)| image)
            .collect::<HashSet<_>>()
            .len();
        if distinct_images >= min_track_length {
            observations.sort_by(|lhs, rhs| stable_observation_cmp(features, lhs, rhs));
            stats.retained_observations += observations.len();
            tracks.push(observations);
        }
    }
    tracks.sort_by(|lhs, rhs| stable_track_cmp(features, lhs, rhs));
    stats.retained_tracks = tracks.len();
    if sfm_debug_enabled() {
        let cycle_edges = adjacency
            .values()
            .map(|directed| directed.values().map(HashSet::len).sum::<usize>())
            .sum::<usize>();
        eprintln!(
            "sfm-debug: cycle-supported tracks edges={} directed_edges={} retained_tracks={} retained_obs={}",
            stats.input_correspondences,
            cycle_edges,
            stats.retained_tracks,
            stats.retained_observations,
        );
    }
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

fn cycle_support_for_edge(
    n_images: usize,
    image_i: usize,
    keypoint_i: usize,
    image_j: usize,
    keypoint_j: usize,
    adjacency: &HashMap<(usize, usize), HashMap<usize, HashSet<usize>>>,
) -> (usize, usize) {
    let mut distinct_third_images = 0;
    let mut exact_cycles = 0;
    for image_k in 0..n_images {
        if image_k == image_i || image_k == image_j {
            continue;
        }
        let left = adjacency
            .get(&(image_i, image_k))
            .and_then(|map| map.get(&keypoint_i));
        let right = adjacency
            .get(&(image_j, image_k))
            .and_then(|map| map.get(&keypoint_j));
        let Some((left, right)) = left.zip(right) else {
            continue;
        };
        let exact_here = left.intersection(right).count();
        if exact_here != 0 {
            distinct_third_images += 1;
            exact_cycles += exact_here;
        }
    }
    (distinct_third_images, exact_cycles)
}

/// Compute the dimensionless normalized Sampson residual for one calibrated
/// correspondence. The essential matrix is scale-invariant, and the pixels
/// are undistorted/normalized through the camera before evaluating
/// `x_jᵀ E x_i`. Returning `None` for any non-finite or degenerate quantity is
/// deliberate: callers must not order an invalid residual ahead of a valid
/// one or compare it with a pixel-space F/H error.
fn normalized_sampson_residual(
    camera: &Camera,
    essential: &Matrix3<f64>,
    point_i: &Point2<f64>,
    point_j: &Point2<f64>,
) -> Option<f64> {
    if !essential.iter().all(|value| value.is_finite())
        || !point_i.x.is_finite()
        || !point_i.y.is_finite()
        || !point_j.x.is_finite()
        || !point_j.y.is_finite()
    {
        return None;
    }
    let normalized_i = camera.normalize_pixel(point_i)?;
    let normalized_j = camera.normalize_pixel(point_j)?;
    if !normalized_i.x.is_finite()
        || !normalized_i.y.is_finite()
        || !normalized_j.x.is_finite()
        || !normalized_j.y.is_finite()
    {
        return None;
    }
    let bearing_i = Vector3::new(normalized_i.x, normalized_i.y, 1.0);
    let bearing_j = Vector3::new(normalized_j.x, normalized_j.y, 1.0);
    let epipolar_i = essential * bearing_i;
    let epipolar_j = essential.transpose() * bearing_j;
    let numerator = bearing_j.dot(&epipolar_i);
    let denominator = epipolar_i.x * epipolar_i.x
        + epipolar_i.y * epipolar_i.y
        + epipolar_j.x * epipolar_j.x
        + epipolar_j.y * epipolar_j.y;
    if !numerator.is_finite() || !denominator.is_finite() || denominator <= 1.0e-24 {
        return None;
    }
    let residual = numerator.abs() / denominator.sqrt();
    residual.is_finite().then_some(residual)
}

/// Build tracks with per-correspondence normalized Sampson confidence where it
/// is safe to do so. Only E-supported matches from a `Calibrated` verifier
/// result receive a residual: F-won, planar/panoramic, watermark, multiple,
/// degenerate, missing-model, and invalid entries all use the pair-level
/// fallback. This keeps model families incomparable by design while retaining
/// the historical pair-level confidence policy for all unsupported edges.
pub(crate) fn build_tracks_geometric_confidence(
    features: &[FeatureSet],
    camera: &Camera,
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> TrackBuildOutput {
    #[derive(Clone, Copy)]
    struct Candidate {
        image_i: usize,
        image_j: usize,
        keypoint_i: usize,
        keypoint_j: usize,
        verified_inliers: usize,
        essential_inliers: usize,
        residual: Option<f64>,
    }

    let mut candidates = Vec::new();
    for pair in pairwise {
        let essential_inliers = pair.essential_matches.as_ref().map_or(0, Vec::len);
        // E/F agreement is the explicit safety gate. An essential matrix on a
        // F-won, planar, or otherwise ambiguous pair is not a comparable
        // confidence score for the winning match set.
        let essential_set = if pair.two_view_config == Some(ConfigurationType::Calibrated) {
            pair.essential_matches
                .as_ref()
                .map(|matches| matches.iter().copied().collect::<HashSet<_>>())
        } else {
            None
        };
        let (image_i, image_j, swapped) = if pair.image_i <= pair.image_j {
            (pair.image_i, pair.image_j, false)
        } else {
            (pair.image_j, pair.image_i, true)
        };
        for &(raw_keypoint_i, raw_keypoint_j) in &pair.matches {
            let residual = if essential_set
                .as_ref()
                .is_some_and(|set| set.contains(&(raw_keypoint_i, raw_keypoint_j)))
            {
                pair.essential_matrix.as_ref().and_then(|essential| {
                    let point_i = features
                        .get(pair.image_i)
                        .and_then(|set| set.keypoints.get(raw_keypoint_i))?;
                    let point_j = features
                        .get(pair.image_j)
                        .and_then(|set| set.keypoints.get(raw_keypoint_j))?;
                    normalized_sampson_residual(camera, essential, point_i, point_j)
                })
            } else {
                None
            };
            let (keypoint_i, keypoint_j) = if swapped {
                (raw_keypoint_j, raw_keypoint_i)
            } else {
                (raw_keypoint_i, raw_keypoint_j)
            };
            candidates.push(Candidate {
                image_i,
                image_j,
                keypoint_i,
                keypoint_j,
                verified_inliers: pair.matches.len(),
                essential_inliers,
                residual,
            });
        }
    }
    // Finite, normalized residuals first (ascending), then the old pair-level
    // support tie-breakers and deterministic endpoint indices. Invalid/model
    // incomparable entries cannot displace a geometrically stronger edge.
    candidates.sort_unstable_by(|a, b| {
        let residual_order = match (a.residual, b.residual) {
            (Some(lhs), Some(rhs)) => lhs.total_cmp(&rhs),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        };
        residual_order
            .then_with(|| b.verified_inliers.cmp(&a.verified_inliers))
            .then_with(|| b.essential_inliers.cmp(&a.essential_inliers))
            .then_with(|| a.image_i.cmp(&b.image_i))
            .then_with(|| a.image_j.cmp(&b.image_j))
            .then_with(|| a.keypoint_i.cmp(&b.keypoint_i))
            .then_with(|| a.keypoint_j.cmp(&b.keypoint_j))
    });

    let mut node_id: HashMap<(usize, usize), usize> = HashMap::new();
    let mut nodes: Vec<(usize, usize)> = Vec::new();
    let mut parent = Vec::new();
    let mut component_size = Vec::new();
    let mut component_images: Vec<HashSet<usize>> = Vec::new();
    let node_of = |image: usize,
                   keypoint: usize,
                   node_id: &mut HashMap<(usize, usize), usize>,
                   nodes: &mut Vec<(usize, usize)>,
                   parent: &mut Vec<usize>,
                   component_size: &mut Vec<usize>,
                   component_images: &mut Vec<HashSet<usize>>|
     -> usize {
        if let Some(&id) = node_id.get(&(image, keypoint)) {
            return id;
        }
        let id = nodes.len();
        node_id.insert((image, keypoint), id);
        nodes.push((image, keypoint));
        parent.push(id);
        component_size.push(1);
        component_images.push(HashSet::from([image]));
        id
    };
    let mut rejected_conflicts = 0usize;
    for candidate in candidates {
        let a = node_of(
            candidate.image_i,
            candidate.keypoint_i,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let b = node_of(
            candidate.image_j,
            candidate.keypoint_j,
            &mut node_id,
            &mut nodes,
            &mut parent,
            &mut component_size,
            &mut component_images,
        );
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            continue;
        }
        if component_images[ra]
            .iter()
            .any(|image| component_images[rb].contains(image))
        {
            rejected_conflicts += 1;
            continue;
        }
        let (root, child) = if component_size[ra] > component_size[rb]
            || (component_size[ra] == component_size[rb] && ra < rb)
        {
            (ra, rb)
        } else {
            (rb, ra)
        };
        parent[child] = root;
        component_size[root] += component_size[child];
        let child_images = std::mem::take(&mut component_images[child]);
        component_images[root].extend(child_images);
    }

    let mut groups: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (id, &(image, keypoint)) in nodes.iter().enumerate() {
        let root = find(&mut parent, id);
        groups.entry(root).or_default().push((image, keypoint));
    }
    let mut stats = TrackBuildStats {
        input_correspondences: pairwise.iter().map(|pair| pair.matches.len()).sum(),
        connected_components: groups.len(),
        ..TrackBuildStats::default()
    };
    let mut tracks = Vec::new();
    for (_root, mut observations) in groups {
        observations.sort_unstable();
        let distinct_images = observations
            .iter()
            .map(|&(image, _)| image)
            .collect::<HashSet<_>>()
            .len();
        if distinct_images >= min_track_length {
            stats.retained_observations += observations.len();
            tracks.push(observations);
        }
    }
    tracks.sort_unstable();
    stats.retained_tracks = tracks.len();
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: geometric-confidence tracks rejected_conflicts={} retained_tracks={} retained_obs={}",
            rejected_conflicts, stats.retained_tracks, stats.retained_observations
        );
    }
    TrackBuildOutput {
        tracks,
        conflicting_components: Vec::new(),
        stats,
    }
}

/// M2 port: build feature tracks by routing through a
/// `visloc_vision::two_view::CorrespondenceGraph` instead of an ad hoc
/// union-find (COLMAP's own `CorrespondenceGraph`, ported in
/// `crates/vision/src/two_view/correspondence_graph.rs` — see that module's
/// doc for full citations). Every `pairwise` entry is added via
/// `CorrespondenceGraph::add_two_view_geometry`; this call site has no
/// per-pair `ConfigurationType` available (that M1 classification, when it
/// runs at all, is consumed upstream by the caller deciding which pairs make
/// it into `pairwise` in the first place — see
/// `examples/unordered_sfm_demo.rs`'s `verify_pairs` and the
/// `correspondence_graph` module doc's "Degenerate-pair policy" section), so
/// every edge is tagged with a placeholder [`ConfigurationType::Calibrated`]
/// that this function never reads back.
///
/// Tracks are then exactly COLMAP's connected components: for every
/// not-yet-visited `(image, keypoint)` observation, pull its **unbounded**
/// transitive closure (`extract_transitive_correspondences(.., ..,
/// usize::MAX)` — see that method's doc for why `usize::MAX` reproduces a
/// full connected component rather than a `num_transitivity`-bounded
/// neighbourhood) and apply the same same-image-conflict rejection and
/// `min_track_length` gate [`build_tracks_with_stats`] does. Because both algorithms
/// partition the exact same node set by the exact same edge set into
/// equivalence classes, and both sort observations within a track and tracks
/// against each other identically, this produces **byte-identical**
/// `Vec<Vec<(usize, usize)>>` output to [`build_tracks_with_stats`] on any input — the
/// M2 acceptance bar (`docs/colmap_port_plan.md`: "byte-identical tracks — a
/// refactor gate, not an accuracy claim"). See the
/// `graph_tracks_match_union_find_tracks_*` tests below.
fn build_tracks_via_graph(
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    min_track_length: usize,
) -> Vec<Vec<(usize, usize)>> {
    let mut graph = CorrespondenceGraph::new();
    for (image_id, feature_set) in features.iter().enumerate() {
        graph.add_image(image_id, feature_set.keypoints.len());
    }

    // `CorrespondenceGraph::add_two_view_geometry` — faithfully to COLMAP's
    // own `THROW_CHECK(inserted)` — accepts a given unordered image pair only
    // *once* (see that method's doc). The legacy union-find track builder
    // has no such restriction: it just unions whatever `(image, keypoint)`
    // pairs every `PairwiseMatches` entry hands it, in either direction,
    // even if the same unordered pair appears more than once (e.g. a
    // pathological/test input, or two independently-verified match sets for
    // the same pair). To keep `build_tracks_via_graph` producing identical
    // tracks on *any* such input, pre-merge every `pairwise` entry into one
    // match list per unordered pair — normalizing direction to the pair's
    // canonical `(min, max)` order — before a single `add_two_view_geometry`
    // call per pair.
    let mut merged: HashMap<(usize, usize), Vec<(usize, usize)>> = HashMap::new();
    for pair in pairwise {
        let key = (
            pair.image_i.min(pair.image_j),
            pair.image_i.max(pair.image_j),
        );
        let entry = merged.entry(key).or_default();
        if pair.image_i <= pair.image_j {
            entry.extend(pair.matches.iter().copied());
        } else {
            entry.extend(pair.matches.iter().map(|&(a, b)| (b, a)));
        }
    }
    for (&(image_id1, image_id2), matches) in &merged {
        // Ignore ingest errors: a self-pair (`image_i == image_j`) is a
        // caller bug the legacy union-find path also has no defence against
        // (it would silently union a node with itself, a no-op); dropping it
        // here preserves the same "garbage in, best-effort out" behaviour
        // rather than panicking.
        let _ = graph.add_two_view_geometry(
            image_id1,
            image_id2,
            matches,
            ConfigurationType::Calibrated,
        );
    }
    graph.finalize();

    let mut visited: HashSet<(usize, usize)> = HashSet::new();
    let mut tracks = Vec::new();
    for (image_id, feature_set) in features.iter().enumerate() {
        if !graph.exists_image(image_id) {
            continue; // dropped by finalize: never received a correspondence
        }
        for point2d_idx in 0..feature_set.keypoints.len() {
            if visited.contains(&(image_id, point2d_idx)) {
                continue;
            }
            if !graph.has_correspondences(image_id, point2d_idx) {
                visited.insert((image_id, point2d_idx));
                continue;
            }

            let closure =
                graph.extract_transitive_correspondences(image_id, point2d_idx, usize::MAX);
            let mut obs: Vec<(usize, usize)> = closure
                .iter()
                .map(|c| (c.image_id, c.point2d_idx))
                .collect();
            obs.push((image_id, point2d_idx));
            for &node in &obs {
                visited.insert(node);
            }

            // Same conflict rule as `build_tracks`: two keypoints from the
            // same image in one component means a bad match chain merged two
            // distinct points — drop the whole track.
            let mut images_seen: HashMap<usize, usize> = HashMap::new();
            let mut conflict = false;
            for &(image, _kp) in &obs {
                let count = images_seen.entry(image).or_insert(0);
                *count += 1;
                if *count > 1 {
                    conflict = true;
                    break;
                }
            }
            if conflict {
                continue;
            }
            if images_seen.len() >= min_track_length {
                obs.sort_unstable();
                tracks.push(obs);
            }
        }
    }
    tracks.sort_unstable();
    tracks
}

fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[ra] = rb;
    }
}

/// Size of the largest connected component of the view graph — images joined by
/// a verified pair. This bounds how many images any single seed can ever reach,
/// so a seed that reaches a large fraction of it is well-connected rather than an
/// isolated local cluster of a few near-identical frames.
fn largest_connected_component(pairwise: &[PairwiseMatches], n_images: usize) -> usize {
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
fn seed_candidate_order(pairwise: &[PairwiseMatches], config: &IncrementalSfmConfig) -> Vec<usize> {
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
fn place_seed_pair(
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
        return false;
    };
    if relative.inliers.len() < config.min_seed_matches {
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
    for &inl in &relative.inliers {
        let (px_i, px_j) = corr_kp[inl];
        let obs = [(pair.image_i, px_i), (pair.image_j, px_j)];
        if triangulate_track(camera, poses, &obs, config).is_some() {
            well_triangulated += 1;
        }
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
fn sequence_fallback_enabled_during_growth(config: &IncrementalSfmConfig) -> bool {
    config.sequence_relative_pose_fallback && !config.sequence_fallback_after_post
}

/// Whether the support-preserving targeted growth fast path is valid. It is
/// intentionally restricted to the plain mapper: correspondence-mode points
/// can be replaced, COLMAP-style local/global BA can complete tracks outside
/// the newly registered image, and sequence fallback has its own full-scan
/// commit helper. All of those modes therefore retain the historical full
/// pending-track scan.
fn targeted_plain_growth_enabled(config: &IncrementalSfmConfig, has_initial_poses: bool) -> bool {
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
fn grow_from_seed_with_sequence_overrides(
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
    let mut poses: Vec<Option<Pose>> = initial_poses
        .map(|poses| poses.to_vec())
        .unwrap_or_else(|| vec![None; n_images]);
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

/// Register at most one image with sequence-relative fallback after an
/// ordinary post-refinement sweep has stalled.  Each accepted pose
/// immediately retriangulates existing tracks; the after-post scheduler can
/// then resume ordinary PnP before asking for another provisional pose.
#[allow(clippy::too_many_arguments)]
fn sequence_relative_pose_registration_once_with_overrides_and_carry(
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
fn sequence_relative_pose_registration_pass_with_overrides(
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

#[derive(Debug, Clone)]
struct StructurelessConstraint {
    neighbor: usize,
    neighbor_center: Point3<f64>,
    missing_rotation: UnitQuaternion<f64>,
    center_direction: Vector3<f64>,
    weight: f64,
}

#[derive(Debug, Clone)]
struct StructurelessPoseProposal {
    pose: Pose,
    neighbor_spread: f64,
    line_error_ratio: f64,
    consensus_indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
enum StructurelessRejection {
    TooFewNeighbors {
        found: usize,
        required: usize,
    },
    RotationDisagreement {
        max_deg: f64,
        allowed_deg: f64,
    },
    WeakCenterGeometry {
        max_angle_deg: f64,
        spread: f64,
    },
    NoCenterConsensus {
        rotation_consensus: usize,
    },
    SingularCenterFit,
    DirectionSign {
        neighbor: usize,
        along_ratio: f64,
        allowed: f64,
    },
    CenterLineResidual {
        ratio: f64,
        allowed: f64,
    },
}

/// Fit the missing camera centre to directed lines originating at registered
/// neighbour centres. Each line direction comes from an independently
/// recovered essential pose, while its origin carries the current model's
/// monocular scale. This is the scale-bearing part of structure-less recovery:
/// one line is deliberately under-constrained and is always rejected.
fn solve_structureless_pose(
    constraints: &[StructurelessConstraint],
    config: &IncrementalSfmConfig,
) -> Result<StructurelessPoseProposal, StructurelessRejection> {
    let required_neighbors = config.structureless_min_neighbors.max(2);
    if constraints.len() < required_neighbors {
        return Err(StructurelessRejection::TooFewNeighbors {
            found: constraints.len(),
            required: required_neighbors,
        });
    }

    let max_rotation_rad = config
        .structureless_max_rotation_disagreement_deg
        .max(0.0)
        .to_radians();
    // A single bad essential edge must not veto an otherwise coherent set.
    // Enumerate every rotation as a deterministic consensus centre and keep
    // the largest, then highest-support, <=threshold subset.
    let mut consensus_indices = Vec::new();
    let mut consensus_weight = -1.0f64;
    let mut rotation_reference_index = None;
    for (reference_index, reference) in constraints.iter().enumerate() {
        let candidate: Vec<usize> = constraints
            .iter()
            .enumerate()
            .filter_map(|(index, constraint)| {
                ((reference.missing_rotation.inverse() * constraint.missing_rotation).angle()
                    <= max_rotation_rad)
                    .then_some(index)
            })
            .collect();
        let weight: f64 = candidate
            .iter()
            .map(|&index| constraints[index].weight)
            .sum();
        if candidate.len() > consensus_indices.len()
            || (candidate.len() == consensus_indices.len() && weight > consensus_weight)
            || (candidate.len() == consensus_indices.len()
                && weight.to_bits() == consensus_weight.to_bits()
                && candidate.first().copied().unwrap_or(reference_index)
                    < consensus_indices.first().copied().unwrap_or(usize::MAX))
        {
            consensus_indices = candidate;
            consensus_weight = weight;
            rotation_reference_index = Some(reference_index);
        }
    }
    if consensus_indices.len() < required_neighbors {
        let strongest = &constraints[0];
        let max_rotation_disagreement = constraints
            .iter()
            .map(|constraint| {
                (strongest.missing_rotation.inverse() * constraint.missing_rotation).angle()
            })
            .fold(0.0f64, f64::max);
        return Err(StructurelessRejection::RotationDisagreement {
            max_deg: max_rotation_disagreement.to_degrees(),
            allowed_deg: max_rotation_rad.to_degrees(),
        });
    }
    // Preserve the actual consensus centre. Choosing the strongest edge after
    // finding the set is not equivalent: two members can each lie within the
    // threshold of the centre yet be almost 2x the threshold apart.
    let reference_index = rotation_reference_index.expect("rotation consensus has a centre");
    let reference = &constraints[reference_index];

    let min_intersection_angle = config
        .structureless_min_intersection_angle_deg
        .max(0.0)
        .to_radians();
    let mut max_intersection_angle = 0.0f64;
    let mut rotation_consensus_spread = 0.0f64;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        let a = &constraints[a_index];
        for &b_index in consensus_indices.iter().skip(position + 1) {
            let b = &constraints[b_index];
            let cosine = a
                .center_direction
                .dot(&b.center_direction)
                .abs()
                .clamp(0.0, 1.0);
            max_intersection_angle = max_intersection_angle.max(cosine.acos());
            rotation_consensus_spread =
                rotation_consensus_spread.max((a.neighbor_center - b.neighbor_center).norm());
        }
    }
    if max_intersection_angle < min_intersection_angle || rotation_consensus_spread <= 1e-9 {
        return Err(StructurelessRejection::WeakCenterGeometry {
            max_angle_deg: max_intersection_angle.to_degrees(),
            spread: rotation_consensus_spread,
        });
    }

    let identity = Matrix3::identity();
    let fit_center = |indices: &[usize]| -> Option<Point3<f64>> {
        let mut normal = Matrix3::zeros();
        let mut rhs = Vector3::zeros();
        for &index in indices {
            let constraint = &constraints[index];
            let direction = constraint.center_direction.try_normalize(1e-12)?;
            let weight = constraint.weight.max(1.0);
            let projector = identity - direction * direction.transpose();
            normal += projector * weight;
            rhs += projector * constraint.neighbor_center.coords * weight;
        }
        Some(Point3::from(normal.try_inverse()? * rhs))
    };

    // Translation directions need their own robust consensus: agreeing
    // rotations do not imply that every essential decomposition has a reliable
    // baseline direction. Seed from every sufficiently non-parallel line pair,
    // score all rotation-consensus lines, then refit the largest 3+ set.
    let max_line_ratio = config.structureless_max_center_line_error_ratio.max(0.0);
    let mut center_consensus = Vec::new();
    let mut center_consensus_weight = -1.0f64;
    let mut center_consensus_error = f64::INFINITY;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        for &b_index in consensus_indices.iter().skip(position + 1) {
            let a = &constraints[a_index];
            let b = &constraints[b_index];
            let angle = a
                .center_direction
                .dot(&b.center_direction)
                .abs()
                .clamp(0.0, 1.0)
                .acos();
            if angle < min_intersection_angle {
                continue;
            }
            let Some(candidate_center) = fit_center(&[a_index, b_index]) else {
                continue;
            };
            let mut inliers = Vec::new();
            let mut squared_error = 0.0;
            let mut weight = 0.0;
            for &index in &consensus_indices {
                let constraint = &constraints[index];
                let displacement = candidate_center - constraint.neighbor_center;
                let along_ratio =
                    displacement.dot(&constraint.center_direction) / rotation_consensus_spread;
                let perpendicular = displacement
                    - constraint.center_direction * displacement.dot(&constraint.center_direction);
                let line_ratio = perpendicular.norm() / rotation_consensus_spread;
                if along_ratio >= config.structureless_min_forward_ratio
                    && line_ratio <= max_line_ratio
                {
                    inliers.push(index);
                    let edge_weight = constraint.weight.max(1.0);
                    squared_error += edge_weight * line_ratio * line_ratio;
                    weight += edge_weight;
                }
            }
            if inliers.len() < required_neighbors {
                continue;
            }
            let rms_error = (squared_error / weight.max(1.0)).sqrt();
            if inliers.len() > center_consensus.len()
                || (inliers.len() == center_consensus.len() && weight > center_consensus_weight)
                || (inliers.len() == center_consensus.len()
                    && weight.to_bits() == center_consensus_weight.to_bits()
                    && rms_error < center_consensus_error)
            {
                center_consensus = inliers;
                center_consensus_weight = weight;
                center_consensus_error = rms_error;
            }
        }
    }
    if center_consensus.len() < required_neighbors {
        return Err(StructurelessRejection::NoCenterConsensus {
            rotation_consensus: consensus_indices.len(),
        });
    }
    consensus_indices = center_consensus;
    // A weighted least-squares refit can move slightly outside the inlier set
    // that generated the winning two-line hypothesis. Reclassify after every
    // refit and discard only the inconsistent lines instead of allowing one
    // marginal edge to veto an otherwise valid 3+ neighbour consensus.
    // Removal is monotonic, so this converges in at most N iterations.
    let center = loop {
        let fitted =
            fit_center(&consensus_indices).ok_or(StructurelessRejection::SingularCenterFit)?;
        let retained: Vec<usize> = consensus_indices
            .iter()
            .copied()
            .filter(|&index| {
                let constraint = &constraints[index];
                let displacement = fitted - constraint.neighbor_center;
                let along_ratio =
                    displacement.dot(&constraint.center_direction) / rotation_consensus_spread;
                let perpendicular = displacement
                    - constraint.center_direction * displacement.dot(&constraint.center_direction);
                let line_ratio = perpendicular.norm() / rotation_consensus_spread;
                along_ratio >= config.structureless_min_forward_ratio
                    && line_ratio <= max_line_ratio
            })
            .collect();
        if retained.len() < required_neighbors {
            return Err(StructurelessRejection::NoCenterConsensus {
                rotation_consensus: consensus_indices.len(),
            });
        }
        if retained.len() == consensus_indices.len() {
            break fitted;
        }
        consensus_indices = retained;
    };
    let mut selected_neighbor_spread = 0.0f64;
    for (position, &a_index) in consensus_indices.iter().enumerate() {
        for &b_index in consensus_indices.iter().skip(position + 1) {
            selected_neighbor_spread = selected_neighbor_spread.max(
                (constraints[a_index].neighbor_center - constraints[b_index].neighbor_center)
                    .norm(),
            );
        }
    }
    if selected_neighbor_spread <= 1e-9 {
        return Err(StructurelessRejection::SingularCenterFit);
    }
    // Use the same rotation-consensus span used while scoring RANSAC centre
    // hypotheses. Switching to the smaller selected-subset span after refit
    // would make an inlier fail a stricter, inconsistent normalized gate.
    let neighbor_spread = rotation_consensus_spread;

    let mut weighted_squared_error = 0.0;
    let mut weight_sum = 0.0;
    for &index in &consensus_indices {
        let constraint = &constraints[index];
        let displacement = center - constraint.neighbor_center;
        // Essential decomposition resolves the sign through cheirality. A
        // negative line parameter means the multi-neighbour fit contradicts
        // that independent two-view geometry.
        let along = displacement.dot(&constraint.center_direction);
        let along_ratio = along / neighbor_spread;
        if along_ratio < config.structureless_min_forward_ratio {
            return Err(StructurelessRejection::DirectionSign {
                neighbor: constraint.neighbor,
                along_ratio,
                allowed: config.structureless_min_forward_ratio,
            });
        }
        let perpendicular = displacement
            - constraint.center_direction * displacement.dot(&constraint.center_direction);
        let weight = constraint.weight.max(1.0);
        weighted_squared_error += weight * perpendicular.norm_squared();
        weight_sum += weight;
    }
    let rms_line_error = (weighted_squared_error / weight_sum.max(1.0)).sqrt();
    let line_error_ratio = rms_line_error / neighbor_spread;
    if !line_error_ratio.is_finite()
        || line_error_ratio > config.structureless_max_center_line_error_ratio.max(0.0)
    {
        return Err(StructurelessRejection::CenterLineResidual {
            ratio: line_error_ratio,
            allowed: config.structureless_max_center_line_error_ratio.max(0.0),
        });
    }

    let rotation = reference.missing_rotation;
    let translation = -rotation.transform_vector(&center.coords);
    Ok(StructurelessPoseProposal {
        pose: Pose::from_world_to_camera(rotation, translation),
        neighbor_spread,
        line_error_ratio,
        consensus_indices,
    })
}

/// Return true when `new_pose` for `image` agrees with independent two-view
/// essentials against already-registered neighbours (same translation
/// hemisphere). With fewer than `min_neighbors` usable checks, accept.
#[allow(clippy::too_many_arguments)]
fn pose_agrees_with_two_view_neighbors(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    image: usize,
    new_pose: &Pose,
    min_neighbors: usize,
    min_agree_fraction: f64,
) -> bool {
    let estimator = RelativePoseEstimator::default();
    let new_c = new_pose.camera_center_world();
    let mut checked = 0usize;
    let mut agree = 0usize;
    for pair in pairwise {
        let (neighbor, matches_ij, new_is_i) = if pair.image_i == image {
            (pair.image_j, &pair.matches, true)
        } else if pair.image_j == image {
            (pair.image_i, &pair.matches, false)
        } else {
            continue;
        };
        let Some(neighbor_pose) = poses.get(neighbor).and_then(|p| p.as_ref()) else {
            continue;
        };
        if matches_ij.len() < 16 {
            continue;
        }
        let correspondences: Vec<TwoViewCorrespondence> = matches_ij
            .iter()
            .filter_map(|&(ki, kj)| {
                Some(TwoViewCorrespondence::new(
                    *features[pair.image_i].keypoints.get(ki)?,
                    *features[pair.image_j].keypoints.get(kj)?,
                ))
            })
            .collect();
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            continue;
        };
        if relative.inliers.len() < 16 {
            continue;
        }
        // Two-view: camera-j centre direction in camera-i frame.
        let r_ij = relative.previous_to_current.rotation;
        let t_ij = relative.previous_to_current.translation;
        let Some(dir_i_to_j) = (-r_ij.inverse().transform_vector(&t_ij)).try_normalize(1e-12)
        else {
            continue;
        };
        let neighbor_c = neighbor_pose.camera_center_world();
        let abs_agree = if new_is_i {
            // Absolute: neighbour in new (image_i) frame.
            let Some(abs_dir) = new_pose
                .world_to_camera
                .rotation
                .transform_vector(&(neighbor_c - new_c))
                .try_normalize(1e-12)
            else {
                continue;
            };
            // Two-view dir is i→j = new→neighbor.
            abs_dir.dot(&dir_i_to_j) > 0.0
        } else {
            // Absolute: new in neighbour (image_i) frame.
            let Some(abs_dir) = neighbor_pose
                .world_to_camera
                .rotation
                .transform_vector(&(new_c - neighbor_c))
                .try_normalize(1e-12)
            else {
                continue;
            };
            abs_dir.dot(&dir_i_to_j) > 0.0
        };
        checked += 1;
        if abs_agree {
            agree += 1;
        }
    }
    if checked < min_neighbors {
        return true;
    }
    (agree as f64 / checked as f64) >= min_agree_fraction
}

fn estimate_structureless_constraints(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    missing: usize,
    config: &IncrementalSfmConfig,
) -> Vec<StructurelessConstraint> {
    let estimator = RelativePoseEstimator::default();
    let mut constraints = Vec::new();
    for pair in pairwise {
        let (neighbor, invert) = if pair.image_j == missing && poses[pair.image_i].is_some() {
            (pair.image_i, false)
        } else if pair.image_i == missing && poses[pair.image_j].is_some() {
            (pair.image_j, true)
        } else {
            continue;
        };
        let Some(neighbor_pose) = poses[neighbor].as_ref() else {
            continue;
        };
        let mut correspondences = Vec::with_capacity(pair.matches.len());
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (Some(pixel_i), Some(pixel_j)) = (
                features[pair.image_i].keypoints.get(keypoint_i),
                features[pair.image_j].keypoints.get(keypoint_j),
            ) else {
                continue;
            };
            correspondences.push(TwoViewCorrespondence::new(*pixel_i, *pixel_j));
        }
        let Some(relative) = estimator.estimate(&correspondences, camera) else {
            continue;
        };
        if relative.inliers.len() < config.structureless_min_pair_inliers {
            continue;
        }
        let neighbor_to_missing = if invert {
            relative.previous_to_current.inverse()
        } else {
            relative.previous_to_current
        };
        let missing_rotation =
            neighbor_to_missing.rotation * neighbor_pose.world_to_camera.rotation;
        let Some(center_direction) = (-missing_rotation
            .inverse()
            .transform_vector(&neighbor_to_missing.translation))
        .try_normalize(1e-12) else {
            continue;
        };
        constraints.push(StructurelessConstraint {
            neighbor,
            neighbor_center: neighbor_pose.camera_center_world(),
            missing_rotation,
            center_direction,
            weight: relative.inliers.len() as f64,
        });
    }
    constraints.sort_by(|a, b| {
        b.weight
            .total_cmp(&a.weight)
            .then_with(|| a.neighbor.cmp(&b.neighbor))
    });
    constraints
}

fn mean_reprojection_for_registered_mask(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    registered_mask: &[bool],
    point_mask: &[bool],
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        if !point_mask.get(track_id).copied().unwrap_or(false) {
            continue;
        }
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        for &(image, keypoint) in track {
            if !registered_mask.get(image).copied().unwrap_or(false) {
                continue;
            }
            let (Some(pose), Some(pixel)) = (
                poses.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint)),
            ) else {
                continue;
            };
            if let Some(error) = reprojection_error_px(camera, pose, point, pixel) {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn supported_tracks_for_image(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    image: usize,
    max_error: f64,
) -> (usize, f64) {
    let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
        return (0, f64::NAN);
    };
    let mut count = 0usize;
    let mut sum = 0.0;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        let Some((_, keypoint)) = track.iter().find(|(track_image, _)| *track_image == image)
        else {
            continue;
        };
        let Some(pixel) = features[image].keypoints.get(*keypoint) else {
            continue;
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            continue;
        };
        if error <= max_error {
            count += 1;
            sum += error;
        }
    }
    if count == 0 {
        (0, f64::NAN)
    } else {
        (count, sum / count as f64)
    }
}

#[derive(Debug, Clone, Copy)]
struct StructurelessPoseConsistency {
    accepted: bool,
    max_rotation_deg: f64,
    min_forward_ratio: f64,
    line_error_ratio: f64,
}

#[cfg(test)]
fn interpolate_structureless_pose(from: &Pose, to: &Pose, alpha: f64) -> Pose {
    interpolate_structureless_pose_components(from, to, alpha, alpha)
}

fn interpolate_structureless_pose_components(
    from: &Pose,
    to: &Pose,
    rotation_alpha: f64,
    center_alpha: f64,
) -> Pose {
    let rotation_alpha = rotation_alpha.clamp(0.0, 1.0);
    let center_alpha = center_alpha.clamp(0.0, 1.0);
    if rotation_alpha <= 0.0 && center_alpha <= 0.0 {
        return from.clone();
    }
    if rotation_alpha >= 1.0 && center_alpha >= 1.0 {
        return to.clone();
    }
    let rotation = from
        .world_to_camera
        .rotation
        .slerp(&to.world_to_camera.rotation, rotation_alpha);
    let from_center = from.camera_center_world();
    let to_center = to.camera_center_world();
    let center =
        Point3::from(from_center.coords * (1.0 - center_alpha) + to_center.coords * center_alpha);
    let translation = -rotation.transform_vector(&center.coords);
    Pose::from_world_to_camera(rotation, translation)
}

fn structureless_pose_consistency(
    pose: &Pose,
    constraints: &[StructurelessConstraint],
    proposal: &StructurelessPoseProposal,
    config: &IncrementalSfmConfig,
) -> StructurelessPoseConsistency {
    let center = pose.camera_center_world();
    let max_rotation = config
        .structureless_max_rotation_disagreement_deg
        .max(0.0)
        .to_radians();
    let mut weighted_squared_error = 0.0;
    let mut weight_sum = 0.0;
    let mut max_rotation_seen = 0.0f64;
    let mut min_forward_seen = f64::INFINITY;
    for &index in &proposal.consensus_indices {
        let constraint = &constraints[index];
        let rotation_error =
            (constraint.missing_rotation.inverse() * pose.world_to_camera.rotation).angle();
        max_rotation_seen = max_rotation_seen.max(rotation_error);
        let displacement = center - constraint.neighbor_center;
        let forward_ratio =
            displacement.dot(&constraint.center_direction) / proposal.neighbor_spread;
        min_forward_seen = min_forward_seen.min(forward_ratio);
        let perpendicular = displacement
            - constraint.center_direction * displacement.dot(&constraint.center_direction);
        let weight = constraint.weight.max(1.0);
        weighted_squared_error += weight * perpendicular.norm_squared();
        weight_sum += weight;
    }
    let ratio =
        (weighted_squared_error / weight_sum.max(1.0)).sqrt() / proposal.neighbor_spread.max(1e-12);
    StructurelessPoseConsistency {
        accepted: max_rotation_seen <= max_rotation
            && min_forward_seen >= config.structureless_min_forward_ratio
            && ratio.is_finite()
            && ratio <= config.structureless_max_center_line_error_ratio.max(0.0),
        max_rotation_deg: max_rotation_seen.to_degrees(),
        min_forward_ratio: min_forward_seen,
        line_error_ratio: ratio,
    }
}

/// Build an independent local submap from verified pairwise edges that were
/// not retained by the global union-find tracks. Observations already owned by
/// a global 3D track are never duplicated. Each new point must be seen by the
/// missing image and the configured number of registered consensus neighbours,
/// triangulate with sufficient parallax, and reproject within the initialization
/// gate in every contributing view.
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn build_structureless_local_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    poses: &[Option<Pose>],
    missing: usize,
    constraints: &[StructurelessConstraint],
    proposal: &StructurelessPoseProposal,
    config: &IncrementalSfmConfig,
) -> Vec<(Vec<(usize, usize)>, Point3<f64>)> {
    let allowed_neighbors: HashSet<usize> = proposal
        .consensus_indices
        .iter()
        .map(|&index| constraints[index].neighbor)
        .collect();
    let occupied: HashSet<(usize, usize)> = tracks
        .iter()
        .enumerate()
        .filter(|(track_id, _)| track_point.get(*track_id).is_some_and(Option::is_some))
        .flat_map(|(_, track)| track.iter().copied())
        .collect();
    let mut by_missing_keypoint: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for pair in pairwise {
        let (neighbor, missing_first) = if pair.image_i == missing
            && allowed_neighbors.contains(&pair.image_j)
            && poses[pair.image_j].is_some()
        {
            (pair.image_j, true)
        } else if pair.image_j == missing
            && allowed_neighbors.contains(&pair.image_i)
            && poses[pair.image_i].is_some()
        {
            (pair.image_i, false)
        } else {
            continue;
        };
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let (missing_keypoint, neighbor_keypoint) = if missing_first {
                (keypoint_i, keypoint_j)
            } else {
                (keypoint_j, keypoint_i)
            };
            if features[missing].keypoints.get(missing_keypoint).is_none()
                || features[neighbor]
                    .keypoints
                    .get(neighbor_keypoint)
                    .is_none()
                || occupied.contains(&(missing, missing_keypoint))
                || occupied.contains(&(neighbor, neighbor_keypoint))
            {
                continue;
            }
            by_missing_keypoint
                .entry(missing_keypoint)
                .or_default()
                .push((neighbor, neighbor_keypoint));
        }
    }

    let mut missing_keypoints: Vec<usize> = by_missing_keypoint.keys().copied().collect();
    missing_keypoints.sort_unstable();
    let mut local_tracks = Vec::new();
    let mut claimed_observations = HashSet::new();
    for missing_keypoint in missing_keypoints {
        let mut neighbors = by_missing_keypoint.remove(&missing_keypoint).unwrap();
        neighbors.sort_unstable();
        neighbors.dedup_by_key(|observation| observation.0);
        let required_registered_views = config
            .structureless_min_local_track_views
            .max(2)
            .saturating_sub(1);
        if neighbors.len() < required_registered_views {
            continue;
        }
        let mut observations = vec![(missing, missing_keypoint)];
        observations.extend(neighbors);
        observations.sort_unstable();
        if observations
            .iter()
            .any(|observation| claimed_observations.contains(observation))
        {
            continue;
        }
        let pixels: Vec<(usize, Point2<f64>)> = observations
            .iter()
            .map(|&(image, keypoint)| (image, features[image].keypoints[keypoint]))
            .collect();
        let Some(point) = triangulate_track(camera, poses, &pixels, config) else {
            continue;
        };
        let mut valid = true;
        for &(image, keypoint) in &observations {
            let Some(error) = reprojection_error_px(
                camera,
                poses[image].as_ref().unwrap(),
                &point,
                &features[image].keypoints[keypoint],
            ) else {
                valid = false;
                break;
            };
            // This is an initialization gate only. The point is subsequently
            // refined in the fixed-pose local submap and must still clear the
            // stricter structure-less admission error below.
            if error > config.max_reprojection_error_px {
                valid = false;
                break;
            }
        }
        if valid {
            claimed_observations.extend(observations.iter().copied());
            local_tracks.push((observations, point));
            if local_tracks.len() >= config.structureless_max_local_tracks.max(1) {
                break;
            }
        }
    }
    local_tracks
}

/// Run [`structureless_registration_pass`] repeatedly until a round registers
/// nothing, the budget [`IncrementalSfmConfig::structureless_max_rounds`] is
/// spent, or every image is registered. Each round scans in the same fixed
/// ascending image order, so the loop is deterministic; images registered by
/// earlier rounds act as neighbours for later ones, which is what lets an
/// island chain inward through its bridge even when the bridge's index is
/// higher than the images it unlocks.
fn structureless_registration_rounds(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut Vec<Option<Point3<f64>>>,
) -> usize {
    let mut total = 0usize;
    let max_rounds = config.structureless_max_rounds.max(1);
    for round in 0..max_rounds {
        if !poses.iter().any(Option::is_none) {
            break;
        }
        let registered = structureless_registration_pass(
            camera,
            features,
            pairwise,
            tracks,
            config,
            poses,
            track_point,
        );
        if sfm_debug_enabled() && registered > 0 {
            eprintln!("sfm-debug: structure-less round {round} registered {registered} image(s)");
        }
        total += registered;
        if registered == 0 {
            break;
        }
    }
    total
}

/// One bounded multi-neighbour recovery sweep. Each missing image is attempted
/// at most once. Failed geometry, local BA, or admission gates restore the
/// complete pose/point state byte-for-byte before moving on.
fn structureless_registration_pass(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut Vec<Option<Point3<f64>>>,
) -> usize {
    let missing_images: Vec<usize> = poses
        .iter()
        .enumerate()
        .filter_map(|(image, pose)| pose.is_none().then_some(image))
        .collect();
    let mut registered = 0usize;
    for image in missing_images {
        let constraints =
            estimate_structureless_constraints(camera, features, pairwise, poses, image, config);
        let proposal = match solve_structureless_pose(&constraints, config) {
            Ok(proposal) => proposal,
            Err(reason) => {
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: structure-less image {image} rejected before insertion \
                         ({} registered relative neighbours): {reason:?}",
                        constraints.len(),
                    );
                }
                continue;
            }
        };
        let tracks_before = tracks.clone();
        let tracks_before_len = tracks_before.len();
        let poses_before = poses.to_vec();
        let points_before = track_point.to_vec();
        let registered_mask: Vec<bool> = poses_before.iter().map(Option::is_some).collect();
        let clean_point_mask: Vec<bool> = points_before.iter().map(Option::is_some).collect();
        let clean_mean_before = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            &poses_before,
            &points_before,
            &registered_mask,
            &clean_point_mask,
        );
        poses[image] = Some(proposal.pose.clone());
        let local_tracks = build_structureless_local_tracks(
            camera,
            features,
            pairwise,
            tracks,
            track_point,
            poses,
            image,
            &constraints,
            &proposal,
            config,
        );
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: structure-less image {image} synthesized {} independent local tracks",
                local_tracks.len()
            );
        }
        let local_observations: HashSet<(usize, usize)> = local_tracks
            .iter()
            .flat_map(|(track, _)| track.iter().copied())
            .collect();
        for (track_id, track) in tracks.iter_mut().enumerate().take(tracks_before_len) {
            if track_point[track_id].is_none() {
                track.retain(|observation| !local_observations.contains(observation));
            }
        }
        for (track, point) in local_tracks {
            tracks.push(track);
            track_point.push(Some(point));
        }
        triangulate_pending_with_config(camera, features, tracks, poses, config, track_point);
        let proposal_poses = poses.to_vec();
        let proposal_points = track_point.to_vec();
        let (proposal_support, proposal_image_mean) = supported_tracks_for_image(
            camera,
            features,
            tracks,
            &proposal_poses,
            &proposal_points,
            image,
            config.max_reprojection_error_px,
        );
        let proposal_clean_mean = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            &proposal_poses,
            &proposal_points,
            &registered_mask,
            &clean_point_mask,
        );
        let proposal_consistency = proposal_poses[image]
            .as_ref()
            .map(|pose| structureless_pose_consistency(pose, &constraints, &proposal, config));
        // A structure-less proposal is already tied to the registered map by
        // several independently estimated relative poses.  Moving its
        // neighbours in the ordinary growth local-BA window can trade that
        // scale-bearing consensus for a lower pixel residual (MH_05 image 86
        // exposed exactly that failure).  Refine only the recovered pose and
        // its incident landmarks; every previously registered observer stays
        // fixed and therefore acts as the local-submap alignment boundary.
        let mut structureless_variable = HashSet::new();
        structureless_variable.insert(image);
        let local_result = bundle_adjust_local(
            camera,
            features,
            tracks,
            config,
            poses,
            track_point,
            &structureless_variable,
        );
        let refined_poses = poses.to_vec();
        let allowed_clean_mean = clean_mean_before
            * (1.0 + config.structureless_max_clean_error_increase_ratio.max(0.0));
        let (mut support, mut image_mean) = supported_tracks_for_image(
            camera,
            features,
            tracks,
            poses,
            track_point,
            image,
            config.max_reprojection_error_px,
        );
        let mut clean_mean_after = mean_reprojection_for_registered_mask(
            camera,
            features,
            tracks,
            poses,
            track_point,
            &registered_mask,
            &clean_point_mask,
        );
        let mut pose_consistency = poses[image]
            .as_ref()
            .map(|pose| structureless_pose_consistency(pose, &constraints, &proposal, config));
        let local_ok = local_result.is_ok();
        let mut support_ok = support >= config.structureless_min_support_tracks;
        let mut image_error_ok =
            image_mean.is_finite() && image_mean <= config.structureless_max_reprojection_error_px;
        let mut clean_ok = clean_mean_before.is_finite()
            && clean_mean_after.is_finite()
            && clean_mean_after <= allowed_clean_mean + 1e-12;
        let mut geometry_ok = pose_consistency.is_some_and(|diagnostic| diagnostic.accepted);
        let mut accepted = local_ok && support_ok && image_error_ok && clean_ok && geometry_ok;
        let mut trust_region_alpha = None;

        // The unconstrained local BA can cross the independently measured
        // relative-geometry boundary while greatly improving reprojection.
        // Search back along the camera part of that BA update. For each pose
        // inside the relative-geometry feasible region, re-solve only the new
        // landmarks against fixed cameras, then commit the largest step that
        // satisfies every admission gate. This is a bounded deterministic
        // local-submap projection, not a relaxed threshold.
        if local_ok && !accepted {
            let proposal_pose = proposal_poses[image].as_ref().unwrap();
            let refined_pose = refined_poses[image].as_ref().unwrap();
            let mut trust_candidates = Vec::with_capacity(400);
            for rotation_step in (0..20).rev() {
                for center_step in (0..20).rev() {
                    trust_candidates.push((rotation_step as f64 / 20.0, center_step as f64 / 20.0));
                }
            }
            let mut candidate_index = 0usize;
            let mut best_near_candidate: Option<(f64, f64, f64)> = None;
            let mut fine_candidates_enqueued = false;
            'trust_region: while candidate_index < trust_candidates.len() {
                let (rotation_alpha, center_alpha) = trust_candidates[candidate_index];
                candidate_index += 1;
                let mut candidate_poses = proposal_poses.clone();
                candidate_poses[image] = Some(interpolate_structureless_pose_components(
                    proposal_pose,
                    refined_pose,
                    rotation_alpha,
                    center_alpha,
                ));
                let candidate_consistency = structureless_pose_consistency(
                    candidate_poses[image].as_ref().unwrap(),
                    &constraints,
                    &proposal,
                    config,
                );
                // Local tracks triangulated at the unconstrained proposal may
                // be invalid at the projected pose (and vice versa). Rebuild
                // the bounded submap at each geometry-feasible trust-region
                // pose from the pre-insertion state. This keeps landmark
                // synthesis consistent with the camera pose being admitted.
                let mut candidate_tracks = tracks_before.clone();
                let mut candidate_points = points_before.clone();
                let candidate_local_tracks = if candidate_consistency.accepted {
                    build_structureless_local_tracks(
                        camera,
                        features,
                        pairwise,
                        &candidate_tracks,
                        &candidate_points,
                        &candidate_poses,
                        image,
                        &constraints,
                        &proposal,
                        config,
                    )
                } else {
                    Vec::new()
                };
                let candidate_local_observations: HashSet<(usize, usize)> = candidate_local_tracks
                    .iter()
                    .flat_map(|(track, _)| track.iter().copied())
                    .collect();
                for (track_id, track) in candidate_tracks.iter_mut().enumerate() {
                    if candidate_points[track_id].is_none() {
                        track.retain(|observation| {
                            !candidate_local_observations.contains(observation)
                        });
                    }
                }
                for (track, point) in candidate_local_tracks {
                    candidate_tracks.push(track);
                    candidate_points.push(Some(point));
                }
                if candidate_consistency.accepted {
                    triangulate_pending_with_config(
                        camera,
                        features,
                        &candidate_tracks,
                        &candidate_poses,
                        config,
                        &mut candidate_points,
                    );
                }
                let submap_ok = candidate_consistency.accepted
                    && refine_structureless_new_landmarks(
                        camera,
                        features,
                        &candidate_tracks,
                        config,
                        &candidate_poses,
                        &mut candidate_points,
                        image,
                        &clean_point_mask,
                    )
                    .is_ok();
                let (candidate_support, candidate_image_mean) = supported_tracks_for_image(
                    camera,
                    features,
                    &candidate_tracks,
                    &candidate_poses,
                    &candidate_points,
                    image,
                    config.max_reprojection_error_px,
                );
                let candidate_clean_mean = mean_reprojection_for_registered_mask(
                    camera,
                    features,
                    &candidate_tracks,
                    &candidate_poses,
                    &candidate_points,
                    &registered_mask,
                    &clean_point_mask,
                );
                let candidate_ok = submap_ok
                    && candidate_support >= config.structureless_min_support_tracks
                    && candidate_image_mean.is_finite()
                    && candidate_image_mean <= config.structureless_max_reprojection_error_px
                    && clean_mean_before.is_finite()
                    && candidate_clean_mean.is_finite()
                    && candidate_clean_mean <= allowed_clean_mean + 1e-12
                    && candidate_consistency.accepted;
                let near_candidate = submap_ok
                    && candidate_support >= config.structureless_min_support_tracks
                    && candidate_image_mean.is_finite()
                    && clean_mean_before.is_finite()
                    && candidate_clean_mean.is_finite()
                    && candidate_clean_mean <= allowed_clean_mean + 1e-12
                    && candidate_consistency.accepted;
                if near_candidate
                    && best_near_candidate
                        .is_none_or(|(_, _, best_mean)| candidate_image_mean < best_mean)
                {
                    best_near_candidate =
                        Some((rotation_alpha, center_alpha, candidate_image_mean));
                }
                if sfm_debug_enabled() {
                    eprintln!(
                        "sfm-debug: structure-less image {image} trust rotation-alpha={rotation_alpha:.2} \
                         center-alpha={center_alpha:.2} \
                         tracks={} support={candidate_support} mean={candidate_image_mean:.3}px \
                         clean={candidate_clean_mean:.6} rot={:.3}deg forward={:.4} \
                         line={:.4} submap-ok={submap_ok} accepted={candidate_ok}",
                        candidate_tracks.len().saturating_sub(tracks_before_len),
                        candidate_consistency.max_rotation_deg,
                        candidate_consistency.min_forward_ratio,
                        candidate_consistency.line_error_ratio,
                    );
                }
                if candidate_ok {
                    poses.clone_from_slice(&candidate_poses);
                    *tracks = candidate_tracks;
                    *track_point = candidate_points;
                    support = candidate_support;
                    image_mean = candidate_image_mean;
                    clean_mean_after = candidate_clean_mean;
                    pose_consistency = Some(candidate_consistency);
                    support_ok = true;
                    image_error_ok = true;
                    clean_ok = true;
                    geometry_ok = true;
                    accepted = true;
                    trust_region_alpha = Some((rotation_alpha, center_alpha));
                    break 'trust_region;
                }
                if candidate_index == trust_candidates.len() && !fine_candidates_enqueued {
                    fine_candidates_enqueued = true;
                    if let Some((best_rotation, best_center, _)) = best_near_candidate {
                        let rotation_percent = (best_rotation * 100.0).round() as i32;
                        let center_percent = (best_center * 100.0).round() as i32;
                        for fine_rotation in
                            ((rotation_percent - 5).max(0)..=(rotation_percent + 5).min(100)).rev()
                        {
                            for fine_center in
                                ((center_percent - 5).max(0)..=(center_percent + 5).min(100)).rev()
                            {
                                if fine_rotation % 5 != 0 || fine_center % 5 != 0 {
                                    trust_candidates.push((
                                        fine_rotation as f64 / 100.0,
                                        fine_center as f64 / 100.0,
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }
        let proposal_accepted = proposal_support >= config.structureless_min_support_tracks
            && proposal_image_mean.is_finite()
            && proposal_image_mean <= config.structureless_max_reprojection_error_px
            && clean_mean_before.is_finite()
            && proposal_clean_mean.is_finite()
            && proposal_clean_mean <= allowed_clean_mean + 1e-12
            && proposal_consistency.is_some_and(|diagnostic| diagnostic.accepted);
        if accepted {
            registered += 1;
            if sfm_debug_enabled() {
                if let Some((rotation_alpha, center_alpha)) = trust_region_alpha {
                    eprintln!(
                        "sfm-debug: structure-less image {image} projected BA step \
                         to trust-region rotation-alpha={rotation_alpha:.2} \
                         center-alpha={center_alpha:.2}"
                    );
                }
                eprintln!(
                    "sfm-debug: structure-less registered image {image} \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6})",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    support,
                    image_mean,
                    clean_mean_before,
                    clean_mean_after,
                );
            }
        } else if proposal_accepted {
            // Local BA is optional for admission: if it leaves the independent
            // relative-pose consensus, retain the already-gated scale-bearing
            // proposal and its newly triangulated structure, not the BA drift.
            poses.clone_from_slice(&proposal_poses);
            track_point.clone_from_slice(&proposal_points);
            registered += 1;
            if sfm_debug_enabled() {
                eprintln!(
                    "sfm-debug: structure-less registered image {image} pose-only \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6}; local BA rejected)",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    proposal_support,
                    proposal_image_mean,
                    clean_mean_before,
                    proposal_clean_mean,
                );
            }
        } else {
            *tracks = tracks_before;
            track_point.truncate(points_before.len());
            poses.clone_from_slice(&poses_before);
            track_point.clone_from_slice(&points_before);
            if sfm_debug_enabled() {
                let (pose_rotation_deg, pose_forward_ratio, pose_line_ratio) = pose_consistency
                    .map(|diagnostic| {
                        (
                            diagnostic.max_rotation_deg,
                            diagnostic.min_forward_ratio,
                            diagnostic.line_error_ratio,
                        )
                    })
                    .unwrap_or((f64::NAN, f64::NAN, f64::NAN));
                eprintln!(
                    "sfm-debug: structure-less image {image} rolled back \
                     (neighbors={} line-ratio={:.4} support={} mean={:.3}px \
                     clean={:.6}->{:.6} allowed={:.6} local-ok={} support-ok={} \
                     image-ok={} clean-ok={} geometry-ok={} pose-rot={:.3}deg \
                     pose-forward={:.4} pose-line={:.4}; proposal-support={} \
                     proposal-mean={:.3}px proposal-clean={:.6} proposal-ok={})",
                    proposal.consensus_indices.len(),
                    proposal.line_error_ratio,
                    support,
                    image_mean,
                    clean_mean_before,
                    clean_mean_after,
                    allowed_clean_mean,
                    local_ok,
                    support_ok,
                    image_error_ok,
                    clean_ok,
                    geometry_ok,
                    pose_rotation_deg,
                    pose_forward_ratio,
                    pose_line_ratio,
                    proposal_support,
                    proposal_image_mean,
                    proposal_clean_mean,
                    proposal_accepted,
                );
            }
        }
    }
    registered
}

/// Triangulate every track that has ≥2 registered observations and is not yet
/// triangulated, accepting only well-conditioned (parallax + reprojection) points.
pub(crate) fn triangulate_pending(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    triangulate_pending_track_ids(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        0..tracks.len(),
    );
}

/// Triangulate only the supplied track ids. In the ordinary, non-COLMAP-style
/// mapper a track that already has a point is never replaced during growth,
/// and an untriangulated track can only gain a newly registered observation
/// from the image just accepted. The growth loop therefore uses this helper
/// for the common path and falls back to a full scan immediately after a BA
/// (where every camera pose may have moved). COLMAP-style growth deliberately
/// retains its historical full scan because local/global BA and completion
/// can change support outside the newly registered image.
fn triangulate_pending_track_ids<I>(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    track_ids: I,
) -> Vec<usize>
where
    I: IntoIterator<Item = usize>,
{
    let mut newly_triangulated = Vec::new();
    for track_id in track_ids {
        let Some(track) = tracks.get(track_id) else {
            continue;
        };
        if track_point[track_id].is_some() {
            continue;
        }
        // Registered observations of this track: (image, pixel, world ray).
        let mut obs: Vec<(usize, Point2<f64>)> = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push((image, px));
            }
        }
        if obs.len() < 2 {
            continue;
        }
        if let Some(point) = triangulate_track(camera, poses, &obs, config) {
            track_point[track_id] = Some(point);
            newly_triangulated.push(track_id);
        }
    }
    newly_triangulated
}

/// Targeted image update with the IDs that became 3D points during the pass.
/// The IDs let the correspondence-count selector update its per-image score
/// cache in proportion to newly-created points rather than rescanning every
/// unregistered image after each successful PnP.
#[allow(clippy::too_many_arguments)]
fn triangulate_pending_for_image_with_new_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    obs_by_image: &[Vec<(usize, usize)>],
    image: usize,
) -> (usize, Vec<usize>) {
    let Some(observations) = obs_by_image.get(image) else {
        return (0, Vec::new());
    };
    let count = observations.len();
    let newly_triangulated = triangulate_pending_track_ids(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        observations.iter().map(|&(_, track_id)| track_id),
    );
    (count, newly_triangulated)
}

/// Seed update with the IDs that became 3D points during the pass.
#[allow(clippy::too_many_arguments)]
fn triangulate_pending_for_images_with_new_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    obs_by_image: &[Vec<(usize, usize)>],
    images: &[usize],
) -> (usize, Vec<usize>) {
    let mut track_ids = HashSet::new();
    for &image in images {
        if let Some(observations) = obs_by_image.get(image) {
            track_ids.extend(observations.iter().map(|&(_, track_id)| track_id));
        }
    }
    let count = track_ids.len();
    let newly_triangulated = triangulate_pending_track_ids(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        track_ids,
    );
    (count, newly_triangulated)
}

/// Build the exact count-policy ranking key once from the current point map.
/// The ordinary count selector used to recompute this join for every
/// registration attempt.  A track becomes a point at most once during plain
/// growth, so the cache can then be maintained from the small set returned by
/// each targeted triangulation pass.
fn build_correspondence_count_cache(
    features: &[FeatureSet],
    obs_by_image: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
) -> Vec<usize> {
    obs_by_image
        .iter()
        .enumerate()
        .map(|(image, observations)| {
            observations
                .iter()
                .filter(|&&(kp, track_id)| {
                    track_point.get(track_id).is_some_and(Option::is_some)
                        && features
                            .get(image)
                            .is_some_and(|set| set.keypoints.get(kp).is_some())
                })
                .count()
        })
        .collect()
}

/// Add newly triangulated observations to the count-policy ranking cache.
/// `newly_triangulated` is unique per pass, so every observation contributes
/// exactly once, matching a fresh count scan.
fn update_correspondence_count_cache(
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
    newly_triangulated: &[usize],
    counts: &mut [usize],
) {
    for &track_id in newly_triangulated {
        if track_point.get(track_id).is_none_or(Option::is_none) {
            continue;
        }
        let Some(track) = tracks.get(track_id) else {
            continue;
        };
        for &(image, kp) in track {
            if features
                .get(image)
                .is_some_and(|set| set.keypoints.get(kp).is_some())
            {
                if let Some(count) = counts.get_mut(image) {
                    *count = count.saturating_add(1);
                }
            }
        }
    }
}

/// Incremental correspondence-mode point update.
///
/// Unlike [`triangulate_pending`], this path owns an explicit
/// observation-to-point map and revisits already-created points after every
/// registration.  Newly registered observations therefore participate in the
/// widest-baseline triangulation immediately, while an existing point is
/// replaced only when the candidate lowers its mean registered-view
/// reprojection.  During ordinary growth the state is retained across calls
/// (and refreshed from `track_point` after a BA); helper callers that do not
/// own a growth state get the same deterministic map rebuilt from the immutable
/// tracks.  Malformed duplicate observations are therefore visible to the
/// same one-image-per-point invariant used by the builder.
fn triangulate_correspondence_pending_with_state(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    state: &mut CorrespondencePointState,
) {
    debug_assert_eq!(state.observations.len(), tracks.len());
    state.points.resize(tracks.len(), None);
    for (state_point, track_point) in state.points.iter_mut().zip(track_point.iter()) {
        *state_point = *track_point;
    }
    debug_assert!(state
        .observation_to_point
        .iter()
        .all(|(&(image, kp), &point)| {
            tracks
                .get(point)
                .is_some_and(|track| track.contains(&(image, kp)))
        }));
    for (track_id, track) in tracks.iter().enumerate() {
        let mut obs: Vec<(usize, Point2<f64>)> = Vec::new();
        for &(image, kp) in track {
            if poses.get(image).and_then(Option::as_ref).is_none() {
                continue;
            }
            if let Some(px) = features
                .get(image)
                .and_then(|set| set.keypoints.get(kp))
                .copied()
            {
                obs.push((image, px));
            }
        }
        if obs.len() < 2 {
            continue;
        }
        let Some(candidate) = triangulate_track(camera, poses, &obs, config) else {
            continue;
        };
        let mean_reprojection = |point: &Point3<f64>| -> f64 {
            let mut sum = 0.0;
            let mut count = 0usize;
            for &(image, pixel) in &obs {
                let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                    continue;
                };
                if let Some(error) = reprojection_error_px(camera, pose, point, &pixel) {
                    sum += error;
                    count += 1;
                }
            }
            if count == 0 {
                f64::INFINITY
            } else {
                sum / count as f64
            }
        };
        let should_replace = match state.points.get(track_id).and_then(Option::as_ref) {
            None => true,
            Some(current) => {
                let current_error = mean_reprojection(current);
                let candidate_error = mean_reprojection(&candidate);
                candidate_error.is_finite() && candidate_error + 1e-9 < current_error
            }
        };
        if should_replace {
            state.retriangulate_point(track_id, candidate);
        }
    }
    track_point.copy_from_slice(&state.points[..track_point.len()]);
}

fn triangulate_correspondence_pending(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    let mut state = CorrespondencePointState::from_tracks(tracks, track_point);
    triangulate_correspondence_pending_with_state(
        camera,
        features,
        tracks,
        poses,
        config,
        track_point,
        &mut state,
    );
}

fn triangulate_pending_with_config(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
) {
    if config.incremental_correspondence_triangulation {
        triangulate_correspondence_pending(camera, features, tracks, poses, config, track_point);
    } else {
        triangulate_pending(camera, features, tracks, poses, config, track_point);
    }
}

fn triangulate_pending_with_config_and_state(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
    track_point: &mut [Option<Point3<f64>>],
    state: Option<&mut CorrespondencePointState>,
) {
    if config.incremental_correspondence_triangulation {
        if let Some(state) = state {
            triangulate_correspondence_pending_with_state(
                camera,
                features,
                tracks,
                poses,
                config,
                track_point,
                state,
            );
        } else {
            triangulate_correspondence_pending(
                camera,
                features,
                tracks,
                poses,
                config,
                track_point,
            );
        }
    } else {
        triangulate_pending(camera, features, tracks, poses, config, track_point);
    }
}

/// Whether a track's widest parallax `angle` (radians) clears the triangulation
/// gate: the strict `min_triangulation_angle_deg`, or — with the multi-view
/// exemption (`low_parallax_min_observations`) configured — the relaxed
/// `low_parallax_min_angle_deg` floor once at least that many views observe it.
fn parallax_angle_ok(angle: f64, num_obs: usize, config: &IncrementalSfmConfig) -> bool {
    if angle >= config.min_triangulation_angle_deg.to_radians() {
        return true;
    }
    match config.low_parallax_min_observations {
        Some(min_obs) => {
            num_obs >= min_obs && angle >= config.low_parallax_min_angle_deg.to_radians()
        }
        None => false,
    }
}

#[allow(clippy::too_many_arguments)]
/// Triangulate one track from its registered observations: choose the
/// widest-parallax view pair, DLT-triangulate, and validate cheirality,
/// parallax, and reprojection in both views.
pub(crate) fn triangulate_track(
    camera: &Camera,
    poses: &[Option<Pose>],
    obs: &[(usize, Point2<f64>)],
    config: &IncrementalSfmConfig,
) -> Option<Point3<f64>> {
    let max_reproj = config.max_reprojection_error_px;
    // Precompute world-frame bearing rays for each observation.
    let mut rays: Vec<Vector3<f64>> = Vec::with_capacity(obs.len());
    for &(image, px) in obs {
        let pose = poses[image].as_ref()?;
        let n = camera.normalize_pixel(&px)?;
        let bearing = Vector3::new(n.x, n.y, 1.0).normalize();
        rays.push(pose.camera_to_world().rotation * bearing);
    }

    // Pick the observation pair with the smallest |cos| (widest parallax).
    let mut best: Option<(usize, usize, f64)> = None;
    for a in 0..obs.len() {
        for b in (a + 1)..obs.len() {
            let cos = rays[a].dot(&rays[b]).clamp(-1.0, 1.0).abs();
            if best.is_none_or(|(_, _, c)| cos < c) {
                best = Some((a, b, cos));
            }
        }
    }
    let (a, b, cos) = best?;
    // Widest-pair parallax angle; accept on the strict gate or the multi-view
    // exemption (a long low-parallax track is well-constrained by its many views).
    if !parallax_angle_ok(cos.acos(), obs.len(), config) {
        return None; // insufficient parallax
    }

    let (image_a, px_a) = obs[a];
    let (image_b, px_b) = obs[b];
    let pose_a = poses[image_a].as_ref()?;
    let pose_b = poses[image_b].as_ref()?;

    // Relative transform mapping camera-a frame to camera-b frame.
    let a_to_b = pose_b.world_to_camera.compose(&pose_a.camera_to_world());
    let point_cam_a = triangulate_two_view_left_frame(camera, camera, &a_to_b, &px_a, &px_b)?;
    if !point_cam_a.z.is_finite() || point_cam_a.z <= 0.0 {
        return None;
    }
    let point_world = pose_a.camera_to_world().transform_point(&point_cam_a);

    // Validate reprojection in both anchor views.
    for (image, px) in [(image_a, px_a), (image_b, px_b)] {
        let pose = poses[image].as_ref()?;
        let err = reprojection_error_px(camera, pose, &point_world, &px)?;
        if err > max_reproj {
            return None;
        }
    }
    Some(point_world)
}

type TrackObservation = (usize, usize);
type TrackEdge = (TrackObservation, TrackObservation);

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct PoseGuidedTrackSplitStats {
    input_components: usize,
    bridge_cuts: usize,
    bridge_cut_components: usize,
    bridge_cut_sizes: Vec<(usize, usize)>,
    preserved_components: usize,
    split_components: usize,
    hypotheses_tested: usize,
    emitted_tracks: usize,
    assigned_observations: usize,
    discarded_observations: usize,
    /// Histogram for graph-support admissions.  Buckets 0..=6 are exact and
    /// bucket 7 contains support from seven or more distinct images.
    graph_support_histogram: [usize; 8],
    graph_supported_tracks: usize,
    graph_length_two_tracks: usize,
    /// Number of complementary track unions accepted by the optional
    /// post-split merge pass.
    merged_tracks: usize,
    /// Number of active track-pair groups whose posed union was tested.
    merge_candidates_tested: usize,
}

#[derive(Debug, Clone, Default)]
struct PoseGuidedTrackSplitOutput {
    tracks: Vec<Vec<TrackObservation>>,
    points: Vec<Option<Point3<f64>>>,
    /// For every final track formed by one or more post-split unions, retain
    /// the exact pre-merge fragments.  The mapper can therefore undo only a
    /// merge whose post-BA observations fail the ordinary hard gate, without
    /// discarding healthy unions from the same candidate model.
    merge_restorations: Vec<PoseGuidedMergeRestoration>,
    stats: PoseGuidedTrackSplitStats,
}

#[derive(Debug, Clone, PartialEq)]
struct PoseGuidedMergeRestoration {
    /// Stable indices in the pre-merge split partition.
    source_track_ids: Vec<usize>,
    source_tracks: Vec<Vec<TrackObservation>>,
    source_points: Vec<Option<Point3<f64>>>,
    merged_track: Vec<TrackObservation>,
}

#[derive(Debug, Clone)]
struct PoseGuidedTrackCandidate {
    observations: Vec<TrackObservation>,
    point: Point3<f64>,
    median_reprojection_px: f64,
    mean_reprojection_px: f64,
    anchor: TrackEdge,
    parallax_rad: f64,
    graph_support_counts: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PoseGuidedTrackMergeCandidate {
    left: usize,
    right: usize,
    observations: Vec<TrackObservation>,
    point: Point3<f64>,
    /// Number of distinct image-pair edges crossing the two tracks.  Multiple
    /// orientation/keypoint rows from one image pair count once: they are not
    /// independent multi-view support for a merge.
    cross_image_edges: usize,
    parallax_rad: f64,
    median_reprojection_px: f64,
    mean_reprojection_px: f64,
}

impl PartialEq for PoseGuidedTrackMergeCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for PoseGuidedTrackMergeCandidate {}

impl PartialOrd for PoseGuidedTrackMergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PoseGuidedTrackMergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.observations
            .len()
            .cmp(&other.observations.len())
            .then_with(|| self.cross_image_edges.cmp(&other.cross_image_edges))
            .then_with(|| self.parallax_rad.total_cmp(&other.parallax_rad))
            // Lower robust reprojection is preferred.  Reversing the operands
            // keeps BinaryHeap's largest item as the best candidate.
            .then_with(|| {
                other
                    .median_reprojection_px
                    .total_cmp(&self.median_reprojection_px)
            })
            .then_with(|| {
                other
                    .mean_reprojection_px
                    .total_cmp(&self.mean_reprojection_px)
            })
            // Stable physical track order is the final deterministic tie
            // break.  Smaller IDs win when every geometric score ties.
            .then_with(|| other.left.cmp(&self.left))
            .then_with(|| other.right.cmp(&self.right))
    }
}

type PoseGuidedMergeTrack = (Vec<TrackObservation>, Option<Point3<f64>>);
type PoseGuidedTrackMergeOutput = (
    Vec<Vec<TrackObservation>>,
    Vec<Option<Point3<f64>>>,
    usize,
    usize,
);

fn pose_guided_merge_reprojection_gate(
    config: &IncrementalSfmConfig,
    split_gate: f64,
) -> Option<f64> {
    let gate = config
        .pose_guided_merge_max_reprojection_error_px
        .unwrap_or(split_gate);
    (gate.is_finite() && gate > 0.0).then_some(gate)
}

/// Fit one prospective merged track against the complete fixed-pose model.
/// `config.max_reprojection_error_px` is the split-only gate supplied by the
/// caller.  Unlike a pair-only check, every observation must be finite,
/// front-facing, and within that gate after local point refinement.
#[allow(clippy::too_many_arguments)]
fn pose_guided_merge_fit(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    config: &IncrementalSfmConfig,
) -> Option<(Point3<f64>, f64, f64, f64)> {
    if observations.len() < 2
        || !config.max_reprojection_error_px.is_finite()
        || config.max_reprojection_error_px <= 0.0
    {
        return None;
    }
    let mut images = HashSet::new();
    if !observations.iter().all(|&(image, _)| images.insert(image)) {
        return None;
    }
    let pixels = observations
        .iter()
        .map(|&(image, keypoint)| {
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint))
                .copied()
                .map(|pixel| (image, pixel))
        })
        .collect::<Option<Vec<_>>>()?;
    let initial = triangulate_track(camera, poses, &pixels, config)?;
    let point =
        refine_pose_guided_point(camera, features, poses, observations, initial).unwrap_or(initial);
    if !point.coords.iter().all(|value| value.is_finite()) {
        return None;
    }
    let mut errors = Vec::with_capacity(observations.len());
    for &(image, keypoint) in observations {
        let pose = poses.get(image)?.as_ref()?;
        let pixel = features.get(image)?.keypoints.get(keypoint)?;
        let error = reprojection_error_px(camera, pose, &point, pixel)?;
        if !error.is_finite() || error > config.max_reprojection_error_px {
            return None;
        }
        errors.push(error);
    }
    errors.sort_by(f64::total_cmp);
    let median = errors[errors.len() / 2];
    let mean = errors.iter().sum::<f64>() / errors.len() as f64;
    let parallax = track_max_parallax(poses, observations, &point);
    (median.is_finite() && mean.is_finite() && parallax.is_finite())
        .then_some((point, parallax, median, mean))
}

/// Build one deterministic candidate for a pair of currently active tracks.
#[allow(clippy::too_many_arguments)]
fn pose_guided_make_merge_candidate(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    left: usize,
    right: usize,
    left_track: &[TrackObservation],
    right_track: &[TrackObservation],
    cross_edges: &BTreeSet<TrackEdge>,
    config: &IncrementalSfmConfig,
) -> Option<PoseGuidedTrackMergeCandidate> {
    if cross_edges.is_empty() {
        return None;
    }
    let mut image_set = HashSet::new();
    if !left_track
        .iter()
        .chain(right_track)
        .all(|&(image, _)| image_set.insert(image))
    {
        return None;
    }
    let mut observations = left_track
        .iter()
        .chain(right_track)
        .copied()
        .collect::<Vec<_>>();
    observations.sort_unstable();
    observations.dedup();
    if observations.len() != left_track.len() + right_track.len() {
        return None;
    }
    let (point, parallax_rad, median_reprojection_px, mean_reprojection_px) =
        pose_guided_merge_fit(camera, features, poses, &observations, config)?;
    let cross_image_edges = cross_edges
        .iter()
        .filter(|&&(first, second)| first.0 != second.0)
        .map(|&(first, second)| (first.0.min(second.0), first.0.max(second.0)))
        .collect::<BTreeSet<_>>()
        .len();
    (cross_image_edges > 0).then_some(PoseGuidedTrackMergeCandidate {
        left,
        right,
        observations,
        point,
        cross_image_edges,
        parallax_rad,
        median_reprojection_px,
        mean_reprojection_px,
    })
}

/// Collect candidates involving one active track.  Candidate groups are
/// keyed by the other active track and by exact verified observation edge, so
/// an orientation-row permutation cannot change either the geometry or the
/// support score.  Only `other > track_id` is emitted to avoid duplicates.
#[allow(clippy::too_many_arguments)]
fn pose_guided_collect_merge_candidates(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    track_id: usize,
    active: &[Option<PoseGuidedMergeTrack>],
    observation_to_track: &HashMap<TrackObservation, usize>,
    edge_adjacency: &HashMap<TrackObservation, Vec<TrackObservation>>,
    config: &IncrementalSfmConfig,
    include_lower_ids: bool,
) -> (Vec<PoseGuidedTrackMergeCandidate>, usize) {
    let Some(Some((track, _))) = active.get(track_id) else {
        return (Vec::new(), 0);
    };
    let mut grouped = BTreeMap::<usize, BTreeSet<TrackEdge>>::new();
    for &observation in track {
        for &other in edge_adjacency
            .get(&observation)
            .into_iter()
            .flat_map(|neighbours| neighbours.iter())
        {
            let Some(&other_id) = observation_to_track.get(&other) else {
                continue;
            };
            if other_id == track_id
                || (!include_lower_ids && other_id < track_id)
                || active.get(other_id).and_then(Option::as_ref).is_none()
            {
                continue;
            }
            let edge = if observation <= other {
                (observation, other)
            } else {
                (other, observation)
            };
            grouped.entry(other_id).or_default().insert(edge);
        }
    }

    let mut candidates = Vec::new();
    let mut tested = 0usize;
    for (other_id, cross_edges) in grouped {
        let Some(Some((other_track, _))) = active.get(other_id) else {
            continue;
        };
        tested += 1;
        let (left_id, left_track, right_id, right_track) = if track_id < other_id {
            (track_id, track, other_id, other_track)
        } else {
            (other_id, other_track, track_id, track)
        };
        if let Some(candidate) = pose_guided_make_merge_candidate(
            camera,
            features,
            poses,
            left_id,
            right_id,
            left_track,
            right_track,
            &cross_edges,
            config,
        ) {
            candidates.push(candidate);
        }
    }
    (candidates, tested)
}

/// Merge complementary pose-guided tracks with a verified cross-track edge.
/// The active-track table gives each union a stable identity; stale heap
/// entries are discarded, while every candidate involving a newly merged
/// track is rebuilt from the new observation set.  Thus geometry is
/// recomputed after every accepted union without rescanning unrelated pairs.
#[allow(clippy::too_many_arguments)]
fn pose_guided_merge_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    poses: &[Option<Pose>],
    tracks: &[Vec<TrackObservation>],
    points: &[Option<Point3<f64>>],
    config: &IncrementalSfmConfig,
) -> PoseGuidedTrackMergeOutput {
    if tracks.len() != points.len() || tracks.is_empty() {
        return (tracks.to_vec(), points.to_vec(), 0, 0);
    }

    // Canonicalise the starting partition so the stable IDs used as final
    // tie-breaks are physical observation order, not caller traversal order.
    let mut initial = tracks
        .iter()
        .zip(points.iter().copied())
        .map(|(track, point)| {
            let mut track = track.clone();
            track.sort_unstable();
            (track, point)
        })
        .collect::<Vec<_>>();
    initial.sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut observation_to_track = HashMap::<TrackObservation, usize>::new();
    let mut active = Vec::<Option<PoseGuidedMergeTrack>>::with_capacity(initial.len());
    for (track_id, (track, point)) in initial.into_iter().enumerate() {
        if track
            .iter()
            .any(|observation| observation_to_track.contains_key(observation))
        {
            // The splitter normally guarantees disjoint observations.  If a
            // future caller violates that invariant, keep the canonical input
            // untouched rather than silently assigning one row twice.
            let mut unchanged = active.into_iter().flatten().collect::<Vec<_>>();
            unchanged.push((track, point));
            unchanged.sort_unstable_by(|left, right| left.0.cmp(&right.0));
            return (
                unchanged.iter().map(|(track, _)| track.clone()).collect(),
                unchanged.into_iter().map(|(_, point)| point).collect(),
                0,
                0,
            );
        }
        for &observation in &track {
            observation_to_track.insert(observation, track_id);
        }
        active.push(Some((track, point)));
    }

    let mut edge_set = BTreeSet::<TrackEdge>::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        for &(left, right) in &pair.matches {
            let left = (pair.image_i, left);
            let right = (pair.image_j, right);
            edge_set.insert(if left <= right {
                (left, right)
            } else {
                (right, left)
            });
        }
    }
    let mut edge_adjacency = HashMap::<TrackObservation, Vec<TrackObservation>>::new();
    for (left, right) in edge_set {
        edge_adjacency.entry(left).or_default().push(right);
        edge_adjacency.entry(right).or_default().push(left);
    }
    for neighbours in edge_adjacency.values_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }

    let mut heap = BinaryHeap::new();
    let mut candidates_tested = 0usize;
    for track_id in 0..active.len() {
        let (candidates, tested) = pose_guided_collect_merge_candidates(
            camera,
            features,
            poses,
            track_id,
            &active,
            &observation_to_track,
            &edge_adjacency,
            config,
            false,
        );
        candidates_tested += tested;
        heap.extend(candidates);
    }

    let mut merges = 0usize;
    while let Some(candidate) = heap.pop() {
        let Some(Some((left_track, _))) = active.get(candidate.left) else {
            continue;
        };
        let Some(Some((right_track, _))) = active.get(candidate.right) else {
            continue;
        };
        // Both tracks remain unchanged while active.  Refit the popped union
        // once more before committing; this guards against accidental future
        // changes to candidate generation and makes the acceptance condition
        // explicit at the mutation point.
        let mut cross_edges = BTreeSet::new();
        for &observation in &candidate.observations {
            for &other in edge_adjacency
                .get(&observation)
                .into_iter()
                .flat_map(|neighbours| neighbours.iter())
            {
                let Some(&left_id) = observation_to_track.get(&observation) else {
                    continue;
                };
                let Some(&right_id) = observation_to_track.get(&other) else {
                    continue;
                };
                if left_id == right_id
                    || !((left_id == candidate.left && right_id == candidate.right)
                        || (left_id == candidate.right && right_id == candidate.left))
                {
                    continue;
                }
                cross_edges.insert(if observation <= other {
                    (observation, other)
                } else {
                    (other, observation)
                });
            }
        }
        let Some(refit) = pose_guided_make_merge_candidate(
            camera,
            features,
            poses,
            candidate.left,
            candidate.right,
            left_track,
            right_track,
            &cross_edges,
            config,
        ) else {
            continue;
        };

        let new_id = active.len();
        active[candidate.left] = None;
        active[candidate.right] = None;
        for &observation in &refit.observations {
            observation_to_track.insert(observation, new_id);
        }
        active.push(Some((refit.observations, Some(refit.point))));
        merges += 1;

        let (new_candidates, tested) = pose_guided_collect_merge_candidates(
            camera,
            features,
            poses,
            new_id,
            &active,
            &observation_to_track,
            &edge_adjacency,
            config,
            true,
        );
        candidates_tested += tested;
        heap.extend(new_candidates);
    }

    let mut result = active.into_iter().flatten().collect::<Vec<_>>();
    result.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    (
        result.iter().map(|(track, _)| track.clone()).collect(),
        result.into_iter().map(|(_, point)| point).collect(),
        merges,
        candidates_tested,
    )
}

/// Recover the exact source fragments for every final track that contains a
/// post-split union.  The merge routine only joins complete, disjoint tracks,
/// so an observation-to-source lookup is sufficient and keeps this provenance
/// independent of heap traversal order.
fn pose_guided_merge_restorations(
    source_tracks: &[Vec<TrackObservation>],
    source_points: &[Option<Point3<f64>>],
    merged_tracks: &[Vec<TrackObservation>],
) -> Vec<PoseGuidedMergeRestoration> {
    if source_tracks.len() != source_points.len() {
        return Vec::new();
    }
    let mut source_by_observation = HashMap::<TrackObservation, usize>::new();
    for (source_id, track) in source_tracks.iter().enumerate() {
        for &observation in track {
            if source_by_observation
                .insert(observation, source_id)
                .is_some()
            {
                // The splitter guarantees disjoint source tracks.  Refuse to
                // manufacture restoration data if a future caller violates
                // that invariant.
                return Vec::new();
            }
        }
    }

    let mut restorations = Vec::new();
    for merged_track in merged_tracks {
        let mut source_track_ids = merged_track
            .iter()
            .filter_map(|observation| source_by_observation.get(observation).copied())
            .collect::<Vec<_>>();
        if source_track_ids.len() != merged_track.len() {
            continue;
        }
        source_track_ids.sort_unstable();
        source_track_ids.dedup();
        if source_track_ids.len() < 2 {
            continue;
        }
        let source_tracks_for_restore = source_track_ids
            .iter()
            .map(|&source_id| source_tracks[source_id].clone())
            .collect::<Vec<_>>();
        let source_points_for_restore = source_track_ids
            .iter()
            .map(|&source_id| source_points[source_id])
            .collect::<Vec<_>>();
        restorations.push(PoseGuidedMergeRestoration {
            source_track_ids,
            source_tracks: source_tracks_for_restore,
            source_points: source_points_for_restore,
            merged_track: merged_track.clone(),
        });
    }
    restorations
}

fn pose_guided_track_reprojection_valid(
    camera: &Camera,
    features: &[FeatureSet],
    track: &[TrackObservation],
    pose_list: &[Option<Pose>],
    point: Option<&Point3<f64>>,
    max_error: f64,
) -> bool {
    let Some(point) = point else {
        return false;
    };
    point.coords.iter().all(|value| value.is_finite())
        && max_error.is_finite()
        && max_error > 0.0
        && track.iter().all(|&(image, keypoint)| {
            let (Some(pose), Some(pixel)) = (
                pose_list.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint)),
            ) else {
                return false;
            };
            reprojection_error_px(camera, pose, point, pixel)
                .is_some_and(|error| error.is_finite() && error <= max_error)
        })
}

/// Restore only merged tracks that fail the ordinary post-BA hard gate.  The
/// caller reruns BA after this mutation; if that second solve fails, its outer
/// candidate snapshot restores the complete pre-split model.
fn pose_guided_restore_invalid_merges(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    tracks: &mut Vec<Vec<TrackObservation>>,
    points: &mut Vec<Option<Point3<f64>>>,
    restorations: &[PoseGuidedMergeRestoration],
    max_error: f64,
) -> (usize, usize) {
    if tracks.len() != points.len() || restorations.is_empty() {
        return (restorations.len(), 0);
    }
    let mut invalid_tracks = BTreeSet::<Vec<TrackObservation>>::new();
    for restoration in restorations {
        if !tracks
            .iter()
            .any(|track| track == &restoration.merged_track)
        {
            continue;
        }
        if !pose_guided_track_reprojection_valid(
            camera,
            features,
            &restoration.merged_track,
            poses,
            tracks
                .iter()
                .position(|track| track == &restoration.merged_track)
                .and_then(|index| points[index].as_ref()),
            max_error,
        ) {
            invalid_tracks.insert(restoration.merged_track.clone());
        }
    }
    if invalid_tracks.is_empty() {
        return (restorations.len(), 0);
    }

    let mut restored = Vec::with_capacity(tracks.len() + invalid_tracks.len());
    for (track, point) in tracks.iter().zip(points.iter()) {
        if invalid_tracks.contains(track) {
            let restoration = restorations
                .iter()
                .find(|restoration| restoration.merged_track == *track)
                .expect("invalid merged track has restoration provenance");
            restored.extend(
                restoration
                    .source_tracks
                    .iter()
                    .cloned()
                    .zip(restoration.source_points.iter().copied()),
            );
        } else {
            restored.push((track.clone(), *point));
        }
    }
    restored.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    *tracks = restored.iter().map(|(track, _)| track.clone()).collect();
    *points = restored.into_iter().map(|(_, point)| point).collect();
    (restorations.len(), invalid_tracks.len())
}

/// Validate only the merged tracks that survived selective restoration.  The
/// split-only tracks have their own candidate/objective guard; a bad unrelated
/// split track must not make a healthy merge appear to be the culprit.
#[allow(clippy::too_many_arguments)]
fn pose_guided_merge_restorations_reprojection_valid(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    tracks: &[Vec<TrackObservation>],
    points: &[Option<Point3<f64>>],
    restorations: &[PoseGuidedMergeRestoration],
    restored_merges: usize,
    max_error: f64,
) -> bool {
    if tracks.len() != points.len() || !max_error.is_finite() || max_error <= 0.0 {
        return false;
    }
    let mut active_merges = 0usize;
    for restoration in restorations {
        let Some(index) = tracks
            .iter()
            .position(|track| track == &restoration.merged_track)
        else {
            continue;
        };
        active_merges += 1;
        if !pose_guided_track_reprojection_valid(
            camera,
            features,
            &tracks[index],
            poses,
            points[index].as_ref(),
            max_error,
        ) {
            return false;
        }
    }
    // If every tentative merged track was restored, there is no surviving
    // union to validate; the outer split/objective guard still applies.  A
    // mismatch indicates lost provenance and is rejected conservatively.
    active_merges + restored_merges == restorations.len()
}

#[derive(Debug, Clone, Default)]
struct PoseGuidedBridgeCutOutput {
    components: Vec<Vec<TrackObservation>>,
    cut_edges: Vec<TrackEdge>,
    cut_sizes: Vec<(usize, usize)>,
}

type PoseGuidedComponentGraph = (
    HashMap<TrackObservation, usize>,
    Vec<Vec<TrackEdge>>,
    Vec<HashMap<TrackObservation, Vec<TrackObservation>>>,
);

/// Build the deterministic verified correspondence graph used by the
/// pose-guided diagnostics.  The returned component-local edge lists are
/// deduplicated and sorted, so callers can safely run graph algorithms without
/// depending on pair or match traversal order.
fn build_pose_guided_component_graph(
    components: &[Vec<TrackObservation>],
    pairwise: &[PairwiseMatches],
) -> Option<PoseGuidedComponentGraph> {
    let mut component_of = HashMap::<TrackObservation, usize>::new();
    for (component_id, component) in components.iter().enumerate() {
        for &observation in component {
            if component_of.insert(observation, component_id).is_some() {
                return None;
            }
        }
    }

    let mut edges_by_component = vec![Vec::<TrackEdge>::new(); components.len()];
    let mut adjacency_by_component =
        vec![HashMap::<TrackObservation, Vec<TrackObservation>>::new(); components.len()];
    let mut edge_seen = HashSet::<(usize, TrackEdge)>::new();
    for pair in pairwise {
        if pair.image_i == pair.image_j {
            continue;
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            let left = (pair.image_i, keypoint_i);
            let right = (pair.image_j, keypoint_j);
            let (Some(&left_component), Some(&right_component)) =
                (component_of.get(&left), component_of.get(&right))
            else {
                continue;
            };
            if left_component != right_component {
                continue;
            }
            let edge = if left <= right {
                (left, right)
            } else {
                (right, left)
            };
            if edge_seen.insert((left_component, edge)) {
                edges_by_component[left_component].push(edge);
                adjacency_by_component[left_component]
                    .entry(edge.0)
                    .or_default()
                    .push(edge.1);
                adjacency_by_component[left_component]
                    .entry(edge.1)
                    .or_default()
                    .push(edge.0);
            }
        }
    }
    for edges in &mut edges_by_component {
        edges.sort_unstable();
    }
    for adjacency in &mut adjacency_by_component {
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
    }
    Some((component_of, edges_by_component, adjacency_by_component))
}

/// Find bridges with an iterative Tarjan DFS.  The iterative form avoids
/// recursion depth depending on a large correspondence component; every
/// adjacency and tie-break is sorted before traversal for permutation
/// invariance.
fn pose_guided_find_bridges(
    observations: &[TrackObservation],
    edges: &[TrackEdge],
) -> Vec<TrackEdge> {
    let mut nodes = observations.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    let node_index = nodes
        .iter()
        .enumerate()
        .map(|(index, &observation)| (observation, index))
        .collect::<HashMap<_, _>>();

    let mut valid_edges = Vec::<TrackEdge>::new();
    let mut edge_nodes = Vec::<(usize, usize)>::new();
    let mut adjacency = vec![Vec::<(usize, usize)>::new(); nodes.len()];
    let mut seen = HashSet::new();
    for &edge @ (left, right) in edges {
        if left == right || !seen.insert(edge) {
            continue;
        }
        let (Some(&left_index), Some(&right_index)) =
            (node_index.get(&left), node_index.get(&right))
        else {
            continue;
        };
        let edge_index = valid_edges.len();
        valid_edges.push(edge);
        edge_nodes.push((left_index, right_index));
        adjacency[left_index].push((right_index, edge_index));
        adjacency[right_index].push((left_index, edge_index));
    }
    for neighbours in &mut adjacency {
        neighbours.sort_unstable();
    }

    #[derive(Clone, Copy)]
    struct Frame {
        node: usize,
        parent_edge: Option<usize>,
        next_neighbour: usize,
    }

    let unvisited = usize::MAX;
    let mut discovery = vec![unvisited; nodes.len()];
    let mut low = vec![unvisited; nodes.len()];
    let mut time = 0usize;
    let mut bridges = Vec::new();
    for root in 0..nodes.len() {
        if discovery[root] != unvisited {
            continue;
        }
        discovery[root] = time;
        low[root] = time;
        time += 1;
        let mut stack = vec![Frame {
            node: root,
            parent_edge: None,
            next_neighbour: 0,
        }];
        while let Some(frame) = stack.last_mut() {
            let node = frame.node;
            if frame.next_neighbour < adjacency[node].len() {
                let (neighbour, edge_index) = adjacency[node][frame.next_neighbour];
                frame.next_neighbour += 1;
                if frame.parent_edge == Some(edge_index) {
                    continue;
                }
                if discovery[neighbour] == unvisited {
                    discovery[neighbour] = time;
                    low[neighbour] = time;
                    time += 1;
                    stack.push(Frame {
                        node: neighbour,
                        parent_edge: Some(edge_index),
                        next_neighbour: 0,
                    });
                } else {
                    low[node] = low[node].min(discovery[neighbour]);
                }
            } else {
                let finished = stack.pop().expect("non-empty Tarjan stack");
                if let Some(parent_edge) = finished.parent_edge {
                    let (left, right) = edge_nodes[parent_edge];
                    let parent = if left == finished.node { right } else { left };
                    low[parent] = low[parent].min(low[finished.node]);
                    if low[finished.node] > discovery[parent] {
                        bridges.push(valid_edges[parent_edge]);
                    }
                }
            }
        }
    }
    bridges.sort_unstable();
    bridges.dedup();
    bridges
}

/// Return the two connected sides after removing one candidate bridge.
fn pose_guided_bridge_sides(
    observations: &[TrackObservation],
    edges: &[TrackEdge],
    bridge: TrackEdge,
) -> Option<(Vec<TrackObservation>, Vec<TrackObservation>)> {
    let mut adjacency = HashMap::<TrackObservation, Vec<TrackObservation>>::new();
    for &observation in observations {
        adjacency.entry(observation).or_default();
    }
    for &edge @ (left, right) in edges {
        if edge == bridge {
            continue;
        }
        adjacency.entry(left).or_default().push(right);
        adjacency.entry(right).or_default().push(left);
    }
    for neighbours in adjacency.values_mut() {
        neighbours.sort_unstable();
        neighbours.dedup();
    }
    let mut left_side = HashSet::new();
    let mut stack = vec![bridge.0];
    while let Some(observation) = stack.pop() {
        if !left_side.insert(observation) {
            continue;
        }
        if let Some(neighbours) = adjacency.get(&observation) {
            for &neighbour in neighbours.iter().rev() {
                if !left_side.contains(&neighbour) {
                    stack.push(neighbour);
                }
            }
        }
    }
    if left_side.len() == observations.len() {
        return None;
    }
    let mut first = left_side.into_iter().collect::<Vec<_>>();
    let mut second = observations
        .iter()
        .copied()
        .filter(|observation| !first.contains(observation))
        .collect::<Vec<_>>();
    first.sort_unstable();
    second.sort_unstable();
    (!first.is_empty() && !second.is_empty()).then_some((first, second))
}

fn pose_guided_bridge_side_is_eligible(observations: &[TrackObservation]) -> bool {
    if observations.len() < 2 {
        return false;
    }
    let mut images = HashSet::new();
    observations.iter().all(|&(image, _)| images.insert(image)) && images.len() >= 2
}

/// Fit one posed point to a component side.  The gate is deliberately the
/// same split gate used by the existing splitter; no bridge-specific pixel or
/// parallax threshold is introduced.  A `None` result means that all side
/// observations cannot be explained by one finite, cheirality-valid point.
fn pose_guided_bridge_fit(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    config: &IncrementalSfmConfig,
) -> Option<Point3<f64>> {
    if observations.len() < 2 {
        return None;
    }
    let pixels = observations
        .iter()
        .map(|&(image, keypoint)| {
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint))
                .copied()
                .map(|pixel| (image, pixel))
        })
        .collect::<Option<Vec<_>>>()?;
    let initial = triangulate_track(camera, poses, &pixels, config)?;
    let point =
        refine_pose_guided_point(camera, features, poses, observations, initial).unwrap_or(initial);
    let mut sum = 0.0;
    let mut max_error = 0.0f64;
    for &(image, keypoint) in observations {
        let pose = poses.get(image)?.as_ref()?;
        let pixel = features.get(image)?.keypoints.get(keypoint)?;
        let error = reprojection_error_px(camera, pose, &point, pixel)?;
        if !error.is_finite() {
            return None;
        }
        sum += error;
        max_error = max_error.max(error);
    }
    let mean_error = sum / observations.len() as f64;
    (mean_error.is_finite()
        && max_error.is_finite()
        && mean_error <= config.max_reprojection_error_px
        && max_error <= config.max_reprojection_error_px)
        .then_some(point)
}

/// Iteratively cut accepted graph bridges, recomputing Tarjan structure after
/// every cut.  A bridge is accepted only when both sides are multi-view posed
/// fits and the combined observations fail the same one-point fit.  This
/// conservative combined-fit test preserves genuine sparse chains while
/// separating a false bridge between two physical points.
fn pose_guided_bridge_cut_component(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    edges: &[TrackEdge],
    config: &IncrementalSfmConfig,
) -> PoseGuidedBridgeCutOutput {
    let mut initial = observations.to_vec();
    initial.sort_unstable();
    initial.dedup();
    let mut partitions = if initial.is_empty() {
        Vec::new()
    } else {
        vec![initial]
    };
    let mut output = PoseGuidedBridgeCutOutput::default();

    loop {
        let mut accepted = None;
        for (partition_index, partition) in partitions.iter().enumerate() {
            let partition_set = partition.iter().copied().collect::<HashSet<_>>();
            let partition_edges = edges
                .iter()
                .copied()
                .filter(|&(left, right)| {
                    partition_set.contains(&left) && partition_set.contains(&right)
                })
                .collect::<Vec<_>>();
            for bridge in pose_guided_find_bridges(partition, &partition_edges) {
                let Some((first, second)) =
                    pose_guided_bridge_sides(partition, &partition_edges, bridge)
                else {
                    continue;
                };
                if !pose_guided_bridge_side_is_eligible(&first)
                    || !pose_guided_bridge_side_is_eligible(&second)
                {
                    continue;
                }
                if pose_guided_bridge_fit(camera, features, poses, &first, config).is_none()
                    || pose_guided_bridge_fit(camera, features, poses, &second, config).is_none()
                {
                    continue;
                }
                let mut combined = first.clone();
                combined.extend(second.iter().copied());
                combined.sort_unstable();
                if pose_guided_bridge_fit(camera, features, poses, &combined, config).is_some() {
                    continue;
                }
                accepted = Some((partition_index, bridge, first, second));
                break;
            }
            if accepted.is_some() {
                break;
            }
        }

        let Some((partition_index, bridge, first, second)) = accepted else {
            break;
        };
        output.cut_edges.push(bridge);
        output.cut_sizes.push((first.len(), second.len()));
        partitions[partition_index] = first;
        partitions.push(second);
        partitions.sort_unstable();
    }

    output.components = partitions;
    output
}

/// Write a pose-guided candidate partition as a compact observation table for
/// offline topology comparisons.  This is deliberately not part of the
/// reconstruction output: the caller must opt in with
/// `VISLOC_SFM_DEBUG_POSE_SPLIT_DUMP=/path/to/file.tsv`.
fn dump_pose_guided_track_split(
    path: &std::path::Path,
    tracks: &[Vec<TrackObservation>],
) -> std::io::Result<usize> {
    let mut output = std::fs::File::create(path)?;
    writeln!(output, "track_id\timage_index\tkeypoint_index")?;
    let mut observations = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        for &(image, keypoint) in track {
            writeln!(output, "{track_id}\t{image}\t{keypoint}")?;
            observations += 1;
        }
    }
    Ok(observations)
}

/// Split legacy union components using the current complete camera model.
///
/// The ordinary union-find path intentionally discards every component that
/// contains two observations from one image.  Once all cameras have been
/// registered, however, their fixed poses provide a safe, GT-independent way
/// to test several 3-D hypotheses inside such a component.  This pass ranks
/// verified anchor edges by posed parallax, retains at most one observation per
/// image whose reprojection and cheirality are valid, locally refines each
/// accepted point, and removes its observations before searching the residual
/// component.  With `pose_guided_graph_support`, every observation after the
/// anchor also needs two direct verified supports from distinct hypothesis
/// images, and multi-view emissions need two independent cross-image edges.
/// Clean components that already fit their existing point are copied
/// byte-for-byte at the observation level.
///
/// A partial pose model is deliberately a no-op: classifying an unregistered
/// observation by an arbitrary image-space proxy would turn this diagnostic
/// into a new growth policy.  The caller may therefore run the pass only after
/// the initial reconstruction is complete, or explicitly compare an oracle
/// complete pose model through the existing initial-pose diagnostic.
#[allow(clippy::too_many_arguments)]
fn pose_guided_split_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    tracks: &[Vec<TrackObservation>],
    conflicting_components: &[Vec<TrackObservation>],
    old_points: &[Option<Point3<f64>>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
) -> Option<PoseGuidedTrackSplitOutput> {
    if !poses.iter().all(Option::is_some)
        || config.conflict_recovery_max_hypotheses == 0
        || tracks.len() != old_points.len()
    {
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: pose-guided track split skipped complete={} hypotheses={} tracks={} points={}",
                poses.iter().all(Option::is_some),
                config.conflict_recovery_max_hypotheses,
                tracks.len(),
                old_points.len(),
            );
        }
        return None;
    }
    let split_max_reprojection_error = config
        .pose_guided_split_max_reprojection_error_px
        .unwrap_or(config.max_reprojection_error_px);
    if !split_max_reprojection_error.is_finite() || split_max_reprojection_error <= 0.0 {
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: pose-guided track split skipped invalid max reprojection gate={split_max_reprojection_error}"
            );
        }
        return None;
    }
    let mut split_config = config.clone();
    split_config.max_reprojection_error_px = split_max_reprojection_error;
    let merge_max_reprojection_error =
        pose_guided_merge_reprojection_gate(config, split_max_reprojection_error);
    if config.pose_guided_track_merging && merge_max_reprojection_error.is_none() {
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: pose-guided track split skipped invalid merge reprojection gate");
        }
        return None;
    }
    let mut merge_config = split_config.clone();
    if let Some(gate) = merge_max_reprojection_error {
        merge_config.max_reprojection_error_px = gate;
    }

    #[derive(Debug)]
    struct InputComponent {
        observations: Vec<TrackObservation>,
        old_point: Option<Point3<f64>>,
        was_conflicting: bool,
    }

    let mut components = Vec::with_capacity(tracks.len() + conflicting_components.len());
    for (track, point) in tracks.iter().zip(old_points.iter()) {
        components.push(InputComponent {
            observations: track.clone(),
            old_point: *point,
            was_conflicting: false,
        });
    }
    components.extend(
        conflicting_components
            .iter()
            .map(|component| InputComponent {
                observations: component.clone(),
                old_point: None,
                was_conflicting: true,
            }),
    );
    if components.is_empty() {
        return Some(PoseGuidedTrackSplitOutput::default());
    }

    let original_component_count = components.len();
    let original_observations = components
        .iter()
        .map(|component| component.observations.clone())
        .collect::<Vec<_>>();
    let Some((_, original_edges_by_component, _)) =
        build_pose_guided_component_graph(&original_observations, pairwise)
    else {
        // The legacy builder should produce disjoint components.  If a future
        // caller violates that invariant, leaving the original model untouched
        // is safer than assigning an observation to two new landmarks.
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: pose-guided track split skipped overlapping observation");
        }
        return None;
    };

    let mut bridge_cut_sizes = Vec::new();
    let mut bridge_cut_components = 0usize;
    if config.pose_guided_bridge_cuts {
        let mut refined_components = Vec::with_capacity(components.len());
        for (component_id, component) in components.into_iter().enumerate() {
            let bridge_cut = pose_guided_bridge_cut_component(
                camera,
                features,
                poses,
                &component.observations,
                &original_edges_by_component[component_id],
                &split_config,
            );
            if bridge_cut.cut_edges.is_empty() {
                refined_components.push(component);
                continue;
            }
            bridge_cut_components += 1;
            bridge_cut_sizes.extend(bridge_cut.cut_sizes);
            for observations in bridge_cut.components {
                refined_components.push(InputComponent {
                    observations,
                    // A cut invalidates the old point as a candidate for the
                    // new sides; each side must be posed/triangulated anew.
                    old_point: None,
                    was_conflicting: component.was_conflicting,
                });
            }
        }
        components = refined_components;
    }

    let component_observations = components
        .iter()
        .map(|component| component.observations.clone())
        .collect::<Vec<_>>();
    let Some((_, edges_by_component, adjacency_by_component)) =
        build_pose_guided_component_graph(&component_observations, pairwise)
    else {
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: pose-guided track split skipped overlapping cut side");
        }
        return None;
    };

    let mut output = PoseGuidedTrackSplitOutput {
        stats: PoseGuidedTrackSplitStats {
            input_components: original_component_count,
            bridge_cuts: bridge_cut_sizes.len(),
            bridge_cut_components,
            bridge_cut_sizes,
            ..PoseGuidedTrackSplitStats::default()
        },
        ..PoseGuidedTrackSplitOutput::default()
    };
    let minimum_support = config.min_track_length.max(2);
    let max_hypotheses = config.conflict_recovery_max_hypotheses;

    for (component_id, component) in components.iter().enumerate() {
        let mut unique_images = HashSet::new();
        let conflict_free = component
            .observations
            .iter()
            .all(|&(image, _)| unique_images.insert(image));

        // Preserve a clean, already-valid component exactly.  This is the
        // important non-regression path: merely enabling the diagnostic does
        // not perturb a track whose current point explains all observations.
        if conflict_free
            && component.observations.len() >= minimum_support
            && component
                .old_point
                .is_some_and(|point| point.coords.iter().all(|value| value.is_finite()))
            && component.observations.iter().all(|&(image, keypoint)| {
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint))
                    .and_then(|pixel| {
                        poses.get(image).and_then(Option::as_ref).and_then(|pose| {
                            reprojection_error_px(
                                camera,
                                pose,
                                &component.old_point.unwrap(),
                                pixel,
                            )
                        })
                    })
                    .is_some_and(|error| error <= split_max_reprojection_error)
            })
        {
            let mut preserved = component.observations.clone();
            preserved.sort_unstable();
            output.tracks.push(preserved);
            output.points.push(component.old_point);
            output.stats.preserved_components += 1;
            output.stats.emitted_tracks += 1;
            output.stats.assigned_observations += component.observations.len();
            continue;
        }

        let mut remaining: HashSet<TrackObservation> =
            component.observations.iter().copied().collect();
        let mut emitted_from_component = 0usize;
        while remaining.len() >= minimum_support {
            let mut ranked_edges = Vec::<(f64, TrackEdge)>::new();
            for &edge in &edges_by_component[component_id] {
                let (left, right) = edge;
                if !remaining.contains(&left) || !remaining.contains(&right) {
                    continue;
                }
                let (Some(pose_a), Some(pose_b), Some(pixel_a), Some(pixel_b)) = (
                    poses.get(left.0).and_then(Option::as_ref),
                    poses.get(right.0).and_then(Option::as_ref),
                    features
                        .get(left.0)
                        .and_then(|set| set.keypoints.get(left.1))
                        .copied(),
                    features
                        .get(right.0)
                        .and_then(|set| set.keypoints.get(right.1))
                        .copied(),
                ) else {
                    continue;
                };
                let Some(normalized_a) = camera.normalize_pixel(&pixel_a) else {
                    continue;
                };
                let Some(normalized_b) = camera.normalize_pixel(&pixel_b) else {
                    continue;
                };
                let bearing_a = pose_a.camera_to_world().rotation
                    * Vector3::new(normalized_a.x, normalized_a.y, 1.0).normalize();
                let bearing_b = pose_b.camera_to_world().rotation
                    * Vector3::new(normalized_b.x, normalized_b.y, 1.0).normalize();
                let parallax = bearing_a.dot(&bearing_b).clamp(-1.0, 1.0).abs().acos();
                if parallax.is_finite() {
                    ranked_edges.push((parallax, edge));
                }
            }
            ranked_edges.sort_unstable_by(|left, right| {
                right
                    .0
                    .total_cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
            });

            let mut best: Option<PoseGuidedTrackCandidate> = None;
            for &(parallax, edge) in ranked_edges.iter().take(max_hypotheses) {
                output.stats.hypotheses_tested += 1;
                let (left, right) = edge;
                let (Some(pixel_a), Some(pixel_b)) = (
                    features
                        .get(left.0)
                        .and_then(|set| set.keypoints.get(left.1))
                        .copied(),
                    features
                        .get(right.0)
                        .and_then(|set| set.keypoints.get(right.1))
                        .copied(),
                ) else {
                    continue;
                };
                let anchor_observations = [(left.0, pixel_a), (right.0, pixel_b)];
                let Some(point) =
                    triangulate_track(camera, poses, &anchor_observations, &split_config)
                else {
                    continue;
                };
                let (mut selected, _) = if config.pose_guided_graph_support {
                    pose_guided_select_observations_with_graph_support(
                        camera,
                        features,
                        poses,
                        &remaining,
                        &point,
                        split_max_reprojection_error,
                        edge,
                        &adjacency_by_component[component_id],
                    )
                } else {
                    (
                        pose_guided_select_observations(
                            camera,
                            features,
                            poses,
                            &remaining,
                            &point,
                            split_max_reprojection_error,
                        ),
                        Vec::new(),
                    )
                };
                if selected.len() < minimum_support
                    || !selected.contains(&left)
                    || !selected.contains(&right)
                {
                    continue;
                }
                let refined = refine_pose_guided_point(camera, features, poses, &selected, point)
                    .unwrap_or(point);
                let (refined_selected, graph_support_counts) = if config.pose_guided_graph_support {
                    pose_guided_select_observations_with_graph_support(
                        camera,
                        features,
                        poses,
                        &remaining,
                        &refined,
                        split_max_reprojection_error,
                        edge,
                        &adjacency_by_component[component_id],
                    )
                } else {
                    (
                        pose_guided_select_observations(
                            camera,
                            features,
                            poses,
                            &remaining,
                            &refined,
                            split_max_reprojection_error,
                        ),
                        Vec::new(),
                    )
                };
                selected = refined_selected;
                if selected.len() < minimum_support
                    || !selected.contains(&left)
                    || !selected.contains(&right)
                {
                    continue;
                }
                let mut errors = selected
                    .iter()
                    .filter_map(|&(image, keypoint)| {
                        let pixel = features.get(image)?.keypoints.get(keypoint)?;
                        let pose = poses.get(image)?.as_ref()?;
                        reprojection_error_px(camera, pose, &refined, pixel)
                    })
                    .collect::<Vec<_>>();
                if errors.len() != selected.len() {
                    continue;
                }
                errors.sort_by(f64::total_cmp);
                let median_reprojection_px = errors[errors.len() / 2];
                let mean_reprojection_px = errors.iter().sum::<f64>() / errors.len() as f64;
                if !mean_reprojection_px.is_finite() {
                    continue;
                }
                let independent_supports = pose_guided_cross_image_support_count(
                    &selected,
                    &adjacency_by_component[component_id],
                );
                if config.pose_guided_graph_support
                    && selected.len() > 2
                    && independent_supports < 2
                {
                    continue;
                }
                let candidate = PoseGuidedTrackCandidate {
                    observations: selected,
                    point: refined,
                    median_reprojection_px,
                    mean_reprojection_px,
                    anchor: edge,
                    parallax_rad: parallax,
                    graph_support_counts,
                };
                let replace = best.as_ref().is_none_or(|current| {
                    candidate.observations.len() > current.observations.len()
                        || (candidate.observations.len() == current.observations.len()
                            && (candidate.median_reprojection_px < current.median_reprojection_px
                                || (candidate.median_reprojection_px
                                    == current.median_reprojection_px
                                    && (candidate.mean_reprojection_px
                                        < current.mean_reprojection_px
                                        || (candidate.mean_reprojection_px
                                            == current.mean_reprojection_px
                                            && (candidate.parallax_rad > current.parallax_rad
                                                || (candidate.parallax_rad
                                                    == current.parallax_rad
                                                    && candidate.anchor < current.anchor)))))))
                });
                if replace {
                    best = Some(candidate);
                }
            }

            let Some(candidate) = best else { break };
            if candidate.observations.len() < minimum_support {
                break;
            }
            for observation in &candidate.observations {
                remaining.remove(observation);
            }
            if config.pose_guided_graph_support {
                for support in &candidate.graph_support_counts {
                    let bucket = (*support).min(output.stats.graph_support_histogram.len() - 1);
                    output.stats.graph_support_histogram[bucket] += 1;
                }
                if candidate.observations.len() > 2 {
                    output.stats.graph_supported_tracks += 1;
                } else {
                    output.stats.graph_length_two_tracks += 1;
                }
            }
            output.tracks.push(candidate.observations);
            output.points.push(Some(candidate.point));
            output.stats.emitted_tracks += 1;
            output.stats.assigned_observations += output.tracks.last().unwrap().len();
            emitted_from_component += 1;
        }

        output.stats.discarded_observations += remaining.len();
        if emitted_from_component > 0 {
            output.stats.split_components += 1;
        } else if component.was_conflicting {
            // A conflicting component without a valid posed hypothesis is
            // intentionally omitted; accepting its old transitive closure
            // would recreate the exact same same-image conflict.
            output.stats.split_components += 1;
        }
    }

    let mut paired = output
        .tracks
        .into_iter()
        .zip(output.points)
        .collect::<Vec<_>>();
    for (track, _) in &mut paired {
        track.sort_unstable();
    }
    paired.sort_unstable_by(|left, right| left.0.cmp(&right.0));
    output.tracks = paired.iter().map(|(track, _)| track.clone()).collect();
    output.points = paired.into_iter().map(|(_, point)| point).collect();
    if config.pose_guided_track_merging {
        let source_tracks = output.tracks.clone();
        let source_points = output.points.clone();
        let (merged_tracks, merged_points, merges, candidates_tested) = pose_guided_merge_tracks(
            camera,
            features,
            pairwise,
            poses,
            &output.tracks,
            &output.points,
            &merge_config,
        );
        output.merge_restorations =
            pose_guided_merge_restorations(&source_tracks, &source_points, &merged_tracks);
        output.tracks = merged_tracks;
        output.points = merged_points;
        output.stats.merged_tracks = merges;
        output.stats.merge_candidates_tested = candidates_tested;
    }
    Some(output)
}

fn pose_guided_split_candidate_gate(
    candidate_support: usize,
    support_floor: usize,
    candidate_mean: f64,
    split_max_reprojection_error: f64,
) -> bool {
    candidate_support >= support_floor
        && candidate_mean.is_finite()
        && split_max_reprojection_error.is_finite()
        && split_max_reprojection_error > 0.0
        && candidate_mean <= split_max_reprojection_error
}

/// Apply the GT-independent acceptance guard shared by every outer split pass.
/// The first pass preserves the historical single-pass behavior: its BA must
/// lower the candidate partition's own mean error.  A later pass must also
/// strictly lower the already accepted model's mean, which provides a
/// deterministic early-stop condition for repeated rebuilding from the same
/// source components.
#[allow(clippy::too_many_arguments)]
fn pose_guided_split_candidate_accepts(
    iteration: usize,
    registered_before: usize,
    registered_after: usize,
    support_floor: usize,
    candidate_support: usize,
    after_support: usize,
    mean_before: f64,
    candidate_mean: f64,
    after_mean: f64,
    split_max_reprojection_error: f64,
) -> bool {
    pose_guided_split_candidate_gate(
        candidate_support,
        support_floor,
        candidate_mean,
        split_max_reprojection_error,
    ) && registered_after >= registered_before
        && after_support >= support_floor
        && after_mean.is_finite()
        && after_mean <= candidate_mean + 1.0e-9
        && (iteration == 0 || after_mean + 1.0e-9 < mean_before)
}

/// Select the best currently posed observation for every image that explains
/// a candidate point within the existing pixel gate.  The input set is
/// converted to a sorted vector before traversal so HashSet iteration cannot
/// affect either the selected topology or its tie-breaks.
fn pose_guided_select_observations(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    remaining: &HashSet<TrackObservation>,
    point: &Point3<f64>,
    max_error: f64,
) -> Vec<TrackObservation> {
    let mut observations = remaining.iter().copied().collect::<Vec<_>>();
    observations.sort_unstable();
    let mut best_by_image = HashMap::<usize, (usize, f64)>::new();
    for (image, keypoint) in observations {
        let (Some(pose), Some(pixel)) = (
            poses.get(image).and_then(Option::as_ref),
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint)),
        ) else {
            continue;
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            continue;
        };
        if !error.is_finite() || error > max_error {
            continue;
        }
        let entry = best_by_image.entry(image).or_insert((keypoint, error));
        if error < entry.1 || (error == entry.1 && keypoint < entry.0) {
            *entry = (keypoint, error);
        }
    }
    let mut selected = best_by_image
        .into_iter()
        .map(|(image, (keypoint, _))| (image, keypoint))
        .collect::<Vec<_>>();
    selected.sort_unstable();
    selected
}

/// Select a posed hypothesis with an explicit verified-graph support rule.
/// The two anchor observations are admitted from one verified edge.  Every
/// later observation must both reproject into the current point and have
/// direct verified edges to at least two distinct observations already in the
/// hypothesis.  The strongest support is admitted first, with reprojection
/// error and physical observation key as deterministic tie-breaks; newly
/// admitted observations can unlock the next round.
#[allow(clippy::too_many_arguments)]
fn pose_guided_select_observations_with_graph_support(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    remaining: &HashSet<TrackObservation>,
    point: &Point3<f64>,
    max_error: f64,
    anchor: TrackEdge,
    adjacency: &HashMap<TrackObservation, Vec<TrackObservation>>,
) -> (Vec<TrackObservation>, Vec<usize>) {
    if anchor.0 == anchor.1
        || anchor.0 .0 == anchor.1 .0
        || !remaining.contains(&anchor.0)
        || !remaining.contains(&anchor.1)
    {
        return (Vec::new(), Vec::new());
    }
    for &(image, keypoint) in &[anchor.0, anchor.1] {
        let (Some(pose), Some(pixel)) = (
            poses.get(image).and_then(Option::as_ref),
            features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint)),
        ) else {
            return (Vec::new(), Vec::new());
        };
        let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
            return (Vec::new(), Vec::new());
        };
        if !error.is_finite() || error > max_error {
            return (Vec::new(), Vec::new());
        }
    }

    let mut selected = vec![anchor.0, anchor.1];
    selected.sort_unstable();
    let mut selected_set = selected.iter().copied().collect::<HashSet<_>>();
    let mut selected_images = selected
        .iter()
        .map(|&(image, _)| image)
        .collect::<HashSet<_>>();
    let mut support_counts = Vec::new();

    loop {
        let mut best: Option<(usize, f64, TrackObservation)> = None;
        let mut observations = remaining.iter().copied().collect::<Vec<_>>();
        observations.sort_unstable();
        for observation @ (image, keypoint) in observations {
            if selected_images.contains(&image) {
                continue;
            }
            let support_images = adjacency
                .get(&observation)
                .into_iter()
                .flat_map(|neighbours| neighbours.iter())
                .filter(|neighbour| selected_set.contains(neighbour))
                .map(|&(support_image, _)| support_image)
                .collect::<HashSet<_>>();
            let support = support_images.len();
            if support < 2 {
                continue;
            }
            let (Some(pose), Some(pixel)) = (
                poses.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|set| set.keypoints.get(keypoint)),
            ) else {
                continue;
            };
            let Some(error) = reprojection_error_px(camera, pose, point, pixel) else {
                continue;
            };
            if !error.is_finite() || error > max_error {
                continue;
            }
            let replace = best.is_none_or(|(best_support, best_error, best_observation)| {
                support > best_support
                    || (support == best_support
                        && (error < best_error
                            || (error == best_error && observation < best_observation)))
            });
            if replace {
                best = Some((support, error, observation));
            }
        }
        let Some((support, _, observation @ (image, _))) = best else {
            break;
        };
        selected.push(observation);
        selected.sort_unstable();
        selected_set.insert(observation);
        selected_images.insert(image);
        support_counts.push(support);
    }
    (selected, support_counts)
}

/// Count independent cross-image correspondence supports inside one posed
/// hypothesis.  An image pair contributes once regardless of how many
/// feature rows happen to connect it; same-image edges are never counted.
fn pose_guided_cross_image_support_count(
    observations: &[TrackObservation],
    adjacency: &HashMap<TrackObservation, Vec<TrackObservation>>,
) -> usize {
    let selected = observations.iter().copied().collect::<HashSet<_>>();
    let mut image_pairs = BTreeSet::new();
    for &(image, keypoint) in observations {
        let observation = (image, keypoint);
        for &(other_image, other_keypoint) in adjacency
            .get(&observation)
            .into_iter()
            .flat_map(|neighbours| neighbours.iter())
        {
            if selected.contains(&(other_image, other_keypoint)) && image != other_image {
                image_pairs.insert((image.min(other_image), image.max(other_image)));
            }
        }
    }
    image_pairs.len()
}

/// Locally refine one pose-guided point while keeping all camera poses fixed.
/// The same projection Jacobian used by the BA conditioning diagnostics is
/// used here, with monotone squared-reprojection acceptance and a tiny
/// diagonal damping term for nearly collinear rays.
fn refine_pose_guided_point(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    observations: &[TrackObservation],
    initial: Point3<f64>,
) -> Option<Point3<f64>> {
    if !initial.coords.iter().all(|value| value.is_finite()) {
        return None;
    }
    let point_cost = |point: &Point3<f64>| -> Option<f64> {
        let mut cost = 0.0;
        let mut count = 0usize;
        for &(image, keypoint) in observations {
            let pose = poses.get(image)?.as_ref()?;
            let pixel = features.get(image)?.keypoints.get(keypoint)?;
            let projected = camera.project(&pose.transform_world_point(point))?;
            let residual = projected - *pixel;
            if !residual.iter().all(|value| value.is_finite()) {
                return None;
            }
            cost += residual.norm_squared();
            count += 1;
        }
        (count >= 2 && cost.is_finite()).then_some(cost)
    };
    let mut point = initial;
    let mut cost = point_cost(&point)?;
    for _ in 0..4 {
        let mut hessian = Matrix3::<f64>::zeros();
        let mut gradient = Vector3::<f64>::zeros();
        for &(image, keypoint) in observations {
            let pose = poses.get(image)?.as_ref()?;
            let pixel = features.get(image)?.keypoints.get(keypoint)?;
            let point_camera = pose.transform_world_point(&point);
            let projected = camera.project(&point_camera)?;
            let residual = projected - *pixel;
            let projection_jacobian = ba_point_projection_jacobian(camera, &point_camera)?;
            let rotation = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let jacobian = projection_jacobian * rotation;
            hessian += jacobian.transpose() * jacobian;
            gradient += jacobian.transpose() * residual;
        }
        let damping = hessian
            .diagonal()
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0_f64, f64::max)
            .max(1.0)
            * 1.0e-8;
        let system = hessian + Matrix3::identity() * damping;
        let delta = system.lu().solve(&(-gradient))?;
        if !delta.iter().all(|value| value.is_finite()) || delta.norm() < 1.0e-10 {
            break;
        }
        let candidate = Point3::from(point.coords + delta);
        let candidate_cost = point_cost(&candidate)?;
        if candidate_cost + 1.0e-12 < cost {
            point = candidate;
            cost = candidate_cost;
        } else {
            break;
        }
    }
    Some(point)
}

#[derive(Debug, Clone)]
struct RecoveredConflictTrack {
    observations: Vec<(usize, usize)>,
    point: Point3<f64>,
    registered_observations: usize,
    mean_reprojection_px: f64,
}

/// Split dropped union-find conflict components against an already-posed model.
///
/// This deliberately does not trust descriptor distance or image-pair support
/// as a global ordering (both were catastrophic on MH_03). A verified edge is
/// only an anchor proposal. The resulting 3D hypothesis must explain a unique
/// observation in at least three registered images, and those selected
/// observations must contain a cycle in the verified correspondence graph.
fn recover_conflict_tracks_geometry(
    camera: &Camera,
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
    conflicting_components: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    config: &IncrementalSfmConfig,
) -> Vec<RecoveredConflictTrack> {
    if conflicting_components.is_empty() || config.conflict_recovery_max_hypotheses == 0 {
        return Vec::new();
    }

    let mut component_of = HashMap::new();
    for (component_id, component) in conflicting_components.iter().enumerate() {
        for &observation in component {
            component_of.insert(observation, component_id);
        }
    }

    type Observation = (usize, usize);
    let mut edges_by_component: Vec<Vec<(Observation, Observation)>> =
        vec![Vec::new(); conflicting_components.len()];
    for pair in pairwise {
        for &(kp_i, kp_j) in &pair.matches {
            let a = (pair.image_i, kp_i);
            let b = (pair.image_j, kp_j);
            let Some(&component_id) = component_of.get(&a) else {
                continue;
            };
            if component_of.get(&b) != Some(&component_id) {
                continue;
            }
            let edge = if a <= b { (a, b) } else { (b, a) };
            edges_by_component[component_id].push(edge);
        }
    }
    for edges in &mut edges_by_component {
        edges.sort_unstable();
        edges.dedup();
    }

    let mut triangulation_config = config.clone();
    triangulation_config.max_reprojection_error_px =
        config.conflict_recovery_max_reprojection_error_px;
    triangulation_config.low_parallax_min_observations = None;
    let min_views = config.conflict_recovery_min_views.max(3);
    let observation_pixel = |&(image, kp): &Observation| {
        features
            .get(image)
            .and_then(|feature_set| feature_set.keypoints.get(kp))
            .copied()
    };

    let mut recovered = Vec::new();
    for (component, edges) in conflicting_components.iter().zip(edges_by_component.iter()) {
        let mut adjacency: HashMap<Observation, Vec<Observation>> = HashMap::new();
        for &(a, b) in edges {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
        for neighbours in adjacency.values_mut() {
            neighbours.sort_unstable();
            neighbours.dedup();
        }
        let mut ranked_anchors = Vec::new();
        for &(a, b) in edges {
            let (Some(pose_a), Some(pose_b), Some(px_a), Some(px_b)) = (
                poses.get(a.0).and_then(Option::as_ref),
                poses.get(b.0).and_then(Option::as_ref),
                observation_pixel(&a),
                observation_pixel(&b),
            ) else {
                continue;
            };
            let Some(n_a) = camera.normalize_pixel(&px_a) else {
                continue;
            };
            let Some(n_b) = camera.normalize_pixel(&px_b) else {
                continue;
            };
            let ray_a =
                pose_a.camera_to_world().rotation * Vector3::new(n_a.x, n_a.y, 1.0).normalize();
            let ray_b =
                pose_b.camera_to_world().rotation * Vector3::new(n_b.x, n_b.y, 1.0).normalize();
            let angle = ray_a.dot(&ray_b).clamp(-1.0, 1.0).abs().acos();
            if angle.is_finite() {
                ranked_anchors.push((angle, a, b, px_a, px_b));
            }
        }
        ranked_anchors.sort_unstable_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
                .then_with(|| left.2.cmp(&right.2))
        });

        let mut best: Option<RecoveredConflictTrack> = None;
        for &(_angle, anchor_a, anchor_b, px_a, px_b) in ranked_anchors
            .iter()
            .take(config.conflict_recovery_max_hypotheses)
        {
            let anchor_observations = [(anchor_a.0, px_a), (anchor_b.0, px_b)];
            let Some(point) =
                triangulate_track(camera, poses, &anchor_observations, &triangulation_config)
            else {
                continue;
            };

            // First retain every registered observation consistent with the 3D
            // hypothesis. A component can contain several keypoints from one
            // image, so keep only that image's lowest-residual observation.
            let mut valid_errors: HashMap<Observation, f64> = HashMap::new();
            for &observation in component {
                let Some(pose) = poses.get(observation.0).and_then(Option::as_ref) else {
                    continue;
                };
                let Some(pixel) = observation_pixel(&observation) else {
                    continue;
                };
                let Some(error) = reprojection_error_px(camera, pose, &point, &pixel) else {
                    continue;
                };
                if error <= config.conflict_recovery_max_reprojection_error_px {
                    valid_errors.insert(observation, error);
                }
            }
            if !valid_errors.contains_key(&anchor_a) || !valid_errors.contains_key(&anchor_b) {
                continue;
            }

            // Restrict evidence to the verified-edge component containing the
            // anchor, then enforce one observation per image.
            let mut reachable = HashSet::from([anchor_a]);
            let mut frontier = vec![anchor_a];
            while let Some(node) = frontier.pop() {
                for &neighbour in adjacency.get(&node).into_iter().flatten() {
                    if valid_errors.contains_key(&neighbour) && reachable.insert(neighbour) {
                        frontier.push(neighbour);
                    }
                }
            }
            let mut best_by_image: HashMap<usize, (usize, f64)> = HashMap::new();
            for observation in reachable {
                let error = valid_errors[&observation];
                let entry = best_by_image
                    .entry(observation.0)
                    .or_insert((observation.1, error));
                if error < entry.1 || (error == entry.1 && observation.1 < entry.0) {
                    *entry = (observation.1, error);
                }
            }
            let mut selected: Vec<Observation> = best_by_image
                .iter()
                .map(|(&image, &(kp, _))| (image, kp))
                .collect();
            selected.sort_unstable();
            if selected.len() < min_views {
                continue;
            }
            let selected_set: HashSet<_> = selected.iter().copied().collect();
            let cycle_edges = edges
                .iter()
                .filter(|(a, b)| selected_set.contains(a) && selected_set.contains(b))
                .count();
            // A connected N-view tree has N-1 edges. Requiring N edges means
            // at least one independent cycle supports the hypothesis.
            if cycle_edges < selected.len() {
                continue;
            }
            let mean_reprojection_px = selected
                .iter()
                .map(|observation| valid_errors[observation])
                .sum::<f64>()
                / selected.len() as f64;
            if mean_reprojection_px > config.conflict_recovery_max_mean_reprojection_px {
                continue;
            }

            let registered_observations = selected.len();
            // An unregistered observation cannot be reprojection-checked yet.
            // Keep one only when it has at least two verified edges into the
            // accepted registered cycle; PnP RANSAC remains the final guard.
            let mut unregistered_support: HashMap<Observation, usize> = HashMap::new();
            for &(edge_a, edge_b) in edges {
                for (candidate, supported) in [(edge_a, edge_b), (edge_b, edge_a)] {
                    if poses.get(candidate.0).is_some_and(|pose| pose.is_none())
                        && selected_set.contains(&supported)
                    {
                        *unregistered_support.entry(candidate).or_insert(0) += 1;
                    }
                }
            }
            let mut inferred_by_image: HashMap<usize, (usize, usize)> = HashMap::new();
            for (observation, support) in unregistered_support {
                if support < 2 {
                    continue;
                }
                let entry = inferred_by_image
                    .entry(observation.0)
                    .or_insert((observation.1, support));
                if support > entry.1 || (support == entry.1 && observation.1 < entry.0) {
                    *entry = (observation.1, support);
                }
            }
            selected.extend(
                inferred_by_image
                    .into_iter()
                    .map(|(image, (kp, _))| (image, kp)),
            );
            selected.sort_unstable();

            let candidate = RecoveredConflictTrack {
                observations: selected,
                point,
                registered_observations,
                mean_reprojection_px,
            };
            let replace = best.as_ref().is_none_or(|current| {
                candidate.registered_observations > current.registered_observations
                    || (candidate.registered_observations == current.registered_observations
                        && (candidate.observations.len() > current.observations.len()
                            || (candidate.observations.len() == current.observations.len()
                                && candidate.mean_reprojection_px < current.mean_reprojection_px)))
            });
            if replace {
                best = Some(candidate);
            }
        }
        if let Some(track) = best {
            recovered.push(track);
        }
    }
    recovered
}

fn mean_reprojection_for_track_range(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
    start: usize,
    end: usize,
) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (track_id, track) in tracks.iter().enumerate().take(end).skip(start) {
        let Some(point) = track_point.get(track_id).and_then(|point| *point) else {
            continue;
        };
        for &(image, kp) in track {
            let (Some(pose), Some(pixel)) = (
                poses.get(image).and_then(Option::as_ref),
                features
                    .get(image)
                    .and_then(|feature_set| feature_set.keypoints.get(kp)),
            ) else {
                continue;
            };
            if let Some(error) = reprojection_error_px(camera, pose, &point, pixel) {
                sum += error;
                count += 1;
            }
        }
    }
    if count == 0 {
        f64::INFINITY
    } else {
        sum / count as f64
    }
}

#[allow(clippy::too_many_arguments)]
/// Among unregistered images still under the per-image trial cap, choose the one
/// observing the most triangulated tracks, returning it with its 2D-3D
/// correspondences.
fn select_next_image(
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

/// A provisional pose proposed by the opt-in sequence fallback.  The pose is
/// deliberately kept separate from the ordinary PnP report: it is admitted
/// only after its consecutive essential edge has passed the same triangulation
/// and reprojection checks used by normal growth.
#[derive(Debug, Clone)]
struct SequenceRelativePoseProposal {
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
    Some(if values.len() % 2 == 0 {
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
fn robust_recent_consecutive_step_scale(
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
struct SequenceProjectedScaleDiagnostic {
    projected_scale: f64,
    recent_median: f64,
    mad: f64,
    sample_count: usize,
    predicted_velocity: Vector3<f64>,
}

/// Apply the intentionally broad safety bounds for the relaxed projected
/// scale policy.  The local MAD fence belongs to the strict policy; this
/// helper only prevents a non-finite, reversed, or catastrophic scale while
/// allowing a genuine turn to change the projected step length.
fn relaxed_projected_scale_is_valid(projected_scale: f64, recent_median: f64) -> bool {
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
fn carried_sequence_scale_or_projection(
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
fn rescale_sequence_pose_translation(
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
fn next_sequence_fallback_carry_state(
    fallback_image: usize,
    fallback_scale: f64,
    resumed_post_registered: usize,
) -> Option<(usize, f64)> {
    (resumed_post_registered == 0).then_some((fallback_image, fallback_scale))
}

/// Compute the un-gated diagnostics for a sequence constant-velocity
/// projection.  Keeping the raw positive/negative and out-of-fence result
/// available lets the opt-in fallback explain why a candidate was rejected.
fn projected_recent_consecutive_step_scale_diagnostic(
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
fn projected_recent_consecutive_step_scale(
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
fn compose_sequence_relative_pose(
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
fn sequence_triangulation_admission_ok(
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
fn sequence_relative_pose_fallback_with_overrides(
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
fn commit_sequence_relative_pose(
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
fn log_registration_track_provenance(
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
fn pnp_track_ids(
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
fn log_pnp_geometry_diagnostic(
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

fn next_image_rank(
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
struct NextImageAutoMetrics {
    registered_images: usize,
    valid_observations: usize,
    tracks: usize,
    mean_reprojection_px: f64,
}

fn next_image_auto_metrics(result: &IncrementalSfmResult) -> NextImageAutoMetrics {
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
fn next_image_auto_count_candidate_is_needed(
    registered_images: usize,
    total_images: usize,
) -> bool {
    registered_images < total_images
}

/// Auto's completion pass is considered only for a genuinely incomplete
/// model.  A complete primary candidate is returned without a second mapper
/// run, preserving both its bytes and its runtime.
fn next_image_auto_post_candidate_is_needed(registered_images: usize, total_images: usize) -> bool {
    next_image_auto_count_candidate_is_needed(registered_images, total_images)
}

/// Post-refinement completion is a registration-only fallback.  A candidate
/// is adopted only when it strictly adds registered images without worsening
/// the finite mean reprojection error.  Equal-support results retain the
/// untouched pre-post candidate, including its tracks, poses, and BA state.
fn next_image_auto_post_candidate_is_better(
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
fn next_image_auto_candidate_is_better(
    candidate: &IncrementalSfmResult,
    incumbent: &IncrementalSfmResult,
) -> bool {
    next_image_auto_metrics_are_better(
        next_image_auto_metrics(candidate),
        next_image_auto_metrics(incumbent),
    )
}

fn next_image_auto_metrics_are_better(
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
fn visibility_pyramid_score(
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
fn track_geometry_observation_weights(
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

/// Global BA over all registered poses + triangulated landmarks. Seed pose
/// (the lowest-index registered image) is fixed for gauge. Writes refined
/// poses and points back in place. When `refine_intrinsics` is set, the BA also
/// refines the pinhole intrinsics (alternating) and the refined camera is
/// returned as `Some` (the caller propagates it); otherwise the second tuple
/// element is `None` and the camera is untouched.
pub(crate) fn run_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
) -> Result<(BaResult, Option<Camera>), BaError> {
    run_bundle_adjustment_impl(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        refine_intrinsics,
        false,
    )
}

/// Summary of the optional camera-fixed landmark warm start that precedes a
/// joint global/periodic BA. All fields are scalar so this remains cheap to
/// report from the registration loop.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct LandmarkBaWarmStartStats {
    pub(crate) requested_iterations: usize,
    pub(crate) attempted: bool,
    pub(crate) accepted: bool,
    pub(crate) points: usize,
    pub(crate) observations: usize,
    pub(crate) initial_cost: f64,
    pub(crate) final_cost: f64,
    pub(crate) solver_iterations: usize,
    pub(crate) accepted_steps: usize,
    pub(crate) rejected_steps: usize,
    pub(crate) converged: bool,
    pub(crate) max_displacement: f64,
    pub(crate) median_displacement: f64,
}

/// Optimize only the currently triangulated landmarks while keeping every
/// registered camera and the intrinsics fixed. This is deliberately separate
/// from [`run_bundle_adjustment_impl`]: it uses the same robust residuals and
/// solver, but never exposes a pose variable to the Schur system. A candidate
/// result is copied back only when the robust cost and every point are finite
/// and the cost is non-increasing; otherwise the caller's points are left
/// untouched and the subsequent ordinary joint BA still runs.
fn run_landmark_ba_warm_start(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> LandmarkBaWarmStartStats {
    let requested_iterations = config.landmark_ba_warm_start_iterations;
    let mut stats = LandmarkBaWarmStartStats {
        requested_iterations,
        attempted: false,
        accepted: false,
        points: 0,
        observations: 0,
        initial_cost: f64::NAN,
        final_cost: f64::NAN,
        solver_iterations: 0,
        accepted_steps: 0,
        rejected_steps: 0,
        converged: false,
        max_displacement: 0.0,
        median_displacement: 0.0,
    };
    if requested_iterations == 0 {
        return stats;
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    for (image, pose) in poses.iter().enumerate() {
        let Some(pose) = pose else { continue };
        ba.add_pose(image as u64, pose.clone());
        ba.fix_pose(image as u64);
    }

    // Keep the input order (track id, then observation order) exactly as the
    // ordinary BA builder. The point-only solve therefore has no alternate
    // ordering or matching policy hidden behind the opt-in switch.
    let mut point_ids = Vec::new();
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point.get(track_id).and_then(Option::as_ref) else {
            continue;
        };
        let mut observations = Vec::new();
        for &(image, keypoint) in track {
            if poses.get(image).and_then(Option::as_ref).is_none() {
                continue;
            }
            let Some(xy) = features
                .get(image)
                .and_then(|set| set.keypoints.get(keypoint))
            else {
                continue;
            };
            observations.push(BaObservation {
                keyframe_id: image as u64,
                landmark_id: track_id as u64,
                xy: *xy,
            });
        }
        if observations.len() < 2 {
            continue;
        }
        ba.add_landmark(track_id as u64, *point);
        for observation in observations {
            ba.add_observation(observation);
        }
        point_ids.push(track_id);
    }
    stats.points = point_ids.len();
    stats.observations = ba.observations.len();
    if point_ids.is_empty() || ba.poses.is_empty() || ba.observations.is_empty() {
        return stats;
    }
    stats.attempted = true;

    let mut warm_config = config.ba_config;
    warm_config.max_iterations = requested_iterations;
    warm_config.refine_intrinsics = false;
    let result = match ba.optimize(&warm_config) {
        Ok(result) => result,
        Err(error) => {
            if sfm_ba_debug_enabled() {
                eprintln!(
                    "sfm-debug-ba-warm-start: solver error={error:?}; keeping input landmarks"
                );
            }
            return stats;
        }
    };
    stats.initial_cost = result.initial_cost;
    stats.final_cost = result.final_cost;
    stats.solver_iterations = result.iterations.len();
    stats.accepted_steps = result
        .iterations
        .iter()
        .filter(|iteration| iteration.step_accepted)
        .count();
    stats.rejected_steps = result.iterations.len().saturating_sub(stats.accepted_steps);
    stats.converged = result.converged;

    let mut displacements = Vec::with_capacity(point_ids.len());
    let mut finite_points = true;
    for &track_id in &point_ids {
        let Some(before) = track_point.get(track_id).and_then(Option::as_ref) else {
            finite_points = false;
            break;
        };
        let Some(after) = ba.landmarks.get(&(track_id as u64)) else {
            finite_points = false;
            break;
        };
        if !after.coords.iter().all(|value| value.is_finite()) {
            finite_points = false;
            break;
        }
        let displacement = (after.coords - before.coords).norm();
        if !displacement.is_finite() {
            finite_points = false;
            break;
        }
        displacements.push(displacement);
    }
    if !displacements.is_empty() {
        stats.max_displacement = displacements.iter().copied().fold(0.0, f64::max);
        stats.median_displacement = sfm_oracle_median(&mut displacements);
    }
    stats.accepted = finite_points
        && stats.initial_cost.is_finite()
        && stats.final_cost.is_finite()
        && stats.final_cost <= stats.initial_cost;
    if stats.accepted {
        for &track_id in &point_ids {
            track_point[track_id] = ba.landmarks.get(&(track_id as u64)).copied();
        }
    }
    if sfm_ba_debug_enabled() {
        eprintln!(
            concat!(
                "sfm-debug-ba-warm-start: requested_iterations={} attempted={} accepted={} ",
                "points={} observations={} initial_cost={:.9e} final_cost={:.9e} ",
                "solver_iterations={} accepted_steps={} rejected_steps={} converged={} ",
                "point_delta_max/median=({:.3e},{:.3e})m"
            ),
            stats.requested_iterations,
            stats.attempted,
            stats.accepted,
            stats.points,
            stats.observations,
            stats.initial_cost,
            stats.final_cost,
            stats.solver_iterations,
            stats.accepted_steps,
            stats.rejected_steps,
            stats.converged,
            stats.max_displacement,
            stats.median_displacement,
        );
    }
    stats
}

/// Implementation shared by the historical BA path and the opt-in final
/// geometry-weighted solve. Keeping the switch here means all ordinary BA
/// callers continue to use the exact legacy `optimize` path.
#[allow(clippy::too_many_arguments)]
fn run_bundle_adjustment_impl(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
    geometry_weighted: bool,
) -> Result<(BaResult, Option<Camera>), BaError> {
    run_bundle_adjustment_impl_with_fixed_rotations(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        refine_intrinsics,
        geometry_weighted,
        None,
    )
}

/// Implementation shared by ordinary BA and the opt-in fixed-rotation
/// diagnostic.  The latter supplies image indices whose pose rotations are
/// constrained while translations/landmarks remain ordinary BA variables.
#[allow(clippy::too_many_arguments)]
fn run_bundle_adjustment_impl_with_fixed_rotations(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    refine_intrinsics: bool,
    geometry_weighted: bool,
    fixed_rotation_images: Option<&BTreeSet<usize>>,
) -> Result<(BaResult, Option<Camera>), BaError> {
    let timing_enabled = std::env::var_os("VISLOC_SFM_TIMING").is_some();
    let total_started = std::time::Instant::now();
    let oracle_poses_before = config.debug_oracle_poses.as_ref().map(|_| poses.to_vec());
    let ba_debug = sfm_ba_debug_enabled();
    let ba_step_debug = sfm_ba_step_debug_enabled();
    let support_before = ba_debug.then(|| {
        (
            poses.iter().filter(|pose| pose.is_some()).count(),
            track_point.iter().filter(|point| point.is_some()).count(),
            count_observations(tracks, poses, track_point),
        )
    });
    let poses_before = ba_debug.then(|| poses.to_vec());
    let points_before = ba_debug.then(|| track_point.to_vec());
    let registered_before = poses.iter().filter(|pose| pose.is_some()).count();
    let warm_start_started = std::time::Instant::now();
    let warm_start_stats = if config.landmark_ba_warm_start_iterations > 0
        && registered_before >= config.landmark_ba_warm_start_min_registered_images
    {
        Some(run_landmark_ba_warm_start(
            camera,
            features,
            tracks,
            config,
            poses,
            track_point,
        ))
    } else {
        if sfm_ba_debug_enabled() && config.landmark_ba_warm_start_iterations > 0 {
            eprintln!(
                "sfm-debug-ba-warm-start: skipped registered={} minimum_registered={}",
                registered_before, config.landmark_ba_warm_start_min_registered_images,
            );
        }
        None
    };
    let warm_start_seconds = warm_start_started.elapsed().as_secs_f64();
    let assembly_started = std::time::Instant::now();
    process_memory::log("ba-before-assembly");
    let mut ba = BundleAdjustment::new(camera.clone());
    let ba_config = BaConfig {
        refine_intrinsics,
        ..config.ba_config
    };

    for (image, pose) in poses.iter().enumerate() {
        if let Some(pose) = pose {
            ba.add_pose(image as u64, pose.clone());
            if fixed_rotation_images.is_some_and(|images| images.contains(&image)) {
                ba.fix_pose_rotation(image as u64);
            }
        }
    }

    // Gauge fixing. A monocular reconstruction (no stereo residual) has 7 gauge
    // freedoms: 6 for the rigid SE(3) frame plus **1 for global scale**. Fixing
    // a single pose pins only the 6 rigid DoF and leaves scale unconstrained, so
    // the BA's normal equations are singular along the scale direction. A single
    // solve from a perturbed state tolerates that (the damping holds the null
    // direction), but **re-optimising from an already-converged state lets the
    // scale drift and the reconstruction collapse**. Pin scale too by also
    // fixing the registered pose whose camera centre is farthest from the
    // anchor — the longest, best-conditioned baseline.
    let anchor = poses.iter().position(|p| p.is_some());
    let mut scale_anchor = None;
    if let Some(anchor) = anchor {
        ba.fix_pose(anchor as u64);
        let anchor_center = poses[anchor]
            .as_ref()
            .unwrap()
            .camera_to_world()
            .translation;
        let mut farthest = None;
        let mut best_d2 = 0.0;
        for (image, pose) in poses.iter().enumerate() {
            if image == anchor {
                continue;
            }
            if let Some(pose) = pose {
                let d2 = (pose.camera_to_world().translation - anchor_center).norm_squared();
                if d2 > best_d2 {
                    best_d2 = d2;
                    farthest = Some(image);
                }
            }
        }
        if let Some(scale_anchor_image) = farthest {
            ba.fix_pose(scale_anchor_image as u64);
            scale_anchor = Some(scale_anchor_image);
        }
    }

    let collect_landmark_geometry =
        config.freeze_ill_conditioned_landmarks || sfm_ba_landmark_debug_enabled();
    let mut landmark_diagnostics = Vec::new();
    let mut excluded_landmarks = 0usize;
    let mut excluded_observations = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        let mut obs = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: px,
                });
            }
        }
        if obs.len() >= 2 {
            let (geometry, excluded) = if collect_landmark_geometry {
                let geometry = ba_landmark_geometry(
                    camera,
                    features,
                    poses,
                    track,
                    &point,
                    &ba_config.robust_kernel,
                );
                let excluded = config.freeze_ill_conditioned_landmarks
                    && ba_landmark_should_exclude(
                        &geometry,
                        config.min_triangulation_angle_deg,
                        config.max_reprojection_error_px,
                    );
                (Some(geometry), excluded)
            } else {
                (None, false)
            };
            if excluded {
                // A fixed point with a large pre-BA residual would still pull
                // the camera Schur system through its observation rows. Such a
                // point is not a trustworthy camera constraint, so this
                // conditioning mode drops its rows for this solve instead of
                // retaining mathematically misleading residual influence.
                excluded_landmarks += 1;
                excluded_observations += obs.len();
            } else {
                ba.add_landmark(track_id as u64, point);
                for o in obs {
                    ba.add_observation(o);
                }
            }
            if collect_landmark_geometry {
                let geometry = geometry.expect("geometry collected for every BA landmark");
                landmark_diagnostics.push(LandmarkBaDiagnostic {
                    id: track_id as u64,
                    geometry,
                    displacement: 0.0,
                    excluded,
                });
            }
        }
    }

    if sfm_ba_jacobian_audit_enabled() {
        let audit = crate::bundle::audit_bundle_visual_jacobians(&ba, 64);
        eprintln!(
            "sfm-debug-ba-jacobian: observations_seen={} samples_audited={} invalid_samples={} normal={:?} far_depth={:?} low_parallax={:?} high_residual={:?}",
            audit.observations_seen,
            audit.samples_audited,
            audit.invalid_samples,
            audit.normal,
            audit.far_depth,
            audit.low_parallax,
            audit.high_residual,
        );
    }

    process_memory::log("ba-after-assembly");
    let initial_l2 = ba_debug.then(|| ba.robust_cost(&RobustKernel::None));
    let initial_robust = ba_debug.then(|| ba.robust_cost(&ba_config.robust_kernel));
    let observation_weights = if geometry_weighted && !refine_intrinsics {
        Some(track_geometry_observation_weights(
            poses,
            tracks,
            track_point,
            &ba.observations,
        ))
    } else {
        None
    };
    let pre_optimize_seconds = assembly_started.elapsed().as_secs_f64();
    let optimize_started = std::time::Instant::now();
    let result = if let Some(weights) = observation_weights.as_deref() {
        ba.optimize_with_observation_weights(&ba_config, weights)?
    } else {
        ba.optimize(&ba_config)?
    };
    let optimize_seconds = optimize_started.elapsed().as_secs_f64();
    process_memory::log("ba-after-optimize");

    if !landmark_diagnostics.is_empty() {
        for diagnostic in &mut landmark_diagnostics {
            let Some(before) = track_point
                .get(diagnostic.id as usize)
                .and_then(Option::as_ref)
            else {
                continue;
            };
            let Some(after) = ba.landmarks.get(&diagnostic.id) else {
                continue;
            };
            diagnostic.displacement = (after.coords - before.coords).norm();
        }
    }

    if ba_debug {
        let accepted = result
            .iterations
            .iter()
            .filter(|iteration| iteration.step_accepted)
            .count();
        let rejected = result.iterations.len().saturating_sub(accepted);
        let last = result.iterations.last();
        let final_l2 = ba.robust_cost(&RobustKernel::None);
        let robust_kernel_cost = ba.robust_cost(&ba_config.robust_kernel);
        let (pose_center_max, pose_center_median, pose_rotation_max, pose_rotation_median) =
            if let Some(before) = poses_before.as_ref() {
                let mut center_displacements = Vec::new();
                let mut rotation_displacements = Vec::new();
                for (image, old_pose) in before.iter().enumerate() {
                    let (Some(old_pose), Some(new_pose)) =
                        (old_pose.as_ref(), ba.poses.get(&(image as u64)))
                    else {
                        continue;
                    };
                    let center_delta = (new_pose.camera_center_world().coords
                        - old_pose.camera_center_world().coords)
                        .norm();
                    let rotation_delta = (old_pose.camera_to_world().rotation.inverse()
                        * new_pose.camera_to_world().rotation)
                        .angle()
                        .to_degrees();
                    if center_delta.is_finite() {
                        center_displacements.push(center_delta);
                    }
                    if rotation_delta.is_finite() {
                        rotation_displacements.push(rotation_delta);
                    }
                }
                (
                    center_displacements.iter().copied().fold(0.0, f64::max),
                    sfm_oracle_median(&mut center_displacements),
                    rotation_displacements.iter().copied().fold(0.0, f64::max),
                    sfm_oracle_median(&mut rotation_displacements),
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            };
        let (point_max, point_median) = if let Some(before) = points_before.as_ref() {
            let mut displacements = Vec::new();
            for (track_id, old_point) in before.iter().enumerate() {
                let (Some(old_point), Some(new_point)) =
                    (old_point.as_ref(), ba.landmarks.get(&(track_id as u64)))
                else {
                    continue;
                };
                let displacement = (new_point.coords - old_point.coords).norm();
                if displacement.is_finite() {
                    displacements.push(displacement);
                }
            }
            (
                displacements.iter().copied().fold(0.0, f64::max),
                sfm_oracle_median(&mut displacements),
            )
        } else {
            (f64::NAN, f64::NAN)
        };
        let (support_poses_before, support_tracks_before, support_observations_before) =
            support_before.unwrap_or((0, 0, 0));
        let support_poses_after = ba.poses.len();
        let support_tracks_after = ba.landmarks.len();
        let support_observations_after = ba.observations.len();
        eprintln!(
            concat!(
                "sfm-debug-ba: poses={} landmarks={} observations={} max_iterations={} ",
                "geometry_weighted={} ",
                "initial_lambda={:?} kernel={:?} iterations={} accepted={} rejected={} ",
                "converged={} initial_cost={:.9e} final_cost={:.9e} final_l2={:.9e} ",
                "last_step=({:.3e},{:.3e}) last_lambda={:.3e} ",
                "initial_robust={:.9e} initial_l2={:.9e} robust_cost={:.9e} ",
                "support_input=({},{},{}) support_ba=({},{},{}) pruning=none ",
                "conditioned_excluded_landmarks={} excluded_observations={} ",
                "landmark_warm_start=(attempted={},accepted={}) ",
                "pose_delta_max/median=({:.3e},{:.3e})m ",
                "pose_rot_delta_max/median=({:.3e},{:.3e})deg ",
                "point_delta_max/median=({:.3e},{:.3e})m ",
                "gauge_anchor={:?} scale_anchor={:?} ",
                "camera_before={:?} camera_after={:?}"
            ),
            ba.poses.len(),
            ba.landmarks.len(),
            ba.observations.len(),
            ba_config.max_iterations,
            observation_weights.is_some(),
            ba_config.initial_lambda,
            ba_config.robust_kernel,
            result.iterations.len(),
            accepted,
            rejected,
            result.converged,
            result.initial_cost,
            result.final_cost,
            final_l2,
            last.map_or(0.0, |iteration| iteration.max_pose_step),
            last.map_or(0.0, |iteration| iteration.max_landmark_step),
            last.map_or(0.0, |iteration| iteration.lambda),
            initial_robust.unwrap_or(f64::NAN),
            initial_l2.unwrap_or(f64::NAN),
            robust_kernel_cost,
            support_poses_before,
            support_tracks_before,
            support_observations_before,
            support_poses_after,
            support_tracks_after,
            support_observations_after,
            excluded_landmarks,
            excluded_observations,
            warm_start_stats.is_some_and(|stats| stats.attempted),
            warm_start_stats.is_some_and(|stats| stats.accepted),
            pose_center_max,
            pose_center_median,
            pose_rotation_max,
            pose_rotation_median,
            point_max,
            point_median,
            anchor,
            scale_anchor,
            camera.params,
            ba.camera.params,
        );
        if ba_step_debug {
            for iteration in &result.iterations {
                eprintln!(
                    concat!(
                        "sfm-debug-ba-step: iteration={} accepted={} ",
                        "cost={:.9e}->{:.9e} delta={:+.9e} lambda={:.3e} ",
                        "step=({:.3e},{:.3e})"
                    ),
                    iteration.iteration,
                    iteration.step_accepted,
                    iteration.cost_before,
                    iteration.cost_after,
                    iteration.cost_after - iteration.cost_before,
                    iteration.lambda,
                    iteration.max_pose_step,
                    iteration.max_landmark_step,
                );
            }
        }
        if sfm_ba_landmark_debug_enabled() {
            sfm_debug_ba_landmarks(&landmark_diagnostics, config.min_triangulation_angle_deg);
        }
    }

    let writeback_started = std::time::Instant::now();
    for (image, pose) in poses.iter_mut().enumerate() {
        if pose.is_some() {
            if let Some(refined) = ba.poses.get(&(image as u64)) {
                *pose = Some(refined.clone());
            }
        }
    }
    for (track_id, point) in track_point.iter_mut().enumerate() {
        if point.is_some() {
            if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
                *point = Some(*refined);
            }
        }
    }
    if timing_enabled {
        let accepted = result
            .iterations
            .iter()
            .filter(|iteration| iteration.step_accepted)
            .count();
        eprintln!(
            "sfm-timing-ba: registered={} landmarks={} observations={} warm_start={:.3}s assemble={:.3}s solve={:.3}s writeback={:.3}s total={:.3}s iterations={} accepted={}",
            ba.poses.len(),
            ba.landmarks.len(),
            ba.observations.len(),
            warm_start_seconds,
            pre_optimize_seconds,
            optimize_seconds,
            writeback_started.elapsed().as_secs_f64(),
            total_started.elapsed().as_secs_f64(),
            result.iterations.len(),
            accepted,
        );
    }
    sfm_debug_oracle_transition(
        &format!(
            "ba weighted={} refine_intrinsics={} poses={} observations={}",
            geometry_weighted,
            refine_intrinsics,
            ba.poses.len(),
            ba.observations.len(),
        ),
        oracle_poses_before.as_deref(),
        poses,
        config.debug_oracle_poses.as_deref(),
    );
    let refined_camera = refine_intrinsics.then(|| ba.camera.clone());
    Ok((result, refined_camera))
}

/// Run one ordinary fixed-support BA solve on an externally supplied pose
/// basin and an already assembled [`SfmTrack`] support.
///
/// This is intentionally a diagnostic API: it does not participate in the
/// incremental path, never changes track membership or observations, and
/// leaves the caller's track positions untouched when the solve fails. It is
/// useful for separating mapper-basin errors from BA errors, for example by
/// injecting a COLMAP sparse-model pose set while retaining our own tracks.
pub fn run_fixed_support_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &mut [SfmTrack],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
) -> Result<(BaResult, Option<Camera>), BaError> {
    let local_tracks: Vec<Vec<(usize, usize)>> = tracks
        .iter()
        .map(|track| {
            track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
                .collect()
        })
        .collect();
    let mut points: Vec<Option<Point3<f64>>> =
        tracks.iter().map(|track| Some(track.position)).collect();
    let points_before = points.clone();
    let result = run_bundle_adjustment_impl(
        camera,
        features,
        &local_tracks,
        config,
        poses,
        &mut points,
        config.refine_intrinsics,
        config.geometry_weighted_ba,
    );
    let (result, refined_camera) = match result {
        Ok(result) => result,
        Err(error) => {
            for (track, point) in tracks.iter_mut().zip(points_before) {
                track.position = point.expect("diagnostic track point is present");
            }
            return Err(error);
        }
    };
    for (track, point) in tracks.iter_mut().zip(points) {
        if let Some(point) = point {
            track.position = point;
        }
    }
    Ok((result, refined_camera))
}

/// Run one fixed-support BA solve while pinning the rotations supplied by
/// `fixed_rotations`.  The entries are index-aligned with `poses`; a `Some`
/// entry replaces only that pose's rotation before solving, while its current
/// translation is retained.  Translation and landmark variables remain free,
/// and the ordinary monocular gauge anchors are rebuilt by the same BA path.
///
/// This is an opt-in diagnostic API for separating rotation from translation
/// error.  It does not change track membership or observations and no caller
/// needs to provide entries for unregistered images.  An empty/all-`None`
/// vector is equivalent to [`run_fixed_support_bundle_adjustment`].
pub fn run_fixed_rotation_support_bundle_adjustment(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &mut [SfmTrack],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    fixed_rotations: &[Option<Pose>],
) -> Result<(BaResult, Option<Camera>), BaError> {
    if fixed_rotations.len() != poses.len() {
        return Err(BaError::InvalidFixedRotationCount {
            expected: poses.len(),
            actual: fixed_rotations.len(),
        });
    }
    let poses_before = poses.to_vec();
    let points_before: Vec<Point3<f64>> = tracks.iter().map(|track| track.position).collect();
    let mut fixed_rotation_images = BTreeSet::new();
    for (image, (pose, desired)) in poses.iter_mut().zip(fixed_rotations).enumerate() {
        let (Some(pose), Some(desired)) = (pose.as_mut(), desired.as_ref()) else {
            continue;
        };
        pose.world_to_camera.rotation = desired.world_to_camera.rotation;
        fixed_rotation_images.insert(image);
    }

    let local_tracks: Vec<Vec<(usize, usize)>> = tracks
        .iter()
        .map(|track| {
            track
                .observations
                .iter()
                .map(|&(image, keypoint, _)| (image, keypoint))
                .collect()
        })
        .collect();
    let mut points: Vec<Option<Point3<f64>>> =
        tracks.iter().map(|track| Some(track.position)).collect();
    let result = run_bundle_adjustment_impl_with_fixed_rotations(
        camera,
        features,
        &local_tracks,
        config,
        poses,
        &mut points,
        config.refine_intrinsics,
        config.geometry_weighted_ba,
        Some(&fixed_rotation_images),
    );
    let (result, refined_camera) = match result {
        Ok(result) => result,
        Err(error) => {
            poses.clone_from_slice(&poses_before);
            for (track, point) in tracks.iter_mut().zip(points_before) {
                track.position = point;
            }
            return Err(error);
        }
    };
    for (track, point) in tracks.iter_mut().zip(points) {
        if let Some(point) = point {
            track.position = point;
        }
    }
    Ok((result, refined_camera))
}

/// COLMAP `IncrementalMapper::AdjustLocalBundle`. After registering `new_image`,
/// bundle-adjust only it and its `local_ba_num_images` most-covisible registered
/// neighbours (sharing the most triangulated tracks) plus the points they see —
/// every *other* registered image that observes one of those points is added as a
/// **fixed** pose, so it constrains the local solve without being moved. This
/// keeps the freshly grown geometry tight after every step at a fraction of a
/// global solve's cost, the schedule that lets COLMAP hold sub-centimetre
/// accuracy as the reconstruction grows. Poses/points outside the variable set
/// are untouched.
fn adjust_local_bundle(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    new_image: usize,
) -> Result<(), BaError> {
    // Covisible registered images: how many triangulated tracks each shares with
    // the newly registered one.
    let mut covis: HashMap<usize, usize> = HashMap::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        if !track
            .iter()
            .any(|&(img, _)| img == new_image && poses[img].is_some())
        {
            continue;
        }
        for &(img, _) in track {
            if img != new_image && poses[img].is_some() {
                *covis.entry(img).or_insert(0) += 1;
            }
        }
    }
    let mut neighbours: Vec<(usize, usize)> = covis.into_iter().collect();
    // Most-covisible first; break ties by index for determinism.
    neighbours.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    let mut variable: HashSet<usize> = neighbours
        .into_iter()
        .take(config.local_ba_num_images)
        .map(|(img, _)| img)
        .collect();
    variable.insert(new_image);

    bundle_adjust_local(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        &variable,
    )
}

#[allow(clippy::too_many_arguments)]
/// With every camera and every pre-existing landmark fixed, refine only the
/// landmarks created by a tentative structure-less insertion. This is the
/// bounded local-submap solve used after projecting the new camera into the
/// independent relative-geometry feasible region.
fn refine_structureless_new_landmarks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    new_image: usize,
    preexisting_points: &[bool],
) -> Result<(), BaError> {
    let mut landmark_ids = Vec::new();
    let mut used_images = HashSet::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if preexisting_points.get(track_id).copied().unwrap_or(false)
            || track_point[track_id].is_none()
            || !track.iter().any(|&(image, kp)| {
                image == new_image
                    && poses[image].is_some()
                    && features[image].keypoints.get(kp).is_some()
            })
        {
            continue;
        }
        let observers: Vec<usize> = track
            .iter()
            .filter_map(|&(image, kp)| {
                (poses[image].is_some() && features[image].keypoints.get(kp).is_some())
                    .then_some(image)
            })
            .collect();
        if observers.len() < 2 {
            continue;
        }
        used_images.extend(observers);
        landmark_ids.push(track_id);
    }
    if landmark_ids.is_empty() {
        return Ok(());
    }

    let mut ba = BundleAdjustment::new(camera.clone());
    for image in used_images {
        ba.add_pose(image as u64, poses[image].clone().unwrap());
        ba.fix_pose(image as u64);
    }
    for &track_id in &landmark_ids {
        ba.add_landmark(track_id as u64, track_point[track_id].unwrap());
        for &(image, kp) in &tracks[track_id] {
            if !ba.poses.contains_key(&(image as u64)) {
                continue;
            }
            if let Some(pixel) = features[image].keypoints.get(kp).copied() {
                ba.add_observation(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: pixel,
                });
            }
        }
    }
    ba.optimize(&config.ba_config)?;
    for track_id in landmark_ids {
        if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
            track_point[track_id] = Some(*refined);
        }
    }
    Ok(())
}

/// Bundle-adjust a chosen `variable` set of poses plus every triangulated track
/// they observe. Other registered images observing those tracks join as fixed
/// poses (constraints). The gauge: with ≥2 fixed observers their baseline pins
/// the 7-DoF monocular gauge for free; otherwise (an early, loosely connected
/// neighbourhood) the variable set's own anchor + farthest pose are fixed, as in
/// the global solve. Only variable poses and the solved landmarks are written back.
fn bundle_adjust_local(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
    variable: &HashSet<usize>,
) -> Result<(), BaError> {
    let mut ba = BundleAdjustment::new(camera.clone());

    // Landmarks touching ≥1 variable image, and the images that participate.
    let mut used: HashSet<usize> = HashSet::new();
    let mut lm_ids: Vec<usize> = Vec::new();
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        let mut obs_images: Vec<usize> = Vec::new();
        let mut touches_variable = false;
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if features[image].keypoints.get(kp).is_none() {
                continue;
            }
            obs_images.push(image);
            if variable.contains(&image) {
                touches_variable = true;
            }
        }
        if obs_images.len() < 2 || !touches_variable {
            continue;
        }
        for image in obs_images {
            used.insert(image);
        }
        lm_ids.push(track_id);
    }
    if lm_ids.is_empty() {
        return Ok(());
    }

    for &image in &used {
        ba.add_pose(image as u64, poses[image].clone().unwrap());
        if !variable.contains(&image) {
            ba.fix_pose(image as u64);
        }
    }

    // Need ≥2 fixed poses to pin metric scale; otherwise fix the variable gauge.
    let n_fixed = used.iter().filter(|i| !variable.contains(i)).count();
    if n_fixed < 2 {
        let var_used: Vec<usize> = used
            .iter()
            .copied()
            .filter(|i| variable.contains(i))
            .collect();
        fix_monocular_scale_gauge(&mut ba, poses, &var_used);
    }

    for &track_id in &lm_ids {
        ba.add_landmark(track_id as u64, track_point[track_id].unwrap());
        for &(image, kp) in &tracks[track_id] {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                ba.add_observation(BaObservation {
                    keyframe_id: image as u64,
                    landmark_id: track_id as u64,
                    xy: px,
                });
            }
        }
    }
    ba.optimize(&config.ba_config)?;

    for &image in &used {
        if variable.contains(&image) {
            if let Some(refined) = ba.poses.get(&(image as u64)) {
                poses[image] = Some(refined.clone());
            }
        }
    }
    for &track_id in &lm_ids {
        if let Some(refined) = ba.landmarks.get(&(track_id as u64)) {
            track_point[track_id] = Some(*refined);
        }
    }
    Ok(())
}

/// Pin the 7-DoF monocular gauge (6 rigid + scale) by fixing two of `candidates`:
/// the lowest-index pose (rigid anchor) and the one farthest from it (the scale
/// anchor — longest, best-conditioned baseline). Mirrors the global solve's gauge
/// handling; used by a local solve that lacks two fixed-observer poses of its own.
fn fix_monocular_scale_gauge(
    ba: &mut BundleAdjustment,
    poses: &[Option<Pose>],
    candidates: &[usize],
) {
    let Some(&anchor) = candidates.iter().min() else {
        return;
    };
    ba.fix_pose(anchor as u64);
    let anchor_center = poses[anchor]
        .as_ref()
        .unwrap()
        .camera_to_world()
        .translation;
    let mut farthest = None;
    let mut best_d2 = 0.0;
    for &image in candidates {
        if image == anchor {
            continue;
        }
        let d2 = (poses[image].as_ref().unwrap().camera_to_world().translation - anchor_center)
            .norm_squared();
        if d2 > best_d2 {
            best_d2 = d2;
            farthest = Some(image);
        }
    }
    if let Some(scale_anchor) = farthest {
        ba.fix_pose(scale_anchor as u64);
    }
}

/// COLMAP `IncrementalMapper::IterativeGlobalRefinement`: a global BA, then a loop
/// of {re-triangulate/complete tracks, filter outliers, global BA} until the
/// changed-observation fraction falls below `global_ba_change_rate` (or
/// `global_ba_max_refinements` rounds run). Re-triangulation is forced on here
/// regardless of `config.retriangulate` — completing tracks between global solves
/// is integral to COLMAP's schedule, not the opt-in density lever of the simple
/// path.
fn iterative_global_refinement(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &mut [Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<BaResult, BaError> {
    // Refine intrinsics on each global solve when enabled; the refined camera is
    // carried forward so the next round's filter / re-triangulation / BA all use
    // it (and the caller reads the final camera back from `*camera`).
    let refine = config.refine_intrinsics;
    let run_ba = |cam: &mut Camera,
                  tr: &[Vec<(usize, usize)>],
                  p: &mut [Option<Pose>],
                  tp: &mut [Option<Point3<f64>>]|
     -> Result<BaResult, BaError> {
        let (res, refined) = run_bundle_adjustment(cam, features, tr, config, p, tp, refine)?;
        if let Some(c) = refined {
            *cam = c;
        }
        Ok(res)
    };

    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: final global refinement begin registered={} points={} observations={} max_followup_rounds={}",
            poses.iter().filter(|pose| pose.is_some()).count(),
            track_point.iter().filter(|point| point.is_some()).count(),
            count_observations(tracks, poses, track_point),
            config.global_ba_max_refinements,
        );
    }
    let mut ba_started = std::time::Instant::now();
    let mut result = run_ba(camera, tracks, poses, track_point)?;
    if sfm_debug_enabled() {
        eprintln!(
            "sfm-debug: final global BA round=0 completed seconds={:.3}",
            ba_started.elapsed().as_secs_f64(),
        );
    }
    for followup in 0..config.global_ba_max_refinements {
        let round = followup + 1;
        let total_obs = count_observations(tracks, poses, track_point).max(1);
        // Filter outlier observations, then complete/re-triangulate tracks the
        // tightened frame can now place. Completing between solves is integral to
        // the schedule — it gives the next global BA more constraints and, on this
        // metric video, measurably beats filter-only (1.64 cm vs 2.21 cm); the
        // forward-motion low-parallax churn it induces against the filter is the
        // price, and the track-density ceiling it leaves is the next lever.
        let support_before = sfm_ba_debug_enabled().then(|| {
            (
                poses.iter().filter(|pose| pose.is_some()).count(),
                track_point.iter().filter(|point| point.is_some()).count(),
                count_observations(tracks, poses, track_point),
            )
        });
        let mut changed =
            filter_outlier_observations(camera, features, tracks, config, poses, track_point);
        let support_after_filter = sfm_ba_debug_enabled().then(|| {
            (
                poses.iter().filter(|pose| pose.is_some()).count(),
                track_point.iter().filter(|point| point.is_some()).count(),
                count_observations(tracks, poses, track_point),
            )
        });
        changed += retriangulate_tracks(camera, features, tracks, config, poses, track_point);
        let support_after_retriangulation = sfm_ba_debug_enabled().then(|| {
            (
                poses.iter().filter(|pose| pose.is_some()).count(),
                track_point.iter().filter(|point| point.is_some()).count(),
                count_observations(tracks, poses, track_point),
            )
        });
        let change_rate = changed as f64 / total_obs as f64;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: final global refinement round={round} changed={changed}/{total_obs} rate={change_rate:.6} threshold={:.6}",
                config.global_ba_change_rate,
            );
        }
        if let (Some(before), Some(after_filter), Some(after_retriangulation)) = (
            support_before,
            support_after_filter,
            support_after_retriangulation,
        ) {
            eprintln!(
                "sfm-debug-ba-support: stage=final_refinement round={round} \
                 before=({},{},{}) after_filter=({},{},{}) \
                 after_retriangulation=({},{},{}) changed={changed}",
                before.0,
                before.1,
                before.2,
                after_filter.0,
                after_filter.1,
                after_filter.2,
                after_retriangulation.0,
                after_retriangulation.1,
                after_retriangulation.2,
            );
        }
        if change_rate < config.global_ba_change_rate {
            if sfm_debug_enabled() {
                eprintln!("sfm-debug: final global refinement converged before BA round={round}");
            }
            break;
        }
        if sfm_debug_enabled() {
            eprintln!("sfm-debug: final global BA round={round} begin");
        }
        ba_started = std::time::Instant::now();
        result = run_ba(camera, tracks, poses, track_point)?;
        if sfm_debug_enabled() {
            eprintln!(
                "sfm-debug: final global BA round={round} completed seconds={:.3}",
                ba_started.elapsed().as_secs_f64(),
            );
        }
    }
    Ok(result)
}

/// In-growth global refinement, used during the seed search where `tracks` is
/// shared read-only across trials: global BA, then up to a couple rounds of
/// {re-triangulate/complete, global BA} while it keeps completing tracks. The
/// completion is what keeps registration moving — a freshly tightened global
/// frame lets [`retriangulate_tracks`] triangulate tracks the narrow growth-time
/// baseline had missed, and those new 3D points give the next PnP enough
/// 2D-3D matches to register (without it, registration stalls well short of full
/// coverage and the trajectory develops ATE-wrecking gaps). The track-membership
/// *filter* (which would mutate the shared tracks) is deferred to the final
/// [`iterative_global_refinement`] after a seed is committed.
fn growth_global_refinement(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<(), BaError> {
    // When intrinsics refinement is on, co-evolve them in these periodic global
    // passes (COLMAP's IterativeGlobalRefinement keeps the camera moving with the
    // structure, so a wrong focal is corrected while the model is still small
    // enough to expose it — the well-conditioned global solve, not the narrow
    // per-registration local one, is where the focal is observable). Otherwise the
    // intrinsics stay fixed and the refined slot is always None.
    let refine = config.refine_intrinsics;
    let run_global = |cam: &mut Camera,
                      p: &mut [Option<Pose>],
                      tp: &mut [Option<Point3<f64>>]|
     -> Result<(), BaError> {
        let (_, refined) = run_bundle_adjustment(cam, features, tracks, config, p, tp, refine)?;
        if let Some(c) = refined {
            *cam = c;
        }
        Ok(())
    };

    run_global(camera, poses, track_point)?;
    for _ in 0..config.global_ba_max_refinements.min(2) {
        let changed = retriangulate_tracks(camera, features, tracks, config, poses, track_point);
        if changed == 0 {
            break;
        }
        run_global(camera, poses, track_point)?;
    }
    Ok(())
}

/// Summary of an optional final fixed-support BA polish.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct FinalBaPolishStats {
    pub(crate) requested_iterations: usize,
    pub(crate) accepted: bool,
    pub(crate) initial_sse: f64,
    pub(crate) final_sse: f64,
    pub(crate) solver_iterations: usize,
    pub(crate) accepted_steps: usize,
    pub(crate) rejected_steps: usize,
    pub(crate) converged: bool,
    pub(crate) max_pose_step: f64,
    pub(crate) max_landmark_step: f64,
    pub(crate) final_lambda: f64,
    /// Number of non-empty final BA landmarks (the output-track support).
    pub(crate) tracks_before: usize,
    pub(crate) tracks_after: usize,
    pub(crate) observations_before: usize,
    pub(crate) observations_after: usize,
}

/// Run an optional fixed-support BA solve on the final support without allowing
/// the solve to change track membership, observation membership, or camera
/// intrinsics. An explicit `final_ba_polish_iterations` uses the historical
/// pure-L2 objective; `geometry_weighted_ba` instead keeps the ordinary robust
/// objective and changes only the fixed pre-BA observation weights. The
/// ordinary robust/refinement schedule has already completed before this
/// function is called. We snapshot all mutable state and commit only a finite,
/// non-increasing weighted-cost result.
pub(crate) fn final_fixed_support_ba_polish(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> Result<(FinalBaPolishStats, Option<BaResult>), BaError> {
    let requested_iterations = if config.final_ba_polish_iterations > 0 {
        config.final_ba_polish_iterations
    } else if config.geometry_weighted_ba {
        // The geometry-weighted mode is itself a final fixed-support solve. If
        // no separate polish cap is supplied, use the ordinary BA cap rather
        // than inventing another tuning knob.
        config.ba_config.max_iterations
    } else {
        0
    };
    let tracks_before = track_point.iter().filter(|point| point.is_some()).count();
    let observations_before = count_observations(tracks, poses, track_point);
    let mut stats = FinalBaPolishStats {
        requested_iterations,
        accepted: false,
        initial_sse: f64::NAN,
        final_sse: f64::NAN,
        solver_iterations: 0,
        accepted_steps: 0,
        rejected_steps: 0,
        converged: false,
        max_pose_step: 0.0,
        max_landmark_step: 0.0,
        final_lambda: 0.0,
        tracks_before,
        tracks_after: tracks_before,
        observations_before,
        observations_after: observations_before,
    };
    if requested_iterations == 0 {
        return Ok((stats, None));
    }

    let poses_before = poses.to_vec();
    let track_point_before = track_point.to_vec();
    let mut polish_config = config.clone();
    polish_config.refine_intrinsics = false;
    polish_config.ba_config = if config.final_ba_polish_iterations > 0 {
        // Preserve the historical explicit fixed-support polish contract:
        // pure L2 with the caller's requested cap.
        BaConfig {
            max_iterations: requested_iterations,
            robust_kernel: RobustKernel::None,
            refine_intrinsics: false,
            ..config.ba_config
        }
    } else {
        // Geometry weighting is a controlled observation-information A/B. Keep
        // the ordinary final objective (usually Huber) so the only changed
        // factor is the fixed pre-BA track weight.
        BaConfig {
            max_iterations: requested_iterations,
            refine_intrinsics: false,
            ..config.ba_config
        }
    };

    let result = match run_bundle_adjustment_impl(
        camera,
        features,
        tracks,
        &polish_config,
        poses,
        track_point,
        false,
        config.geometry_weighted_ba,
    ) {
        Ok(result) => result,
        Err(error) => {
            poses.clone_from_slice(&poses_before);
            track_point.clone_from_slice(&track_point_before);
            return Err(error);
        }
    };
    let (result, _) = result;
    stats.initial_sse = result.initial_cost;
    stats.final_sse = result.final_cost;
    stats.solver_iterations = result.iterations.len();
    stats.accepted_steps = result
        .iterations
        .iter()
        .filter(|iteration| iteration.step_accepted)
        .count();
    stats.rejected_steps = result.iterations.len().saturating_sub(stats.accepted_steps);
    stats.converged = result.converged;
    if let Some(last) = result.iterations.last() {
        stats.max_pose_step = last.max_pose_step;
        stats.max_landmark_step = last.max_landmark_step;
        stats.final_lambda = last.lambda;
    }

    let finite_state = poses.iter().all(|pose| {
        pose.as_ref().is_none_or(|pose| {
            pose.world_to_camera
                .translation
                .iter()
                .all(|value| value.is_finite())
                && pose
                    .world_to_camera
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
        })
    }) && track_point.iter().all(|point| {
        point
            .as_ref()
            .is_none_or(|point| point.coords.iter().all(|value| value.is_finite()))
    });
    stats.tracks_after = track_point.iter().filter(|point| point.is_some()).count();
    stats.observations_after = count_observations(tracks, poses, track_point);
    let support_unchanged = stats.tracks_after == tracks_before
        && stats.observations_after == observations_before
        && poses
            .iter()
            .zip(poses_before.iter())
            .all(|(after, before)| after.is_some() == before.is_some())
        && track_point
            .iter()
            .zip(track_point_before.iter())
            .all(|(after, before)| after.is_some() == before.is_some());
    let cost_nonincreasing = stats.initial_sse.is_finite()
        && stats.final_sse.is_finite()
        && stats.final_sse <= stats.initial_sse;
    stats.accepted = finite_state && support_unchanged && cost_nonincreasing;

    if !stats.accepted {
        poses.clone_from_slice(&poses_before);
        track_point.clone_from_slice(&track_point_before);
        stats.observations_after = observations_before;
    }
    if sfm_ba_debug_enabled() {
        eprintln!(
            concat!(
                "sfm-debug-ba-polish: requested_iterations={} accepted={} ",
                "geometry_weighted={} ",
                "support_tracks={}=>{} support_observations={}=>{} ",
                "initial_sse={:.9e} final_sse={:.9e} solver_iterations={} ",
                "accepted_steps={} rejected_steps={} converged={} ",
                "last_step=({:.3e},{:.3e}) last_lambda={:.3e}"
            ),
            stats.requested_iterations,
            stats.accepted,
            config.geometry_weighted_ba,
            stats.tracks_before,
            stats.tracks_after,
            stats.observations_before,
            stats.observations_after,
            stats.initial_sse,
            stats.final_sse,
            stats.solver_iterations,
            stats.accepted_steps,
            stats.rejected_steps,
            stats.converged,
            stats.max_pose_step,
            stats.max_landmark_step,
            stats.final_lambda,
        );
    }
    Ok((stats, stats.accepted.then_some(result)))
}

/// Total triangulated observations: for every track with a 3D point, the number
/// of its registered observations. The denominator for the refinement-loop
/// change-rate stop test.
fn count_observations(
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> usize {
    let mut n = 0usize;
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point[track_id].is_none() {
            continue;
        }
        n += track
            .iter()
            .filter(|&&(img, _)| poses[img].is_some())
            .count();
    }
    n
}

/// Whether a plain-growth periodic BA is due at the current registration
/// boundary. Keeping this decision in a pure helper makes the opt-in schedule
/// explicit and testable: `minimum_registered == 0` is exactly the historical
/// behavior, while a positive minimum only defers a due solve and never
/// disables the final BA.
fn periodic_ba_due(
    ba_every: usize,
    minimum_registered: usize,
    registrations_since_ba: usize,
    registered: usize,
) -> bool {
    ba_every > 0
        && registrations_since_ba >= ba_every
        && (minimum_registered == 0 || registered >= minimum_registered)
}

/// Number of post-BA filter/retriangulation rounds for the plain final pass.
/// A disabled final BA also disables this post-BA stage: running it anyway can
/// discover outliers and launch a second global solve after the caller asked
/// for a growth-only result. The default final-BA schedule is unchanged.
fn simple_final_refinement_rounds(config: &IncrementalSfmConfig) -> usize {
    if !config.final_global_ba {
        0
    } else if config.retriangulate {
        config.track_filter_iterations.max(1)
    } else {
        config.track_filter_iterations
    }
}

/// COLMAP `Reconstruction::FilterImages`: de-register registered images whose
/// well-supported observation count has collapsed. For each registered image,
/// count its observations that are triangulated and reproject within
/// `max_reprojection_error_px`; if that count is below
/// `config.filter_min_image_observations`, set its pose to `None`. The two
/// lowest-index registered images (the seed pair) are protected as the gauge
/// anchor, and the registered count is never driven below 3. Returns how many
/// images were de-registered. The caller's grow loop resets the trial counter of
/// any now-unregistered image, so a filtered image can re-register once the
/// surrounding structure improves.
fn filter_images(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &mut [Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> usize {
    let threshold = config.max_reprojection_error_px;
    let min_obs = config.filter_min_image_observations;

    // Per-image count of well-supported (triangulated, in-threshold) observations.
    let mut good_obs = vec![0usize; poses.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        for &(image, kp) in track {
            let Some(pose) = &poses[image] else { continue };
            let Some(px) = features[image].keypoints.get(kp).copied() else {
                continue;
            };
            if matches!(reprojection_error_px(camera, pose, &point, &px), Some(e) if e <= threshold)
            {
                good_obs[image] += 1;
            }
        }
    }

    // Protect the seed pair (the two lowest-index registered images) — they pin the
    // 7-DoF monocular gauge — and keep at least three registered images alive.
    let registered: Vec<usize> = (0..poses.len()).filter(|&i| poses[i].is_some()).collect();
    let protected: std::collections::HashSet<usize> = registered.iter().take(2).copied().collect();
    let mut remaining = registered.len();

    let mut removed = 0usize;
    for &image in &registered {
        if remaining <= 3 || protected.contains(&image) {
            continue;
        }
        if good_obs[image] < min_obs {
            poses[image] = None;
            removed += 1;
            remaining -= 1;
        }
    }
    removed
}

/// Clean every triangulated track after the current BA, on two grounds:
///
/// 1. **Reprojection.** A contaminated union-find track — two distinct 3D points
///    merged into one — has a BA'd point that fits neither cluster, so its
///    observations reproject past `max_reprojection_error_px` and are stripped;
///    a track left below the minimum posed observations is dropped.
/// 2. **Parallax.** A point first triangulated just over the parallax gate is
///    depth-unstable: BA can slide it far along its viewing ray without changing
///    any reprojection (low parallax = depth ambiguity), so it survives the
///    reprojection test while sitting thousands of units from the scene — these
///    far-flung outliers wreck the scene scale for downstream 3DGS / MVS. So
///    re-measure parallax against the *current* point and all observing camera
///    centres (the widest angle subtended at the point), and drop the track if
///    it is below `min_triangulation_angle_deg`.
///
/// Observations in *unregistered* images are kept untouched (the BA already
/// ignores them); no pose is ever removed, so the registered-image count is
/// invariant. Returns how many tracks/observations changed (zero ⇒ converged).
fn filter_outlier_observations(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &mut [Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> usize {
    let threshold = config.max_reprojection_error_px;
    let min_obs = config.min_track_length.max(2);
    let mut changed = 0usize;

    for (track_id, track) in tracks.iter_mut().enumerate() {
        let Some(point) = track_point[track_id] else {
            continue;
        };
        let before = track.len();
        track.retain(|&(image, kp)| {
            let Some(pose) = &poses[image] else {
                return true; // unregistered view: BA ignores it, cannot judge.
            };
            let Some(px) = features[image].keypoints.get(kp).copied() else {
                return false;
            };
            match reprojection_error_px(camera, pose, &point, &px) {
                Some(err) => err <= threshold,
                None => false, // behind the camera => outlier.
            }
        });
        changed += before - track.len();

        let posed_obs = track
            .iter()
            .filter(|&&(image, _)| poses[image].is_some())
            .count();
        if posed_obs < min_obs {
            if track_point[track_id].take().is_some() {
                changed += 1;
            }
            continue;
        }

        // Drop a low-parallax track unless the multi-view exemption keeps it: a
        // long forward-motion track below the strict angle but seen by many views
        // is well-constrained, while a 2-view depth-ambiguous one is not.
        if !parallax_angle_ok(track_max_parallax(poses, track, &point), posed_obs, config)
            && track_point[track_id].take().is_some()
        {
            changed += 1;
        }
    }
    changed
}

/// Re-triangulate tracks after a bundle adjustment has moved the poses — the
/// COLMAP completeness/refinement step the single-pass growth lacks. For each
/// track with ≥2 registered observations, triangulate a fresh point from the
/// current widest-parallax view pair ([`triangulate_track`], so it still passes
/// the parallax + reprojection gates) and either:
///
///  1. **Complete** an un-triangulated track. At growth time its registered
///     views were a narrow baseline and the parallax gate rejected it; the
///     BA-refined geometry (more views registered, wider baselines) can now place
///     it. The new point constrains the next BA.
///  2. **Re-seed** an existing point, but only as a **guarded swap**: keep the
///     re-triangulation only if it lowers the track's mean reprojection over its
///     registered observations. A point a multi-view BA already placed better is
///     never regressed, so the step is monotone per track.
///
/// Poses are read-only here; the caller re-runs the BA afterwards. Returns how
/// many tracks gained or improved a point (zero ⇒ nothing changed, converged).
fn retriangulate_tracks(
    camera: &Camera,
    features: &[FeatureSet],
    tracks: &[Vec<(usize, usize)>],
    config: &IncrementalSfmConfig,
    poses: &[Option<Pose>],
    track_point: &mut [Option<Point3<f64>>],
) -> usize {
    let mut changed = 0usize;

    for (track_id, track) in tracks.iter().enumerate() {
        // Registered observations of this track: (image, pixel).
        let mut obs: Vec<(usize, Point2<f64>)> = Vec::new();
        for &(image, kp) in track {
            if poses[image].is_none() {
                continue;
            }
            if let Some(px) = features[image].keypoints.get(kp).copied() {
                obs.push((image, px));
            }
        }
        if obs.len() < 2 {
            continue;
        }
        let Some(candidate) = triangulate_track(camera, poses, &obs, config) else {
            continue;
        };

        match track_point[track_id] {
            None => {
                track_point[track_id] = Some(candidate);
                changed += 1;
            }
            Some(current) => {
                // Mean reprojection of a point over this track's registered obs.
                let mean_reproj = |p: &Point3<f64>| -> f64 {
                    let mut sum = 0.0;
                    let mut n = 0usize;
                    for &(image, px) in &obs {
                        let Some(pose) = &poses[image] else { continue };
                        if let Some(err) = reprojection_error_px(camera, pose, p, &px) {
                            sum += err;
                            n += 1;
                        }
                    }
                    if n > 0 {
                        sum / n as f64
                    } else {
                        f64::INFINITY
                    }
                };
                if mean_reproj(&candidate) + 1e-9 < mean_reproj(&current) {
                    track_point[track_id] = Some(candidate);
                    changed += 1;
                }
            }
        }
    }
    changed
}

/// Summary of the optional final minimum-track-length gate.  The gate is
/// intentionally a post-registration operation: it never participates in
/// seed selection or PnP growth, and a failed refinement restores the complete
/// pre-gate state.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct FinalTrackLengthGateStats {
    requested_min_length: usize,
    attempted: bool,
    accepted: bool,
    tracks_before: usize,
    tracks_removed: usize,
    tracks_after: usize,
    observations_before: usize,
    observations_removed: usize,
    observations_after: usize,
    retriangulated_tracks: usize,
    registered_before: usize,
    registered_after: usize,
    mean_before_ba: f64,
    mean_after_ba: f64,
    finite_state: bool,
    support_valid: bool,
    objective_valid: bool,
}

/// Keep only tracks meeting a final minimum observation count while preserving
/// the parallel `track_point` indexing.  This small pure helper is also used
/// by unit tests so the length-2 removal/length-3 preservation contract does
/// not depend on a camera or solver.
fn retain_final_track_length(
    tracks: &mut Vec<Vec<(usize, usize)>>,
    track_point: &mut Vec<Option<Point3<f64>>>,
    min_length: usize,
) -> (usize, usize) {
    debug_assert_eq!(tracks.len(), track_point.len());
    if min_length <= 2 {
        return (0, 0);
    }

    let old_tracks = std::mem::take(tracks);
    let old_points = std::mem::take(track_point);
    let mut removed_tracks = 0usize;
    let mut removed_observations = 0usize;
    let mut kept_tracks = Vec::with_capacity(old_tracks.len());
    let mut kept_points = Vec::with_capacity(old_points.len());
    for (track, point) in old_tracks.into_iter().zip(old_points) {
        if track.len() < min_length {
            removed_tracks += 1;
            removed_observations += track.len();
        } else {
            kept_tracks.push(track);
            kept_points.push(point);
        }
    }
    *tracks = kept_tracks;
    *track_point = kept_points;
    (removed_tracks, removed_observations)
}

/// A final support gate is valid only when every registered camera still has
/// at least one triangulated observation.  This is deliberately weaker than a
/// density heuristic: the gate is allowed to remove all two-view landmarks,
/// but must never turn a registered camera into an unsupported pose.
fn final_track_length_support_is_valid(
    tracks: &[Vec<(usize, usize)>],
    poses: &[Option<Pose>],
    track_point: &[Option<Point3<f64>>],
) -> bool {
    let mut image_support = vec![0usize; poses.len()];
    for (track_id, track) in tracks.iter().enumerate() {
        if track_point.get(track_id).and_then(Option::as_ref).is_none() {
            continue;
        }
        for &(image, _) in track {
            if poses.get(image).is_some_and(Option::is_some) {
                image_support[image] += 1;
            }
        }
    }
    poses
        .iter()
        .enumerate()
        .all(|(image, pose)| pose.is_none() || image_support.get(image).copied().unwrap_or(0) > 0)
}

fn final_track_length_state_is_finite(
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[Vec<(usize, usize)>],
    track_point: &[Option<Point3<f64>>],
) -> bool {
    let camera_finite = camera.params.iter().all(|value| value.is_finite());
    camera_finite
        && poses.iter().all(|pose| {
            pose.as_ref().is_none_or(|pose| {
                pose.world_to_camera
                    .rotation
                    .coords
                    .iter()
                    .all(|value| value.is_finite())
                    && pose
                        .world_to_camera
                        .translation
                        .iter()
                        .all(|value| value.is_finite())
            })
        })
        && tracks.len() == track_point.len()
        && track_point.iter().all(|point| {
            point
                .as_ref()
                .is_none_or(|point| point.coords.iter().all(|value| value.is_finite()))
        })
}

/// Remove short landmarks after all registration/splitting work, re-triangulate
/// the remaining support, and run one guarded final BA.  Any solver error,
/// non-finite state, loss of registered-camera support, or increase of the
/// remaining-support reprojection objective rolls back the whole operation.
/// `None` and values below three are no-ops; the example CLI currently exposes
/// only the source-motivated value three.
fn apply_final_track_length_gate(
    camera: &mut Camera,
    features: &[FeatureSet],
    tracks: &mut Vec<Vec<(usize, usize)>>,
    config: &IncrementalSfmConfig,
    poses: &mut Vec<Option<Pose>>,
    track_point: &mut Vec<Option<Point3<f64>>>,
    ba_result: &mut Option<BaResult>,
) -> FinalTrackLengthGateStats {
    let Some(min_length) = config.final_min_track_length else {
        return FinalTrackLengthGateStats::default();
    };
    let mut stats = FinalTrackLengthGateStats {
        requested_min_length: min_length,
        tracks_before: tracks.len(),
        observations_before: tracks.iter().map(Vec::len).sum(),
        registered_before: poses.iter().filter(|pose| pose.is_some()).count(),
        ..FinalTrackLengthGateStats::default()
    };
    stats.tracks_after = stats.tracks_before;
    stats.observations_after = stats.observations_before;
    stats.registered_after = stats.registered_before;
    if min_length <= 2
        || !config.final_global_ba
        || !poses.iter().all(Option::is_some)
        || tracks.len() != track_point.len()
    {
        return stats;
    }
    stats.attempted = true;

    let tracks_before = tracks.clone();
    let points_before = track_point.clone();
    let poses_before = poses.clone();
    let camera_before = camera.clone();
    let ba_before = ba_result.clone();
    let (removed_tracks, removed_observations) =
        retain_final_track_length(tracks, track_point, min_length);
    stats.tracks_removed = removed_tracks;
    stats.observations_removed = removed_observations;
    stats.tracks_after = tracks.len();
    stats.observations_after = tracks.iter().map(Vec::len).sum();
    if removed_tracks == 0 {
        stats.accepted = true;
        return stats;
    }

    stats.retriangulated_tracks =
        retriangulate_tracks(camera, features, tracks, config, poses, track_point);
    let mean_before_ba = mean_reprojection_for_track_range(
        camera,
        features,
        tracks,
        poses,
        track_point,
        0,
        tracks.len(),
    );
    stats.mean_before_ba = mean_before_ba;
    let ba = run_bundle_adjustment(
        camera,
        features,
        tracks,
        config,
        poses,
        track_point,
        config.refine_intrinsics,
    );
    let (candidate_ba, refined_camera) = match ba {
        Ok(result) => result,
        Err(_) => {
            *tracks = tracks_before;
            *track_point = points_before;
            *poses = poses_before;
            *camera = camera_before;
            *ba_result = ba_before;
            return stats;
        }
    };
    if let Some(refined_camera) = refined_camera {
        *camera = refined_camera;
    }
    let mean_after = mean_reprojection_for_track_range(
        camera,
        features,
        tracks,
        poses,
        track_point,
        0,
        tracks.len(),
    );
    stats.mean_after_ba = mean_after;
    let registered_after = poses.iter().filter(|pose| pose.is_some()).count();
    stats.registered_after = registered_after;
    let finite = final_track_length_state_is_finite(camera, poses, tracks, track_point);
    let support_valid = final_track_length_support_is_valid(tracks, poses, track_point);
    // The solver's reported robust objective is the acceptance objective.  A
    // mean-pixel change can move slightly upward when short, weak tracks are
    // removed (the denominator and residual population both change), even
    // while the fixed-support BA objective decreases.  Keep that diagnostic
    // pair of means in the log, but do not reject a finite, support-preserving
    // solve solely for this population-statistic effect.
    let objective_valid = candidate_ba.initial_cost.is_finite()
        && candidate_ba.final_cost.is_finite()
        && candidate_ba.final_cost <= candidate_ba.initial_cost + 1.0e-9;
    stats.finite_state = finite;
    stats.support_valid = support_valid;
    stats.objective_valid = objective_valid;
    if finite && support_valid && registered_after == stats.registered_before && objective_valid {
        *ba_result = Some(candidate_ba);
        stats.accepted = true;
    } else {
        *tracks = tracks_before;
        *track_point = points_before;
        *poses = poses_before;
        *camera = camera_before;
        *ba_result = ba_before;
        stats.tracks_after = stats.tracks_before;
        stats.observations_after = stats.observations_before;
        stats.retriangulated_tracks = 0;
    }
    stats
}

/// Numerical cutoff for the geometry-only point block condition proxy used by
/// [`ba_landmark_should_freeze`].  A condition number above 1e8 leaves fewer
/// than eight reliable decimal digits in a 3-D point solve, which is the
/// conventional double-precision boundary for treating a normal-equation
/// block as ill-conditioned.  This is deliberately a fixed numerical
/// criterion, not a dataset/accuracy knob.
const BA_POINT_BLOCK_MAX_CONDITION: f64 = 1.0e8;

/// Geometry captured at the beginning of one BA solve for one landmark.  The
/// point-block condition is the condition number of the accumulated 3×3
/// landmark Jacobian block (with the configured robust observation weights),
/// before any LM step.  It is a local Schur-block proxy: the full reduced
/// camera system is intentionally not approximated here.
#[derive(Debug, Clone, Copy, PartialEq)]
struct LandmarkBaGeometry {
    track_length: usize,
    baseline_depth_ratio: f64,
    max_parallax_deg: f64,
    median_reprojection_px: f64,
    point_condition: f64,
    point_min_eigenvalue: f64,
    point_max_eigenvalue: f64,
    invalid_depth: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LandmarkBaDiagnostic {
    id: u64,
    geometry: LandmarkBaGeometry,
    displacement: f64,
    excluded: bool,
}

/// Projection Jacobian with respect to a camera-frame 3-D point.  The
/// pinhole/no-distortion path mirrors the BA normal-equation Jacobian exactly.
/// For a camera carrying radial distortion, a deterministic central difference
/// keeps this diagnostic conservative without pretending that the pinhole
/// formula is exact for a distorted projection.
fn ba_point_projection_jacobian(
    camera: &Camera,
    point_camera: &Point3<f64>,
) -> Option<Matrix2x3<f64>> {
    let (fx, fy, _, _) = camera.intrinsics()?;
    if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
        return None;
    }
    let has_distortion = camera
        .radial_distortion()
        .is_some_and(|(k1, k2)| k1 != 0.0 || k2 != 0.0);
    if !has_distortion {
        let z_inv = 1.0 / point_camera.z;
        let mut jacobian = Matrix2x3::<f64>::zeros();
        jacobian[(0, 0)] = fx * z_inv;
        jacobian[(0, 2)] = -fx * point_camera.x * z_inv * z_inv;
        jacobian[(1, 1)] = fy * z_inv;
        jacobian[(1, 2)] = -fy * point_camera.y * z_inv * z_inv;
        return jacobian
            .iter()
            .all(|value| value.is_finite())
            .then_some(jacobian);
    }

    let epsilon = (point_camera.coords.norm().max(1.0) * 1.0e-6).max(1.0e-8);
    let mut jacobian = Matrix2x3::<f64>::zeros();
    for axis in 0..3 {
        let mut plus = point_camera.coords;
        let mut minus = point_camera.coords;
        plus[axis] += epsilon;
        minus[axis] -= epsilon;
        let plus = camera.project(&Point3::from(plus))?;
        let minus = camera.project(&Point3::from(minus))?;
        if !plus.coords.iter().all(|value| value.is_finite())
            || !minus.coords.iter().all(|value| value.is_finite())
        {
            return None;
        }
        let derivative = (plus - minus) / (2.0 * epsilon);
        jacobian[(0, axis)] = derivative.x;
        jacobian[(1, axis)] = derivative.y;
    }
    jacobian
        .iter()
        .all(|value| value.is_finite())
        .then_some(jacobian)
}

/// Compute the fixed, pre-BA geometry used both by the diagnostic dump and by
/// the opt-in landmark freeze gate.  Invalid/behind-camera observations mark
/// the point invalid; valid observations still contribute to the condition
/// proxy so the report explains how the gate was reached.
fn ba_landmark_geometry(
    camera: &Camera,
    features: &[FeatureSet],
    poses: &[Option<Pose>],
    track: &[(usize, usize)],
    point: &Point3<f64>,
    kernel: &RobustKernel,
) -> LandmarkBaGeometry {
    let point_finite = point.coords.iter().all(|value| value.is_finite());
    let mut registered_images = HashSet::new();
    let mut centres = Vec::new();
    let mut depths = Vec::new();
    let mut reprojections = Vec::new();
    let mut hessian = Matrix3::<f64>::zeros();
    let mut invalid_depth = !point_finite;

    if point_finite {
        for &(image, keypoint) in track {
            let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                continue;
            };
            registered_images.insert(image);
            let centre = pose.camera_center_world().coords;
            if centre.iter().all(|value| value.is_finite()) {
                centres.push(centre);
            }
            let point_camera = pose.transform_world_point(point);
            if !point_camera.coords.iter().all(|value| value.is_finite()) || point_camera.z <= 0.0 {
                invalid_depth = true;
                continue;
            }
            depths.push(point_camera.z);
            let Some(pixel) = features
                .get(image)
                .and_then(|feature| feature.keypoints.get(keypoint))
                .copied()
            else {
                continue;
            };
            let Some(error) = reprojection_error_px(camera, pose, point, &pixel) else {
                invalid_depth = true;
                continue;
            };
            if error.is_finite() {
                reprojections.push(error);
            }
            let Some(j_projection) = ba_point_projection_jacobian(camera, &point_camera) else {
                invalid_depth = true;
                continue;
            };
            let rotation = pose
                .world_to_camera
                .rotation
                .to_rotation_matrix()
                .into_inner();
            let j_landmark = j_projection * rotation;
            let weight = kernel.weight(error * error);
            if weight.is_finite() && weight > 0.0 {
                hessian += weight * (j_landmark.transpose() * j_landmark);
            }
        }
    }

    depths.sort_by(f64::total_cmp);
    reprojections.sort_by(f64::total_cmp);
    let median_depth = depths
        .get(depths.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::NAN);
    let median_reprojection = reprojections
        .get(reprojections.len().saturating_sub(1) / 2)
        .copied()
        .unwrap_or(f64::NAN);
    let mut max_baseline = 0.0f64;
    for i in 0..centres.len() {
        for j in (i + 1)..centres.len() {
            let baseline = (centres[i] - centres[j]).norm();
            if baseline.is_finite() {
                max_baseline = max_baseline.max(baseline);
            }
        }
    }
    let baseline_depth_ratio = if median_depth.is_finite() && median_depth > 0.0 {
        max_baseline / median_depth
    } else {
        f64::NAN
    };
    let max_parallax_deg = if point_finite {
        track_max_parallax(poses, track, point).to_degrees()
    } else {
        f64::NAN
    };

    let eigenvalues = hessian.symmetric_eigen().eigenvalues;
    let point_min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let point_max_eigenvalue = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let point_condition = if point_min_eigenvalue.is_finite()
        && point_max_eigenvalue.is_finite()
        && point_min_eigenvalue > 0.0
        && point_max_eigenvalue > 0.0
    {
        point_max_eigenvalue / point_min_eigenvalue
    } else {
        f64::INFINITY
    };

    LandmarkBaGeometry {
        track_length: registered_images.len(),
        baseline_depth_ratio,
        max_parallax_deg,
        median_reprojection_px: median_reprojection,
        point_condition,
        point_min_eigenvalue,
        point_max_eigenvalue,
        invalid_depth,
    }
}

fn ba_landmark_is_ill_conditioned(geometry: &LandmarkBaGeometry, min_parallax_deg: f64) -> bool {
    geometry.invalid_depth
        || !geometry.point_condition.is_finite()
        || geometry.point_condition > BA_POINT_BLOCK_MAX_CONDITION
        || (min_parallax_deg.is_finite()
            && min_parallax_deg > 0.0
            && geometry.max_parallax_deg.is_finite()
            && geometry.max_parallax_deg < min_parallax_deg)
}

/// A weak point with a small reprojection residual can still be a useful
/// camera observation, whereas a weak point whose current residual is already
/// outside the ordinary reprojection gate is not a trustworthy fixed camera
/// constraint.  The opt-in safeguard therefore excludes only the latter; the
/// classification is computed once before the solve and never changes during
/// LM iterations.
fn ba_landmark_should_exclude(
    geometry: &LandmarkBaGeometry,
    min_parallax_deg: f64,
    max_reprojection_error_px: f64,
) -> bool {
    if !ba_landmark_is_ill_conditioned(geometry, min_parallax_deg) {
        return false;
    }
    geometry.invalid_depth
        || !geometry.median_reprojection_px.is_finite()
        || !max_reprojection_error_px.is_finite()
        || geometry.median_reprojection_px > max_reprojection_error_px
}

fn sfm_pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let mean_x = xs.iter().sum::<f64>() / xs.len() as f64;
    let mean_y = ys.iter().sum::<f64>() / ys.len() as f64;
    let mut covariance = 0.0;
    let mut variance_x = 0.0;
    let mut variance_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        covariance += dx * dy;
        variance_x += dx * dx;
        variance_y += dy * dy;
    }
    (variance_x > 0.0 && variance_y > 0.0 && covariance.is_finite())
        .then_some(covariance / (variance_x * variance_y).sqrt())
}

/// Print a compact per-landmark report and correlations for one BA solve.
/// Rows are sorted by the *observed* post-solve displacement, making the
/// potentially tiny set driving an extreme point step immediately visible.
fn sfm_debug_ba_landmarks(records: &[LandmarkBaDiagnostic], min_parallax_deg: f64) {
    if records.is_empty() {
        return;
    }
    let mut rows = records.to_vec();
    rows.sort_by(|a, b| {
        b.displacement
            .total_cmp(&a.displacement)
            .then(a.id.cmp(&b.id))
    });
    let displacements: Vec<f64> = rows
        .iter()
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .collect();
    let total_displacement = displacements.iter().sum::<f64>();
    let top_n = rows.len().min(10);
    let top_displacement = rows
        .iter()
        .take(top_n)
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let low_parallax = rows.iter().filter(|row| {
        min_parallax_deg.is_finite()
            && min_parallax_deg > 0.0
            && row.geometry.max_parallax_deg.is_finite()
            && row.geometry.max_parallax_deg < min_parallax_deg
    });
    let low_parallax_count = low_parallax.clone().count();
    let low_parallax_displacement = low_parallax
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let near_condition = rows
        .iter()
        .filter(|row| row.geometry.point_condition > BA_POINT_BLOCK_MAX_CONDITION);
    let near_condition_count = near_condition.clone().count();
    let near_condition_displacement = near_condition
        .map(|row| row.displacement)
        .filter(|value| value.is_finite())
        .sum::<f64>();
    let invalid_depth_count = rows.iter().filter(|row| row.geometry.invalid_depth).count();
    let mut median_displacement = displacements.clone();
    let median_displacement = sfm_oracle_median(&mut median_displacement);
    let finite_rows: Vec<&LandmarkBaDiagnostic> = rows
        .iter()
        .filter(|row| {
            row.displacement.is_finite()
                && row.geometry.baseline_depth_ratio.is_finite()
                && row.geometry.max_parallax_deg.is_finite()
                && row.geometry.median_reprojection_px.is_finite()
                && row.geometry.point_condition.is_finite()
                && row.geometry.point_condition > 0.0
        })
        .collect();
    let correlation = |value: fn(&LandmarkBaGeometry) -> f64| {
        let mut x = Vec::with_capacity(finite_rows.len());
        let mut y = Vec::with_capacity(finite_rows.len());
        for row in &finite_rows {
            let feature = value(&row.geometry);
            if feature.is_finite() && feature > 0.0 && row.displacement > 0.0 {
                x.push(feature.ln());
                y.push(row.displacement.ln());
            }
        }
        sfm_pearson(&x, &y).unwrap_or(f64::NAN)
    };
    let excluded = rows.iter().filter(|row| row.excluded).count();
    eprintln!(
        concat!(
            "sfm-debug-ba-landmarks: count={} excluded={} condition_limit={:.3e} ",
            "disp_max={:.3e} disp_median={:.3e} top10_fraction={:.6} ",
            "low_parallax(<{:.3}deg)={}/{} fraction={:.6} ",
            "near_condition(>{:.3e})={}/{} fraction={:.6} invalid_depth={} ",
            "corr_log_disp=(baseline_depth={:.4},parallax_deg={:.4},reproj={:.4},condition={:.4})"
        ),
        rows.len(),
        excluded,
        BA_POINT_BLOCK_MAX_CONDITION,
        rows.first().map_or(f64::NAN, |row| row.displacement),
        median_displacement,
        if total_displacement > 0.0 {
            top_displacement / total_displacement
        } else {
            f64::NAN
        },
        min_parallax_deg,
        low_parallax_count,
        rows.len(),
        if total_displacement > 0.0 {
            low_parallax_displacement / total_displacement
        } else {
            f64::NAN
        },
        BA_POINT_BLOCK_MAX_CONDITION,
        near_condition_count,
        rows.len(),
        if total_displacement > 0.0 {
            near_condition_displacement / total_displacement
        } else {
            f64::NAN
        },
        invalid_depth_count,
        correlation(|geometry| geometry.baseline_depth_ratio),
        correlation(|geometry| geometry.max_parallax_deg),
        correlation(|geometry| geometry.median_reprojection_px),
        correlation(|geometry| geometry.point_condition),
    );
    for row in rows.iter().take(10) {
        let geometry = row.geometry;
        eprintln!(
            concat!(
                "sfm-debug-ba-landmark: track={} excluded={} len={} baseline_depth={:.6e} ",
                "parallax_deg={:.6e} reproj_px={:.6e} condition={:.6e} ",
                "eig=({:.6e},{:.6e}) displacement={:.6e} invalid_depth={}"
            ),
            row.id,
            row.excluded,
            geometry.track_length,
            geometry.baseline_depth_ratio,
            geometry.max_parallax_deg,
            geometry.median_reprojection_px,
            geometry.point_condition,
            geometry.point_min_eigenvalue,
            geometry.point_max_eigenvalue,
            row.displacement,
            geometry.invalid_depth,
        );
    }
}

/// Widest angle (radians) subtended at `point` by any pair of registered camera
/// centres that observe it — the post-BA triangulation angle. Zero if fewer than
/// two registered views remain.
fn track_max_parallax(
    poses: &[Option<Pose>],
    track: &[(usize, usize)],
    point: &Point3<f64>,
) -> f64 {
    let dirs: Vec<Vector3<f64>> = track
        .iter()
        .filter_map(|&(image, _)| poses[image].as_ref())
        .filter_map(|pose| {
            let v = pose.camera_to_world().translation - point.coords;
            (v.norm() > f64::EPSILON).then(|| v.normalize())
        })
        .collect();
    let mut max_angle = 0.0;
    for a in 0..dirs.len() {
        for b in (a + 1)..dirs.len() {
            let angle = dirs[a].dot(&dirs[b]).clamp(-1.0, 1.0).acos();
            if angle > max_angle {
                max_angle = angle;
            }
        }
    }
    max_angle
}

/// Reprojection error (px) of `point_world` against pixel `px` in a camera.
/// `None` if the point is behind the camera or projection is degenerate.
pub(crate) fn reprojection_error_px(
    camera: &Camera,
    pose: &Pose,
    point_world: &Point3<f64>,
    px: &Point2<f64>,
) -> Option<f64> {
    let cam = pose.transform_world_point(point_world);
    if !cam.z.is_finite() || cam.z <= 0.0 {
        return None;
    }
    let projected = camera.project(&cam)?;
    Some((projected - px).norm())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{UnitQuaternion, Vector3};

    #[test]
    fn sfm_debug_image_filter_parses_trimmed_unique_indices() {
        let expected: HashSet<usize> = [20, 21].into_iter().collect();
        assert_eq!(parse_sfm_debug_images(" 20, 21,20 ").unwrap(), expected);
        assert!(parse_sfm_debug_images("20,nope").is_err());
        assert!(parse_sfm_debug_images(" , \t").is_err());
    }

    #[test]
    fn periodic_ba_schedule_default_identity_and_deferred_boundary() {
        let config = IncrementalSfmConfig::default();
        assert_eq!(config.periodic_ba_min_registered_images, 0);
        for registered in [5, 27, 32, 38] {
            assert!(
                periodic_ba_due(config.ba_every, 0, config.ba_every, registered),
                "minimum=0 must retain the historical schedule"
            );
        }

        // The observed champion growth has five registrations at the first
        // basin-jump boundary (27 cameras), followed by the next periodic
        // boundary at 32. The evidence-derived minimum defers only the former.
        assert!(!periodic_ba_due(5, 32, 5, 27));
        assert!(periodic_ba_due(5, 32, 10, 32));
        assert!(!periodic_ba_due(0, 32, 10, 32));
    }

    #[test]
    fn disabled_final_ba_does_not_run_post_ba_refinement_rounds() {
        let mut config = IncrementalSfmConfig::default();
        assert_eq!(simple_final_refinement_rounds(&config), 2);

        config.final_global_ba = false;
        assert_eq!(
            simple_final_refinement_rounds(&config),
            0,
            "a growth-only run must not launch post-filter BA"
        );

        config.retriangulate = true;
        config.track_filter_iterations = 0;
        assert_eq!(simple_final_refinement_rounds(&config), 0);

        config.final_global_ba = true;
        assert_eq!(simple_final_refinement_rounds(&config), 1);
    }

    #[test]
    fn targeted_growth_fast_path_excludes_support_changing_modes() {
        let defaults = IncrementalSfmConfig::default();
        assert!(targeted_plain_growth_enabled(&defaults, false));
        assert!(!targeted_plain_growth_enabled(&defaults, true));

        let mut colmap = defaults.clone();
        colmap.colmap_style_mapper = true;
        assert!(!targeted_plain_growth_enabled(&colmap, false));

        let mut correspondence = defaults.clone();
        correspondence.incremental_correspondence_triangulation = true;
        assert!(!targeted_plain_growth_enabled(&correspondence, false));

        let mut sequence = defaults;
        sequence.sequence_relative_pose_fallback = true;
        assert!(!targeted_plain_growth_enabled(&sequence, false));
    }

    #[test]
    fn correspondence_count_cache_matches_fresh_scan_after_point_additions() {
        let feature = |count| {
            FeatureSet::new(
                (0..count)
                    .map(|index| Point2::new(index as f64, index as f64))
                    .collect(),
                (0..count).map(|_| vec![0.0f32]).collect(),
            )
            .expect("synthetic feature set")
        };
        let features = vec![feature(2), feature(2), feature(2)];
        let tracks = vec![vec![(0, 0), (1, 0)], vec![(1, 1), (2, 1)]];
        let obs_by_image = vec![vec![(0, 0)], vec![(0, 0), (1, 1)], vec![(1, 1)]];
        let mut points = vec![None, None];
        let mut cache = build_correspondence_count_cache(&features, &obs_by_image, &points);
        assert_eq!(cache, vec![0, 0, 0]);

        points[0] = Some(Point3::new(0.0, 0.0, 1.0));
        update_correspondence_count_cache(&features, &tracks, &points, &[0], &mut cache);
        assert_eq!(cache, vec![1, 1, 0]);
        assert_eq!(
            cache,
            build_correspondence_count_cache(&features, &obs_by_image, &points)
        );

        points[1] = Some(Point3::new(0.0, 0.0, 1.0));
        update_correspondence_count_cache(&features, &tracks, &points, &[1], &mut cache);
        assert_eq!(cache, vec![1, 2, 1]);
        assert_eq!(
            cache,
            build_correspondence_count_cache(&features, &obs_by_image, &points)
        );
    }

    fn pose_with_world_center(x: f64) -> Pose {
        Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-x, 0.0, 0.0))
    }

    #[test]
    fn sequence_fallback_scale_uses_latest_consecutive_median_and_needs_two_steps() {
        let poses = vec![
            Some(pose_with_world_center(0.0)),
            Some(pose_with_world_center(1.0)),
            Some(pose_with_world_center(3.0)),
            Some(pose_with_world_center(6.0)),
            None,
        ];
        let stems = vec![286, 287, 288, 289, 290];
        let (scale, mad, samples) =
            robust_recent_consecutive_step_scale(&poses, &stems).expect("three steps");
        assert!((scale - 2.0).abs() < 1.0e-12);
        assert!((mad - 1.0).abs() < 1.0e-12);
        assert_eq!(samples, 3);

        let insufficient = vec![
            Some(pose_with_world_center(0.0)),
            Some(pose_with_world_center(1.0)),
            None,
        ];
        assert!(robust_recent_consecutive_step_scale(&insufficient, &[1, 2, 3]).is_none());
    }

    #[test]
    fn sequence_projected_scale_follows_straight_velocity() {
        let poses = vec![
            Some(pose_with_world_center(0.0)),
            Some(pose_with_world_center(1.0)),
            Some(pose_with_world_center(2.0)),
            Some(pose_with_world_center(3.0)),
        ];
        let estimate = projected_recent_consecutive_step_scale(
            &poses,
            &[1, 2, 3, 4],
            3,
            Vector3::new(1.0, 0.0, 0.0),
        )
        .expect("two straight steps should project");
        assert!((estimate.0 - 1.0).abs() < 1.0e-12);
        assert!((estimate.1 - 1.0).abs() < 1.0e-12);
        assert!(estimate.2.abs() < 1.0e-12);
        assert_eq!(estimate.3, 2);
        assert!((estimate.4 - Vector3::new(1.0, 0.0, 0.0)).norm() < 1.0e-12);
    }

    #[test]
    fn sequence_projected_scale_handles_turn_and_rejects_bad_direction() {
        let poses = vec![
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(0.0, 0.0, 0.0),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-1.0, 0.0, 0.0),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-2.0, -1.0, 0.0),
            )),
        ];
        let turn = projected_recent_consecutive_step_scale(
            &poses,
            &[1, 2, 3],
            3,
            Vector3::new(1.0, 0.0, 0.0),
        )
        .expect("a bounded turn should retain a positive projection");
        assert!(turn.0 > 0.0);
        assert!((turn.0 - 1.0).abs() < 1.0e-12);
        assert!(projected_recent_consecutive_step_scale(
            &poses,
            &[1, 2, 3],
            3,
            Vector3::new(-1.0, 0.0, 0.0),
        )
        .is_none());
        assert!(projected_recent_consecutive_step_scale(
            &poses,
            &[1, 2, 3],
            3,
            Vector3::new(0.0, 0.0, 1.0),
        )
        .is_none());
    }

    #[test]
    fn sequence_projected_scale_uses_mad_robustly_and_requires_history() {
        let poses = vec![
            Some(pose_with_world_center(0.0)),
            Some(pose_with_world_center(1.0)),
            Some(pose_with_world_center(3.0)),
            Some(pose_with_world_center(13.0)),
        ];
        let estimate = projected_recent_consecutive_step_scale(
            &poses,
            &[1, 2, 3, 4],
            4,
            Vector3::new(1.0, 0.0, 0.0),
        )
        .expect("the median velocity should reject the isolated long step");
        assert!((estimate.0 - 1.5).abs() < 1.0e-12);
        assert_eq!(estimate.3, 2);
        let insufficient = vec![
            Some(pose_with_world_center(0.0)),
            Some(pose_with_world_center(1.0)),
            None,
        ];
        assert!(projected_recent_consecutive_step_scale(
            &insufficient,
            &[1, 2, 3],
            2,
            Vector3::new(1.0, 0.0, 0.0),
        )
        .is_none());

        // A zero-MAD magnitude sample is still bounded: a component-wise
        // velocity median that points between equal-length turns must not
        // invent a larger step than the recent median.
        let turning = vec![
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(0.0, 0.0, 0.0),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-1.0, 0.0, 0.0),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-1.0, -1.0, 0.0),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-6.0, -6.0, 0.0),
            )),
        ];
        assert!(projected_recent_consecutive_step_scale(
            &turning,
            &[1, 2, 3, 4],
            4,
            Vector3::new(1.0, 1.0, 0.0),
        )
        .is_none());
        let diagnostic = projected_recent_consecutive_step_scale_diagnostic(
            &turning,
            &[1, 2, 3, 4],
            4,
            Vector3::new(1.0, 1.0, 0.0),
        )
        .expect("relaxed mode still has a finite projection");
        assert!(relaxed_projected_scale_is_valid(
            diagnostic.projected_scale,
            diagnostic.recent_median
        ));
    }

    #[test]
    fn relaxed_projected_scale_uses_only_broad_bounds() {
        assert!(relaxed_projected_scale_is_valid(0.25, 1.0));
        assert!(relaxed_projected_scale_is_valid(4.0, 1.0));
        assert!(!relaxed_projected_scale_is_valid(0.249, 1.0));
        assert!(!relaxed_projected_scale_is_valid(4.001, 1.0));
        assert!(!relaxed_projected_scale_is_valid(1.0, 0.0));
        assert!(!relaxed_projected_scale_is_valid(-1.0, 1.0));
        assert!(!relaxed_projected_scale_is_valid(f64::NAN, 1.0));
    }

    #[test]
    fn carried_sequence_scale_uses_first_projection_then_previous_baseline() {
        // The first fallback has no carry state and therefore keeps the
        // freshly projected scale.  The following consecutive fallback uses
        // the accepted baseline, not a newly projected value.
        assert_eq!(
            carried_sequence_scale_or_projection(None, 1.4, 1.0),
            (1.4, false)
        );
        assert_eq!(
            carried_sequence_scale_or_projection(Some(1.4), 2.8, 1.0),
            (1.4, true)
        );
    }

    #[test]
    fn carried_sequence_scale_invalid_value_falls_back_and_pose_rescale_preserves_rotation() {
        assert_eq!(
            carried_sequence_scale_or_projection(Some(4.001), 1.25, 1.0),
            (1.25, false)
        );
        assert_eq!(
            carried_sequence_scale_or_projection(Some(f64::NAN), 1.25, 1.0),
            (1.25, false)
        );
        assert_eq!(
            next_sequence_fallback_carry_state(23, 1.4, 0),
            Some((23, 1.4))
        );
        assert_eq!(next_sequence_fallback_carry_state(23, 1.4, 1), None);

        let previous = pose_with_world_center(2.0);
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.3);
        let proposed = Pose::from_world_to_camera(rotation, Vector3::new(-5.0, 0.0, 0.0));
        let carried = rescale_sequence_pose_translation(&previous, &proposed, 1.5)
            .expect("finite proposed displacement should rescale");
        assert!(
            ((carried.camera_center_world() - previous.camera_center_world()).norm() - 1.5).abs()
                < 1.0e-12
        );
        assert!(
            carried
                .world_to_camera
                .rotation
                .rotation_to(&rotation)
                .angle()
                < 1.0e-12
        );
        assert!(rescale_sequence_pose_translation(&previous, &proposed, 0.0).is_none());
    }

    #[test]
    fn sequence_fallback_pose_composition_preserves_relative_pose_convention() {
        let previous = pose_with_world_center(4.0);
        let relative_rotation = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.2);
        let relative_translation = Vector3::new(0.0, 1.0, 0.0);
        let relative =
            visloc_core::geometry::SE3::new(relative_rotation, relative_translation * 2.0);
        let expected = relative.compose(&previous.world_to_camera);
        let actual =
            compose_sequence_relative_pose(&previous, relative_rotation, relative_translation, 2.0)
                .expect("finite relative pose");
        assert!((actual.world_to_camera.translation - expected.translation).norm() < 1.0e-12);
        assert!(
            actual
                .world_to_camera
                .rotation
                .rotation_to(&expected.rotation)
                .angle()
                < 1.0e-12
        );
    }

    #[test]
    fn sequence_triangulation_admission_high_support_boundaries() {
        // The relaxed path is inclusive at both evidence-backed boundaries:
        // 100 valid points and exactly 30% of the selected support.
        assert!(sequence_triangulation_admission_ok(100, 300, 30, true));
        assert!(sequence_triangulation_admission_ok(100, 333, 30, true));
        assert!(!sequence_triangulation_admission_ok(100, 334, 30, true));
        assert!(!sequence_triangulation_admission_ok(99, 300, 30, true));
        assert!(!sequence_triangulation_admission_ok(120, 401, 30, true));

        // The override never weakens the configured absolute seed floor.
        assert!(!sequence_triangulation_admission_ok(100, 300, 101, true));

        // A sequence pair without the explicit high-support mark retains the
        // historical half-support gate, including the same selected count.
        assert!(!sequence_triangulation_admission_ok(100, 300, 30, false));
        assert!(sequence_triangulation_admission_ok(150, 300, 30, false));
        assert!(!sequence_triangulation_admission_ok(149, 300, 30, false));
    }

    #[test]
    fn sequence_fallback_is_default_off_and_rejects_bad_essential() {
        let defaults = IncrementalSfmConfig::default();
        assert!(!defaults.sequence_relative_pose_fallback);
        assert!(!defaults.sequence_fallback_after_post);
        assert!(defaults.sequence_stem_values.is_none());

        let features = (0..4)
            .map(|_| {
                FeatureSet::new(
                    (0..30)
                        .map(|index| Point2::new(index as f64 + 1.0, 100.0))
                        .collect(),
                    (0..30).map(|_| vec![0.0f32, 1.0]).collect(),
                )
                .expect("synthetic feature set")
            })
            .collect::<Vec<_>>();
        let matches = (0..30).map(|index| (index, index)).collect::<Vec<_>>();
        let pair = PairwiseMatches {
            image_i: 2,
            image_j: 3,
            matches: matches.clone(),
            two_view_config: Some(ConfigurationType::Uncalibrated),
            essential_matches: Some(matches),
            essential_matrix: Some(Matrix3::zeros()),
        };
        let mut config = defaults;
        config.sequence_relative_pose_fallback = true;
        config.sequence_stem_values = Some(vec![286, 287, 288, 289]);
        let poses = vec![
            Some(pose_with_world_center(0.0)),
            Some(pose_with_world_center(1.0)),
            Some(pose_with_world_center(2.0)),
            None,
        ];
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        assert!(sequence_relative_pose_fallback_with_overrides(
            &camera,
            &features,
            &[pair],
            &poses,
            &config,
            None,
        )
        .is_none());
    }

    #[test]
    fn sequence_fallback_after_post_defers_eager_growth_only() {
        let defaults = IncrementalSfmConfig::default();
        assert!(!sequence_fallback_enabled_during_growth(&defaults));

        let mut eager = defaults.clone();
        eager.sequence_relative_pose_fallback = true;
        assert!(sequence_fallback_enabled_during_growth(&eager));

        let mut deferred = eager;
        deferred.sequence_fallback_after_post = true;
        assert!(!sequence_fallback_enabled_during_growth(&deferred));
        // The scheduling bit is orthogonal to the scale policy; it only moves
        // the same fallback proposal out of the ordinary growth loop.
        deferred.sequence_constant_velocity_scale = true;
        assert!(!sequence_fallback_enabled_during_growth(&deferred));
    }

    #[test]
    fn sfm_oracle_metrics_are_sim3_and_rotation_invariant() {
        let centres = [
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(1.0, 0.0, 0.2),
            Vector3::new(0.1, 1.4, 0.7),
            Vector3::new(-0.5, 0.4, 1.8),
        ];
        let make_pose = |centre: Vector3<f64>, camera_to_world: UnitQuaternion<f64>| {
            Pose::from_world_to_camera(
                camera_to_world.inverse(),
                -(camera_to_world.inverse() * centre),
            )
        };
        let oracle: Vec<Option<Pose>> = centres
            .iter()
            .copied()
            .map(|centre| make_pose(centre, UnitQuaternion::identity()))
            .map(Some)
            .collect();
        let transform = UnitQuaternion::from_axis_angle(&Vector3::z_axis(), 0.37);
        let scale = 2.4;
        let offset = Vector3::new(3.0, -1.0, 0.8);
        let mapped: Vec<Option<Pose>> = centres
            .iter()
            .copied()
            .map(|centre| make_pose(scale * (transform * centre) + offset, transform))
            .map(Some)
            .collect();
        let metrics = sfm_oracle_metrics(&mapped, &oracle).expect("four common poses align");
        assert!(
            metrics.center_rmse < 1.0e-10,
            "center rmse={}",
            metrics.center_rmse
        );
        assert!(
            metrics.rotation_mean < 1.0e-10,
            "rotation={}",
            metrics.rotation_mean
        );
        assert!(sfm_oracle_metrics(&mapped[..2], &oracle[..2]).is_none());
    }

    /// `LocalSubmapBuilder::build`'s scale-pathology retry
    /// (`NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(b)) relies on
    /// `seed_candidate_order` walking to the *next*-ranked seed candidate,
    /// deterministically, once the previously tried pair is excluded. This
    /// pins that mechanism directly: descending match-count order by
    /// default, and excluding a pair (regardless of which of its two image
    /// orderings is recorded) removes exactly that pair and nothing else,
    /// repeatably.
    #[test]
    fn seed_candidate_order_skips_excluded_pairs_deterministically() {
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0); 50],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0); 40],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 2,
                image_j: 3,
                matches: vec![(0, 0); 30],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let config = IncrementalSfmConfig::default();
        assert_eq!(seed_candidate_order(&pairwise, &config), vec![0, 1, 2]);

        let mut selected_pair = config.clone();
        selected_pair.seed_pair = Some((1, 2));
        assert_eq!(seed_candidate_order(&pairwise, &selected_pair), vec![1]);
        // The library field is documented as normalized; the CLI parser
        // canonicalizes reversed user input before constructing this config.
        selected_pair.seed_pair = Some((2, 1));
        assert_eq!(
            seed_candidate_order(&pairwise, &selected_pair),
            Vec::<usize>::new()
        );
        selected_pair.seed_pair = Some((1, 2));
        assert_eq!(seed_candidate_order(&pairwise, &selected_pair), vec![1]);

        let mut excluded_first = config.clone();
        excluded_first.excluded_seed_pairs.insert((0, 1));
        assert_eq!(seed_candidate_order(&pairwise, &excluded_first), vec![1, 2]);
        // Deterministic: repeated calls on the same (excluded) config agree.
        assert_eq!(seed_candidate_order(&pairwise, &excluded_first), vec![1, 2]);

        // The pairwise-side key is normalized regardless of which image is
        // recorded as `image_i`/`image_j`: a reversed-direction entry for
        // the same underlying pair still matches a normalized `(0, 1)`
        // exclusion key.
        let mut reversed_first_pair = pairwise.clone();
        reversed_first_pair[0] = PairwiseMatches {
            image_i: 1,
            image_j: 0,
            matches: vec![(0, 0); 50],
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        };
        let mut excluded_normalized = config.clone();
        excluded_normalized.excluded_seed_pairs.insert((0, 1));
        assert_eq!(
            seed_candidate_order(&reversed_first_pair, &excluded_normalized),
            vec![1, 2]
        );

        // Excluding the two strongest pairs walks to the third-ranked
        // candidate, still in descending order among what remains.
        let mut excluded_two = config;
        excluded_two.excluded_seed_pairs.insert((0, 1));
        excluded_two.excluded_seed_pairs.insert((1, 2));
        assert_eq!(seed_candidate_order(&pairwise, &excluded_two), vec![2]);
    }

    fn physical_track_signature(
        features: &[FeatureSet],
        tracks: &[Vec<(usize, usize)>],
    ) -> Vec<Vec<(usize, i64, i64)>> {
        let mut signature = tracks
            .iter()
            .map(|track| {
                track
                    .iter()
                    .map(|&(image, keypoint)| {
                        let point = features[image].keypoints[keypoint];
                        (
                            image,
                            (point.x * 1_000_000.0).round() as i64,
                            (point.y * 1_000_000.0).round() as i64,
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        signature.sort();
        signature
    }

    #[test]
    fn cycle_supported_tracks_prefer_supported_three_view_edges() {
        let features = vec![
            FeatureSet::new(
                vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(0.0, 1.0), Point2::new(10.0, 1.0)],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap(),
            FeatureSet::new(vec![Point2::new(0.0, 2.0)], vec![vec![0.0f32]]).unwrap(),
        ];
        let pairwise = vec![
            // An unsupported edge is intentionally first. Legacy union-find
            // would absorb it and later drop the whole same-image conflict.
            PairwiseMatches::new(0, 1, vec![(0, 1)]),
            PairwiseMatches::new(0, 1, vec![(0, 0)]),
            PairwiseMatches::new(0, 2, vec![(0, 0)]),
            PairwiseMatches::new(1, 2, vec![(0, 0)]),
        ];
        let adjacency = {
            let mut lookup = HashMap::new();
            for pair in &pairwise {
                let forward = lookup
                    .entry((pair.image_i, pair.image_j))
                    .or_insert_with(HashMap::new);
                for &(a, b) in &pair.matches {
                    forward.entry(a).or_insert_with(HashSet::new).insert(b);
                }
                let reverse = lookup
                    .entry((pair.image_j, pair.image_i))
                    .or_insert_with(HashMap::new);
                for &(a, b) in &pair.matches {
                    reverse.entry(b).or_insert_with(HashSet::new).insert(a);
                }
            }
            lookup
        };
        assert_eq!(cycle_support_for_edge(3, 0, 0, 1, 0, &adjacency), (1, 1));
        assert_eq!(cycle_support_for_edge(3, 0, 0, 1, 1, &adjacency), (0, 0));

        let cycle = build_tracks_cycle_supported(&features, None, &pairwise, 3);
        assert_eq!(cycle.tracks, vec![vec![(0, 0), (1, 0), (2, 0)]]);
        assert_eq!(cycle.stats.retained_tracks, 1);
        assert_eq!(cycle.stats.retained_observations, 3);
        assert!(!cycle.tracks[0].contains(&(1, 1)));
    }

    #[test]
    fn cycle_supported_tracks_are_permutation_invariant() {
        let features = vec![
            FeatureSet::new(
                vec![Point2::new(20.0, 0.0), Point2::new(10.0, 0.0)],
                vec![vec![20.0f32], vec![10.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(20.0, 1.0), Point2::new(10.0, 1.0)],
                vec![vec![20.0f32], vec![10.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(20.0, 2.0), Point2::new(10.0, 2.0)],
                vec![vec![20.0f32], vec![10.0]],
            )
            .unwrap(),
        ];
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(0, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        ];
        let permutations = [[1usize, 0], [1, 0], [1, 0]];
        let permuted_features = features
            .iter()
            .zip(permutations)
            .map(|(set, permutation)| {
                FeatureSet::new(
                    permutation
                        .iter()
                        .map(|&index| set.keypoints[index])
                        .collect(),
                    permutation
                        .iter()
                        .map(|&index| set.descriptors[index].clone())
                        .collect(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let remapped = pairwise
            .iter()
            .map(|pair| {
                PairwiseMatches::new(
                    pair.image_i,
                    pair.image_j,
                    pair.matches
                        .iter()
                        .map(|&(lhs, rhs)| {
                            (
                                permutations[pair.image_i][lhs],
                                permutations[pair.image_j][rhs],
                            )
                        })
                        .rev()
                        .collect(),
                )
            })
            .rev()
            .collect::<Vec<_>>();
        let original = build_tracks_cycle_supported(&features, None, &pairwise, 2);
        let permuted = build_tracks_cycle_supported(&permuted_features, None, &remapped, 2);
        assert_eq!(
            physical_track_signature(&features, &original.tracks),
            physical_track_signature(&permuted_features, &permuted.tracks)
        );
        assert_eq!(original.stats, permuted.stats);
    }

    #[test]
    fn cycle_supported_tracks_have_deterministic_no_cycle_fallback() {
        let features = vec![
            FeatureSet::new(
                vec![Point2::new(0.0, 0.0), Point2::new(10.0, 0.0)],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(0.0, 1.0), Point2::new(10.0, 1.0)],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap(),
            FeatureSet::new(vec![Point2::new(100.0, 2.0)], vec![vec![0.0f32]]).unwrap(),
        ];
        let first = PairwiseMatches::new(0, 1, vec![(0, 1), (0, 0)]);
        let second = PairwiseMatches::new(0, 1, vec![(0, 0), (0, 1)]);
        let output_a =
            build_tracks_cycle_supported(&features, None, &[first.clone(), second.clone()], 2);
        let output_b = build_tracks_cycle_supported(&features, None, &[second, first], 2);
        assert_eq!(output_a.tracks, vec![vec![(0, 0), (1, 0)]]);
        assert_eq!(output_a.tracks, output_b.tracks);
        assert_eq!(output_a.stats, output_b.stats);
    }

    /// A synthetic 3D point cloud and a ring of cameras looking at it, used to
    /// exercise the full unordered pipeline end-to-end.
    struct Scene {
        camera: Camera,
        points: Vec<Point3<f64>>,
        poses: Vec<Pose>,
    }

    fn build_scene() -> Scene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        // A 3D grid of points around the origin.
        let mut points = Vec::new();
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in 0..=2 {
                    points.push(Point3::new(
                        xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.3,
                    ));
                }
            }
        }
        // Cameras on an arc, all looking roughly toward the cloud centre from
        // ~3 m away (enough parallax between neighbours).
        let mut poses = Vec::new();
        for k in 0..6 {
            let angle = -0.5 + k as f64 * 0.2; // radians along the arc
            let radius = 3.0;
            let cam_center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            // Look-at the origin: build world_to_camera.
            let forward = (Point3::origin() - cam_center).normalize();
            let world_up = Vector3::new(0.0, 1.0, 0.0);
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward);
            // Rotation columns map camera axes (x=right, y=down, z=forward) to world.
            let r_cam_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let rot_c2w = nalgebra::Rotation3::from_matrix_unchecked(r_cam_to_world);
            let q_c2w = UnitQuaternion::from_rotation_matrix(&rot_c2w);
            let q_w2c = q_c2w.inverse();
            let t_w2c = -(q_w2c * cam_center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }
        Scene {
            camera,
            points,
            poses,
        }
    }

    /// Project a world point into a pose; `None` if behind camera or off-image.
    fn project(camera: &Camera, pose: &Pose, p: &Point3<f64>) -> Option<Point2<f64>> {
        let cam = pose.transform_world_point(p);
        if cam.z <= 0.05 {
            return None;
        }
        let px = camera.project(&cam)?;
        if px.x < 0.0 || px.x >= camera.width as f64 || px.y < 0.0 || px.y >= camera.height as f64 {
            return None;
        }
        Some(px)
    }

    /// Render the scene to per-image features (keypoint per visible point, the
    /// point index baked into a trivial descriptor) and ground-truth pairwise
    /// matches between every image pair that co-observes ≥8 points.
    fn render(scene: &Scene) -> (Vec<FeatureSet>, Vec<PairwiseMatches>) {
        let n = scene.poses.len();
        // visible[image] = map point_index -> keypoint_index
        let mut features = Vec::new();
        let mut visible: Vec<HashMap<usize, usize>> = Vec::new();
        for pose in &scene.poses {
            let mut kps = Vec::new();
            let mut descs = Vec::new();
            let mut vis = HashMap::new();
            for (pidx, p) in scene.points.iter().enumerate() {
                if let Some(px) = project(&scene.camera, pose, p) {
                    vis.insert(pidx, kps.len());
                    kps.push(px);
                    // Descriptor is irrelevant here (matches are ground truth),
                    // but FeatureSet wants one; use a tiny unique vector.
                    descs.push(vec![pidx as f32, 1.0, 0.0, 0.0]);
                }
            }
            features.push(FeatureSet::new(kps, descs).unwrap());
            visible.push(vis);
        }

        let mut pairwise = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let mut matches = Vec::new();
                for (pidx, &ki) in &visible[i] {
                    if let Some(&kj) = visible[j].get(pidx) {
                        matches.push((ki, kj));
                    }
                }
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                        two_view_config: None,
                        essential_matches: None,
                        essential_matrix: None,
                    });
                }
            }
        }
        (features, pairwise)
    }

    /// M2 acceptance test: on the same realistic multi-image synthetic scene
    /// every other integration test in this module uses
    /// (`build_scene`/`render` — a 45-point cloud seen by a 6-camera ring),
    /// [`build_tracks_via_graph`] must produce **byte-identical** tracks to
    /// the legacy [`build_tracks`] union-find — the refactor gate
    /// `docs/colmap_port_plan.md`'s M2 milestone specifies ("byte-identical
    /// tracks... a refactor gate, not an accuracy claim"), exercised here on
    /// real transitive (multi-hop, multi-image) structure rather than the
    /// small hand-built fixtures above.
    #[test]
    fn graph_tracks_match_union_find_tracks_on_synthetic_scene() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        assert!(
            !pairwise.is_empty(),
            "fixture sanity: the scene must produce at least one verified pair"
        );

        let union_find_tracks = build_tracks(features.len(), &pairwise, 2);
        let graph_tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert_eq!(
            union_find_tracks, graph_tracks,
            "CorrespondenceGraph-derived tracks must byte-match the legacy union-find's"
        );
        assert!(
            !union_find_tracks.is_empty(),
            "fixture sanity: some tracks must form"
        );
    }

    #[test]
    fn post_refinement_pass_registers_against_tightened_structure_once() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let mut poses = vec![None; features.len()];
        poses[0] = Some(scene.poses[0].clone());
        poses[1] = Some(scene.poses[1].clone());
        let mut track_point = vec![None; tracks.len()];
        let config = IncrementalSfmConfig {
            min_pnp_inliers: 8,
            max_reprojection_error_px: 2.0,
            ..IncrementalSfmConfig::default()
        };
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        assert!(track_point.iter().filter(|p| p.is_some()).count() >= 8);

        let added = post_refinement_registration_pass(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses,
            &mut track_point,
        )
        .unwrap();
        assert!(added > 0);
        assert_eq!(poses.iter().filter(|p| p.is_some()).count(), 2 + added);
    }

    #[test]
    fn pose_guided_split_recovers_two_points_prunes_outlier_and_is_deterministic() {
        let scene = build_scene();
        let mut visible_points = scene
            .points
            .iter()
            .filter(|point| {
                (0..4).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
            })
            .copied();
        let point_a = visible_points
            .next()
            .expect("a point visible in four views");
        let point_b = visible_points
            .next()
            .expect("a second point visible in four views");
        let mut features = Vec::new();
        for image in 0..4 {
            let pixel_a = project(&scene.camera, &scene.poses[image], &point_a).unwrap();
            let pixel_b = project(&scene.camera, &scene.poses[image], &point_b).unwrap();
            let outlier = if image == 3 {
                Point2::new(pixel_a.x + 80.0, pixel_a.y + 45.0)
            } else {
                pixel_a
            };
            features.push(
                FeatureSet::new(
                    vec![pixel_a, pixel_b, outlier],
                    vec![vec![0.0f32], vec![1.0], vec![2.0]],
                )
                .unwrap(),
            );
        }

        // The false cross-edge joins two physical points into one legacy
        // component; the final edge joins one of those points to an outlier
        // in an already represented image.  A posed split must recover the
        // two four-view tracks and discard the lone outlier.
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(2, 3, vec![(0, 0), (1, 1), (0, 2)]),
        ];
        let component = vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (3, 2),
        ];
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            min_track_length: 2,
            max_reprojection_error_px: 4.0,
            conflict_recovery_max_hypotheses: 16,
            pose_guided_track_splitting: true,
            ..IncrementalSfmConfig::default()
        };
        let mut incomplete_poses = poses.clone();
        incomplete_poses[3] = None;
        assert!(pose_guided_split_tracks(
            &scene.camera,
            &features,
            &pairwise,
            &[],
            std::slice::from_ref(&component),
            &[],
            &incomplete_poses,
            &config,
        )
        .is_none());
        let first = pose_guided_split_tracks(
            &scene.camera,
            &features,
            &pairwise,
            &[],
            std::slice::from_ref(&component),
            &[],
            &poses,
            &config,
        )
        .expect("complete poses should enable the split");
        assert_eq!(first.stats.split_components, 1);
        assert_eq!(first.stats.emitted_tracks, 2);
        assert!(first.stats.discarded_observations >= 1);
        assert!(first
            .tracks
            .iter()
            .any(|track| track == &vec![(0, 0), (1, 0), (2, 0), (3, 0)]));
        assert!(first
            .tracks
            .iter()
            .any(|track| track == &vec![(0, 1), (1, 1), (2, 1), (3, 1)]));

        let reordered_pairwise = pairwise
            .iter()
            .rev()
            .map(|pair| {
                let mut pair = pair.clone();
                pair.matches.reverse();
                pair
            })
            .collect::<Vec<_>>();
        let mut reversed_component = component.clone();
        reversed_component.reverse();
        let second = pose_guided_split_tracks(
            &scene.camera,
            &features,
            &reordered_pairwise,
            &[],
            std::slice::from_ref(&reversed_component),
            &[],
            &poses,
            &config,
        )
        .expect("reordered input should remain deterministic");
        assert_eq!(first.tracks, second.tracks);
        assert_eq!(first.points, second.points);
        assert_eq!(first.stats, second.stats);
        assert!(!IncrementalSfmConfig::default().pose_guided_track_splitting);
        assert_eq!(
            IncrementalSfmConfig::default().pose_guided_split_max_reprojection_error_px,
            None
        );
        assert_eq!(
            IncrementalSfmConfig::default().pose_guided_track_splitting_iterations,
            1
        );
        assert!(!IncrementalSfmConfig::default().pose_guided_bridge_cuts);
        assert!(!IncrementalSfmConfig::default().pose_guided_track_merging);
        assert_eq!(
            IncrementalSfmConfig::default().pose_guided_merge_max_reprojection_error_px,
            None
        );
        assert_eq!(IncrementalSfmConfig::default().final_min_track_length, None);
    }

    #[test]
    fn pose_guided_track_merge_requires_geometry_and_is_permutation_invariant() {
        let scene = build_scene();
        let defaults = IncrementalSfmConfig::default();
        assert_eq!(
            pose_guided_merge_reprojection_gate(&defaults, 2.0),
            Some(2.0)
        );
        let explicit = IncrementalSfmConfig {
            pose_guided_merge_max_reprojection_error_px: Some(4.0),
            ..defaults.clone()
        };
        assert_eq!(
            pose_guided_merge_reprojection_gate(&explicit, 2.0),
            Some(4.0)
        );
        let invalid = IncrementalSfmConfig {
            pose_guided_merge_max_reprojection_error_px: Some(0.0),
            ..defaults
        };
        assert_eq!(pose_guided_merge_reprojection_gate(&invalid, 2.0), None);
        let point = scene
            .points
            .iter()
            .find(|point| {
                (0..6).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
            })
            .copied()
            .expect("fixture point must be visible in every camera");
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            max_reprojection_error_px: 0.1,
            min_track_length: 2,
            ..IncrementalSfmConfig::default()
        };
        let features = (0..6)
            .map(|image| {
                FeatureSet::new(
                    vec![project(&scene.camera, &scene.poses[image], &point).unwrap()],
                    vec![vec![0.0f32]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        // A verified edge is enough for a complementary pair of tracks.  The
        // union is refit from all four observations, not accepted merely from
        // the edge's two endpoints.
        let fragments = vec![vec![(0, 0), (1, 0)], vec![(2, 0), (3, 0)]];
        let pairwise = vec![PairwiseMatches::new(1, 2, vec![(0, 0)])];
        let (merged, merged_points, merges, tested) = pose_guided_merge_tracks(
            &scene.camera,
            &features,
            &pairwise,
            &poses,
            &fragments,
            &[None, None],
            &config,
        );
        assert_eq!(merges, 1);
        assert_eq!(tested, 1);
        assert_eq!(merged, vec![vec![(0, 0), (1, 0), (2, 0), (3, 0)]]);
        assert!(merged_points[0].is_some());
        assert!(pose_guided_track_reprojection_valid(
            &scene.camera,
            &features,
            &merged[0],
            &poses,
            merged_points[0].as_ref(),
            0.1,
        ));
        assert!(!pose_guided_track_reprojection_valid(
            &scene.camera,
            &features,
            &merged[0],
            &poses,
            merged_points[0].as_ref(),
            0.0,
        ));

        // Two distinct physical points on disjoint image sets can still have
        // a false verified edge.  The all-observation reprojection gate must
        // reject their union even though the pair itself is geometrically
        // valid as an edge.
        let point_b = scene
            .points
            .iter()
            .copied()
            .find(|candidate| {
                *candidate != point
                    && (0..6).all(|image| {
                        project(&scene.camera, &scene.poses[image], candidate).is_some()
                    })
            })
            .expect("fixture needs a second visible point");
        let mixed_features = (0..4)
            .map(|image| {
                let physical = if image < 2 { point } else { point_b };
                FeatureSet::new(
                    vec![project(&scene.camera, &scene.poses[image], &physical).unwrap()],
                    vec![vec![0.0f32]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let (not_merged, _, false_merges, _) = pose_guided_merge_tracks(
            &scene.camera,
            &mixed_features,
            &[PairwiseMatches::new(1, 2, vec![(0, 0)])],
            &poses[..4],
            &fragments,
            &[None, None],
            &config,
        );
        assert_eq!(false_merges, 0);
        assert_eq!(not_merged, fragments);

        // Same-image overlap is rejected before any triangulation, even when
        // a verified edge crosses the two fragments.
        let conflict_tracks = vec![vec![(0, 0), (1, 0)], vec![(1, 1), (2, 0)]];
        let conflict_features = vec![
            FeatureSet::new(
                vec![project(&scene.camera, &scene.poses[0], &point).unwrap()],
                vec![vec![0.0f32]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![
                    project(&scene.camera, &scene.poses[1], &point).unwrap(),
                    project(&scene.camera, &scene.poses[1], &point_b).unwrap(),
                ],
                vec![vec![0.0f32], vec![1.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![project(&scene.camera, &scene.poses[2], &point).unwrap()],
                vec![vec![0.0f32]],
            )
            .unwrap(),
        ];
        let (conflict_result, _, conflict_merges, _) = pose_guided_merge_tracks(
            &scene.camera,
            &conflict_features,
            &[PairwiseMatches::new(0, 2, vec![(0, 0)])],
            &poses[..3],
            &conflict_tracks,
            &[None, None],
            &config,
        );
        assert_eq!(conflict_merges, 0);
        assert_eq!(conflict_result, conflict_tracks);

        // A chain requires recomputing candidates after the first union: the
        // second edge touches the newly created four-view track.
        let chain_tracks = vec![
            vec![(0, 0), (1, 0)],
            vec![(2, 0), (3, 0)],
            vec![(4, 0), (5, 0)],
        ];
        let chain_edges = vec![
            PairwiseMatches::new(1, 2, vec![(0, 0)]),
            PairwiseMatches::new(3, 4, vec![(0, 0)]),
        ];
        let (chain_result, _, chain_merges, _) = pose_guided_merge_tracks(
            &scene.camera,
            &features,
            &chain_edges,
            &poses,
            &chain_tracks,
            &[None, None, None],
            &config,
        );
        assert_eq!(chain_merges, 2);
        assert_eq!(
            chain_result,
            vec![vec![(0, 0), (1, 0), (2, 0), (3, 0), (4, 0), (5, 0)]]
        );

        let mut reversed_tracks = chain_tracks.clone();
        reversed_tracks.reverse();
        let reversed_edges = chain_edges
            .iter()
            .rev()
            .map(|pair| {
                let mut pair = pair.clone();
                pair.matches.reverse();
                pair
            })
            .collect::<Vec<_>>();
        let (reordered_result, _, reordered_merges, _) = pose_guided_merge_tracks(
            &scene.camera,
            &features,
            &reversed_edges,
            &poses,
            &reversed_tracks,
            &[None, None, None],
            &config,
        );
        assert_eq!(chain_result, reordered_result);
        assert_eq!(chain_merges, reordered_merges);
    }

    #[test]
    fn pose_guided_invalid_merge_restores_only_that_fragment_set() {
        let scene = build_scene();
        let point = scene.points[0];
        let point_b = scene
            .points
            .iter()
            .copied()
            .find(|candidate| *candidate != point)
            .expect("fixture needs two points");
        let poses = scene.poses[..4]
            .iter()
            .cloned()
            .map(Some)
            .collect::<Vec<_>>();
        let features = (0..4)
            .map(|image| {
                FeatureSet::new(
                    vec![
                        project(&scene.camera, &scene.poses[image], &point).unwrap(),
                        project(&scene.camera, &scene.poses[image], &point_b).unwrap(),
                    ],
                    vec![vec![0.0f32], vec![1.0]],
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let good_left = vec![(0, 0), (1, 0)];
        let good_right = vec![(2, 0), (3, 0)];
        let bad_left = vec![(0, 1), (1, 1)];
        let bad_right = vec![(2, 1), (3, 1)];
        let good_merged = vec![(0, 0), (1, 0), (2, 0), (3, 0)];
        let bad_merged = vec![(0, 1), (1, 1), (2, 1), (3, 1)];
        let restorations = vec![
            PoseGuidedMergeRestoration {
                source_track_ids: vec![0, 1],
                source_tracks: vec![good_left.clone(), good_right.clone()],
                source_points: vec![Some(point), Some(point)],
                merged_track: good_merged.clone(),
            },
            PoseGuidedMergeRestoration {
                source_track_ids: vec![2, 3],
                source_tracks: vec![bad_left.clone(), bad_right.clone()],
                source_points: vec![Some(point_b), Some(point_b)],
                merged_track: bad_merged.clone(),
            },
        ];
        assert_eq!(
            pose_guided_merge_restorations(
                &[
                    good_left.clone(),
                    good_right.clone(),
                    bad_left.clone(),
                    bad_right.clone(),
                ],
                &[Some(point), Some(point), Some(point_b), Some(point_b)],
                &[good_merged.clone(), bad_merged.clone()],
            ),
            restorations
        );
        let mut tracks = vec![good_merged.clone(), bad_merged.clone()];
        let mut points = vec![Some(point), Some(point)];
        let result = pose_guided_restore_invalid_merges(
            &scene.camera,
            &features,
            &poses,
            &mut tracks,
            &mut points,
            &restorations,
            0.1,
        );
        assert_eq!(result, (2, 1));
        assert_eq!(
            tracks,
            vec![good_merged, bad_left.clone(), bad_right.clone()]
        );
        assert_eq!(points, vec![Some(point), Some(point_b), Some(point_b)]);
        assert!(pose_guided_merge_restorations_reprojection_valid(
            &scene.camera,
            &features,
            &poses,
            &tracks,
            &points,
            &restorations,
            1,
            0.1,
        ));

        // Input traversal order does not affect which merged set is restored
        // or the exact source-fragment order in the final partition.
        let mut reversed_tracks = vec![
            vec![(0, 1), (1, 1), (2, 1), (3, 1)],
            vec![(0, 0), (1, 0), (2, 0), (3, 0)],
        ];
        let mut reversed_points = vec![Some(point), Some(point)];
        let mut reversed_restorations = restorations.clone();
        reversed_restorations.reverse();
        let reversed_result = pose_guided_restore_invalid_merges(
            &scene.camera,
            &features,
            &poses,
            &mut reversed_tracks,
            &mut reversed_points,
            &reversed_restorations,
            0.1,
        );
        assert_eq!(reversed_result, result);
        assert_eq!(reversed_tracks, tracks);
        assert_eq!(reversed_points, points);
    }

    #[test]
    fn pose_guided_split_iteration_guard_covers_identity_improvement_rollback_and_stop() {
        let defaults = IncrementalSfmConfig::default();
        assert_eq!(defaults.pose_guided_track_splitting_iterations, 1);

        // The first pass retains the historical acceptance rule: a denser
        // candidate may have a slightly larger pre-BA mean, provided its own
        // guarded BA lowers that candidate objective.
        let first =
            pose_guided_split_candidate_accepts(0, 38, 38, 100, 120, 120, 0.30, 0.31, 0.30, 1.0);
        assert!(first);

        // A genuinely improving second pass is admitted from the same source
        // components, while a non-improving pass is the deterministic stop.
        let second_improves =
            pose_guided_split_candidate_accepts(1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.28, 1.0);
        assert!(second_improves);
        let second_stops =
            pose_guided_split_candidate_accepts(1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.30, 1.0);
        assert!(!second_stops);

        // A support or registration regression, non-finite objective, and an
        // invalid gate all force rollback. Re-evaluating the same values is
        // pure/deterministic.
        assert!(!pose_guided_split_candidate_accepts(
            1, 38, 37, 120, 125, 125, 0.30, 0.29, 0.28, 1.0,
        ));
        assert!(!pose_guided_split_candidate_accepts(
            1, 38, 38, 120, 125, 119, 0.30, 0.29, 0.28, 1.0,
        ));
        assert!(!pose_guided_split_candidate_accepts(
            1,
            38,
            38,
            120,
            125,
            125,
            0.30,
            f64::NAN,
            0.28,
            1.0,
        ));
        assert!(!pose_guided_split_candidate_accepts(
            1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.28, 0.0,
        ));
        assert_eq!(
            second_improves,
            pose_guided_split_candidate_accepts(1, 38, 38, 120, 125, 125, 0.30, 0.29, 0.28, 1.0,)
        );
    }

    #[test]
    fn pose_guided_composition_snapshots_original_components_and_is_default_off() {
        let tracks = vec![vec![(0, 0), (1, 1)]];
        let conflicts = vec![vec![(0, 2), (1, 3)]];
        let points = vec![Some(Point3::new(1.0, 2.0, 3.0))];
        let source = capture_pose_guided_split_source(true, None, &tracks, &conflicts, &points)
            .expect("enabled composition must capture its source");

        // Simulate geometry recovery appending/replacing state after the
        // snapshot.  The splitter must still receive the original components,
        // not recovered tracks recursively.
        let mut recovered_tracks = tracks.clone();
        recovered_tracks.push(vec![(0, 4), (1, 5), (2, 6)]);
        assert_eq!(source.0, tracks);
        assert_eq!(source.1, conflicts);
        assert_eq!(source.2, points);
        assert_ne!(source.0, recovered_tracks);

        // The ordinary/default path and imported membership diagnostics never
        // allocate a composition snapshot.
        assert!(
            capture_pose_guided_split_source(false, None, &tracks, &conflicts, &points).is_none()
        );
        let membership = vec![vec![(0, 0), (1, 1)]];
        assert!(capture_pose_guided_split_source(
            true,
            Some(&membership),
            &tracks,
            &conflicts,
            &points,
        )
        .is_none());
        assert!(!IncrementalSfmConfig::default().pose_guided_track_splitting);
    }

    #[test]
    fn pose_guided_bridge_cuts_require_two_valid_sides_and_reject_sparse_chain_cuts() {
        let scene = build_scene();
        let (features, _) = render(&scene);
        let visible_points = scene
            .points
            .iter()
            .enumerate()
            .filter(|(_, point)| {
                (0..4).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
            })
            .take(2)
            .map(|(point, _)| point)
            .collect::<Vec<_>>();
        assert_eq!(visible_points.len(), 2);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        let observations_for = |point: usize| {
            (0..4)
                .map(|image| (image, keypoint_for_point(image, point)))
                .collect::<Vec<_>>()
        };
        let edge = |left: TrackObservation, right: TrackObservation| {
            if left <= right {
                (left, right)
            } else {
                (right, left)
            }
        };
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            max_reprojection_error_px: 0.1,
            conflict_recovery_max_hypotheses: 32,
            ..IncrementalSfmConfig::default()
        };
        let first = observations_for(visible_points[0]);
        let second = observations_for(visible_points[1]);
        let mut false_bridge_edges = vec![
            edge(first[0], first[1]),
            edge(first[1], first[2]),
            edge(first[2], first[3]),
            edge(second[0], second[1]),
            edge(second[1], second[2]),
            edge(second[2], second[3]),
            edge(first[0], second[1]),
        ];
        false_bridge_edges.sort_unstable();
        let mut false_bridge_observations = first.clone();
        false_bridge_observations.extend(second.clone());
        let cut = pose_guided_bridge_cut_component(
            &scene.camera,
            &features,
            &poses,
            &false_bridge_observations,
            &false_bridge_edges,
            &config,
        );
        assert_eq!(cut.cut_edges, vec![edge(first[0], second[1])]);
        assert_eq!(cut.cut_sizes, vec![(4, 4)]);
        assert_eq!(cut.components, vec![first.clone(), second.clone()]);

        let mut reversed_edges = false_bridge_edges.clone();
        reversed_edges.reverse();
        let mut reversed_observations = false_bridge_observations.clone();
        reversed_observations.reverse();
        let reordered = pose_guided_bridge_cut_component(
            &scene.camera,
            &features,
            &poses,
            &reversed_observations,
            &reversed_edges,
            &config,
        );
        assert_eq!(cut.components, reordered.components);
        assert_eq!(cut.cut_edges, reordered.cut_edges);
        assert_eq!(cut.cut_sizes, reordered.cut_sizes);

        // A genuine one-point chain has bridge edges, but the complete side
        // still fits one posed point, so no bridge is cut.
        let chain_edges = (0..3)
            .map(|index| edge(first[index], first[index + 1]))
            .collect::<Vec<_>>();
        let chain = pose_guided_bridge_cut_component(
            &scene.camera,
            &features,
            &poses,
            &first,
            &chain_edges,
            &config,
        );
        assert!(chain.cut_edges.is_empty());
        assert_eq!(chain.components, vec![first.clone()]);

        // A singleton leaf is not an eligible side even when the other side
        // is a valid multi-view point.
        let singleton_observation = (4, keypoint_for_point(4, visible_points[1]));
        let mut singleton_observations = first.clone();
        singleton_observations.push(singleton_observation);
        let singleton_edges = chain_edges
            .into_iter()
            .chain(std::iter::once(edge(first[3], singleton_observation)))
            .collect::<Vec<_>>();
        let singleton = pose_guided_bridge_cut_component(
            &scene.camera,
            &features,
            &poses,
            &singleton_observations,
            &singleton_edges,
            &config,
        );
        assert!(singleton.cut_edges.is_empty());
        assert_eq!(singleton.components, vec![singleton_observations]);
    }

    #[test]
    fn final_track_length_gate_removes_only_short_tracks_and_preserves_support() {
        let tracks = vec![
            vec![(0, 0), (1, 0)],
            vec![(0, 1), (1, 1), (2, 1)],
            vec![(0, 2), (1, 2), (2, 2), (3, 2)],
        ];
        let points = vec![
            Some(Point3::new(0.0, 0.0, 1.0)),
            Some(Point3::new(0.1, 0.0, 1.0)),
            Some(Point3::new(0.2, 0.0, 1.0)),
        ];
        let tracks_before = tracks.clone();
        let points_before = points.clone();
        let mut no_op_tracks = tracks.clone();
        let mut no_op_points = points.clone();
        assert_eq!(
            retain_final_track_length(&mut no_op_tracks, &mut no_op_points, 2),
            (0, 0)
        );
        assert_eq!(no_op_tracks, tracks_before);
        assert_eq!(no_op_points, points_before);

        let mut filtered_tracks = tracks;
        let mut filtered_points = points;
        assert_eq!(
            retain_final_track_length(&mut filtered_tracks, &mut filtered_points, 3),
            (1, 2)
        );
        assert_eq!(
            filtered_tracks,
            vec![
                vec![(0, 1), (1, 1), (2, 1)],
                vec![(0, 2), (1, 2), (2, 2), (3, 2)],
            ]
        );
        assert_eq!(filtered_points, vec![points_before[1], points_before[2]]);

        let pose = || Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let poses = vec![Some(pose()), Some(pose()), Some(pose()), Some(pose())];
        assert!(final_track_length_support_is_valid(
            &filtered_tracks,
            &poses,
            &filtered_points,
        ));
        let unsupported = vec![vec![(0, 1), (1, 1), (2, 1)], vec![(0, 2), (1, 2), (2, 2)]];
        let unsupported_points = vec![Some(Point3::new(0.0, 0.0, 1.0)); 2];
        assert!(!final_track_length_support_is_valid(
            &unsupported,
            &poses,
            &unsupported_points,
        ));

        // Repeating the same operation is deterministic and keeps the same
        // point/track pairing, which is the only state the final BA consumes.
        let mut repeat_tracks = tracks_before;
        let mut repeat_points = points_before;
        let first = retain_final_track_length(&mut repeat_tracks, &mut repeat_points, 3);
        let mut repeat_tracks_again = vec![
            vec![(0, 0), (1, 0)],
            vec![(0, 1), (1, 1), (2, 1)],
            vec![(0, 2), (1, 2), (2, 2), (3, 2)],
        ];
        let mut repeat_points_again = vec![
            Some(Point3::new(0.0, 0.0, 1.0)),
            Some(Point3::new(0.1, 0.0, 1.0)),
            Some(Point3::new(0.2, 0.0, 1.0)),
        ];
        let second =
            retain_final_track_length(&mut repeat_tracks_again, &mut repeat_points_again, 3);
        assert_eq!(first, second);
        assert_eq!(repeat_tracks, repeat_tracks_again);
        assert_eq!(repeat_points, repeat_points_again);
    }

    #[test]
    fn pose_guided_graph_support_rejects_single_bridge_and_keeps_two_view_fallback() {
        let scene = build_scene();
        let mut visible_points = scene
            .points
            .iter()
            .filter(|point| {
                (0..4).all(|image| project(&scene.camera, &scene.poses[image], point).is_some())
            })
            .copied();
        let point_a = visible_points
            .next()
            .expect("a point visible in four views");
        let point_b = visible_points
            .next()
            .expect("a second point visible in four views");
        let mut features = Vec::new();
        for image in 0..4 {
            features.push(
                FeatureSet::new(
                    vec![
                        project(&scene.camera, &scene.poses[image], &point_a).unwrap(),
                        project(&scene.camera, &scene.poses[image], &point_b).unwrap(),
                    ],
                    vec![vec![0.0f32], vec![1.0]],
                )
                .unwrap(),
            );
        }

        // Every physical point has two independent supports for every
        // additional view.  One cross-point edge is deliberately present, but
        // it cannot win over a geometrically valid multi-view hypothesis.
        let mut pairwise = Vec::new();
        for image_i in 0..4 {
            for image_j in (image_i + 1)..4 {
                let mut matches = vec![(0, 0), (1, 1)];
                if image_i == 0 && image_j == 1 {
                    matches.push((0, 1));
                }
                pairwise.push(PairwiseMatches::new(image_i, image_j, matches));
            }
        }
        let component = (0..4)
            .flat_map(|image| [(image, 0), (image, 1)])
            .collect::<Vec<_>>();
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            min_track_length: 2,
            max_reprojection_error_px: 4.0,
            conflict_recovery_max_hypotheses: 32,
            pose_guided_track_splitting: true,
            pose_guided_graph_support: true,
            ..IncrementalSfmConfig::default()
        };
        let first = pose_guided_split_tracks(
            &scene.camera,
            &features,
            &pairwise,
            &[],
            std::slice::from_ref(&component),
            &[],
            &poses,
            &config,
        )
        .expect("complete poses should enable graph-supported splitting");
        assert_eq!(first.stats.emitted_tracks, 2);
        assert_eq!(first.stats.graph_supported_tracks, 2);
        assert!(first.stats.graph_support_histogram[2] > 0);
        assert!(first.tracks.iter().all(|track| track
            .iter()
            .filter(|(image, _)| *image == 0)
            .count()
            <= 1));
        assert!(first
            .tracks
            .iter()
            .any(|track| track == &vec![(0, 0), (1, 0), (2, 0), (3, 0)]));
        assert!(first
            .tracks
            .iter()
            .any(|track| track == &vec![(0, 1), (1, 1), (2, 1), (3, 1)]));
        assert!(first
            .tracks
            .iter()
            .all(|track| !track.contains(&(0, 0)) || !track.contains(&(1, 1))));

        // A component with no third-view support retains its genuine two-view
        // fallback rather than being dropped by the admission rule.
        let two_view = pose_guided_split_tracks(
            &scene.camera,
            &features,
            &[PairwiseMatches::new(0, 1, vec![(0, 0)])],
            &[],
            &[vec![(0, 0), (1, 0)]],
            &[],
            &poses,
            &config,
        )
        .expect("two-view fallback should remain deterministic");
        assert_eq!(two_view.tracks, vec![vec![(0, 0), (1, 0)]]);
        assert_eq!(two_view.stats.graph_length_two_tracks, 1);

        let reordered_pairwise = pairwise
            .iter()
            .rev()
            .map(|pair| {
                let mut pair = pair.clone();
                pair.matches.reverse();
                pair
            })
            .collect::<Vec<_>>();
        let mut reversed_component = component.clone();
        reversed_component.reverse();
        let second = pose_guided_split_tracks(
            &scene.camera,
            &features,
            &reordered_pairwise,
            &[],
            std::slice::from_ref(&reversed_component),
            &[],
            &poses,
            &config,
        )
        .expect("graph-supported split should be permutation invariant");
        assert_eq!(first.tracks, second.tracks);
        assert_eq!(first.points, second.points);
        assert_eq!(first.stats, second.stats);
    }

    /// M2 acceptance test, end-to-end: running the *full* `incremental_sfm`
    /// pipeline with [`TrackSource::CorrespondenceGraph`] instead of the
    /// default [`TrackSource::UnionFind`] on the same synthetic scene must
    /// register the same images and produce the same track count and mean
    /// reprojection error — i.e. the track-builder swap is invisible to
    /// every downstream stage (seeding, growth, bundle adjustment).
    #[test]
    fn incremental_sfm_matches_between_track_sources() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);

        let base_config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let union_find_config = IncrementalSfmConfig {
            track_source: TrackSource::UnionFind,
            ..base_config.clone()
        };
        let graph_config = IncrementalSfmConfig {
            track_source: TrackSource::CorrespondenceGraph,
            ..base_config
        };

        let union_find_result =
            incremental_sfm(&scene.camera, &features, &pairwise, &union_find_config)
                .expect("union-find track source must reconstruct this scene");
        let graph_result = incremental_sfm(&scene.camera, &features, &pairwise, &graph_config)
            .expect("CorrespondenceGraph track source must reconstruct this scene");

        assert_eq!(
            union_find_result.registered_images, graph_result.registered_images,
            "both track sources must register the same number of images"
        );
        assert_eq!(
            union_find_result.tracks.len(),
            graph_result.tracks.len(),
            "both track sources must produce the same number of output tracks"
        );
        assert!(
            (union_find_result.mean_reprojection_px - graph_result.mean_reprojection_px).abs()
                < 1.0e-6,
            "both track sources must reach the same mean reprojection error: {} vs {}",
            union_find_result.mean_reprojection_px,
            graph_result.mean_reprojection_px,
        );
    }

    #[test]
    fn seeded_incremental_growth_keeps_supplied_poses_fixed_and_registers_missing_images() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let initial_poses = (0..scene.poses.len())
            .map(|image| (image < 2).then(|| scene.poses[image].clone()))
            .collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            // Keep this test focused on the staged growth phase. The normal
            // final-BA release is covered by the following test.
            final_global_ba: false,
            ba_every: 1,
            ..IncrementalSfmConfig::default()
        };

        let result = incremental_sfm_with_initial_poses(
            &scene.camera,
            &features,
            &pairwise,
            &config,
            Some(&initial_poses),
        )
        .expect("two exact supplied poses must seed the synthetic scene");

        assert!(
            result.registered_images >= 3,
            "the missing-camera PnP loop must grow beyond the supplied seed"
        );
        assert_eq!(result.poses[0], initial_poses[0]);
        assert_eq!(result.poses[1], initial_poses[1]);

        // The public legacy entry point remains the same path as an explicit
        // `None` staged seed. Compare the observable result rather than the
        // internal timing fields so this also guards the default no-op.
        let legacy = incremental_sfm(&scene.camera, &features, &pairwise, &config)
            .expect("legacy synthetic reconstruction must still succeed");
        let explicit_none =
            incremental_sfm_with_initial_poses(&scene.camera, &features, &pairwise, &config, None)
                .expect("explicit None must use the legacy growth path");
        assert_eq!(legacy.poses, explicit_none.poses);
        assert_eq!(legacy.tracks, explicit_none.tracks);
        assert_eq!(legacy.registered_images, explicit_none.registered_images);
        assert_eq!(
            legacy.mean_reprojection_px,
            explicit_none.mean_reprojection_px
        );
    }

    #[test]
    fn seeded_growth_is_deterministic_and_final_ba_is_run_after_fixed_phase() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let initial_poses = (0..scene.poses.len())
            .map(|image| (image < 2).then(|| scene.poses[image].clone()))
            .collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            track_filter_iterations: 0,
            ..IncrementalSfmConfig::default()
        };

        let first = incremental_sfm_with_initial_poses(
            &scene.camera,
            &features,
            &pairwise,
            &config,
            Some(&initial_poses),
        )
        .expect("seeded final refinement must succeed");
        let second = incremental_sfm_with_initial_poses(
            &scene.camera,
            &features,
            &pairwise,
            &config,
            Some(&initial_poses),
        )
        .expect("the same staged input must be repeatable");
        assert_eq!(first.poses, second.poses);
        assert_eq!(first.tracks, second.tracks);
        assert_eq!(first.registered_images, second.registered_images);
        assert_eq!(first.mean_reprojection_px, second.mean_reprojection_px);
        assert!(
            first.ba_result.is_some(),
            "normal final BA must still run after staged growth, releasing non-gauge poses"
        );
        assert!(first.registered_images >= 3);
    }

    #[test]
    fn seeded_incremental_rejects_short_or_nonfinite_pose_vectors() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            final_global_ba: false,
            ..IncrementalSfmConfig::default()
        };
        let short = vec![Some(scene.poses[0].clone())];
        assert!(matches!(
            incremental_sfm_with_initial_poses(
                &scene.camera,
                &features,
                &pairwise,
                &config,
                Some(&short),
            ),
            Err(IncrementalSfmError::InvalidInitialPoses(_))
        ));

        let mut one_seed = vec![None; scene.poses.len()];
        one_seed[0] = Some(scene.poses[0].clone());
        assert!(matches!(
            incremental_sfm_with_initial_poses(
                &scene.camera,
                &features,
                &pairwise,
                &config,
                Some(&one_seed),
            ),
            Err(IncrementalSfmError::InvalidInitialPoses(_))
        ));

        let mut nonfinite = vec![None; scene.poses.len()];
        nonfinite[0] = Some(scene.poses[0].clone());
        nonfinite[1] = Some(Pose::from_world_to_camera(
            UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(f64::NAN, 0.0, 0.0, 1.0)),
            Vector3::zeros(),
        ));
        assert!(matches!(
            incremental_sfm_with_initial_poses(
                &scene.camera,
                &features,
                &pairwise,
                &config,
                Some(&nonfinite),
            ),
            Err(IncrementalSfmError::InvalidInitialPoses(_))
        ));
    }

    /// M2.1 acceptance: `docs/colmap_port_plan.md`'s M2.1 milestone widens
    /// `examples/unordered_sfm_demo.rs`'s verified-pair keep-list so a
    /// `PANORAMIC` (pure-rotation, zero-baseline) pair now reaches
    /// `PairwiseMatches`/this mapper, matching COLMAP's own
    /// `database_cache.cc` `UseInlierMatchesCheck` gate. This must not make
    /// such a pair *seedable*: COLMAP's own
    /// `IncrementalMapperImpl::EstimateInitialTwoViewGeometry` re-derives its
    /// own relative pose and rejects init candidates whose triangulation
    /// angle doesn't clear `init_min_tri_angle`, independent of any stored
    /// `ConfigurationType` — this mapper's [`place_seed_pair`] already has
    /// the same independent architecture (re-estimate the relative pose,
    /// gate on how many inliers actually triangulate), so no new exclusion
    /// mechanism is needed; this test pins that the existing gate covers the
    /// newly-admitted pair type too.
    #[test]
    fn pure_rotation_pair_is_rejected_as_a_seed_even_though_it_now_reaches_pairwise() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        // A scattered point cloud with real depth variation (same shape as
        // `colmap_verification.rs`'s `general_scene_points` fixture).
        let mut points = Vec::new();
        for i in 0..6 {
            for j in 0..4 {
                points.push(Point3::new(
                    -1.5 + 0.6 * i as f64,
                    -1.0 + 0.7 * j as f64,
                    3.0 + 0.8 * ((i + j) % 5) as f64,
                ));
            }
        }

        // Camera 0 at the world origin; camera 1 at the SAME origin, only
        // rotated — a pure-rotation pair, zero baseline, exactly the
        // `PANORAMIC` configuration `TwoViewGeometryVerifier` would classify
        // this as (see `colmap_verification.rs`'s
        // `pure_rotation_classifies_panoramic`).
        let pose0 = Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::zeros());
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.12);
        let pose1 = Pose::from_world_to_camera(yaw, Vector3::zeros());

        let mut kp0 = Vec::new();
        let mut kp1 = Vec::new();
        let mut matches = Vec::new();
        for p in &points {
            if let (Some(px0), Some(px1)) =
                (project(&camera, &pose0, p), project(&camera, &pose1, p))
            {
                matches.push((kp0.len(), kp1.len()));
                kp0.push(px0);
                kp1.push(px1);
            }
        }
        assert!(
            matches.len() >= 15,
            "fixture sanity: pure rotation should still leave most points in both views"
        );

        let features = vec![
            FeatureSet::new(kp0, vec![vec![0.0f32; 4]; matches.len()]).unwrap(),
            FeatureSet::new(kp1, vec![vec![0.0f32; 4]; matches.len()]).unwrap(),
        ];
        let pair = PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches,
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        };
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            ..IncrementalSfmConfig::default()
        };
        let mut poses = vec![None, None];
        assert!(
            !place_seed_pair(&camera, &features, &pair, &config, &mut poses),
            "a zero-baseline (panoramic) pair must never bootstrap a seed, \
             even though M2.1 now lets its correspondences reach PairwiseMatches"
        );
        assert!(
            poses[0].is_none() && poses[1].is_none(),
            "rejected seed must leave poses untouched"
        );
    }

    #[test]
    fn build_tracks_merges_shared_observations() {
        // Two images both see point P (kp 0 in each) and image-2 sees it too.
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let tracks = build_tracks(3, &pairwise, 2);
        assert_eq!(tracks.len(), 1, "the chained matches form one track");
        assert_eq!(tracks[0].len(), 3, "track spans all three images");
    }

    #[test]
    fn track_build_preview_matches_union_find_topology_without_mapping() {
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let features = vec![
            FeatureSet {
                keypoints: vec![Point2::new(0.0, 0.0)],
                descriptors: vec![vec![0.0]],
            };
            3
        ];
        let stats =
            preview_track_build_stats(&features, &pairwise, &IncrementalSfmConfig::default());
        assert_eq!(stats.input_correspondences, 2);
        assert_eq!(stats.connected_components, 1);
        assert_eq!(stats.retained_tracks, 1);
        assert_eq!(stats.retained_observations, 3);
    }

    #[test]
    fn incremental_correspondence_tracks_create_continue_merge_and_are_permutation_invariant() {
        let features = vec![
            FeatureSet {
                keypoints: vec![Point2::new(0.0, 0.0)],
                descriptors: vec![vec![0.0]],
            };
            4
        ];
        let pairwise = vec![
            PairwiseMatches::new(2, 3, vec![(0, 0)]),
            PairwiseMatches::new(1, 2, vec![(0, 0)]),
            PairwiseMatches::new(0, 1, vec![(0, 0)]),
        ];
        let output = build_tracks_incremental_correspondence(&features, &pairwise, 2);
        assert_eq!(output.stats.connected_components, 1);
        assert_eq!(output.stats.retained_tracks, 1);
        assert_eq!(output.tracks, vec![vec![(0, 0), (1, 0), (2, 0), (3, 0)]]);

        let mut permuted = pairwise.clone();
        permuted.reverse();
        permuted[0].matches.reverse();
        let permuted_output = build_tracks_incremental_correspondence(&features, &permuted, 2);
        assert_eq!(output.tracks, permuted_output.tracks);
        assert_eq!(output.stats, permuted_output.stats);
    }

    #[test]
    fn incremental_correspondence_rejects_only_conflicting_edge() {
        let features = vec![
            FeatureSet {
                keypoints: vec![Point2::new(0.0, 0.0), Point2::new(1.0, 0.0)],
                descriptors: vec![vec![0.0], vec![1.0]],
            };
            2
        ];
        let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0), (0, 1)])];
        let output = build_tracks_incremental_correspondence(&features, &pairwise, 2);
        assert_eq!(output.stats.conflicting_components, 1);
        assert_eq!(output.stats.conflicting_observations, 2);
        assert_eq!(output.stats.retained_tracks, 1);
        assert_eq!(output.stats.retained_observations, 2);
    }

    #[test]
    fn correspondence_point_state_enforces_conflicts_and_retriangulates() {
        let mut state = CorrespondencePointState::default();
        let first = state
            .create_point(&[(0, 0), (1, 0)], Point3::new(0.0, 0.0, 1.0))
            .unwrap();
        assert!(state.continue_point(first, (2, 0)));
        assert!(!state.continue_point(first, (1, 1)));
        let second = state
            .create_point(&[(3, 0), (4, 0)], Point3::new(1.0, 0.0, 1.0))
            .unwrap();
        assert!(state.merge_points(first, second, Point3::new(0.5, 0.0, 1.0)));
        let conflicting = state
            .create_point(&[(1, 99), (6, 0)], Point3::new(2.0, 0.0, 1.0))
            .unwrap();
        assert!(!state.merge_points(first, conflicting, Point3::new(0.0, 0.0, 1.0)));
        assert!(state.retriangulate_point(first, Point3::new(0.25, 0.0, 1.0)));
        assert_eq!(state.points[first], Some(Point3::new(0.25, 0.0, 1.0)));
        assert!(!state.retriangulate_point(first, Point3::new(f64::NAN, 0.0, 1.0)));
    }

    #[test]
    fn incremental_correspondence_mode_is_default_noop() {
        let features = vec![
            FeatureSet {
                keypoints: vec![Point2::new(0.0, 0.0)],
                descriptors: vec![vec![0.0]],
            };
            2
        ];
        let pairwise = vec![PairwiseMatches::new(0, 1, vec![(0, 0)])];
        let config = IncrementalSfmConfig::default();
        let default_output = build_track_output(&features, &pairwise, &config, None);
        let legacy_output =
            build_tracks_detailed(features.len(), &pairwise, config.min_track_length);
        assert_eq!(default_output.tracks, legacy_output.tracks);
        assert_eq!(default_output.stats, legacy_output.stats);
        assert!(!config.incremental_correspondence_triangulation);
    }

    #[test]
    fn build_tracks_drops_same_image_conflict() {
        // kp0 and kp1 of image 1 get merged into one component -> inconsistent.
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 1)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let (tracks, stats) = build_tracks_with_stats(2, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
        assert_eq!(stats.input_correspondences, 2);
        assert_eq!(stats.connected_components, 1);
        assert_eq!(stats.conflicting_components, 1);
        assert_eq!(stats.conflicting_observations, 3);
        assert_eq!(stats.retained_tracks, 0);
        assert_eq!(stats.retained_observations, 0);
    }

    #[test]
    fn stable_track_order_is_permutation_invariant_at_coordinate_level() {
        let features = vec![
            FeatureSet::new(
                vec![Point2::new(20.0, 0.0), Point2::new(10.0, 0.0)],
                vec![vec![20.0], vec![10.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(20.0, 1.0), Point2::new(10.0, 1.0)],
                vec![vec![20.0], vec![10.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![Point2::new(20.0, 2.0), Point2::new(10.0, 2.0)],
                vec![vec![20.0], vec![10.0]],
            )
            .unwrap(),
        ];
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
        ];

        let permutations = [[1usize, 0], [1, 0], [1, 0]];
        let permuted_features: Vec<FeatureSet> = features
            .iter()
            .zip(permutations)
            .map(|(set, permutation)| {
                FeatureSet::new(
                    permutation
                        .iter()
                        .map(|&index| set.keypoints[index])
                        .collect(),
                    permutation
                        .iter()
                        .map(|&index| set.descriptors[index].clone())
                        .collect(),
                )
                .unwrap()
            })
            .collect();
        let remapped = pairwise
            .iter()
            .map(|pair| {
                PairwiseMatches::new(
                    pair.image_i,
                    pair.image_j,
                    pair.matches
                        .iter()
                        .map(|&(lhs, rhs)| {
                            (
                                permutations[pair.image_i][lhs],
                                permutations[pair.image_j][rhs],
                            )
                        })
                        .rev()
                        .collect(),
                )
            })
            .rev()
            .collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            stable_track_order: true,
            ..IncrementalSfmConfig::default()
        };
        let original = build_track_output(&features, &pairwise, &config, None).tracks;
        let permuted = build_track_output(&permuted_features, &remapped, &config, None).tracks;
        let physical = |sets: &[FeatureSet], tracks: &[Vec<(usize, usize)>]| {
            let mut output: Vec<Vec<(usize, i64, i64)>> = tracks
                .iter()
                .map(|track| {
                    let mut observations = track
                        .iter()
                        .map(|&(image, index)| {
                            let point = sets[image].keypoints[index];
                            (
                                image,
                                (point.x * 1_000_000.0).round() as i64,
                                (point.y * 1_000_000.0).round() as i64,
                            )
                        })
                        .collect::<Vec<_>>();
                    observations.sort_unstable();
                    observations
                })
                .collect();
            output.sort_unstable();
            output
        };
        assert_eq!(
            physical(&features, &original),
            physical(&permuted_features, &permuted),
            "physical track components must not depend on feature/match order"
        );
    }

    #[test]
    fn confidence_ordered_tracks_keep_strong_multiview_chain() {
        // The weak edge maps image 0's point to the wrong image-1 keypoint.
        // Legacy union-find merges it into the good chain and drops the whole
        // component; confidence ordering accepts the two stronger pair sets
        // first and rejects only that conflicting edge.
        let weak_conflict = PairwiseMatches::new(0, 1, vec![(0, 1)]);
        let strong_01 = PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]);
        let strong_12 = PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]);
        let pairwise = vec![weak_conflict, strong_12, strong_01];

        let legacy = build_tracks(3, &pairwise, 2);
        assert!(legacy.is_empty(), "the weak chain should poison legacy UF");
        let default_config = IncrementalSfmConfig::default();
        assert!(!default_config.confidence_ordered_tracks);
        assert_eq!(
            build_track_output(
                &dummy_features(&[2, 2, 2]),
                &pairwise,
                &default_config,
                None,
            )
            .tracks,
            legacy,
            "the confidence policy must remain opt-in"
        );

        let ordered = build_tracks_confidence_ordered(3, &pairwise, 2);
        assert_eq!(ordered.stats.retained_tracks, 2);
        assert_eq!(ordered.stats.retained_observations, 6);
        assert_eq!(
            ordered.tracks,
            vec![vec![(0, 0), (1, 0), (2, 0)], vec![(0, 1), (1, 1), (2, 1)],]
        );
    }

    #[test]
    fn confidence_ordered_tracks_are_permutation_deterministic() {
        let pairwise = vec![
            PairwiseMatches::new(0, 1, vec![(0, 1)]),
            PairwiseMatches::new(1, 2, vec![(0, 0), (1, 1)]),
            PairwiseMatches::new(0, 1, vec![(0, 0), (1, 1)]),
        ];
        let expected = build_tracks_confidence_ordered(3, &pairwise, 2).tracks;
        for permutation in [vec![2, 0, 1], vec![1, 2, 0], vec![0, 2, 1]] {
            let reordered = permutation
                .into_iter()
                .map(|index| pairwise[index].clone())
                .collect::<Vec<_>>();
            assert_eq!(
                build_tracks_confidence_ordered(3, &reordered, 2).tracks,
                expected
            );
        }
    }

    #[test]
    fn geometry_observation_weights_are_parallax_ordered_and_deterministic() {
        let poses = vec![
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::zeros(),
            )),
            Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(-1.0, 0.0, 0.0),
            )),
        ];
        let tracks = vec![vec![(0, 0), (1, 0)], vec![(0, 1), (1, 1)]];
        let points = vec![
            Some(Point3::new(0.0, 0.0, 100.0)), // weak baseline information
            Some(Point3::new(0.0, 0.0, 2.0)),   // strong baseline information
        ];
        let observations = vec![
            BaObservation {
                keyframe_id: 0,
                landmark_id: 0,
                xy: Point2::origin(),
            },
            BaObservation {
                keyframe_id: 1,
                landmark_id: 0,
                xy: Point2::origin(),
            },
            BaObservation {
                keyframe_id: 0,
                landmark_id: 1,
                xy: Point2::origin(),
            },
            BaObservation {
                keyframe_id: 1,
                landmark_id: 1,
                xy: Point2::origin(),
            },
        ];
        let weights = track_geometry_observation_weights(&poses, &tracks, &points, &observations);
        assert_eq!(weights.len(), observations.len());
        assert!(weights[0] < weights[2]);
        assert_eq!(weights[0], weights[1]);
        assert_eq!(weights[2], weights[3]);
        assert!((0.25..=4.0).contains(&weights[0]));
        assert!((0.25..=4.0).contains(&weights[2]));
        assert_eq!(
            weights,
            track_geometry_observation_weights(&poses, &tracks, &points, &observations)
        );
        assert!(!IncrementalSfmConfig::default().geometry_weighted_ba);
    }

    #[test]
    fn ill_conditioned_landmark_gate_is_deterministic_and_freeze_preserves_point() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let make_pose =
            |centre: Vector3<f64>| Pose::from_world_to_camera(UnitQuaternion::identity(), -centre);
        let poses = vec![
            Some(make_pose(Vector3::zeros())),
            Some(make_pose(Vector3::new(1.0, 0.0, 0.0))),
        ];
        let point = Point3::new(0.0, 0.0, 2.0);
        let pixels: Vec<Point2<f64>> = poses
            .iter()
            .map(|pose| {
                camera
                    .project(&pose.as_ref().unwrap().transform_world_point(&point))
                    .unwrap()
            })
            .collect();
        let features = vec![
            FeatureSet::new(vec![pixels[0]], vec![vec![0.0f32]]).unwrap(),
            FeatureSet::new(vec![pixels[1]], vec![vec![0.0f32]]).unwrap(),
        ];
        let track = vec![(0, 0), (1, 0)];
        let healthy = ba_landmark_geometry(
            &camera,
            &features,
            &poses,
            &track,
            &point,
            &RobustKernel::None,
        );
        assert_eq!(healthy.track_length, 2);
        assert!(healthy.point_condition.is_finite());
        assert!(!ba_landmark_is_ill_conditioned(&healthy, 2.0));
        assert_eq!(
            healthy,
            ba_landmark_geometry(
                &camera,
                &features,
                &poses,
                &track,
                &point,
                &RobustKernel::None,
            )
        );

        let weak_poses = vec![
            Some(make_pose(Vector3::zeros())),
            Some(make_pose(Vector3::new(1.0e-6, 0.0, 0.0))),
        ];
        let weak_pixels: Vec<Point2<f64>> = weak_poses
            .iter()
            .map(|pose| {
                camera
                    .project(&pose.as_ref().unwrap().transform_world_point(&point))
                    .unwrap()
            })
            .collect();
        let weak_features = vec![
            FeatureSet::new(vec![weak_pixels[0]], vec![vec![0.0f32]]).unwrap(),
            FeatureSet::new(vec![weak_pixels[1]], vec![vec![0.0f32]]).unwrap(),
        ];
        let weak = ba_landmark_geometry(
            &camera,
            &weak_features,
            &weak_poses,
            &track,
            &point,
            &RobustKernel::None,
        );
        assert!(weak.point_condition > BA_POINT_BLOCK_MAX_CONDITION);
        assert!(ba_landmark_is_ill_conditioned(&weak, 2.0));
        assert!(!ba_landmark_should_exclude(&weak, 2.0, 4.0));
        let weak_bad = LandmarkBaGeometry {
            median_reprojection_px: 5.0,
            ..weak
        };
        assert!(ba_landmark_should_exclude(&weak_bad, 2.0, 4.0));

        // The freeze is implemented through BundleAdjustment's existing fixed
        // landmark semantics: residual rows remain present, while the point
        // has no Schur variable. A healthy point is free to move instead.
        let initial = Point3::new(0.2, 0.1, 2.2);
        let mut healthy_ba = BundleAdjustment::new(camera.clone());
        for (id, pose) in poses.iter().enumerate() {
            healthy_ba.add_pose(id as u64, pose.as_ref().unwrap().clone());
            healthy_ba.fix_pose(id as u64);
        }
        healthy_ba.add_landmark(0, initial);
        for (id, pixel) in pixels.iter().enumerate() {
            healthy_ba.add_observation(BaObservation {
                keyframe_id: id as u64,
                landmark_id: 0,
                xy: *pixel,
            });
        }
        healthy_ba
            .optimize(&BaConfig {
                robust_kernel: RobustKernel::None,
                ..BaConfig::default()
            })
            .unwrap();
        assert!((healthy_ba.landmarks[&0].coords - initial.coords).norm() > 1.0e-6);

        let mut frozen_ba = healthy_ba.clone();
        frozen_ba.landmarks.insert(0, initial);
        frozen_ba.fix_landmark(0);
        // Leave one pose variable so the solver has a legitimate camera block;
        // the test is about the landmark variable being absent, not an
        // all-fixed empty solve.
        frozen_ba.fixed_poses.remove(&1);
        frozen_ba
            .optimize(&BaConfig {
                robust_kernel: RobustKernel::None,
                ..BaConfig::default()
            })
            .unwrap();
        assert_eq!(frozen_ba.landmarks[&0], initial);
        assert!(!IncrementalSfmConfig::default().freeze_ill_conditioned_landmarks);
    }

    #[test]
    fn landmark_ba_warm_start_is_camera_fixed_monotone_and_deterministic() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let make_pose =
            |centre: Vector3<f64>| Pose::from_world_to_camera(UnitQuaternion::identity(), -centre);
        let poses = vec![
            Some(make_pose(Vector3::zeros())),
            Some(make_pose(Vector3::new(1.0, 0.0, 0.0))),
        ];
        let truth = Point3::new(0.15, -0.08, 2.4);
        let pixels: Vec<Point2<f64>> = poses
            .iter()
            .map(|pose| {
                camera
                    .project(&pose.as_ref().unwrap().transform_world_point(&truth))
                    .unwrap()
            })
            .collect();
        let features = vec![
            FeatureSet::new(vec![pixels[0]], vec![vec![0.0f32]]).unwrap(),
            FeatureSet::new(vec![pixels[1]], vec![vec![0.0f32]]).unwrap(),
        ];
        let tracks = vec![vec![(0, 0), (1, 0)]];
        let initial = Point3::new(0.35, 0.12, 2.8);
        let mut config = IncrementalSfmConfig {
            landmark_ba_warm_start_iterations: 5,
            ..IncrementalSfmConfig::default()
        };
        config.ba_config.robust_kernel = RobustKernel::None;

        let mut points_a = vec![Some(initial)];
        let poses_before = poses.clone();
        let stats_a =
            run_landmark_ba_warm_start(&camera, &features, &tracks, &config, &poses, &mut points_a);
        assert!(stats_a.attempted);
        assert!(stats_a.accepted);
        assert!(stats_a.final_cost < stats_a.initial_cost);
        assert!(stats_a.max_displacement > 0.0);
        assert_eq!(poses, poses_before, "warm start must not mutate cameras");

        let mut points_b = vec![Some(initial)];
        let stats_b =
            run_landmark_ba_warm_start(&camera, &features, &tracks, &config, &poses, &mut points_b);
        assert_eq!(stats_a, stats_b);
        assert_eq!(points_a, points_b);

        let mut untouched = vec![Some(initial)];
        let no_op = run_landmark_ba_warm_start(
            &camera,
            &features,
            &tracks,
            &IncrementalSfmConfig::default(),
            &poses,
            &mut untouched,
        );
        assert!(!no_op.attempted);
        assert!(!no_op.accepted);
        assert_eq!(untouched, vec![Some(initial)]);
    }

    #[test]
    fn normalized_sampson_residual_is_stable_and_rejects_invalid_inputs() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let essential = Matrix3::new(
            0.0, 0.0, 0.0, // translation along x, identity rotation
            0.0, 0.0, -1.0, 0.0, 1.0, 0.0,
        );
        let centre = Point2::new(320.0, 240.0);
        let off_epipolar_line = Point2::new(320.0, 340.0);
        let good = normalized_sampson_residual(&camera, &essential, &centre, &centre)
            .expect("finite E residual");
        let bad = normalized_sampson_residual(&camera, &essential, &centre, &off_epipolar_line)
            .expect("finite off-line residual");
        assert_eq!(good, 0.0);
        assert!(bad > good && bad.is_finite());

        let mut invalid_essential = essential;
        invalid_essential[(0, 0)] = f64::NAN;
        assert!(
            normalized_sampson_residual(&camera, &invalid_essential, &centre, &centre).is_none()
        );
        let invalid_point = Point2::new(f64::NAN, 240.0);
        assert!(
            normalized_sampson_residual(&camera, &essential, &invalid_point, &centre).is_none()
        );
        assert!(
            normalized_sampson_residual(&camera, &Matrix3::zeros(), &centre, &centre,).is_none()
        );
    }

    #[test]
    fn geometric_confidence_prefers_low_residual_and_is_permutation_deterministic() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let centre = Point2::new(320.0, 240.0);
        let off_epipolar_line = Point2::new(320.0, 340.0);
        let features = vec![
            FeatureSet::new(
                vec![centre, centre],
                vec![vec![0.0f32; 2], vec![1.0f32, 0.0]],
            )
            .unwrap(),
            FeatureSet::new(
                vec![centre, off_epipolar_line, off_epipolar_line],
                vec![vec![0.0f32; 2], vec![1.0f32, 0.0], vec![2.0f32, 0.0]],
            )
            .unwrap(),
            FeatureSet::new(vec![centre], vec![vec![0.0f32; 2]]).unwrap(),
        ];
        let essential = Matrix3::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
        // This pair has more matches and therefore wins the old pair-level
        // ordering, but both of its E-supported matches have a larger
        // normalized residual than the single correct edge below.
        let high_support_wrong = PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 1), (1, 2)],
            two_view_config: Some(ConfigurationType::Calibrated),
            essential_matches: Some(vec![(0, 1), (1, 2)]),
            essential_matrix: Some(essential),
        };
        let low_support_correct = PairwiseMatches {
            image_i: 0,
            image_j: 1,
            matches: vec![(0, 0)],
            two_view_config: Some(ConfigurationType::Calibrated),
            essential_matches: Some(vec![(0, 0)]),
            essential_matrix: Some(essential),
        };
        let continuation = PairwiseMatches::new(1, 2, vec![(0, 0)]);
        let pairwise = vec![high_support_wrong, continuation, low_support_correct];

        let legacy = build_tracks_confidence_ordered(3, &pairwise, 2).tracks;
        assert!(!legacy.contains(&vec![(0, 0), (1, 0), (2, 0)]));
        let default_config = IncrementalSfmConfig::default();
        assert!(!default_config.geometric_confidence_tracks);
        assert_eq!(
            build_track_output(&features, &pairwise, &default_config, Some(&camera)).tracks,
            build_tracks(3, &pairwise, 2),
            "the geometric strategy must remain opt-in"
        );
        let geometric = build_tracks_geometric_confidence(&features, &camera, &pairwise, 2);
        assert!(geometric.tracks.contains(&vec![(0, 0), (1, 0), (2, 0)]));

        // Every tie-break after the residual is explicit, so input pair order
        // cannot alter the selected topology.
        let expected = geometric.tracks;
        for permutation in [[2, 1, 0], [1, 0, 2], [0, 2, 1]] {
            let reordered = permutation
                .into_iter()
                .map(|index| pairwise[index].clone())
                .collect::<Vec<_>>();
            assert_eq!(
                build_tracks_geometric_confidence(&features, &camera, &reordered, 2).tracks,
                expected
            );
        }
    }

    #[test]
    fn geometry_recovery_splits_conflict_from_trusted_multiview_poses() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        let kp_0_image_0 = keypoint_for_point(0, 0);
        let kp_1_image_1 = keypoint_for_point(1, 1);
        let pair_0_1 = pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap();
        // One erroneous bridge merges two otherwise-complete six-view tracks.
        pair_0_1.matches.push((kp_0_image_0, kp_1_image_1));

        let built = build_tracks_detailed(features.len(), &pairwise, 2);
        assert_eq!(built.stats.conflicting_components, 1);
        assert_eq!(built.conflicting_components.len(), 1);

        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let config = IncrementalSfmConfig {
            geometry_guided_conflict_recovery: true,
            conflict_recovery_min_views: 3,
            conflict_recovery_max_hypotheses: 32,
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            ..IncrementalSfmConfig::default()
        };
        let recovered = recover_conflict_tracks_geometry(
            &scene.camera,
            &features,
            &pairwise,
            &built.conflicting_components,
            &poses,
            &config,
        );
        assert_eq!(
            recovered.len(),
            1,
            "first slice keeps one guarded hypothesis"
        );
        let track = &recovered[0];
        assert!(track.registered_observations >= 3);
        assert!(track.mean_reprojection_px < 1e-6);
        let unique_images: HashSet<_> =
            track.observations.iter().map(|&(image, _)| image).collect();
        assert_eq!(unique_images.len(), track.observations.len());
        let nearest_truth = scene
            .points
            .iter()
            .take(2)
            .map(|point| (track.point - point).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(
            nearest_truth < 1e-6,
            "recovered point error {nearest_truth}"
        );
    }

    #[test]
    fn geometry_recovery_rejects_three_view_chain_without_cycle() {
        let scene = build_scene();
        let (features, _) = render(&scene);
        let observation = |image: usize| {
            let kp = features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == 0.0)
                .unwrap();
            (image, kp)
        };
        let component = vec![observation(0), observation(1), observation(2)];
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(component[0].1, component[1].1)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(component[1].1, component[2].1)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let recovered = recover_conflict_tracks_geometry(
            &scene.camera,
            &features,
            &pairwise,
            &[component],
            &poses,
            &IncrementalSfmConfig {
                conflict_recovery_max_reprojection_error_px: 0.1,
                conflict_recovery_max_mean_reprojection_px: 0.05,
                ..IncrementalSfmConfig::default()
            },
        );
        assert!(
            recovered.is_empty(),
            "a tree is not independent multi-view evidence"
        );
    }

    #[test]
    fn incremental_sfm_admits_geometry_recovery_only_after_clean_model() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        // Leave image 5 outside the verified component so the clean model is
        // intentionally incomplete and recovery may exercise its guarded BA.
        pairwise.retain(|pair| pair.image_i != 5 && pair.image_j != 5);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap()
            .matches
            .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));

        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            geometry_guided_conflict_recovery: true,
            conflict_recovery_max_reprojection_error_px: 0.1,
            conflict_recovery_max_mean_reprojection_px: 0.05,
            // Noise-free BA can move at floating-point epsilon around zero.
            conflict_recovery_max_clean_error_increase_ratio: 0.01,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert_eq!(result.track_build_stats.conflicting_components, 1);
        assert_eq!(result.geometry_recovered_tracks, 1);
        assert!(result.geometry_recovered_observations >= 3);
        assert!(result.geometry_recovery_pose_ba_applied);
        assert_eq!(result.registered_images, 5);
        assert!(result.mean_reprojection_px < 0.1);
    }

    #[test]
    fn complete_model_geometry_recovery_keeps_poses_byte_identical() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap()
            .matches
            .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));
        let base = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let control = incremental_sfm(&scene.camera, &features, &pairwise, &base).unwrap();
        let recovered = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                geometry_guided_conflict_recovery: true,
                conflict_recovery_max_reprojection_error_px: 0.1,
                conflict_recovery_max_mean_reprojection_px: 0.05,
                ..base
            },
        )
        .unwrap();
        assert_eq!(control.registered_images, features.len());
        assert_eq!(recovered.registered_images, features.len());
        assert_eq!(recovered.geometry_recovered_tracks, 1);
        assert!(!recovered.geometry_recovery_pose_ba_applied);
        assert_eq!(recovered.poses, control.poses);
        assert_eq!(recovered.tracks.len(), control.tracks.len() + 1);
    }

    #[test]
    fn rejected_geometry_recovery_rolls_back_byte_identical_clean_model() {
        let scene = build_scene();
        let (features, mut pairwise) = render(&scene);
        let keypoint_for_point = |image: usize, point: usize| {
            features[image]
                .descriptors
                .iter()
                .position(|descriptor| descriptor[0] == point as f32)
                .unwrap()
        };
        pairwise
            .iter_mut()
            .find(|pair| pair.image_i == 0 && pair.image_j == 1)
            .unwrap()
            .matches
            .push((keypoint_for_point(0, 0), keypoint_for_point(1, 1)));

        let base = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let control = incremental_sfm(&scene.camera, &features, &pairwise, &base).unwrap();
        let rejected = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                geometry_guided_conflict_recovery: true,
                conflict_recovery_max_reprojection_error_px: 0.1,
                // Force the post-BA acceptance gate to reject every proposal.
                conflict_recovery_max_mean_reprojection_px: -1.0,
                ..base
            },
        )
        .unwrap();
        assert_eq!(rejected.geometry_recovered_tracks, 0);
        assert_eq!(rejected.geometry_recovered_observations, 0);
        assert_eq!(rejected.poses, control.poses);
        assert_eq!(rejected.tracks, control.tracks);
        assert_eq!(rejected.registered_images, control.registered_images);
        assert_eq!(rejected.mean_reprojection_px, control.mean_reprojection_px);
    }

    /// Minimal `FeatureSet`s with `kp_counts[i]` dummy keypoints per image —
    /// enough for `build_tracks_via_graph` to declare each image's point2D
    /// capacity; keypoint/descriptor content is irrelevant to track building.
    fn dummy_features(kp_counts: &[usize]) -> Vec<FeatureSet> {
        kp_counts
            .iter()
            .map(|&n| {
                let kps = vec![Point2::new(0.0, 0.0); n];
                let descs = vec![vec![0.0f32; 4]; n];
                FeatureSet::new(kps, descs).unwrap()
            })
            .collect()
    }

    /// M2: the [`TrackSource::CorrespondenceGraph`] path reproduces
    /// [`build_tracks_merges_shared_observations`] exactly.
    #[test]
    fn graph_tracks_merges_shared_observations() {
        let features = dummy_features(&[1, 1, 1]);
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 2,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert_eq!(tracks.len(), 1, "the chained matches form one track");
        assert_eq!(tracks[0].len(), 3, "track spans all three images");
    }

    /// M2: the [`TrackSource::CorrespondenceGraph`] path reproduces
    /// [`build_tracks_drops_same_image_conflict`] exactly — including the
    /// repeated-pair-entry input shape that exercises this function's
    /// pre-merge step (see `build_tracks_via_graph`'s doc).
    #[test]
    fn graph_tracks_drops_same_image_conflict() {
        let features = dummy_features(&[2, 2]);
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 1)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
    }

    /// M2 acceptance bar: on a repeated-pair input in *swapped* direction
    /// (`(1, 0)` instead of `(0, 1)`), the graph path's pre-merge
    /// canonicalization must still see both entries as the same unordered
    /// pair and produce the identical conflict-drop as
    /// [`graph_tracks_drops_same_image_conflict`] — proving the merge step
    /// doesn't silently drop the second entry via `DuplicatePair`.
    #[test]
    fn graph_tracks_drops_same_image_conflict_with_swapped_pair_direction() {
        let features = dummy_features(&[2, 2]);
        let pairwise = vec![
            PairwiseMatches {
                image_i: 0,
                image_j: 1,
                matches: vec![(0, 0)],
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
            PairwiseMatches {
                image_i: 1,
                image_j: 0,
                matches: vec![(1, 0)], // (kp1 in image1, kp0 in image0),
                two_view_config: None,
                essential_matches: None,
                essential_matrix: None,
            },
        ];
        let tracks = build_tracks_via_graph(&features, &pairwise, 2);
        assert!(tracks.is_empty(), "same-image conflict track is dropped");
    }

    #[test]
    fn visibility_pyramid_prefers_distribution_over_count() {
        assert_eq!(
            NextImagePolicy::default(),
            NextImagePolicy::CorrespondenceCount
        );
        // A small cluster of MANY points in one corner versus FEWER points spread
        // across the frame: the COLMAP visibility score must rank the spread set
        // higher (better-conditioned PnP), even though it has fewer correspondences.
        let (w, h) = (640u32, 480u32);
        let clustered: Vec<Point2<f64>> = (0..50)
            .map(|i| Point2::new(2.0 + (i % 5) as f64, 2.0 + (i / 5) as f64))
            .collect();
        let spread: Vec<Point2<f64>> = (0..5)
            .flat_map(|gy| {
                (0..4).map(move |gx| {
                    Point2::new(
                        (gx as f64 + 0.5) * w as f64 / 4.0,
                        (gy as f64 + 0.5) * h as f64 / 5.0,
                    )
                })
            })
            .collect();
        let clustered_score = visibility_pyramid_score(w, h, clustered.iter().copied());
        let spread_score = visibility_pyramid_score(w, h, spread.iter().copied());
        assert!(
            spread_score > clustered_score,
            "spread ({spread_score}, {} pts) should beat clustered ({clustered_score}, {} pts)",
            spread.len(),
            clustered.len(),
        );
        // The 50 clustered points collapse onto a handful of cells (occupancy
        // saturates), unlike a raw count which would have ranked them first.
        assert!(
            clustered_score < clustered.len(),
            "clustered occupancy {clustered_score} must saturate below the point count"
        );

        let camera = Camera::pinhole(0, w, h, 500.0, 500.0, 320.0, 240.0);
        let to_corrs = |points: &[Point2<f64>]| {
            points
                .iter()
                .copied()
                .map(|point2d| Correspondence2D3D {
                    point2d,
                    point3d: Point3::new(0.0, 0.0, 5.0),
                    confidence: None,
                })
                .collect::<Vec<_>>()
        };
        let clustered_corrs = to_corrs(&clustered);
        let spread_corrs = to_corrs(&spread);
        assert!(
            next_image_rank(&camera, NextImagePolicy::VisibilityPyramid, &spread_corrs,)
                > next_image_rank(
                    &camera,
                    NextImagePolicy::VisibilityPyramid,
                    &clustered_corrs,
                ),
            "visibility policy must prefer coverage"
        );
        assert!(
            next_image_rank(
                &camera,
                NextImagePolicy::CorrespondenceCount,
                &clustered_corrs,
            ) > next_image_rank(&camera, NextImagePolicy::CorrespondenceCount, &spread_corrs,),
            "count policy must reproduce the legacy ordering"
        );
    }

    #[test]
    fn auto_compares_count_for_every_incomplete_visibility_candidate() {
        assert!(next_image_auto_count_candidate_is_needed(89, 100));
        assert!(next_image_auto_count_candidate_is_needed(90, 100));
        assert!(next_image_auto_count_candidate_is_needed(9, 10));
        assert!(!next_image_auto_count_candidate_is_needed(10, 10));
        assert!(!next_image_auto_count_candidate_is_needed(0, 0));

        assert!(next_image_auto_post_candidate_is_needed(9, 10));
        assert!(!next_image_auto_post_candidate_is_needed(10, 10));

        let visibility = NextImageAutoMetrics {
            registered_images: 17,
            valid_observations: 100,
            tracks: 20,
            mean_reprojection_px: 2.0,
        };
        let count = NextImageAutoMetrics {
            registered_images: 18,
            valid_observations: 1,
            tracks: 1,
            mean_reprojection_px: 100.0,
        };
        assert!(next_image_auto_metrics_are_better(count, visibility));
    }

    #[test]
    fn auto_post_completion_requires_strict_registration_gain() {
        let mut baseline = IncrementalSfmResult {
            poses: Vec::new(),
            tracks: Vec::new(),
            track_build_stats: TrackBuildStats::default(),
            registered_images: 9,
            post_refinement_registered_images: 0,
            structureless_registered_images: 0,
            geometry_recovered_tracks: 0,
            geometry_recovered_observations: 0,
            geometry_recovery_pose_ba_applied: false,
            mean_reprojection_px: 1.0,
            ba_result: None,
            refined_camera: None,
            seed_image_i: 0,
            seed_image_j: 1,
            seed_match_count: 0,
        };
        let mut candidate = baseline.clone();
        candidate.registered_images = 10;
        assert!(next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));

        // A registration gain must not be allowed to replace a materially
        // cleaner incumbent with a finite but much worse post candidate.
        candidate.mean_reprojection_px = 100.0;
        assert!(!next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));
        candidate.mean_reprojection_px = 1.0;
        assert!(next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));

        // A post pass may improve reprojection or change tracks while leaving
        // registration unchanged. Auto must retain the untouched candidate in
        // that case, so the primary model's bytes/trajectory are stable.
        candidate.registered_images = baseline.registered_images;
        candidate.mean_reprojection_px = 0.1;
        assert!(!next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));

        candidate.registered_images = baseline.registered_images.saturating_sub(1);
        assert!(!next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));
        candidate.registered_images = baseline.registered_images + 1;
        candidate.mean_reprojection_px = f64::NAN;
        assert!(!next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));

        // A finite post candidate can repair an incumbent whose aggregate
        // reprojection metric is unavailable, while a non-finite candidate
        // can never be adopted.
        baseline.mean_reprojection_px = f64::NAN;
        candidate.mean_reprojection_px = 0.5;
        assert!(next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));
        baseline.registered_images = 0;
        candidate.registered_images = 1;
        assert!(next_image_auto_post_candidate_is_better(
            &candidate, &baseline
        ));
    }

    #[test]
    fn auto_selection_is_lexicographic_and_ties_keep_visibility() {
        let visibility = NextImageAutoMetrics {
            registered_images: 17,
            valid_observations: 100,
            tracks: 20,
            mean_reprojection_px: 2.0,
        };
        let more_observations = NextImageAutoMetrics {
            valid_observations: 101,
            ..visibility
        };
        assert!(next_image_auto_metrics_are_better(
            more_observations,
            visibility
        ));

        let more_tracks = NextImageAutoMetrics {
            tracks: 21,
            ..visibility
        };
        assert!(next_image_auto_metrics_are_better(more_tracks, visibility));

        let lower_reprojection = NextImageAutoMetrics {
            mean_reprojection_px: 1.0,
            ..visibility
        };
        assert!(next_image_auto_metrics_are_better(
            lower_reprojection,
            visibility
        ));
        assert!(!next_image_auto_metrics_are_better(visibility, visibility));

        let nonfinite = NextImageAutoMetrics {
            mean_reprojection_px: f64::NAN,
            ..visibility
        };
        assert!(!next_image_auto_metrics_are_better(nonfinite, visibility));
    }

    #[test]
    fn reconstructs_synthetic_ring_scene() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        assert!(pairwise.len() >= 5, "expected an overlapping view graph");

        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();

        // Most images register and most points triangulate.
        assert!(
            result.registered_images >= 5,
            "registered only {}",
            result.registered_images
        );
        assert!(
            result.tracks.len() >= 20,
            "triangulated only {} tracks",
            result.tracks.len()
        );
        // Reprojection is tight (synthetic, noise-free).
        assert!(
            result.mean_reprojection_px < 1.0,
            "mean reprojection {} px too high",
            result.mean_reprojection_px
        );

        // The reconstruction is correct up to a similarity transform. Check the
        // recovered camera-center geometry matches GT up to scale by comparing
        // pairwise center-distance ratios between two registered images.
        let registered: Vec<usize> = (0..scene.poses.len())
            .filter(|&i| result.poses[i].is_some())
            .collect();
        assert!(registered.len() >= 3);
        let center = |i: usize| {
            result.poses[i]
                .as_ref()
                .unwrap()
                .camera_to_world()
                .translation
        };
        let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
        let (a, b, c) = (registered[0], registered[1], registered[2]);
        let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
        let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
        assert!(
            (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
            "camera-spacing ratio {est_ratio} != GT {gt_ratio} (similarity-invariant)"
        );
    }

    #[test]
    fn auto_policy_runs_through_public_incremental_api() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            next_image_policy: NextImagePolicy::Auto,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert!(
            result.registered_images >= 5,
            "Auto policy registered only {} synthetic images",
            result.registered_images
        );
    }

    /// Look-at world→camera poses on an arc of `n` cameras at `radius` from
    /// `target`, spanning `span` radians (so neighbours keep a real baseline).
    fn arc_cameras(n: usize, target: Point3<f64>, radius: f64, span: f64) -> Vec<Pose> {
        let mut poses = Vec::new();
        let denom = (n.max(2) - 1) as f64;
        for k in 0..n {
            let angle = -span / 2.0 + span * (k as f64) / denom;
            let cam_center =
                target + Vector3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            let forward = (target - cam_center).normalize();
            let right = forward.cross(&Vector3::new(0.0, 1.0, 0.0)).normalize();
            let up = right.cross(&forward);
            let r_c2w = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let q_c2w = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r_c2w),
            );
            let q_w2c = q_c2w.inverse();
            let t_w2c = -(q_w2c * cam_center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }
        poses
    }

    /// A scene with two geometrically disjoint components: a small dense "trap"
    /// cluster (3 cameras, ~100 co-visible points → the *strongest-match* pairs in
    /// the whole graph) far to one side, and a larger "main" component (8 cameras
    /// over a grid). The trap's frustums never see the main grid and vice versa,
    /// so they form two connected components; the strongest seed reconstructs only
    /// the 3-camera trap, and recovering the main component needs the multi-seed
    /// search to look past it. Cameras: indices 0..3 trap, 3..11 main.
    fn build_two_component_scene() -> Scene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        // Trap cluster: a dense cube at the origin (every trap camera sees all of
        // it, so each trap pair carries the most matches).
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in -2..=1 {
                    points.push(Point3::new(
                        xi as f64 * 0.2,
                        yi as f64 * 0.2,
                        zi as f64 * 0.2,
                    ));
                }
            }
        }
        // Main grid: a separate, larger structure offset far along +x.
        for xi in -2..=2 {
            for yi in -2..=2 {
                for zi in 0..=2 {
                    points.push(Point3::new(
                        20.0 + xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.3,
                    ));
                }
            }
        }
        let mut poses = arc_cameras(3, Point3::origin(), 3.0, 0.5);
        poses.extend(arc_cameras(8, Point3::new(20.0, 0.0, 0.0), 3.0, 1.2));
        Scene {
            camera,
            points,
            poses,
        }
    }

    #[test]
    fn multi_seed_escapes_strongest_isolated_cluster() {
        let scene = build_two_component_scene();
        let (features, pairwise) = render(&scene);

        // The strongest-match pair is inside the 3-camera trap.
        let strongest = pairwise
            .iter()
            .max_by_key(|p| p.matches.len())
            .expect("a view graph");
        assert!(
            strongest.image_i < 3 && strongest.image_j < 3,
            "expected the densest pair to be inside the trap cluster, got ({},{})",
            strongest.image_i,
            strongest.image_j
        );

        // One trial commits to that strongest seed and is trapped in the cluster.
        let trapped = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                min_seed_matches: 8,
                min_pnp_inliers: 6,
                seed_trials: 1,
                ..IncrementalSfmConfig::default()
            },
        )
        .unwrap();
        assert!(
            trapped.registered_images <= 3,
            "single-seed should be stuck in the 3-camera trap, got {}",
            trapped.registered_images
        );

        // The multi-seed search looks past the trap and recovers the 8-camera
        // main component instead.
        let escaped = incremental_sfm(
            &scene.camera,
            &features,
            &pairwise,
            &IncrementalSfmConfig {
                min_seed_matches: 8,
                min_pnp_inliers: 6,
                ..IncrementalSfmConfig::default() // seed_trials = 12
            },
        )
        .unwrap();
        assert!(
            escaped.registered_images >= 7,
            "multi-seed should recover the 8-camera main component, got {}",
            escaped.registered_images
        );
    }

    type OutlierTrackFixture = (
        Camera,
        Vec<FeatureSet>,
        Vec<Option<Pose>>,
        Vec<Vec<(usize, usize)>>,
        Vec<Option<Point3<f64>>>,
    );

    /// Build three views (identity rotation, small lateral offsets) of one world
    /// point, with `outlier_views` images observing it at a planted off-by-50px
    /// outlier keypoint instead of the true projection.
    fn outlier_track_fixture(outlier_views: &[usize]) -> OutlierTrackFixture {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.1, -0.2, 5.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.3 - 0.3, 0.0, 0.0),
            );
            let mut px = camera.project(&pose.transform_world_point(&point)).unwrap();
            if outlier_views.contains(&k) {
                px += Vector3::new(50.0, 50.0, 0.0).xy();
            }
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let track_point = vec![Some(point)];
        (camera, features, poses, tracks, track_point)
    }

    #[test]
    fn filter_strips_single_outlier_observation_keeps_track() {
        let (camera, features, poses, mut tracks, mut track_point) = outlier_track_fixture(&[2]);
        let config = IncrementalSfmConfig::default();
        let removed = filter_outlier_observations(
            &camera,
            &features,
            &mut tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(removed, 1, "the planted outlier observation is removed");
        assert_eq!(
            tracks[0],
            vec![(0, 0), (1, 0)],
            "only the two inliers remain"
        );
        assert!(track_point[0].is_some(), "track survives with >= 2 inliers");
    }

    #[test]
    fn filter_drops_low_parallax_far_point() {
        // A point 500 units away, seen by three cameras 0.6 units apart, projects
        // with ZERO reprojection error (perfect) yet has ~0.07 deg parallax — the
        // depth-ambiguous far-flung outlier the reprojection test cannot catch.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.0, 0.0, 500.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.3 - 0.3, 0.0, 0.0),
            );
            let px = camera.project(&pose.transform_world_point(&point)).unwrap();
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let mut tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let mut track_point = vec![Some(point)];
        let config = IncrementalSfmConfig::default(); // min_triangulation_angle_deg = 2.0
        let changed = filter_outlier_observations(
            &camera,
            &features,
            &mut tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(changed, 1, "the low-parallax track is dropped");
        assert!(
            track_point[0].is_none(),
            "depth-ambiguous far point dropped despite zero reprojection error"
        );
    }

    #[test]
    fn filter_drops_track_below_min_observations() {
        // Two of three views are outliers -> a single inlier left -> drop track.
        let (camera, features, poses, mut tracks, mut track_point) = outlier_track_fixture(&[1, 2]);
        let config = IncrementalSfmConfig::default();
        let removed = filter_outlier_observations(
            &camera,
            &features,
            &mut tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(removed, 3, "2 observations stripped + 1 track dropped");
        assert!(
            track_point[0].is_none(),
            "track with < 2 inlier observations is dropped"
        );
    }

    #[test]
    fn filter_images_deregisters_unsupported_pose_and_protects_seed() {
        // Register all six ring cameras with their true poses, triangulate, then
        // corrupt one non-seed image's pose so none of its observations reproject.
        // FilterImages must de-register exactly that image, keep the well-supported
        // ones, and never touch the two seed (lowest-index) images.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let config = IncrementalSfmConfig {
            filter_images: true,
            filter_min_image_observations: 5,
            ..IncrementalSfmConfig::default()
        };
        let mut poses: Vec<Option<Pose>> = scene.poses.iter().map(|p| Some(p.clone())).collect();
        let mut track_point = vec![None; tracks.len()];
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );

        // Corrupt image 3 (not in the protected seed pair): aim it away from the
        // cloud so every observation reprojects far off or behind the camera.
        let bad = Pose::from_world_to_camera(
            UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::PI),
            Vector3::new(0.0, 0.0, 0.0),
        );
        poses[3] = Some(bad);

        let removed = filter_images(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses,
            &track_point,
        );
        assert_eq!(removed, 1, "only the unsupported image is de-registered");
        assert!(poses[3].is_none(), "the corrupted-pose image is filtered");
        assert!(
            poses[0].is_some() && poses[1].is_some(),
            "the seed pair is protected from filtering"
        );
        assert!(
            poses[2].is_some() && poses[4].is_some() && poses[5].is_some(),
            "well-supported images stay registered"
        );
    }

    #[test]
    fn retriangulate_completes_untriangulated_track() {
        // Three identity-rotation views with a real lateral baseline see one
        // world point. The track exists in the union-find but was never given a
        // 3D point (it failed the parallax gate at growth time, say). With the
        // poses now fixed, re-triangulation must complete it.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.1, -0.2, 5.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.5 - 0.5, 0.0, 0.0),
            );
            let px = camera.project(&pose.transform_world_point(&point)).unwrap();
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let mut track_point: Vec<Option<Point3<f64>>> = vec![None]; // not yet triangulated
        let config = IncrementalSfmConfig::default();
        let changed = retriangulate_tracks(
            &camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(changed, 1, "the un-triangulated track is completed");
        let p = track_point[0].expect("track now has a 3D point");
        assert!(
            (p - point).norm() < 1e-6,
            "re-triangulated point {p:?} should recover the true point {point:?}"
        );
    }

    #[test]
    fn retriangulate_guarded_swap_replaces_noisy_point_only_when_better() {
        // Same three-view geometry, but the track already carries a *noisy* point
        // displaced far along the depth ray. Re-triangulation from the true
        // observations fits them better, so the guarded swap must replace it; a
        // second pass (now exact) must be a no-op (never regress an exact point).
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let point = Point3::new(0.1, -0.2, 5.0);
        let mut features = Vec::new();
        let mut poses = Vec::new();
        for k in 0..3 {
            let pose = Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(k as f64 * 0.5 - 0.5, 0.0, 0.0),
            );
            let px = camera.project(&pose.transform_world_point(&point)).unwrap();
            features.push(FeatureSet::new(vec![px], vec![vec![k as f32, 1.0]]).unwrap());
            poses.push(Some(pose));
        }
        let tracks = vec![vec![(0, 0), (1, 0), (2, 0)]];
        let noisy = Point3::new(0.3, -0.6, 8.0);
        let mut track_point = vec![Some(noisy)];
        let config = IncrementalSfmConfig::default();

        let changed = retriangulate_tracks(
            &camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(changed, 1, "the noisy point is replaced by a better fit");
        let p = track_point[0].unwrap();
        assert!(
            (p - point).norm() < 1e-6,
            "guarded swap should land on the true point, got {p:?}"
        );

        // Re-running on the now-exact point changes nothing.
        let again = retriangulate_tracks(
            &camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
        );
        assert_eq!(again, 0, "an already-exact point must not be regressed");
    }

    #[test]
    fn colmap_style_mapper_reconstructs_ring_scene() {
        // The COLMAP schedule (per-registration local BA + growth-triggered
        // iterative global refinement + registration retries) must reconstruct the
        // synthetic ring at least as completely as the simple schedule, with tight
        // reprojection and a similarity-correct camera geometry.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            colmap_style_mapper: true,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert!(
            result.registered_images >= 5,
            "registered only {}",
            result.registered_images
        );
        assert!(
            result.tracks.len() >= 20,
            "triangulated only {} tracks",
            result.tracks.len()
        );
        assert!(
            result.mean_reprojection_px < 1.0,
            "mean reprojection {} px too high",
            result.mean_reprojection_px
        );
        let registered: Vec<usize> = (0..scene.poses.len())
            .filter(|&i| result.poses[i].is_some())
            .collect();
        let center = |i: usize| {
            result.poses[i]
                .as_ref()
                .unwrap()
                .camera_to_world()
                .translation
        };
        let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
        let (a, b, c) = (registered[0], registered[1], registered[2]);
        let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
        let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
        assert!(
            (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
            "camera-spacing ratio {est_ratio} != GT {gt_ratio} (similarity-invariant)"
        );
    }

    #[test]
    fn colmap_style_co_evolves_intrinsics_toward_truth() {
        // The synthetic ring is observable geometry, so a focal error is
        // recoverable. Render with the TRUE camera (fx=fy=500) but reconstruct from
        // a WRONG horizontal focal (fx=530). The arc moves the cameras only in the
        // x-z plane, so the *horizontal* focal fx is well constrained by the
        // azimuthal parallax (fy would need elevation change — exercised instead by
        // the anisotropic South-Building benchmark). The joint solve must pull fx
        // substantially back toward 500 — the COLMAP self-calibration formulation
        // (intrinsics co-estimated inside the Schur camera system, using the coupled
        // landmark-eliminated gradient), which a final-only alternating refinement
        // against converged structure cannot do. The orthogonal, un-perturbed
        // vertical axis (fy, cy) must stay fixed. The horizontal principal point cx
        // is allowed to co-adjust: on a pure look-at arc fx and cx are only *jointly*
        // constrained (the focal/principal-point ambiguity), so the joint solve
        // legitimately distributes the correction across both — this confound is
        // absent on the richer South-Building viewpoints, where cx stays put.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let wrong = Camera::pinhole(0, 640, 480, 530.0, 500.0, 320.0, 240.0);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            colmap_style_mapper: true,
            refine_intrinsics: true,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&wrong, &features, &pairwise, &config).unwrap();
        let cam = result
            .refined_camera
            .expect("refine_intrinsics returns the refined camera");
        let (fx, fy, cx, cy) = cam.intrinsics().unwrap();
        eprintln!("joint-refined intrinsics: fx {fx} fy {fy} cx {cx} cy {cy}");
        // fx recovers at least a third of its injected error (530 - 500 = 30).
        assert!(
            fx < 530.0 - 0.33 * 30.0,
            "fx {fx} should recover substantially toward 500 from 530"
        );
        // The orthogonal vertical axis was not perturbed and must not drift.
        assert!((fy - 500.0).abs() < 2.0, "fy {fy} drifted from 500");
        assert!((cy - 240.0).abs() < 2.0, "cy {cy} drifted from 240");
        // cx co-adjusts with fx within the look-at arc's focal/centre ambiguity, but
        // must stay sane (no blow-up).
        assert!((cx - 320.0).abs() < 20.0, "cx {cx} blew up from 320");
    }

    #[test]
    fn colmap_style_mapper_retries_a_filtered_image_up_to_its_trial_budget_then_gives_up() {
        // M4 (`docs/colmap_port_plan.md`'s "M4 results"): the growth loop's
        // stall-triggered recovery must give a `filter_images`-demoted image
        // genuine retry attempts across multiple growth stalls — not filter it
        // once and abandon it, the pre-M4 behaviour, since pre-M4
        // `growth_global_refinement` (and the `filter_images` call inside it)
        // only ever ran on the growth-*ratio* trigger, never on a stall — while
        // still terminating cleanly once `max_registration_trials` is spent,
        // rather than cycling forever. `global_ba_images_ratio` is set absurdly
        // high so the *only* thing that can ever invoke `growth_global_refinement`
        // / `filter_images` in this test is the stall path, isolating exactly
        // the mechanism this milestone added.
        //
        // Scene: 4 cameras looking at the same 40-point cloud (two z-layers so
        // the essential-matrix seed estimator sees a non-degenerate, non-planar
        // point set). The seed pair (0, 1) and camera 3 all see all 40 points;
        // camera 2 is built (by construction, not by frustum geometry) to see
        // only the first 15 — enough to clear `min_pnp_inliers` and register,
        // but below `filter_min_image_observations` (16), so every time
        // `filter_images` runs it demotes camera 2 and nothing else (the seed
        // pair is exempt from filtering by construction, and camera 3 is
        // well-supported). A 4th, well-supported camera is needed because
        // `filter_images` refuses to drop *anyone* once the registered count
        // is already at its floor of 3 (`incremental_sfm.rs`'s
        // `filter_images`: `if remaining <= 3 { continue; }`) — with only 3
        // total cameras, camera 2 could never be filtered no matter how weak
        // its support, which would make this test vacuous. Since camera 2's
        // supporting-observation count can never improve (it structurally
        // only ever sees 15 points), this is a fixed point: register, demote,
        // retry, register, demote, … — bounded only by
        // `max_registration_trials`. Never resetting `trials` on the stall
        // (see `grow_from_seed`'s module-level doc on `stalled_once`) is what
        // makes this terminate at all instead of cycling indefinitely.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for xi in -2..=2 {
            for yi in -1..=2 {
                for zi in 0..=1 {
                    points.push(Point3::new(
                        xi as f64 * 0.25,
                        yi as f64 * 0.25,
                        1.0 + zi as f64 * 0.3,
                    ));
                }
            }
        }
        assert_eq!(points.len(), 40, "test fixture must have exactly 40 points");
        let poses = arc_cameras(4, Point3::origin(), 3.0, 0.6);

        // Camera 2 ("the weak straggler") only ever observes the first 15 of
        // the 40 points; cameras 0, 1 (the seed pair) and 3 observe all 40.
        let mut features = Vec::new();
        let mut visible: Vec<HashMap<usize, usize>> = Vec::new();
        for (cam_idx, pose) in poses.iter().enumerate() {
            let n_visible = if cam_idx == 2 { 15 } else { points.len() };
            let mut kps = Vec::new();
            let mut descs = Vec::new();
            let mut vis = HashMap::new();
            for (pidx, p) in points.iter().enumerate().take(n_visible) {
                let px = project(&camera, pose, p)
                    .expect("fixture point must project in front of every camera");
                vis.insert(pidx, kps.len());
                kps.push(px);
                descs.push(vec![pidx as f32, 1.0, 0.0, 0.0]);
            }
            features.push(FeatureSet::new(kps, descs).unwrap());
            visible.push(vis);
        }

        let n_cams = poses.len();
        let mut pairwise = Vec::new();
        for i in 0..n_cams {
            for j in (i + 1)..n_cams {
                let mut matches = Vec::new();
                for (pidx, &ki) in &visible[i] {
                    if let Some(&kj) = visible[j].get(pidx) {
                        matches.push((ki, kj));
                    }
                }
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i: i,
                        image_j: j,
                        matches,
                        two_view_config: None,
                        essential_matches: None,
                        essential_matrix: None,
                    });
                }
            }
        }

        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 8,
            colmap_style_mapper: true,
            filter_images: true,
            filter_min_image_observations: 16,
            global_ba_images_ratio: 1000.0,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&camera, &features, &pairwise, &config).unwrap();

        assert_eq!(
            result.registered_images, 3,
            "camera 2 can never clear filter_min_image_observations=16 with its \
             fixed 15 supporting observations, so it must end up excluded, not \
             stuck mid-retry or wrongly kept, leaving the other 3 cameras registered"
        );
        assert!(
            result.poses[0].is_some() && result.poses[1].is_some(),
            "the seed pair stays registered (protected from filter_images)"
        );
        assert!(
            result.poses[2].is_none(),
            "the weakly-supported straggler ends up filtered, not registered"
        );
        assert!(
            result.poses[3].is_some(),
            "the well-supported 4th camera stays registered"
        );
    }

    #[test]
    fn colmap_style_mapper_is_deterministic_across_repeated_runs() {
        // M4 regression pin: multi-seed search (`seed_trials`) and the new
        // stall-triggered recovery must stay fully deterministic (fixed PnP
        // RANSAC seed, no reset-driven or iteration-order-driven nondeterminism)
        // — running the identical config against the identical view graph twice
        // must produce byte-identical registered counts, track counts, and mean
        // reprojection error. Uses `build_two_component_scene` (multiple seed
        // candidates, one of them a trap) with `colmap_style_mapper` on so both
        // the multi-seed sweep and the stall-recovery path are exercised.
        let scene = build_two_component_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            colmap_style_mapper: true,
            ..IncrementalSfmConfig::default()
        };
        let a = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        let b = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        assert_eq!(a.registered_images, b.registered_images);
        assert_eq!(a.tracks.len(), b.tracks.len());
        assert_eq!(
            a.mean_reprojection_px.to_bits(),
            b.mean_reprojection_px.to_bits(),
            "mean reprojection error must be bit-identical across repeated runs"
        );
        for i in 0..scene.poses.len() {
            assert_eq!(
                a.poses[i].is_some(),
                b.poses[i].is_some(),
                "image {i}'s registration outcome must be identical across runs"
            );
        }
    }

    #[test]
    fn repeated_bundle_adjustment_does_not_collapse_scale() {
        // A monocular reconstruction has a free scale gauge; without anchoring it
        // a second BA from the converged state collapses the reconstruction.
        // run_bundle_adjustment fixes a second (farthest) pose to pin scale, so
        // re-optimising must be stable — track refinement relies on this.
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let config = IncrementalSfmConfig {
            min_seed_matches: 8,
            min_pnp_inliers: 6,
            track_filter_iterations: 4,
            ..IncrementalSfmConfig::default()
        };
        let result = incremental_sfm(&scene.camera, &features, &pairwise, &config).unwrap();
        // A scale collapse manifests as nearly all tracks dropping out (the EuRoC
        // symptom was 630 -> 1) and the camera geometry degenerating; with the
        // gauge anchored, structure and registration survive four BA rounds.
        assert!(
            result.registered_images >= 5,
            "registration must survive repeated BA, got {}",
            result.registered_images
        );
        assert!(
            result.tracks.len() >= 20,
            "structure must survive repeated BA, got {} tracks",
            result.tracks.len()
        );
        assert!(
            result.mean_reprojection_px < 1.0,
            "reprojection {} px too high after repeated BA",
            result.mean_reprojection_px
        );
        // Camera-spacing ratio stays similarity-correct (a collapse would warp it).
        let registered: Vec<usize> = (0..scene.poses.len())
            .filter(|&i| result.poses[i].is_some())
            .collect();
        let center = |i: usize| {
            result.poses[i]
                .as_ref()
                .unwrap()
                .camera_to_world()
                .translation
        };
        let gt_center = |i: usize| scene.poses[i].camera_to_world().translation;
        let (a, b, c) = (registered[0], registered[1], registered[2]);
        let est_ratio = (center(a) - center(b)).norm() / (center(b) - center(c)).norm();
        let gt_ratio = (gt_center(a) - gt_center(b)).norm() / (gt_center(b) - gt_center(c)).norm();
        assert!(
            (est_ratio - gt_ratio).abs() / gt_ratio < 0.1,
            "camera geometry warped after repeated BA: {est_ratio} vs GT {gt_ratio}"
        );
    }

    #[test]
    fn final_ba_polish_keeps_support_and_is_deterministic() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let mut track_point = vec![None; tracks.len()];
        let config = IncrementalSfmConfig {
            final_ba_polish_iterations: 5,
            ..IncrementalSfmConfig::default()
        };
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        let poses_initial = poses.clone();
        let points_initial = track_point.clone();
        let mut poses_a = poses_initial.clone();
        let mut points_a = points_initial.clone();
        let (stats_a, result_a) = final_fixed_support_ba_polish(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses_a,
            &mut points_a,
        )
        .expect("fixed-support polish should solve the synthetic scene");
        assert!(stats_a.accepted);
        assert!(stats_a.initial_sse.is_finite());
        assert!(stats_a.final_sse <= stats_a.initial_sse);
        assert_eq!(stats_a.tracks_before, stats_a.tracks_after);
        assert_eq!(
            stats_a.observations_before, stats_a.observations_after,
            "polish must not change the supported observation set"
        );
        assert!(result_a.is_some());

        let mut poses_b = poses_initial;
        let mut points_b = points_initial;
        let (stats_b, result_b) = final_fixed_support_ba_polish(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses_b,
            &mut points_b,
        )
        .expect("repeated fixed-support polish should solve identically");
        assert_eq!(stats_a, stats_b);
        assert_eq!(poses_a, poses_b);
        assert_eq!(points_a, points_b);
        assert_eq!(result_a, result_b);

        let mut poses_disabled = poses_a.clone();
        let mut points_disabled = points_a.clone();
        let disabled = final_fixed_support_ba_polish(
            &scene.camera,
            &features,
            &tracks,
            &IncrementalSfmConfig::default(),
            &mut poses_disabled,
            &mut points_disabled,
        )
        .expect("disabled polish is a no-op");
        assert_eq!(disabled.0.requested_iterations, 0);
        assert!(!disabled.0.accepted);
        assert!(disabled.1.is_none());
        assert_eq!(poses_disabled, poses_a);
        assert_eq!(points_disabled, points_a);
    }

    #[test]
    fn structureless_local_bundle_keeps_registered_boundary_poses_exactly_fixed() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let config = IncrementalSfmConfig::default();
        let mut poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let mut track_point = vec![None; tracks.len()];
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );

        // Mimic a slightly inaccurate multi-neighbour structure-less proposal.
        // Only image 2 is allowed to move during the admission refinement.
        let truth = scene.poses[2].clone();
        poses[2].as_mut().unwrap().world_to_camera.translation += Vector3::new(0.03, -0.02, 0.01);
        let before = poses.clone();
        let mut variable = HashSet::new();
        variable.insert(2usize);
        bundle_adjust_local(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &mut poses,
            &mut track_point,
            &variable,
        )
        .expect("fixed-boundary structure-less BA should converge");

        for image in [0usize, 1, 3, 4, 5] {
            assert_eq!(
                poses[image], before[image],
                "registered boundary pose {image} must remain byte-for-byte unchanged"
            );
        }
        let error_before = (before[2].as_ref().unwrap().matrix() - truth.matrix()).norm();
        let error_after = (poses[2].as_ref().unwrap().matrix() - truth.matrix()).norm();
        assert!(
            error_after < error_before,
            "recovered pose should improve while its registered boundary stays fixed: \
             {error_before} -> {error_after}"
        );
    }

    #[test]
    fn structureless_fixed_pose_submap_refines_only_new_landmarks() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let tracks = build_tracks(features.len(), &pairwise, 2);
        let config = IncrementalSfmConfig::default();
        let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let mut track_point = vec![None; tracks.len()];
        triangulate_pending(
            &scene.camera,
            &features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        let new_track = tracks
            .iter()
            .position(|track| track.iter().any(|&(image, _)| image == 2))
            .unwrap();
        let truth = track_point[new_track].unwrap();
        track_point[new_track] = Some(truth + Vector3::new(0.08, -0.05, 0.12));
        let points_before = track_point.clone();
        let mut preexisting = vec![true; track_point.len()];
        preexisting[new_track] = false;

        refine_structureless_new_landmarks(
            &scene.camera,
            &features,
            &tracks,
            &config,
            &poses,
            &mut track_point,
            2,
            &preexisting,
        )
        .expect("fixed-pose local submap should converge");

        for track_id in 0..track_point.len() {
            if track_id != new_track {
                assert_eq!(track_point[track_id], points_before[track_id]);
            }
        }
        assert!(
            (track_point[new_track].unwrap() - truth).norm()
                < (points_before[new_track].unwrap() - truth).norm()
        );
    }

    #[test]
    fn structureless_local_tracks_use_consensus_edges_and_unowned_observations() {
        let scene = build_scene();
        let (features, pairwise) = render(&scene);
        let poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        let missing = 0usize;
        let missing_center = scene.poses[missing].camera_center_world();
        let missing_rotation = scene.poses[missing].world_to_camera.rotation;
        let constraints: Vec<_> = [1usize, 2, 3]
            .into_iter()
            .map(|neighbor| {
                structureless_constraint(
                    neighbor,
                    scene.poses[neighbor].camera_center_world(),
                    missing_center,
                    missing_rotation,
                )
            })
            .collect();
        let proposal = StructurelessPoseProposal {
            pose: scene.poses[missing].clone(),
            neighbor_spread: 1.0,
            line_error_ratio: 0.0,
            consensus_indices: vec![0, 1, 2],
        };
        let local = build_structureless_local_tracks(
            &scene.camera,
            &features,
            &pairwise,
            &[],
            &[],
            &poses,
            missing,
            &constraints,
            &proposal,
            &IncrementalSfmConfig::default(),
        );
        assert!(!local.is_empty());
        for (track, point) in local {
            assert!(track.len() >= 2);
            assert_eq!(
                track.iter().filter(|(image, _)| *image == missing).count(),
                1
            );
            let unique_images: HashSet<_> = track.iter().map(|(image, _)| *image).collect();
            assert_eq!(unique_images.len(), track.len());
            for &(image, keypoint) in &track {
                let error = reprojection_error_px(
                    &scene.camera,
                    poses[image].as_ref().unwrap(),
                    &point,
                    &features[image].keypoints[keypoint],
                )
                .unwrap();
                assert!(error <= 2.0);
            }
        }
    }

    fn structureless_constraint(
        neighbor: usize,
        neighbor_center: Point3<f64>,
        missing_center: Point3<f64>,
        missing_rotation: UnitQuaternion<f64>,
    ) -> StructurelessConstraint {
        StructurelessConstraint {
            neighbor,
            neighbor_center,
            missing_rotation,
            center_direction: (missing_center - neighbor_center).normalize(),
            weight: 100.0 - neighbor as f64,
        }
    }

    #[test]
    fn structureless_multineighbor_lines_recover_scaled_camera_pose() {
        let missing_center = Point3::new(1.2, -0.4, 3.5);
        let rotation = UnitQuaternion::from_euler_angles(0.05, -0.12, 0.08);
        let constraints = vec![
            structureless_constraint(0, Point3::new(-1.0, 0.0, 0.0), missing_center, rotation),
            structureless_constraint(1, Point3::new(2.0, 0.5, 0.2), missing_center, rotation),
            structureless_constraint(2, Point3::new(0.0, -2.0, 0.4), missing_center, rotation),
        ];
        let config = IncrementalSfmConfig {
            structureless_min_intersection_angle_deg: 1.0,
            structureless_max_center_line_error_ratio: 1e-8,
            ..IncrementalSfmConfig::default()
        };
        let proposal = solve_structureless_pose(&constraints, &config).unwrap();
        assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
        assert!((proposal.pose.world_to_camera.rotation.inverse() * rotation).angle() < 1e-12);
        assert!(proposal.line_error_ratio < 1e-9);
    }

    #[test]
    fn structureless_pose_interpolation_uses_camera_centers_and_slerp() {
        let from_center = Point3::new(-1.0, 0.5, 2.0);
        let to_center = Point3::new(3.0, -0.5, 4.0);
        let from_rotation = UnitQuaternion::identity();
        let to_rotation = UnitQuaternion::from_euler_angles(0.0, 0.4, 0.0);
        let from = Pose::from_world_to_camera(
            from_rotation,
            -from_rotation.transform_vector(&from_center.coords),
        );
        let to = Pose::from_world_to_camera(
            to_rotation,
            -to_rotation.transform_vector(&to_center.coords),
        );
        let midpoint = interpolate_structureless_pose(&from, &to, 0.5);
        let expected_center = Point3::from((from_center.coords + to_center.coords) * 0.5);
        assert!((midpoint.camera_center_world() - expected_center).norm() < 1e-12);
        let expected_rotation = from_rotation.slerp(&to_rotation, 0.5);
        assert!((midpoint.world_to_camera.rotation.inverse() * expected_rotation).angle() < 1e-12);
        assert_eq!(interpolate_structureless_pose(&from, &to, 0.0), from);
        assert_eq!(interpolate_structureless_pose(&from, &to, 1.0), to);
    }

    #[test]
    fn structureless_pose_rejects_single_neighbor_arbitrary_scale() {
        let missing_center = Point3::new(0.0, 0.0, 3.0);
        let constraints = vec![structureless_constraint(
            0,
            Point3::origin(),
            missing_center,
            UnitQuaternion::identity(),
        )];
        assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
    }

    #[test]
    fn structureless_pose_rejects_rotation_disagreement() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let constraints = vec![
            structureless_constraint(
                0,
                Point3::new(-1.0, 0.0, 0.0),
                missing_center,
                UnitQuaternion::identity(),
            ),
            structureless_constraint(
                1,
                Point3::new(1.0, 0.0, 0.0),
                missing_center,
                UnitQuaternion::from_euler_angles(0.0, 0.2, 0.0),
            ),
        ];
        assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
    }

    #[test]
    fn structureless_pose_uses_largest_rotation_consensus() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let good_rotation = UnitQuaternion::from_euler_angles(0.01, -0.02, 0.03);
        let mut constraints = vec![
            structureless_constraint(
                0,
                Point3::new(-1.0, 0.0, 0.0),
                missing_center,
                good_rotation,
            ),
            structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, good_rotation),
            structureless_constraint(
                2,
                Point3::new(0.0, -1.0, 0.0),
                missing_center,
                good_rotation,
            ),
            structureless_constraint(
                3,
                Point3::new(0.0, 1.0, 0.0),
                missing_center,
                UnitQuaternion::from_euler_angles(0.0, 0.8, 0.0),
            ),
        ];
        constraints[3].weight = 1000.0;
        let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
            .expect("three coherent rotations must outvote one high-support outlier");
        assert_eq!(proposal.consensus_indices.len(), 3);
        assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
        assert!((proposal.pose.world_to_camera.rotation.inverse() * good_rotation).angle() < 1e-12);
    }

    #[test]
    fn structureless_pose_keeps_rotation_consensus_centre_not_strongest_edge() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let centre_rotation = UnitQuaternion::identity();
        let positive_edge = UnitQuaternion::from_euler_angles(0.0, 2.5f64.to_radians(), 0.0);
        let negative_edge = UnitQuaternion::from_euler_angles(0.0, -2.5f64.to_radians(), 0.0);
        let mut constraints = vec![
            structureless_constraint(
                0,
                Point3::new(-1.0, 0.0, 0.0),
                missing_center,
                centre_rotation,
            ),
            structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, positive_edge),
            structureless_constraint(
                2,
                Point3::new(0.0, -1.0, 0.0),
                missing_center,
                negative_edge,
            ),
        ];
        constraints[1].weight = 1000.0;
        let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
            .expect("the centre edge supports a valid three-rotation consensus");
        assert!(
            (proposal.pose.world_to_camera.rotation.inverse() * centre_rotation).angle() < 1e-12,
            "the high-weight +2.5deg edge would be 5deg from the negative edge"
        );
        let consistency = structureless_pose_consistency(
            &proposal.pose,
            &constraints,
            &proposal,
            &IncrementalSfmConfig::default(),
        );
        assert!(consistency.accepted);
        assert!(consistency.max_rotation_deg <= 3.0);
    }

    #[test]
    fn structureless_pose_uses_robust_translation_consensus() {
        let missing_center = Point3::new(0.5, 0.2, 3.0);
        let rotation = UnitQuaternion::identity();
        let mut constraints = vec![
            structureless_constraint(0, Point3::new(-1.0, 0.0, 0.0), missing_center, rotation),
            structureless_constraint(1, Point3::new(1.0, 0.0, 0.0), missing_center, rotation),
            structureless_constraint(2, Point3::new(0.0, -1.0, 0.0), missing_center, rotation),
            StructurelessConstraint {
                neighbor: 3,
                neighbor_center: Point3::new(0.0, 1.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::x(),
                weight: 1000.0,
            },
        ];
        constraints.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        let config = IncrementalSfmConfig {
            structureless_max_center_line_error_ratio: 0.01,
            ..IncrementalSfmConfig::default()
        };
        let proposal = solve_structureless_pose(&constraints, &config)
            .expect("three coherent directions must reject one high-support translation outlier");
        assert_eq!(proposal.consensus_indices.len(), 3);
        assert!((proposal.pose.camera_center_world() - missing_center).norm() < 1e-9);
    }

    #[test]
    fn structureless_pose_reclassifies_lines_after_weighted_refit() {
        let rotation = UnitQuaternion::identity();
        let mut constraints = vec![
            StructurelessConstraint {
                neighbor: 0,
                neighbor_center: Point3::new(-1.0, 0.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::x(),
                weight: 100.0,
            },
            StructurelessConstraint {
                neighbor: 1,
                neighbor_center: Point3::new(0.0, -1.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::y(),
                weight: 100.0,
            },
            StructurelessConstraint {
                neighbor: 2,
                neighbor_center: Point3::new(0.0, 1.0, 0.0),
                missing_rotation: rotation,
                center_direction: Vector3::new(0.1, -1.0, 0.0).normalize(),
                weight: 1000.0,
            },
            // This short directed baseline agrees with the winning pairwise
            // hypothesis at the origin, but the high-weight tilted line moves
            // the least-squares refit behind it. It must be reclassified as an
            // outlier instead of vetoing the other three consistent lines.
            StructurelessConstraint {
                neighbor: 3,
                neighbor_center: Point3::new(0.01, 0.0, 0.0),
                missing_rotation: rotation,
                center_direction: -Vector3::x(),
                weight: 100.0,
            },
        ];
        constraints.sort_by(|a, b| b.weight.total_cmp(&a.weight));
        let proposal = solve_structureless_pose(&constraints, &IncrementalSfmConfig::default())
            .expect("a marginal directed line must not veto a stable 3-line refit");
        assert_eq!(proposal.consensus_indices.len(), 3);
        assert!(proposal
            .consensus_indices
            .iter()
            .all(|&index| constraints[index].neighbor != 3));
        assert!(
            structureless_pose_consistency(
                &proposal.pose,
                &constraints,
                &proposal,
                &IncrementalSfmConfig::default(),
            )
            .accepted
        );
    }

    #[test]
    fn structureless_pose_rejects_parallel_center_directions() {
        let constraints = vec![
            StructurelessConstraint {
                neighbor: 0,
                neighbor_center: Point3::new(0.0, 0.0, 0.0),
                missing_rotation: UnitQuaternion::identity(),
                center_direction: Vector3::z(),
                weight: 100.0,
            },
            StructurelessConstraint {
                neighbor: 1,
                neighbor_center: Point3::new(1.0, 0.0, 0.0),
                missing_rotation: UnitQuaternion::identity(),
                center_direction: Vector3::z(),
                weight: 90.0,
            },
        ];
        assert!(solve_structureless_pose(&constraints, &IncrementalSfmConfig::default()).is_err());
    }

    /// Island-chain fixture. Ten arc cameras all observing one point cloud,
    /// with the verified pair graph pruned into a main component
    /// `{0, 1, 2, 3, 6, 7, 8, 9}` and a two-image island `{4, 5}` where the
    /// bridge image `5` has a *higher* index than its dependent `4`:
    ///
    /// - `4` pairs only with `{2, 3, 5}` — two registered neighbours while
    ///   `5` is unregistered, below [`IncrementalSfmConfig::
    ///   structureless_min_neighbors`];
    /// - `5` pairs with registered `{3, 6, 7}` plus the island partner `4`.
    ///
    /// Every island pair is narrow-baseline (adjacent arc steps) because the
    /// two-view essential estimate degrades on this synthetic cloud beyond
    /// ~0.4 rad of arc separation. Disjoint keypoint bands keep every
    /// island-touching union-find component at two images, below the
    /// track-length floor: the clean global model triangulates from
    /// main-component tracks only, leaving the island's observations free
    /// for local-submap synthesis — exactly the thin-per-image-structure
    /// regime the courtyard second component exposed.
    struct IslandScene {
        camera: Camera,
        poses: Vec<Pose>,
        features: Vec<FeatureSet>,
        pairwise: Vec<PairwiseMatches>,
    }

    fn build_island_scene() -> IslandScene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for xi in -3..=3 {
            for yi in -2..=2 {
                for zi in 0..=4 {
                    points.push(Point3::new(
                        xi as f64 * 0.3,
                        yi as f64 * 0.3,
                        zi as f64 * 0.25,
                    ));
                }
            }
        }
        let mut poses = Vec::new();
        for k in 0..10 {
            let angle = -0.585 + k as f64 * 0.13;
            let radius = 3.0;
            let cam_center = Point3::new(radius * angle.sin(), 0.0, -radius * angle.cos());
            let forward = (Point3::origin() - cam_center).normalize();
            let world_up = Vector3::new(0.0, 1.0, 0.0);
            let right = forward.cross(&world_up).normalize();
            let up = right.cross(&forward);
            let r_cam_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let q_c2w = UnitQuaternion::from_rotation_matrix(
                &nalgebra::Rotation3::from_matrix_unchecked(r_cam_to_world),
            );
            let q_w2c = q_c2w.inverse();
            let t_w2c = -(q_w2c * cam_center.coords);
            poses.push(Pose::from_world_to_camera(q_w2c, t_w2c));
        }

        // Every camera sees every point; keypoint index == point index.
        let features: Vec<FeatureSet> = poses
            .iter()
            .map(|pose| {
                let (kps, descs): (Vec<_>, Vec<_>) = points
                    .iter()
                    .enumerate()
                    .filter_map(|(pidx, p)| {
                        project(&camera, pose, p).map(|px| (px, vec![pidx as f32, 1.0, 0.0, 0.0]))
                    })
                    .unzip();
                FeatureSet::new(kps, descs).unwrap()
            })
            .collect();

        // Strided keypoint bands. Every band must mix points across all
        // three grid axes: a band confined to one grid slice is exactly
        // planar, and a planar correspondence set makes the two-view
        // essential estimate chirality-degenerate (the failure that
        // motivated this design).
        let all: Vec<usize> = (0..points.len()).collect();
        let main_points: Vec<usize> = all.iter().step_by(4).copied().collect();
        let remainder: Vec<usize> = {
            let main_set: HashSet<usize> = main_points.iter().copied().collect();
            all.into_iter().filter(|p| !main_set.contains(p)).collect()
        };
        let island_band =
            |k: usize| -> Vec<usize> { remainder.iter().skip(k).step_by(6).copied().collect() };
        let band_a = island_band(0);
        let band_b = island_band(1);
        let band_c = island_band(2);
        let band_d = island_band(3);
        let band_e = island_band(4);
        let band_f = island_band(5);

        let pair = |image_i: usize, image_j: usize, band: &[usize]| PairwiseMatches {
            image_i,
            image_j,
            matches: band.iter().map(|&p| (p, p)).collect(),
            two_view_config: None,
            essential_matches: None,
            essential_matrix: None,
        };

        let mut pairwise = Vec::new();
        let main = [0usize, 1, 2, 3, 6, 7, 8, 9];
        for (a, &i) in main.iter().enumerate() {
            for &j in main.iter().skip(a + 1) {
                pairwise.push(pair(i, j, &main_points));
            }
        }
        pairwise.push(pair(4, 3, &band_a));
        pairwise.push(pair(4, 2, &band_b));
        pairwise.push(pair(4, 5, &band_c));
        pairwise.push(pair(5, 6, &band_d));
        pairwise.push(pair(5, 3, &band_e));
        pairwise.push(pair(5, 7, &band_f));

        IslandScene {
            camera,
            poses,
            features,
            pairwise,
        }
    }

    #[test]
    fn structureless_rounds_chain_an_island_through_a_higher_indexed_bridge() {
        let scene = build_island_scene();
        let min_track_length = 5;
        let mut tracks = build_tracks(scene.features.len(), &scene.pairwise, min_track_length);
        assert!(!tracks.is_empty(), "fixture sanity: main tracks must form");
        for track in &tracks {
            let images: HashSet<usize> = track.iter().map(|&(image, _)| image).collect();
            assert!(
                !images.contains(&4) && !images.contains(&5),
                "fixture sanity: island observations must not join global tracks"
            );
        }

        // Register only the main component with ground-truth poses and
        // triangulate the clean model.
        let mut poses: Vec<Option<Pose>> = scene.poses.iter().cloned().map(Some).collect();
        poses[4] = None;
        poses[5] = None;
        let mut track_point = vec![None; tracks.len()];
        let config = IncrementalSfmConfig {
            colmap_style_mapper: true,
            structureless_registration: true,
            structureless_min_pair_inliers: 5,
            structureless_min_support_tracks: 6,
            // The 20-point synthetic essentials carry ~1 deg of rotation
            // noise, which at fx=500 is ~9 px of reprojection — far beyond
            // the production-default 2 px admission gate that real
            // hundreds-of-inlier matches easily meet. This fixture exercises
            // the round-chaining mechanics, not the pixel gate (which has
            // its own dedicated tests), so the gate is widened accordingly.
            structureless_max_reprojection_error_px: 12.0,
            max_reprojection_error_px: 12.0,
            ..IncrementalSfmConfig::default()
        };
        triangulate_pending(
            &scene.camera,
            &scene.features,
            &tracks,
            &poses,
            &config,
            &mut track_point,
        );
        let clean_points = track_point.iter().filter(|p| p.is_some()).count();
        assert!(
            clean_points >= 10,
            "fixture sanity: clean model must triangulate ({clean_points} points)"
        );

        // A single ascending scan must register the bridge `6` but leave `3`
        // behind: when the scan reaches `3`, `6` is still unregistered and `3`
        // has only two admissible neighbours.
        let single_round_config = IncrementalSfmConfig {
            structureless_max_rounds: 1,
            ..config.clone()
        };
        let single_registered = structureless_registration_rounds(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &mut tracks.clone(),
            &single_round_config,
            &mut poses.clone(),
            &mut track_point.clone(),
        );
        assert_eq!(
            single_registered, 1,
            "one ascending pass must recover exactly the bridge image"
        );

        // Multiple rounds feed `6` back in as a neighbour and chain `3`.
        let total_registered = structureless_registration_rounds(
            &scene.camera,
            &scene.features,
            &scene.pairwise,
            &mut tracks,
            &config,
            &mut poses,
            &mut track_point,
        );
        assert_eq!(
            total_registered, 2,
            "rounds must chain the dependent island image through the bridge"
        );
        assert!(poses.iter().all(Option::is_some));

        // The chained pose must sit at the true (metric) geometry: rotation
        // tight, centre within a fraction of the neighbour spread.
        for image in [4usize, 5] {
            let pose = poses[image].as_ref().unwrap();
            let rotation_error = (pose.world_to_camera.rotation.inverse()
                * scene.poses[image].world_to_camera.rotation)
                .angle();
            let center_error =
                (pose.camera_center_world() - scene.poses[image].camera_center_world()).norm();
            assert!(
                rotation_error < 0.01,
                "image {image} rotation error {rotation_error} rad too large"
            );
            assert!(
                center_error < 0.05,
                "image {image} centre error {center_error} m too large"
            );
        }
    }
}
