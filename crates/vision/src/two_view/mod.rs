//! Classical two-view geometry building blocks.
//!
//! The module exposes a small, testable pipeline:
//! 1. Normalize pixel correspondences with camera intrinsics.
//! 2. Estimate the essential matrix with the 8-point algorithm in a
//!    Sampson-distance scored RANSAC loop (`EssentialRansac`).
//! 3. Decompose the essential matrix into the four (R, t) candidates and pick
//!    the one with the most correspondences in front of both cameras
//!    (`recover_relative_pose`).
//! 4. Compose the above as a `RelativePoseEstimator`, optionally applying a
//!    caller-supplied translation scale.
//!
//! These components are intentionally independent of `Frame`, `FrameId`, or
//! the tracking pipeline so they can be used as the geometric core of any
//! `VisualOdometryFrontend` implementation. The
//! `EssentialMatrixVisualOdometryFrontend` in the top-level `visloc-rs` crate
//! wires this module into the existing tracking layer.
//!
//! [`colmap_verification`] adds a second, opt-in tier on top of the essential-
//! matrix-only pipeline above: COLMAP-style multi-model (essential /
//! fundamental / homography) two-view verification with
//! [`colmap_verification::ConfigurationType`] classification
//! (`CALIBRATED`/`UNCALIBRATED`/`PLANAR`/`PANORAMIC`/`WATERMARK`/`DEGENERATE`/
//! `MULTIPLE`). It is a drop-in alternative — every existing caller of
//! `EssentialRansac`/`RelativePoseEstimator` is unchanged; see
//! `examples/unordered_sfm_demo.rs`'s `--colmap-verification` flag for the
//! A/B wiring. [`homography`] and [`fundamental`] hold the estimators the
//! classifier composes.
//!
//! [`correspondence_graph`] adds a third, independent tier: COLMAP's
//! persistent `CorrespondenceGraph` view-graph structure (M2 in
//! `docs/colmap_port_plan.md`), which stores per-image-pair
//! [`colmap_verification::ConfigurationType`] and per-feature adjacency for
//! track building and transitive-correspondence queries. It is consumed by
//! `pipelines/slam/src/incremental_sfm.rs`'s `build_tracks_via_graph` as an
//! opt-in alternative to the legacy ad hoc union-find.
//!
//! [`rescue`] adds a fourth, independent tier: M5's disconnection detector +
//! cross-component candidate generator, used by `examples/unordered_sfm_demo
//! .rs`'s `--rescue-bridging` pass to propose and (after re-verification)
//! admit bridge pairs between components the standard pipeline left
//! disconnected — see `docs/colmap_port_plan.md`'s "M5 results".

use nalgebra::{DMatrix, Matrix3, Matrix3x4, Point2, UnitQuaternion, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use visloc_core::geometry::SE3;
use visloc_core::types::Camera;

pub mod colmap_verification;
pub mod correspondence_graph;
pub mod fundamental;
pub mod homography;
pub mod rescue;

pub use colmap_verification::{
    ConfigurationType, TwoViewGeometryOptions, TwoViewGeometryReport, TwoViewGeometryVerifier,
};
pub use correspondence_graph::{
    Correspondence, CorrespondenceGraph, CorrespondenceGraphError, EdgeMetadata, IngestStats,
};
pub use fundamental::{
    estimate_fundamental_dlt, fundamental_ransac, fundamental_squared_sampson_error,
    FundamentalRansacConfig, FundamentalReport,
};
pub use homography::{
    decompose_homography_matrix, estimate_homography_dlt, homography_ransac,
    homography_squared_error, pose_from_homography_matrix, HomographyMotion,
    HomographyRansacConfig, HomographyReport,
};
pub use rescue::{connected_components, generate_bridge_candidates, BridgeCandidateOptions};

/// One pixel-space correspondence between a previous and a current frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoViewCorrespondence {
    pub previous_xy: Point2<f64>,
    pub current_xy: Point2<f64>,
}

impl TwoViewCorrespondence {
    pub fn new(previous_xy: Point2<f64>, current_xy: Point2<f64>) -> Self {
        Self {
            previous_xy,
            current_xy,
        }
    }
}

/// Output of the essential-matrix RANSAC loop.
#[derive(Debug, Clone, PartialEq)]
pub struct EssentialRansacReport {
    pub essential: Matrix3<f64>,
    pub inliers: Vec<usize>,
    pub mean_sampson_error: f64,
}

/// Output of relative-pose recovery: rotation, unit translation direction,
/// applied scale, the implied `SE3` previous-to-current transform, inlier
/// indices, and Sampson diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePose {
    pub previous_to_current: SE3,
    pub translation_unit: Vector3<f64>,
    pub translation_scale: f64,
    pub inliers: Vec<usize>,
    pub mean_sampson_error: f64,
    /// Runner-up (R, t_unit) when essential decomposition is chirality-
    /// ambiguous. Global SfM can keep this as an alternate view-graph edge
    /// hypothesis; `None` when the margin is decisive or unused.
    pub alternate: Option<(UnitQuaternion<f64>, Vector3<f64>)>,
    /// `(best - second) / best` from cheirality scoring; near zero means the
    /// essential was ambiguous. View-graph builders can down-weight such edges.
    pub chirality_margin: f64,
}

/// Estimator for an essential matrix from pixel correspondences plus
/// intrinsics. Implementors return `None` when the input is degenerate.
pub trait EssentialMatrixEstimator {
    fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<Matrix3<f64>>;
    fn minimum_correspondences(&self) -> usize;
}

/// Hartley-normalized 8-point essential-matrix estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EightPointEssentialMatrixEstimator {
    pub min_correspondences: usize,
}

impl Default for EightPointEssentialMatrixEstimator {
    fn default() -> Self {
        Self {
            min_correspondences: 8,
        }
    }
}

impl EssentialMatrixEstimator for EightPointEssentialMatrixEstimator {
    fn minimum_correspondences(&self) -> usize {
        self.min_correspondences
    }

    fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<Matrix3<f64>> {
        if correspondences.len() < self.min_correspondences {
            return None;
        }

        let normalized = normalize_pairs(correspondences, camera)?;
        let (previous_normalization, previous_points) =
            hartley_normalization(normalized.iter().map(|(p, _)| *p))?;
        let (current_normalization, current_points) =
            hartley_normalization(normalized.iter().map(|(_, c)| *c))?;

        let mut a = DMatrix::<f64>::zeros(normalized.len(), 9);
        for row in 0..normalized.len() {
            let (x, y) = (previous_points[row].x, previous_points[row].y);
            let (xp, yp) = (current_points[row].x, current_points[row].y);
            a[(row, 0)] = xp * x;
            a[(row, 1)] = xp * y;
            a[(row, 2)] = xp;
            a[(row, 3)] = yp * x;
            a[(row, 4)] = yp * y;
            a[(row, 5)] = yp;
            a[(row, 6)] = x;
            a[(row, 7)] = y;
            a[(row, 8)] = 1.0;
        }

        // The 8-point linear system A * f = 0 has 9 unknowns. When there are
        // fewer than 9 rows the thin SVD that nalgebra computes for non-square
        // inputs drops the last right singular vector — the very direction we
        // want. Multiply by A^T A so the SVD always operates on a 9x9 matrix.
        let ata = a.transpose() * a;
        let svd = ata.svd(true, true);
        let v_t = svd.v_t?;
        let last = v_t.row(v_t.nrows() - 1);
        let mut essential_normalized = Matrix3::new(
            last[0], last[1], last[2], last[3], last[4], last[5], last[6], last[7], last[8],
        );

        // Project E_normalized onto the essential manifold.
        let essential_norm_svd = essential_normalized.svd(true, true);
        let u_n = essential_norm_svd.u?;
        let v_t_n = essential_norm_svd.v_t?;
        let s_n =
            (essential_norm_svd.singular_values[0] + essential_norm_svd.singular_values[1]) * 0.5;
        let constrained_n = Matrix3::from_diagonal(&Vector3::new(s_n, s_n, 0.0));
        essential_normalized = u_n * constrained_n * v_t_n;

        let essential_calibrated: Matrix3<f64> =
            current_normalization.transpose() * essential_normalized * previous_normalization;
        Some(essential_calibrated)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssentialRansacConfig {
    pub iterations: usize,
    /// Sampson distance threshold in normalized image-plane units. A 1-pixel
    /// reprojection error at focal length `f` corresponds to a normalized
    /// distance of `1.0 / f`.
    pub sampson_threshold: f64,
    pub seed: u64,
}

impl Default for EssentialRansacConfig {
    fn default() -> Self {
        Self {
            iterations: 256,
            sampson_threshold: 5.0e-3,
            seed: 7,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EssentialRansac<E = EightPointEssentialMatrixEstimator> {
    pub estimator: E,
    pub config: EssentialRansacConfig,
}

impl Default for EssentialRansac {
    fn default() -> Self {
        Self {
            estimator: EightPointEssentialMatrixEstimator::default(),
            config: EssentialRansacConfig::default(),
        }
    }
}

impl<E> EssentialRansac<E>
where
    E: EssentialMatrixEstimator,
{
    pub fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<EssentialRansacReport> {
        self.estimate_with_optional_weights(correspondences, camera, None)
    }

    /// PROSAC-style confidence-weighted variant: sort correspondences by
    /// `weights` descending, then for iteration `k` draw the minimal
    /// sample uniformly from the top-`m_k` correspondences where `m_k`
    /// expands from `sample_size` to `correspondences.len()` over the
    /// configured iteration budget. High-confidence matches anchor the
    /// early iterations (so RANSAC finds a consensus quickly when the
    /// weights are informative) and the worst candidates get evaluated
    /// only late in the schedule. Falls back to the uniform-shuffle
    /// behaviour when `weights` is `None` or all weights are zero.
    pub fn estimate_with_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
        weights: &[f32],
    ) -> Option<EssentialRansacReport> {
        if weights.len() != correspondences.len() {
            return self.estimate(correspondences, camera);
        }
        self.estimate_with_optional_weights(correspondences, camera, Some(weights))
    }

    fn estimate_with_optional_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
        weights: Option<&[f32]>,
    ) -> Option<EssentialRansacReport> {
        let sample_size = self.estimator.minimum_correspondences();
        if correspondences.len() < sample_size {
            return None;
        }

        // PROSAC ordering: sort indices by descending weight when weights
        // are usable; otherwise fall back to natural order + uniform
        // shuffle (the original behaviour).
        let weighted = weights.filter(|w| w.iter().any(|&v| v.is_finite() && v > 0.0));
        let mut sorted_indices: Vec<usize> = (0..correspondences.len()).collect();
        if let Some(w) = weighted {
            sorted_indices
                .sort_by(|&a, &b| w[b].partial_cmp(&w[a]).unwrap_or(std::cmp::Ordering::Equal));
        }

        let mut rng = SmallRng::seed_from_u64(self.config.seed);
        let mut best_inliers: Vec<usize> = Vec::new();
        let mut best_essential: Option<Matrix3<f64>> = None;
        let threshold_sq = self.config.sampson_threshold * self.config.sampson_threshold;
        let n = correspondences.len();
        let total_iters = self.config.iterations.max(1);

        for iteration in 0..self.config.iterations {
            // PROSAC shrinking sample-set: m_k expands linearly from
            // `sample_size` to `n` over the iteration budget. When
            // weights are absent this collapses to `m_k = n` and the
            // shuffle samples uniformly across all correspondences.
            let m_k = if weighted.is_some() {
                let progress = iteration as f64 / total_iters as f64;
                let m = sample_size as f64 + (n - sample_size) as f64 * progress;
                (m.ceil() as usize).clamp(sample_size, n)
            } else {
                n
            };

            // Sample `sample_size` distinct indices uniformly from the
            // top-m_k of the (possibly sorted) index list.
            let mut subset: Vec<usize> = sorted_indices[..m_k].to_vec();
            subset.shuffle(&mut rng);
            let sample: Vec<TwoViewCorrespondence> = subset[..sample_size]
                .iter()
                .map(|&i| correspondences[i])
                .collect();

            let Some(candidate) = self.estimator.estimate(&sample, camera) else {
                continue;
            };

            let inliers = score_inliers(&candidate, correspondences, camera, threshold_sq);
            if inliers.len() > best_inliers.len() {
                best_inliers = inliers;
                best_essential = Some(candidate);
            }
        }

        let essential = best_essential?;
        if best_inliers.len() < sample_size {
            return None;
        }

        let inlier_correspondences: Vec<TwoViewCorrespondence> =
            best_inliers.iter().map(|&i| correspondences[i]).collect();
        let refined = self
            .estimator
            .estimate(&inlier_correspondences, camera)
            .unwrap_or(essential);
        let final_inliers = score_inliers(&refined, correspondences, camera, threshold_sq);
        let final_inliers = if final_inliers.len() >= best_inliers.len() {
            final_inliers
        } else {
            best_inliers
        };
        let mean = mean_sampson_error(&refined, correspondences, camera, &final_inliers);

        Some(EssentialRansacReport {
            essential: refined,
            inliers: final_inliers,
            mean_sampson_error: mean,
        })
    }
}

/// Gates applied when selecting among the four essential-matrix (R, t)
/// hypotheses. Default values reproduce the legacy positive-depth-only
/// count that [`recover_relative_pose`] has always used.
///
/// Hardened settings ([`CheiralityOptions::hardened`]) additionally require a
/// minimum triangulation angle, reject poses whose second-best hypothesis
/// scores nearly as well as the winner (chirality / façade ambiguity), and
/// demand that a minimum fraction of inliers pass the positive-depth test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CheiralityOptions {
    /// Minimum ray-intersection angle (degrees) for a triangulated point to
    /// count toward the cheirality score. `0.0` = legacy (depth-only).
    pub min_tri_angle_deg: f64,
    /// Reject the recovery when `second_best / best` exceeds this ratio.
    /// `1.0` never rejects on ratio alone (legacy); values like `0.85` drop
    /// chirality-ambiguous essentials that survive plain inlier counts on
    /// repetitive façades.
    pub max_ambiguity_ratio: f64,
    /// Require `best_score >= fraction * inliers.len()`. `0.0` = legacy.
    pub min_positive_depth_fraction: f64,
}

impl Default for CheiralityOptions {
    fn default() -> Self {
        Self {
            min_tri_angle_deg: 0.0,
            max_ambiguity_ratio: 1.0,
            min_positive_depth_fraction: 0.0,
        }
    }
}

impl CheiralityOptions {
    /// Courtyard-class edge quality: angle-gated cheirality, ambiguity
    /// rejection, and a majority positive-depth requirement.
    pub fn hardened() -> Self {
        Self {
            min_tri_angle_deg: 1.0,
            max_ambiguity_ratio: 0.85,
            min_positive_depth_fraction: 0.5,
        }
    }

    /// Like [`Self::hardened`] but never drops on `second/best` ratio alone —
    /// the runner-up is exposed via [`RelativePoseRecovery::alternate`] so the
    /// view-graph can carry multi-hypothesis edges instead of discarding the
    /// pair.
    pub fn hardened_keep_ambiguous() -> Self {
        Self {
            max_ambiguity_ratio: 1.0,
            ..Self::hardened()
        }
    }
}

/// Result of decomposing an essential matrix, including the margin that
/// separates the winning hypothesis from the runner-up. Callers that build
/// view-graph edges can down-weight low-margin recoveries or keep the
/// runner-up as an alternate.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativePoseRecovery {
    pub rotation: UnitQuaternion<f64>,
    pub translation_unit: Vector3<f64>,
    /// Cheirality (positive-depth [+ angle]) count for the winning hypothesis.
    pub best_score: i64,
    /// Same count for the second-best of the four hypotheses.
    pub second_score: i64,
    /// Runner-up (R, t) when its score is positive and the rotation differs
    /// from the winner; used for multi-hypothesis view-graph edges.
    pub alternate: Option<(UnitQuaternion<f64>, Vector3<f64>)>,
}

impl RelativePoseRecovery {
    /// `(best - second) / best`, or `0` when `best == 0`. Near-zero means the
    /// essential was chirality-ambiguous.
    pub fn chirality_margin(&self) -> f64 {
        if self.best_score <= 0 {
            return 0.0;
        }
        (self.best_score - self.second_score) as f64 / self.best_score as f64
    }
}

/// Composes essential-matrix RANSAC with relative-pose recovery, applying a
/// caller-controlled translation scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RelativePoseEstimator<E = EightPointEssentialMatrixEstimator> {
    pub ransac: EssentialRansac<E>,
    /// Default translation scale applied when no per-call scale is supplied.
    /// Stays at 1.0 unless the caller knows the metric scale (e.g., from a
    /// GNSS displacement, the previous frame's translation, or a configured
    /// default).
    pub default_translation_scale: f64,
    /// Cheirality gates applied during essential decomposition. Default is
    /// byte-identical to the historical positive-depth-only selector.
    pub cheirality: CheiralityOptions,
}

impl Default for RelativePoseEstimator {
    fn default() -> Self {
        Self {
            ransac: EssentialRansac::default(),
            default_translation_scale: 1.0,
            cheirality: CheiralityOptions::default(),
        }
    }
}

impl<E> RelativePoseEstimator<E>
where
    E: EssentialMatrixEstimator,
{
    pub fn estimate(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> Option<RelativePose> {
        self.estimate_with_scale(correspondences, camera, self.default_translation_scale)
    }

    pub fn estimate_with_scale(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
        translation_scale: f64,
    ) -> Option<RelativePose> {
        self.estimate_with_scale_and_optional_weights(
            correspondences,
            camera,
            translation_scale,
            None,
        )
    }

    /// Estimate a relative pose when the two images have different pinhole
    /// calibrations.  Each endpoint is first converted to its own normalized
    /// bearing and the existing deterministic estimator then runs in a unit
    /// pinhole convention.  Inlier indices and pose conventions are identical
    /// to [`Self::estimate`]; no descriptor or correspondence order changes.
    pub fn estimate_with_cameras(
        &self,
        correspondences: &[TwoViewCorrespondence],
        previous_camera: &Camera,
        current_camera: &Camera,
    ) -> Option<RelativePose> {
        let normalized = normalize_correspondences_with_cameras(
            correspondences,
            previous_camera,
            current_camera,
        )?;
        let unit_camera = Camera::pinhole(0, 1, 1, 1.0, 1.0, 0.0, 0.0);
        self.estimate(&normalized, &unit_camera)
    }

    /// PROSAC-flavoured variant: order RANSAC sampling by `weights` (e.g.
    /// matcher confidence) so high-confidence correspondences anchor early
    /// iterations. Falls back to the uniform path when `weights` is the
    /// wrong length / all-zero / all-non-finite.
    pub fn estimate_with_scale_and_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
        translation_scale: f64,
        weights: &[f32],
    ) -> Option<RelativePose> {
        self.estimate_with_scale_and_optional_weights(
            correspondences,
            camera,
            translation_scale,
            Some(weights),
        )
    }

    fn estimate_with_scale_and_optional_weights(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
        translation_scale: f64,
        weights: Option<&[f32]>,
    ) -> Option<RelativePose> {
        let report = match weights {
            Some(w) if w.len() == correspondences.len() => {
                self.ransac
                    .estimate_with_weights(correspondences, camera, w)
            }
            _ => self.ransac.estimate(correspondences, camera),
        }?;
        let recovered = recover_relative_pose_with_options(
            &report.essential,
            correspondences,
            camera,
            &report.inliers,
            &self.cheirality,
        )?;
        let se3 = SE3::new(
            recovered.rotation,
            recovered.translation_unit * translation_scale,
        );
        Some(RelativePose {
            previous_to_current: se3,
            translation_unit: recovered.translation_unit,
            translation_scale,
            inliers: report.inliers,
            mean_sampson_error: report.mean_sampson_error,
            alternate: recovered.alternate,
            chirality_margin: recovered.chirality_margin(),
        })
    }
}

/// Decompose an essential matrix into the (R, t_unit) pair that puts the most
/// inlier correspondences in front of both cameras. Legacy gates only
/// ([`CheiralityOptions::default`]).
pub fn recover_relative_pose(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
) -> Option<(UnitQuaternion<f64>, Vector3<f64>)> {
    let recovered = recover_relative_pose_with_options(
        essential,
        correspondences,
        camera,
        inliers,
        &CheiralityOptions::default(),
    )?;
    Some((recovered.rotation, recovered.translation_unit))
}

/// Decompose an essential matrix with explicit [`CheiralityOptions`] gates.
/// Returns `None` when every hypothesis fails the positive-depth test, when
/// the winner's margin against the runner-up is too thin (ambiguous
/// chirality), or when too few inliers pass the depth/angle gates.
pub fn recover_relative_pose_with_options(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
    options: &CheiralityOptions,
) -> Option<RelativePoseRecovery> {
    let svd = essential.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;

    let w = Matrix3::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let mut r1 = u * w * v_t;
    let mut r2 = u * w.transpose() * v_t;
    if r1.determinant() < 0.0 {
        r1 = -r1;
    }
    if r2.determinant() < 0.0 {
        r2 = -r2;
    }
    let t_unit = u.column(2).into_owned();

    let candidates = [(r1, t_unit), (r1, -t_unit), (r2, t_unit), (r2, -t_unit)];
    let mut ranked: Vec<(i64, Matrix3<f64>, Vector3<f64>)> = Vec::with_capacity(4);
    let min_angle_rad = options.min_tri_angle_deg.to_radians();
    for (rotation, translation) in candidates {
        let score = cheirality_score(
            &rotation,
            &translation,
            correspondences,
            camera,
            inliers,
            min_angle_rad,
        );
        if score > 0 {
            ranked.push((score, rotation, translation));
        }
    }
    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    let (best_score, rotation, translation) = ranked.first().copied()?;
    let second_score = ranked.get(1).map(|(s, _, _)| *s).unwrap_or(0);
    if options.min_positive_depth_fraction > 0.0 {
        let required = (options.min_positive_depth_fraction * inliers.len() as f64).ceil() as i64;
        if best_score < required {
            return None;
        }
    }
    // Ambiguity gate: when the runner-up scores nearly as well as the winner,
    // the essential is chirality-ambiguous (typical on repetitive façades) and
    // the recovered translation direction is not trustworthy as a view-graph
    // bearing. `max_ambiguity_ratio == 1.0` never rejects on ratio alone.
    // Callers that want multi-hypothesis edges should use
    // `max_ambiguity_ratio = 1.0` and read `alternate`.
    if options.max_ambiguity_ratio < 1.0
        && second_score > 0
        && (second_score as f64) > (best_score as f64) * options.max_ambiguity_ratio
    {
        return None;
    }
    let rotation = UnitQuaternion::from_matrix(&rotation);
    let alternate = ranked.get(1).and_then(|(score, r, t)| {
        if *score <= 0 {
            return None;
        }
        let r = UnitQuaternion::from_matrix(r);
        // Keep only if it is a meaningfully different pose (different R or
        // anti-aligned t).
        let rot_diff = r.rotation_to(&rotation).angle();
        let t_anti = t.dot(&translation) < 0.0;
        if rot_diff > 1e-2 || t_anti {
            Some((r, *t))
        } else {
            None
        }
    });
    Some(RelativePoseRecovery {
        rotation,
        translation_unit: translation,
        best_score,
        second_score: second_score.max(0),
        alternate,
    })
}

fn cheirality_score(
    rotation: &Matrix3<f64>,
    translation: &Vector3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
    min_tri_angle_rad: f64,
) -> i64 {
    let p_prev = Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let mut p_curr = Matrix3x4::zeros();
    p_curr.fixed_view_mut::<3, 3>(0, 0).copy_from(rotation);
    p_curr.fixed_view_mut::<3, 1>(0, 3).copy_from(translation);

    // Camera-2 centre in camera-1 coordinates: C2 = −Rᵀ t.
    let cam2_centre = -(rotation.transpose() * translation);

    let mut score: i64 = 0;
    for &index in inliers {
        let correspondence = &correspondences[index];
        let Some(prev) = camera.normalize_pixel(&correspondence.previous_xy) else {
            continue;
        };
        let Some(curr) = camera.normalize_pixel(&correspondence.current_xy) else {
            continue;
        };

        let mut a = DMatrix::<f64>::zeros(4, 4);
        for column in 0..4 {
            a[(0, column)] = prev.x * p_prev[(2, column)] - p_prev[(0, column)];
            a[(1, column)] = prev.y * p_prev[(2, column)] - p_prev[(1, column)];
            a[(2, column)] = curr.x * p_curr[(2, column)] - p_curr[(0, column)];
            a[(3, column)] = curr.y * p_curr[(2, column)] - p_curr[(1, column)];
        }
        let svd = a.svd(true, true);
        let Some(v_t) = svd.v_t else {
            continue;
        };
        let solution = v_t.row(v_t.nrows() - 1);
        let w = solution[3];
        if w.abs() < 1e-12 {
            continue;
        }
        let world = Vector3::new(solution[0] / w, solution[1] / w, solution[2] / w);
        let camera_curr = rotation * world + translation;
        if world.z <= 0.0 || camera_curr.z <= 0.0 {
            continue;
        }
        if min_tri_angle_rad > 0.0 {
            let ray1 = world;
            let ray2 = world - cam2_centre;
            let n1 = ray1.norm();
            let n2 = ray2.norm();
            if n1 < 1e-12 || n2 < 1e-12 {
                continue;
            }
            let cos = (ray1.dot(&ray2) / (n1 * n2)).clamp(-1.0, 1.0);
            if cos.acos() < min_tri_angle_rad {
                continue;
            }
        }
        score += 1;
    }
    score
}

/// Hartley normalization for a set of 2D points: translate so the centroid is
/// at the origin and scale so the average distance to the origin is sqrt(2).
/// Returns the 3x3 transform `T` (so `T * [x, y, 1]` gives the normalized
/// point) and the normalized points themselves.
fn hartley_normalization<I>(points: I) -> Option<(Matrix3<f64>, Vec<Point2<f64>>)>
where
    I: IntoIterator<Item = Point2<f64>>,
{
    let collected: Vec<Point2<f64>> = points.into_iter().collect();
    if collected.is_empty() {
        return None;
    }

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    for point in &collected {
        mean_x += point.x;
        mean_y += point.y;
    }
    let count = collected.len() as f64;
    mean_x /= count;
    mean_y /= count;

    let mut mean_distance = 0.0;
    for point in &collected {
        let dx = point.x - mean_x;
        let dy = point.y - mean_y;
        mean_distance += (dx * dx + dy * dy).sqrt();
    }
    mean_distance /= count;
    if mean_distance < 1.0e-12 {
        return None;
    }
    let scale = std::f64::consts::SQRT_2 / mean_distance;
    let transform = Matrix3::new(
        scale,
        0.0,
        -scale * mean_x,
        0.0,
        scale,
        -scale * mean_y,
        0.0,
        0.0,
        1.0,
    );

    let normalized = collected
        .into_iter()
        .map(|point| Point2::new(scale * (point.x - mean_x), scale * (point.y - mean_y)))
        .collect();
    Some((transform, normalized))
}

fn normalize_pairs(
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
) -> Option<Vec<(Point2<f64>, Point2<f64>)>> {
    correspondences
        .iter()
        .map(|correspondence| {
            Some((
                camera.normalize_pixel(&correspondence.previous_xy)?,
                camera.normalize_pixel(&correspondence.current_xy)?,
            ))
        })
        .collect()
}

fn normalize_correspondences_with_cameras(
    correspondences: &[TwoViewCorrespondence],
    previous_camera: &Camera,
    current_camera: &Camera,
) -> Option<Vec<TwoViewCorrespondence>> {
    correspondences
        .iter()
        .map(|correspondence| {
            Some(TwoViewCorrespondence::new(
                previous_camera.normalize_pixel(&correspondence.previous_xy)?,
                current_camera.normalize_pixel(&correspondence.current_xy)?,
            ))
        })
        .collect()
}

fn score_inliers(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    threshold_sq: f64,
) -> Vec<usize> {
    let mut inliers = Vec::with_capacity(correspondences.len());
    for (index, correspondence) in correspondences.iter().enumerate() {
        let Some(distance_sq) = sampson_distance_squared(essential, correspondence, camera) else {
            continue;
        };
        if distance_sq <= threshold_sq {
            inliers.push(index);
        }
    }
    inliers
}

fn mean_sampson_error(
    essential: &Matrix3<f64>,
    correspondences: &[TwoViewCorrespondence],
    camera: &Camera,
    inliers: &[usize],
) -> f64 {
    if inliers.is_empty() {
        return f64::INFINITY;
    }
    let mut total = 0.0;
    let mut count = 0.0;
    for &index in inliers {
        if let Some(distance_sq) =
            sampson_distance_squared(essential, &correspondences[index], camera)
        {
            total += distance_sq.sqrt();
            count += 1.0;
        }
    }
    if count > 0.0 {
        total / count
    } else {
        f64::INFINITY
    }
}

fn sampson_distance_squared(
    essential: &Matrix3<f64>,
    correspondence: &TwoViewCorrespondence,
    camera: &Camera,
) -> Option<f64> {
    let prev = camera.normalize_pixel(&correspondence.previous_xy)?;
    let curr = camera.normalize_pixel(&correspondence.current_xy)?;
    let prev_h = Vector3::new(prev.x, prev.y, 1.0);
    let curr_h = Vector3::new(curr.x, curr.y, 1.0);
    let e_prev = essential * prev_h;
    let et_curr = essential.transpose() * curr_h;
    let numerator = curr_h.dot(&e_prev).powi(2);
    let denominator = e_prev.x.powi(2) + e_prev.y.powi(2) + et_curr.x.powi(2) + et_curr.y.powi(2);
    if denominator < 1e-18 {
        return None;
    }
    Some(numerator / denominator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point3, UnitQuaternion};
    use visloc_core::geometry::Pose;

    fn synthetic_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn project(pose: &Pose, camera: &Camera, point: &Point3<f64>) -> Point2<f64> {
        camera
            .project(&pose.transform_world_point(point))
            .expect("synthetic point must project in front of the camera")
    }

    fn synthetic_world_points() -> Vec<Point3<f64>> {
        vec![
            Point3::new(-1.0, -1.0, 5.0),
            Point3::new(1.0, -1.0, 5.0),
            Point3::new(-1.0, 1.0, 5.0),
            Point3::new(1.0, 1.0, 5.0),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.5, -0.25, 6.0),
            Point3::new(-0.7, 0.4, 4.5),
            Point3::new(0.6, 0.8, 5.5),
            Point3::new(-0.3, -0.6, 4.8),
            Point3::new(0.2, 0.2, 6.5),
        ]
    }

    fn correspondences(
        previous_pose: &Pose,
        current_pose: &Pose,
        camera: &Camera,
        points: &[Point3<f64>],
    ) -> Vec<TwoViewCorrespondence> {
        points
            .iter()
            .map(|point| TwoViewCorrespondence {
                previous_xy: project(previous_pose, camera, point),
                current_xy: project(current_pose, camera, point),
            })
            .collect()
    }

    #[test]
    fn essential_ransac_recovers_pure_translation() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());

        let estimator = RelativePoseEstimator::default();
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, 0.3)
            .expect("relative pose must be recovered");

        let translation = pose.previous_to_current.translation;
        assert!(
            (translation - Vector3::new(-0.3, 0.0, 0.0)).norm() < 5.0e-3,
            "translation drifted: {translation:?}"
        );
        let rotation = pose.previous_to_current.rotation.angle();
        assert!(
            rotation < 5.0e-3,
            "rotation should be near zero: {rotation}"
        );
        assert!(pose.inliers.len() >= 8);
        assert!(pose.mean_sampson_error < 5.0e-3);
    }

    #[test]
    fn essential_ransac_accepts_distinct_pinhole_cameras() {
        // The two endpoint images deliberately have different sizes and
        // focal lengths.  A shared-camera call would interpret the right
        // pixels with the wrong bearing; the camera-aware entry point must
        // normalize each endpoint before running the unchanged estimator.
        let previous_camera = synthetic_camera();
        let current_camera = Camera::pinhole(2, 800, 600, 700.0, 680.0, 400.0, 300.0);
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences = synthetic_world_points()
            .iter()
            .map(|point| {
                TwoViewCorrespondence::new(
                    project(&previous, &previous_camera, point),
                    project(&current, &current_camera, point),
                )
            })
            .collect::<Vec<_>>();

        let estimated = RelativePoseEstimator::default()
            .estimate_with_cameras(&correspondences, &previous_camera, &current_camera)
            .expect("distinct pinhole calibrations must still recover a pose");
        assert!(estimated.inliers.len() >= 8);
        assert!(estimated.previous_to_current.rotation.angle() < 5.0e-3);
        assert!(
            (estimated.translation_unit - Vector3::new(-1.0, 0.0, 0.0)).norm() < 5.0e-3,
            "translation direction drifted: {:?}",
            estimated.translation_unit
        );
    }

    #[test]
    fn essential_ransac_recovers_translation_with_yaw() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.05);
        let current_world_to_camera = SE3::new(yaw, Vector3::new(-0.2, 0.0, -0.05));
        let current = Pose {
            world_to_camera: current_world_to_camera.clone(),
        };
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());

        let estimator = RelativePoseEstimator::default();
        let scale = current_world_to_camera.translation.norm();
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, scale)
            .expect("relative pose must be recovered");

        assert!(pose.inliers.len() >= 8);
        let translation_error =
            (pose.previous_to_current.translation - current_world_to_camera.translation).norm();
        assert!(
            translation_error < 5.0e-3,
            "translation drifted: error={translation_error}"
        );
        let rotation_error = pose
            .previous_to_current
            .rotation
            .rotation_to(&current_world_to_camera.rotation)
            .angle()
            .abs();
        assert!(
            rotation_error < 5.0e-3,
            "rotation drifted: error_rad={rotation_error}"
        );
    }

    #[test]
    fn essential_ransac_recovers_pure_translation_with_eight_points() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.30, 0.0, 0.0));
        let points = [
            Point3::new(-1.0, -1.0, 4.5),
            Point3::new(1.0, -1.0, 4.6),
            Point3::new(-1.0, 1.0, 5.5),
            Point3::new(1.0, 1.0, 5.4),
            Point3::new(0.0, 0.0, 5.0),
            Point3::new(0.5, -0.25, 6.0),
            Point3::new(-0.6, 0.4, 4.8),
            Point3::new(0.4, 0.7, 5.2),
        ];
        let correspondences = correspondences(&previous, &current, &camera, &points);

        let estimator = RelativePoseEstimator::default();
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, 0.30)
            .expect("relative pose must be recovered");
        let translation = pose.previous_to_current.translation;
        assert!(
            (translation - Vector3::new(-0.30, 0.0, 0.0)).norm() < 5.0e-3,
            "translation drifted: {translation:?}"
        );
        assert!(pose.inliers.len() >= 8);
    }

    #[test]
    fn essential_ransac_returns_none_for_too_few_points() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.2, 0.0, 0.0));
        let mut points = synthetic_world_points();
        points.truncate(6);
        let correspondences = correspondences(&previous, &current, &camera, &points);

        let estimator = RelativePoseEstimator::default();
        assert!(estimator.estimate(&correspondences, &camera).is_none());
    }

    #[test]
    fn weighted_ransac_recovers_pose_with_correctly_ordered_confidence_weights() {
        // 12 inlier correspondences from a real essential geometry, plus
        // 18 outlier correspondences sprinkled in random positions. The
        // inliers carry confidence ~ 0.9, the outliers ~ 0.05. The
        // weighted estimator must recover the same pose as the uniform
        // path (this checks that the PROSAC ordering doesn't break clean
        // inputs and that the confidence-driven priority correctly anchors
        // the early iterations on the inliers).
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let mut correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());
        let n_inliers = correspondences.len();
        let mut weights: Vec<f32> = vec![0.9; n_inliers];

        // Inject outlier correspondences whose pixel positions are
        // shuffled — they cannot satisfy the true essential geometry.
        let outlier_seeds = [
            (50.0, 60.0, 200.0, 90.0),
            (310.0, 100.0, 70.0, 410.0),
            (250.0, 200.0, 100.0, 100.0),
            (440.0, 330.0, 200.0, 60.0),
            (90.0, 350.0, 320.0, 240.0),
            (10.0, 410.0, 540.0, 430.0),
            (280.0, 70.0, 110.0, 200.0),
            (190.0, 30.0, 460.0, 160.0),
            (370.0, 250.0, 50.0, 50.0),
            (230.0, 410.0, 600.0, 100.0),
            (100.0, 110.0, 300.0, 350.0),
            (420.0, 90.0, 30.0, 220.0),
            (150.0, 300.0, 480.0, 60.0),
            (350.0, 150.0, 70.0, 320.0),
            (60.0, 250.0, 230.0, 70.0),
            (270.0, 380.0, 540.0, 290.0),
            (390.0, 20.0, 90.0, 270.0),
            (210.0, 170.0, 470.0, 380.0),
        ];
        for (px, py, cx, cy) in outlier_seeds {
            correspondences.push(TwoViewCorrespondence::new(
                nalgebra::Point2::new(px, py),
                nalgebra::Point2::new(cx, cy),
            ));
            weights.push(0.05);
        }
        assert_eq!(weights.len(), correspondences.len());

        let ransac = EssentialRansac {
            estimator: EightPointEssentialMatrixEstimator::default(),
            config: EssentialRansacConfig {
                iterations: 64,
                sampson_threshold: 5.0e-3,
                seed: 11,
            },
        };

        let weighted_report = ransac
            .estimate_with_weights(&correspondences, &camera, &weights)
            .expect("weighted RANSAC must recover a model from clean inliers");

        // The first n_inliers correspondences are the geometric inliers;
        // the weighted RANSAC should recover essentially all of them.
        let recovered_inliers: usize = weighted_report
            .inliers
            .iter()
            .filter(|&&i| i < n_inliers)
            .count();
        assert!(
            recovered_inliers >= n_inliers - 1,
            "weighted RANSAC should recover the geometric inliers, got {recovered_inliers}/{n_inliers}"
        );
        assert!(weighted_report.mean_sampson_error < 5.0e-3);
    }

    #[test]
    fn weighted_ransac_falls_back_to_uniform_when_weights_are_all_zero() {
        // Same clean correspondences, but all weights are zero — the
        // PROSAC ordering should fall back to the uniform sampling path
        // (no iteration_progress shrinking) and the recovery should
        // match the unweighted estimate.
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());
        let weights: Vec<f32> = vec![0.0; correspondences.len()];

        let ransac = EssentialRansac::default();
        let unweighted = ransac.estimate(&correspondences, &camera).unwrap();
        let weighted = ransac
            .estimate_with_weights(&correspondences, &camera, &weights)
            .unwrap();
        assert_eq!(unweighted.inliers, weighted.inliers);
    }

    #[test]
    fn hardened_cheirality_recovers_clean_translation() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());
        let estimator = RelativePoseEstimator {
            cheirality: CheiralityOptions::hardened(),
            ..RelativePoseEstimator::default()
        };
        let pose = estimator
            .estimate_with_scale(&correspondences, &camera, 0.3)
            .expect("hardened recovery must succeed on a clean baseline");
        assert!(
            (pose.previous_to_current.translation - Vector3::new(-0.3, 0.0, 0.0)).norm() < 5e-3
        );
    }

    #[test]
    fn hardened_cheirality_rejects_near_planar_pure_rotation() {
        // Near-zero baseline: every (R, ±t) hypothesis triangulates with tiny
        // angles, so the angle gate + ambiguity ratio should refuse recovery.
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.02);
        let current = Pose::from_world_to_camera(yaw, Vector3::new(-1e-4, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());
        let report = EssentialRansac::default()
            .estimate(&correspondences, &camera)
            .expect("essential still fits a near-planar pair");
        let recovered = recover_relative_pose_with_options(
            &report.essential,
            &correspondences,
            &camera,
            &report.inliers,
            &CheiralityOptions::hardened(),
        );
        assert!(
            recovered.is_none(),
            "pure-rotation / tiny-baseline essentials must fail hardened gates"
        );
    }

    #[test]
    fn chirality_margin_is_high_on_clean_baseline() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());
        let report = EssentialRansac::default()
            .estimate(&correspondences, &camera)
            .unwrap();
        let recovered = recover_relative_pose_with_options(
            &report.essential,
            &correspondences,
            &camera,
            &report.inliers,
            &CheiralityOptions::default(),
        )
        .unwrap();
        assert!(
            recovered.chirality_margin() > 0.5,
            "clean baseline should separate winner from runner-up, margin={}",
            recovered.chirality_margin()
        );
    }

    #[test]
    fn hardened_keep_ambiguous_exposes_alternate_on_clean_pair() {
        // Even a clean pair has a runner-up (usually −t or the other R); with
        // keep-ambiguous gates that runner-up is returned rather than rejected.
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences(&previous, &current, &camera, &synthetic_world_points());
        let report = EssentialRansac::default()
            .estimate(&correspondences, &camera)
            .unwrap();
        let recovered = recover_relative_pose_with_options(
            &report.essential,
            &correspondences,
            &camera,
            &report.inliers,
            &CheiralityOptions::hardened_keep_ambiguous(),
        )
        .expect("clean pair must recover under keep-ambiguous");
        assert!(recovered.best_score > 0);
        // Runner-up score may be low; alternate is only set when it differs.
        if recovered.second_score > 0 {
            assert!(
                recovered.alternate.is_some() || recovered.chirality_margin() > 0.99,
                "expected an alternate or a decisive margin"
            );
        }
    }
}
