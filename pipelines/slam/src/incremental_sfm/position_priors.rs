//! GPS / position priors for incremental SfM, after COLMAP's
//! `use_prior_position` mapper (`PosePriorBundleAdjuster`, COLMAP ≥ 3.10, and
//! the `colmap-gps` fork it grew from).
//!
//! Each image may carry a prior camera centre in a metric Cartesian frame
//! (typically a local East-North-Up frame built from its GPS fix, see
//! `visloc_io::gps_prior`) with a per-axis standard deviation. Once at least
//! [`SfmPositionPriors::min_num_priors`] registered images have one, every
//! global / periodic bundle adjustment of [`crate::incremental_sfm`]:
//!
//! 1. **Aligns** the reconstruction to the priors with a robust Sim(3)
//!    (deterministic RANSAC over three-point Umeyama fits, then a refit on the
//!    inliers). The inlier threshold is
//!    [`SfmPositionPriors::alignment_max_error`], by default COLMAP's
//!    `3 · mean(σ)`. The monocular reconstruction leaves the arbitrary scale
//!    and frame of its seed pair and lands in the metric prior frame.
//! 2. **Constrains** the BA with a covariance-weighted camera-centre residual
//!    `(C − p) / σ` per prior image (the existing
//!    [`crate::PositionPrior`] factor, weight `1/σ²` per axis, in the same
//!    squared-pixel units as the reprojection terms). The priors fix the
//!    7-DoF gauge, so the anchor and scale-anchor poses of the ordinary BA are
//!    left free and GPS can correct the scale and bending drift of a long
//!    capture.
//!
//! As in COLMAP, every prior enters the BA, outliers of the alignment
//! included: a camera can disagree with its prior because the *model* is
//! wrong there (drift, a bad registration), and its prior is what pulls it
//! back. Bad fixes (a GPS jump) are handled by a Cauchy loss on the
//! normalised residual ([`SfmPositionPriors::robust_loss_scale`], COLMAP's
//! `use_robust_loss_on_prior_position`).
//!
//! Deviations from COLMAP:
//!
//! - The Cauchy loss is on by default (COLMAP's default is plain least
//!   squares). With COLMAP-compatible features, on a drone block with real
//!   onboard GNSS the two are equal, and on simulated GPS with 30 m jumps the
//!   Cauchy loss keeps the model at 0.30 m where least squares leaves 1.21 m
//!   (`docs/gps_prior_sfm.md`).

//! - The Cauchy loss is applied by reweighting: each BA call weights a prior
//!   by `1 / (1 + e²/c²)` for its normalised residual `e` at the start of that
//!   call. The mapper runs many BAs, which makes this iteratively reweighted
//!   least squares.
//! - [`SfmPositionPriors::reject_outliers`] (off by default) drops the
//!   alignment's outliers from the BA instead.
//! - When the inlier priors are close to collinear (a straight drive), GPS
//!   does not observe the roll about the line: the first registered pose's
//!   rotation is then held fixed in the BA to keep the problem regular.
//! - The alignment is deterministic (a fixed-seed sampler), so a run is
//!   reproducible.

use nalgebra::{Matrix3, Point3, Rotation3, UnitQuaternion, Vector3};
use visloc_core::geometry::Pose;
use visloc_tracking::umeyama_similarity_transform;

/// One image's position prior.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePositionPrior {
    /// Prior camera centre in a metric Cartesian frame (e.g. local ENU, metres).
    pub position: Point3<f64>,
    /// Per-axis 1-σ standard deviation of `position`, in the same units.
    /// Non-positive or non-finite entries drop that axis from the BA prior.
    pub std_dev: Vector3<f64>,
}

impl ImagePositionPrior {
    /// A prior with the same standard deviation on every axis.
    pub fn isotropic(position: Point3<f64>, std_dev: f64) -> Self {
        Self {
            position,
            std_dev: Vector3::repeat(std_dev),
        }
    }

    /// Mean of the valid per-axis standard deviations, if any.
    fn mean_std_dev(&self) -> Option<f64> {
        let valid: Vec<f64> = self
            .std_dev
            .iter()
            .copied()
            .filter(|s| s.is_finite() && *s > 0.0)
            .collect();
        (!valid.is_empty()).then(|| valid.iter().sum::<f64>() / valid.len() as f64)
    }

    /// Per-axis BA weights `1/σ²` (zero for an invalid axis).
    fn axis_weights(&self) -> Vector3<f64> {
        self.std_dev.map(|s| {
            if s.is_finite() && s > 0.0 {
                1.0 / (s * s)
            } else {
                0.0
            }
        })
    }
}

/// `sqrt(χ²₀.₉₅(3))`: COLMAP's default Cauchy scale on a covariance-normalised
/// position residual.
pub const DEFAULT_PRIOR_LOSS_SCALE: f64 = 2.795_483_5;

/// Position priors and their options for [`crate::incremental_sfm`]
/// (`IncrementalSfmConfig::position_priors`).
#[derive(Debug, Clone, PartialEq)]
pub struct SfmPositionPriors {
    /// Indexed like the SfM's `features`; `None` for an image without a prior.
    /// A shorter vector leaves the remaining images without priors.
    pub priors: Vec<Option<ImagePositionPrior>>,
    /// Registered images with a prior needed before the priors are used
    /// (at least 3, the minimum a Sim(3) needs).
    pub min_num_priors: usize,
    /// Inlier threshold of the robust Sim(3) alignment, in prior units.
    /// `None`: `3 · mean σ` over the priors (COLMAP's default).
    pub alignment_max_error: Option<f64>,
    /// Leave the alignment's outlier priors out of the BA prior factor
    /// (default off: a prior the model disagrees with may be the one that
    /// corrects it).
    pub reject_outliers: bool,
    /// Cauchy loss scale `c` on the normalised prior residual
    /// `e = ‖(C − p) / σ‖`, applied by reweighting each BA's prior by
    /// `1 / (1 + e²/c²)`. Default [`DEFAULT_PRIOR_LOSS_SCALE`] (COLMAP's
    /// `prior_position_loss_scale`); `None`: plain least squares, COLMAP's
    /// default.
    pub robust_loss_scale: Option<f64>,
    /// RANSAC iterations of the alignment.
    pub ransac_iterations: usize,
}

impl SfmPositionPriors {
    /// Priors with the default options.
    pub const fn new(priors: Vec<Option<ImagePositionPrior>>) -> Self {
        Self {
            priors,
            min_num_priors: 3,
            alignment_max_error: None,
            reject_outliers: false,
            robust_loss_scale: Some(DEFAULT_PRIOR_LOSS_SCALE),
            ransac_iterations: 1000,
        }
    }

    /// Priors from `visloc_io::gps_prior` (e.g. EXIF GPS or a prior file
    /// converted to local ENU), with the default options.
    pub fn from_local_priors(priors: &[Option<visloc_io::gps_prior::LocalPositionPrior>]) -> Self {
        Self::new(
            priors
                .iter()
                .map(|prior| {
                    prior.map(|p| ImagePositionPrior {
                        position: p.position,
                        std_dev: p.std_dev,
                    })
                })
                .collect(),
        )
    }

    /// Number of images with a prior.
    pub fn num_priors(&self) -> usize {
        self.priors.iter().filter(|p| p.is_some()).count()
    }

    fn get(&self, image: usize) -> Option<&ImagePositionPrior> {
        self.priors.get(image).and_then(Option::as_ref)
    }

    /// The alignment's inlier threshold for these priors.
    pub fn effective_alignment_max_error(&self) -> Option<f64> {
        if let Some(max_error) = self.alignment_max_error {
            return (max_error.is_finite() && max_error > 0.0).then_some(max_error);
        }
        let sigmas: Vec<f64> = self
            .priors
            .iter()
            .flatten()
            .filter_map(ImagePositionPrior::mean_std_dev)
            .collect();
        (!sigmas.is_empty()).then(|| 3.0 * sigmas.iter().sum::<f64>() / sigmas.len() as f64)
    }
}

/// A similarity `x ↦ scale · rotation · x + translation` taking the
/// reconstruction frame to the prior frame.
#[derive(Debug, Clone, PartialEq)]
pub struct PriorSimilarity {
    pub scale: f64,
    pub rotation: Rotation3<f64>,
    pub translation: Vector3<f64>,
}

impl PriorSimilarity {
    pub fn transform_point(&self, point: &Point3<f64>) -> Point3<f64> {
        Point3::from(self.scale * (self.rotation * point.coords) + self.translation)
    }

    /// The same camera expressed in the transformed world frame.
    pub fn transform_pose(&self, pose: &Pose) -> Pose {
        let center = self.transform_point(&pose.camera_center_world());
        let rotation = pose.world_to_camera.rotation
            * UnitQuaternion::from_rotation_matrix(&self.rotation).inverse();
        let translation = -(rotation * center.coords);
        Pose::from_world_to_camera(rotation, translation)
    }
}

/// Outcome of [`align_to_position_priors`].
#[derive(Debug, Clone, PartialEq)]
pub struct PositionPriorAlignment {
    /// Reconstruction frame → prior frame.
    pub similarity: PriorSimilarity,
    /// Registered images with a prior.
    pub num_candidates: usize,
    /// Images whose prior agrees with the aligned reconstruction.
    pub inliers: Vec<usize>,
    /// RMS centre-to-prior distance over the inliers, after alignment.
    pub inlier_rmse: f64,
    /// Inlier threshold used.
    pub max_error: f64,
    /// The inlier priors are close to collinear: GPS does not observe the
    /// roll about their line.
    pub degenerate: bool,
}

/// SplitMix64: a tiny fixed-seed sampler so the alignment is reproducible.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Ratio of the second to the first singular value of the centred points:
/// near zero for (close to) collinear points.
fn planarity_ratio(points: &[Point3<f64>]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let mean = points
        .iter()
        .fold(Vector3::zeros(), |acc, p| acc + p.coords)
        / points.len() as f64;
    let mut cov = Matrix3::zeros();
    for p in points {
        let d = p.coords - mean;
        cov += d * d.transpose();
    }
    let mut eig: Vec<f64> = cov
        .symmetric_eigenvalues()
        .iter()
        .map(|v| v.max(0.0))
        .collect();
    eig.sort_by(|a, b| b.total_cmp(a));
    if eig[0] <= 0.0 {
        return 0.0;
    }
    (eig[1] / eig[0]).sqrt()
}

/// Below this second/first singular-value ratio the priors count as collinear.
const COLLINEAR_RATIO: f64 = 1e-2;

fn fit(source: &[Point3<f64>], target: &[Point3<f64>]) -> Option<PriorSimilarity> {
    let fit = umeyama_similarity_transform(source, target, true)?;
    let ok = fit.scale.is_finite()
        && fit.scale > 0.0
        && fit.translation.iter().all(|v| v.is_finite())
        && fit.rotation.matrix().iter().all(|v| v.is_finite());
    ok.then_some(PriorSimilarity {
        scale: fit.scale,
        rotation: fit.rotation,
        translation: fit.translation,
    })
}

/// Robustly estimate the similarity taking the registered camera centres onto
/// their priors. Returns `None` with fewer than `priors.min_num_priors`
/// (at least 3) registered prior images, without a usable threshold, or when
/// no sample yields three inliers.
pub fn estimate_position_prior_alignment(
    poses: &[Option<Pose>],
    priors: &SfmPositionPriors,
) -> Option<PositionPriorAlignment> {
    let max_error = priors.effective_alignment_max_error()?;
    let mut images = Vec::new();
    let mut source = Vec::new();
    let mut target = Vec::new();
    for (image, pose) in poses.iter().enumerate() {
        let (Some(pose), Some(prior)) = (pose.as_ref(), priors.get(image)) else {
            continue;
        };
        let center = pose.camera_center_world();
        if !center
            .coords
            .iter()
            .chain(prior.position.coords.iter())
            .all(|v| v.is_finite())
        {
            continue;
        }
        images.push(image);
        source.push(center);
        target.push(prior.position);
    }
    let n = images.len();
    if n < priors.min_num_priors.max(3) {
        return None;
    }

    let inliers_of = |sim: &PriorSimilarity| -> (Vec<usize>, f64) {
        let mut inliers = Vec::new();
        let mut sum_sq = 0.0;
        for k in 0..n {
            let r2 = (sim.transform_point(&source[k]) - target[k]).norm_squared();
            if r2 <= max_error * max_error {
                inliers.push(k);
                sum_sq += r2;
            }
        }
        (inliers, sum_sq)
    };
    let better = |a: &(Vec<usize>, f64), b: &(Vec<usize>, f64)| {
        a.0.len() > b.0.len() || (a.0.len() == b.0.len() && a.1 < b.1)
    };

    // Every subset when that is cheaper than the sampling budget.
    let mut best: Option<(PriorSimilarity, (Vec<usize>, f64))> = None;
    let consider = |sample: [usize; 3], best: &mut Option<(PriorSimilarity, (Vec<usize>, f64))>| {
        let src: Vec<Point3<f64>> = sample.iter().map(|&k| source[k]).collect();
        let dst: Vec<Point3<f64>> = sample.iter().map(|&k| target[k]).collect();
        if planarity_ratio(&src) < COLLINEAR_RATIO || planarity_ratio(&dst) < COLLINEAR_RATIO {
            // Collinear samples leave the roll free; fall back to them only
            // through the refit below (which sees all inliers).
            return;
        }
        let Some(sim) = fit(&src, &dst) else { return };
        let score = inliers_of(&sim);
        if best.as_ref().is_none_or(|(_, s)| better(&score, s)) {
            *best = Some((sim, score));
        }
    };
    let exhaustive = n * (n - 1) * (n - 2) / 6 <= priors.ransac_iterations;
    if exhaustive {
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    consider([a, b, c], &mut best);
                }
            }
        }
    } else {
        let mut rng = SplitMix64(0x5EED_6E55_0000_0001);
        for _ in 0..priors.ransac_iterations {
            let a = rng.below(n);
            let mut b = rng.below(n);
            while b == a {
                b = rng.below(n);
            }
            let mut c = rng.below(n);
            while c == a || c == b {
                c = rng.below(n);
            }
            consider([a, b, c], &mut best);
        }
    }
    // All priors (nearly) collinear: no non-degenerate sample exists. Fit
    // everything; the roll about the line is then arbitrary.
    let (mut sim, mut score) = match best {
        Some(found) => found,
        None => {
            let sim = fit(&source, &target)?;
            let score = inliers_of(&sim);
            (sim, score)
        }
    };
    // Local optimisation: refit on the inliers until the set stops growing.
    for _ in 0..5 {
        if score.0.len() < 3 {
            break;
        }
        let src: Vec<Point3<f64>> = score.0.iter().map(|&k| source[k]).collect();
        let dst: Vec<Point3<f64>> = score.0.iter().map(|&k| target[k]).collect();
        let Some(refit) = fit(&src, &dst) else { break };
        let refit_score = inliers_of(&refit);
        if refit_score.0.len() < score.0.len() {
            break;
        }
        let same = refit_score.0 == score.0;
        sim = refit;
        score = refit_score;
        if same {
            break;
        }
    }
    if score.0.len() < 3 {
        return None;
    }
    let inlier_points: Vec<Point3<f64>> = score.0.iter().map(|&k| target[k]).collect();
    let degenerate = planarity_ratio(&inlier_points) < COLLINEAR_RATIO;
    let inlier_rmse = (score.1 / score.0.len() as f64).sqrt();
    Some(PositionPriorAlignment {
        similarity: sim,
        num_candidates: n,
        inliers: score.0.iter().map(|&k| images[k]).collect(),
        inlier_rmse,
        max_error,
        degenerate,
    })
}

/// Apply a similarity to every registered pose and triangulated point.
pub fn apply_prior_similarity(
    similarity: &PriorSimilarity,
    poses: &mut [Option<Pose>],
    points: &mut [Option<Point3<f64>>],
) {
    for pose in poses.iter_mut().flatten() {
        *pose = similarity.transform_pose(pose);
    }
    for point in points.iter_mut().flatten() {
        *point = similarity.transform_point(point);
    }
}

/// Robustly align a reconstruction to its position priors in place
/// ([`estimate_position_prior_alignment`] then [`apply_prior_similarity`]).
/// Leaves the inputs untouched and returns `None` when no alignment exists.
pub fn align_to_position_priors(
    poses: &mut [Option<Pose>],
    points: &mut [Option<Point3<f64>>],
    priors: &SfmPositionPriors,
) -> Option<PositionPriorAlignment> {
    let alignment = estimate_position_prior_alignment(poses, priors)?;
    apply_prior_similarity(&alignment.similarity, poses, points);
    Some(alignment)
}

/// The BA prior factor for an aligned reconstruction: one observation per
/// registered prior image (only the alignment's inliers when
/// `reject_outliers`), Cauchy-reweighted at the current residual when
/// `robust_loss_scale` is set.
pub(crate) fn position_prior_factor(
    poses: &[Option<Pose>],
    priors: &SfmPositionPriors,
    alignment: &PositionPriorAlignment,
) -> crate::PositionPrior {
    let mut factor = crate::PositionPrior::new();
    let mut push = |image: usize| {
        if poses.get(image).and_then(Option::as_ref).is_none() {
            return;
        }
        let Some(prior) = priors.get(image) else {
            return;
        };
        let mut axis_weights = prior.axis_weights();
        if axis_weights.iter().all(|w| *w == 0.0) {
            return;
        }
        if let Some(scale) = priors
            .robust_loss_scale
            .filter(|c| c.is_finite() && *c > 0.0)
        {
            let center = poses[image].as_ref().map(Pose::camera_center_world);
            if let Some(center) = center {
                let r = center - prior.position;
                let e2 = axis_weights.component_mul(&r.component_mul(&r)).sum();
                if e2.is_finite() {
                    axis_weights *= 1.0 / (1.0 + e2 / (scale * scale));
                }
            }
        }
        factor.push(crate::PositionPriorObservation {
            keyframe_id: image as u64,
            camera_center_world: prior.position,
            axis_weights,
        });
    };
    if priors.reject_outliers {
        for &image in &alignment.inliers {
            push(image);
        }
    } else {
        for image in 0..poses.len() {
            push(image);
        }
    }
    factor
}

/// `VISLOC_SFM_DEBUG`-style log switch for the prior path.
pub(crate) fn position_prior_debug_enabled() -> bool {
    std::env::var_os("VISLOC_SFM_DEBUG").is_some()
        || std::env::var_os("VISLOC_SFM_PRIOR_DEBUG").is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pose_at(center: Point3<f64>, yaw: f64) -> Pose {
        let q_c2w = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
        let q_w2c = q_c2w.inverse();
        Pose::from_world_to_camera(q_w2c, -(q_w2c * center.coords))
    }

    fn sample_sim() -> PriorSimilarity {
        PriorSimilarity {
            scale: 37.5,
            rotation: Rotation3::from_euler_angles(0.3, -0.7, 1.9),
            translation: Vector3::new(120.0, -45.0, 8.0),
        }
    }

    #[test]
    fn transform_pose_moves_centre_and_keeps_projections() {
        let sim = sample_sim();
        let pose = pose_at(Point3::new(0.4, -0.2, 1.3), 0.6);
        let point = Point3::new(0.1, 0.5, 4.0);
        let moved = sim.transform_pose(&pose);
        let c = moved.camera_center_world();
        let expected = sim.transform_point(&pose.camera_center_world());
        assert!((c - expected).norm() < 1e-9);
        // The camera-frame direction of a point is unchanged (depth scales).
        let before = pose.transform_world_point(&point).coords.normalize();
        let after = moved
            .transform_world_point(&sim.transform_point(&point))
            .coords
            .normalize();
        assert!((before - after).norm() < 1e-9);
    }

    #[test]
    fn alignment_recovers_similarity_and_rejects_gross_outliers() {
        let sim = sample_sim();
        let mut poses = Vec::new();
        let mut priors = Vec::new();
        for k in 0..20 {
            let t = k as f64 * 0.3;
            let center = Point3::new(t.cos() * 2.0, 0.1 * t, t.sin() * 2.0);
            poses.push(Some(pose_at(center, t)));
            let mut prior = sim.transform_point(&center);
            if k == 4 || k == 11 {
                prior.x += 500.0; // GPS jump
            }
            priors.push(Some(ImagePositionPrior::isotropic(prior, 1.0)));
        }
        let priors = SfmPositionPriors::new(priors);
        let alignment = estimate_position_prior_alignment(&poses, &priors).unwrap();
        assert_eq!(alignment.num_candidates, 20);
        assert_eq!(alignment.inliers.len(), 18);
        assert!(!alignment.inliers.contains(&4) && !alignment.inliers.contains(&11));
        assert!((alignment.similarity.scale - sim.scale).abs() < 1e-6);
        assert!(alignment.inlier_rmse < 1e-6);
        assert!(!alignment.degenerate);
        assert!((alignment.max_error - 3.0).abs() < 1e-12);

        // Every prior enters the BA by default; rejection is opt-in.
        let factor = position_prior_factor(&poses, &priors, &alignment);
        assert_eq!(factor.observations.len(), 20);
        let rejecting = SfmPositionPriors {
            reject_outliers: true,
            ..priors.clone()
        };
        assert_eq!(
            position_prior_factor(&poses, &rejecting, &alignment)
                .observations
                .len(),
            18
        );
        // The Cauchy loss leaves an agreeing prior at full weight and all but
        // silences the 500 m jump (normalised residual 500 against c = 2.8).
        let robust = SfmPositionPriors {
            robust_loss_scale: Some(DEFAULT_PRIOR_LOSS_SCALE),
            ..priors.clone()
        };
        let mut aligned = poses.clone();
        apply_prior_similarity(&alignment.similarity, &mut aligned, &mut []);
        let weighted = position_prior_factor(&aligned, &robust, &alignment);
        let weight = |image: u64| {
            weighted
                .observations
                .iter()
                .find(|o| o.keyframe_id == image)
                .unwrap()
                .axis_weights
                .x
        };
        assert!((weight(0) - 1.0).abs() < 1e-9);
        assert!(weight(4) < 1e-4);
    }

    #[test]
    fn alignment_is_deterministic_on_the_sampled_path() {
        let sim = sample_sim();
        let mut poses = Vec::new();
        let mut priors = Vec::new();
        for k in 0..60 {
            let t = k as f64 * 0.1;
            let center = Point3::new(t, (3.0 * t).sin(), 0.2 * t * t);
            poses.push(Some(pose_at(center, 0.0)));
            let noise = Vector3::new(
                ((k * 7919) % 13) as f64 / 13.0 - 0.5,
                ((k * 104_729) % 17) as f64 / 17.0 - 0.5,
                ((k * 1_299_709) % 19) as f64 / 19.0 - 0.5,
            );
            priors.push(Some(ImagePositionPrior::isotropic(
                sim.transform_point(&center) + noise,
                2.0,
            )));
        }
        let priors = SfmPositionPriors {
            ransac_iterations: 200,
            ..SfmPositionPriors::new(priors)
        };
        let a = estimate_position_prior_alignment(&poses, &priors).unwrap();
        let b = estimate_position_prior_alignment(&poses, &priors).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.inliers.len(), 60);
        assert!((a.similarity.scale / sim.scale - 1.0).abs() < 0.02);
    }

    #[test]
    fn alignment_needs_enough_registered_priors_and_flags_collinear_ones() {
        let sim = sample_sim();
        let line: Vec<Option<Pose>> = (0..5)
            .map(|k| Some(pose_at(Point3::new(k as f64, 0.0, 0.0), 0.0)))
            .collect();
        let line_priors: Vec<Option<ImagePositionPrior>> = (0..5)
            .map(|k| {
                Some(ImagePositionPrior::isotropic(
                    sim.transform_point(&Point3::new(k as f64, 0.0, 0.0)),
                    1.0,
                ))
            })
            .collect();
        let mut two = line_priors.clone();
        for prior in two.iter_mut().skip(2) {
            *prior = None;
        }
        assert!(estimate_position_prior_alignment(&line, &SfmPositionPriors::new(two)).is_none());
        let collinear =
            estimate_position_prior_alignment(&line, &SfmPositionPriors::new(line_priors)).unwrap();
        assert!(collinear.degenerate);
        assert_eq!(collinear.inliers.len(), 5);
        assert!((collinear.similarity.scale - sim.scale).abs() < 1e-6);
    }
}
