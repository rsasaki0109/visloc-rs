//! COLMAP-style two-view geometric verification with multi-model
//! classification (`TwoViewGeometry` / `ConfigurationType`).
//!
//! This module ports the *semantics* of COLMAP's estimate-three-models-and-
//! compare-inlier-ratios decision procedure — not a line-for-line C++
//! transliteration — from (all BSD-3-Clause, ETH Zurich / UNC Chapel Hill):
//! - `src/colmap/scene/two_view_geometry.h` — `TwoViewGeometry::ConfigurationType`.
//! - `src/colmap/estimators/two_view_geometry.h` — `TwoViewGeometryOptions`
//!   (every default threshold below cites its line).
//! - `src/colmap/estimators/two_view_geometry.cc` —
//!   `EstimateCalibratedTwoViewGeometry` (the E/F/H decision tree),
//!   `DetectWatermarkMatches`, `EstimateMultipleTwoViewGeometries`.
//!
//! **Scope note.** COLMAP has two top-level entry points:
//! `EstimateCalibratedTwoViewGeometry` (both cameras have a focal-length
//! prior) and `EstimateUncalibratedTwoViewGeometry` (at least one does not,
//! `two_view_geometry.cc:186-268`). Every caller in this repo already knows
//! its camera intrinsics — the unordered-SfM view graph in
//! `examples/unordered_sfm_demo.rs` shares one calibrated [`Camera`] across
//! the whole VLAD-retrieved photo collection — so only the **calibrated**
//! entry point has a caller here and is the one ported. Note that the
//! calibrated path still visits the `UNCALIBRATED` *configuration*: COLMAP's
//! own decision tree falls back to it whenever the shared-intrinsics prior
//! disagrees with what the fundamental matrix sees (the `E_F_inlier_ratio`
//! test below) — exactly the failure mode a single assumed cam-0 pinhole
//! prior shared across an ETH3D scene's several DSLR groups can hit.

use std::collections::HashSet;

use nalgebra::{Matrix3, Vector3};
use visloc_core::types::Camera;

use super::fundamental::{fundamental_ransac, FundamentalRansacConfig};
use super::homography::{homography_ransac, pose_from_homography_matrix, HomographyRansacConfig};
use super::{
    EightPointEssentialMatrixEstimator, EssentialRansac, EssentialRansacConfig,
    TwoViewCorrespondence,
};

/// Port of `TwoViewGeometry::ConfigurationType`
/// (`src/colmap/scene/two_view_geometry.h:42-63`). COLMAP's `CALIBRATED_RIG`
/// (multi-camera-rig relative pose) is omitted: this repo's unordered SfM path
/// has no rig concept to classify into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConfigurationType {
    /// Not yet classified (never returned by [`TwoViewGeometryVerifier`]; kept
    /// for parity with COLMAP's default-constructed state).
    Undefined,
    /// No overlap, or too few inliers for any model. `two_view_geometry.cc:
    /// 804, 858, 921`.
    Degenerate,
    /// Essential matrix: real 3D scene, moving calibrated camera.
    /// `two_view_geometry.cc:897`.
    Uncalibrated,
    /// Fundamental matrix: real 3D scene, but the essential matrix did not
    /// agree with it well enough — see the module-doc scope note.
    /// `two_view_geometry.cc:913`.
    Calibrated,
    /// Homography, planar scene with a real baseline (has triangulatable
    /// parallax). Resolved from `PlanarOrPanoramic` by decomposition.
    Planar,
    /// Homography, pure rotation (no baseline). Resolved from
    /// `PlanarOrPanoramic` by decomposition (`tri_angle` pinned to 0 in
    /// COLMAP, `tvg.cc:711-713`).
    Panoramic,
    /// Homography explains the inliers at least as well as the epipolar
    /// model, but decomposition has not (yet) told planar and panoramic
    /// apart. Only escapes this repo's [`TwoViewGeometryVerifier::classify`]
    /// if homography decomposition fails outright (degenerate `H`).
    PlanarOrPanoramic,
    /// Pure 2D translation confined to the image border — the classic
    /// stock-photo/stamp watermark artifact. `two_view_geometry.cc:946`.
    Watermark,
    /// Inliers are explained by more than one non-degenerate configuration
    /// (`multiple_models` option). `two_view_geometry.cc:304`.
    Multiple,
}

/// Port of `TwoViewGeometryOptions`
/// (`src/colmap/estimators/two_view_geometry.h:45-129`). Field names follow
/// COLMAP's own names translated to `snake_case`; doc comments quote COLMAP's
/// own descriptions where unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoViewGeometryOptions {
    /// "Minimum number of inliers for non-degenerate two-view geometry."
    /// COLMAP default `15` (`two_view_geometry.h:47`).
    pub min_num_inliers: usize,
    /// "Minimum ratio of inliers to total matches for non-degenerate
    /// geometry. Disabled by default, only effective when > 0."
    /// COLMAP default `0.0` (`:51`).
    pub min_inlier_ratio: f64,
    /// E/F cross-check: "the calibration is assumed to be correct" once the
    /// essential matrix explains at least this fraction of the fundamental
    /// matrix's inlier count. COLMAP default `0.95` (`:58`).
    pub min_e_f_inlier_ratio: f64,
    /// "If the inlier ratio of a homography comes close to the inlier ratio
    /// of the epipolar geometry, a planar or panoramic configuration is
    /// assumed." COLMAP default `0.8` (`:66`).
    pub max_h_inlier_ratio: f64,
    /// "If more than a certain ratio of inlier points conform with a pure
    /// image translation, a watermark is assumed." COLMAP default `0.7`
    /// (`:72`).
    pub watermark_min_inlier_ratio: f64,
    /// Border region width as a fraction of the image diagonal. COLMAP
    /// default `0.1` (`:77`).
    pub watermark_border_size: f64,
    /// Whether to run watermark detection at all. COLMAP default `true`
    /// (`:81`).
    pub detect_watermark: bool,
    /// "Maximum translational error of matched points to be considered
    /// inliers of a watermark," in pixels. COLMAP default `4.0` (`:88`).
    pub watermark_detection_max_error_px: f64,
    /// Max pixel Sampson/reprojection error for the fundamental-matrix and
    /// homography RANSACs. COLMAP's `TwoViewGeometryOptions()` constructor
    /// sets `ransac_options.max_error = 4.0` (`:124`).
    pub max_error_px: f64,
    /// Max Sampson error for the essential-matrix RANSAC, in *normalized*
    /// image-plane units (COLMAP derives the equivalent per-pair bound via
    /// `Camera::CamFromImgThreshold`, averaged across both cameras,
    /// `estimators/two_view_geometry.cc:831-835`). Use
    /// [`TwoViewGeometryOptions::for_camera`] to derive this from
    /// `max_error_px` and a known focal length instead of setting it by hand.
    pub essential_sampson_threshold: f64,
    /// RANSAC iteration budget shared by the E/F/H estimators.
    pub ransac_iterations: usize,
    /// RNG seed shared by the E/F/H estimators (deterministic classification).
    pub seed: u64,
    /// "Recursively estimate multiple configurations by removing the
    /// previous set of inliers from the matches until not enough inliers are
    /// found... the configuration type is `MULTIPLE` if multiple models
    /// could be estimated." COLMAP default `false` (`:118`).
    pub multiple_models: bool,
    /// When true and the pair classifies as Calibrated, keep E inliers even
    /// if F has a larger count (COLMAP picks max(E,F)). Useful when known
    /// intrinsics make E the metric model. Default false.
    pub calibrated_prefer_essential: bool,
}

impl Default for TwoViewGeometryOptions {
    fn default() -> Self {
        Self {
            min_num_inliers: 15,
            min_inlier_ratio: 0.0,
            min_e_f_inlier_ratio: 0.95,
            max_h_inlier_ratio: 0.8,
            watermark_min_inlier_ratio: 0.7,
            watermark_border_size: 0.1,
            detect_watermark: true,
            watermark_detection_max_error_px: 4.0,
            max_error_px: 4.0,
            essential_sampson_threshold: 5.0e-3,
            ransac_iterations: 256,
            seed: 7,
            multiple_models: false,
            calibrated_prefer_essential: false,
        }
    }
}

impl TwoViewGeometryOptions {
    /// Derive `essential_sampson_threshold` from `max_error_px` and the
    /// camera's focal length, mirroring COLMAP's per-pair
    /// `Camera::CamFromImgThreshold` conversion
    /// (`estimators/two_view_geometry.cc:831-835` — there averaged across two
    /// possibly-different cameras; this repo's unordered SfM path always
    /// shares one [`Camera`], so no averaging is needed). Also sets
    /// `watermark_detection_max_error_px` to the same pixel budget.
    pub fn for_camera(camera: &Camera, max_error_px: f64) -> Self {
        let mut options = Self {
            max_error_px,
            watermark_detection_max_error_px: max_error_px,
            ..Self::default()
        };
        if let Some((fx, fy, _, _)) = camera.intrinsics() {
            let focal = 0.5 * (fx + fy);
            if focal > 0.0 {
                options.essential_sampson_threshold = max_error_px / focal;
            }
        }
        options
    }
}

/// Result of classifying one candidate pair: which configuration it fell
/// into, the winning model's inlier indices (into the caller's
/// `correspondences` slice), the estimated models, and — when a
/// planar/panoramic homography was resolved — the decomposed relative pose.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoViewGeometryReport {
    pub config: ConfigurationType,
    pub inliers: Vec<usize>,
    pub essential: Option<Matrix3<f64>>,
    pub fundamental: Option<Matrix3<f64>>,
    pub homography: Option<Matrix3<f64>>,
    /// `(rotation, translation)` from PLANAR/PANORAMIC homography
    /// decomposition (`pose_from_homography_matrix`), when resolved.
    pub relative_pose: Option<(Matrix3<f64>, Vector3<f64>)>,
    /// Essential-matrix RANSAC inliers (indices into the input
    /// correspondences), even when the winning config selected F/H inliers
    /// for [`Self::inliers`]. Empty when E failed. Used by opt-in
    /// `--prefer-essential-inliers` so global edges are built from E, not F.
    pub essential_inliers: Vec<usize>,
    pub e_inlier_count: usize,
    pub f_inlier_count: usize,
    pub h_inlier_count: usize,
}

fn degenerate_report() -> TwoViewGeometryReport {
    TwoViewGeometryReport {
        config: ConfigurationType::Degenerate,
        inliers: Vec::new(),
        essential: None,
        fundamental: None,
        homography: None,
        relative_pose: None,
        essential_inliers: Vec::new(),
        e_inlier_count: 0,
        f_inlier_count: 0,
        h_inlier_count: 0,
    }
}

/// The drop-in COLMAP-style verifier. Construct with `Default::default()` for
/// COLMAP's own default thresholds, [`TwoViewGeometryOptions::for_camera`] to
/// also derive the essential-matrix threshold from a known focal length, or
/// [`TwoViewGeometryVerifier::new`] to set every field explicitly.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TwoViewGeometryVerifier {
    pub options: TwoViewGeometryOptions,
}

impl TwoViewGeometryVerifier {
    pub fn new(options: TwoViewGeometryOptions) -> Self {
        Self { options }
    }

    /// Classify one candidate pair. Dispatches to [`Self::classify_multiple`]
    /// when `options.multiple_models` is set, else [`Self::classify_single`].
    pub fn classify(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> TwoViewGeometryReport {
        if self.options.multiple_models {
            self.classify_multiple(correspondences, camera)
        } else {
            self.classify_single(correspondences, camera)
        }
    }

    /// Port of `EstimateCalibratedTwoViewGeometry`
    /// (`src/colmap/estimators/two_view_geometry.cc:786-956`); see module doc
    /// for the calibrated-path-only scope note.
    fn classify_single(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> TwoViewGeometryReport {
        let opts = &self.options;
        // `tvg.cc:802-806`: too few raw matches to even attempt a model.
        if correspondences.len() < opts.min_num_inliers {
            return degenerate_report();
        }

        let essential_ransac = EssentialRansac {
            estimator: EightPointEssentialMatrixEstimator::default(),
            config: EssentialRansacConfig {
                iterations: opts.ransac_iterations,
                sampson_threshold: opts.essential_sampson_threshold,
                seed: opts.seed,
            },
        };
        let e_report = essential_ransac.estimate(correspondences, camera);
        let f_report = fundamental_ransac(
            correspondences,
            &FundamentalRansacConfig {
                iterations: opts.ransac_iterations,
                max_error_px: opts.max_error_px,
                seed: opts.seed,
            },
        );
        let h_report = homography_ransac(
            correspondences,
            &HomographyRansacConfig {
                iterations: opts.ransac_iterations,
                max_error_px: opts.max_error_px,
                seed: opts.seed,
            },
        );

        let e_inliers = e_report.as_ref().map_or(0, |r| r.inliers.len());
        let f_inliers = f_report.as_ref().map_or(0, |r| r.inliers.len());
        let h_inliers = h_report.as_ref().map_or(0, |r| r.inliers.len());

        // `tvg.cc:854-860`: all three models failed, or none cleared the
        // minimum-inlier gate.
        if (e_report.is_none() && f_report.is_none() && h_report.is_none())
            || (e_inliers < opts.min_num_inliers
                && f_inliers < opts.min_num_inliers
                && h_inliers < opts.min_num_inliers)
        {
            return degenerate_report();
        }

        // `tvg.cc:864-872`: inlier-count ratios drive model selection. A
        // zero denominator (the reference model failed outright) reads as
        // "ratio not exceeded" — COLMAP's C++ divides by zero and gets
        // inf/nan there, which never compares `> threshold` as true either,
        // so this mirrors the observable behaviour without the UB.
        let ratio = |num: usize, den: usize| -> f64 {
            if den == 0 {
                0.0
            } else {
                num as f64 / den as f64
            }
        };
        let e_f_inlier_ratio = ratio(e_inliers, f_inliers);
        let h_f_inlier_ratio = ratio(h_inliers, f_inliers);
        let h_e_inlier_ratio = ratio(h_inliers, e_inliers);

        // Owned inlier-index snapshots, bound once via `if let`/`let-else`
        // below instead of `Option::is_some()` followed by a later
        // `.unwrap()` — same decision tree, clippy-clean.
        let e_inlier_indices = e_report.as_ref().map(|r| r.inliers.clone());
        let f_inlier_indices = f_report.as_ref().map(|r| r.inliers.clone());
        let h_inlier_indices = h_report.as_ref().map(|r| r.inliers.clone());

        let mut config;
        let mut chosen_inliers: Vec<usize>;

        if let Some(e) = e_inlier_indices.as_ref().filter(|_| {
            e_f_inlier_ratio > opts.min_e_f_inlier_ratio && e_inliers >= opts.min_num_inliers
        }) {
            // `tvg.cc:877-898`: calibrated configuration — use whichever of
            // E/F has more inliers (unless `calibrated_prefer_essential`).
            let mut num_inliers = e_inliers;
            chosen_inliers = e.clone();
            if !opts.calibrated_prefer_essential && f_inliers > e_inliers {
                if let Some(f) = &f_inlier_indices {
                    num_inliers = f_inliers;
                    chosen_inliers = f.clone();
                }
            }
            if h_e_inlier_ratio > opts.max_h_inlier_ratio {
                config = ConfigurationType::PlanarOrPanoramic;
                if h_inliers > num_inliers {
                    if let Some(h) = &h_inlier_indices {
                        chosen_inliers = h.clone();
                    }
                }
            } else {
                config = ConfigurationType::Calibrated;
            }
        } else if f_inliers >= opts.min_num_inliers {
            // `tvg.cc:899-914`: uncalibrated configuration (E did not agree
            // well enough with F, or failed outright). `f_inliers >=
            // min_num_inliers > 0` (COLMAP default 15) can only hold when
            // `f_report` actually succeeded, so `f_inlier_indices` is `Some`
            // here in practice; the `let-else` below is defensive, not a
            // reachable path.
            let Some(f) = &f_inlier_indices else {
                return degenerate_report();
            };
            let num_inliers = f_inliers;
            chosen_inliers = f.clone();
            if h_f_inlier_ratio > opts.max_h_inlier_ratio {
                config = ConfigurationType::PlanarOrPanoramic;
                if h_inliers > num_inliers {
                    if let Some(h) = &h_inlier_indices {
                        chosen_inliers = h.clone();
                    }
                }
            } else {
                config = ConfigurationType::Uncalibrated;
            }
        } else if h_inliers >= opts.min_num_inliers {
            // `tvg.cc:915-919`: only the homography cleared the gate. Same
            // defensive note as the `f_inlier_indices` branch above.
            let Some(h) = &h_inlier_indices else {
                return degenerate_report();
            };
            chosen_inliers = h.clone();
            config = ConfigurationType::PlanarOrPanoramic;
        } else {
            // `tvg.cc:920-922`.
            return degenerate_report();
        }

        // `tvg.cc:929-937`: optional global inlier-ratio gate (disabled by
        // COLMAP's own default, `min_inlier_ratio == 0.0`).
        if opts.min_inlier_ratio > 0.0 {
            let inlier_ratio = chosen_inliers.len() as f64 / correspondences.len() as f64;
            if inlier_ratio < opts.min_inlier_ratio {
                return degenerate_report();
            }
        }

        if opts.detect_watermark && detect_watermark(camera, correspondences, &chosen_inliers, opts)
        {
            // `tvg.cc:939-947`.
            config = ConfigurationType::Watermark;
        }

        let essential_inliers = e_inlier_indices.clone().unwrap_or_default();
        let mut report = TwoViewGeometryReport {
            config,
            inliers: chosen_inliers,
            essential: e_report.map(|r| r.essential),
            fundamental: f_report.map(|r| r.fundamental),
            homography: h_report.map(|r| r.homography),
            relative_pose: None,
            essential_inliers,
            e_inlier_count: e_inliers,
            f_inlier_count: f_inliers,
            h_inlier_count: h_inliers,
        };

        if report.config == ConfigurationType::PlanarOrPanoramic {
            self.resolve_planar_or_panoramic(&mut report, correspondences, camera);
        }

        report
    }

    /// Splits `PLANAR_OR_PANORAMIC` into `PLANAR` or `PANORAMIC` by
    /// decomposing the winning homography and checking the recovered
    /// translation, port of `EstimateTwoViewGeometryPoseFromCamRays`
    /// (`src/colmap/estimators/two_view_geometry.cc:687-718`, the
    /// `PLANAR_OR_PANORAMIC` branch). Leaves the report as
    /// `PLANAR_OR_PANORAMIC` if decomposition fails outright (degenerate
    /// homography) — see [`ConfigurationType::PlanarOrPanoramic`]'s doc.
    fn resolve_planar_or_panoramic(
        &self,
        report: &mut TwoViewGeometryReport,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) {
        let Some(h) = report.homography else {
            return;
        };
        let Some((fx, fy, cx, cy)) = camera.intrinsics() else {
            return;
        };
        let k = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
        let inlier_correspondences: Vec<TwoViewCorrespondence> =
            report.inliers.iter().map(|&i| correspondences[i]).collect();
        let Some((rotation, translation, _normal)) =
            pose_from_homography_matrix(&h, &k, &k, &inlier_correspondences, camera)
        else {
            return;
        };
        report.config = if translation.norm_squared() < 1e-12 {
            ConfigurationType::Panoramic
        } else {
            ConfigurationType::Planar
        };
        report.relative_pose = Some((rotation, translation));
    }

    /// Port of `EstimateMultipleTwoViewGeometries`
    /// (`src/colmap/estimators/two_view_geometry.cc:270-313`): repeatedly
    /// classify the remaining (not-yet-explained) correspondences, removing
    /// each round's inlier set (`ExtractOutlierMatches`, `tvg.cc:105-126`)
    /// before the next round, until a round comes back `DEGENERATE`. Reports
    /// `MULTIPLE` when more than one non-degenerate, non-watermark round
    /// succeeded (COLMAP's default `multiple_ignore_watermark = true`,
    /// `two_view_geometry.h:84`).
    fn classify_multiple(
        &self,
        correspondences: &[TwoViewCorrespondence],
        camera: &Camera,
    ) -> TwoViewGeometryReport {
        let mut remaining: Vec<usize> = (0..correspondences.len()).collect();
        let mut geometries: Vec<TwoViewGeometryReport> = Vec::new();

        let mut single_options = self.options;
        single_options.multiple_models = false;
        let single_verifier = TwoViewGeometryVerifier::new(single_options);

        loop {
            let remaining_correspondences: Vec<TwoViewCorrespondence> =
                remaining.iter().map(|&i| correspondences[i]).collect();
            let sub = single_verifier.classify_single(&remaining_correspondences, camera);
            if sub.config == ConfigurationType::Degenerate {
                break;
            }

            let global_inliers: Vec<usize> = sub.inliers.iter().map(|&i| remaining[i]).collect();
            let inlier_set: HashSet<usize> = global_inliers.iter().copied().collect();
            remaining.retain(|i| !inlier_set.contains(i));

            if sub.config != ConfigurationType::Watermark {
                let mut sub_global = sub;
                sub_global.inliers = global_inliers;
                geometries.push(sub_global);
            }

            if remaining.len() < self.options.min_num_inliers {
                break;
            }
        }

        if geometries.is_empty() {
            degenerate_report()
        } else if geometries.len() == 1 {
            geometries.into_iter().next().unwrap()
        } else {
            // SfM must not concatenate incompatible models into one inlier set
            // (that poisons a later essential RANSAC). Prefer the strongest
            // Calibrated (E) sub-model; otherwise the largest inlier set.
            // Still label `Multiple` so callers can see multi-model admission.
            let best_idx = geometries
                .iter()
                .enumerate()
                .max_by_key(|(_, g)| {
                    let calibrated_bonus = if matches!(g.config, ConfigurationType::Calibrated) {
                        1_000_000usize
                    } else {
                        0
                    };
                    calibrated_bonus + g.inliers.len()
                })
                .map(|(i, _)| i)
                .expect("geometries non-empty");
            let mut best = geometries.swap_remove(best_idx);
            best.config = ConfigurationType::Multiple;
            best
        }
    }
}

/// Port of `DetectWatermarkMatches`
/// (`src/colmap/estimators/two_view_geometry.cc:958-1023`): a watermark is a
/// pure 2D translation confined to the image border. Two gates:
/// 1. Border test — at least `watermark_min_inlier_ratio` of the inliers must
///    lie in the border region of *both* images (`box.contains` inverted,
///    `tvg.cc:990-998`).
/// 2. Translation test — a single 2D translation must explain at least
///    `watermark_min_inlier_ratio` of those (border-restricted) inliers.
///    COLMAP samples this with `TranslationTransformEstimator<2>` inside a
///    `LORANSAC` (`kMinNumSamples = 1`,
///    `src/colmap/estimators/solvers/translation_transform.h:44-118`); since
///    the minimal sample is a single point, this port evaluates every
///    candidate translation exhaustively (`n` candidates from `n` points)
///    rather than randomly sampling a subset — strictly at least as good an
///    approximation of COLMAP's own RANSAC target, and deterministic.
fn detect_watermark(
    camera: &Camera,
    correspondences: &[TwoViewCorrespondence],
    inliers: &[usize],
    opts: &TwoViewGeometryOptions,
) -> bool {
    if inliers.is_empty() {
        return false;
    }

    let diagonal = ((camera.width as f64).powi(2) + (camera.height as f64).powi(2)).sqrt();
    let border = opts.watermark_border_size * diagonal;
    let in_interior = |p: &nalgebra::Point2<f64>| {
        p.x >= border
            && p.x <= camera.width as f64 - border
            && p.y >= border
            && p.y <= camera.height as f64 - border
    };

    let mut inlier_points1 = Vec::with_capacity(inliers.len());
    let mut inlier_points2 = Vec::with_capacity(inliers.len());
    let mut num_in_border = 0usize;
    for &i in inliers {
        let c = &correspondences[i];
        inlier_points1.push(c.previous_xy);
        inlier_points2.push(c.current_xy);
        if !in_interior(&c.previous_xy) && !in_interior(&c.current_xy) {
            num_in_border += 1;
        }
    }
    let border_ratio = num_in_border as f64 / inliers.len() as f64;
    if border_ratio < opts.watermark_min_inlier_ratio {
        return false;
    }

    let n = inlier_points1.len();
    let threshold_sq =
        opts.watermark_detection_max_error_px * opts.watermark_detection_max_error_px;
    let mut best_count = 0usize;
    for i in 0..n {
        let t = inlier_points2[i] - inlier_points1[i];
        let count = (0..n)
            .filter(|&j| {
                let d = inlier_points2[j] - (inlier_points1[j] + t);
                d.norm_squared() <= threshold_sq
            })
            .count();
        if count > best_count {
            best_count = count;
        }
    }
    let translation_inlier_ratio = best_count as f64 / n as f64;
    translation_inlier_ratio >= opts.watermark_min_inlier_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, Point3, UnitQuaternion, Vector3};
    use visloc_core::geometry::{Pose, SE3};

    fn synthetic_camera() -> Camera {
        Camera::pinhole(1, 640, 480, 500.0, 500.0, 320.0, 240.0)
    }

    fn project(pose: &Pose, camera: &Camera, point: &Point3<f64>) -> Option<Point2<f64>> {
        camera.project(&pose.transform_world_point(point))
    }

    fn correspondences_for(
        previous: &Pose,
        current: &Pose,
        camera: &Camera,
        points: &[Point3<f64>],
    ) -> Vec<TwoViewCorrespondence> {
        points
            .iter()
            .filter_map(|point| {
                let p1 = project(previous, camera, point)?;
                let p2 = project(current, camera, point)?;
                Some(TwoViewCorrespondence::new(p1, p2))
            })
            .collect()
    }

    /// A scattered non-planar point cloud, enough (>15) points and real
    /// depth variation that a single homography cannot explain the true
    /// parallax as well as the essential matrix does.
    fn general_scene_points() -> Vec<Point3<f64>> {
        let mut points = Vec::new();
        for i in 0..6 {
            for j in 0..4 {
                let x = -1.5 + 0.6 * i as f64;
                let y = -1.0 + 0.7 * j as f64;
                let z = 3.0 + 0.8 * ((i + j) % 5) as f64;
                points.push(Point3::new(x, y, z));
            }
        }
        points
    }

    /// The same grid of points, all pinned to a single plane `z = 5`.
    fn planar_scene_points() -> Vec<Point3<f64>> {
        let mut points = Vec::new();
        for i in 0..6 {
            for j in 0..4 {
                let x = -1.5 + 0.6 * i as f64;
                let y = -1.0 + 0.7 * j as f64;
                points.push(Point3::new(x, y, 5.0));
            }
        }
        points
    }

    #[test]
    fn general_scene_classifies_calibrated_or_uncalibrated() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.08);
        let current = Pose {
            world_to_camera: SE3::new(yaw, Vector3::new(-0.6, 0.05, -0.1)),
        };
        let correspondences =
            correspondences_for(&previous, &current, &camera, &general_scene_points());
        assert!(correspondences.len() >= 20);

        let verifier =
            TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(&camera, 4.0));
        let report = verifier.classify(&correspondences, &camera);
        assert_eq!(
            report.config,
            ConfigurationType::Calibrated,
            "general 3D parallax with known intrinsics should classify CALIBRATED when E/F agree"
        );
        assert!(!report.inliers.is_empty());
        assert!(
            !report.essential_inliers.is_empty() && report.e_inlier_count > 0,
            "essential RANSAC must expose its inlier set for prefer-essential-inliers"
        );
    }

    #[test]
    fn planar_scene_with_baseline_classifies_planar() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.06);
        let current = Pose {
            world_to_camera: SE3::new(yaw, Vector3::new(-0.5, 0.0, 0.0)),
        };
        let correspondences =
            correspondences_for(&previous, &current, &camera, &planar_scene_points());
        assert!(correspondences.len() >= 20);

        let verifier =
            TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(&camera, 4.0));
        let report = verifier.classify(&correspondences, &camera);
        assert_eq!(report.config, ConfigurationType::Planar);
        let (_, translation) = report.relative_pose.expect("planar pose must resolve");
        assert!(translation.norm() > 1.0e-6);
    }

    #[test]
    fn pure_rotation_classifies_panoramic() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let yaw = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.12);
        let current = Pose::from_world_to_camera(yaw, Vector3::new(0.0, 0.0, 0.0));
        let correspondences =
            correspondences_for(&previous, &current, &camera, &general_scene_points());
        assert!(correspondences.len() >= 20);

        let verifier =
            TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(&camera, 4.0));
        let report = verifier.classify(&correspondences, &camera);
        assert_eq!(report.config, ConfigurationType::Panoramic);
        let (_, translation) = report.relative_pose.expect("panoramic pose must resolve");
        assert!(translation.norm_squared() < 1.0e-9);
    }

    #[test]
    fn too_few_correspondences_is_degenerate() {
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let mut points = general_scene_points();
        points.truncate(10); // below the default min_num_inliers = 15
        let correspondences = correspondences_for(&previous, &current, &camera, &points);

        let verifier = TwoViewGeometryVerifier::default();
        let report = verifier.classify(&correspondences, &camera);
        assert_eq!(report.config, ConfigurationType::Degenerate);
        assert!(report.inliers.is_empty());
    }

    #[test]
    fn border_translation_detected_as_watermark() {
        // Direct unit test of the ported COLMAP rule
        // (`DetectWatermarkMatches`): correspondences confined to the image
        // border, related by one fixed 2D translation.
        let camera = synthetic_camera();
        let opts = TwoViewGeometryOptions::default();
        let translation = Vector3::new(6.0, -4.0, 0.0);
        let mut correspondences = Vec::new();
        // Border strip is `watermark_border_size * diagonal` wide; diagonal
        // for 640x480 is 800, so the strip is 80px. Scatter points inside the
        // top-left border box, comfortably clear of the interior box.
        for i in 0..20 {
            let x = 5.0 + (i as f64 * 3.0) % 60.0;
            let y = 5.0 + (i as f64 * 7.0) % 60.0;
            let p1 = Point2::new(x, y);
            let p2 = Point2::new(x + translation.x, y + translation.y);
            correspondences.push(TwoViewCorrespondence::new(p1, p2));
        }
        let all_inliers: Vec<usize> = (0..correspondences.len()).collect();
        assert!(detect_watermark(
            &camera,
            &correspondences,
            &all_inliers,
            &opts
        ));
    }

    #[test]
    fn interior_matches_are_not_flagged_as_watermark() {
        let camera = synthetic_camera();
        let opts = TwoViewGeometryOptions::default();
        let translation = Vector3::new(6.0, -4.0, 0.0);
        let mut correspondences = Vec::new();
        // Same pure translation, but comfortably inside the interior box —
        // must not trigger the border gate.
        for i in 0..20 {
            let x = 300.0 + (i as f64 * 3.0) % 40.0;
            let y = 220.0 + (i as f64 * 5.0) % 40.0;
            let p1 = Point2::new(x, y);
            let p2 = Point2::new(x + translation.x, y + translation.y);
            correspondences.push(TwoViewCorrespondence::new(p1, p2));
        }
        let all_inliers: Vec<usize> = (0..correspondences.len()).collect();
        assert!(!detect_watermark(
            &camera,
            &correspondences,
            &all_inliers,
            &opts
        ));
    }

    #[test]
    fn legacy_essential_only_path_is_unaffected_by_new_module() {
        // "Bit-preserved when the flag is off": the pre-existing
        // essential-only path (`RelativePoseEstimator`, used by
        // `examples/unordered_sfm_demo.rs` unless `--colmap-verification` is
        // passed) is untouched code, exercised here side-by-side with the new
        // verifier on the same data to document that adding this module does
        // not change its behaviour or its callers.
        use super::super::RelativePoseEstimator;
        let camera = synthetic_camera();
        let previous =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, 0.0));
        let current =
            Pose::from_world_to_camera(UnitQuaternion::identity(), Vector3::new(-0.3, 0.0, 0.0));
        let correspondences =
            correspondences_for(&previous, &current, &camera, &general_scene_points());

        let legacy = RelativePoseEstimator::default();
        let legacy_pose = legacy
            .estimate(&correspondences, &camera)
            .expect("legacy essential-only path must still recover a pose");
        assert!(legacy_pose.inliers.len() >= 8);

        // The new verifier runs independently and does not mutate any shared
        // state the legacy path depends on.
        let verifier =
            TwoViewGeometryVerifier::new(TwoViewGeometryOptions::for_camera(&camera, 4.0));
        let _ = verifier.classify(&correspondences, &camera);
        let legacy_pose_again = legacy
            .estimate(&correspondences, &camera)
            .expect("legacy essential-only path must still recover a pose");
        assert_eq!(legacy_pose.inliers, legacy_pose_again.inliers);
    }
}
