//! Rectified-stereo VO frontend primitives.
//!
//! These helpers turn left/right [`FeatureSet`]s into metric 3D points in the
//! left camera's frame (via [`triangulate_stereo_features`]) and stitch
//! consecutive triangulations into a relative-pose hypothesis (via
//! [`build_stereo_temporal_correspondences`] + a caller-supplied PnP RANSAC).
//! Compared with monocular VO the recovered pose is in metric units — the
//! baseline anchors scale on every frame, so there is no per-pair scale
//! ambiguity to resolve later.
//!
//! The frontend is split so callers can plug in alternative feature
//! extractors and matchers, keep their own per-frame caches, or skip the PnP
//! step when they only need the triangulations themselves.

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector, Matrix3, Point2, Point3, UnitQuaternion, Vector3};
use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;

use crate::features::{
    CornerFeatureConfig, CornerFeatureError, CornerFeatureExtractor, FeatureExtractor, FeatureSet,
    GrayscaleImage,
};
use crate::matching::{BruteForceMatcher, DescriptorMatch, Matcher};
use crate::pnp::Correspondence2D3D;
use crate::ransac::{PnPRansac, RobustPoseEstimator};
use crate::stereo::triangulate_stereo_pixel;

/// One triangulated stereo feature, anchored to its descriptor index in the
/// left [`FeatureSet`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoFeature {
    /// Index of the keypoint in the left [`FeatureSet`].
    pub left_index: usize,
    /// Index of the keypoint in the right [`FeatureSet`].
    pub right_index: usize,
    /// Disparity `u_l − u_r` (always positive, in pixels).
    pub disparity: f64,
    /// 3D point in the left camera's frame, in metric units derived from
    /// `baseline`.
    pub point_cam: Point3<f64>,
}

/// Configuration for [`triangulate_stereo_features`]. Defaults are tuned for
/// rectified KITTI imagery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoFeatureConfig {
    /// Minimum acceptable disparity (`u_l − u_r`) in pixels. Sub-threshold
    /// matches indicate either a far-field point (numerically unstable) or
    /// a wrong correspondence (e.g., a sky pixel). Set on the same scale
    /// as the descriptor matcher's expected sub-pixel accuracy.
    pub min_disparity_px: f64,
    /// Maximum allowed `|v_l − v_r|` row residual in pixels. A non-zero
    /// row residual on rectified imagery means the match is wrong (or the
    /// rectification is imperfect); reject so triangulation does not blend
    /// in a Y-direction error.
    pub max_row_residual_px: f64,
    /// Minimum `Z` (depth) accepted in the left camera's frame. Points
    /// nearer than this disappear behind / below the camera within one
    /// frame on a moving vehicle; their stereo disparity is noise-dominated
    /// because the L↔R parallax dwarfs descriptor sub-pixel accuracy.
    pub min_depth_m: f64,
    /// Maximum `Z` (depth) accepted in the left camera's frame. KITTI sky
    /// pixels triangulate to `Z ≫ 100 m` because their disparity is at the
    /// matcher's noise floor — those points are useless for downstream
    /// pose estimation and inflate landmark counts.
    pub max_depth_m: f64,
    /// Lowe's ratio threshold for the row-restricted L↔R descriptor search.
    /// `Some(0.85)` keeps matches whose best-distance / second-best-distance
    /// in the same row band is strictly below the threshold. `None`
    /// disables the ratio check (every left keypoint with a band candidate
    /// is matched).
    pub ratio: Option<f64>,
}

impl Default for StereoFeatureConfig {
    fn default() -> Self {
        Self {
            min_disparity_px: 1.0,
            max_row_residual_px: 1.5,
            min_depth_m: 3.0,
            max_depth_m: 80.0,
            ratio: Some(0.85),
        }
    }
}

/// Match the left and right [`FeatureSet`]s using the rectified-stereo
/// epipolar constraint and triangulate every accepted correspondence into a
/// metric 3D point. Returns one [`StereoFeature`] per match that passes the
/// [`StereoFeatureConfig`] gates.
///
/// Unlike a plain descriptor matcher, this routine restricts each left
/// keypoint's right-image search to the same row band (`|v_l − v_r| ≤
/// max_row_residual_px`) and to positive disparity (`u_r < u_l`). Without
/// the row gate, patch descriptors on KITTI-like imagery routinely match
/// across rows and produce 3D points consistent with the left view but
/// wildly wrong in 3D space — those points then poison the downstream PnP.
/// The gated matcher also runs Lowe's ratio test on the within-band best /
/// second-best descriptor distance.
///
/// The returned points live in the left-camera frame (positive `Z` ahead of
/// the camera). `baseline` is the rectified-stereo baseline magnitude in
/// metric units (same units the points come out in).
pub fn triangulate_stereo_features(
    left: &FeatureSet,
    right: &FeatureSet,
    camera: &Camera,
    baseline: f64,
    config: &StereoFeatureConfig,
) -> Vec<StereoFeature> {
    if left.is_empty() || right.is_empty() {
        return Vec::new();
    }
    let mut out: Vec<StereoFeature> = Vec::with_capacity(left.len() / 2);
    for (l_idx, l_kp) in left.keypoints.iter().enumerate() {
        let l_desc = &left.descriptors[l_idx];
        // Row-restricted descriptor search: only consider right keypoints
        // sharing the left's image row (within max_row_residual_px) and
        // with positive disparity (u_r < u_l).
        let mut best: Option<(usize, f64)> = None;
        let mut second_best: Option<f64> = None;
        for (r_idx, r_kp) in right.keypoints.iter().enumerate() {
            if (l_kp.y - r_kp.y).abs() > config.max_row_residual_px {
                continue;
            }
            let disparity = l_kp.x - r_kp.x;
            if disparity < config.min_disparity_px {
                continue;
            }
            let r_desc = &right.descriptors[r_idx];
            let dist = descriptor_distance(l_desc, r_desc);
            match best {
                None => best = Some((r_idx, dist)),
                Some((_, b_dist)) if dist < b_dist => {
                    second_best = Some(b_dist);
                    best = Some((r_idx, dist));
                }
                Some(_) => {
                    if second_best.is_none_or(|s| dist < s) {
                        second_best = Some(dist);
                    }
                }
            }
        }
        let Some((r_idx, b_dist)) = best else {
            continue;
        };
        // Lowe's ratio test (rectified-stereo variant: only if a runner-up
        // exists in the same band).
        if let (Some(s), Some(ratio)) = (second_best, config.ratio) {
            if s > 0.0 && b_dist / s > ratio {
                continue;
            }
        } else {
            // Suppress unused-variable warning when ratio is disabled.
            let _ = b_dist;
        }
        let r_kp = right.keypoints[r_idx];
        let disparity = l_kp.x - r_kp.x;
        let Some(point) = triangulate_stereo_pixel(
            camera,
            baseline,
            (l_kp.x, l_kp.y),
            (r_kp.x, r_kp.y),
            config.min_disparity_px,
        ) else {
            continue;
        };
        if !point.coords.iter().all(|v| v.is_finite()) {
            continue;
        }
        if point.z < config.min_depth_m || point.z > config.max_depth_m {
            continue;
        }
        out.push(StereoFeature {
            left_index: l_idx,
            right_index: r_idx,
            disparity,
            point_cam: point,
        });
    }
    out
}

/// Triangulate explicit rectified-stereo matches into metric 3D points.
///
/// This is the file-backed / learned-frontend companion to
/// [`triangulate_stereo_features`]. It trusts the caller's matcher
/// (SuperPoint+LightGlue, ONNX, a remote service, etc.) for candidate
/// association, then applies the same stereo geometry gates: positive
/// disparity, row residual, finite 3D point, and depth range.
pub fn triangulate_stereo_feature_matches(
    left: &FeatureSet,
    right: &FeatureSet,
    matches: &[DescriptorMatch],
    camera: &Camera,
    baseline: f64,
    config: &StereoFeatureConfig,
) -> Vec<StereoFeature> {
    if left.is_empty() || right.is_empty() || matches.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(matches.len());
    for descriptor_match in matches {
        let Some(l_kp) = left.keypoints.get(descriptor_match.query_index) else {
            continue;
        };
        let Some(r_kp) = right.keypoints.get(descriptor_match.train_index) else {
            continue;
        };
        if (l_kp.y - r_kp.y).abs() > config.max_row_residual_px {
            continue;
        }
        let disparity = l_kp.x - r_kp.x;
        if disparity < config.min_disparity_px {
            continue;
        }
        let Some(point) = triangulate_stereo_pixel(
            camera,
            baseline,
            (l_kp.x, l_kp.y),
            (r_kp.x, r_kp.y),
            config.min_disparity_px,
        ) else {
            continue;
        };
        if !point.coords.iter().all(|v| v.is_finite()) {
            continue;
        }
        if point.z < config.min_depth_m || point.z > config.max_depth_m {
            continue;
        }
        out.push(StereoFeature {
            left_index: descriptor_match.query_index,
            right_index: descriptor_match.train_index,
            disparity,
            point_cam: point,
        });
    }
    out
}

/// One pair of metric 3D points (frame `a` and frame `b`) believed to be the
/// same physical scene point. Used by [`estimate_relative_pose_kabsch_ransac`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoPairCorrespondence {
    pub a: Point3<f64>,
    pub b: Point3<f64>,
    /// Optional temporal-match confidence. Deep matchers such as
    /// `MutualSoftmaxMatcher` populate this so Kabsch RANSAC can sample
    /// high-confidence 3D-3D pairs first; classical matchers leave it `None`.
    pub confidence: Option<f32>,
}

/// Configuration for [`estimate_relative_pose_kabsch_ransac`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KabschRansacConfig {
    pub iterations: usize,
    /// Maximum allowed `|X_b − (R·X_a + t)|` per inlier (in metric units).
    pub inlier_threshold_m: f64,
    /// Minimum number of inliers for a pose to be accepted. Below this, the
    /// estimator returns `None`.
    pub min_inliers: usize,
    /// Discard correspondences with `Z` (in either frame) above this. Stereo
    /// noise scales as `Z² · σ_disp / (fx · b)`, so far-field samples are
    /// unreliable for the metric pose recovery.
    pub max_depth_m: f64,
    pub seed: u64,
}

impl Default for KabschRansacConfig {
    fn default() -> Self {
        Self {
            iterations: 256,
            inlier_threshold_m: 0.30,
            min_inliers: 12,
            max_depth_m: 25.0,
            seed: 7,
        }
    }
}

/// Output of [`estimate_relative_pose_kabsch_ransac`].
#[derive(Debug, Clone, PartialEq)]
pub struct KabschRansacReport {
    /// `T_a_to_b`: applies as `X_b = relative · X_a` for any matched scene
    /// point, with `world_to_camera.translation` carrying metric scale.
    pub relative_pose: Pose,
    /// Indices into the original `correspondences` slice that fit the
    /// recovered pose within [`KabschRansacConfig::inlier_threshold_m`].
    pub inliers: Vec<usize>,
    pub mean_residual_m: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoRelativePoseSource {
    Pnp,
    PnpFallback,
    Kabsch,
    KabschFallback,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoPairDiagnostics {
    pub from_frame: usize,
    pub to_frame: usize,
    pub source: StereoRelativePoseSource,
    pub temporal_match_count: usize,
    pub temporal_row_gate_px: Option<f64>,
    pub temporal_confidence_gate: Option<f32>,
    pub pnp_correspondence_count: usize,
    pub stereo_pair_correspondence_count: usize,
    pub inlier_count: usize,
    pub raw_translation_m: f64,
    pub raw_rotation_deg: f64,
    pub translation_m: f64,
    pub rotation_deg: f64,
    pub motion_scale_rescued: bool,
    pub stereo_vertical_aligned: bool,
    pub translation_direction_rescued: bool,
    pub rotation_spike_rescued: bool,
    pub rotation_vector_rescued: bool,
    pub pnp_mean_reprojection_error_px: Option<f64>,
    pub kabsch_mean_residual_m: Option<f64>,
}

/// 3D-3D Kabsch (Procrustes) registration with RANSAC.
///
/// Robust to the forward-motion degeneracy that breaks essential-matrix VO
/// on driving sequences. Each correspondence supplies the same physical
/// point's metric coordinates in two frames; the metric translation
/// magnitude is recovered directly from the data with no scale ambiguity.
///
/// Three correspondences are the minimum sample (one solves a 3D-3D
/// alignment up to a one-DoF reflection that the cheirality picker
/// resolves). 256 iterations find a clean inlier set even when the input
/// has 50 % outliers.
pub fn estimate_relative_pose_kabsch_ransac(
    correspondences: &[StereoPairCorrespondence],
    config: &KabschRansacConfig,
) -> Option<KabschRansacReport> {
    // Pre-filter to within-depth correspondences (stereo accuracy
    // degrades as Z²).
    let usable: Vec<usize> = (0..correspondences.len())
        .filter(|&i| {
            let c = &correspondences[i];
            c.a.z > 0.0 && c.b.z > 0.0 && c.a.z <= config.max_depth_m && c.b.z <= config.max_depth_m
        })
        .collect();
    if usable.len() < 3.max(config.min_inliers) {
        return None;
    }
    let mut rng = SmallRng::seed_from_u64(config.seed);
    let has_confidence = usable.iter().any(|&i| {
        correspondences[i]
            .confidence
            .is_some_and(|w| w.is_finite() && w > 0.0)
    });
    let mut ranked = usable.clone();
    if has_confidence {
        ranked.sort_by(|&left, &right| {
            let l = correspondences[left].confidence.unwrap_or(0.0);
            let r = correspondences[right].confidence.unwrap_or(0.0);
            r.total_cmp(&l)
        });
    }
    let mut indices = usable.clone();
    let mut best: Option<(Pose, Vec<usize>, f64)> = None;
    let threshold_sq = config.inlier_threshold_m * config.inlier_threshold_m;
    for iter in 0..config.iterations {
        let sample_indices: Vec<usize> = if has_confidence {
            let span = ranked.len().saturating_sub(3);
            let subset_len = 3 + (span * (iter + 1) / config.iterations.max(1));
            ranked[..subset_len]
                .choose_multiple(&mut rng, 3)
                .copied()
                .collect()
        } else {
            indices.shuffle(&mut rng);
            indices.iter().take(3).copied().collect()
        };
        let sample: Vec<&StereoPairCorrespondence> = sample_indices
            .iter()
            .map(|&i| &correspondences[i])
            .collect();
        let Some(pose) = kabsch_3pt(&sample) else {
            continue;
        };
        let (inliers, sse) = score_pose_3d3d(&pose, correspondences, threshold_sq);
        let better = best.as_ref().is_none_or(|(_, b_inliers, b_sse)| {
            inliers.len() > b_inliers.len() || (inliers.len() == b_inliers.len() && sse < *b_sse)
        });
        if better {
            best = Some((pose, inliers, sse));
        }
    }
    let (_, best_inliers, _) = best?;
    if best_inliers.len() < config.min_inliers {
        return None;
    }
    // Refine on the full inlier set (Kabsch on N≥3 is the closed-form
    // least-squares solution, no iteration needed).
    let inlier_corrs: Vec<&StereoPairCorrespondence> =
        best_inliers.iter().map(|&i| &correspondences[i]).collect();
    let refined = kabsch_n(&inlier_corrs)?;
    let (final_inliers, sse) = score_pose_3d3d(&refined, correspondences, threshold_sq);
    let mean_residual_m = if final_inliers.is_empty() {
        f64::INFINITY
    } else {
        (sse / final_inliers.len() as f64).sqrt()
    };
    Some(KabschRansacReport {
        relative_pose: refined,
        inliers: final_inliers,
        mean_residual_m,
    })
}

fn kabsch_3pt(samples: &[&StereoPairCorrespondence]) -> Option<Pose> {
    if samples.len() < 3 {
        return None;
    }
    kabsch_n(samples)
}

/// Closed-form Kabsch / Procrustes least-squares pose from N matched 3D
/// pairs. Returns `T_a_to_b` such that `X_b ≈ R·X_a + t`.
fn kabsch_n(samples: &[&StereoPairCorrespondence]) -> Option<Pose> {
    let n = samples.len();
    if n < 3 {
        return None;
    }
    let mut centroid_a = Vector3::<f64>::zeros();
    let mut centroid_b = Vector3::<f64>::zeros();
    for c in samples {
        centroid_a += c.a.coords;
        centroid_b += c.b.coords;
    }
    centroid_a /= n as f64;
    centroid_b /= n as f64;
    let mut h = Matrix3::<f64>::zeros();
    for c in samples {
        let a = c.a.coords - centroid_a;
        let b = c.b.coords - centroid_b;
        h += a * b.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let v_t = svd.v_t?;
    let v = v_t.transpose();
    let mut s = Matrix3::<f64>::identity();
    let det = (v * u.transpose()).determinant();
    if det < 0.0 {
        s[(2, 2)] = -1.0;
    }
    let r_mat = v * s * u.transpose();
    let translation = centroid_b - r_mat * centroid_a;
    let rotation = UnitQuaternion::from_matrix(&r_mat);
    Some(Pose::from_world_to_camera(rotation, translation))
}

fn score_pose_3d3d(
    pose: &Pose,
    correspondences: &[StereoPairCorrespondence],
    threshold_sq: f64,
) -> (Vec<usize>, f64) {
    let r_mat = pose
        .world_to_camera
        .rotation
        .to_rotation_matrix()
        .into_inner();
    let t = pose.world_to_camera.translation;
    let mut inliers = Vec::new();
    let mut sse = 0.0;
    for (i, c) in correspondences.iter().enumerate() {
        let predicted = r_mat * c.a.coords + t;
        let residual = c.b.coords - predicted;
        let err_sq = residual.dot(&residual);
        if err_sq <= threshold_sq {
            inliers.push(i);
            sse += err_sq;
        }
    }
    (inliers, sse)
}

/// Build [`StereoPairCorrespondence`]s by joining the per-frame
/// triangulations on temporal descriptor matches: each match links a left
/// keypoint at frame `a` to a left keypoint at frame `b`; if both have a
/// stereo triangulation the pair contributes one 3D-3D correspondence.
pub fn stereo_pair_correspondences(
    a_features: &[StereoFeature],
    b_features: &[StereoFeature],
    temporal_matches: &[DescriptorMatch],
) -> Vec<StereoPairCorrespondence> {
    if a_features.is_empty() || b_features.is_empty() || temporal_matches.is_empty() {
        return Vec::new();
    }
    let a_lookup: HashMap<usize, &StereoFeature> =
        a_features.iter().map(|f| (f.left_index, f)).collect();
    let b_lookup: HashMap<usize, &StereoFeature> =
        b_features.iter().map(|f| (f.left_index, f)).collect();
    let mut out: Vec<StereoPairCorrespondence> = Vec::with_capacity(temporal_matches.len());
    for m in temporal_matches {
        let (Some(sa), Some(sb)) = (a_lookup.get(&m.query_index), b_lookup.get(&m.train_index))
        else {
            continue;
        };
        out.push(StereoPairCorrespondence {
            a: sa.point_cam,
            b: sb.point_cam,
            confidence: m.confidence,
        });
    }
    out
}

/// One observation contributed by a multi-frame [`StereoTrack`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoTrackObservation {
    pub frame_index: usize,
    pub left_index: usize,
    pub right_index: usize,
}

/// One multi-frame stereo track: a single landmark's world-frame position
/// (anchored from the seed frame's stereo triangulation, lifted into world
/// using the seed frame's pose) plus its observations across one or more
/// frames. Produced by [`extend_stereo_tracks_via_projection`].
#[derive(Debug, Clone, PartialEq)]
pub struct StereoTrack {
    pub landmark_world: Point3<f64>,
    pub observations: Vec<StereoTrackObservation>,
}

/// Configuration for [`extend_stereo_tracks_via_projection`]. Defaults are
/// tuned for KITTI-scale per-frame motion (~1 m/frame at stride 1, ~7 m/frame
/// at stride 8) and the project's `CornerFeatureExtractor` descriptor scale.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackExtensionConfig {
    /// Maximum image-pixel distance from the projected landmark within
    /// which to look for a matching keypoint at the next frame. Has to be
    /// generous enough to cover VO pose noise + reprojection error from
    /// per-frame stereo-triangulation inaccuracy, but small enough to keep
    /// the descriptor search discriminative — `25 px` works on KITTI 00.
    pub search_radius_px: f64,
    /// Lowe's ratio test on the within-radius descriptor search. The best
    /// match's descriptor distance must be below this fraction of the
    /// second best's, otherwise the track ends.
    pub ratio: f64,
    /// Tracks may extend by this many frames past their seed before being
    /// force-cut. Without a cap, tracks that follow distant landmarks can
    /// span ~all frames at the cost of a quadratic match. `30` is a
    /// reasonable upper bound for stride-1 KITTI windows.
    pub max_extension: usize,
    /// Each `(frame, left_kp_index)` is consumed by at most one track —
    /// once a feature has been added to a longer track its later
    /// re-seeding as a new track is suppressed. Set `false` to allow
    /// overlapping tracks (rarely useful).
    pub deduplicate: bool,
}

impl Default for TrackExtensionConfig {
    fn default() -> Self {
        Self {
            search_radius_px: 25.0,
            ratio: 0.8,
            max_extension: 30,
            deduplicate: true,
        }
    }
}

/// Build multi-frame stereo tracks by projecting each frame's stereo
/// triangulations forward through subsequent VO poses and searching for
/// descriptor-matching stereo features near the predicted pixel.
///
/// Why projection-guided rather than chained brute-force matching: KITTI's
/// per-frame motion takes individual features out of the camera's FOV
/// quickly (≤ 4 frames at stride 8), and the brute-force matcher between
/// `left_i ↔ left_(i+1)` does not transitively reuse the same `left_(i+1)`
/// keypoint across `(i → i+1)` and `(i+1 → i+2)`. Projecting the landmark
/// through the next pose narrows the search to a small image patch and
/// allows the same physical point to be re-identified across many frames,
/// which is what gives the bundle adjuster the long-baseline constraints
/// needed to attack rotation drift.
pub fn extend_stereo_tracks_via_projection(
    poses: &[Pose],
    left_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    camera: &Camera,
    config: &TrackExtensionConfig,
) -> Vec<StereoTrack> {
    let n = poses.len();
    if n < 2 || left_features.len() < n || stereo_per_frame.len() < n {
        return Vec::new();
    }
    // Per-frame: index every stereo triangulation by its left keypoint id
    // so projection-based extension can ask "is there a stereo feature at
    // keypoint k of frame j?" in O(1).
    let stereo_lookup: Vec<HashMap<usize, &StereoFeature>> = stereo_per_frame
        .iter()
        .map(|sf| sf.iter().map(|f| (f.left_index, f)).collect())
        .collect();
    let mut consumed: Vec<std::collections::HashSet<usize>> =
        (0..n).map(|_| std::collections::HashSet::new()).collect();
    let mut out: Vec<StereoTrack> = Vec::new();

    for seed_frame in 0..n - 1 {
        let pose_seed = &poses[seed_frame];
        let r_inv = pose_seed.world_to_camera.rotation.inverse();
        let t_seed = pose_seed.world_to_camera.translation;
        for stereo in &stereo_per_frame[seed_frame] {
            if config.deduplicate && consumed[seed_frame].contains(&stereo.left_index) {
                continue;
            }
            let landmark_w =
                nalgebra::Point3::from(r_inv.transform_vector(&(stereo.point_cam.coords - t_seed)));
            let descriptor = left_features[seed_frame].descriptors[stereo.left_index].clone();
            let mut observations: Vec<StereoTrackObservation> = vec![StereoTrackObservation {
                frame_index: seed_frame,
                left_index: stereo.left_index,
                right_index: stereo.right_index,
            }];
            let extension_cap = (seed_frame + config.max_extension).min(n - 1);
            for next_frame in seed_frame + 1..=extension_cap {
                let pose_next = &poses[next_frame];
                let xc = pose_next.transform_world_point(&landmark_w);
                if xc.z <= 0.0 {
                    break;
                }
                let Some(predicted) = camera.project(&xc) else {
                    break;
                };
                if predicted.x < 0.0
                    || predicted.x >= camera.width as f64
                    || predicted.y < 0.0
                    || predicted.y >= camera.height as f64
                {
                    break;
                }
                // Within-radius descriptor search constrained to keypoints
                // that have a stereo triangulation (otherwise the
                // landmark's BA observation is incomplete and the track
                // cannot contribute a stereo residual).
                let mut best: Option<(usize, f64)> = None;
                let mut second_best: Option<f64> = None;
                let radius_sq = config.search_radius_px * config.search_radius_px;
                for (kp_idx, kp) in left_features[next_frame].keypoints.iter().enumerate() {
                    let dx = kp.x - predicted.x;
                    let dy = kp.y - predicted.y;
                    if dx * dx + dy * dy > radius_sq {
                        continue;
                    }
                    if !stereo_lookup[next_frame].contains_key(&kp_idx) {
                        continue;
                    }
                    if config.deduplicate && consumed[next_frame].contains(&kp_idx) {
                        continue;
                    }
                    let dist = descriptor_distance(
                        &descriptor,
                        &left_features[next_frame].descriptors[kp_idx],
                    );
                    match best {
                        None => best = Some((kp_idx, dist)),
                        Some((_, b_dist)) if dist < b_dist => {
                            second_best = Some(b_dist);
                            best = Some((kp_idx, dist));
                        }
                        Some(_) => {
                            if second_best.is_none_or(|s| dist < s) {
                                second_best = Some(dist);
                            }
                        }
                    }
                }
                let Some((kp_idx, b_dist)) = best else {
                    break;
                };
                if let Some(s) = second_best {
                    if s > 0.0 && b_dist / s > config.ratio {
                        break;
                    }
                }
                let stereo_next = stereo_lookup[next_frame][&kp_idx];
                observations.push(StereoTrackObservation {
                    frame_index: next_frame,
                    left_index: kp_idx,
                    right_index: stereo_next.right_index,
                });
            }
            if observations.len() >= 2 {
                if config.deduplicate {
                    for obs in &observations {
                        consumed[obs.frame_index].insert(obs.left_index);
                    }
                }
                out.push(StereoTrack {
                    landmark_world: landmark_w,
                    observations,
                });
            }
        }
    }
    out
}

fn descriptor_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut sum = 0.0;
    let len = a.len().min(b.len());
    for i in 0..len {
        let d = (a[i] - b[i]) as f64;
        sum += d * d;
    }
    sum
}

/// Build 2D-3D correspondences for PnP between two consecutive frames, using
/// frame-`a`'s stereo triangulations and frame-`b`'s left features.
///
/// `temporal_matches` is the descriptor matcher's output of `left_a → left_b`.
/// Only matches whose query keypoint also has a triangulated stereo feature
/// in `a_features` are kept; the resulting [`Correspondence2D3D`] pairs the
/// 3D point (in `a`'s left-camera frame) with the 2D pixel in `b`'s left
/// image.
///
/// The PnP solver then recovers `T_a_to_b` directly: feed the result into
/// `PnPRansac::estimate` with `b_left`'s keypoints as the 2D side.
pub fn build_stereo_temporal_correspondences(
    a_features: &[StereoFeature],
    b_left: &FeatureSet,
    temporal_matches: &[crate::matching::DescriptorMatch],
) -> Vec<Correspondence2D3D> {
    if a_features.is_empty() || b_left.is_empty() || temporal_matches.is_empty() {
        return Vec::new();
    }
    // Index a_features by their left-keypoint index for O(1) lookup.
    let mut a_by_left_idx: HashMap<usize, &StereoFeature> =
        HashMap::with_capacity(a_features.len());
    for f in a_features {
        a_by_left_idx.insert(f.left_index, f);
    }
    let mut out: Vec<Correspondence2D3D> = Vec::with_capacity(temporal_matches.len());
    for m in temporal_matches {
        let Some(stereo) = a_by_left_idx.get(&m.query_index) else {
            continue;
        };
        let b_kp = b_left.keypoints[m.train_index];
        out.push(Correspondence2D3D {
            point2d: b_kp,
            point3d: stereo.point_cam,
            confidence: m.confidence,
        });
    }
    out
}

#[derive(Debug, Clone, PartialEq)]
struct PnpSelectedReport {
    pose: Pose,
    inlier_count: usize,
    mean_reprojection_error: f64,
    correspondence_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PnpFullScore {
    inlier_count: usize,
    mean_error: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Pose3d3dScore {
    inlier_count: usize,
    mean_residual_m: f64,
}

fn estimate_pnp_with_depth_hypotheses(
    pnp: &PnPRansac,
    all_corrs: &[Correspondence2D3D],
    camera: &Camera,
    min_inliers: usize,
    primary_max_depth_m: Option<f64>,
    depth_hypotheses_m: &[f64],
    adaptive_depth_hypothesis_m: Option<f64>,
    adaptive_depth_min_primary_inlier_ratio: f64,
) -> Option<PnpSelectedReport> {
    let primary_depth = primary_max_depth_m.filter(|d| d.is_finite() && *d > 0.0);
    let mut best: Option<PnpSelectedReport> = None;
    let mut best_score: Option<PnpFullScore> = None;

    update_best_pnp_depth_candidate(
        pnp,
        all_corrs,
        camera,
        min_inliers,
        primary_depth,
        &mut best,
        &mut best_score,
    );

    let mut candidate_depths = pnp_candidate_depths(primary_max_depth_m, depth_hypotheses_m);
    if depth_hypotheses_m.is_empty()
        && should_try_adaptive_pnp_depth(
            best_score.as_ref(),
            all_corrs.len(),
            adaptive_depth_min_primary_inlier_ratio,
        )
    {
        if let Some(adaptive_depth_m) =
            adaptive_depth_hypothesis_m.filter(|d| d.is_finite() && *d > 0.0)
        {
            let candidate = Some(adaptive_depth_m);
            if !candidate_depths
                .iter()
                .any(|existing| same_optional_depth(*existing, candidate))
            {
                candidate_depths.push(candidate);
            }
        }
    }

    for max_depth_m in candidate_depths {
        if same_optional_depth(max_depth_m, primary_depth) {
            continue;
        }
        update_best_pnp_depth_candidate(
            pnp,
            all_corrs,
            camera,
            min_inliers,
            max_depth_m,
            &mut best,
            &mut best_score,
        );
    }
    best
}

fn update_best_pnp_depth_candidate(
    pnp: &PnPRansac,
    all_corrs: &[Correspondence2D3D],
    camera: &Camera,
    min_inliers: usize,
    max_depth_m: Option<f64>,
    best: &mut Option<PnpSelectedReport>,
    best_score: &mut Option<PnpFullScore>,
) {
    let corrs = pnp_correspondences_for_depth(all_corrs, max_depth_m);
    let weights: Vec<f32> = corrs.iter().map(|c| c.confidence.unwrap_or(0.0)).collect();
    let Some(report) = estimate_pnp_candidate(pnp, &corrs, camera, &weights) else {
        return;
    };
    if report.inliers.len() < min_inliers {
        return;
    }
    let full_score = score_pnp_pose_on_correspondences(
        &report.pose,
        all_corrs,
        camera,
        pnp.reprojection_threshold,
    );
    if full_score.inlier_count < min_inliers {
        return;
    }
    let better = best_score
        .as_ref()
        .is_none_or(|score| is_pnp_candidate_better(&full_score, score));
    if better {
        *best_score = Some(full_score);
        *best = Some(PnpSelectedReport {
            pose: report.pose,
            inlier_count: full_score.inlier_count,
            mean_reprojection_error: full_score.mean_error,
            correspondence_count: corrs.len(),
        });
    }
}

fn should_try_adaptive_pnp_depth(
    primary_score: Option<&PnpFullScore>,
    total_correspondences: usize,
    min_primary_inlier_ratio: f64,
) -> bool {
    if total_correspondences == 0
        || !min_primary_inlier_ratio.is_finite()
        || min_primary_inlier_ratio <= 0.0
    {
        return false;
    }
    let Some(primary_score) = primary_score else {
        return true;
    };
    let primary_ratio = primary_score.inlier_count as f64 / total_correspondences as f64;
    primary_ratio < min_primary_inlier_ratio
}

fn pnp_candidate_depths(
    primary_max_depth_m: Option<f64>,
    depth_hypotheses_m: &[f64],
) -> Vec<Option<f64>> {
    let mut depths = Vec::new();
    depths.push(primary_max_depth_m.filter(|d| d.is_finite() && *d > 0.0));
    for &depth in depth_hypotheses_m {
        if !depth.is_finite() || depth <= 0.0 {
            continue;
        }
        let candidate = Some(depth);
        if !depths
            .iter()
            .any(|existing| same_optional_depth(*existing, candidate))
        {
            depths.push(candidate);
        }
    }
    depths
}

fn same_optional_depth(a: Option<f64>, b: Option<f64>) -> bool {
    match (a, b) {
        (None, None) => true,
        (Some(a), Some(b)) => (a - b).abs() <= 1.0e-9,
        _ => false,
    }
}

fn pnp_correspondences_for_depth(
    all_corrs: &[Correspondence2D3D],
    max_depth_m: Option<f64>,
) -> Vec<Correspondence2D3D> {
    match max_depth_m {
        Some(max_depth_m) => all_corrs
            .iter()
            .filter(|corr| corr.point3d.z <= max_depth_m)
            .cloned()
            .collect(),
        None => all_corrs.to_vec(),
    }
}

fn estimate_pnp_candidate(
    pnp: &PnPRansac,
    corrs: &[Correspondence2D3D],
    camera: &Camera,
    weights: &[f32],
) -> Option<crate::ransac::RansacReport> {
    if weights.iter().any(|w| w.is_finite() && *w > 0.0) {
        pnp.estimate_with_weights(corrs, camera, weights)
    } else {
        pnp.estimate(corrs, camera)
    }
}

fn score_pnp_pose_on_correspondences(
    pose: &Pose,
    corrs: &[Correspondence2D3D],
    camera: &Camera,
    threshold_px: f64,
) -> PnpFullScore {
    let mut inlier_count = 0usize;
    let mut error_sum = 0.0;
    for corr in corrs {
        let projected_point = pose.transform_world_point(&corr.point3d);
        let Some(projected) = camera.project(&projected_point) else {
            continue;
        };
        let err = (projected - corr.point2d).norm();
        if err <= threshold_px {
            inlier_count += 1;
            error_sum += err;
        }
    }
    let mean_error = if inlier_count == 0 {
        f64::INFINITY
    } else {
        error_sum / inlier_count as f64
    };
    PnpFullScore {
        inlier_count,
        mean_error,
    }
}

fn is_pnp_candidate_better(candidate: &PnpFullScore, baseline: &PnpFullScore) -> bool {
    candidate.inlier_count > baseline.inlier_count
        || (candidate.inlier_count == baseline.inlier_count
            && candidate.mean_error + 1.0e-9 < baseline.mean_error)
}

fn score_pose_on_stereo_pairs(
    pose: &Pose,
    correspondences: &[StereoPairCorrespondence],
    inlier_threshold_m: f64,
) -> Pose3d3dScore {
    if correspondences.is_empty() || !inlier_threshold_m.is_finite() || inlier_threshold_m <= 0.0 {
        return Pose3d3dScore {
            inlier_count: 0,
            mean_residual_m: f64::INFINITY,
        };
    }
    let (inliers, sse) = score_pose_3d3d(pose, correspondences, inlier_threshold_m.powi(2));
    let mean_residual_m = if inliers.is_empty() {
        f64::INFINITY
    } else {
        (sse / inliers.len() as f64).sqrt()
    };
    Pose3d3dScore {
        inlier_count: inliers.len(),
        mean_residual_m,
    }
}

fn should_try_kabsch_challenger(
    pnp_report: &PnpSelectedReport,
    config: &StereoVoFrontendConfig,
) -> bool {
    if pnp_report.correspondence_count == 0 {
        return true;
    }
    if !config.pnp_kabsch_challenge_max_inlier_ratio.is_finite()
        || config.pnp_kabsch_challenge_max_inlier_ratio <= 0.0
    {
        return false;
    }
    let ratio = pnp_report.inlier_count as f64 / pnp_report.correspondence_count as f64;
    ratio < config.pnp_kabsch_challenge_max_inlier_ratio
}

fn is_kabsch_challenger_better(
    kabsch_report: &KabschRansacReport,
    pnp_stereo_score: Pose3d3dScore,
    config: &StereoVoFrontendConfig,
) -> bool {
    if kabsch_report.inliers.len()
        < pnp_stereo_score
            .inlier_count
            .saturating_add(config.pnp_kabsch_challenge_min_3d_inlier_gain)
    {
        return false;
    }
    if !pnp_stereo_score.mean_residual_m.is_finite() {
        return true;
    }
    kabsch_report.mean_residual_m
        <= pnp_stereo_score.mean_residual_m * config.pnp_kabsch_challenge_max_residual_ratio
}

fn filter_temporal_matches_by_row_delta(
    a_left: &FeatureSet,
    b_left: &FeatureSet,
    temporal_matches: &[DescriptorMatch],
    max_row_delta_px: f64,
) -> Vec<DescriptorMatch> {
    if !max_row_delta_px.is_finite() || max_row_delta_px <= 0.0 {
        return temporal_matches.to_vec();
    }
    temporal_matches
        .iter()
        .filter(|m| {
            let Some(a_kp) = a_left.keypoints.get(m.query_index) else {
                return false;
            };
            let Some(b_kp) = b_left.keypoints.get(m.train_index) else {
                return false;
            };
            (a_kp.y - b_kp.y).abs() <= max_row_delta_px
        })
        .cloned()
        .collect()
}

fn filter_temporal_matches_by_confidence(
    temporal_matches: &[DescriptorMatch],
    min_confidence: f32,
) -> Vec<DescriptorMatch> {
    if !min_confidence.is_finite() || min_confidence <= 0.0 {
        return temporal_matches.to_vec();
    }
    if !temporal_matches
        .iter()
        .any(|m| m.confidence.is_some_and(|c| c.is_finite()))
    {
        return temporal_matches.to_vec();
    }
    temporal_matches
        .iter()
        .filter(|m| {
            m.confidence
                .is_some_and(|confidence| confidence.is_finite() && confidence >= min_confidence)
        })
        .cloned()
        .collect()
}

fn effective_temporal_max_row_delta_px(
    translation_history_m: &[f64],
    config: &StereoVoFrontendConfig,
) -> Option<f64> {
    if let Some(max_row_delta_px) = config.temporal_max_row_delta_px {
        return finite_positive(max_row_delta_px);
    }
    let auto_max = finite_positive(config.temporal_auto_max_row_delta_px?)?;
    if translation_history_m.len() < config.temporal_auto_min_history {
        return None;
    }
    let recent_median =
        recent_translation_median(translation_history_m, config.temporal_auto_min_history)?;
    if recent_median >= config.temporal_auto_min_median_translation_m {
        Some(auto_max)
    } else {
        None
    }
}

fn effective_temporal_min_confidence(
    translation_history_m: &[f64],
    rotation_history_deg: &[f64],
    config: &StereoVoFrontendConfig,
) -> Option<f32> {
    let min_confidence = config.temporal_auto_min_confidence?;
    if !min_confidence.is_finite() || min_confidence <= 0.0 {
        return None;
    }
    if translation_history_m.len() < config.temporal_auto_confidence_min_history {
        return None;
    }
    let recent_translation_median = recent_translation_median(
        translation_history_m,
        config.temporal_auto_confidence_min_history,
    )?;
    let max_rotation_deg = config
        .temporal_auto_confidence_max_median_rotation_deg
        .and_then(finite_positive);
    let needs_rotation_median = max_rotation_deg.is_some()
        || config
            .temporal_auto_confidence_curve_min_median_translation_m
            .and_then(finite_positive)
            .is_some()
        || config
            .temporal_auto_confidence_curve_min_median_rotation_deg
            .and_then(finite_positive)
            .is_some();
    let recent_rotation_median = if needs_rotation_median {
        if rotation_history_deg.len() < config.temporal_auto_confidence_min_history {
            return None;
        }
        Some(recent_positive_median(
            rotation_history_deg,
            config.temporal_auto_confidence_min_history,
        )?)
    } else {
        None
    };
    if recent_translation_median >= config.temporal_auto_confidence_min_median_translation_m
        && max_rotation_deg
            .is_none_or(|max| recent_rotation_median.is_some_and(|rotation| rotation <= max))
    {
        return Some(min_confidence);
    }
    let Some(curve_min_translation_m) = config
        .temporal_auto_confidence_curve_min_median_translation_m
        .and_then(finite_positive)
    else {
        return None;
    };
    let Some(curve_min_rotation_deg) = config
        .temporal_auto_confidence_curve_min_median_rotation_deg
        .and_then(finite_positive)
    else {
        return None;
    };
    if recent_translation_median < curve_min_translation_m {
        return None;
    }
    let Some(recent_rotation_median) = recent_rotation_median else {
        return None;
    };
    if recent_rotation_median < curve_min_rotation_deg {
        return None;
    }
    if max_rotation_deg.is_some_and(|max| recent_rotation_median > max) {
        return None;
    }
    Some(min_confidence)
}

fn finite_positive(value: f64) -> Option<f64> {
    if value.is_finite() && value > 0.0 {
        Some(value)
    } else {
        None
    }
}

#[derive(Debug, Clone, PartialEq)]
struct StereoPoseObservation {
    point_prev: Point3<f64>,
    left_px: Point2<f64>,
    right_px: Point2<f64>,
}

fn refine_relative_pose_with_current_stereo(
    initial_pose: &Pose,
    prev_stereo: &[StereoFeature],
    current_stereo: &[StereoFeature],
    current_left: &FeatureSet,
    current_right: &FeatureSet,
    temporal_matches: &[DescriptorMatch],
    camera: &Camera,
    baseline: f64,
    max_initial_left_error_px: f64,
) -> Option<Pose> {
    let observations = stereo_pose_observations(
        initial_pose,
        prev_stereo,
        current_stereo,
        current_left,
        current_right,
        temporal_matches,
        camera,
        max_initial_left_error_px,
    );
    if observations.len() < 12 {
        return None;
    }
    refine_pose_stereo_reprojection(initial_pose, &observations, camera, baseline)
}

fn stereo_pose_observations(
    initial_pose: &Pose,
    prev_stereo: &[StereoFeature],
    current_stereo: &[StereoFeature],
    current_left: &FeatureSet,
    current_right: &FeatureSet,
    temporal_matches: &[DescriptorMatch],
    camera: &Camera,
    max_initial_left_error_px: f64,
) -> Vec<StereoPoseObservation> {
    let prev_lookup: HashMap<usize, &StereoFeature> =
        prev_stereo.iter().map(|f| (f.left_index, f)).collect();
    let current_lookup: HashMap<usize, &StereoFeature> =
        current_stereo.iter().map(|f| (f.left_index, f)).collect();
    let mut observations = Vec::new();
    for m in temporal_matches {
        let (Some(prev), Some(curr)) = (
            prev_lookup.get(&m.query_index),
            current_lookup.get(&m.train_index),
        ) else {
            continue;
        };
        let left_px = current_left.keypoints[m.train_index];
        let right_px = current_right.keypoints[curr.right_index];
        let Some(projected) = camera.project(&initial_pose.transform_world_point(&prev.point_cam))
        else {
            continue;
        };
        if (projected - left_px).norm() > max_initial_left_error_px {
            continue;
        }
        observations.push(StereoPoseObservation {
            point_prev: prev.point_cam,
            left_px,
            right_px,
        });
    }
    observations
}

fn refine_pose_stereo_reprojection(
    initial_pose: &Pose,
    observations: &[StereoPoseObservation],
    camera: &Camera,
    baseline: f64,
) -> Option<Pose> {
    let mut pose = initial_pose.clone();
    let mut best_error =
        stereo_reprojection_mean_squared_error(&pose, observations, camera, baseline)?;
    for _ in 0..6 {
        let residual = stereo_reprojection_residual(&pose, observations, camera, baseline)?;
        let jacobian = stereo_translation_jacobian(&pose, observations, camera, baseline, 1.0e-6)?;
        let j_t = jacobian.transpose();
        let mut hessian = &j_t * &jacobian;
        for diagonal in 0..hessian.nrows().min(hessian.ncols()) {
            hessian[(diagonal, diagonal)] += 1.0e-5;
        }
        let gradient = &j_t * residual;
        let step = hessian.lu().solve(&(-gradient))?;
        if !step.iter().all(|value| value.is_finite()) {
            return Some(pose);
        }
        let candidate = perturb_stereo_translation(&pose, &step);
        let Some(candidate_error) =
            stereo_reprojection_mean_squared_error(&candidate, observations, camera, baseline)
        else {
            break;
        };
        if candidate_error + 1.0e-9 >= best_error {
            break;
        }
        pose = candidate;
        best_error = candidate_error;
    }
    Some(pose)
}

fn stereo_reprojection_mean_squared_error(
    pose: &Pose,
    observations: &[StereoPoseObservation],
    camera: &Camera,
    baseline: f64,
) -> Option<f64> {
    let residual = stereo_reprojection_residual(pose, observations, camera, baseline)?;
    Some(residual.norm_squared() / observations.len() as f64)
}

fn stereo_reprojection_residual(
    pose: &Pose,
    observations: &[StereoPoseObservation],
    camera: &Camera,
    baseline: f64,
) -> Option<DVector<f64>> {
    let mut residual = DVector::<f64>::zeros(observations.len() * 4);
    for (index, obs) in observations.iter().enumerate() {
        let point_left = pose.transform_world_point(&obs.point_prev);
        let left_projected = camera.project(&point_left)?;
        let right_projected = camera.project(&Point3::new(
            point_left.x - baseline,
            point_left.y,
            point_left.z,
        ))?;
        let offset = index * 4;
        residual[offset] = left_projected.x - obs.left_px.x;
        residual[offset + 1] = left_projected.y - obs.left_px.y;
        residual[offset + 2] = right_projected.x - obs.right_px.x;
        residual[offset + 3] = right_projected.y - obs.right_px.y;
    }
    Some(residual)
}

fn stereo_translation_jacobian(
    pose: &Pose,
    observations: &[StereoPoseObservation],
    camera: &Camera,
    baseline: f64,
    epsilon: f64,
) -> Option<DMatrix<f64>> {
    let base = stereo_reprojection_residual(pose, observations, camera, baseline)?;
    let mut jacobian = DMatrix::<f64>::zeros(base.len(), 3);
    for parameter_index in 0..3 {
        let mut delta = DVector::<f64>::zeros(3);
        delta[parameter_index] = epsilon;
        let perturbed_pose = perturb_stereo_translation(pose, &delta);
        let perturbed =
            stereo_reprojection_residual(&perturbed_pose, observations, camera, baseline)?;
        let column = (perturbed - &base) / epsilon;
        jacobian.set_column(parameter_index, &column);
    }
    Some(jacobian)
}

fn perturb_stereo_translation(pose: &Pose, delta: &DVector<f64>) -> Pose {
    let translation_delta = Vector3::new(delta[0], delta[1], delta[2]);
    Pose::from_world_to_camera(
        pose.world_to_camera.rotation,
        pose.world_to_camera.translation + translation_delta,
    )
}

/// Configuration for [`StereoVoFrontend`]. Defaults are tuned for KITTI-
/// scale rectified stereo with the project's `CornerFeatureExtractor`
/// patch descriptors at radius 9; tighter `kabsch.inlier_threshold_m` /
/// looser `stereo.max_depth_m` may be appropriate for sequences with
/// faster motion or longer-range structure.
#[derive(Debug, Clone, PartialEq)]
pub struct StereoVoFrontendConfig {
    pub feature: CornerFeatureConfig,
    pub stereo: StereoFeatureConfig,
    pub kabsch: KabschRansacConfig,
    /// Pixel threshold for the 2D-3D PnP RANSAC reprojection inlier test.
    /// Lower values reject descriptor-consistent but geometrically loose
    /// matches; higher values are more tolerant when features are noisy.
    pub pnp_reprojection_threshold_px: f64,
    /// Minimum accepted 2D-3D PnP inlier count. If PnP recovers fewer
    /// inliers, the frontend treats it as a failed primary estimate and
    /// falls back to the 3D-3D Kabsch path when available.
    pub pnp_min_inliers: usize,
    /// Optional maximum depth for 3D points passed into consecutive-frame
    /// PnP. Stereo depth noise grows quadratically with distance, so
    /// dropping very far landmarks can reduce rotation drift on driving
    /// sequences. `None` keeps every triangulated point accepted by the
    /// stereo frontend.
    pub pnp_max_depth_m: Option<f64>,
    /// Optional extra max-depth hypotheses for PnP. When populated, the
    /// frontend estimates PnP once with the primary `pnp_max_depth_m` setting
    /// and once for each extra depth cap, then selects the pose that scores
    /// best when reprojected against the full uncapped correspondence set.
    /// This lets far-field pruning act as a candidate generator without
    /// making a depth-capped pose eligible when it loses global consensus.
    pub pnp_depth_hypotheses_m: Vec<f64>,
    /// Optional automatic PnP max-depth candidate. Unlike
    /// `pnp_depth_hypotheses_m`, this candidate is tried only when the
    /// primary uncapped/depth-capped PnP estimate has weak consensus.
    pub pnp_adaptive_depth_hypothesis_m: Option<f64>,
    /// Primary PnP inlier ratio below which the adaptive depth candidate is
    /// evaluated. This is intentionally conservative so clean short windows
    /// keep the primary path unchanged.
    pub pnp_adaptive_depth_min_primary_inlier_ratio: f64,
    /// Number of prior relative-motion estimates required before the frontend
    /// can rescue a collapsed translation magnitude from recent motion scale.
    pub motion_scale_rescue_min_history: usize,
    /// Minimum recent median translation required before motion-scale rescue
    /// is considered. This keeps slow urban windows on the raw estimator path.
    pub motion_scale_rescue_min_median_translation_m: f64,
    /// Trigger rescue when the current translation magnitude is below this
    /// fraction of the recent median and the pose consensus is weak.
    pub motion_scale_rescue_min_translation_ratio: f64,
    /// Trigger rescue when the current translation magnitude is above this
    /// multiple of the recent median and the pose consensus is weak.
    pub motion_scale_rescue_max_translation_ratio: f64,
    /// Percentile of the recent translation history used as the replacement
    /// scale once motion-scale rescue triggers. Values above the median make
    /// the fast-motion guard less laggy on accelerating driving sequences.
    pub motion_scale_rescue_target_percentile: f64,
    /// Maximum PnP inlier ratio eligible for motion-scale rescue. Keeping
    /// this slightly above one lets the fast-motion scale guard smooth clean
    /// but laggy PnP pairs, while the median-translation gate keeps slow
    /// urban windows on the raw estimator path.
    pub motion_scale_rescue_max_pnp_inlier_ratio: f64,
    /// Number of prior relative-motion estimates required before translation
    /// direction rescue can clamp a weak-consensus lateral outlier.
    pub translation_direction_rescue_min_history: usize,
    /// Minimum recent median translation required before translation direction
    /// rescue is considered.
    pub translation_direction_rescue_min_median_translation_m: f64,
    /// Trigger translation direction rescue when the current translation vector
    /// deviates from the recent average direction by more than this angle.
    pub translation_direction_rescue_max_angle_deg: f64,
    /// Maximum PnP inlier ratio still considered weak enough for translation
    /// direction rescue. Kabsch fallbacks are considered weak.
    pub translation_direction_rescue_max_pnp_inlier_ratio: f64,
    /// Number of prior relative-motion estimates required before rotation
    /// spike rescue can clamp an implausibly large weak-consensus rotation.
    pub rotation_spike_rescue_min_history: usize,
    /// Minimum recent median translation required before rotation spike rescue
    /// is considered.
    pub rotation_spike_rescue_min_median_translation_m: f64,
    /// Minimum current relative-rotation angle before rotation spike rescue can
    /// trigger, even if the recent-median ratio is exceeded.
    pub rotation_spike_rescue_min_angle_deg: f64,
    /// Trigger rotation spike rescue when the current rotation angle exceeds
    /// this multiple of the recent median rotation angle.
    pub rotation_spike_rescue_max_angle_ratio: f64,
    /// Maximum PnP inlier ratio still considered weak enough for rotation
    /// spike rescue. Kabsch fallbacks are considered weak.
    pub rotation_spike_rescue_max_pnp_inlier_ratio: f64,
    /// Number of prior relative-motion estimates required before weak
    /// rotations can be clamped to the recent scaled-axis trend.
    pub rotation_vector_rescue_min_history: usize,
    /// Minimum recent median translation required before rotation-vector
    /// rescue is considered.
    pub rotation_vector_rescue_min_median_translation_m: f64,
    /// Trigger rotation-vector rescue when the current scaled-axis rotation
    /// differs from the recent average scaled-axis rotation by more than this
    /// angle.
    pub rotation_vector_rescue_max_delta_deg: f64,
    /// Maximum PnP inlier ratio still considered weak enough for
    /// rotation-vector rescue. Kabsch fallbacks are considered weak.
    pub rotation_vector_rescue_max_pnp_inlier_ratio: f64,
    /// Minimum PnP RANSAC trials before the high-consensus early-stop guard
    /// can terminate the search. This only applies when
    /// `pnp_early_stop_inlier_ratio` is `Some`.
    pub pnp_early_stop_min_iterations: usize,
    /// Optional high-consensus early-stop ratio for consecutive-frame PnP.
    /// KITTI-like stereo pairs often reach a very large inlier set quickly;
    /// stopping after that point avoids pathological long frames while
    /// preserving the full budget for lower-consensus pairs.
    pub pnp_early_stop_inlier_ratio: Option<f64>,
    /// When PnP technically succeeds but has a very low inlier ratio, also
    /// estimate a 3D-3D Kabsch pose and let it challenge the PnP pose.
    /// This targets highway windows where a small descriptor-consistent PnP
    /// consensus can carry a rotation/translation outlier even though the
    /// stereo pair set contains a stronger metric alignment.
    pub pnp_kabsch_challenge_max_inlier_ratio: f64,
    /// Minimum 3D-3D inlier-count gain required before the weak-PnP Kabsch
    /// challenger can replace PnP.
    pub pnp_kabsch_challenge_min_3d_inlier_gain: usize,
    /// Required residual improvement for the Kabsch challenger relative to
    /// scoring the PnP pose on the same 3D-3D stereo pairs.
    pub pnp_kabsch_challenge_max_residual_ratio: f64,
    /// If enabled, refine only the selected relative pose translation
    /// against current-frame left and right stereo reprojection residuals
    /// for temporal matches that have stereo triangulations in both frames.
    /// Rotation stays anchored to PnP/Kabsch so current-frame stereo
    /// disparity noise cannot pull the orientation estimate around.
    pub stereo_pose_refinement: bool,
    /// If enabled, align only the selected relative pose's vertical
    /// translation component against matched stereo 3D pairs. This keeps
    /// PnP's rotation and forward/lateral translation while using the
    /// current stereo triangulation as a robust vertical residual check.
    pub stereo_vertical_alignment: bool,
    /// Minimum matched stereo 3D pairs required before vertical-only
    /// alignment can run.
    pub stereo_vertical_alignment_min_pairs: usize,
    /// Maximum absolute per-pair vertical correction in meters.
    pub stereo_vertical_alignment_max_correction_m: f64,
    /// Number of prior relative-motion estimates required before automatic
    /// stereo pose refinement can run without the explicit
    /// `stereo_pose_refinement` override.
    pub stereo_pose_refinement_auto_min_history: usize,
    /// Minimum recent median translation required before automatic stereo
    /// pose refinement is considered.
    pub stereo_pose_refinement_auto_min_median_translation_m: f64,
    /// Maximum PnP inlier ratio eligible for automatic stereo pose
    /// refinement once the recent motion is fast enough. Kabsch fallbacks are
    /// considered eligible by construction.
    pub stereo_pose_refinement_auto_max_pnp_inlier_ratio: f64,
    /// Consecutive-frame relative-pose source. `PnpThenKabsch` preserves
    /// the default 2D-3D PnP path and uses 3D-3D Kabsch only as a fallback;
    /// `KabschThenPnp` is useful for driving/stereo ablations where metric
    /// 3D-3D alignment may reduce forward-motion PnP drift.
    pub relative_pose_mode: StereoRelativePoseMode,
    /// Lowe's ratio for the per-pair temporal `left_i ↔ left_(i+1)`
    /// descriptor matcher. Native to the project's `BruteForceMatcher`
    /// (`f32`), kept here as `Option<f32>` to avoid a lossy cast at
    /// every `process_pair`.
    pub temporal_ratio: Option<f32>,
    /// Optional row-motion gate for consecutive-frame temporal matches.
    /// Rectified automotive stereo has limited vertical frame-to-frame
    /// motion; setting this can remove descriptor-consistent but implausible
    /// temporal matches before PnP/Kabsch. `None` preserves unconstrained
    /// temporal matching.
    pub temporal_max_row_delta_px: Option<f64>,
    /// Optional automatic row-motion gate for faster recent motion. Unlike
    /// `temporal_max_row_delta_px`, this only activates after enough prior
    /// relative translations have been integrated and their recent median is
    /// above `temporal_auto_min_median_translation_m`.
    pub temporal_auto_max_row_delta_px: Option<f64>,
    pub temporal_auto_min_history: usize,
    pub temporal_auto_min_median_translation_m: f64,
    /// Optional automatic confidence floor for consecutive-frame temporal
    /// matches during faster recent motion. Matchers without confidence
    /// metadata are left unchanged, so this only affects deep-style matchers
    /// such as [`crate::matching::MutualSoftmaxMatcher`].
    pub temporal_auto_min_confidence: Option<f32>,
    pub temporal_auto_confidence_min_history: usize,
    pub temporal_auto_confidence_min_median_translation_m: f64,
    pub temporal_auto_confidence_curve_min_median_translation_m: Option<f64>,
    pub temporal_auto_confidence_curve_min_median_rotation_deg: Option<f64>,
    pub temporal_auto_confidence_max_median_rotation_deg: Option<f64>,
}

impl Default for StereoVoFrontendConfig {
    fn default() -> Self {
        Self {
            feature: CornerFeatureConfig {
                max_features: 1500,
                min_score: 0.02,
                descriptor_radius: 9,
            },
            stereo: StereoFeatureConfig {
                min_disparity_px: 1.0,
                max_row_residual_px: 2.0,
                min_depth_m: 3.0,
                max_depth_m: 80.0,
                ratio: Some(0.85),
            },
            kabsch: KabschRansacConfig {
                iterations: 4000,
                inlier_threshold_m: 2.0,
                min_inliers: 8,
                max_depth_m: 30.0,
                seed: 7,
            },
            pnp_reprojection_threshold_px: 3.32,
            pnp_min_inliers: 12,
            pnp_max_depth_m: None,
            pnp_depth_hypotheses_m: Vec::new(),
            pnp_adaptive_depth_hypothesis_m: Some(60.0),
            pnp_adaptive_depth_min_primary_inlier_ratio: 0.65,
            motion_scale_rescue_min_history: 20,
            motion_scale_rescue_min_median_translation_m: 1.5,
            motion_scale_rescue_min_translation_ratio: 0.97,
            motion_scale_rescue_max_translation_ratio: 1.6,
            motion_scale_rescue_target_percentile: 0.75,
            motion_scale_rescue_max_pnp_inlier_ratio: 1.05,
            translation_direction_rescue_min_history: 20,
            translation_direction_rescue_min_median_translation_m: 1.5,
            translation_direction_rescue_max_angle_deg: 10.0,
            translation_direction_rescue_max_pnp_inlier_ratio: 0.45,
            rotation_spike_rescue_min_history: 20,
            rotation_spike_rescue_min_median_translation_m: 1.5,
            rotation_spike_rescue_min_angle_deg: 1.0,
            rotation_spike_rescue_max_angle_ratio: 3.0,
            rotation_spike_rescue_max_pnp_inlier_ratio: 0.45,
            rotation_vector_rescue_min_history: 20,
            rotation_vector_rescue_min_median_translation_m: 1.5,
            rotation_vector_rescue_max_delta_deg: 0.4,
            rotation_vector_rescue_max_pnp_inlier_ratio: 0.45,
            pnp_early_stop_min_iterations: 100,
            pnp_early_stop_inlier_ratio: Some(0.85),
            pnp_kabsch_challenge_max_inlier_ratio: 0.0,
            pnp_kabsch_challenge_min_3d_inlier_gain: 8,
            pnp_kabsch_challenge_max_residual_ratio: 0.85,
            stereo_pose_refinement: false,
            stereo_vertical_alignment: false,
            stereo_vertical_alignment_min_pairs: 80,
            stereo_vertical_alignment_max_correction_m: 0.25,
            stereo_pose_refinement_auto_min_history: 20,
            stereo_pose_refinement_auto_min_median_translation_m: 1.5,
            stereo_pose_refinement_auto_max_pnp_inlier_ratio: 1.0,
            relative_pose_mode: StereoRelativePoseMode::PnpThenKabsch,
            temporal_ratio: Some(0.9),
            temporal_max_row_delta_px: None,
            temporal_auto_max_row_delta_px: None,
            temporal_auto_min_history: 20,
            temporal_auto_min_median_translation_m: 1.05,
            temporal_auto_min_confidence: Some(0.20),
            temporal_auto_confidence_min_history: 20,
            temporal_auto_confidence_min_median_translation_m: 1.45,
            temporal_auto_confidence_curve_min_median_translation_m: Some(0.95),
            temporal_auto_confidence_curve_min_median_rotation_deg: Some(0.26),
            temporal_auto_confidence_max_median_rotation_deg: Some(0.45),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StereoRelativePoseMode {
    PnpThenKabsch,
    KabschThenPnp,
}

/// Errors returned by [`StereoVoFrontend::process_pair`]. The feature
/// extractor's error is type-erased into a `Box<dyn Error>` so the same
/// `StereoVoError` type accepts any [`FeatureExtractor`] implementation
/// (classical `CornerFeatureExtractor`, learned `HogLikeFeatureExtractor`,
/// or a future ONNX-backed deep head).
#[derive(Debug)]
pub enum StereoVoError {
    Feature(Box<dyn std::error::Error + Send + Sync>),
    KabschFailed {
        pair_index: usize,
        correspondence_count: usize,
        min_inliers: usize,
    },
}

impl From<CornerFeatureError> for StereoVoError {
    fn from(value: CornerFeatureError) -> Self {
        StereoVoError::Feature(Box::new(value))
    }
}

impl std::fmt::Display for StereoVoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StereoVoError::Feature(e) => write!(f, "feature extraction failed: {e}"),
            StereoVoError::KabschFailed {
                pair_index,
                correspondence_count,
                min_inliers,
            } => write!(
                f,
                "relative-pose RANSAC failed at pair {pair_index}→{}: {correspondence_count} \
                 correspondences (need ≥{min_inliers} inliers)",
                pair_index + 1,
            ),
        }
    }
}

impl std::error::Error for StereoVoError {}

/// Stateful rectified-stereo VO frontend. Each `process_pair` call extracts
/// features in both eyes, triangulates them, matches the previous frame's
/// left features against the new frame's, and recovers the relative pose via
/// 2D-3D PnP RANSAC with a 3D-3D Kabsch fallback. The first call seeds the
/// world frame at identity; subsequent calls compose the recovered relative
/// pose into the running world-frame trajectory.
///
/// Per-frame state (FeatureSets, stereo triangulations, world-frame poses)
/// is exposed via the public fields so a downstream BA orchestrator can
/// build multi-frame tracks across the trajectory and refine without
/// re-running the front-end.
#[derive(Debug)]
pub struct StereoVoFrontend<E = CornerFeatureExtractor, M = BruteForceMatcher>
where
    E: FeatureExtractor<Image = GrayscaleImage>,
    M: Matcher,
{
    pub camera: Camera,
    pub baseline: f64,
    pub config: StereoVoFrontendConfig,

    extractor: E,
    matcher: M,

    /// World-frame pose for every processed frame, beginning with the
    /// identity pose at the seed frame.
    pub poses: Vec<Pose>,
    pub left_features: Vec<FeatureSet>,
    pub right_features: Vec<FeatureSet>,
    pub stereo_per_frame: Vec<Vec<StereoFeature>>,
    /// Per-pair Kabsch translation magnitude in metric units (one entry
    /// per call after the first). Used by track-extension callers to
    /// adapt the projection search radius to the observed motion.
    pub per_pair_translation_m: Vec<f64>,
    /// Per-pair relative translation vectors in the current camera frame,
    /// after any guarded frontend rescues have been applied.
    pub per_pair_translation_vectors: Vec<Vector3<f64>>,
    /// Per-pair relative-rotation magnitudes in degrees, after any guarded
    /// frontend rescues have been applied.
    pub per_pair_rotation_deg: Vec<f64>,
    /// Per-pair relative-rotation scaled-axis vectors in radians, after any
    /// guarded frontend rescues have been applied.
    pub per_pair_rotation_vectors: Vec<Vector3<f64>>,
    /// Per-pair relative-pose RANSAC inlier counts.
    pub kabsch_inlier_counts: Vec<usize>,
    /// Per-pair relative-pose diagnostics, including which estimator was
    /// actually selected. Kept separate from `kabsch_inlier_counts` because
    /// the latter is a legacy name used by existing examples.
    pub pair_diagnostics: Vec<StereoVoPairDiagnostics>,
    /// Per-pair temporal matches actually used (after row-gate and
    /// confidence filters). One entry per pose pair, i.e. `poses.len() - 1`
    /// entries after the seed frame. Captured so downstream multi-frame BA
    /// refiners (e.g. `visloc_slam::refine_stereo_vo_with_ba`) can be driven
    /// online without re-running the matcher.
    pub temporal_matches_per_pair: Vec<Vec<DescriptorMatch>>,
}

impl StereoVoFrontend<CornerFeatureExtractor, BruteForceMatcher> {
    /// Build the classical Corner + brute-force frontend that the original
    /// stereo VO demos use. Equivalent to
    /// `new_with(camera, baseline, config, CornerFeatureExtractor::new(...),
    /// BruteForceMatcher { ratio: config.temporal_ratio })`.
    pub fn new(camera: Camera, baseline: f64, config: StereoVoFrontendConfig) -> Self {
        let extractor = CornerFeatureExtractor::new(config.feature.clone());
        let matcher = BruteForceMatcher {
            ratio: config.temporal_ratio,
        };
        Self::new_with(camera, baseline, config, extractor, matcher)
    }
}

impl<E, M> StereoVoFrontend<E, M>
where
    E: FeatureExtractor<Image = GrayscaleImage>,
    E::Error: std::error::Error + Send + Sync + 'static,
    M: Matcher,
{
    /// Build a stereo VO frontend with caller-provided feature extractor
    /// and matcher. Use this constructor to plug in a deep frontend
    /// (`HogLikeFeatureExtractor` + `MutualSoftmaxMatcher`) or a future
    /// ONNX-backed learned head — the rest of the pipeline (stereo
    /// triangulation, Kabsch RANSAC, world-frame composition) is shared.
    pub fn new_with(
        camera: Camera,
        baseline: f64,
        config: StereoVoFrontendConfig,
        extractor: E,
        matcher: M,
    ) -> Self {
        Self {
            camera,
            baseline,
            config,
            extractor,
            matcher,
            poses: Vec::new(),
            left_features: Vec::new(),
            right_features: Vec::new(),
            stereo_per_frame: Vec::new(),
            per_pair_translation_m: Vec::new(),
            per_pair_translation_vectors: Vec::new(),
            per_pair_rotation_deg: Vec::new(),
            per_pair_rotation_vectors: Vec::new(),
            kabsch_inlier_counts: Vec::new(),
            pair_diagnostics: Vec::new(),
            temporal_matches_per_pair: Vec::new(),
        }
    }

    /// Number of frames processed so far.
    pub fn frame_count(&self) -> usize {
        self.poses.len()
    }

    /// Process a rectified stereo pair. The first call seeds the world
    /// frame at identity; each subsequent call extends the trajectory
    /// with one new pose and returns it.
    pub fn process_pair(
        &mut self,
        left: &GrayscaleImage,
        right: &GrayscaleImage,
    ) -> Result<Pose, StereoVoError> {
        let left_features = self
            .extractor
            .extract(left)
            .map_err(|e| StereoVoError::Feature(Box::new(e)))?;
        let right_features = self
            .extractor
            .extract(right)
            .map_err(|e| StereoVoError::Feature(Box::new(e)))?;
        self.process_feature_pair(left_features, right_features)
    }

    /// Process caller-provided left/right features using the frontend's
    /// configured descriptor matcher for stereo-temporal association.
    ///
    /// This is useful when feature extraction lives outside the frontend but
    /// descriptor matching should still use the Rust matcher.
    pub fn process_feature_pair(
        &mut self,
        left_features: FeatureSet,
        right_features: FeatureSet,
    ) -> Result<Pose, StereoVoError> {
        self.process_feature_pair_with_matches(left_features, right_features, None, None)
    }

    /// Process caller-provided left/right features and optional explicit
    /// matches.
    ///
    /// `stereo_matches` are interpreted as `left.query_index -> right.train_index`.
    /// `temporal_matches` are interpreted as
    /// `previous_left.query_index -> current_left.train_index`. When either
    /// argument is `None`, the frontend falls back to its configured internal
    /// matcher for that association.
    pub fn process_feature_pair_with_matches(
        &mut self,
        left_features: FeatureSet,
        right_features: FeatureSet,
        stereo_matches: Option<&[DescriptorMatch]>,
        temporal_matches: Option<&[DescriptorMatch]>,
    ) -> Result<Pose, StereoVoError> {
        let stereo = if let Some(stereo_matches) = stereo_matches {
            triangulate_stereo_feature_matches(
                &left_features,
                &right_features,
                stereo_matches,
                &self.camera,
                self.baseline,
                &self.config.stereo,
            )
        } else {
            triangulate_stereo_features(
                &left_features,
                &right_features,
                &self.camera,
                self.baseline,
                &self.config.stereo,
            )
        };

        let new_pose = if let (Some(prev_left), Some(prev_stereo)) =
            (self.left_features.last(), self.stereo_per_frame.last())
        {
            let temporal_matches_raw = temporal_matches.map_or_else(
                || {
                    self.matcher
                        .match_descriptors(&prev_left.descriptors, &left_features.descriptors)
                },
                |matches| matches.to_vec(),
            );
            let temporal_row_gate_px =
                effective_temporal_max_row_delta_px(&self.per_pair_translation_m, &self.config);
            let temporal_confidence_gate = effective_temporal_min_confidence(
                &self.per_pair_translation_m,
                &self.per_pair_rotation_deg,
                &self.config,
            );
            let temporal_matches = if let Some(max_row_delta_px) = temporal_row_gate_px {
                filter_temporal_matches_by_row_delta(
                    prev_left,
                    &left_features,
                    &temporal_matches_raw,
                    max_row_delta_px,
                )
            } else {
                temporal_matches_raw
            };
            let temporal_matches = if let Some(min_confidence) = temporal_confidence_gate {
                filter_temporal_matches_by_confidence(&temporal_matches, min_confidence)
            } else {
                temporal_matches
            };
            let pnp_corrs_all = build_stereo_temporal_correspondences(
                prev_stereo,
                &left_features,
                &temporal_matches,
            );
            let pnp = PnPRansac {
                iterations: self.config.kabsch.iterations,
                reprojection_threshold: self.config.pnp_reprojection_threshold_px,
                seed: self.config.kabsch.seed,
                early_stop_min_iterations: self.config.pnp_early_stop_min_iterations,
                early_stop_inlier_ratio: self.config.pnp_early_stop_inlier_ratio,
                ..PnPRansac::default()
            };
            let pnp_report = estimate_pnp_with_depth_hypotheses(
                &pnp,
                &pnp_corrs_all,
                &self.camera,
                self.config.pnp_min_inliers,
                self.config.pnp_max_depth_m,
                &self.config.pnp_depth_hypotheses_m,
                self.config.pnp_adaptive_depth_hypothesis_m,
                self.config.pnp_adaptive_depth_min_primary_inlier_ratio,
            );
            let pair_corrs = stereo_pair_correspondences(prev_stereo, &stereo, &temporal_matches);
            let (
                mut relative_pose,
                inlier_count,
                source,
                pnp_mean_reprojection_error_px,
                kabsch_mean_residual_m,
            ) = match self.config.relative_pose_mode {
                StereoRelativePoseMode::PnpThenKabsch => {
                    if let Some(report) = &pnp_report {
                        let kabsch_challenger =
                            if should_try_kabsch_challenger(report, &self.config) {
                                estimate_relative_pose_kabsch_ransac(
                                    &pair_corrs,
                                    &self.config.kabsch,
                                )
                            } else {
                                None
                            };
                        if let Some(kabsch_report) = &kabsch_challenger {
                            let pnp_stereo_score = score_pose_on_stereo_pairs(
                                &report.pose,
                                &pair_corrs,
                                self.config.kabsch.inlier_threshold_m,
                            );
                            if is_kabsch_challenger_better(
                                kabsch_report,
                                pnp_stereo_score,
                                &self.config,
                            ) {
                                (
                                    kabsch_report.relative_pose.clone(),
                                    kabsch_report.inliers.len(),
                                    StereoRelativePoseSource::Kabsch,
                                    Some(report.mean_reprojection_error),
                                    Some(kabsch_report.mean_residual_m),
                                )
                            } else {
                                (
                                    report.pose.clone(),
                                    report.inlier_count,
                                    StereoRelativePoseSource::Pnp,
                                    Some(report.mean_reprojection_error),
                                    kabsch_challenger.as_ref().map(|r| r.mean_residual_m),
                                )
                            }
                        } else {
                            (
                                report.pose.clone(),
                                report.inlier_count,
                                StereoRelativePoseSource::Pnp,
                                Some(report.mean_reprojection_error),
                                None,
                            )
                        }
                    } else {
                        let kabsch_report =
                            estimate_relative_pose_kabsch_ransac(&pair_corrs, &self.config.kabsch);
                        let report =
                            kabsch_report
                                .as_ref()
                                .ok_or_else(|| StereoVoError::KabschFailed {
                                    pair_index: self.poses.len() - 1,
                                    correspondence_count: pair_corrs.len().max(pnp_corrs_all.len()),
                                    min_inliers: self.config.kabsch.min_inliers,
                                })?;
                        (
                            report.relative_pose.clone(),
                            report.inliers.len(),
                            StereoRelativePoseSource::KabschFallback,
                            pnp_report.as_ref().map(|r| r.mean_reprojection_error),
                            Some(report.mean_residual_m),
                        )
                    }
                }
                StereoRelativePoseMode::KabschThenPnp => {
                    let kabsch_report =
                        estimate_relative_pose_kabsch_ransac(&pair_corrs, &self.config.kabsch);
                    if let Some(report) = &kabsch_report {
                        (
                            report.relative_pose.clone(),
                            report.inliers.len(),
                            StereoRelativePoseSource::Kabsch,
                            pnp_report.as_ref().map(|r| r.mean_reprojection_error),
                            Some(report.mean_residual_m),
                        )
                    } else {
                        let report =
                            pnp_report
                                .as_ref()
                                .ok_or_else(|| StereoVoError::KabschFailed {
                                    pair_index: self.poses.len() - 1,
                                    correspondence_count: pair_corrs.len().max(pnp_corrs_all.len()),
                                    min_inliers: self.config.kabsch.min_inliers,
                                })?;
                        (
                            report.pose.clone(),
                            report.inlier_count,
                            StereoRelativePoseSource::PnpFallback,
                            Some(report.mean_reprojection_error),
                            kabsch_report.as_ref().map(|r| r.mean_residual_m),
                        )
                    }
                }
            };
            if should_refine_current_stereo_pose(
                &self.per_pair_translation_m,
                pnp_report.as_ref(),
                source,
                &self.config,
            ) {
                if let Some(refined) = refine_relative_pose_with_current_stereo(
                    &relative_pose,
                    prev_stereo,
                    &stereo,
                    &left_features,
                    &right_features,
                    &temporal_matches,
                    &self.camera,
                    self.baseline,
                    self.config.pnp_reprojection_threshold_px * 2.0,
                ) {
                    relative_pose = refined;
                }
            }
            let raw_translation_m = relative_pose.world_to_camera.translation.norm();
            let raw_rotation_deg = relative_rotation_angle_deg(&relative_pose);
            let stereo_vertical_aligned = align_vertical_translation_to_stereo_pairs(
                &mut relative_pose,
                prev_stereo,
                &stereo,
                prev_left,
                &left_features,
                &temporal_matches,
                &self.camera,
                &self.config,
            );
            let motion_scale_rescued = rescue_collapsed_motion_scale(
                &mut relative_pose,
                &self.per_pair_translation_m,
                pnp_report.as_ref(),
                source,
                &self.config,
            );
            let translation_direction_rescued = rescue_translation_direction(
                &mut relative_pose,
                &self.per_pair_translation_m,
                &self.per_pair_translation_vectors,
                pnp_report.as_ref(),
                source,
                &self.config,
            );
            let rotation_spike_rescued = rescue_rotation_spike(
                &mut relative_pose,
                &self.per_pair_translation_m,
                &self.per_pair_rotation_deg,
                pnp_report.as_ref(),
                source,
                &self.config,
            );
            let rotation_vector_rescued = rescue_rotation_vector(
                &mut relative_pose,
                &self.per_pair_translation_m,
                &self.per_pair_rotation_vectors,
                pnp_report.as_ref(),
                source,
                &self.config,
            );
            self.kabsch_inlier_counts.push(inlier_count);
            let translation_m = relative_pose.world_to_camera.translation.norm();
            let rotation_deg = relative_rotation_angle_deg(&relative_pose);
            let rotation_vector = relative_pose.world_to_camera.rotation.scaled_axis();
            self.per_pair_translation_m.push(translation_m);
            self.per_pair_translation_vectors
                .push(relative_pose.world_to_camera.translation);
            self.per_pair_rotation_deg.push(rotation_deg);
            self.per_pair_rotation_vectors.push(rotation_vector);
            self.temporal_matches_per_pair
                .push(temporal_matches.clone());
            self.pair_diagnostics.push(StereoVoPairDiagnostics {
                from_frame: self.poses.len() - 1,
                to_frame: self.poses.len(),
                source,
                temporal_match_count: temporal_matches.len(),
                temporal_row_gate_px,
                temporal_confidence_gate,
                pnp_correspondence_count: pnp_report
                    .as_ref()
                    .map(|report| report.correspondence_count)
                    .unwrap_or_else(|| pnp_corrs_all.len()),
                stereo_pair_correspondence_count: pair_corrs.len(),
                inlier_count,
                raw_translation_m,
                raw_rotation_deg,
                translation_m,
                rotation_deg,
                motion_scale_rescued,
                stereo_vertical_aligned,
                translation_direction_rescued,
                rotation_spike_rescued,
                rotation_vector_rescued,
                pnp_mean_reprojection_error_px,
                kabsch_mean_residual_m,
            });
            let relative = relative_pose.world_to_camera;
            let last = self.poses.last().unwrap();
            Pose {
                world_to_camera: relative.compose(&last.world_to_camera),
            }
        } else {
            Pose::identity()
        };

        self.poses.push(new_pose.clone());
        self.left_features.push(left_features);
        self.right_features.push(right_features);
        self.stereo_per_frame.push(stereo);
        Ok(new_pose)
    }

    /// Adaptive projection-track search radius in pixels: scales with the
    /// observed median per-pair translation, with a 20 px floor that
    /// covers KITTI's patch-descriptor confusion band even when motion is
    /// small. Returns 25 px when no pairs have been processed yet.
    pub fn adaptive_track_search_radius_px(&self) -> f64 {
        if self.per_pair_translation_m.is_empty() {
            return 25.0;
        }
        let mut sorted = self.per_pair_translation_m.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        (12.0 + 2.0 * median).clamp(20.0, 35.0)
    }

    /// Sum of consecutive camera-center deltas in the recovered trajectory
    /// (in metric units).
    pub fn trajectory_length_m(&self) -> f64 {
        let centers: Vec<_> = self.poses.iter().map(|p| p.camera_center_world()).collect();
        centers.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
    }
}

fn rescue_collapsed_motion_scale(
    relative_pose: &mut Pose,
    translation_history_m: &[f64],
    pnp_report: Option<&PnpSelectedReport>,
    source: StereoRelativePoseSource,
    config: &StereoVoFrontendConfig,
) -> bool {
    if translation_history_m.len() < config.motion_scale_rescue_min_history {
        return false;
    }
    let Some(recent_median) = recent_translation_median(
        translation_history_m,
        config.motion_scale_rescue_min_history,
    ) else {
        return false;
    };
    if recent_median < config.motion_scale_rescue_min_median_translation_m {
        return false;
    }
    let current = relative_pose.world_to_camera.translation.norm();
    if !current.is_finite() || current <= 1.0e-9 {
        return false;
    }
    let collapsed = current < recent_median * config.motion_scale_rescue_min_translation_ratio;
    let inflated = current > recent_median * config.motion_scale_rescue_max_translation_ratio;
    if !collapsed && !inflated {
        return false;
    }
    if !is_weak_motion_scale_consensus(
        pnp_report,
        source,
        config.motion_scale_rescue_max_pnp_inlier_ratio,
    ) {
        return false;
    }
    let target_scale = recent_positive_percentile(
        translation_history_m,
        config.motion_scale_rescue_min_history,
        config.motion_scale_rescue_target_percentile,
    )
    .filter(|target| target.is_finite() && *target > 0.0)
    .unwrap_or(recent_median);
    relative_pose.world_to_camera.translation *= target_scale / current;
    true
}

fn rescue_rotation_spike(
    relative_pose: &mut Pose,
    translation_history_m: &[f64],
    rotation_history_deg: &[f64],
    pnp_report: Option<&PnpSelectedReport>,
    source: StereoRelativePoseSource,
    config: &StereoVoFrontendConfig,
) -> bool {
    if translation_history_m.len() < config.rotation_spike_rescue_min_history
        || rotation_history_deg.len() < config.rotation_spike_rescue_min_history
    {
        return false;
    }
    let Some(recent_translation_median) = recent_translation_median(
        translation_history_m,
        config.rotation_spike_rescue_min_history,
    ) else {
        return false;
    };
    if recent_translation_median < config.rotation_spike_rescue_min_median_translation_m {
        return false;
    }
    if !is_weak_motion_scale_consensus(
        pnp_report,
        source,
        config.rotation_spike_rescue_max_pnp_inlier_ratio,
    ) {
        return false;
    }
    let Some(recent_rotation_median) = recent_positive_median(
        rotation_history_deg,
        config.rotation_spike_rescue_min_history,
    ) else {
        return false;
    };
    let current_deg = relative_rotation_angle_deg(relative_pose);
    if !current_deg.is_finite()
        || current_deg < config.rotation_spike_rescue_min_angle_deg
        || current_deg < recent_rotation_median * config.rotation_spike_rescue_max_angle_ratio
    {
        return false;
    }
    let scaled_axis = relative_pose.world_to_camera.rotation.scaled_axis();
    let current_rad = scaled_axis.norm();
    if !current_rad.is_finite() || current_rad <= 1.0e-12 {
        return false;
    }
    let target_deg = recent_rotation_median.max(1.0e-6);
    let target_axis = scaled_axis * (target_deg.to_radians() / current_rad);
    relative_pose.world_to_camera.rotation = UnitQuaternion::from_scaled_axis(target_axis);
    true
}

fn rescue_rotation_vector(
    relative_pose: &mut Pose,
    translation_history_m: &[f64],
    rotation_vector_history: &[Vector3<f64>],
    pnp_report: Option<&PnpSelectedReport>,
    source: StereoRelativePoseSource,
    config: &StereoVoFrontendConfig,
) -> bool {
    if translation_history_m.len() < config.rotation_vector_rescue_min_history
        || rotation_vector_history.len() < config.rotation_vector_rescue_min_history
    {
        return false;
    }
    let Some(recent_translation_median) = recent_translation_median(
        translation_history_m,
        config.rotation_vector_rescue_min_history,
    ) else {
        return false;
    };
    if recent_translation_median < config.rotation_vector_rescue_min_median_translation_m {
        return false;
    }
    if !is_weak_motion_scale_consensus(
        pnp_report,
        source,
        config.rotation_vector_rescue_max_pnp_inlier_ratio,
    ) {
        return false;
    }
    let Some(recent_vector) = recent_rotation_vector(
        rotation_vector_history,
        config.rotation_vector_rescue_min_history,
    ) else {
        return false;
    };
    let current_vector = relative_pose.world_to_camera.rotation.scaled_axis();
    if !current_vector.iter().all(|v| v.is_finite()) {
        return false;
    }
    let delta_deg = (current_vector - recent_vector).norm().to_degrees();
    if !delta_deg.is_finite() || delta_deg <= config.rotation_vector_rescue_max_delta_deg {
        return false;
    }
    relative_pose.world_to_camera.rotation = UnitQuaternion::from_scaled_axis(recent_vector);
    true
}

fn rescue_translation_direction(
    relative_pose: &mut Pose,
    translation_history_m: &[f64],
    translation_vector_history: &[Vector3<f64>],
    pnp_report: Option<&PnpSelectedReport>,
    source: StereoRelativePoseSource,
    config: &StereoVoFrontendConfig,
) -> bool {
    if translation_history_m.len() < config.translation_direction_rescue_min_history
        || translation_vector_history.len() < config.translation_direction_rescue_min_history
    {
        return false;
    }
    let Some(recent_translation_median) = recent_translation_median(
        translation_history_m,
        config.translation_direction_rescue_min_history,
    ) else {
        return false;
    };
    if recent_translation_median < config.translation_direction_rescue_min_median_translation_m {
        return false;
    }
    if !is_weak_motion_scale_consensus(
        pnp_report,
        source,
        config.translation_direction_rescue_max_pnp_inlier_ratio,
    ) {
        return false;
    }
    let current = relative_pose.world_to_camera.translation;
    let current_norm = current.norm();
    if !current_norm.is_finite() || current_norm <= 1.0e-9 {
        return false;
    }
    let Some(recent_direction) = recent_translation_direction(
        translation_vector_history,
        config.translation_direction_rescue_min_history,
    ) else {
        return false;
    };
    let current_direction = current / current_norm;
    let cos_angle = current_direction.dot(&recent_direction).clamp(-1.0, 1.0);
    let angle_deg = cos_angle.acos().to_degrees();
    if !angle_deg.is_finite() || angle_deg <= config.translation_direction_rescue_max_angle_deg {
        return false;
    }
    relative_pose.world_to_camera.translation = recent_direction * current_norm;
    true
}

fn align_vertical_translation_to_stereo_pairs(
    relative_pose: &mut Pose,
    prev_stereo: &[StereoFeature],
    current_stereo: &[StereoFeature],
    prev_left: &FeatureSet,
    current_left: &FeatureSet,
    temporal_matches: &[DescriptorMatch],
    camera: &Camera,
    config: &StereoVoFrontendConfig,
) -> bool {
    if !config.stereo_vertical_alignment
        || temporal_matches.len() < config.stereo_vertical_alignment_min_pairs
        || !config
            .stereo_vertical_alignment_max_correction_m
            .is_finite()
        || config.stereo_vertical_alignment_max_correction_m <= 0.0
    {
        return false;
    }
    let rotation = relative_pose.world_to_camera.rotation;
    let current_ty = relative_pose.world_to_camera.translation.y;
    let prev_lookup: HashMap<usize, &StereoFeature> =
        prev_stereo.iter().map(|f| (f.left_index, f)).collect();
    let current_lookup: HashMap<usize, &StereoFeature> =
        current_stereo.iter().map(|f| (f.left_index, f)).collect();
    let lower_row = camera.height as f64 * 0.55;
    let mut residuals = Vec::with_capacity(temporal_matches.len());
    for m in temporal_matches {
        let (Some(prev), Some(curr)) = (
            prev_lookup.get(&m.query_index),
            current_lookup.get(&m.train_index),
        ) else {
            continue;
        };
        let (Some(prev_kp), Some(curr_kp)) = (
            prev_left.keypoints.get(m.query_index),
            current_left.keypoints.get(m.train_index),
        ) else {
            continue;
        };
        if prev_kp.y < lower_row || curr_kp.y < lower_row {
            continue;
        }
        if prev.point_cam.z > 45.0
            || curr.point_cam.z > 45.0
            || prev.point_cam.y < -0.5
            || curr.point_cam.y < -0.5
            || !prev.point_cam.coords.iter().all(|v| v.is_finite())
            || !curr.point_cam.coords.iter().all(|v| v.is_finite())
        {
            continue;
        }
        let rotated = rotation.transform_point(&prev.point_cam);
        let residual_y = curr.point_cam.y - (rotated.y + current_ty);
        if residual_y.is_finite() {
            residuals.push(residual_y);
        }
    }
    if residuals.len() < config.stereo_vertical_alignment_min_pairs {
        return false;
    }
    align_vertical_translation_from_residuals(
        relative_pose,
        residuals,
        config.stereo_vertical_alignment_max_correction_m,
    )
}

fn align_vertical_translation_from_residuals(
    relative_pose: &mut Pose,
    mut residuals: Vec<f64>,
    max_correction_m: f64,
) -> bool {
    residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let correction = residuals[residuals.len() / 2].clamp(-max_correction_m, max_correction_m);
    if correction.abs() <= 1.0e-6 {
        return false;
    }
    relative_pose.world_to_camera.translation.y += correction;
    true
}

fn should_refine_current_stereo_pose(
    translation_history_m: &[f64],
    pnp_report: Option<&PnpSelectedReport>,
    source: StereoRelativePoseSource,
    config: &StereoVoFrontendConfig,
) -> bool {
    if config.stereo_pose_refinement {
        return true;
    }
    if translation_history_m.len() < config.stereo_pose_refinement_auto_min_history {
        return false;
    }
    let Some(recent_median) = recent_translation_median(
        translation_history_m,
        config.stereo_pose_refinement_auto_min_history,
    ) else {
        return false;
    };
    if recent_median < config.stereo_pose_refinement_auto_min_median_translation_m {
        return false;
    }
    is_weak_motion_scale_consensus(
        pnp_report,
        source,
        config.stereo_pose_refinement_auto_max_pnp_inlier_ratio,
    )
}

fn recent_translation_median(history: &[f64], window: usize) -> Option<f64> {
    recent_positive_median(history, window)
}

fn recent_positive_median(history: &[f64], window: usize) -> Option<f64> {
    recent_positive_percentile(history, window, 0.5)
}

fn recent_positive_percentile(history: &[f64], window: usize, percentile: f64) -> Option<f64> {
    if history.is_empty() || window == 0 {
        return None;
    }
    let start = history.len().saturating_sub(window);
    let mut values = history[start..]
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v > 0.0)
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let clamped = if percentile.is_finite() {
        percentile.clamp(0.0, 1.0)
    } else {
        0.5
    };
    let index = ((values.len() - 1) as f64 * clamped).round() as usize;
    Some(values[index])
}

fn relative_rotation_angle_deg(relative_pose: &Pose) -> f64 {
    relative_pose.world_to_camera.rotation.angle().to_degrees()
}

fn recent_translation_direction(history: &[Vector3<f64>], window: usize) -> Option<Vector3<f64>> {
    if history.is_empty() || window == 0 {
        return None;
    }
    let start = history.len().saturating_sub(window);
    let mut sum = Vector3::zeros();
    let mut count = 0usize;
    for t in &history[start..] {
        let norm = t.norm();
        if norm.is_finite() && norm > 1.0e-9 {
            sum += *t / norm;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let norm = sum.norm();
    if !norm.is_finite() || norm <= 1.0e-9 {
        return None;
    }
    Some(sum / norm)
}

fn recent_rotation_vector(history: &[Vector3<f64>], window: usize) -> Option<Vector3<f64>> {
    if history.is_empty() || window == 0 {
        return None;
    }
    let start = history.len().saturating_sub(window);
    let mut sum = Vector3::zeros();
    let mut count = 0usize;
    for v in &history[start..] {
        if v.iter().all(|x| x.is_finite()) {
            sum += *v;
            count += 1;
        }
    }
    if count == 0 {
        return None;
    }
    let mean = sum / count as f64;
    if mean.iter().all(|v| v.is_finite()) {
        Some(mean)
    } else {
        None
    }
}

fn is_weak_motion_scale_consensus(
    pnp_report: Option<&PnpSelectedReport>,
    source: StereoRelativePoseSource,
    max_pnp_inlier_ratio: f64,
) -> bool {
    if source != StereoRelativePoseSource::Pnp {
        return true;
    }
    let Some(report) = pnp_report else {
        return true;
    };
    if report.correspondence_count == 0 {
        return true;
    }
    let ratio = report.inlier_count as f64 / report.correspondence_count as f64;
    ratio < max_pnp_inlier_ratio
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::FeatureSet;
    use crate::matching::{BruteForceMatcher, DescriptorMatch, Matcher};
    use nalgebra::{Point2, UnitQuaternion, Vector3};
    use visloc_core::geometry::{Pose, SE3};

    fn camera() -> Camera {
        Camera::pinhole(1, 1241, 376, 718.856, 718.856, 607.193, 185.216)
    }

    #[test]
    fn default_rescue_config_keeps_adopted_guardrails() {
        let config = StereoVoFrontendConfig::default();
        assert_eq!(config.motion_scale_rescue_min_history, 20);
        assert_eq!(config.motion_scale_rescue_min_median_translation_m, 1.5);
        assert_eq!(config.motion_scale_rescue_min_translation_ratio, 0.97);
        assert_eq!(config.motion_scale_rescue_max_translation_ratio, 1.6);
        assert_eq!(config.motion_scale_rescue_target_percentile, 0.75);
        assert_eq!(config.motion_scale_rescue_max_pnp_inlier_ratio, 1.05);
        assert_eq!(config.translation_direction_rescue_min_history, 20);
        assert_eq!(
            config.translation_direction_rescue_min_median_translation_m,
            1.5
        );
        assert_eq!(config.translation_direction_rescue_max_angle_deg, 10.0);
        assert_eq!(
            config.translation_direction_rescue_max_pnp_inlier_ratio,
            0.45
        );
        assert_eq!(config.rotation_spike_rescue_min_history, 20);
        assert_eq!(config.rotation_spike_rescue_min_angle_deg, 1.0);
        assert_eq!(config.rotation_spike_rescue_max_angle_ratio, 3.0);
        assert_eq!(config.rotation_vector_rescue_min_history, 20);
        assert_eq!(config.rotation_vector_rescue_max_delta_deg, 0.4);
        assert_eq!(config.rotation_vector_rescue_max_pnp_inlier_ratio, 0.45);
        assert_eq!(config.pnp_kabsch_challenge_max_inlier_ratio, 0.0);
        assert_eq!(config.pnp_kabsch_challenge_min_3d_inlier_gain, 8);
        assert_eq!(config.pnp_kabsch_challenge_max_residual_ratio, 0.85);
        assert!(!config.stereo_vertical_alignment);
        assert_eq!(config.stereo_vertical_alignment_min_pairs, 80);
        assert_eq!(config.stereo_vertical_alignment_max_correction_m, 0.25);
        assert_eq!(config.stereo_pose_refinement_auto_min_history, 20);
        assert_eq!(config.stereo_pose_refinement_auto_max_pnp_inlier_ratio, 1.0);
        assert_eq!(config.temporal_max_row_delta_px, None);
        assert_eq!(config.temporal_auto_max_row_delta_px, None);
        assert_eq!(config.temporal_auto_min_history, 20);
        assert_eq!(config.temporal_auto_min_median_translation_m, 1.05);
        assert_eq!(config.temporal_auto_min_confidence, Some(0.20));
        assert_eq!(config.temporal_auto_confidence_min_history, 20);
        assert_eq!(
            config.temporal_auto_confidence_min_median_translation_m,
            1.45
        );
        assert_eq!(
            config.temporal_auto_confidence_curve_min_median_translation_m,
            Some(0.95)
        );
        assert_eq!(
            config.temporal_auto_confidence_curve_min_median_rotation_deg,
            Some(0.26)
        );
        assert_eq!(
            config.temporal_auto_confidence_max_median_rotation_deg,
            Some(0.45)
        );
    }

    #[test]
    fn temporal_row_gate_prefers_explicit_override() {
        let mut config = StereoVoFrontendConfig::default();
        config.temporal_max_row_delta_px = Some(4.0);
        config.temporal_auto_max_row_delta_px = Some(8.0);
        config.temporal_auto_min_history = 3;
        config.temporal_auto_min_median_translation_m = 10.0;
        assert_eq!(
            effective_temporal_max_row_delta_px(&[0.1, 0.2, 0.3], &config),
            Some(4.0)
        );
    }

    #[test]
    fn temporal_row_gate_activates_only_for_fast_recent_motion() {
        let mut config = StereoVoFrontendConfig::default();
        config.temporal_max_row_delta_px = None;
        config.temporal_auto_max_row_delta_px = Some(8.0);
        config.temporal_auto_min_history = 3;
        config.temporal_auto_min_median_translation_m = 0.9;

        assert_eq!(
            effective_temporal_max_row_delta_px(&[0.8, 0.85], &config),
            None
        );
        assert_eq!(
            effective_temporal_max_row_delta_px(&[0.7, 0.8, 0.85], &config),
            None
        );
        assert_eq!(
            effective_temporal_max_row_delta_px(&[0.7, 0.95, 1.1], &config),
            Some(8.0)
        );
    }

    #[test]
    fn temporal_confidence_gate_activates_only_for_fast_recent_motion() {
        let mut config = StereoVoFrontendConfig::default();
        config.temporal_auto_min_confidence = Some(0.20);
        config.temporal_auto_confidence_min_history = 3;
        config.temporal_auto_confidence_min_median_translation_m = 0.8;

        assert_eq!(
            effective_temporal_min_confidence(&[0.7, 0.75], &[0.2, 0.2], &config),
            None
        );
        assert_eq!(
            effective_temporal_min_confidence(&[0.65, 0.7, 0.75], &[0.2, 0.2, 0.2], &config),
            None
        );
        assert_eq!(
            effective_temporal_min_confidence(&[0.7, 0.85, 0.9], &[0.2, 0.2, 0.2], &config),
            Some(0.20)
        );
        assert_eq!(
            effective_temporal_min_confidence(&[0.7, 0.85, 0.9], &[0.6, 0.7, 0.8], &config),
            None
        );
    }

    #[test]
    fn temporal_confidence_gate_activates_for_medium_curved_motion() {
        let mut config = StereoVoFrontendConfig::default();
        config.temporal_auto_min_confidence = Some(0.20);
        config.temporal_auto_confidence_min_history = 3;
        config.temporal_auto_confidence_min_median_translation_m = 1.5;
        config.temporal_auto_confidence_curve_min_median_translation_m = Some(0.9);
        config.temporal_auto_confidence_curve_min_median_rotation_deg = Some(0.18);
        config.temporal_auto_confidence_max_median_rotation_deg = Some(0.45);

        assert_eq!(
            effective_temporal_min_confidence(&[0.8, 0.95, 1.0], &[0.15, 0.16, 0.17], &config),
            None
        );
        assert_eq!(
            effective_temporal_min_confidence(&[0.8, 0.95, 1.0], &[0.18, 0.22, 0.24], &config),
            Some(0.20)
        );
        assert_eq!(
            effective_temporal_min_confidence(&[0.8, 0.95, 1.0], &[0.4, 0.5, 0.6], &config),
            None
        );
    }

    #[test]
    fn temporal_confidence_filter_preserves_matchers_without_confidence() {
        let matches = vec![
            DescriptorMatch {
                query_index: 0,
                train_index: 1,
                distance: 0.1,
                second_best_distance: None,
                ratio: None,
                confidence: None,
            },
            DescriptorMatch {
                query_index: 2,
                train_index: 3,
                distance: 0.2,
                second_best_distance: None,
                ratio: None,
                confidence: None,
            },
        ];

        assert_eq!(
            filter_temporal_matches_by_confidence(&matches, 0.20),
            matches
        );
    }

    #[test]
    fn temporal_confidence_filter_drops_low_confidence_deep_matches() {
        let matches = vec![
            DescriptorMatch {
                query_index: 0,
                train_index: 1,
                distance: 0.1,
                second_best_distance: None,
                ratio: None,
                confidence: Some(0.19),
            },
            DescriptorMatch {
                query_index: 2,
                train_index: 3,
                distance: 0.2,
                second_best_distance: None,
                ratio: None,
                confidence: Some(0.20),
            },
        ];

        let filtered = filter_temporal_matches_by_confidence(&matches, 0.20);

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].query_index, 2);
    }

    #[test]
    fn stereo_vertical_alignment_uses_median_y_residual_only() {
        let mut config = StereoVoFrontendConfig::default();
        config.stereo_vertical_alignment = true;
        config.stereo_vertical_alignment_min_pairs = 3;
        config.stereo_vertical_alignment_max_correction_m = 0.25;
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.5, 0.0, -1.0)),
        };
        assert!(align_vertical_translation_from_residuals(
            &mut pose,
            vec![0.2, 0.2, 0.2],
            config.stereo_vertical_alignment_max_correction_m,
        ));
        assert!((pose.world_to_camera.translation.x - 0.5).abs() < 1.0e-12);
        assert!((pose.world_to_camera.translation.y - 0.2).abs() < 1.0e-12);
        assert!((pose.world_to_camera.translation.z + 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn stereo_vertical_alignment_clamps_large_corrections() {
        let mut config = StereoVoFrontendConfig::default();
        config.stereo_vertical_alignment = true;
        config.stereo_vertical_alignment_min_pairs = 3;
        config.stereo_vertical_alignment_max_correction_m = 0.25;
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::zeros()),
        };
        assert!(align_vertical_translation_from_residuals(
            &mut pose,
            vec![1.0, 1.0, 1.0],
            config.stereo_vertical_alignment_max_correction_m,
        ));
        assert!((pose.world_to_camera.translation.y - 0.25).abs() < 1.0e-12);
    }

    #[test]
    fn weak_pnp_triggers_kabsch_challenger_only_below_ratio_gate() {
        let mut config = StereoVoFrontendConfig::default();
        config.pnp_kabsch_challenge_max_inlier_ratio = 0.20;
        let weak = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 19,
            correspondence_count: 100,
            mean_reprojection_error: 1.0,
        };
        let strong = PnpSelectedReport {
            inlier_count: 20,
            ..weak.clone()
        };

        assert!(should_try_kabsch_challenger(&weak, &config));
        assert!(!should_try_kabsch_challenger(&strong, &config));
    }

    #[test]
    fn kabsch_challenger_needs_inlier_gain_and_residual_gain() {
        let config = StereoVoFrontendConfig::default();
        let pnp_score = Pose3d3dScore {
            inlier_count: 20,
            mean_residual_m: 1.0,
        };
        let better = KabschRansacReport {
            relative_pose: Pose::identity(),
            inliers: (0..30).collect(),
            mean_residual_m: 0.7,
        };
        let too_few_extra_inliers = KabschRansacReport {
            inliers: (0..27).collect(),
            ..better.clone()
        };
        let too_noisy = KabschRansacReport {
            mean_residual_m: 0.9,
            ..better.clone()
        };

        assert!(is_kabsch_challenger_better(&better, pnp_score, &config));
        assert!(!is_kabsch_challenger_better(
            &too_few_extra_inliers,
            pnp_score,
            &config
        ));
        assert!(!is_kabsch_challenger_better(&too_noisy, pnp_score, &config));
    }

    /// Project a world-frame point through `pose` and the camera. Used by
    /// the round-trip test below.
    fn project(camera: &Camera, pose: &Pose, x_world: &Point3<f64>) -> Point2<f64> {
        let xc = pose.transform_world_point(x_world);
        camera.project(&xc).expect("point in front of camera")
    }

    /// Build a synthetic FeatureSet at the given pixel locations. Each
    /// keypoint gets a 1-D descriptor encoding its `landmark id`, so a
    /// brute-force matcher can recover the unique correspondence.
    fn synthetic_feature_set(keypoints: &[(usize, Point2<f64>)]) -> FeatureSet {
        let kps: Vec<Point2<f64>> = keypoints.iter().map(|(_, p)| *p).collect();
        let descriptors: Vec<Vec<f32>> = keypoints.iter().map(|(id, _)| vec![*id as f32]).collect();
        FeatureSet::new(kps, descriptors).expect("synthetic FeatureSet")
    }

    fn descriptor_match(query_index: usize, train_index: usize) -> DescriptorMatch {
        DescriptorMatch {
            query_index,
            train_index,
            distance: 0.0,
            second_best_distance: None,
            ratio: None,
            confidence: Some(0.95),
        }
    }

    #[test]
    fn triangulate_stereo_features_round_trips_metric_points() {
        let camera = camera();
        let baseline = 0.537150888;
        let (fx, _, _, _) = camera.intrinsics().unwrap();
        // Choose a few world-frame points in front of the camera.
        let truth: Vec<(usize, Point3<f64>)> = vec![
            (0, Point3::new(-1.5, 0.2, 12.0)),
            (1, Point3::new(0.5, -0.4, 8.0)),
            (2, Point3::new(2.0, 0.0, 25.0)),
            (3, Point3::new(-0.5, 0.5, 5.0)),
        ];
        let pose = Pose::identity();
        let mut left_kps: Vec<(usize, Point2<f64>)> = Vec::new();
        let mut right_kps: Vec<(usize, Point2<f64>)> = Vec::new();
        for (id, p) in &truth {
            let l = project(&camera, &pose, p);
            // Right pixel: apply `u_r = u_l − fx · b / Z`, same row.
            let right = Point2::new(l.x - fx * baseline / p.z, l.y);
            left_kps.push((*id, l));
            right_kps.push((*id, right));
        }
        let left = synthetic_feature_set(&left_kps);
        let right = synthetic_feature_set(&right_kps);
        let stereo = triangulate_stereo_features(
            &left,
            &right,
            &camera,
            baseline,
            &StereoFeatureConfig {
                ratio: None,
                ..StereoFeatureConfig::default()
            },
        );
        assert_eq!(stereo.len(), truth.len());
        for f in &stereo {
            // The "id" field is the same for left and right since we made
            // the descriptor encode it; verify the matcher actually paired
            // by id and that the recovered point matches truth.
            let id = left.descriptors[f.left_index][0] as usize;
            let expected = truth.iter().find(|(i, _)| *i == id).unwrap().1;
            let err = (f.point_cam - expected).norm();
            assert!(err < 1.0e-9, "stereo triangulation error {err} for id {id}");
        }
    }

    #[test]
    fn triangulate_stereo_features_rejects_row_mismatched_pairs() {
        let camera = camera();
        let baseline = 0.5;
        // Row mismatch by 5 px should be rejected (default max 1.5 px).
        let left = synthetic_feature_set(&[(0, Point2::new(700.0, 200.0))]);
        let right = synthetic_feature_set(&[(0, Point2::new(680.0, 205.0))]);
        let stereo = triangulate_stereo_features(
            &left,
            &right,
            &camera,
            baseline,
            &StereoFeatureConfig {
                ratio: None,
                ..StereoFeatureConfig::default()
            },
        );
        assert!(stereo.is_empty(), "row-mismatched pair must be rejected");
    }

    #[test]
    fn triangulate_stereo_features_rejects_far_field_points() {
        let camera = camera();
        let baseline = 0.5;
        // Disparity = 0.4 < default 1.0 px threshold.
        let left = synthetic_feature_set(&[(0, Point2::new(700.0, 200.0))]);
        let right = synthetic_feature_set(&[(0, Point2::new(699.6, 200.0))]);
        let stereo = triangulate_stereo_features(
            &left,
            &right,
            &camera,
            baseline,
            &StereoFeatureConfig {
                ratio: None,
                ..StereoFeatureConfig::default()
            },
        );
        assert!(
            stereo.is_empty(),
            "sub-threshold disparity must be rejected"
        );
    }

    #[test]
    fn triangulate_stereo_feature_matches_uses_explicit_pairs() {
        let camera = camera();
        let baseline = 0.5;
        let left = FeatureSet::new(vec![Point2::new(700.0, 200.0)], vec![vec![0.0]]).unwrap();
        let right = FeatureSet::new(
            vec![Point2::new(699.0, 210.0), Point2::new(670.0, 200.0)],
            vec![vec![0.0], vec![0.0]],
        )
        .unwrap();
        let matches = vec![descriptor_match(0, 1), descriptor_match(0, 0)];

        let stereo = triangulate_stereo_feature_matches(
            &left,
            &right,
            &matches,
            &camera,
            baseline,
            &StereoFeatureConfig {
                ratio: None,
                ..StereoFeatureConfig::default()
            },
        );

        assert_eq!(stereo.len(), 1);
        assert_eq!(stereo[0].left_index, 0);
        assert_eq!(stereo[0].right_index, 1);
        assert!((stereo[0].disparity - 30.0).abs() < 1.0e-9);
    }

    #[test]
    fn build_stereo_temporal_correspondences_links_3d_with_b_pixels() {
        // Three triangulated stereo features in frame A, and a temporal
        // matcher that pairs each one with a known frame-B pixel.
        let a_features = vec![
            StereoFeature {
                left_index: 5,
                right_index: 6,
                disparity: 10.0,
                point_cam: Point3::new(0.0, 0.0, 5.0),
            },
            StereoFeature {
                left_index: 7,
                right_index: 8,
                disparity: 5.0,
                point_cam: Point3::new(1.0, 0.5, 10.0),
            },
            StereoFeature {
                left_index: 9,
                right_index: 10,
                disparity: 4.0,
                point_cam: Point3::new(-2.0, 0.5, 15.0),
            },
        ];
        let b_left = synthetic_feature_set(&[
            (0, Point2::new(100.0, 100.0)),
            (1, Point2::new(200.0, 200.0)),
            (2, Point2::new(300.0, 300.0)),
        ]);
        let temporal_matches = vec![
            DescriptorMatch {
                query_index: 5,
                train_index: 0,
                distance: 0.1,
                second_best_distance: None,
                ratio: None,
                confidence: None,
            },
            DescriptorMatch {
                query_index: 7,
                train_index: 1,
                distance: 0.1,
                second_best_distance: None,
                ratio: None,
                confidence: None,
            },
            // This match's query index has no triangulation in A → drop.
            DescriptorMatch {
                query_index: 99,
                train_index: 2,
                distance: 0.1,
                second_best_distance: None,
                ratio: None,
                confidence: None,
            },
        ];
        let pairs = build_stereo_temporal_correspondences(&a_features, &b_left, &temporal_matches);
        assert_eq!(pairs.len(), 2);
        assert_eq!(pairs[0].point2d, Point2::new(100.0, 100.0));
        assert_eq!(pairs[0].point3d, Point3::new(0.0, 0.0, 5.0));
        assert_eq!(pairs[1].point2d, Point2::new(200.0, 200.0));
        assert_eq!(pairs[1].point3d, Point3::new(1.0, 0.5, 10.0));
    }

    #[test]
    fn adaptive_pnp_depth_triggers_only_on_weak_primary_consensus() {
        assert!(!should_try_adaptive_pnp_depth(
            Some(&PnpFullScore {
                inlier_count: 65,
                mean_error: 1.0,
            }),
            100,
            0.65,
        ));
        assert!(should_try_adaptive_pnp_depth(
            Some(&PnpFullScore {
                inlier_count: 64,
                mean_error: 1.0,
            }),
            100,
            0.65,
        ));
        assert!(should_try_adaptive_pnp_depth(None, 100, 0.65));
        assert!(!should_try_adaptive_pnp_depth(None, 0, 0.65));
        assert!(!should_try_adaptive_pnp_depth(
            Some(&PnpFullScore {
                inlier_count: 1,
                mean_error: 1.0,
            }),
            100,
            f64::NAN,
        ));
    }

    #[test]
    fn motion_scale_rescue_requires_fast_history_and_weak_consensus() {
        let mut config = StereoVoFrontendConfig::default();
        config.motion_scale_rescue_min_history = 4;
        config.motion_scale_rescue_min_median_translation_m = 1.5;
        config.motion_scale_rescue_min_translation_ratio = 0.35;
        config.motion_scale_rescue_max_translation_ratio = 1.6;
        config.motion_scale_rescue_max_pnp_inlier_ratio = 0.25;

        let history = [2.2, 2.5, 2.7, 2.6];
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
        };
        assert!(rescue_collapsed_motion_scale(
            &mut pose,
            &history,
            None,
            StereoRelativePoseSource::KabschFallback,
            &config,
        ));
        assert!((pose.world_to_camera.translation.norm() - 2.6).abs() < 1.0e-9);

        let slow_history = [0.6, 0.7, 0.8, 0.7];
        let mut slow_pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
        };
        assert!(!rescue_collapsed_motion_scale(
            &mut slow_pose,
            &slow_history,
            None,
            StereoRelativePoseSource::KabschFallback,
            &config,
        ));

        let strong_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 80,
            correspondence_count: 100,
            mean_reprojection_error: 1.0,
        };
        let mut strong_pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 0.0)),
        };
        assert!(!rescue_collapsed_motion_scale(
            &mut strong_pose,
            &history,
            Some(&strong_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));

        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 20,
            correspondence_count: 100,
            mean_reprojection_error: 1.0,
        };
        let mut inflated_pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(5.0, 0.0, 0.0)),
        };
        assert!(rescue_collapsed_motion_scale(
            &mut inflated_pose,
            &history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!((inflated_pose.world_to_camera.translation.norm() - 2.6).abs() < 1.0e-9);
    }

    #[test]
    fn motion_scale_rescue_uses_configured_target_percentile() {
        let mut config = StereoVoFrontendConfig::default();
        config.motion_scale_rescue_min_history = 5;
        config.motion_scale_rescue_min_median_translation_m = 1.5;
        config.motion_scale_rescue_min_translation_ratio = 0.65;
        config.motion_scale_rescue_target_percentile = 0.75;

        let history = [2.0, 2.1, 2.2, 2.8, 3.0];
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(1.0, 0.0, 0.0)),
        };
        assert!(rescue_collapsed_motion_scale(
            &mut pose,
            &history,
            None,
            StereoRelativePoseSource::KabschFallback,
            &config,
        ));
        assert!((pose.world_to_camera.translation.norm() - 2.8).abs() < 1.0e-9);
    }

    #[test]
    fn stereo_pose_refinement_auto_requires_fast_history_and_weak_consensus() {
        let mut config = StereoVoFrontendConfig::default();
        config.stereo_pose_refinement = false;
        config.stereo_pose_refinement_auto_min_history = 4;
        config.stereo_pose_refinement_auto_min_median_translation_m = 1.5;
        config.stereo_pose_refinement_auto_max_pnp_inlier_ratio = 0.65;
        let history = [1.9, 2.0, 2.1, 2.2];
        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 12,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        assert!(should_refine_current_stereo_pose(
            &history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));

        let strong_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 32,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        assert!(!should_refine_current_stereo_pose(
            &history,
            Some(&strong_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!(!should_refine_current_stereo_pose(
            &[0.6, 0.7, 0.8, 0.9],
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!(should_refine_current_stereo_pose(
            &history,
            Some(&strong_report),
            StereoRelativePoseSource::KabschFallback,
            &config,
        ));

        config.stereo_pose_refinement = true;
        assert!(should_refine_current_stereo_pose(
            &[],
            Some(&strong_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
    }

    #[test]
    fn translation_direction_rescue_requires_fast_history_and_weak_consensus() {
        let mut config = StereoVoFrontendConfig::default();
        config.translation_direction_rescue_min_history = 4;
        config.translation_direction_rescue_min_median_translation_m = 1.5;
        config.translation_direction_rescue_max_angle_deg = 12.0;
        config.translation_direction_rescue_max_pnp_inlier_ratio = 0.45;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let direction_history = [
            Vector3::new(0.0, 0.0, 2.1),
            Vector3::new(0.02, 0.0, 2.2),
            Vector3::new(-0.01, 0.0, 2.3),
            Vector3::new(0.0, 0.0, 2.4),
        ];
        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 10,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(1.2, 0.0, 2.0)),
        };
        let original_norm = pose.world_to_camera.translation.norm();
        assert!(rescue_translation_direction(
            &mut pose,
            &translation_history,
            &direction_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!((pose.world_to_camera.translation.norm() - original_norm).abs() < 1.0e-9);
        assert!(pose.world_to_camera.translation.x.abs() < 0.02);
        assert!(pose.world_to_camera.translation.z > 0.0);

        let strong_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 30,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut strong_pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(1.2, 0.0, 2.0)),
        };
        assert!(!rescue_translation_direction(
            &mut strong_pose,
            &translation_history,
            &direction_history,
            Some(&strong_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));

        let mut slow_pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(1.2, 0.0, 2.0)),
        };
        assert!(!rescue_translation_direction(
            &mut slow_pose,
            &[0.7, 0.8, 0.9, 0.8],
            &direction_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));

        let mut aligned_pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.1, 0.0, 2.0)),
        };
        assert!(!rescue_translation_direction(
            &mut aligned_pose,
            &translation_history,
            &direction_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
    }

    #[test]
    fn translation_direction_rescue_requires_vector_history_window() {
        let mut config = StereoVoFrontendConfig::default();
        config.translation_direction_rescue_min_history = 4;
        config.translation_direction_rescue_min_median_translation_m = 1.5;
        config.translation_direction_rescue_max_angle_deg = 10.0;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let short_direction_history = [
            Vector3::new(0.0, 0.0, 2.1),
            Vector3::new(0.0, 0.0, 2.2),
            Vector3::new(0.0, 0.0, 2.3),
        ];
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(1.2, 0.0, 2.0)),
        };
        let before = pose.world_to_camera.translation;

        assert!(!rescue_translation_direction(
            &mut pose,
            &translation_history,
            &short_direction_history,
            None,
            StereoRelativePoseSource::KabschFallback,
            &config,
        ));
        assert_eq!(pose.world_to_camera.translation, before);
    }

    #[test]
    fn translation_direction_rescue_handles_kabsch_fallback_without_pnp_report() {
        let mut config = StereoVoFrontendConfig::default();
        config.translation_direction_rescue_min_history = 4;
        config.translation_direction_rescue_min_median_translation_m = 1.5;
        config.translation_direction_rescue_max_angle_deg = 10.0;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let direction_history = [
            Vector3::new(0.0, 0.0, 2.1),
            Vector3::new(0.0, 0.0, 2.2),
            Vector3::new(0.0, 0.0, 2.3),
            Vector3::new(0.0, 0.0, 2.4),
        ];
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(-1.5, 0.0, 1.5)),
        };
        let original_norm = pose.world_to_camera.translation.norm();

        assert!(rescue_translation_direction(
            &mut pose,
            &translation_history,
            &direction_history,
            None,
            StereoRelativePoseSource::KabschFallback,
            &config,
        ));
        assert!((pose.world_to_camera.translation.norm() - original_norm).abs() < 1.0e-9);
        assert!(pose.world_to_camera.translation.x.abs() < 1.0e-9);
        assert!((pose.world_to_camera.translation.z - original_norm).abs() < 1.0e-9);
    }

    #[test]
    fn translation_direction_rescue_clamps_backward_outlier_but_keeps_magnitude() {
        let mut config = StereoVoFrontendConfig::default();
        config.translation_direction_rescue_min_history = 4;
        config.translation_direction_rescue_min_median_translation_m = 1.5;
        config.translation_direction_rescue_max_angle_deg = 10.0;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let direction_history = [
            Vector3::new(0.0, 0.0, 2.1),
            Vector3::new(0.0, 0.0, 2.2),
            Vector3::new(0.0, 0.0, 2.3),
            Vector3::new(0.0, 0.0, 2.4),
        ];
        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 10,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut pose = Pose {
            world_to_camera: SE3::new(UnitQuaternion::identity(), Vector3::new(0.0, 0.0, -2.4)),
        };

        assert!(rescue_translation_direction(
            &mut pose,
            &translation_history,
            &direction_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!(pose.world_to_camera.translation.x.abs() < 1.0e-9);
        assert!(pose.world_to_camera.translation.y.abs() < 1.0e-9);
        assert!((pose.world_to_camera.translation.z - 2.4).abs() < 1.0e-9);
    }

    #[test]
    fn recent_translation_direction_uses_recent_window_and_ignores_bad_vectors() {
        let history = [
            Vector3::new(3.0, 0.0, 0.0),
            Vector3::new(f64::NAN, 0.0, 0.0),
            Vector3::zeros(),
            Vector3::new(0.0, 0.0, 2.0),
            Vector3::new(0.0, 0.0, 3.0),
        ];
        let direction = recent_translation_direction(&history, 4).expect("recent direction");
        assert!(direction.x.abs() < 1.0e-9);
        assert!(direction.y.abs() < 1.0e-9);
        assert!((direction.z - 1.0).abs() < 1.0e-9);

        assert!(recent_translation_direction(
            &[Vector3::new(1.0, 0.0, 0.0), Vector3::new(-1.0, 0.0, 0.0)],
            2,
        )
        .is_none());
        assert!(recent_translation_direction(&history, 0).is_none());
    }

    #[test]
    fn rotation_spike_rescue_requires_fast_history_and_weak_consensus() {
        let mut config = StereoVoFrontendConfig::default();
        config.rotation_spike_rescue_min_history = 4;
        config.rotation_spike_rescue_min_median_translation_m = 1.5;
        config.rotation_spike_rescue_min_angle_deg = 1.0;
        config.rotation_spike_rescue_max_angle_ratio = 3.0;
        config.rotation_spike_rescue_max_pnp_inlier_ratio = 0.45;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let rotation_history = [0.2, 0.3, 0.4, 0.3];
        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 10,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut pose = Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 3.0_f64.to_radians()),
                Vector3::new(2.0, 0.0, 0.0),
            ),
        };
        assert!(rescue_rotation_spike(
            &mut pose,
            &translation_history,
            &rotation_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!((relative_rotation_angle_deg(&pose) - 0.3).abs() < 1.0e-9);

        let strong_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 30,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut strong_pose = Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 3.0_f64.to_radians()),
                Vector3::new(2.0, 0.0, 0.0),
            ),
        };
        assert!(!rescue_rotation_spike(
            &mut strong_pose,
            &translation_history,
            &rotation_history,
            Some(&strong_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));

        let mut slow_pose = Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 3.0_f64.to_radians()),
                Vector3::new(2.0, 0.0, 0.0),
            ),
        };
        assert!(!rescue_rotation_spike(
            &mut slow_pose,
            &[0.7, 0.8, 0.9, 0.8],
            &rotation_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
    }

    #[test]
    fn rotation_vector_rescue_clamps_weak_rotation_to_recent_trend() {
        let mut config = StereoVoFrontendConfig::default();
        config.rotation_vector_rescue_min_history = 4;
        config.rotation_vector_rescue_min_median_translation_m = 1.5;
        config.rotation_vector_rescue_max_delta_deg = 0.4;
        config.rotation_vector_rescue_max_pnp_inlier_ratio = 0.45;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let rotation_history = [
            Vector3::new(0.0, 0.0, 0.2_f64.to_radians()),
            Vector3::new(0.0, 0.0, 0.3_f64.to_radians()),
            Vector3::new(0.0, 0.0, 0.4_f64.to_radians()),
            Vector3::new(0.0, 0.0, 0.3_f64.to_radians()),
        ];
        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 10,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut pose = Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 1.1_f64.to_radians()),
                Vector3::new(2.0, 0.0, 0.0),
            ),
        };

        assert!(rescue_rotation_vector(
            &mut pose,
            &translation_history,
            &rotation_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
        assert!((relative_rotation_angle_deg(&pose) - 0.3).abs() < 1.0e-9);
    }

    #[test]
    fn rotation_vector_rescue_requires_fast_history_and_weak_consensus() {
        let mut config = StereoVoFrontendConfig::default();
        config.rotation_vector_rescue_min_history = 4;
        config.rotation_vector_rescue_min_median_translation_m = 1.5;
        config.rotation_vector_rescue_max_delta_deg = 0.4;
        config.rotation_vector_rescue_max_pnp_inlier_ratio = 0.45;
        let translation_history = [2.1, 2.2, 2.3, 2.4];
        let rotation_history = [
            Vector3::new(0.0, 0.0, 0.2_f64.to_radians()),
            Vector3::new(0.0, 0.0, 0.3_f64.to_radians()),
            Vector3::new(0.0, 0.0, 0.4_f64.to_radians()),
            Vector3::new(0.0, 0.0, 0.3_f64.to_radians()),
        ];
        let strong_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 30,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut strong_pose = Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 1.1_f64.to_radians()),
                Vector3::new(2.0, 0.0, 0.0),
            ),
        };
        assert!(!rescue_rotation_vector(
            &mut strong_pose,
            &translation_history,
            &rotation_history,
            Some(&strong_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));

        let weak_report = PnpSelectedReport {
            pose: Pose::identity(),
            inlier_count: 10,
            correspondence_count: 40,
            mean_reprojection_error: 1.0,
        };
        let mut slow_pose = Pose {
            world_to_camera: SE3::new(
                UnitQuaternion::from_euler_angles(0.0, 0.0, 1.1_f64.to_radians()),
                Vector3::new(2.0, 0.0, 0.0),
            ),
        };
        assert!(!rescue_rotation_vector(
            &mut slow_pose,
            &[0.7, 0.8, 0.9, 0.8],
            &rotation_history,
            Some(&weak_report),
            StereoRelativePoseSource::Pnp,
            &config,
        ));
    }

    /// 3D-3D Kabsch round-trip: synthesize matched 3D points in two
    /// frames with known relative pose plus 3 outliers, run RANSAC,
    /// confirm pose recovery and outlier rejection.
    #[test]
    fn kabsch_ransac_recovers_metric_pose_and_rejects_outliers() {
        let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.1)
            * UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -0.05);
        let truth_t = Vector3::new(0.3, -0.05, 1.5);
        let r_mat = r.to_rotation_matrix().into_inner();
        let inliers: Vec<Point3<f64>> = vec![
            Point3::new(-1.0, 0.0, 8.0),
            Point3::new(1.5, 0.5, 12.0),
            Point3::new(0.0, -0.3, 6.0),
            Point3::new(2.0, 0.5, 15.0),
            Point3::new(-2.5, -0.4, 10.0),
            Point3::new(0.5, 0.6, 5.0),
            Point3::new(-1.5, 0.2, 9.0),
            Point3::new(1.0, -0.2, 7.0),
            Point3::new(-0.5, 0.0, 11.0),
            Point3::new(2.5, -0.5, 14.0),
            Point3::new(-2.0, 0.4, 8.5),
            Point3::new(0.0, 0.5, 6.5),
        ];
        let mut corrs: Vec<StereoPairCorrespondence> = inliers
            .iter()
            .map(|p| StereoPairCorrespondence {
                a: *p,
                b: Point3::from(r_mat * p.coords + truth_t),
                confidence: None,
            })
            .collect();
        // Three gross outliers (random offsets).
        corrs.push(StereoPairCorrespondence {
            a: Point3::new(0.0, 0.0, 10.0),
            b: Point3::new(8.0, 0.0, 4.0),
            confidence: None,
        });
        corrs.push(StereoPairCorrespondence {
            a: Point3::new(1.0, 0.0, 12.0),
            b: Point3::new(-4.0, 0.0, 6.0),
            confidence: None,
        });
        corrs.push(StereoPairCorrespondence {
            a: Point3::new(0.5, 0.5, 9.0),
            b: Point3::new(0.5, -3.5, 3.0),
            confidence: None,
        });
        let report = estimate_relative_pose_kabsch_ransac(&corrs, &KabschRansacConfig::default())
            .expect("Kabsch RANSAC must succeed on mostly-inlier data");
        let recovered_t = report.relative_pose.world_to_camera.translation;
        assert!(
            (recovered_t - truth_t).norm() < 1.0e-6,
            "translation err {:?}",
            recovered_t,
        );
        let recovered_r = report
            .relative_pose
            .world_to_camera
            .rotation
            .to_rotation_matrix()
            .into_inner();
        assert!(
            (recovered_r - r_mat).norm() < 1.0e-6,
            "rotation err {:?}",
            recovered_r,
        );
        // All 12 inliers should be recovered, the 3 outliers rejected.
        assert!(
            report.inliers.len() >= 12,
            "should keep all inliers: {}",
            report.inliers.len()
        );
        assert!(report.inliers.len() <= 12, "should reject outliers");
    }

    /// `StereoVoFrontend` round-trip on a synthetic 4-frame KITTI-like
    /// stereo sequence. Verifies that (a) the first frame seeds at
    /// identity, (b) the last frame's recovered camera center matches
    /// truth in metric units, and (c) the per-call state collections
    /// stay in lockstep.
    #[test]
    fn stereo_vo_frontend_recovers_synthetic_metric_trajectory() {
        // Pack a synthetic image so CornerFeatureExtractor returns
        // exactly the keypoints / descriptors we want — i.e., construct
        // a frontend that doesn't actually depend on the FAST detector.
        // Simpler path: use the lower-level helpers directly to validate
        // the frontend's state-tracking is correct on synthesized inputs.
        let camera = camera();
        let baseline = 0.537150888;
        let (fx, _, _, _) = camera.intrinsics().unwrap();
        let landmarks: Vec<Point3<f64>> = vec![
            Point3::new(-2.0, 0.5, 12.0),
            Point3::new(1.5, -0.3, 9.0),
            Point3::new(0.0, 0.0, 18.0),
            Point3::new(-1.0, 0.6, 7.0),
            Point3::new(2.0, 0.0, 14.0),
            Point3::new(-3.0, -0.2, 11.0),
        ];
        let mut poses_truth: Vec<Pose> = Vec::new();
        for i in 0..3 {
            let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.03 * i as f64);
            let center = Vector3::new(0.05 * i as f64, 0.0, 1.2 * i as f64);
            poses_truth.push(Pose::from_world_to_camera(
                r,
                -(r.transform_vector(&center)),
            ));
        }
        let mut frontend = StereoVoFrontend::new(
            camera.clone(),
            baseline,
            StereoVoFrontendConfig {
                kabsch: KabschRansacConfig {
                    iterations: 64,
                    inlier_threshold_m: 0.01,
                    min_inliers: 4,
                    max_depth_m: 30.0,
                    seed: 7,
                },
                ..StereoVoFrontendConfig::default()
            },
        );
        // Inject synthetic state directly: skip the image-extractor
        // path (which the test doesn't aim to exercise) and validate
        // only the per-pair Kabsch + composition logic.
        for pose in &poses_truth {
            let mut kps: Vec<Point2<f64>> = Vec::new();
            let mut descs: Vec<Vec<f32>> = Vec::new();
            let mut sf: Vec<StereoFeature> = Vec::new();
            for (id, p) in landmarks.iter().enumerate() {
                let xc = pose.transform_world_point(p);
                let l = camera.project(&xc).unwrap();
                let u_r = l.x - fx * baseline / xc.z;
                let kp_idx = kps.len();
                kps.push(l);
                descs.push(vec![id as f32]);
                sf.push(StereoFeature {
                    left_index: kp_idx,
                    right_index: kp_idx,
                    disparity: l.x - u_r,
                    point_cam: Point3::from(xc.coords),
                });
            }
            let lf = FeatureSet::new(kps.clone(), descs.clone()).unwrap();
            let rf = FeatureSet::new(kps, descs).unwrap();
            // Manually advance frontend state, mimicking process_pair
            // without invoking the FAST extractor on a synthetic image.
            let new_pose = if let (Some(prev_left), Some(prev_stereo)) = (
                frontend.left_features.last(),
                frontend.stereo_per_frame.last(),
            ) {
                let temporal_matches = frontend
                    .matcher
                    .match_descriptors(&prev_left.descriptors, &lf.descriptors);
                let pair_corrs = stereo_pair_correspondences(prev_stereo, &sf, &temporal_matches);
                let report =
                    estimate_relative_pose_kabsch_ransac(&pair_corrs, &frontend.config.kabsch)
                        .expect("kabsch should succeed on synthetic data");
                frontend.kabsch_inlier_counts.push(report.inliers.len());
                frontend
                    .per_pair_translation_m
                    .push(report.relative_pose.world_to_camera.translation.norm());
                frontend
                    .per_pair_translation_vectors
                    .push(report.relative_pose.world_to_camera.translation);
                let relative = report.relative_pose.world_to_camera.clone();
                let last = frontend.poses.last().unwrap();
                Pose {
                    world_to_camera: relative.compose(&last.world_to_camera),
                }
            } else {
                Pose::identity()
            };
            frontend.poses.push(new_pose);
            frontend.left_features.push(lf);
            frontend.right_features.push(rf);
            frontend.stereo_per_frame.push(sf);
        }
        assert_eq!(frontend.frame_count(), 3);
        assert_eq!(frontend.left_features.len(), 3);
        assert_eq!(frontend.stereo_per_frame.len(), 3);
        assert_eq!(frontend.per_pair_translation_m.len(), 2);
        assert_eq!(frontend.per_pair_translation_vectors.len(), 2);
        assert_eq!(frontend.kabsch_inlier_counts.len(), 2);
        for (magnitude, vector) in frontend
            .per_pair_translation_m
            .iter()
            .zip(frontend.per_pair_translation_vectors.iter())
        {
            assert!((vector.norm() - magnitude).abs() < 1.0e-9);
        }
        // Frame 0 should be identity, last-frame center should match truth.
        let center_0 = frontend.poses[0].camera_center_world();
        assert!(center_0.coords.norm() < 1.0e-9);
        let recovered_last = frontend.poses[2].camera_center_world();
        let truth_last = -poses_truth[2]
            .world_to_camera
            .rotation
            .inverse()
            .transform_vector(&poses_truth[2].world_to_camera.translation);
        assert!(
            (recovered_last.coords - truth_last).norm() < 1.0e-6,
            "recovered {recovered_last:?} truth {truth_last:?}",
        );
        // Trajectory length should be ≈ sum of per-pair translations.
        let len_traj = frontend.trajectory_length_m();
        let len_pairs: f64 = frontend.per_pair_translation_m.iter().sum();
        assert!((len_traj - len_pairs).abs() < 1.0e-6);
        // Adaptive search radius respects the floor.
        assert!(frontend.adaptive_track_search_radius_px() >= 20.0);
    }

    #[test]
    fn stereo_vo_frontend_accepts_external_features_and_matches() {
        let camera = camera();
        let baseline = 0.537150888;
        let (fx, _, _, _) = camera.intrinsics().unwrap();
        let landmarks: Vec<Point3<f64>> = vec![
            Point3::new(-2.0, 0.5, 12.0),
            Point3::new(1.5, -0.3, 9.0),
            Point3::new(0.0, 0.0, 18.0),
            Point3::new(-1.0, 0.6, 7.0),
            Point3::new(2.0, 0.0, 14.0),
            Point3::new(-3.0, -0.2, 11.0),
        ];
        let mut poses_truth: Vec<Pose> = Vec::new();
        for i in 0..3 {
            let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.03 * i as f64);
            let center = Vector3::new(0.05 * i as f64, 0.0, 1.2 * i as f64);
            poses_truth.push(Pose::from_world_to_camera(
                r,
                -(r.transform_vector(&center)),
            ));
        }
        let mut frontend = StereoVoFrontend::new(
            camera.clone(),
            baseline,
            StereoVoFrontendConfig {
                kabsch: KabschRansacConfig {
                    iterations: 64,
                    inlier_threshold_m: 0.01,
                    min_inliers: 4,
                    max_depth_m: 30.0,
                    seed: 7,
                },
                relative_pose_mode: StereoRelativePoseMode::KabschThenPnp,
                ..StereoVoFrontendConfig::default()
            },
        );

        for (frame_index, pose) in poses_truth.iter().enumerate() {
            let mut left_keypoints = Vec::new();
            let mut right_keypoints = Vec::new();
            let mut left_descriptors = Vec::new();
            let mut right_descriptors = Vec::new();
            let mut stereo_matches = Vec::new();

            for (id, point_world) in landmarks.iter().enumerate() {
                let point_cam = pose.transform_world_point(point_world);
                let left = camera.project(&point_cam).unwrap();
                let right = Point2::new(left.x - fx * baseline / point_cam.z, left.y);
                left_keypoints.push(left);
                right_keypoints.push(right);
                // All descriptors are deliberately identical. The explicit
                // LightGlue-style matches below, not the internal matcher,
                // are what make this sequence trackable.
                left_descriptors.push(vec![0.0]);
                right_descriptors.push(vec![0.0]);
                stereo_matches.push(descriptor_match(id, id));
            }

            let left_features = FeatureSet::new(left_keypoints, left_descriptors).unwrap();
            let right_features = FeatureSet::new(right_keypoints, right_descriptors).unwrap();
            let temporal_matches = if frame_index == 0 {
                None
            } else {
                Some(
                    (0..landmarks.len())
                        .map(|id| descriptor_match(id, id))
                        .collect::<Vec<_>>(),
                )
            };

            frontend
                .process_feature_pair_with_matches(
                    left_features,
                    right_features,
                    Some(&stereo_matches),
                    temporal_matches.as_deref(),
                )
                .unwrap();
        }

        assert_eq!(frontend.frame_count(), 3);
        assert_eq!(frontend.stereo_per_frame[0].len(), landmarks.len());
        assert_eq!(frontend.pair_diagnostics.len(), 2);
        for diagnostics in &frontend.pair_diagnostics {
            assert_eq!(diagnostics.source, StereoRelativePoseSource::Kabsch);
            assert_eq!(diagnostics.temporal_match_count, landmarks.len());
            assert_eq!(
                diagnostics.stereo_pair_correspondence_count,
                landmarks.len()
            );
        }
        let recovered_last = frontend.poses[2].camera_center_world();
        let truth_last = -poses_truth[2]
            .world_to_camera
            .rotation
            .inverse()
            .transform_vector(&poses_truth[2].world_to_camera.translation);
        assert!(
            (recovered_last.coords - truth_last).norm() < 1.0e-6,
            "recovered {recovered_last:?} truth {truth_last:?}",
        );
    }

    /// Projection-guided extension: synthesize a 4-frame world-frame
    /// trajectory plus 5 fixed world-frame landmarks. Each frame's
    /// FeatureSet contains the projected pixels (left + right with
    /// rectified-stereo offset). Run extension, confirm tracks span all
    /// 4 frames and consume each (frame, kp) at most once.
    #[test]
    fn extend_stereo_tracks_via_projection_finds_long_chains() {
        let camera = camera();
        let baseline = 0.537150888;
        let (fx, _, _, _) = camera.intrinsics().unwrap();
        let landmarks: Vec<Point3<f64>> = vec![
            Point3::new(-2.0, 0.5, 12.0),
            Point3::new(1.5, -0.3, 9.0),
            Point3::new(0.0, 0.0, 18.0),
            Point3::new(-1.0, 0.6, 7.0),
            Point3::new(2.0, 0.0, 14.0),
        ];
        // Forward camera motion of 0.8 m / frame, slight yaw rotation.
        let mut poses: Vec<Pose> = Vec::new();
        for i in 0..4 {
            let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.02 * i as f64);
            let center = Vector3::new(0.0, 0.0, 0.8 * i as f64);
            let t = -(r.transform_vector(&center));
            poses.push(Pose::from_world_to_camera(r, t));
        }
        // Build per-frame FeatureSets and stereo features.
        let mut left_features: Vec<FeatureSet> = Vec::new();
        let mut stereo_per_frame: Vec<Vec<StereoFeature>> = Vec::new();
        for pose in &poses {
            let mut kps: Vec<Point2<f64>> = Vec::new();
            let mut descs: Vec<Vec<f32>> = Vec::new();
            let mut sf: Vec<StereoFeature> = Vec::new();
            for (id, p) in landmarks.iter().enumerate() {
                let xc = pose.transform_world_point(p);
                let l = camera.project(&xc).unwrap();
                let u_r = l.x - fx * baseline / xc.z;
                let kp_idx = kps.len();
                kps.push(l);
                descs.push(vec![id as f32]);
                sf.push(StereoFeature {
                    left_index: kp_idx,
                    right_index: kp_idx,
                    disparity: l.x - u_r,
                    point_cam: Point3::from(xc.coords),
                });
            }
            left_features.push(FeatureSet::new(kps, descs).unwrap());
            stereo_per_frame.push(sf);
        }
        let tracks = extend_stereo_tracks_via_projection(
            &poses,
            &left_features,
            &stereo_per_frame,
            &camera,
            &TrackExtensionConfig::default(),
        );
        // Each landmark should produce exactly one 4-frame track (with
        // deduplication on, the same physical point isn't re-seeded).
        assert_eq!(tracks.len(), 5, "got {} tracks", tracks.len());
        for track in &tracks {
            assert_eq!(track.observations.len(), 4, "track len {:?}", track);
        }
    }

    /// End-to-end PnP round-trip: synthesize a metric scene, triangulate
    /// stereo features at frame A, build correspondences against frame B
    /// (with a known relative pose), run PnP, and verify the recovered
    /// pose matches truth in metric units.
    #[test]
    fn stereo_vo_pnp_round_trips_metric_relative_pose() {
        use crate::pnp::{DltPnP, PoseEstimator};
        let camera = camera();
        let baseline = 0.5;
        let (fx, _, _, _) = camera.intrinsics().unwrap();
        let truth_t_a_to_b = Vector3::new(0.4, 0.0, 1.2);
        let r = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.05);
        let pose_a = Pose::identity();
        let pose_b = Pose::from_world_to_camera(r, -(r.transform_vector(&truth_t_a_to_b)));

        let world_points: Vec<Point3<f64>> = vec![
            Point3::new(-1.0, -0.3, 6.0),
            Point3::new(1.0, 0.3, 8.0),
            Point3::new(0.0, 0.0, 10.0),
            Point3::new(-1.5, 0.2, 4.0),
            Point3::new(2.0, -0.5, 12.0),
            Point3::new(0.5, 0.6, 5.5),
        ];
        let mut left_a: Vec<(usize, Point2<f64>)> = Vec::new();
        let mut right_a: Vec<(usize, Point2<f64>)> = Vec::new();
        let mut left_b: Vec<(usize, Point2<f64>)> = Vec::new();
        for (id, p) in world_points.iter().enumerate() {
            let xa = pose_a.transform_world_point(p);
            let xb = pose_b.transform_world_point(p);
            let la = camera.project(&xa).unwrap();
            let lb = camera.project(&xb).unwrap();
            let ra = Point2::new(la.x - fx * baseline / xa.z, la.y);
            left_a.push((id, la));
            right_a.push((id, ra));
            left_b.push((id, lb));
        }
        let left_a_fs = synthetic_feature_set(&left_a);
        let right_a_fs = synthetic_feature_set(&right_a);
        let left_b_fs = synthetic_feature_set(&left_b);
        let stereo_a = triangulate_stereo_features(
            &left_a_fs,
            &right_a_fs,
            &camera,
            baseline,
            &StereoFeatureConfig {
                ratio: None,
                ..StereoFeatureConfig::default()
            },
        );
        let matcher = BruteForceMatcher { ratio: None };
        let temporal_matches =
            matcher.match_descriptors(&left_a_fs.descriptors, &left_b_fs.descriptors);
        let corrs = build_stereo_temporal_correspondences(&stereo_a, &left_b_fs, &temporal_matches);
        assert_eq!(corrs.len(), world_points.len());
        let estimator = DltPnP::default();
        let recovered = estimator
            .estimate_pose(&corrs, &camera)
            .expect("DLT must succeed on noise-free input");
        // PnP recovers T_a_to_b directly because the 3D points were given
        // in frame-a's camera frame and the 2D pixels are in frame-b.
        let recovered_center = recovered.camera_center_world();
        let truth_center = -(pose_b
            .world_to_camera
            .rotation
            .inverse()
            .transform_vector(&pose_b.world_to_camera.translation));
        let err = (recovered_center.coords - truth_center).norm();
        assert!(
            err < 1.0e-9,
            "stereo VO PnP center error {err}: recovered {recovered_center:?} truth {truth_center:?}"
        );
    }
}
