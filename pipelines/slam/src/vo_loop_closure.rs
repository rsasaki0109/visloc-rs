//! Loop-closure pose-graph optimization for an open stereo-VO trajectory.
//!
//! A streaming stereo-VO frontend (classical or learned) produces a
//! drift-accumulating *open* trajectory: only the first pose is gauge-fixed and
//! nothing ties a revisited place back to its earlier observation. On a loopy
//! sequence (KITTI seq00 revisits frame 86 around frame 1561, and closes the
//! grand loop near frame 4404) the accumulated drift is exactly what a loop
//! closure removes — and dense global bundle adjustment provably cannot, because
//! a loop-free reprojection-minimum just deforms the trajectory without a
//! constraint pulling the revisit back.
//!
//! This module is the glue that turns the per-frame VO products — poses, left
//! [`FeatureSet`]s, and per-frame stereo depth ([`StereoFeature`]) — into a
//! [`PoseGraph`] and runs the robust GNC SE(3) optimizer:
//!
//! 1. **Appearance** — a VLAD global descriptor per frame, from a k-means
//!    vocabulary pooled over (subsampled) left descriptors.
//! 2. **Proposal** — for each frame, the most similar earlier frames beyond a
//!    temporal gap, above a cosine-similarity floor.
//! 3. **Verification** — brute-force descriptor matching between the two frames'
//!    left features, lifted to 2D-3D via the older frame's stereo depth, then
//!    [`PnPLoopClosureVerifier`]. PnP yields a *metric* relative pose, so no
//!    separate scale source is needed.
//! 4. **Optimization** — sequential odometry edges (from the VO poses) plus the
//!    verified loop edges, anchored at frame 0, solved with [`PoseGraph::optimize_se3_gnc`]
//!    so a surviving spurious loop is down-weighted rather than trusted.
//!
//! The map-bound online-SLAM helpers ([`crate::verify_loop_closure_candidates_pnp`])
//! require a [`crate::VisualMap`]/[`crate::Keyframe`] that the file-backed VO path
//! never builds; this module reuses the same map-free primitives
//! ([`PnPLoopClosureVerifier::verify`], [`LoopClosureConstraint`], [`PoseGraph`])
//! over raw feature sets instead.

use std::collections::HashMap;

use nalgebra::Point3;
use rayon::prelude::*;

use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;
use visloc_vision::matching::{BruteForceMatcher, Matcher};
use visloc_vision::pnp::Correspondence2D3D;
use visloc_vision::place_recognition::{cosine_similarity, vlad, Vocabulary};
use visloc_vision::ransac::PnPRansac;
use visloc_vision::stereo_vo::StereoFeature;

use crate::gnc::{GncConfig, AUTO_SCALE_K};
use crate::{
    relative_world_to_camera, LinearSolver, LoopClosureConstraint, PnPLoopClosureVerifier,
    PnPLoopClosureVerifierConfig, PoseGraph, PoseGraphError, PoseGraphGncResult,
    PoseGraphSe3Config, RobustKernel,
};

/// Configuration for [`close_loops_on_vo_trajectory`].
#[derive(Debug, Clone)]
pub struct VoLoopClosureConfig {
    /// Visual-word count for the VLAD vocabulary (k-means centroids).
    pub vocab_k: usize,
    /// Lloyd iterations for vocabulary construction.
    pub vocab_iterations: usize,
    /// Deterministic seed for vocabulary k-means++ seeding.
    pub vocab_seed: u64,
    /// Keep every `vocab_descriptor_stride`-th left descriptor when pooling for
    /// vocabulary construction (`1` = use all; larger keeps k-means tractable on
    /// thousand-frame sequences).
    pub vocab_descriptor_stride: usize,
    /// Optional hard cap on the pooled descriptor count fed to k-means, applied
    /// after striding by evenly sub-sampling the pool. k-means is
    /// `O(pool · k · dim · iters)`, so without a cap a 4500-frame sequence pools
    /// millions of descriptors and vocabulary construction alone runs for the
    /// better part of an hour. A few tens of thousands of descriptors already
    /// seed a stable `k=64` vocabulary. `None` uses the full strided pool.
    pub vocab_max_pool: Option<usize>,
    /// Minimum temporal gap (in frames) between the two frames of a loop
    /// candidate. A cheap floor; the portable gate is `min_path_length`.
    pub min_frame_gap: usize,
    /// Optional minimum accumulated VO path length (metres travelled along the
    /// trajectory) between a candidate's two frames. This is the *frame-rate-
    /// and speed-independent* loop gate: a loop only corrects drift if drift had
    /// room to accumulate between the revisit and its original observation, and
    /// drift grows with distance travelled — not with frame index. A small
    /// frame gap during slow motion (e.g. a hovering UAV at 20 Hz) travels
    /// almost nothing and yields odometry-consistent "loops" that contribute no
    /// correction; gating on path length rejects them without per-dataset
    /// frame-gap tuning. `None` disables the gate (frame gap only). Measured on
    /// EuRoC MH_03: frame-gap 30 alone left ATE unchanged (2.46 m), while a few
    /// metres of required travel recovers the genuine long-range revisits.
    pub min_path_length: Option<f64>,
    /// Minimum VLAD cosine similarity for a frame pair to become a candidate.
    pub min_similarity: f32,
    /// Keep at most this many earlier frames per query frame as candidates
    /// (strongest similarity first).
    pub max_candidates_per_frame: usize,
    /// Optional global cap on how many candidates reach PnP verification, taken
    /// in descending appearance similarity. Bounds the cost of the dominant
    /// stage — per-pair brute-force descriptor matching is `O(N_kp² · dim)` — on
    /// long sequences where appearance alone proposes thousands of (mostly
    /// redundant) pairs. `None` verifies every candidate.
    pub max_verifications: Option<usize>,
    /// Lowe-ratio for the loop-pair descriptor matcher.
    pub match_ratio: Option<f32>,
    /// PnP loop-closure verifier thresholds.
    pub verifier: PnPLoopClosureVerifierConfig,
    /// PnP RANSAC backend used by the verifier.
    pub ransac: PnPRansac,
    /// SE(3) pose-graph solver configuration.
    pub se3: PoseGraphSe3Config,
    /// GNC robust-optimization configuration.
    pub gnc: GncConfig,
}

impl Default for VoLoopClosureConfig {
    fn default() -> Self {
        Self {
            vocab_k: 64,
            vocab_iterations: 10,
            vocab_seed: 0xC0FFEE,
            vocab_descriptor_stride: 4,
            vocab_max_pool: Some(60_000),
            min_frame_gap: 50,
            min_path_length: Some(5.0),
            min_similarity: 0.20,
            max_candidates_per_frame: 3,
            max_verifications: Some(400),
            match_ratio: Some(0.8),
            verifier: PnPLoopClosureVerifierConfig::default(),
            ransac: PnPRansac::default(),
            se3: PoseGraphSe3Config {
                robust_kernel: RobustKernel::None,
                linear_solver: LinearSolver::Sparse,
                ..PoseGraphSe3Config::default()
            },
            gnc: GncConfig {
                auto_scale: Some(AUTO_SCALE_K),
                ..GncConfig::default()
            },
        }
    }
}

/// An appearance-proposed loop candidate (older revisited frame ← newer query
/// frame), before geometric verification.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoopCandidatePair {
    /// Earlier (revisited) frame index.
    pub older: usize,
    /// Later (query) frame index. Always `older + min_frame_gap <= newer`.
    pub newer: usize,
    /// VLAD cosine similarity that proposed the pair.
    pub similarity: f32,
}

/// Output of [`close_loops_on_vo_trajectory`].
#[derive(Debug, Clone)]
pub struct VoLoopClosureResult {
    /// Trajectory after loop-closure PGO, in input frame order. Equal to the
    /// input poses when no loop was verified.
    pub refined_poses: Vec<Pose>,
    /// Verified metric loop constraints fed into the pose graph.
    pub loop_constraints: Vec<LoopClosureConstraint>,
    /// Appearance-proposed candidate count (pre-verification).
    pub candidate_count: usize,
    /// GNC solver report. `None` when no loop was verified (PGO is skipped and
    /// the trajectory is returned unchanged).
    pub gnc: Option<PoseGraphGncResult>,
}

impl VoLoopClosureResult {
    /// Number of verified loop constraints.
    pub fn verified_count(&self) -> usize {
        self.loop_constraints.len()
    }
}

/// Error returned by [`close_loops_on_vo_trajectory`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VoLoopClosureError {
    /// `poses`, `left_features`, and `stereo_per_frame` disagree in length.
    InputLengthMismatch,
    /// Fewer than two frames — nothing to close.
    TooFewFrames(usize),
    /// k-means vocabulary construction failed (too few / zero-dim descriptors).
    VocabularyBuildFailed,
    /// The pose-graph solver failed.
    PoseGraph(PoseGraphError),
}

impl std::fmt::Display for VoLoopClosureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InputLengthMismatch => write!(
                f,
                "poses, left_features, and stereo_per_frame must have equal length"
            ),
            Self::TooFewFrames(n) => write!(f, "need at least 2 frames, got {n}"),
            Self::VocabularyBuildFailed => write!(
                f,
                "VLAD vocabulary construction failed (too few or zero-dimension descriptors)"
            ),
            Self::PoseGraph(err) => write!(f, "pose-graph optimization failed: {err:?}"),
        }
    }
}

impl std::error::Error for VoLoopClosureError {}

impl From<PoseGraphError> for VoLoopClosureError {
    fn from(err: PoseGraphError) -> Self {
        Self::PoseGraph(err)
    }
}

/// Build a VLAD global descriptor per frame from a k-means vocabulary pooled
/// over (subsampled) left descriptors.
fn compute_frame_globals(
    left_features: &[FeatureSet],
    config: &VoLoopClosureConfig,
) -> Result<Vec<Vec<f32>>, VoLoopClosureError> {
    let stride = config.vocab_descriptor_stride.max(1);
    let mut pool: Vec<&[f32]> = Vec::new();
    for features in left_features {
        for descriptor in features.descriptors.iter().step_by(stride) {
            pool.push(descriptor.as_slice());
        }
    }
    // Cap the k-means pool independent of sequence length: k-means is
    // O(pool · k · dim · iters), so a multi-thousand-frame sequence would
    // otherwise spend tens of minutes here. Evenly sub-sample to the cap.
    if let Some(cap) = config.vocab_max_pool {
        if cap > 0 && pool.len() > cap {
            let step = pool.len() as f64 / cap as f64;
            pool = (0..cap).map(|i| pool[(i as f64 * step) as usize]).collect();
        }
    }
    let vocab = Vocabulary::build(
        &pool,
        config.vocab_k,
        config.vocab_iterations,
        config.vocab_seed,
    )
    .ok_or(VoLoopClosureError::VocabularyBuildFailed)?;

    // VLAD per frame is independent → parallelize (each is a full nearest-
    // centroid pass over that frame's descriptors). `collect` preserves order.
    Ok(left_features
        .par_iter()
        .map(|features| vlad(&features.descriptors, &vocab))
        .collect())
}

/// Propose loop candidates by gap-constrained appearance retrieval: for each
/// query frame, its strongest earlier matches beyond `min_frame_gap` whose
/// cosine similarity clears `min_similarity`.
///
/// Unlike a full mutual-NN over one set (which collapses to the identity
/// diagonal), this scans only the strictly-earlier prefix and leaves the
/// false-positive rejection to PnP verification + GNC, mirroring the standard
/// appearance-then-geometry loop-detection pipeline.
pub fn detect_loop_candidates(
    globals: &[Vec<f32>],
    min_frame_gap: usize,
    min_similarity: f32,
    max_candidates_per_frame: usize,
) -> Vec<LoopCandidatePair> {
    if min_frame_gap == 0 || max_candidates_per_frame == 0 {
        return Vec::new();
    }
    // Each query frame scans its earlier prefix independently — the O(N²·dim)
    // retrieval is the dominant cost on long sequences, so parallelize across
    // query frames. `flat_map_iter` over an indexed range keeps the output
    // order deterministic.
    (min_frame_gap..globals.len())
        .into_par_iter()
        .flat_map_iter(|newer| {
            let last_older = newer - min_frame_gap;
            let mut scored: Vec<(usize, f32)> = (0..=last_older)
                .map(|older| (older, cosine_similarity(&globals[newer], &globals[older])))
                .filter(|&(_, similarity)| similarity >= min_similarity)
                .collect();
            scored.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.0.cmp(&b.0))
            });
            scored
                .into_iter()
                .take(max_candidates_per_frame)
                .map(move |(older, similarity)| LoopCandidatePair {
                    older,
                    newer,
                    similarity,
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

/// Geometrically verify one candidate pair. Matches the newer frame's left
/// descriptors against the older frame's, lifts the older keypoints to world
/// 3D via the older frame's stereo depth and pose, and runs PnP. Returns a
/// metric loop constraint (`older -> newer`) when the verifier accepts it.
#[allow(clippy::too_many_arguments)]
fn verify_loop_candidate(
    camera: &Camera,
    poses: &[Pose],
    left_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    older: usize,
    newer: usize,
    matcher: &BruteForceMatcher,
    verifier: &PnPLoopClosureVerifier,
) -> Option<LoopClosureConstraint> {
    let older_pose = &poses[older];
    let camera_to_world = older_pose.camera_to_world();

    // older left-keypoint index -> world 3D point (only keypoints with a valid
    // stereo depth contribute a 2D-3D correspondence).
    let mut world_by_index: HashMap<usize, Point3<f64>> = HashMap::new();
    for stereo in &stereo_per_frame[older] {
        world_by_index.insert(
            stereo.left_index,
            camera_to_world.transform_point(&stereo.point_cam),
        );
    }
    if world_by_index.is_empty() {
        return None;
    }

    let matches =
        matcher.match_descriptors(&left_features[newer].descriptors, &left_features[older].descriptors);
    let mut correspondences = Vec::with_capacity(matches.len());
    for descriptor_match in &matches {
        let Some(&point3d) = world_by_index.get(&descriptor_match.train_index) else {
            continue;
        };
        let Some(&point2d) = left_features[newer].keypoints.get(descriptor_match.query_index) else {
            continue;
        };
        correspondences.push(Correspondence2D3D {
            point2d,
            point3d,
            confidence: descriptor_match.confidence,
        });
    }

    let verification = verifier.verify(&correspondences, older_pose, camera);
    if !verification.verified {
        return None;
    }
    let relative_pose = verification.relative_pose.clone()?;
    Some(LoopClosureConstraint {
        from_keyframe_id: older as u64,
        to_keyframe_id: newer as u64,
        relative_pose,
        inlier_count: verification.inlier_count,
        inlier_ratio: verification.inlier_ratio,
        mean_sampson_error: verification.mean_sampson_error,
        score: verification.score,
    })
}

/// Accumulated trajectory length up to each frame: `cum[0] = 0`,
/// `cum[i] = cum[i-1] + ‖center_i − center_{i-1}‖`. The arc length between two
/// frames `a ≤ b` is `cum[b] − cum[a]`.
fn cumulative_path_length(poses: &[Pose]) -> Vec<f64> {
    let mut cum = vec![0.0; poses.len()];
    for i in 1..poses.len() {
        let step = (poses[i].camera_center_world().coords
            - poses[i - 1].camera_center_world().coords)
            .norm();
        cum[i] = cum[i - 1] + step;
    }
    cum
}

/// Close loops on an open stereo-VO trajectory and return the globally
/// consistent poses.
///
/// `poses[i]` is frame `i`'s `world_to_camera`; `left_features[i]` and
/// `stereo_per_frame[i]` are the same products the BA refiner consumes
/// ([`crate::refine_stereo_vo_with_ba`]). When no loop survives verification the
/// trajectory is returned unchanged (`gnc == None`).
pub fn close_loops_on_vo_trajectory(
    camera: &Camera,
    poses: &[Pose],
    left_features: &[FeatureSet],
    stereo_per_frame: &[Vec<StereoFeature>],
    config: &VoLoopClosureConfig,
) -> Result<VoLoopClosureResult, VoLoopClosureError> {
    let n = poses.len();
    if left_features.len() != n || stereo_per_frame.len() != n {
        return Err(VoLoopClosureError::InputLengthMismatch);
    }
    if n < 2 {
        return Err(VoLoopClosureError::TooFewFrames(n));
    }

    let globals = compute_frame_globals(left_features, config)?;
    let mut candidates = detect_loop_candidates(
        &globals,
        config.min_frame_gap,
        config.min_similarity,
        config.max_candidates_per_frame,
    );

    // Path-length gate: keep only candidates whose two frames are separated by
    // enough accumulated travel for drift to have built up. Speed-/frame-rate-
    // independent, so it needs no per-dataset frame-gap tuning.
    if let Some(min_path) = config.min_path_length {
        if min_path > 0.0 {
            let cum = cumulative_path_length(poses);
            candidates.retain(|c| cum[c.newer] - cum[c.older] >= min_path);
        }
    }

    let matcher = BruteForceMatcher {
        ratio: config.match_ratio,
    };
    let verifier = PnPLoopClosureVerifier::new(config.ransac, config.verifier);

    // Verify in descending appearance similarity so a `max_verifications` cap
    // spends the (expensive) geometric stage on the most promising pairs first.
    let mut to_verify: Vec<&LoopCandidatePair> = candidates.iter().collect();
    to_verify.sort_by(|a, b| {
        b.similarity
            .partial_cmp(&a.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    if let Some(cap) = config.max_verifications {
        to_verify.truncate(cap);
    }

    // Each candidate's geometric check (a brute-force descriptor match +
    // PnP RANSAC, O(N_kp²·dim)) is independent and is the other dominant cost,
    // so verify in parallel. The PnP RANSAC seed is fixed, so results are
    // thread-count-independent; `collect` preserves the (similarity-sorted)
    // order so the pose graph's edge order is deterministic.
    let loop_constraints: Vec<LoopClosureConstraint> = to_verify
        .par_iter()
        .filter_map(|candidate| {
            verify_loop_candidate(
                camera,
                poses,
                left_features,
                stereo_per_frame,
                candidate.older,
                candidate.newer,
                &matcher,
                &verifier,
            )
        })
        .collect();

    if loop_constraints.is_empty() {
        return Ok(VoLoopClosureResult {
            refined_poses: poses.to_vec(),
            loop_constraints,
            candidate_count: candidates.len(),
            gnc: None,
        });
    }

    let mut graph = PoseGraph::new();
    for (index, pose) in poses.iter().enumerate() {
        graph.add_pose(index as u64, pose.clone());
    }
    graph.anchor(0);
    for index in 0..n - 1 {
        graph.add_sequential_edge(
            index as u64,
            (index + 1) as u64,
            relative_world_to_camera(&poses[index], &poses[index + 1]),
        );
    }
    for constraint in &loop_constraints {
        graph.add_loop_closure_constraint(constraint);
    }

    let gnc = graph.optimize_se3_gnc(&config.se3, &config.gnc)?;

    let refined_poses = (0..n)
        .map(|index| {
            graph
                .poses
                .get(&(index as u64))
                .cloned()
                .unwrap_or_else(|| poses[index].clone())
        })
        .collect();

    Ok(VoLoopClosureResult {
        refined_poses,
        loop_constraints,
        candidate_count: candidates.len(),
        gnc: Some(gnc),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Point2, UnitQuaternion, Vector3};
    use visloc_core::geometry::SE3;

    fn camera() -> Camera {
        Camera::pinhole(0, 1280, 480, 600.0, 600.0, 640.0, 240.0)
    }

    /// A landmark's descriptor: a deterministic, *continuous* per-id vector so
    /// the same physical point matches across frames, while no k-means centroid
    /// sits exactly on it (which would collapse the VLAD residual to zero, an
    /// artifact of sparse one-hot synthetic descriptors that real 256-d
    /// SuperPoint vectors never exhibit).
    fn descriptor_for(id: usize, dim: usize) -> Vec<f32> {
        let mut d = vec![0.0f32; dim];
        let phase = id as f32 * 0.61803398;
        for (j, slot) in d.iter_mut().enumerate() {
            *slot = ((phase + j as f32 * 0.27).sin() * 0.5 + 0.5)
                + ((id * 13 + j * 7) % 5) as f32 * 0.05;
        }
        let norm: f32 = d.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for slot in d.iter_mut() {
                *slot /= norm;
            }
        }
        d
    }

    /// Build a frame's `FeatureSet` + stereo depth by projecting the visible
    /// landmarks through the *true* pose, so PnP recovers the true geometry
    /// regardless of how drifted the VO estimate is.
    fn observe(
        camera: &Camera,
        true_pose: &Pose,
        landmarks: &[(usize, Point3<f64>)],
        desc_dim: usize,
    ) -> (FeatureSet, Vec<StereoFeature>) {
        let mut keypoints = Vec::new();
        let mut descriptors = Vec::new();
        let mut stereo = Vec::new();
        for (id, world) in landmarks {
            let point_cam = true_pose.transform_world_point(world);
            if point_cam.z <= 0.5 {
                continue;
            }
            let Some(pixel) = camera.project(&point_cam) else {
                continue;
            };
            if pixel.x < 0.0
                || pixel.x >= camera.width as f64
                || pixel.y < 0.0
                || pixel.y >= camera.height as f64
            {
                continue;
            }
            let index = keypoints.len();
            keypoints.push(Point2::new(pixel.x, pixel.y));
            descriptors.push(descriptor_for(*id, desc_dim));
            stereo.push(StereoFeature {
                left_index: index,
                right_index: index,
                disparity: 600.0 * 0.5 / point_cam.z,
                point_cam,
            });
        }
        (FeatureSet::new(keypoints, descriptors).unwrap(), stereo)
    }

    fn pose_at(translation: Vector3<f64>, yaw: f64) -> Pose {
        // world_to_camera: rotate then translate. Build from a camera-center +
        // orientation so the geometry is easy to reason about.
        let rotation = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), yaw);
        let world_to_camera_rot = rotation.inverse();
        let translation = world_to_camera_rot * (-translation);
        Pose::from_world_to_camera(world_to_camera_rot, translation)
    }

    fn ate_rmse(estimate: &[Pose], truth: &[Pose]) -> f64 {
        let sum: f64 = estimate
            .iter()
            .zip(truth)
            .map(|(e, t)| {
                (e.camera_center_world().coords - t.camera_center_world().coords).norm_squared()
            })
            .sum();
        (sum / estimate.len() as f64).sqrt()
    }

    #[test]
    fn closes_a_synthetic_square_loop() {
        let camera = camera();
        let desc_dim = 16;

        // A circular loop of 41 frames returning to the start; landmarks scattered
        // ahead so each frame sees a stable subset, and frame 0 / frame 40 (same
        // place) re-see the same landmarks for the loop match.
        let n = 41;
        let mut centers = Vec::with_capacity(n);
        let mut angles = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64; // 0..1 around the loop
            let angle = t * std::f64::consts::TAU;
            let radius = 6.0;
            centers.push(Vector3::new(radius * angle.sin(), 0.0, radius * (1.0 - angle.cos())));
            angles.push(angle);
        }
        let true_poses: Vec<Pose> = centers
            .iter()
            .zip(&angles)
            .map(|(c, &a)| pose_at(*c, a))
            .collect();

        // Landmarks on a ring slightly outside the path so they stay in front of
        // the camera as it goes around.
        let mut landmarks = Vec::new();
        for k in 0..160 {
            let a = k as f64 / 160.0 * std::f64::consts::TAU;
            let r = 9.0;
            landmarks.push((
                k,
                Point3::new(r * a.sin(), (k % 5) as f64 * 0.3 - 0.6, r * (1.0 - a.cos()) + 3.0),
            ));
        }

        let mut left_features = Vec::with_capacity(n);
        let mut stereo_per_frame = Vec::with_capacity(n);
        for true_pose in &true_poses {
            let (features, stereo) = observe(&camera, true_pose, &landmarks, desc_dim);
            left_features.push(features);
            stereo_per_frame.push(stereo);
        }

        // Drifted VO estimate: integrate the *true* frame-to-frame motions with a
        // small systematic per-step error (slight yaw + forward bias) injected
        // into each relative pose. Error accumulates, so the trajectory spirals
        // open and the endpoint no longer coincides with the start — exactly the
        // drift a loop closure removes (a linear position offset would not).
        let mut vo_poses = Vec::with_capacity(n);
        vo_poses.push(true_poses[0].clone());
        let delta = SE3 {
            rotation: UnitQuaternion::from_axis_angle(&Vector3::y_axis(), 0.012),
            translation: Vector3::new(0.0, 0.0, 0.03),
        };
        for i in 1..n {
            let true_rel = relative_world_to_camera(&true_poses[i - 1], &true_poses[i]);
            let drifted_rel = delta.compose(&true_rel);
            let prev = vo_poses[i - 1].world_to_camera.clone();
            vo_poses.push(Pose {
                world_to_camera: drifted_rel.compose(&prev),
            });
        }

        let config = VoLoopClosureConfig {
            min_frame_gap: 20,
            min_similarity: 0.1,
            max_candidates_per_frame: 5,
            vocab_k: 24,
            vocab_descriptor_stride: 1,
            ..VoLoopClosureConfig::default()
        };

        let result =
            close_loops_on_vo_trajectory(&camera, &vo_poses, &left_features, &stereo_per_frame, &config)
                .expect("loop closure runs");

        assert!(
            result.verified_count() >= 1,
            "expected at least one verified loop, got {} (candidates {})",
            result.verified_count(),
            result.candidate_count
        );

        let before = ate_rmse(&vo_poses, &true_poses);
        let after = ate_rmse(&result.refined_poses, &true_poses);
        assert!(
            after < before,
            "PGO should reduce ATE: before {before:.3} m, after {after:.3} m"
        );
    }

    #[test]
    fn no_loop_returns_input_unchanged() {
        let camera = camera();
        let desc_dim = 16;
        let n = 10;
        let mut poses = Vec::new();
        let mut left_features = Vec::new();
        let mut stereo_per_frame = Vec::new();
        let landmarks: Vec<(usize, Point3<f64>)> = (0..40)
            .map(|k| (k, Point3::new((k % 7) as f64 - 3.0, (k % 3) as f64 - 1.0, 8.0)))
            .collect();
        for i in 0..n {
            let pose = pose_at(Vector3::new(0.0, 0.0, i as f64 * 2.0), 0.0);
            let (features, stereo) = observe(&camera, &pose, &landmarks, desc_dim);
            poses.push(pose);
            left_features.push(features);
            stereo_per_frame.push(stereo);
        }
        // Straight line, no revisits: huge min_gap so nothing is proposed.
        let config = VoLoopClosureConfig {
            min_frame_gap: 8,
            min_similarity: 0.99,
            vocab_k: 12,
            vocab_descriptor_stride: 1,
            ..VoLoopClosureConfig::default()
        };
        let result =
            close_loops_on_vo_trajectory(&camera, &poses, &left_features, &stereo_per_frame, &config)
                .expect("runs");
        assert_eq!(result.verified_count(), 0);
        assert!(result.gnc.is_none());
        assert_eq!(result.refined_poses.len(), n);
    }
}
