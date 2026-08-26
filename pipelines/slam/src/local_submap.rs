//! Independently reconstructed local submaps shared by sequential SfM and SLAM.
//!
//! A local submap is rebuilt from verified image correspondences. It never
//! consumes a live tracker pose or a live tracker depth, so its monocular gauge
//! is independent of the state it may later diagnose or correct. This is the
//! required boundary for sound scale-bearing loop measurements: two accepted
//! local submaps may later be aligned with a `Sim3`, while a low-parallax window
//! is rejected here before it can pretend to observe scale.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

use nalgebra::{Point2, Point3};
use visloc_core::geometry::Pose;
use visloc_core::types::Camera;
use visloc_vision::features::FeatureSet;

use crate::{
    incremental_sfm, BaResult, IncrementalSfmConfig, IncrementalSfmError, PairwiseMatches,
    TrackBuildStats,
};

/// Quality gates applied after the independent reconstruction has completed.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapQualityConfig {
    pub min_registered_images: usize,
    pub min_registration_fraction: f64,
    pub min_landmarks: usize,
    pub min_observations: usize,
    pub min_median_track_length: f64,
    pub min_median_max_parallax_deg: f64,
    pub max_mean_reprojection_px: f64,
    /// Minimum fraction of held-out observations explained after rebuilding
    /// their landmark from the other views in the same track.
    pub min_leave_one_out_support_fraction: f64,
    /// Maximum median held-out reprojection error over successful rebuilds.
    pub max_median_leave_one_out_reprojection_px: f64,
    /// Enable the build-time internal scale-sanity gate
    /// (`NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(a)): reject a submap whose own
    /// registered camera-centre trajectory contains one implausibly large
    /// consecutive-frame jump relative to its own typical step size. This is
    /// GT-free and purely a function of one submap's own registered poses —
    /// no neighbour, no cross-submap comparison — exactly like the two gates
    /// it sits beside, and it is the only statistic that caught the
    /// diagnosed MH_03 submap-10 failure: `camera_center_diameter` of
    /// `159,616.9` versus `16.4`/`42.6` for its immediate, identically-seeded
    /// (same seed pair, same seed match count), motion-comparable neighbours,
    /// even though `mean_reprojection_px` (`0.8146`) and
    /// `median_max_parallax_deg` (`4.6577`) were both unremarkable on the
    /// exploded submap — those two measure pixel/angular conditioning, not
    /// absolute 3D scale, and are provably blind to this failure mode. `true`
    /// by default: a single scale-exploded submap is a structural build
    /// defect, not a tuning choice.
    pub scale_pathology_gate_enabled: bool,
    /// Reject the build when `max / median` of the consecutive-registered-
    /// frame camera-centre displacement sequence exceeds this ratio. Default
    /// `30.0`, inside the diagnosis's recommended 20-50x range: comfortably
    /// above ordinary accel/decel variability inside an 88-frame/~4.4s EuRoC
    /// cruise window (the diagnosis's per-10-frame GT trace around the
    /// failure never shows more than a ~2-3x jump between adjacent 10-frame
    /// buckets, and the whole span is a smooth, monotonically-accelerating,
    /// non-kinematically-anomalous flight segment) and comfortably below the
    /// ~3,700x-9,700x per-registered-image proxy the actual failure produced
    /// (`159616.9/88 \u{2248} 1814` vs `16.4/88 \u{2248} 0.19` for one healthy
    /// neighbour, `42.6/88 \u{2248} 0.48` for the other). See
    /// `NOROBUSTFIT_CLUSTER_DIAGNOSIS.md` §6(a).
    pub max_camera_center_displacement_outlier_ratio: f64,
    /// Reject the build when the seed pair's *final* camera-centre distance
    /// (`LocalSubmapQuality::seed_pair_final_distance`) has drifted by more
    /// than this factor from its known seed-time value of exactly `1.0`
    /// (`max(distance, 1.0 / distance)`, symmetric for a blow-up or a
    /// collapse). Empirically calibrated, not from the diagnosis's original
    /// text: the diagnosed submap-10 failure turned out to be a *uniform*
    /// scale explosion (every consecutive camera-centre step inflated
    /// together, not one outlier step), which leaves
    /// `max_camera_center_displacement_outlier_ratio` blind to it by
    /// construction (measured `camera_center_step_max /
    /// camera_center_step_median \u{2248} 1.9` on the actual pathological
    /// submap -- indistinguishable from a healthy submap's own ratio) but
    /// leaves this statistic squarely triggered (measured
    /// `seed_pair_final_distance` orders of magnitude past its `1.0`
    /// seed-time value on the actual pathological submap; see
    /// `hierarchical-scale-pathology` log lines for the exact figure on a
    /// given run). Default `30.0`, the same order of
    /// magnitude headroom as the other gate. Both gates independently feed
    /// the same `ImplausibleScale` rejection -- this one catches a
    /// submap-wide uniform rescale (the diagnosed mechanism: an outlier pose
    /// getting fixed as the monocular BA gauge's scale anchor,
    /// `fix_monocular_scale_gauge` in `incremental_sfm.rs`), the other
    /// catches a single/few-frame local explosion; neither subsumes the
    /// other.
    pub max_seed_pair_scale_drift_ratio: f64,
    /// How many contiguous sub-windows the registered-frame camera-centre
    /// step sequence is split into for
    /// [`LocalSubmapQuality::camera_center_window_drift_ratio`] (each as
    /// close to equal length as possible; `n` steps split as evenly as
    /// `n / window_count` allows). Default `4`. `0` or `1` disables this
    /// specific statistic (reported as `0.0`, never flagged) since a single
    /// window has nothing to compare against.
    pub camera_center_drift_window_count: usize,
    /// Reject the build when `max / min` of the *per-sub-window median*
    /// camera-centre step (see `camera_center_drift_window_count`) exceeds
    /// this ratio. This is the follow-on to the `LowInlierRatio` seam-cluster
    /// diagnosis after the scale-sanity gate above eliminated the
    /// `NoRobustFit` cluster: submaps 9 and 13 in the MH_03 `1100..1500`
    /// subrange each have a genuine *within-window* scale drift (confirmed
    /// by comparing each submap's own camera-centre step shape, restricted
    /// to the frames it shares with a neighbour, self-normalized by its own
    /// median step over that span, against the same statistic computed for
    /// that neighbour -- submap 9: `2.0-2.3x`, submap 13: `3.5-5.1x`,
    /// corroborated by two independent seam comparisons each) too mild to
    /// trip either whole-submap-aggregate gate above
    /// (`camera_center_step_outlier_ratio`, blind to it because the drift is
    /// gradual across many steps, not one outlier; `seed_pair_final_distance`
    /// only `0.70`/`0.33`, nowhere near either gate's `30.0` threshold) but
    /// still a real, GT-free-detectable defect: `mean_reprojection_px` is
    /// also visibly elevated on both submaps (`1.137`, `1.142` vs. `~0.7-0.9`
    /// typical), an independent corroborating signal. This statistic makes
    /// the same "internal scale self-consistency" measurement as the other
    /// two, just at sub-window granularity instead of single-step or
    /// whole-submap. Calibrated against the actual `subrange_1100_1500`
    /// data (all 20 submaps, `camera_center_drift_window_count = 4`); see
    /// `LOWINLIERRATIO_DIAGNOSIS.md` and the windowed-drift-gate follow-on
    /// evidence directory for the calibration table this default was chosen
    /// from.
    pub max_camera_center_window_drift_ratio: f64,
    /// Bound on the cheap same-window seed retry (§6(b)): before declaring a
    /// window's build failed for `ImplausibleScale` specifically, retry with
    /// the next seed candidate from `seed_candidate_order`'s deterministic
    /// descending-match-count order, excluding every seed pair already tried
    /// this build. `0` disables the retry (the first pathological result
    /// fails immediately, still routed to the widen/merge machinery like
    /// `NoSeedPair`). This is *not* a substitute for the gate itself: if the
    /// underlying defect is tied to a frame/correspondence combination
    /// reachable from several seeds rather than to seed choice, retries burn
    /// CPU without resolving anything and the window falls through to the
    /// gate's normal widen/merge path anyway once retries are exhausted.
    pub max_scale_pathology_seed_retries: usize,
}

impl Default for LocalSubmapQualityConfig {
    fn default() -> Self {
        Self {
            min_registered_images: 3,
            min_registration_fraction: 0.75,
            min_landmarks: 30,
            min_observations: 90,
            min_median_track_length: 3.0,
            min_median_max_parallax_deg: 2.0,
            max_mean_reprojection_px: 2.0,
            min_leave_one_out_support_fraction: 0.5,
            max_median_leave_one_out_reprojection_px: 2.0,
            scale_pathology_gate_enabled: true,
            max_camera_center_displacement_outlier_ratio: 30.0,
            max_seed_pair_scale_drift_ratio: 30.0,
            camera_center_drift_window_count: 4,
            // Calibration placeholder -- effectively disabled (no real
            // submap should ever reach 1000x) until the calibration run
            // against `subrange_1100_1500` picks a real value with margin.
            // See the windowed-drift-gate evidence directory.
            max_camera_center_window_drift_ratio: 1000.0,
            max_scale_pathology_seed_retries: 3,
        }
    }
}

/// Construction and acceptance policy for [`LocalSubmapBuilder`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalSubmapConfig {
    pub sfm: IncrementalSfmConfig,
    pub quality: LocalSubmapQualityConfig,
}

/// A registered frame expressed in this submap's independent local gauge.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapFrame {
    pub local_frame_index: usize,
    pub source_frame_id: u64,
    pub pose: Pose,
}

/// One landmark observation, retaining both local and source frame identity.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapObservation {
    pub local_frame_index: usize,
    pub source_frame_id: u64,
    pub keypoint_index: usize,
    pub pixel: Point2<f64>,
}

/// A multi-view point reconstructed only from this submap's image evidence.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalSubmapLandmark {
    pub local_landmark_id: u64,
    pub position: Point3<f64>,
    pub observations: Vec<LocalSubmapObservation>,
}

/// Auditable uncertainty/conditioning proxy for an independently built submap.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalSubmapQuality {
    pub requested_images: usize,
    pub registered_images: usize,
    pub registration_fraction: f64,
    pub landmarks: usize,
    pub observations: usize,
    pub median_track_length: f64,
    /// Median, over landmarks, of the widest viewing-ray angle in that track.
    pub median_max_parallax_deg: f64,
    /// Largest distance between reconstructed camera centres in local gauge.
    pub camera_center_diameter: f64,
    /// Median, over consecutive *registered* frames (in temporal/index
    /// order, gaps from unregistered frames skipped), of the camera-centre
    /// displacement between one frame and the next. Robust to one outlier
    /// step, unlike `camera_center_diameter` (a single max-over-all-pairs
    /// statistic already dominated by whatever pair is farthest apart).
    pub camera_center_step_median: f64,
    /// Maximum consecutive-registered-frame camera-centre displacement.
    /// `camera_center_step_max / camera_center_step_median` is the ratio the
    /// scale-sanity gate thresholds; see
    /// [`LocalSubmapQualityConfig::max_camera_center_displacement_outlier_ratio`].
    pub camera_center_step_max: f64,
    /// Distance between the winning seed pair's two camera centres in the
    /// *final* (post-growth, post-BA) reconstruction. The seed is always
    /// bootstrapped from a unit-normalized two-view translation (see
    /// `place_seed_pair`, `incremental_sfm.rs`), so this is `1.0` exactly at
    /// seed time; how far it has since drifted from `1.0` is a scale-drift
    /// signal that stays discriminating even when every *other* pairwise
    /// statistic among the current poses has scaled together and lost the
    /// signal. `NaN` if either seed image is (unexpectedly) unregistered in
    /// the final result.
    pub seed_pair_final_distance: f64,
    /// `max / min` of the per-sub-window median camera-centre step, where
    /// the registered-frame step sequence is split into
    /// [`LocalSubmapQualityConfig::camera_center_drift_window_count`]
    /// contiguous sub-windows. Unlike `camera_center_step_median`/`_max`
    /// (dominated by a single outlier step) or `seed_pair_final_distance`
    /// (a single-point-in-time reference), this catches a *gradual* drift
    /// spread across many consecutive steps -- the diagnosed submap 9/13
    /// `LowInlierRatio` mechanism -- by comparing coarser-grained chunks of
    /// the trajectory against each other. `0.0` when
    /// `camera_center_drift_window_count` is `0`/`1` or there are fewer
    /// steps than windows (nothing to compare).
    pub camera_center_window_drift_ratio: f64,
    pub mean_reprojection_px: f64,
    /// Number of observations in tracks long enough for leave-one-view-out.
    pub leave_one_out_attempts: usize,
    /// Attempts whose remaining views triangulate a point that explains the
    /// held-out pixel within the mapper's reprojection gate.
    pub leave_one_out_supported: usize,
    pub leave_one_out_support_fraction: f64,
    pub median_leave_one_out_reprojection_px: f64,
}

/// A reconstruction with no inherited pose/depth state from the live tracker.
#[derive(Debug, Clone)]
pub struct LocalSubmap {
    pub camera: Camera,
    /// Full input mapping, including frames that failed registration.
    pub source_frame_ids: Vec<u64>,
    pub frames: Vec<LocalSubmapFrame>,
    pub landmarks: Vec<LocalSubmapLandmark>,
    pub quality: LocalSubmapQuality,
    pub track_build_stats: TrackBuildStats,
    pub ba_result: Option<BaResult>,
    /// `source_frame_id` of the first image of the seed pair the winning
    /// growth trial started from (see [`crate::IncrementalSfmResult::seed_image_i`]).
    /// Purely observational.
    pub seed_source_frame_i: u64,
    /// `source_frame_id` of the second image of the winning seed pair.
    pub seed_source_frame_j: u64,
    /// Number of verified matches in the winning seed pair.
    pub seed_match_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LocalSubmapRejectionReason {
    TooFewRegisteredImages,
    LowRegistrationFraction,
    TooFewLandmarks,
    TooFewObservations,
    ShortTracks,
    LowParallax,
    HighReprojectionError,
    InsufficientLeaveOneOutSupport,
    HighLeaveOneOutReprojection,
    NonFiniteGeometry,
    /// Build-time scale-sanity gate (`NOROBUSTFIT_CLUSTER_DIAGNOSIS.md`
    /// §6(a)): the submap's own registered camera-centre trajectory contains
    /// a consecutive-frame displacement far larger than its typical step,
    /// the signature of an internal scale explosion invisible to the
    /// reprojection/parallax gates.
    ImplausibleScale,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LocalSubmapBuildError {
    SourceFrameCountMismatch {
        ids: usize,
        features: usize,
    },
    DuplicateSourceFrameId(u64),
    InvalidPairImageIndex {
        pair_index: usize,
        image_index: usize,
        image_count: usize,
    },
    SelfPair {
        pair_index: usize,
        image_index: usize,
    },
    InvalidMatchKeypointIndex {
        pair_index: usize,
        image_index: usize,
        keypoint_index: usize,
        keypoint_count: usize,
    },
    Reconstruction(IncrementalSfmError),
    QualityRejected {
        reason: LocalSubmapRejectionReason,
        quality: LocalSubmapQuality,
    },
}

impl fmt::Display for LocalSubmapBuildError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceFrameCountMismatch { ids, features } => {
                write!(f, "source frame id count {ids} != feature count {features}")
            }
            Self::DuplicateSourceFrameId(id) => write!(f, "duplicate source frame id {id}"),
            Self::InvalidPairImageIndex {
                pair_index,
                image_index,
                image_count,
            } => write!(
                f,
                "pair {pair_index} references image {image_index}; image count is {image_count}"
            ),
            Self::SelfPair {
                pair_index,
                image_index,
            } => write!(f, "pair {pair_index} references image {image_index} twice"),
            Self::InvalidMatchKeypointIndex {
                pair_index,
                image_index,
                keypoint_index,
                keypoint_count,
            } => write!(
                f,
                "pair {pair_index} references keypoint {keypoint_index} in image {image_index}; keypoint count is {keypoint_count}"
            ),
            Self::Reconstruction(error) => write!(f, "local reconstruction failed: {error}"),
            Self::QualityRejected { reason, .. } => {
                write!(f, "local reconstruction rejected by {reason:?}")
            }
        }
    }
}

impl Error for LocalSubmapBuildError {}

impl From<IncrementalSfmError> for LocalSubmapBuildError {
    fn from(value: IncrementalSfmError) -> Self {
        Self::Reconstruction(value)
    }
}

/// Stateless builder; all gauge and geometric state belongs to each output.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LocalSubmapBuilder {
    pub config: LocalSubmapConfig,
}

impl LocalSubmapBuilder {
    pub fn new(config: LocalSubmapConfig) -> Self {
        Self { config }
    }

    /// Reconstruct a local map exclusively from `features` and pre-verified
    /// `pairwise` correspondences. `source_frame_ids` only preserve identity;
    /// their numeric values never influence geometry or selection.
    #[allow(clippy::result_large_err)]
    pub fn build(
        &self,
        camera: &Camera,
        source_frame_ids: &[u64],
        features: &[FeatureSet],
        pairwise: &[PairwiseMatches],
    ) -> Result<LocalSubmap, LocalSubmapBuildError> {
        validate_inputs(source_frame_ids, features, pairwise)?;

        // (b) cheap same-window seed retry (NOROBUSTFIT_CLUSTER_DIAGNOSIS.md
        // §6(b)): a build that trips the (a) scale-sanity gate specifically
        // is retried, up to `max_scale_pathology_seed_retries` times, on the
        // *next* seed candidate in `seed_candidate_order`'s deterministic
        // descending-match-count order rather than being declared a failure
        // immediately. Every other rejection reason (including a plain
        // `IncrementalSfmError`, e.g. `NoSeedPair`) is unaffected and still
        // fails on the first attempt. Bounded and deterministic: each retry
        // excludes every seed pair already tried this call, so the sequence
        // of candidates tried is fixed by the input alone.
        let max_retries = self.config.quality.max_scale_pathology_seed_retries;
        let mut sfm_config = self.config.sfm.clone();
        let mut attempt = 0usize;
        let (result, camera, quality, reason) = loop {
            let result = incremental_sfm(camera, features, pairwise, &sfm_config)?;
            let refined_camera = result
                .refined_camera
                .clone()
                .unwrap_or_else(|| camera.clone());
            let quality = measure_quality(
                features.len(),
                &refined_camera,
                &result.poses,
                &result.tracks,
                result.mean_reprojection_px,
                &self.config.sfm,
                result.seed_image_i,
                result.seed_image_j,
                self.config.quality.camera_center_drift_window_count,
            );
            let reason = rejection_reason(&quality, &self.config.quality);
            if reason == Some(LocalSubmapRejectionReason::ImplausibleScale) && attempt < max_retries
            {
                attempt += 1;
                let step_ratio = camera_center_step_outlier_ratio(&quality);
                let seed_drift_ratio = seed_pair_scale_drift_ratio(&quality);
                eprintln!(
                    "hierarchical-scale-pathology-retry: seed=({}, {}) attempt={}/{} \
                     diameter={:.4} median_step={:.6} max_step={:.6} step_ratio={:.4} \
                     step_threshold={:.4} seed_pair_final_distance={:.6} \
                     seed_drift_ratio={:.4} seed_drift_threshold={:.4}; \
                     retrying with next seed candidate",
                    result.seed_image_i,
                    result.seed_image_j,
                    attempt,
                    max_retries,
                    quality.camera_center_diameter,
                    quality.camera_center_step_median,
                    quality.camera_center_step_max,
                    step_ratio,
                    self.config
                        .quality
                        .max_camera_center_displacement_outlier_ratio,
                    quality.seed_pair_final_distance,
                    seed_drift_ratio,
                    self.config.quality.max_seed_pair_scale_drift_ratio,
                );
                if quality.camera_center_window_drift_ratio
                    > self.config.quality.max_camera_center_window_drift_ratio
                {
                    eprintln!(
                        "hierarchical-scale-drift-retry: seed=({}, {}) attempt={}/{} \
                         window_count={} window_drift_ratio={:.4} window_drift_threshold={:.4}; \
                         retrying with next seed candidate",
                        result.seed_image_i,
                        result.seed_image_j,
                        attempt,
                        max_retries,
                        self.config.quality.camera_center_drift_window_count,
                        quality.camera_center_window_drift_ratio,
                        self.config.quality.max_camera_center_window_drift_ratio,
                    );
                }
                let key = (
                    result.seed_image_i.min(result.seed_image_j),
                    result.seed_image_i.max(result.seed_image_j),
                );
                sfm_config.excluded_seed_pairs.insert(key);
                continue;
            }
            break (result, refined_camera, quality, reason);
        };
        if let Some(reason) = reason {
            return Err(LocalSubmapBuildError::QualityRejected { reason, quality });
        }

        let frames = result
            .poses
            .iter()
            .enumerate()
            .filter_map(|(local_frame_index, pose)| {
                pose.clone().map(|pose| LocalSubmapFrame {
                    local_frame_index,
                    source_frame_id: source_frame_ids[local_frame_index],
                    pose,
                })
            })
            .collect();
        let landmarks = result
            .tracks
            .iter()
            .enumerate()
            .map(|(local_landmark_id, track)| LocalSubmapLandmark {
                local_landmark_id: local_landmark_id as u64,
                position: track.position,
                observations: track
                    .observations
                    .iter()
                    .map(
                        |&(local_frame_index, keypoint_index, pixel)| LocalSubmapObservation {
                            local_frame_index,
                            source_frame_id: source_frame_ids[local_frame_index],
                            keypoint_index,
                            pixel,
                        },
                    )
                    .collect(),
            })
            .collect();

        let seed_source_frame_i = source_frame_ids
            .get(result.seed_image_i)
            .copied()
            .unwrap_or(0);
        let seed_source_frame_j = source_frame_ids
            .get(result.seed_image_j)
            .copied()
            .unwrap_or(0);
        Ok(LocalSubmap {
            camera,
            source_frame_ids: source_frame_ids.to_vec(),
            frames,
            landmarks,
            quality,
            track_build_stats: result.track_build_stats,
            ba_result: result.ba_result,
            seed_source_frame_i,
            seed_source_frame_j,
            seed_match_count: result.seed_match_count,
        })
    }
}

#[allow(clippy::result_large_err)]
fn validate_inputs(
    source_frame_ids: &[u64],
    features: &[FeatureSet],
    pairwise: &[PairwiseMatches],
) -> Result<(), LocalSubmapBuildError> {
    if source_frame_ids.len() != features.len() {
        return Err(LocalSubmapBuildError::SourceFrameCountMismatch {
            ids: source_frame_ids.len(),
            features: features.len(),
        });
    }
    let mut unique_ids = HashSet::with_capacity(source_frame_ids.len());
    for &id in source_frame_ids {
        if !unique_ids.insert(id) {
            return Err(LocalSubmapBuildError::DuplicateSourceFrameId(id));
        }
    }
    #[allow(clippy::result_large_err)]
    for (pair_index, pair) in pairwise.iter().enumerate() {
        for image_index in [pair.image_i, pair.image_j] {
            if image_index >= features.len() {
                return Err(LocalSubmapBuildError::InvalidPairImageIndex {
                    pair_index,
                    image_index,
                    image_count: features.len(),
                });
            }
        }
        if pair.image_i == pair.image_j {
            return Err(LocalSubmapBuildError::SelfPair {
                pair_index,
                image_index: pair.image_i,
            });
        }
        for &(keypoint_i, keypoint_j) in &pair.matches {
            for (image_index, keypoint_index) in
                [(pair.image_i, keypoint_i), (pair.image_j, keypoint_j)]
            {
                let keypoint_count = features[image_index].keypoints.len();
                if keypoint_index >= keypoint_count {
                    return Err(LocalSubmapBuildError::InvalidMatchKeypointIndex {
                        pair_index,
                        image_index,
                        keypoint_index,
                        keypoint_count,
                    });
                }
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn measure_quality(
    requested_images: usize,
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[crate::SfmTrack],
    mean_reprojection_px: f64,
    sfm_config: &IncrementalSfmConfig,
    seed_image_i: usize,
    seed_image_j: usize,
    drift_window_count: usize,
) -> LocalSubmapQuality {
    let registered_images = poses.iter().filter(|pose| pose.is_some()).count();
    #[allow(clippy::too_many_arguments)]
    let registration_fraction = if requested_images == 0 {
        0.0
    } else {
        registered_images as f64 / requested_images as f64
    };
    let observations = tracks.iter().map(|track| track.observations.len()).sum();
    let median_track_length = median(
        tracks
            .iter()
            .map(|track| track.observations.len() as f64)
            .collect(),
    );
    let median_max_parallax_deg = median(
        tracks
            .iter()
            .filter_map(|track| track_max_parallax_deg(track, poses))
            .collect(),
    );
    let centres = poses
        .iter()
        .filter_map(|pose| pose.as_ref().map(Pose::camera_center_world))
        .collect::<Vec<_>>();
    let mut camera_center_diameter: f64 = 0.0;
    for i in 0..centres.len() {
        for j in (i + 1)..centres.len() {
            camera_center_diameter = camera_center_diameter.max((centres[i] - centres[j]).norm());
        }
    }
    // Consecutive-registered-frame displacement sequence for the scale-sanity
    // gate (NOROBUSTFIT_CLUSTER_DIAGNOSIS.md §6(a)). `centres` is already in
    // registration/temporal order (poses are filtered in index order above),
    // so adjacent entries here are genuinely temporally adjacent registered
    // frames, not an arbitrary pairing.
    let camera_center_steps: Vec<f64> = centres
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).norm())
        .collect();
    let camera_center_step_median = median(camera_center_steps.clone());
    let camera_center_step_max = camera_center_steps.iter().copied().fold(0.0_f64, f64::max);
    // Windowed drift statistic (LowInlierRatio seam-cluster follow-on): split
    // the step sequence into `drift_window_count` contiguous, near-equal
    // chunks and compare their medians. A gradual drift spread across many
    // steps (the diagnosed submap 9/13 mechanism) shifts an entire chunk's
    // median together, which the single-step max/median statistic above
    // cannot see (it is dominated by whichever single step is largest, not
    // by a sustained shift) but this can.
    let camera_center_window_drift_ratio =
        windowed_camera_center_drift_ratio(&camera_center_steps, drift_window_count);
    // Seed-pair scale-drift statistic. `place_seed_pair`
    // (`incremental_sfm.rs`) bootstraps every reconstruction's gauge from a
    // `RelativePoseEstimator::default()` two-view decomposition, whose
    // translation is *always* unit-normalized
    // (`default_translation_scale = 1.0`, `crates/vision/src/two_view/mod.rs`)
    // with the seed's first image placed at the world origin — so the
    // distance between the two seed images' camera centres is *exactly*
    // `1.0` at the moment the seed is placed, deterministically, regardless
    // of the scene. Unlike `camera_center_step_median`/`_max` (which the
    // diagnosed submap-10 failure showed can *both* explode together, keeping
    // their ratio scale-invariant and blind to a uniform blow-up: measured
    // `median=2056.5`, `max=3969.4`, ratio `~1.9` on the actual pathological
    // submap -- statistically indistinguishable from a healthy submap's own
    // ratio), this is a true absolute reference baked into the code, not
    // just another quantity computed from the same (possibly corrupted)
    // pose set. If subsequent growth + bundle adjustment drastically
    // rescales the reconstruction -- e.g. by fixing an outlier pose as the
    // BA gauge's "farthest/scale anchor" (`fix_monocular_scale_gauge`,
    // `incremental_sfm.rs`) -- the seed pair's own *final* distance drifts
    // far from that known `1.0` starting point even though every pairwise
    // ratio measured purely among the *current* poses stays unremarkable.
    let seed_pair_final_distance = match (poses.get(seed_image_i), poses.get(seed_image_j)) {
        (Some(Some(pose_i)), Some(Some(pose_j))) => {
            (pose_i.camera_center_world() - pose_j.camera_center_world()).norm()
        }
        _ => f64::NAN,
    };
    let (leave_one_out_attempts, leave_one_out_supported, leave_one_out_errors) =
        leave_one_out_support(camera, poses, tracks, sfm_config);
    let leave_one_out_support_fraction = if leave_one_out_attempts == 0 {
        0.0
    } else {
        leave_one_out_supported as f64 / leave_one_out_attempts as f64
    };
    let median_leave_one_out_reprojection_px = median(leave_one_out_errors);
    LocalSubmapQuality {
        requested_images,
        registered_images,
        registration_fraction,
        landmarks: tracks.len(),
        observations,
        median_track_length,
        median_max_parallax_deg,
        camera_center_diameter,
        camera_center_step_median,
        camera_center_step_max,
        seed_pair_final_distance,
        camera_center_window_drift_ratio,
        mean_reprojection_px,
        leave_one_out_attempts,
        leave_one_out_supported,
        leave_one_out_support_fraction,
        median_leave_one_out_reprojection_px,
    }
}

fn leave_one_out_support(
    camera: &Camera,
    poses: &[Option<Pose>],
    tracks: &[crate::SfmTrack],
    config: &IncrementalSfmConfig,
) -> (usize, usize, Vec<f64>) {
    let mut attempts = 0;
    let mut supported = 0;
    let mut errors = Vec::new();
    for track in tracks.iter().filter(|track| track.observations.len() >= 3) {
        for held_out in 0..track.observations.len() {
            attempts += 1;
            let remaining = track
                .observations
                .iter()
                .enumerate()
                .filter_map(|(index, &(image, _, pixel))| {
                    (index != held_out).then_some((image, pixel))
                })
                .collect::<Vec<_>>();
            let Some(point) = incremental_sfm::triangulate_track(camera, poses, &remaining, config)
            else {
                continue;
            };
            let (image, _, pixel) = track.observations[held_out];
            let Some(pose) = poses.get(image).and_then(Option::as_ref) else {
                continue;
            };
            let Some(error) = incremental_sfm::reprojection_error_px(camera, pose, &point, &pixel)
            else {
                continue;
            };
            errors.push(error);
            if error <= config.max_reprojection_error_px {
                supported += 1;
            }
        }
    }
    (attempts, supported, errors)
}

fn track_max_parallax_deg(track: &crate::SfmTrack, poses: &[Option<Pose>]) -> Option<f64> {
    let rays = track
        .observations
        .iter()
        .filter_map(|(image_index, _, _)| {
            let centre = poses.get(*image_index)?.as_ref()?.camera_center_world();
            (track.position - centre).try_normalize(1.0e-12)
        })
        .collect::<Vec<_>>();
    if rays.len() < 2 {
        return None;
    }
    let mut max_angle: f64 = 0.0;
    for i in 0..rays.len() {
        for j in (i + 1)..rays.len() {
            max_angle = max_angle.max(rays[i].dot(&rays[j]).clamp(-1.0, 1.0).acos());
        }
    }
    Some(max_angle.to_degrees())
}

fn median(mut values: Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn rejection_reason(
    quality: &LocalSubmapQuality,
    config: &LocalSubmapQualityConfig,
) -> Option<LocalSubmapRejectionReason> {
    if !quality.registration_fraction.is_finite()
        || !quality.median_track_length.is_finite()
        || !quality.median_max_parallax_deg.is_finite()
        || !quality.camera_center_diameter.is_finite()
        || !quality.camera_center_step_median.is_finite()
        || !quality.camera_center_step_max.is_finite()
        || !quality.mean_reprojection_px.is_finite()
        || !quality.leave_one_out_support_fraction.is_finite()
        || !quality.median_leave_one_out_reprojection_px.is_finite()
    {
        return Some(LocalSubmapRejectionReason::NonFiniteGeometry);
    }
    if quality.registered_images < config.min_registered_images {
        return Some(LocalSubmapRejectionReason::TooFewRegisteredImages);
    }
    if quality.registration_fraction < config.min_registration_fraction {
        return Some(LocalSubmapRejectionReason::LowRegistrationFraction);
    }
    if quality.landmarks < config.min_landmarks {
        return Some(LocalSubmapRejectionReason::TooFewLandmarks);
    }
    if quality.observations < config.min_observations {
        return Some(LocalSubmapRejectionReason::TooFewObservations);
    }
    if quality.median_track_length < config.min_median_track_length {
        return Some(LocalSubmapRejectionReason::ShortTracks);
    }
    if quality.median_max_parallax_deg < config.min_median_max_parallax_deg {
        return Some(LocalSubmapRejectionReason::LowParallax);
    }
    if quality.mean_reprojection_px > config.max_mean_reprojection_px {
        return Some(LocalSubmapRejectionReason::HighReprojectionError);
    }
    if quality.leave_one_out_support_fraction < config.min_leave_one_out_support_fraction {
        return Some(LocalSubmapRejectionReason::InsufficientLeaveOneOutSupport);
    }
    if quality.median_leave_one_out_reprojection_px
        > config.max_median_leave_one_out_reprojection_px
    {
        return Some(LocalSubmapRejectionReason::HighLeaveOneOutReprojection);
    }
    if config.scale_pathology_gate_enabled
        && (seed_pair_scale_drift_ratio(quality) > config.max_seed_pair_scale_drift_ratio
            || quality.camera_center_window_drift_ratio
                > config.max_camera_center_window_drift_ratio)
    {
        return Some(LocalSubmapRejectionReason::ImplausibleScale);
    }
    None
}

/// `max / median` of the consecutive-registered-frame camera-centre
/// displacement sequence — the statistic
/// [`LocalSubmapQualityConfig::max_camera_center_displacement_outlier_ratio`]
/// thresholds. A zero median with a nonzero max (every step but one is
/// exactly zero, e.g. a degenerate duplicated pose) is treated as an
/// unbounded ratio rather than a division artifact; a zero max (a
/// perfectly static or single-step submap, already caught by the
/// registered-image/parallax gates) is `0.0`, never flagged.
pub(crate) fn camera_center_step_outlier_ratio(quality: &LocalSubmapQuality) -> f64 {
    if quality.camera_center_step_median > 1.0e-9 {
        quality.camera_center_step_max / quality.camera_center_step_median
    } else if quality.camera_center_step_max > 1.0e-9 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// How far the seed pair's final camera-centre distance has drifted from its
/// known seed-time value of exactly `1.0` -- the statistic
/// [`LocalSubmapQualityConfig::max_seed_pair_scale_drift_ratio`] thresholds.
/// Symmetric: `distance = 4.0` and `distance = 0.25` both report `4.0`, since
/// either a blow-up or a collapse is an equally implausible scale drift from
/// the known starting point. `NaN`/non-finite `seed_pair_final_distance`
/// (the seed images unexpectedly missing from the final result) reports
/// infinity rather than silently passing.
pub(crate) fn seed_pair_scale_drift_ratio(quality: &LocalSubmapQuality) -> f64 {
    let distance = quality.seed_pair_final_distance;
    if !distance.is_finite() || distance <= 0.0 {
        return f64::INFINITY;
    }
    distance.max(1.0 / distance)
}

/// Split `steps` into `window_count` contiguous, near-equal chunks (the
/// first `steps.len() % window_count` chunks get one extra element), take
/// each chunk's median, and return `max(chunk medians) / min(chunk medians)`.
/// `0.0` if `window_count < 2` or there are fewer steps than windows (an
/// empty chunk would be meaningless to compare). A zero minimum with a
/// nonzero maximum reports infinity, mirroring
/// [`camera_center_step_outlier_ratio`]'s zero-division handling.
pub(crate) fn windowed_camera_center_drift_ratio(steps: &[f64], window_count: usize) -> f64 {
    if window_count < 2 || steps.len() < window_count {
        return 0.0;
    }
    let base = steps.len() / window_count;
    let remainder = steps.len() % window_count;
    let mut chunk_medians = Vec::with_capacity(window_count);
    let mut start = 0;
    for window_index in 0..window_count {
        let size = base + usize::from(window_index < remainder);
        let end = start + size;
        chunk_medians.push(median(steps[start..end].to_vec()));
        start = end;
    }
    let max_median = chunk_medians.iter().copied().fold(0.0_f64, f64::max);
    let min_median = chunk_medians.iter().copied().fold(f64::INFINITY, f64::min);
    if min_median > 1.0e-9 {
        max_median / min_median
    } else if max_median > 1.0e-9 {
        f64::INFINITY
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use nalgebra::{Point3, UnitQuaternion, Vector3};

    struct Scene {
        camera: Camera,
        points: Vec<Point3<f64>>,
        poses: Vec<Pose>,
    }

    fn scene() -> Scene {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut points = Vec::new();
        for x in -2..=2 {
            for y in -2..=2 {
                for z in 0..=2 {
                    points.push(Point3::new(x as f64 * 0.3, y as f64 * 0.3, z as f64 * 0.3));
                }
            }
        }
        let mut poses = Vec::new();
        for index in 0..6 {
            let angle = -0.5 + index as f64 * 0.2;
            let centre = Point3::new(3.0 * angle.sin(), 0.0, -3.0 * angle.cos());
            let forward = (Point3::origin() - centre).normalize();
            let right = forward.cross(&Vector3::y()).normalize();
            let up = right.cross(&forward);
            let camera_to_world = nalgebra::Matrix3::from_columns(&[right, -up, forward]);
            let rotation = nalgebra::Rotation3::from_matrix_unchecked(camera_to_world);
            let world_to_camera = UnitQuaternion::from_rotation_matrix(&rotation).inverse();
            let translation = -(world_to_camera * centre.coords);
            poses.push(Pose::from_world_to_camera(world_to_camera, translation));
        }
        Scene {
            camera,
            points,
            poses,
        }
    }

    fn render(scene: &Scene) -> (Vec<FeatureSet>, Vec<PairwiseMatches>) {
        let mut features = Vec::new();
        let mut visible = Vec::new();
        for pose in &scene.poses {
            let mut keypoints = Vec::new();
            let mut descriptors = Vec::new();
            let mut point_to_keypoint = HashMap::new();
            for (point_index, point) in scene.points.iter().enumerate() {
                let camera_point = pose.transform_world_point(point);
                if camera_point.z <= 0.05 {
                    continue;
                }
                let Some(pixel) = scene.camera.project(&camera_point) else {
                    continue;
                };
                if pixel.x < 0.0
                    || pixel.x >= scene.camera.width as f64
                    || pixel.y < 0.0
                    || pixel.y >= scene.camera.height as f64
                {
                    continue;
                }
                point_to_keypoint.insert(point_index, keypoints.len());
                keypoints.push(pixel);
                descriptors.push(vec![point_index as f32, 1.0, 0.0, 0.0]);
            }
            features.push(FeatureSet::new(keypoints, descriptors).unwrap());
            visible.push(point_to_keypoint);
        }
        let mut pairwise = Vec::new();
        for image_i in 0..features.len() {
            for image_j in (image_i + 1)..features.len() {
                let matches = visible[image_i]
                    .iter()
                    .filter_map(|(point, keypoint_i)| {
                        visible[image_j]
                            .get(point)
                            .map(|keypoint_j| (*keypoint_i, *keypoint_j))
                    })
                    .collect::<Vec<_>>();
                if matches.len() >= 8 {
                    pairwise.push(PairwiseMatches {
                        image_i,
                        image_j,
                        matches,
                    });
                }
            }
        }
        (features, pairwise)
    }

    #[test]
    fn independently_builds_a_well_conditioned_multiview_submap() {
        let scene = scene();
        let (features, pairwise) = render(&scene);
        let frame_ids = (100..106).collect::<Vec<_>>();
        let config = LocalSubmapConfig {
            sfm: IncrementalSfmConfig {
                min_seed_matches: 8,
                ba_every: 0,
                ..IncrementalSfmConfig::default()
            },
            quality: LocalSubmapQualityConfig {
                min_registered_images: 6,
                min_registration_fraction: 1.0,
                min_landmarks: 30,
                min_observations: 180,
                min_median_track_length: 3.0,
                min_median_max_parallax_deg: 2.0,
                max_mean_reprojection_px: 0.1,
                min_leave_one_out_support_fraction: 0.5,
                max_median_leave_one_out_reprojection_px: 0.1,
                ..LocalSubmapQualityConfig::default()
            },
        };
        let submap = LocalSubmapBuilder::new(config)
            .build(&scene.camera, &frame_ids, &features, &pairwise)
            .expect("synthetic multiview scene should build an independent submap");

        assert_eq!(submap.frames.len(), 6);
        assert_eq!(submap.quality.registered_images, 6);
        assert!(submap.quality.landmarks >= 30);
        assert!(submap.quality.observations >= 180);
        assert!(submap.quality.median_max_parallax_deg >= 2.0);
        assert!(submap.quality.mean_reprojection_px < 0.1);
        assert!(submap.quality.leave_one_out_attempts >= 180);
        assert!(submap.quality.leave_one_out_support_fraction >= 0.5);
        assert!(submap.quality.median_leave_one_out_reprojection_px < 0.1);
        assert_eq!(submap.frames[0].source_frame_id, 100);
        assert!(submap
            .landmarks
            .iter()
            .flat_map(|landmark| &landmark.observations)
            .all(|observation| (100..106).contains(&observation.source_frame_id)));
    }

    #[test]
    fn source_ids_are_identity_only_but_must_be_unique() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let features = vec![
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
        ];
        let error = LocalSubmapBuilder::default()
            .build(&camera, &[7, 7], &features, &[])
            .unwrap_err();
        assert_eq!(error, LocalSubmapBuildError::DuplicateSourceFrameId(7));
    }

    #[test]
    fn invalid_pair_indices_are_rejected_before_mapper_indexing() {
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let features = vec![
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
            FeatureSet::new(Vec::new(), Vec::new()).unwrap(),
        ];
        let pairwise = vec![PairwiseMatches {
            image_i: 0,
            image_j: 2,
            matches: Vec::new(),
        }];
        let error = LocalSubmapBuilder::default()
            .build(&camera, &[7, 8], &features, &pairwise)
            .unwrap_err();
        assert_eq!(
            error,
            LocalSubmapBuildError::InvalidPairImageIndex {
                pair_index: 0,
                image_index: 2,
                image_count: 2,
            }
        );
    }

    #[test]
    fn quality_gate_reports_one_deterministic_reason() {
        let quality = LocalSubmapQuality {
            requested_images: 4,
            registered_images: 2,
            registration_fraction: 0.5,
            landmarks: 0,
            observations: 0,
            median_track_length: 0.0,
            median_max_parallax_deg: 0.0,
            camera_center_diameter: 0.0,
            camera_center_step_median: 0.0,
            camera_center_step_max: 0.0,
            seed_pair_final_distance: 0.0,
            camera_center_window_drift_ratio: 0.0,
            mean_reprojection_px: 0.0,
            leave_one_out_attempts: 0,
            leave_one_out_supported: 0,
            leave_one_out_support_fraction: 0.0,
            median_leave_one_out_reprojection_px: 0.0,
        };
        assert_eq!(
            rejection_reason(&quality, &LocalSubmapQualityConfig::default()),
            Some(LocalSubmapRejectionReason::TooFewRegisteredImages)
        );

        let otherwise_valid = LocalSubmapQuality {
            requested_images: 4,
            registered_images: 4,
            registration_fraction: 1.0,
            landmarks: 40,
            observations: 120,
            median_track_length: 3.0,
            median_max_parallax_deg: 3.0,
            camera_center_diameter: 1.0,
            camera_center_step_median: 0.5,
            camera_center_step_max: 1.0,
            seed_pair_final_distance: 1.0,
            camera_center_window_drift_ratio: 1.0,
            mean_reprojection_px: 0.5,
            leave_one_out_attempts: 120,
            leave_one_out_supported: 48,
            leave_one_out_support_fraction: 0.4,
            median_leave_one_out_reprojection_px: 0.5,
        };
        assert_eq!(
            rejection_reason(&otherwise_valid, &LocalSubmapQualityConfig::default()),
            Some(LocalSubmapRejectionReason::InsufficientLeaveOneOutSupport)
        );
        assert_eq!(
            rejection_reason(
                &LocalSubmapQuality {
                    leave_one_out_supported: 120,
                    leave_one_out_support_fraction: 1.0,
                    median_leave_one_out_reprojection_px: 2.5,
                    ..otherwise_valid
                },
                &LocalSubmapQualityConfig::default(),
            ),
            Some(LocalSubmapRejectionReason::HighLeaveOneOutReprojection)
        );
    }

    /// A relaxed gate config that only lets the scale-sanity check itself
    /// decide the outcome -- every other gate is set to accept anything so a
    /// synthetic `tracks = []` fixture (no landmarks needed to exercise the
    /// pose-only statistic) isn't rejected for an unrelated reason first.
    fn scale_gate_only_config() -> LocalSubmapQualityConfig {
        LocalSubmapQualityConfig {
            min_registered_images: 0,
            min_registration_fraction: 0.0,
            min_landmarks: 0,
            min_observations: 0,
            min_median_track_length: 0.0,
            min_median_max_parallax_deg: 0.0,
            max_mean_reprojection_px: f64::MAX,
            min_leave_one_out_support_fraction: 0.0,
            max_median_leave_one_out_reprojection_px: f64::MAX,
            scale_pathology_gate_enabled: true,
            max_camera_center_displacement_outlier_ratio: 30.0,
            max_seed_pair_scale_drift_ratio: 30.0,
            camera_center_drift_window_count: 4,
            max_camera_center_window_drift_ratio: 1000.0,
            max_scale_pathology_seed_retries: 3,
        }
    }

    fn evenly_spaced_poses(n: usize, step: f64) -> Vec<Option<Pose>> {
        (0..n)
            .map(|i| {
                Some(Pose::from_world_to_camera(
                    UnitQuaternion::identity(),
                    Vector3::new(i as f64 * step, 0.0, 0.0),
                ))
            })
            .collect()
    }

    #[test]
    fn scale_pathology_gate_does_not_fire_on_healthy_geometry() {
        // Six registered frames on a straight, uniformly-spaced path -- like
        // the diagnosis's healthy neighbouring submaps (9 and 11), which had
        // ordinary, smoothly varying consecutive camera-centre steps and were
        // not flagged.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let poses = evenly_spaced_poses(6, 0.1);
        let quality = measure_quality(
            6,
            &camera,
            &poses,
            &[],
            0.5,
            &IncrementalSfmConfig::default(),
            0,
            1,
            0,
        );
        assert!((quality.camera_center_step_median - 0.1).abs() < 1.0e-9);
        assert!((quality.camera_center_step_max - 0.1).abs() < 1.0e-9);
        assert!((quality.seed_pair_final_distance - 0.1).abs() < 1.0e-9);
        assert_eq!(
            rejection_reason(&quality, &scale_gate_only_config()),
            None,
            "a uniform, unremarkable trajectory must not trip the scale-sanity gate"
        );
    }

    #[test]
    fn scale_pathology_gate_does_not_fire_on_step_ratio_alone() {
        // Same five ordinary steps as the healthy fixture, but the final
        // registered frame is placed far outside the trajectory. This makes
        // the max/median-consecutive-step statistic enormous while keeping
        // both active scale-pathology trigger statistics below threshold.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut poses = evenly_spaced_poses(6, 0.1);
        poses[5] = Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(50_000.0, 0.0, 0.0),
        ));
        // Seed pair kept among the untouched, ordinary frames (0, 1) so this
        // test isolates the max/median-step statistic from the seed-pair-drift
        // statistic added below.
        let quality = measure_quality(
            6,
            &camera,
            &poses,
            &[],
            0.5,
            &IncrementalSfmConfig::default(),
            0,
            1,
            0,
        );
        assert!((quality.camera_center_step_median - 0.1).abs() < 1.0e-9);
        assert!(quality.camera_center_step_max > 40_000.0);
        assert_eq!(
            rejection_reason(&quality, &scale_gate_only_config()),
            None,
            "step ratio alone is diagnostic and must not trip the scale-pathology gate"
        );

        // Disabling the gate leaves the same result unchanged.
        let disabled = LocalSubmapQualityConfig {
            scale_pathology_gate_enabled: false,
            ..scale_gate_only_config()
        };
        assert_eq!(rejection_reason(&quality, &disabled), None);
    }

    #[test]
    fn scale_pathology_gate_fires_on_a_uniform_submap_wide_scale_explosion() {
        // The *actual* diagnosed submap-10 failure shape (empirically
        // confirmed on the real MH_03 subrange, not hypothesized): the
        // consecutive-step max/median ratio is *unremarkable* because the
        // scale explosion is uniform across nearly every step, not
        // concentrated in one outlier -- measured
        // `camera_center_step_median = 2056.5`, `camera_center_step_max =
        // 3969.4` on the real pathological submap, ratio `~1.9`, statistically
        // indistinguishable from the healthy neighbours' own ratios (`~2.06`,
        // `~2.78`). This fixture reproduces that shape synthetically: a
        // uniformly-scaled-up trajectory (`step = 2000.0` instead of a
        // healthy `~0.1-1.0`) with *no* single-step outlier at all -- the
        // max/median statistic alone would report `1.0`, comfortably under
        // threshold. Only the seed-pair-drift statistic (comparing the seed
        // pair's final distance to its known seed-time value of exactly
        // `1.0`) catches this.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let poses = evenly_spaced_poses(6, 2000.0);
        let quality = measure_quality(
            6,
            &camera,
            &poses,
            &[],
            0.5,
            &IncrementalSfmConfig::default(),
            0,
            1,
            0,
        );
        assert!((quality.camera_center_step_median - 2000.0).abs() < 1.0e-6);
        assert!((quality.camera_center_step_max - 2000.0).abs() < 1.0e-6);
        assert_eq!(
            camera_center_step_outlier_ratio(&quality),
            1.0,
            "a uniform explosion must NOT trip the max/median statistic -- that is exactly \
             the gap this second statistic exists to close"
        );
        assert!((quality.seed_pair_final_distance - 2000.0).abs() < 1.0e-6);
        assert_eq!(
            rejection_reason(&quality, &scale_gate_only_config()),
            Some(LocalSubmapRejectionReason::ImplausibleScale)
        );

        // A healthy-scale trajectory (seed-pair distance close to the known
        // `1.0` seed-time value) must not trip this statistic.
        let healthy = evenly_spaced_poses(6, 0.3);
        let healthy_quality = measure_quality(
            6,
            &camera,
            &healthy,
            &[],
            0.5,
            &IncrementalSfmConfig::default(),
            0,
            1,
            0,
        );
        assert_eq!(
            rejection_reason(&healthy_quality, &scale_gate_only_config()),
            None
        );
    }

    #[test]
    fn scale_pathology_gate_fires_on_a_within_window_drift_the_other_statistics_miss() {
        // The `LowInlierRatio` seam-cluster follow-on (submaps 9/13,
        // LOWINLIERRATIO_DIAGNOSIS.md): a *gradual* internal drift spread
        // across many consecutive steps, mild enough that neither the
        // single-step max/median ratio nor the seed-pair-drift ratio trips.
        // Twelve steps: the first half at `0.1`, the second half at `1.0` --
        // a smooth-looking ramp, not a single spike. `camera_center_step_max
        // / camera_center_step_median` stays a mild ~1.8x (median sits at the
        // `0.1`/`1.0` midpoint), but a 4-window split cleanly separates the
        // `0.1`-median early windows from the `1.0`-median late windows: a
        // 10x windowed ratio.
        let camera = Camera::pinhole(0, 640, 480, 500.0, 500.0, 320.0, 240.0);
        let mut steps = vec![0.1_f64; 6];
        steps.extend(vec![1.0_f64; 6]);
        let mut x = 0.0;
        let mut poses = vec![Some(Pose::from_world_to_camera(
            UnitQuaternion::identity(),
            Vector3::new(0.0, 0.0, 0.0),
        ))];
        for step in &steps {
            x += step;
            poses.push(Some(Pose::from_world_to_camera(
                UnitQuaternion::identity(),
                Vector3::new(x, 0.0, 0.0),
            )));
        }
        let quality = measure_quality(
            poses.len(),
            &camera,
            &poses,
            &[],
            0.5,
            &IncrementalSfmConfig::default(),
            0,
            1,
            4,
        );
        let step_ratio = camera_center_step_outlier_ratio(&quality);
        assert!(
            step_ratio < 3.0,
            "single-step max/median ratio should stay mild for a smooth ramp, got {step_ratio}"
        );
        assert!(
            (quality.camera_center_window_drift_ratio - 10.0).abs() < 1.0e-6,
            "expected a clean 10x windowed ratio (0.1 -> 1.0 median across the 4 windows), got {}",
            quality.camera_center_window_drift_ratio
        );

        let relaxed_step_gate = LocalSubmapQualityConfig {
            max_camera_center_window_drift_ratio: 5.0,
            ..scale_gate_only_config()
        };
        assert_eq!(
            rejection_reason(&quality, &relaxed_step_gate),
            Some(LocalSubmapRejectionReason::ImplausibleScale),
            "the windowed statistic alone must be sufficient to trip the gate"
        );

        // A submap with no such within-window disagreement (uniform steps)
        // must not trip it.
        let uniform = evenly_spaced_poses(13, 0.3);
        let uniform_quality = measure_quality(
            13,
            &camera,
            &uniform,
            &[],
            0.5,
            &IncrementalSfmConfig::default(),
            0,
            1,
            4,
        );
        assert!((uniform_quality.camera_center_window_drift_ratio - 1.0).abs() < 1.0e-6);
        assert_eq!(rejection_reason(&uniform_quality, &relaxed_step_gate), None);
    }

    #[test]
    fn windowed_drift_ratio_disabled_below_two_windows_or_too_few_steps() {
        // `window_count < 2` disables the statistic entirely (reported as
        // `0.0`, matching `camera_center_step_outlier_ratio`'s "nothing to
        // compare" convention).
        assert_eq!(windowed_camera_center_drift_ratio(&[1.0, 2.0, 3.0], 0), 0.0);
        assert_eq!(windowed_camera_center_drift_ratio(&[1.0, 2.0, 3.0], 1), 0.0);
        // Fewer steps than windows: nothing meaningful to split.
        assert_eq!(windowed_camera_center_drift_ratio(&[1.0], 4), 0.0);
        // Deterministic and order-sensitive: reversing the sequence gives the
        // same ratio (the statistic is direction-agnostic) but a shuffled,
        // non-contiguous grouping would not -- contiguity is what makes this
        // a *windowed* statistic rather than a global one.
        let ratio = windowed_camera_center_drift_ratio(&[0.1, 0.1, 0.1, 1.0, 1.0, 1.0], 2);
        assert!((ratio - 10.0).abs() < 1.0e-9);
        let reversed = windowed_camera_center_drift_ratio(&[1.0, 1.0, 1.0, 0.1, 0.1, 0.1], 2);
        assert!((reversed - 10.0).abs() < 1.0e-9);
    }

    #[test]
    fn leave_one_out_support_detects_an_observation_not_explained_by_other_views() {
        let scene = scene();
        let point = scene.points[scene.points.len() / 2];
        let poses = scene.poses.iter().cloned().map(Some).collect::<Vec<_>>();
        let observations = scene
            .poses
            .iter()
            .enumerate()
            .map(|(image, pose)| {
                let pixel = scene
                    .camera
                    .project(&pose.transform_world_point(&point))
                    .expect("scene point projects");
                (image, 0, pixel)
            })
            .collect::<Vec<_>>();
        let clean = crate::SfmTrack {
            position: point,
            observations: observations.clone(),
        };
        let config = IncrementalSfmConfig {
            min_triangulation_angle_deg: 0.1,
            max_reprojection_error_px: 2.0,
            ..IncrementalSfmConfig::default()
        };
        let (clean_attempts, clean_supported, clean_errors) =
            leave_one_out_support(&scene.camera, &poses, &[clean], &config);
        assert_eq!(clean_attempts, observations.len());
        assert_eq!(clean_supported, clean_attempts);
        assert!(median(clean_errors) < 1.0e-6);

        let mut corrupted_observations = observations;
        corrupted_observations[2].2.x += 100.0;
        let corrupted = crate::SfmTrack {
            position: point,
            observations: corrupted_observations,
        };
        let (corrupt_attempts, corrupt_supported, _) =
            leave_one_out_support(&scene.camera, &poses, &[corrupted], &config);
        assert_eq!(corrupt_attempts, clean_attempts);
        assert!(corrupt_supported < clean_supported);
    }
}
