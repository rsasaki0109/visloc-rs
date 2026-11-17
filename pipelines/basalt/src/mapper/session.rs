//! Stateful offline mapper ingestion at Basalt's `NfrMapper::addMargData`
//! boundary.
//!
//! The lower-level mapper functions intentionally operate on one
//! [`MargData`] packet at a time.  Upstream
//! `NfrMapper` does not, however: it reduces and factor-recovers each packet,
//! then retains the valid frame poses, recovered factors, and raw images in a
//! long-lived mapper object.  Keeping that boundary explicit prevents a
//! caller from accidentally optimizing only the last packet or dropping the
//! image records needed by the later feature stages.
//!
//! This module is independent of the estimator's AOM construction and IMU
//! factor implementation.  It consumes the versioned `MargData` contract
//! produced by those components and mirrors only the pinned upstream mapper
//! ingestion order.
//!
//! The pinned C++ mapper keys its aligned pose/image maps by camera timestamp.
//! The Rust MargData contract carries a compact `frame_id` alongside that
//! timestamp, so this stateful adapter keeps poses keyed by frame ID and
//! retains `frame_timestamps` for trajectory/result serialization.  Image
//! buckets remain timestamp-keyed, with their records translated back to the
//! frame-ID graph identity at detection/stereo boundaries.

use std::collections::{BTreeMap, BTreeSet};

use nalgebra::{Matrix2, Matrix3, Matrix6, UnitQuaternion, Vector2, Vector3, Vector6};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use visloc_core::geometry::SE3;

use crate::{
    calibration::BasaltCalibration,
    mapper::features::{
        extract_mapper_features, match_stereo_features, match_temporal_ransac,
        match_temporal_ransac_seeded, match_temporal_stage, mutual_descriptor_matches,
        query_bow_candidates, BowQueryCandidate, FeaturePipelineError, MapperImageFeatures,
        MapperImageId,
    },
    pyramid::{ImageError, RawU16Image},
    vio::margdata::{MargData, OfImageData},
};
// Only exercised by the `tests` module's fixtures below (via `use super::*;`).
#[cfg(test)]
use crate::mapper::features::BowEntry;
#[cfg(test)]
use crate::vio::margdata::MARGDATA_SCHEMA_VERSION_V3;

use super::{
    extract_nonlinear_factors, global_ba_with_state, linearize_mapper_observation,
    local_ba_with_state, process_marg_data, rel_pose_error, roll_pitch_error,
    setup_opt as run_setup_opt, triangulate_pair, FeatureId, FeatureTracks, GlobalBaConfig,
    GlobalBaIteration, GlobalBaOptimizerState, ImagePair, MapperConfig, MapperFactors,
    MapperLandmark, MapperObservation, MapperSummary, MargDataProcessError, MatchData, Matches,
    NfrExtractionError, OfflineMapperConfig, RelativePoseFactor, SetupOptInput, SetupOptReport,
    TemporalRansacResult, TimeCamId, TrackBuilder, TrackFilterReport,
};

/// Errors raised while processing an individual mapper packet.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NfrMapperError {
    /// The mixed absolute system or its AOM ordering is malformed.
    MargData(MargDataProcessError),
    /// The packet failed the versioned MargData contract before any mapper
    /// state or packet-owned value was mutated.
    InvalidMargDataContract,
    /// Factor recovery failed for a reason other than the upstream
    /// rank-deficient rejection (which is reported as `accepted = false`).
    FactorRecovery(NfrExtractionError),
}

/// Errors raised by the stateful feature-detection handoff.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NfrMapperFeatureError {
    #[error("mapper feature detection requires calibration")]
    MissingCalibration,
    #[error("raw mapper image {timestamp_ns}/{camera_id} is invalid: {source}")]
    Image {
        timestamp_ns: i64,
        camera_id: u16,
        #[source]
        source: ImageError,
    },
    #[error("mapper feature extraction failed for image {timestamp_ns}/{camera_id}: {source}")]
    Extraction {
        timestamp_ns: i64,
        camera_id: u16,
        #[source]
        source: FeaturePipelineError,
    },
    #[error("mapper calibration has no camera {camera_id} for image {timestamp_ns}")]
    MissingCamera { timestamp_ns: i64, camera_id: u16 },
}

/// Result of one `NfrMapper::addMargData` call.
///
/// Upstream returns no report, but these packet-local counts make it possible
/// to assert that the stateful boundary retained exactly the data from an
/// accepted packet without exposing internal container choices as a parity
/// contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NfrMapperIngestReport {
    pub input_size: usize,
    pub output_size: usize,
    pub accepted: bool,
    pub frame_pose_count: usize,
    pub relative_pose_factor_count: usize,
    pub roll_pitch_factor_count: usize,
    pub image_timestamp_count: usize,
}

/// Counts emitted by one stateful `NfrMapper::detect_keypoints` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NfrMapperDetectionReport {
    /// Number of timestamp buckets retained by `addMargData`.
    pub input_timestamp_count: usize,
    /// Number of timestamp buckets containing an image whose frame ID has a
    /// corresponding installed pose.
    pub eligible_timestamp_count: usize,
    /// Number of raw camera images converted into feature products.
    pub processed_image_count: usize,
    /// Total number of detected corners across the processed images.
    pub feature_count: usize,
}

/// Errors raised by the stateful `NfrMapper::match_stereo` handoff.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NfrMapperStereoError {
    #[error("stereo matching requires calibration")]
    MissingCalibration,
    #[error("stereo matching requires calibration for camera {camera_id}")]
    MissingCamera { camera_id: u16 },
    #[error("stereo matching failed for timestamp {timestamp_ns}: {source}")]
    Pipeline {
        timestamp_ns: i64,
        #[source]
        source: FeaturePipelineError,
    },
}

/// Errors raised by the stateful `NfrMapper::setup_opt` handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NfrMapperSetupOptError {
    #[error("mapper setup_opt requires calibration")]
    MissingCalibration,
}

/// Errors raised by the stateful `NfrMapper::optimize` handoff.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Error)]
pub enum NfrMapperOptimizeError {
    #[error("mapper optimize requires calibration")]
    MissingCalibration,
}

/// Aggregate state transition counts from one `NfrMapper::match_stereo` pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NfrMapperStereoReport {
    /// Number of raw-image timestamp buckets considered by the pass.
    pub input_timestamp_count: usize,
    /// Number of non-negative timestamp buckets for which the stereo pair was
    /// attempted.  Missing feature maps still count as attempts, matching the
    /// upstream `feature_corners[tcid]` default construction.
    pub attempted_timestamp_count: usize,
    /// Buckets where one or both camera feature maps were absent before the
    /// source-compatible empty-map materialization.
    pub missing_feature_timestamp_count: usize,
    pub raw_match_count: usize,
    pub essential_inlier_count: usize,
    /// Accepted pair insertions from this pass (`inliers.size() > 16`).
    pub accepted_pair_count: usize,
    /// Total persistent pair count after this pass.
    pub total_pair_count: usize,
}

/// Aggregate state transition counts from one stateful `NfrMapper::match_all`
/// pass.  `candidate_pairs` is retained in the deterministic Rust report so
/// fixtures can assert the source key/id-to-index and HashBoW ordering rather
/// than only a final graph cardinality.
#[derive(Debug, Clone, PartialEq)]
pub struct NfrMapperMatchAllReport {
    /// Number of entries in `feature_corners` when the key/index vectors were
    /// built.
    pub input_feature_count: usize,
    /// Number of feature entries queried against the HashBoW database.
    pub query_count: usize,
    /// Candidate pairs surviving the source strict frame and score gates.
    pub candidate_pair_count: usize,
    /// Candidate pair order and score after the deterministic database query.
    pub candidate_pairs: Vec<(ImagePair, f64)>,
    /// Sum of all mutual descriptor matches across candidate pairs.
    pub raw_match_count: usize,
    /// Cumulative raw descriptor matches retained in the persistent source
    /// graph after this pass.  Upstream's diagnostic `match_all` counter walks
    /// the whole `feature_matches` map, so this includes the already stored
    /// stereo edges as well as the temporal edges from this pass.
    pub cumulative_raw_match_count: usize,
    /// Cumulative inlier count in the persistent source graph after this pass.
    pub cumulative_inlier_match_count: usize,
    /// Raw/inlier counts and accepted-pair counts split by the source's
    /// same-frame stereo edges versus cross-frame temporal edges.
    pub stereo_raw_match_count: usize,
    pub stereo_inlier_match_count: usize,
    pub stereo_match_pair_count: usize,
    pub temporal_raw_match_count: usize,
    pub temporal_inlier_match_count: usize,
    pub temporal_match_pair_count: usize,
    /// Candidate pairs rejected by the strict `raw_matches > min_matches`
    /// gate before RANSAC.
    pub raw_gate_rejected_pair_count: usize,
    /// Candidate pairs for which relative-pose RANSAC was attempted.
    pub ransac_attempt_count: usize,
    /// RANSAC-attempted pairs which did not satisfy the source min-inlier
    /// gate after geometric refinement.
    pub geometric_rejected_pair_count: usize,
    /// Accepted pair insertions from this pass.
    pub accepted_pair_count: usize,
    /// Total persistent pair count after this pass.
    pub total_pair_count: usize,
}

/// Aggregate state transition counts from one stateful
/// `NfrMapper::build_tracks` pass.  The embedded filter report retains the
/// source TrackBuilder's unrenumbered roots and complete-component rejection
/// decisions while the scalar counts mirror the mapper's diagnostic output.
#[derive(Debug, Clone, PartialEq)]
pub struct NfrMapperTrackReport {
    pub input_pair_count: usize,
    pub inlier_match_count: usize,
    pub node_count: usize,
    pub component_count_before: usize,
    pub component_count_after: usize,
    pub rejected_conflict_ids: Vec<u64>,
    pub rejected_short_ids: Vec<u64>,
    pub rejected_track_ids: Vec<u64>,
    pub total_track_obs_count: usize,
    /// `None` is the safe representation of the source's zero-track
    /// `total_obs / feature_tracks.size()` diagnostic division.
    pub average_track_length: Option<f64>,
    pub exported_track_count: usize,
    pub filter: TrackFilterReport,
}

/// Source-ordered trace and persistent-state counts from one stateful
/// `NfrMapper::optimize` call.  The embedded global-BA iterations retain the
/// visual RelLinData/Schur solve, factor rows, damping trial, and
/// accept/reject records needed for the pinned oracle comparison.
#[derive(Debug, Clone, PartialEq)]
pub struct NfrMapperOptimizeReport {
    pub requested_iterations: usize,
    pub iterations: usize,
    pub pose_count: usize,
    pub landmark_count: usize,
    pub initial_cost: f64,
    pub final_cost: f64,
    pub accepted_step_count: usize,
    pub rejected_trial_count: usize,
    pub initial_lambda: f64,
    pub min_lambda: f64,
    pub max_lambda: f64,
    pub initial_lambda_vee: f64,
    pub final_lambda: f64,
    pub final_lambda_vee: f64,
    pub final_state_hash: u64,
    pub trace_hash: u64,
    pub trace: Vec<GlobalBaIteration>,
}

/// Pose-map view exposed by the mapper result boundary.
///
/// `frame_id` is the mapper's ordered key (the Rust frame index used by
/// `TimeCamId`).  `timestamp_ns` is retained separately from that key because
/// MargData carries both values.  The pose itself is `T_w_i` and follows the
/// on-disk MargData convention `[tx, ty, tz, qw, qx, qy, qz]`; naming the
/// quaternion explicitly prevents an accidental xyzw/wxyz swap at export
/// call sites.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperPoseRecord {
    pub frame_id: u64,
    pub timestamp_ns: i64,
    pub translation: [f64; 3],
    pub quaternion_wxyz: [f64; 4],
}

/// Deterministic map record corresponding to one initialized mapper landmark.
///
/// The full observation payload remains available through `lmdb`; this record
/// is the compact result/export form used by the headless mapping boundary and
/// preserves the source `LandmarkDatabase` track identity and host/second
/// camera ordering.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperImageRecord {
    pub frame_id: u64,
    pub camera_id: u16,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperLandmarkRecord {
    pub track_id: u64,
    pub host: NfrMapperImageRecord,
    pub second: NfrMapperImageRecord,
    pub direction: [f64; 2],
    pub inverse_distance: f64,
    pub observation_count: usize,
}

/// Result snapshot after the stateful mapper lifecycle has reached BA.
///
/// This is intentionally a snapshot rather than a second mutable database:
/// subsequent `addMargData`/frontend passes continue to mutate `NfrMapper`,
/// while callers that need a stable artifact can retain this value and
/// serialize it with serde.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperResult {
    pub poses: Vec<NfrMapperPoseRecord>,
    pub landmarks: Vec<NfrMapperLandmarkRecord>,
}

/// World-point snapshot produced by the inherited
/// `BundleAdjustmentBase::get_current_points` boundary.  The upstream helper
/// emits one point for every host/target observation-index entry (so the same
/// landmark can appear more than once) and one constant visualization ID
/// (`1`) per point. Retaining both details keeps the Rust headless result
/// shape source-compatible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperCurrentPoints {
    pub points: Vec<[f64; 3]>,
    pub ids: Vec<i32>,
}

/// Counts and reprojection cost from one inherited `filterOutliers` pass.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperFilterReport {
    pub outlier_threshold: f64,
    pub min_num_obs: usize,
    pub reprojection_error: f64,
    pub before_landmark_count: usize,
    pub after_landmark_count: usize,
    pub before_observation_count: usize,
    pub after_observation_count: usize,
    pub candidate_landmark_count: usize,
    pub outlier_observation_count: usize,
    pub invalid_observation_count: usize,
    pub removed_landmark_count: usize,
    pub removed_observation_count: usize,
}

/// Configuration for the top-level headless mapping workflow.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperHeadlessConfig {
    pub num_opt_iter: usize,
    pub outlier_threshold: f64,
    pub min_num_obs: usize,
    /// Production uses `None` (OpenGV's time-derived seed).  A fixed seed is
    /// available solely for release fixtures and deterministic diagnostics.
    pub temporal_seed: Option<u32>,
}

impl Default for NfrMapperHeadlessConfig {
    fn default() -> Self {
        Self {
            num_opt_iter: 10,
            outlier_threshold: 3.0,
            min_num_obs: 4,
            temporal_seed: None,
        }
    }
}

/// One ordered transition in the headless mapper workflow.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NfrMapperHeadlessStage {
    pub name: String,
    pub pose_count: usize,
    pub landmark_count: usize,
    pub observation_count: usize,
    pub point_count: usize,
    pub reprojection_error: Option<f64>,
}

/// Full source-ordered headless mapper report.
#[derive(Debug, Clone, PartialEq)]
pub struct NfrMapperHeadlessReport {
    pub config: NfrMapperHeadlessConfig,
    pub detection: NfrMapperDetectionReport,
    pub stereo: NfrMapperStereoReport,
    pub match_all: NfrMapperMatchAllReport,
    pub tracks: NfrMapperTrackReport,
    pub setup: SetupOptReport,
    pub first_optimize: NfrMapperOptimizeReport,
    pub filter: NfrMapperFilterReport,
    pub second_optimize: NfrMapperOptimizeReport,
    pub initial_points: NfrMapperCurrentPoints,
    pub filtered_points: NfrMapperCurrentPoints,
    pub final_points: NfrMapperCurrentPoints,
    pub result: NfrMapperResult,
    pub stages: Vec<NfrMapperHeadlessStage>,
}

/// Errors raised by the top-level headless workflow.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum NfrMapperHeadlessError {
    #[error("headless detection failed: {0}")]
    Detection(NfrMapperFeatureError),
    #[error("headless stereo matching failed: {0}")]
    Stereo(NfrMapperStereoError),
    #[error("headless setup_opt failed: {0}")]
    Setup(NfrMapperSetupOptError),
    #[error("headless optimize failed: {0}")]
    Optimize(NfrMapperOptimizeError),
}

/// Persistent Rust representation of Basalt's `LandmarkDatabase<double>`.
///
/// `landmarks` retains the full initialized track payload used by the mapper
/// BA port.  `observations` mirrors the source host→target→landmark index that
/// `LandmarkDatabase::addObservation` maintains alongside each keypoint.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NfrMapperLandmarkDb {
    pub landmarks: BTreeMap<u64, MapperLandmark>,
    pub observations: BTreeMap<TimeCamId, BTreeMap<TimeCamId, BTreeSet<u64>>>,
}

impl NfrMapperLandmarkDb {
    fn from_landmarks(landmarks: BTreeMap<u64, MapperLandmark>) -> Self {
        let mut observations = BTreeMap::<TimeCamId, BTreeMap<TimeCamId, BTreeSet<u64>>>::new();
        for (&track_id, landmark) in &landmarks {
            for observation in &landmark.observations {
                observations
                    .entry(landmark.host)
                    .or_default()
                    .entry(observation.image)
                    .or_default()
                    .insert(track_id);
            }
        }
        Self {
            landmarks,
            observations,
        }
    }

    pub fn rebuild_observation_index(&mut self) {
        let landmarks = std::mem::take(&mut self.landmarks);
        *self = Self::from_landmarks(landmarks);
    }

    pub fn num_landmarks(&self) -> usize {
        self.landmarks.len()
    }

    pub fn num_observations(&self) -> usize {
        self.observations
            .values()
            .flat_map(|targets| targets.values())
            .map(BTreeSet::len)
            .sum()
    }

    pub fn num_hosts(&self) -> usize {
        self.observations.len()
    }

    /// Return the ordered landmark IDs hosted by one camera image.
    pub fn landmarks_for_host(&self, host: TimeCamId) -> Vec<u64> {
        self.observations
            .get(&host)
            .into_iter()
            .flat_map(|targets| targets.values())
            .flat_map(|ids| ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    /// Return the first landmark observed at `(image, feature_id)`, if any.
    ///
    /// The landmark store is small (thousands) and loop-factor construction
    /// runs only on accepted loop pairs, so a linear scan is fine; the
    /// `observations` index is host-centric and does not key by feature.
    pub fn landmark_for_observation(
        &self,
        image: TimeCamId,
        feature_id: FeatureId,
    ) -> Option<&MapperLandmark> {
        self.landmarks.values().find(|landmark| {
            landmark
                .observations
                .iter()
                .any(|o| o.image == image && o.feature_id == feature_id)
        })
    }

    pub fn num_landmark_observations(&self, track_id: u64) -> Option<usize> {
        self.landmarks
            .get(&track_id)
            .map(|landmark| landmark.observations.len())
    }

    fn remove_landmark_raw(&mut self, track_id: u64) -> bool {
        self.landmarks.remove(&track_id).is_some()
    }

    fn remove_observations_raw(&mut self, track_id: u64, outliers: &BTreeSet<TimeCamId>) -> usize {
        let Some(landmark) = self.landmarks.get_mut(&track_id) else {
            return 0;
        };
        let before = landmark.observations.len();
        landmark
            .observations
            .retain(|observation| !outliers.contains(&observation.image));
        before.saturating_sub(landmark.observations.len())
    }

    /// Remove one complete source landmark and rebuild its host/target index.
    pub fn remove_landmark(&mut self, track_id: u64) -> bool {
        let removed = self.remove_landmark_raw(track_id);
        if removed {
            self.rebuild_observation_index();
        }
        removed
    }

    /// Remove only the selected target-image observations from one landmark,
    /// preserving the source observation vector order and track ID.
    pub fn remove_observations(&mut self, track_id: u64, outliers: &BTreeSet<TimeCamId>) -> usize {
        let removed = self.remove_observations_raw(track_id, outliers);
        if removed > 0 {
            self.rebuild_observation_index();
        }
        removed
    }
}

/// Full source-compatible match payload retained for accepted stereo/temporal
/// pairs.
///
/// The existing public [`MatchData`] intentionally remains the track-builder
/// projection containing only inliers, because `TrackBuilder` consumes only
/// that field and existing fixture/serialization contracts use it.  The
/// stateful mapper keeps the upstream `T_i_j` and raw descriptor matches here
/// so the lifecycle does not discard information merely to feed tracks.
#[derive(Debug, Clone, PartialEq)]
pub struct NfrMapperMatchData {
    pub t_i_j: SE3,
    pub matches: Vec<(FeatureId, FeatureId)>,
    pub inliers: Vec<(FeatureId, FeatureId)>,
}

/// Long-lived state corresponding to the pinned upstream `NfrMapper` object.
///
/// The maps use ordered Rust containers so serialization and release tests are
/// reproducible.  This does not change the upstream data contract: frame IDs,
/// factor append order, and image timestamps are retained verbatim.
#[derive(Debug, Clone, PartialEq)]
pub struct NfrMapper {
    pub config: MapperConfig,
    /// Optional calibration needed by the image frontend.  MargData factor
    /// ingestion remains usable without it, matching the source lifecycle in
    /// which calibration is supplied to the mapper constructor but is not
    /// needed by `addMargData` itself.
    pub calibration: Option<BasaltCalibration>,
    /// M8c settings used by the stateful feature handoff.
    pub feature_config: OfflineMapperConfig,
    /// All valid pose-only states accumulated from accepted MargData packets.
    pub frame_poses: BTreeMap<u64, SE3>,
    /// Camera timestamps paired with the mapper frame IDs above.  Upstream's
    /// C++ map is keyed directly by `t_ns`; the Rust MargData contract keeps a
    /// compact frame index and timestamp as distinct fields, so retaining this
    /// side table is required for faithful trajectory export.
    pub frame_timestamps: BTreeMap<u64, i64>,
    /// Frozen (velocity_world, gyro_bias, accel_bias) per keyframe.
    pub frame_velocity_bias: BTreeMap<u64, (Vector3<f64>, Vector3<f64>, Vector3<f64>)>,
    /// Live, mapper-refined per-keyframe world velocity, seeded from
    /// `frame_velocity_bias` and updated by periodic IMU velocity refinement;
    /// distinct from the frozen original VIO estimates in `frame_velocity_bias`.
    pub frame_velocities: BTreeMap<u64, Vector3<f64>>,
    /// Live, joint-VI-BA-refined per-keyframe gyro bias, seeded from
    /// `frame_velocity_bias` and updated by [`Self::optimize_joint_with_extra_factors`].
    /// Distinct from `frame_velocity_bias`'s frozen original VIO estimate, the
    /// same way `frame_velocities` is distinct from it. Unused (stays empty)
    /// unless the joint solver runs.
    pub frame_gyro_bias: BTreeMap<u64, Vector3<f64>>,
    /// Live, joint-VI-BA-refined per-keyframe accel bias; see
    /// `frame_gyro_bias`.
    pub frame_accel_bias: BTreeMap<u64, Vector3<f64>>,
    /// Recovered factors are appended in packet order, matching upstream's
    /// persistent `rel_pose_factors` and `roll_pitch_factors` vectors.
    pub factors: MapperFactors,
    /// Raw image vectors keyed by the upstream camera timestamp (`t_ns`).
    /// A later packet with the same timestamp replaces the previous vector,
    /// exactly like `img_data[t_ns] = input_images` in C++.
    pub img_data: BTreeMap<i64, Vec<OfImageData>>,
    /// Feature products keyed by the Rust `(frame_id, camera)` identity.
    /// `img_data` remains timestamp-keyed, but the versioned image contract
    /// carries the frame index separately; retaining that distinction keeps
    /// M7 packets with compact frame IDs usable by the later graph stages.
    pub feature_corners: BTreeMap<TimeCamId, MapperImageFeatures>,
    /// Inverted HashBoW index: hash bucket -> images whose `bow_vector`
    /// contains that hash.  `query_bow_candidates` (`features.rs`) only ever
    /// scores a database entry that shares a hash bucket with the query (its
    /// `shared` flag), so restricting `match_all`'s per-query scan to the
    /// union of the query's own hash-bucket members is an *exact* reduction
    /// of the candidate set, not an approximation -- ported from the online
    /// mapper's `OnlineNfrMapper::hash_index`
    /// (`pipelines/basalt/src/mapper/online.rs`, commit 8b55648) together
    /// with its `bow_candidates_via_index` equivalence argument.  Populated
    /// additively wherever `feature_corners` gains a non-empty `bow_vector`
    /// (`detect_keypoints`); `match_stereo`'s empty-feature placeholders
    /// contribute nothing since their `bow_vector` is empty.
    pub hash_index: BTreeMap<u32, BTreeSet<TimeCamId>>,
    /// Persistent stereo/temporal match graph input.  `match_stereo` inserts
    /// only strict essential-inlier winners and `match_all` inserts only
    /// geometrically verified temporal winners, leaving prior accepted pairs
    /// in place when a later pass rejects the same key.
    pub feature_matches: Matches,
    /// Full accepted-pair payload corresponding to upstream `MatchData`.
    /// `feature_matches` remains the stable inlier-only track contract.
    pub feature_match_data: BTreeMap<ImagePair, NfrMapperMatchData>,
    /// Persistent result of the source `build_tracks` lifecycle.  Each pass
    /// rebuilds this map from the current match graph and replaces the prior
    /// export, retaining unrenumbered union-find roots.
    pub feature_tracks: FeatureTracks,
    /// Persistent initialized landmark database populated by `setup_opt`.
    pub lmdb: NfrMapperLandmarkDb,
    /// Mapper BA controls corresponding to the pinned `VioConfig` mapper
    /// fields.  `optimize` replaces only `max_iterations` with its explicit
    /// argument; all factor/LM/robustification controls remain persistent.
    pub optimize_config: GlobalBaConfig,
    /// Live source damping state.  Unlike a one-shot BA config, this survives
    /// separate `optimize` calls exactly like NfrMapper's lambda members.
    pub optimizer_state: GlobalBaOptimizerState,
    pub accepted_packets: usize,
}

impl NfrMapper {
    /// Minimal copy for the background optimizer pipeline
    /// (`build_tracks` -> `setup_opt` -> `optimize` -> `filter_outliers`).
    ///
    /// Those stages read only the match graph, feature *pixel* corners,
    /// poses, factors, calibration and optimizer configuration and state.
    /// `build_tracks` always replaces `feature_tracks`, and `setup_opt`
    /// (given calibration) always replaces `lmdb`. Descriptors, rays, BoW,
    /// match payloads, raw images, frozen velocity/bias history and the old
    /// tracks and landmarks are therefore dead weight in the snapshot. For
    /// the pose-only frozen-velocity IMU factor (`imu_relative_pose`), that
    /// used to include the live velocities too: they are refreshed into
    /// baked relative-pose factors (retained in `factors`) *before* this
    /// snapshot is taken, so `optimize()` on the snapshot never reads
    /// `frame_velocities` itself. The joint solver
    /// (`optimize_joint_with_extra_factors`) is different: it optimizes
    /// velocity/gyro-bias/accel-bias as first-class state alongside pose, so
    /// those three maps *are* cloned into the snapshot (cheap -- one
    /// `Vector3` per keyframe) so a background joint pass has real state to
    /// iterate from and `merge_background_result` has something to merge
    /// back. A full clone doubled the mapper's persistent memory at every
    /// periodic optimize, which this snapshot still avoids for every other
    /// (large) container.
    ///
    /// Without calibration `setup_opt` would keep the old `lmdb`, so the
    /// full clone is returned in that case.
    pub fn optimizer_snapshot(&self) -> Self {
        if self.calibration.is_none() {
            return self.clone();
        }
        let feature_corners = self
            .feature_corners
            .iter()
            .map(|(&key, features)| {
                (
                    key,
                    MapperImageFeatures {
                        corners: features.corners.clone(),
                        corner_angles: Vec::new(),
                        descriptors: Vec::new(),
                        rays: Vec::new(),
                        hashes: Vec::new(),
                        bow_vector: Vec::new(),
                    },
                )
            })
            .collect();
        Self {
            config: self.config,
            calibration: self.calibration.clone(),
            feature_config: self.feature_config,
            frame_poses: self.frame_poses.clone(),
            frame_timestamps: self.frame_timestamps.clone(),
            frame_velocity_bias: BTreeMap::new(),
            frame_velocities: self.frame_velocities.clone(),
            frame_gyro_bias: self.frame_gyro_bias.clone(),
            frame_accel_bias: self.frame_accel_bias.clone(),
            factors: self.factors.clone(),
            img_data: BTreeMap::new(),
            feature_corners,
            hash_index: BTreeMap::new(),
            feature_matches: self.feature_matches.clone(),
            feature_match_data: BTreeMap::new(),
            feature_tracks: FeatureTracks::new(),
            lmdb: NfrMapperLandmarkDb::default(),
            optimize_config: self.optimize_config,
            optimizer_state: self.optimizer_state.clone(),
            accepted_packets: self.accepted_packets,
        }
    }

    /// Approximate heap payload per persistent container, in bytes.
    ///
    /// Counts element payload (`len * size_of`) only, ignoring allocator and
    /// B-tree node overhead, so it is a lower bound suited to comparing
    /// containers and keyframe densities, not an RSS reconstruction.
    pub fn approx_heap_breakdown(&self) -> Vec<(&'static str, usize)> {
        use std::mem::size_of;
        let features: usize = self
            .feature_corners
            .values()
            .map(|f| {
                f.corners.len() * size_of::<nalgebra::Point2<f64>>()
                    + f.corner_angles.len() * size_of::<f64>()
                    + f.descriptors.len() * 32
                    + f.rays.len() * size_of::<[f64; 4]>()
                    + f.hashes.len() * size_of::<u32>()
                    + f.bow_vector.len() * size_of::<super::features::BowEntry>()
            })
            .sum();
        let pair = size_of::<(FeatureId, FeatureId)>();
        let match_data: usize = self
            .feature_match_data
            .values()
            .map(|m| (m.matches.len() + m.inliers.len()) * pair + size_of::<NfrMapperMatchData>())
            .sum();
        let matches: usize = self
            .feature_matches
            .values()
            .map(|m| m.inliers.len() * pair)
            .sum();
        let tracks: usize = self
            .feature_tracks
            .values()
            .map(|t| t.len() * size_of::<(TimeCamId, FeatureId)>())
            .sum();
        let landmarks: usize = self
            .lmdb
            .landmarks
            .values()
            .map(|l| {
                size_of::<MapperLandmark>()
                    + l.observations.len() * size_of::<super::triangulation::MapperObservation>()
            })
            .sum();
        let observation_index: usize = self
            .lmdb
            .observations
            .values()
            .flat_map(BTreeMap::values)
            .map(|ids| ids.len() * size_of::<u64>() + 2 * size_of::<TimeCamId>())
            .sum();
        let hash_index: usize = self
            .hash_index
            .values()
            .map(|ids| ids.len() * size_of::<TimeCamId>() + size_of::<u32>())
            .sum();
        vec![
            ("keyframe_images", self.feature_corners.len()),
            ("feature_corners", features),
            ("feature_match_data", match_data),
            ("feature_matches", matches),
            ("feature_tracks", tracks),
            ("lmdb_landmarks", landmarks),
            ("lmdb_observation_index", observation_index),
            ("hash_index", hash_index),
            ("match_pairs", self.feature_match_data.len()),
        ]
    }

    pub fn new(config: MapperConfig) -> Self {
        Self {
            config,
            calibration: None,
            feature_config: OfflineMapperConfig::default(),
            frame_poses: BTreeMap::new(),
            frame_timestamps: BTreeMap::new(),
            frame_velocity_bias: BTreeMap::new(),
            frame_velocities: BTreeMap::new(),
            frame_gyro_bias: BTreeMap::new(),
            frame_accel_bias: BTreeMap::new(),
            factors: MapperFactors {
                imu_relative_pose: Vec::new(),
                provenance_version: String::new(),
                relative_pose: Vec::new(),
                roll_pitch: Vec::new(),
                ba_covisibility: Vec::new(),
            },
            img_data: BTreeMap::new(),
            feature_corners: BTreeMap::new(),
            hash_index: BTreeMap::new(),
            feature_matches: Matches::new(),
            feature_match_data: BTreeMap::new(),
            feature_tracks: FeatureTracks::new(),
            lmdb: NfrMapperLandmarkDb::default(),
            optimize_config: GlobalBaConfig::default(),
            optimizer_state: GlobalBaOptimizerState::new(
                GlobalBaConfig::default().lambda_min,
                GlobalBaConfig::default().lambda_max,
            ),
            accepted_packets: 0,
        }
    }

    /// Constructs a mapper with the calibration needed by the image
    /// frontend.  `new` remains available for packet-only factor recovery.
    pub fn with_calibration(config: MapperConfig, calibration: BasaltCalibration) -> Self {
        let mut mapper = Self::new(config);
        mapper.calibration = Some(calibration);
        mapper
    }

    /// Installs or replaces the image calibration before feature detection.
    pub fn set_calibration(&mut self, calibration: BasaltCalibration) {
        self.calibration = Some(calibration);
    }

    /// Sets the pinned M8c feature configuration used by subsequent frontend
    /// calls.  Existing feature products are intentionally retained, matching
    /// the upstream mapper's persistent `feature_corners` map.
    pub const fn set_feature_config(&mut self, config: OfflineMapperConfig) {
        self.feature_config = config;
    }

    /// Install mapper BA controls and reset the source constructor damping
    /// state.  The pinned constructor starts `lambda` at the configured
    /// minimum and `lambda_vee` at two, so changing bounds is a new mapper
    /// configuration rather than a continuation of the old schedule.
    pub const fn set_optimize_config(&mut self, config: GlobalBaConfig) {
        self.optimize_config = config;
        self.optimizer_state = GlobalBaOptimizerState::new(config.lambda_min, config.lambda_max);
    }

    /// Validate the versioned packet contract at the stateful mapper ingress.
    ///
    /// `MargData::from_json` is intentionally parse-only because diagnostic and
    /// legacy callers may need to inspect incomplete records.  The mapper
    /// consumer, however, is a mutation boundary: schema-4 packets must pass
    /// their complete FEJ/AOM/image checks before reduction, image retention,
    /// or any other state transition.  The contract validator also retains the
    /// existing schema 1--3 compatibility that is part of the on-disk reader;
    /// this function adds no compatibility fallback or field synthesis.
    fn validate_marg_data_ingress(data: &MargData) -> Result<(), NfrMapperError> {
        data.validate_contract()
            .map_err(|_| NfrMapperError::InvalidMargDataContract)
    }

    /// Ingest one packet in the same order as `NfrMapper::addMargData`:
    /// reduce the mixed absolute system, retain its raw input images, recover
    /// nonlinear factors, reject a rank-deficient packet, then merge poses and
    /// factors only when recovery succeeds.  Image retention intentionally
    /// happens before the rank gate because the pinned `processMargData`
    /// stores `img_data[t_ns]` before `addMargData` checks the factor result.
    ///
    /// The packet is intentionally borrowed mutably because upstream mutates
    /// `MargData` in `processMargData` before factor extraction.  A rank-
    /// deficient packet therefore retains the same processed state while its
    /// contents are not merged into this mapper.
    pub fn add_marg_data(
        &mut self,
        data: &mut MargData,
    ) -> Result<NfrMapperIngestReport, NfrMapperError> {
        self.add_marg_data_with_velocity_bias(data, true)
    }

    /// Online ingestion can skip navigation-state retention when IMU factors are off.
    pub(crate) fn add_marg_data_with_velocity_bias(
        &mut self,
        data: &mut MargData,
        retain_velocity_bias: bool,
    ) -> Result<NfrMapperIngestReport, NfrMapperError> {
        // Contract validation is deliberately the first operation.  In
        // particular, process_marg_data mutates its input and retain_images
        // mutates the long-lived mapper before the rank gate.
        Self::validate_marg_data_ingress(data)?;
        // Processing removes navigation states as it converts keyframes to poses.
        // Capture first, but publish only after successful factor recovery below.
        let navigation_states = if retain_velocity_bias {
            data.frame_states
                .iter()
                .map(|state| {
                    (
                        state.frame_id,
                        (
                            Vector3::from(state.velocity),
                            Vector3::from(state.gyro_bias),
                            Vector3::from(state.accel_bias),
                        ),
                    )
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let process = process_marg_data(data).map_err(NfrMapperError::MargData)?;
        self.retain_images(data);
        let factors = match extract_nonlinear_factors(data, self.config) {
            Ok(factors) => factors,
            // The pinned implementation's only expected false return is the
            // full-column-rank gate.  Do not turn malformed/singular inputs
            // into an apparently valid empty packet.
            Err(NfrExtractionError::RankDeficient) => {
                return Ok(NfrMapperIngestReport {
                    input_size: process.input_size,
                    output_size: process.output_size,
                    accepted: false,
                    frame_pose_count: self.frame_poses.len(),
                    relative_pose_factor_count: self.factors.relative_pose.len(),
                    roll_pitch_factor_count: self.factors.roll_pitch.len(),
                    image_timestamp_count: self.img_data.len(),
                });
            }
            Err(error) => return Err(NfrMapperError::FactorRecovery(error)),
        };

        // `process_marg_data` converts keyframe navigation states into
        // explicit frame poses.  Retain those first, then mirror the
        // upstream safety net for callers carrying an unprocessed keyframe
        // state in `frame_states`.
        for pose in &data.frame_poses {
            self.frame_poses
                .insert(pose.frame_id, pose_to_se3(pose.pose));
            self.frame_timestamps
                .insert(pose.frame_id, pose.timestamp_ns);
        }
        let keyframes = data.kfs_all.iter().copied().collect::<BTreeSet<_>>();
        for (frame_id, velocity_bias) in navigation_states {
            if keyframes.contains(&frame_id) {
                self.frame_velocity_bias.insert(frame_id, velocity_bias);
                self.frame_velocities
                    .entry(frame_id)
                    .or_insert(velocity_bias.0);
                self.frame_gyro_bias
                    .entry(frame_id)
                    .or_insert(velocity_bias.1);
                self.frame_accel_bias
                    .entry(frame_id)
                    .or_insert(velocity_bias.2);
            }
        }
        for state in &data.frame_states {
            if keyframes.contains(&state.frame_id) {
                self.frame_poses
                    .insert(state.frame_id, pose_to_se3(state.pose));
                self.frame_timestamps
                    .insert(state.frame_id, state.timestamp_ns);
            }
        }

        if self.factors.provenance_version.is_empty() {
            self.factors.provenance_version = factors.provenance_version.clone();
        }
        self.factors.relative_pose.extend(factors.relative_pose);
        self.factors.roll_pitch.extend(factors.roll_pitch);
        self.factors.ba_covisibility.extend(factors.ba_covisibility);

        self.accepted_packets += 1;
        Ok(NfrMapperIngestReport {
            input_size: process.input_size,
            output_size: process.output_size,
            accepted: true,
            frame_pose_count: self.frame_poses.len(),
            relative_pose_factor_count: self.factors.relative_pose.len(),
            roll_pitch_factor_count: self.factors.roll_pitch.len(),
            image_timestamp_count: self.img_data.len(),
        })
    }

    fn retain_images(&mut self, data: &MargData) {
        // Upstream stores one OpticalFlowInput vector for each timestamp.  The
        // versioned MargData contract already emits each vector in camera
        // order; retain that order and do not transform or filter samples.
        let mut packet_images = BTreeMap::<i64, Vec<OfImageData>>::new();
        for image in data.of_images.iter().cloned() {
            packet_images
                .entry(image.timestamp_ns)
                .or_default()
                .push(image);
        }
        for (timestamp, images) in packet_images {
            self.img_data.insert(timestamp, images);
        }
    }

    /// Source-compatible spelling for code following the pinned C++ API.
    #[allow(non_snake_case)]
    pub fn addMargData(
        &mut self,
        data: &mut MargData,
    ) -> Result<NfrMapperIngestReport, NfrMapperError> {
        self.add_marg_data(data)
    }

    /// Run the first stateful frontend stage after `addMargData`.
    ///
    /// The pinned implementation first enumerates image timestamps whose key
    /// is already present in `frame_poses`; the Rust MargData image contract
    /// stores that identity as `OfImageData::frame_id` while retaining the
    /// camera timestamp as the `img_data` bucket key.  Only images whose frame
    /// ID has an installed pose are converted to camera features,
    /// angle/descriptors, calibrated rays, and BoW entries.
    /// The generated map is committed only after every extraction succeeds so
    /// a malformed packet cannot leave a partially updated frontend state.
    pub fn detect_keypoints(&mut self) -> Result<NfrMapperDetectionReport, NfrMapperFeatureError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(NfrMapperFeatureError::MissingCalibration)?;
        let input_timestamp_count = self.img_data.len();
        let mut eligible_timestamp_count = 0;
        let mut processed_image_count = 0;
        let mut feature_count = 0;
        let mut generated = BTreeMap::new();

        for (&timestamp_ns, images) in &self.img_data {
            if !images
                .iter()
                .any(|image| self.frame_poses.contains_key(&image.frame_id))
            {
                continue;
            }
            eligible_timestamp_count += 1;

            for image in images {
                if !self.frame_poses.contains_key(&image.frame_id) {
                    continue;
                }
                let camera_id = image.camera_id;
                let camera =
                    calibration
                        .camera(camera_id)
                        .ok_or(NfrMapperFeatureError::MissingCamera {
                            timestamp_ns,
                            camera_id,
                        })?;
                let width =
                    usize::try_from(image.width).map_err(|_| NfrMapperFeatureError::Image {
                        timestamp_ns,
                        camera_id,
                        source: ImageError::DimensionOverflow,
                    })?;
                let height =
                    usize::try_from(image.height).map_err(|_| NfrMapperFeatureError::Image {
                        timestamp_ns,
                        camera_id,
                        source: ImageError::DimensionOverflow,
                    })?;
                let raw =
                    RawU16Image::new(width, height, image.data.clone()).map_err(|source| {
                        NfrMapperFeatureError::Image {
                            timestamp_ns,
                            camera_id,
                            source,
                        }
                    })?;
                let features = extract_mapper_features(&raw, camera, self.feature_config).map_err(
                    |source| NfrMapperFeatureError::Extraction {
                        timestamp_ns,
                        camera_id,
                        source,
                    },
                )?;
                feature_count += features.len();
                processed_image_count += 1;
                let key = TimeCamId::new(image.frame_id, camera_id);
                for entry in &features.bow_vector {
                    self.hash_index.entry(entry.hash).or_default().insert(key);
                }
                generated.insert(key, features);
            }
        }

        self.feature_corners.extend(generated);
        Ok(NfrMapperDetectionReport {
            input_timestamp_count,
            eligible_timestamp_count,
            processed_image_count,
            feature_count,
        })
    }

    /// Source-compatible spelling for the pinned C++ lifecycle method.
    #[allow(non_snake_case)]
    pub fn detectKeypoints(&mut self) -> Result<NfrMapperDetectionReport, NfrMapperFeatureError> {
        self.detect_keypoints()
    }

    /// Match the two calibrated cameras for every retained image timestamp.
    ///
    /// This follows the pinned `NfrMapper::match_stereo` order: derive
    /// `T_0_1 = T_i_c[0]⁻¹ T_i_c[1]`, run mutual descriptor matching, apply
    /// the known-pose essential gate at `1e-3`, and persist a pair only when
    /// the inlier count is strictly greater than sixteen.  Upstream uses
    /// `operator[]` on missing feature maps; the Rust map materializes empty
    /// products for the same pair instead of silently manufacturing matches.
    pub fn match_stereo(&mut self) -> Result<NfrMapperStereoReport, NfrMapperStereoError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(NfrMapperStereoError::MissingCalibration)?;
        // Monocular rig: no stereo pair exists, so the stage is a no-op and
        // every landmark comes from temporal matches (`match_all`).
        if calibration.cameras.len() < 2 {
            return Ok(NfrMapperStereoReport {
                input_timestamp_count: self.img_data.len(),
                attempted_timestamp_count: 0,
                missing_feature_timestamp_count: 0,
                raw_match_count: 0,
                essential_inlier_count: 0,
                accepted_pair_count: 0,
                total_pair_count: self.feature_matches.len(),
            });
        }
        let camera_0 = calibration
            .camera_to_imu(0)
            .ok_or(NfrMapperStereoError::MissingCamera { camera_id: 0 })?;
        let camera_1 = calibration
            .camera_to_imu(1)
            .ok_or(NfrMapperStereoError::MissingCamera { camera_id: 1 })?;
        let t_0_1 = camera_0.inverse().compose(camera_1);

        let input_timestamp_count = self.img_data.len();
        let mut attempted_timestamp_count = 0;
        let mut missing_feature_timestamp_count = 0;
        let mut raw_match_count = 0;
        let mut essential_inlier_count = 0;
        let mut accepted_pair_count = 0;

        for (&timestamp_ns, images) in &self.img_data {
            let Some(frame_id) = stereo_frame_id(timestamp_ns, images, 0) else {
                continue;
            };
            attempted_timestamp_count += 1;
            let left_id = TimeCamId::new(frame_id, 0);
            let right_id = TimeCamId::new(
                stereo_frame_id(timestamp_ns, images, 1).unwrap_or(frame_id),
                1,
            );
            let missing = !self.feature_corners.contains_key(&left_id)
                || !self.feature_corners.contains_key(&right_id);
            if missing {
                missing_feature_timestamp_count += 1;
                self.feature_corners
                    .entry(left_id)
                    .or_insert_with(empty_mapper_features);
                self.feature_corners
                    .entry(right_id)
                    .or_insert_with(empty_mapper_features);
            }

            let stereo = {
                let left = self
                    .feature_corners
                    .get(&left_id)
                    .expect("stereo left feature map materialized");
                let right = self
                    .feature_corners
                    .get(&right_id)
                    .expect("stereo right feature map materialized");
                match_stereo_features(left, right, &t_0_1, self.feature_config).map_err(
                    |source| NfrMapperStereoError::Pipeline {
                        timestamp_ns,
                        source,
                    },
                )?
            };
            raw_match_count += stereo.raw_matches.len();
            essential_inlier_count += stereo.essential_inliers.len();

            if stereo.mapper_feature_matches_stored {
                let inliers = stereo
                    .essential_inliers
                    .into_iter()
                    .map(|(left, right)| (left as u64, right as u64))
                    .collect::<Vec<_>>();
                self.feature_match_data.insert(
                    (left_id, right_id),
                    NfrMapperMatchData {
                        t_i_j: t_0_1.clone(),
                        matches: stereo
                            .raw_matches
                            .into_iter()
                            .map(|match_| (match_.left, match_.right))
                            .collect(),
                        inliers: inliers.clone(),
                    },
                );
                self.feature_matches
                    .insert((left_id, right_id), MatchData::new(inliers));
                accepted_pair_count += 1;
            }
        }

        Ok(NfrMapperStereoReport {
            input_timestamp_count,
            attempted_timestamp_count,
            missing_feature_timestamp_count,
            raw_match_count,
            essential_inlier_count,
            accepted_pair_count,
            total_pair_count: self.feature_matches.len(),
        })
    }

    /// Source-compatible spelling for the pinned C++ lifecycle method.
    #[allow(non_snake_case)]
    pub fn matchStereo(&mut self) -> Result<NfrMapperStereoReport, NfrMapperStereoError> {
        self.match_stereo()
    }

    /// Match all detected images using the pinned `NfrMapper::match_all`
    /// lifecycle.  HashBoW candidates are generated in feature-map key order,
    /// filtered by the strict source frame/score gates, then passed through
    /// mutual descriptor matching and relative-pose RANSAC.  Accepted pairs
    /// update both the track-builder projection and the full source payload.
    ///
    /// The production RANSAC helper keeps the pinned wall-clock seed behavior.
    /// Use [`Self::match_all_seeded`] for a deterministic release fixture.
    pub fn match_all(&mut self) -> NfrMapperMatchAllReport {
        self.match_all_impl(None)
    }

    /// Deterministic fixture form of [`Self::match_all`].  The explicit seed is
    /// passed unchanged to each candidate's OpenGV-compatible RANSAC call;
    /// all key, candidate, descriptor, and insertion ordering remains the
    /// same as the production method.
    pub fn match_all_seeded(&mut self, seed: u32) -> NfrMapperMatchAllReport {
        self.match_all_impl(Some(seed))
    }

    /// Source-compatible spelling for the pinned C++ lifecycle method.
    #[allow(non_snake_case)]
    pub fn matchAll(&mut self) -> NfrMapperMatchAllReport {
        self.match_all()
    }

    /// Source-compatible deterministic fixture spelling.
    #[allow(non_snake_case)]
    pub fn matchAllSeeded(&mut self, seed: u32) -> NfrMapperMatchAllReport {
        self.match_all_seeded(seed)
    }

    /// Build and filter persistent feature tracks in the exact source order:
    /// `TrackBuilder::Build(feature_matches)`, `Filter(min_track_length)`,
    /// then `Export(feature_tracks)`.  The builder is intentionally local for
    /// each pass, while the exported map is retained on `NfrMapper` exactly as
    /// upstream retains its `feature_tracks` member.
    pub fn build_tracks(&mut self) -> NfrMapperTrackReport {
        let input_pair_count = self.feature_matches.len();
        let inlier_match_count = self
            .feature_matches
            .values()
            .map(|match_data| match_data.inliers.len())
            .sum::<usize>();

        let mut builder = TrackBuilder::new();
        builder.Build(&self.feature_matches);
        let _ = builder.Filter(self.feature_config.min_track_length);
        let filter = builder
            .last_filter_report()
            .cloned()
            .expect("TrackBuilder::Filter records its source-compatible report");
        let mut exported = FeatureTracks::new();
        builder.Export(&mut exported);

        let total_track_obs_count = exported
            .values()
            .map(|observations| observations.len())
            .sum::<usize>();
        let exported_track_count = exported.len();
        let average_track_length = (exported_track_count != 0)
            .then(|| total_track_obs_count as f64 / exported_track_count as f64);
        self.feature_tracks = exported;

        NfrMapperTrackReport {
            input_pair_count,
            inlier_match_count,
            node_count: filter.node_count,
            component_count_before: filter.component_count_before,
            component_count_after: filter.component_count_after,
            rejected_conflict_ids: filter.rejected_conflict_ids.clone(),
            rejected_short_ids: filter.rejected_short_ids.clone(),
            rejected_track_ids: filter.rejected_track_ids.clone(),
            total_track_obs_count,
            average_track_length,
            exported_track_count,
            filter,
        }
    }

    /// Source-compatible spelling for the pinned C++ lifecycle method.
    #[allow(non_snake_case)]
    pub fn buildTracks(&mut self) -> NfrMapperTrackReport {
        self.build_tracks()
    }

    /// Initialize persistent mapper landmarks from the exported
    /// `feature_tracks` in the exact pinned `setup_opt` order.  The existing
    /// lower-level primitive owns the source gates and deterministic
    /// candidate decisions; this stateful boundary commits its accepted
    /// landmarks and host/target observation index atomically to `lmdb`.
    pub fn setup_opt(&mut self) -> Result<SetupOptReport, NfrMapperSetupOptError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(NfrMapperSetupOptError::MissingCalibration)?;
        let result = run_setup_opt(SetupOptInput::new(
            &self.feature_tracks,
            &self.feature_corners,
            &self.frame_poses,
            calibration,
            self.feature_config.min_triangulation_distance,
        ));
        let report = result.report.clone();
        self.lmdb = NfrMapperLandmarkDb::from_landmarks(result.landmarks);
        Ok(report)
    }

    /// Source-compatible spelling for the pinned C++ lifecycle method.
    #[allow(non_snake_case)]
    pub fn setupOpt(&mut self) -> Result<SetupOptReport, NfrMapperSetupOptError> {
        self.setup_opt()
    }

    /// Incremental local mapping for freshly matched keyframes.
    ///
    /// Unlike [`Self::setup_opt`] (a full track rebuild that *replaces*
    /// `lmdb`), this seeds `lmdb` continuously: for every match pair that
    /// touches one of `keys`, it either appends the new observation to an
    /// existing landmark or triangulates a new landmark with
    /// [`triangulate_pair`]'s exact `setup_opt` gates. The durable product is
    /// the `feature_matches` edge (which survives background merges); the
    /// periodic `setup_opt` remains the canonical reconciler and renumbers all
    /// ids, so incremental ids live in a disjoint namespace and are treated as
    /// transient.
    ///
    /// Returns `(attempted, accepted, rejected)` pair counts.
    pub fn local_map_new_keyframes(&mut self, keys: &[TimeCamId]) -> (usize, usize, usize) {
        let Some(calibration) = self.calibration.clone() else {
            return (0, 0, 0);
        };
        // `(image, feature) -> track_id` for landmarks already in the map.
        let mut owner: BTreeMap<(TimeCamId, FeatureId), u64> = BTreeMap::new();
        for (&track_id, landmark) in &self.lmdb.landmarks {
            for observation in &landmark.observations {
                owner.insert((observation.image, observation.feature_id), track_id);
            }
        }
        let min_distance = self.feature_config.min_triangulation_distance;
        // Disjoint from union-find-root ids (`setup_opt`), which are far below.
        let mut next_incremental_id: u64 = 1 << 62;
        let mut attempted = 0usize;
        let mut accepted = 0usize;
        let mut rejected = 0usize;
        let mut added_landmarks: Vec<(u64, MapperLandmark)> = Vec::new();
        let mut appended: Vec<(u64, MapperObservation)> = Vec::new();
        let mut touched_pairs: BTreeSet<ImagePair> = BTreeSet::new();

        let keys: BTreeSet<TimeCamId> = keys.iter().copied().collect();
        for (&(left, right), data) in &self.feature_matches {
            let query = if keys.contains(&right) {
                right
            } else if keys.contains(&left) {
                left
            } else {
                continue;
            };
            let other = if query == right { left } else { right };
            // Host must be the older image in TimeCamId order.
            let (host, second) = if other < query {
                (other, query)
            } else {
                (query, other)
            };
            if host == second {
                continue;
            }
            for &(left_feature, right_feature) in &data.inliers {
                // Map the pair's endpoints back onto (host, second).
                let (host_feature, second_feature) = if other == host {
                    (left_feature, right_feature)
                } else {
                    (right_feature, left_feature)
                };
                let host_owner = owner.get(&(host, host_feature)).copied();
                let second_owner = owner.get(&(second, second_feature)).copied();
                if let Some(track_id) = second_owner {
                    // The query feature already belongs to a landmark (e.g. a
                    // stereo/short link): re-observe it in `host` instead.
                    if host_owner != Some(track_id) {
                        if let Some(&pixel) = self.feature_corners.get(&host).and_then(|features| {
                            features.corners.get(usize::try_from(host_feature).ok()?)
                        }) {
                            appended.push((
                                track_id,
                                MapperObservation {
                                    image: host,
                                    feature_id: host_feature,
                                    pixel,
                                },
                            ));
                            owner.insert((host, host_feature), track_id);
                        }
                    }
                    continue;
                }
                if let Some(track_id) = host_owner {
                    // Symmetric case: re-observe the landmark in `second`.
                    if let Some(&pixel) = self.feature_corners.get(&second).and_then(|features| {
                        features.corners.get(usize::try_from(second_feature).ok()?)
                    }) {
                        appended.push((
                            track_id,
                            MapperObservation {
                                image: second,
                                feature_id: second_feature,
                                pixel,
                            },
                        ));
                        owner.insert((second, second_feature), track_id);
                    }
                    continue;
                }
                // Neither endpoint owned: triangulate a new landmark.
                attempted += 1;
                match triangulate_pair(
                    host,
                    host_feature,
                    second,
                    second_feature,
                    &self.feature_corners,
                    &self.frame_poses,
                    &calibration,
                    min_distance,
                ) {
                    Ok((direction, inverse_distance, host_obs, second_obs)) => {
                        let track_id = next_incremental_id;
                        next_incremental_id += 1;
                        owner.insert((host, host_feature), track_id);
                        owner.insert((second, second_feature), track_id);
                        added_landmarks.push((
                            track_id,
                            MapperLandmark {
                                track_id,
                                host,
                                second,
                                direction,
                                inverse_distance,
                                observations: vec![host_obs, second_obs],
                            },
                        ));
                        touched_pairs.insert((host, second));
                        accepted += 1;
                    }
                    Err(_reason) => rejected += 1,
                }
            }
        }

        if added_landmarks.is_empty() && appended.is_empty() {
            return (attempted, accepted, rejected);
        }
        for (track_id, landmark) in added_landmarks {
            self.lmdb.landmarks.insert(track_id, landmark);
        }
        for (track_id, observation) in appended {
            if let Some(landmark) = self.lmdb.landmarks.get_mut(&track_id) {
                if !landmark
                    .observations
                    .iter()
                    .any(|existing| existing.image == observation.image)
                {
                    landmark.observations.push(observation);
                }
            }
        }
        // Keep the reverse index consistent with `landmarks` (required by
        // `filter_outliers`/`get_current_points`; see `NfrMapperLandmarkDb`).
        self.lmdb.rebuild_observation_index();
        // The durable record is the match edge: the next full `setup_opt`
        // rebuilds tracks from it. Projection-only pairs are already in
        // `feature_matches`; nothing further to insert here.
        let _ = touched_pairs;
        (attempted, accepted, rejected)
    }

    /// Optimize the persistent mapper state in the pinned NFR order.
    ///
    /// The lower-level in-place BA primitive linearizes all landmark
    /// observations (the source `RelLinData` vector), then appends the
    /// recovered roll/pitch and relative-pose rows when configured.  Each LM
    /// trial applies the source-sign pose/landmark increment, evaluates the
    /// real-vs-updated cost, and either retains the maps or restores both
    /// backups before updating lambda/lambda-vee.  Because the maps are passed
    /// by mutable reference, accepted steps are persisted directly on
    /// `frame_poses` and `lmdb.landmarks`; the observation index is rebuilt
    /// afterward from the unchanged full observation payload.
    pub fn optimize(
        &mut self,
        num_iterations: usize,
    ) -> Result<NfrMapperOptimizeReport, NfrMapperOptimizeError> {
        self.optimize_with_extra_factors(&[], num_iterations)
    }

    /// Global BA with transient extra relative-pose factors appended to the
    /// persistent factor set (which is left untouched).
    ///
    /// This is how the online mapper injects **loop-closure** factors: they are
    /// re-derived from the current map after every `setup_opt`, so they must not
    /// accumulate on `self.factors` the way the VIO marginalization factors do.
    pub fn optimize_with_extra_factors(
        &mut self,
        extra_relative_pose: &[RelativePoseFactor],
        num_iterations: usize,
    ) -> Result<NfrMapperOptimizeReport, NfrMapperOptimizeError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(NfrMapperOptimizeError::MissingCalibration)?;
        let requested_iterations = num_iterations;
        let initial_lambda = self.optimizer_state.lambda;
        let initial_lambda_vee = self.optimizer_state.lambda_vee;
        let mut config = self.optimize_config;
        config.max_iterations = requested_iterations;
        let mut factors = self.factors.clone();
        factors.relative_pose.extend_from_slice(extra_relative_pose);
        let summary: MapperSummary = global_ba_with_state(
            &mut self.frame_poses,
            &factors,
            &mut self.lmdb.landmarks,
            calibration,
            config,
            &mut self.optimizer_state,
        );
        self.lmdb.rebuild_observation_index();

        let accepted_step_count = summary
            .trace
            .iter()
            .flat_map(|iteration| iteration.trials.iter())
            .filter(|trial| trial.accepted)
            .count();
        let rejected_trial_count = summary
            .trace
            .iter()
            .flat_map(|iteration| iteration.trials.iter())
            .filter(|trial| !trial.accepted)
            .count();
        Ok(NfrMapperOptimizeReport {
            requested_iterations,
            iterations: summary.iterations,
            pose_count: summary.pose_count,
            landmark_count: summary.track_count,
            initial_cost: summary.initial_cost,
            final_cost: summary.final_cost,
            accepted_step_count,
            rejected_trial_count,
            initial_lambda,
            min_lambda: self.optimizer_state.min_lambda,
            max_lambda: self.optimizer_state.max_lambda,
            initial_lambda_vee,
            final_lambda: self.optimizer_state.lambda,
            final_lambda_vee: self.optimizer_state.lambda_vee,
            final_state_hash: summary.final_state_hash,
            trace_hash: summary.trace_hash,
            trace: summary.trace,
        })
    }

    /// Joint visual-inertial global BA: like [`Self::optimize_with_extra_factors`],
    /// but solves the full 15-dof-per-keyframe navigation state (pose,
    /// velocity, gyro bias, accel bias) via
    /// [`super::global_ba_with_state_joint`], adding `imu_factors`
    /// (preintegrated relative-state) and `bias_factors` (bias random-walk)
    /// between consecutive keyframes on top of the same persistent
    /// vision/relative-pose/roll-pitch factor set. `frame_velocities`,
    /// `frame_gyro_bias` and `frame_accel_bias` are updated in place;
    /// missing entries (a keyframe never seeded with a VIO velocity/bias
    /// estimate) default to zero, matching
    /// `global_ba_with_state_joint`'s own contract.
    ///
    /// `final_state_hash`/`trace_hash` on the returned report are always
    /// `0`: this path keeps no per-iteration LM trace (see
    /// [`super::JointGlobalBaSummary`]), unlike the pose-only solver's
    /// provenance-hashed trace.
    pub fn optimize_joint_with_extra_factors(
        &mut self,
        extra_relative_pose: &[RelativePoseFactor],
        imu_factors: &[super::imu_ba::ImuPreintegratedFactor],
        bias_factors: &[super::imu_ba::BiasRandomWalkFactor],
        gravity_world: Vector3<f64>,
        num_iterations: usize,
    ) -> Result<NfrMapperOptimizeReport, NfrMapperOptimizeError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(NfrMapperOptimizeError::MissingCalibration)?;
        let requested_iterations = num_iterations;
        let initial_lambda = self.optimizer_state.lambda;
        let initial_lambda_vee = self.optimizer_state.lambda_vee;
        let mut config = self.optimize_config;
        config.max_iterations = requested_iterations;
        let mut factors = self.factors.clone();
        factors.relative_pose.extend_from_slice(extra_relative_pose);
        let summary = super::global_ba_with_state_joint(
            &mut self.frame_poses,
            &mut self.frame_velocities,
            &mut self.frame_gyro_bias,
            &mut self.frame_accel_bias,
            &factors,
            imu_factors,
            bias_factors,
            &mut self.lmdb.landmarks,
            calibration,
            config,
            gravity_world,
            &mut self.optimizer_state,
        );
        self.lmdb.rebuild_observation_index();
        if std::env::var_os("BASALT_ONLINE_MAPPER_TRACE").is_some() {
            eprintln!(
                "[mapper-trace] joint_ba cost: pose_factor={:.3} imu_factor={:.3} total={:.3}->{:.3} \
                 accepted={} rejected={}",
                summary.initial_pose_factor_cost,
                summary.initial_imu_factor_cost,
                summary.initial_cost,
                summary.final_cost,
                summary.accepted_step_count,
                summary.rejected_trial_count,
            );
        }

        Ok(NfrMapperOptimizeReport {
            requested_iterations,
            iterations: summary.iterations,
            pose_count: summary.pose_count,
            landmark_count: summary.track_count,
            initial_cost: summary.initial_cost,
            final_cost: summary.final_cost,
            accepted_step_count: summary.accepted_step_count,
            rejected_trial_count: summary.rejected_trial_count,
            initial_lambda,
            min_lambda: self.optimizer_state.min_lambda,
            max_lambda: self.optimizer_state.max_lambda,
            initial_lambda_vee,
            final_lambda: self.optimizer_state.lambda,
            final_lambda_vee: self.optimizer_state.lambda_vee,
            final_state_hash: 0,
            trace_hash: 0,
            trace: Vec::new(),
        })
    }

    /// Source-compatible spelling for the pinned C++ lifecycle method.
    #[allow(non_snake_case)]
    pub fn optimizeNfr(
        &mut self,
        num_iterations: usize,
    ) -> Result<NfrMapperOptimizeReport, NfrMapperOptimizeError> {
        self.optimize(num_iterations)
    }

    /// Select a covisibility window around the newest keyframe: the newest
    /// `window_keyframes` keyframes ranked by shared-landmark count with the
    /// active frame, plus the active frame itself.
    ///
    /// Returns frame IDs (not image IDs). Empty when there are fewer than two
    /// poses or no landmarks.
    pub fn covisibility_window(&self, active: u64, window_keyframes: usize) -> BTreeSet<u64> {
        let mut window = BTreeSet::new();
        window.insert(active);
        if window_keyframes == 0 {
            return window;
        }
        // Count shared landmarks between the active frame and every other frame.
        let mut shared: BTreeMap<u64, usize> = BTreeMap::new();
        let mut active_landmarks: BTreeSet<u64> = BTreeSet::new();
        for (&track_id, landmark) in &self.lmdb.landmarks {
            let touches_active = landmark
                .observations
                .iter()
                .any(|o| o.image.frame_id == active)
                || landmark.host.frame_id == active;
            if !touches_active {
                continue;
            }
            active_landmarks.insert(track_id);
            // Every other frame observing this landmark shares it.
            for observation in &landmark.observations {
                if observation.image.frame_id != active {
                    *shared.entry(observation.image.frame_id).or_default() += 1;
                }
            }
            if landmark.host.frame_id != active {
                *shared.entry(landmark.host.frame_id).or_default() += 1;
            }
        }
        let _ = active_landmarks;
        // Rank by shared count, tie-break by frame id for determinism.
        let mut ranked = shared.into_iter().collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        for (frame_id, _) in ranked.into_iter().take(window_keyframes) {
            window.insert(frame_id);
        }
        window
    }

    /// Local bundle adjustment over the covisibility window around the newest
    /// keyframe. Reuses the global-BA factor machinery on the window's pose
    /// block (see [`local_ba_with_state`]).
    pub fn optimize_local_window(
        &mut self,
        window_keyframes: usize,
        num_iterations: usize,
    ) -> Result<NfrMapperOptimizeReport, NfrMapperOptimizeError> {
        let calibration = self
            .calibration
            .as_ref()
            .ok_or(NfrMapperOptimizeError::MissingCalibration)?;
        let Some(&active) = self.frame_poses.keys().next_back() else {
            return Err(NfrMapperOptimizeError::MissingCalibration);
        };
        let window = self.covisibility_window(active, window_keyframes);
        let requested_iterations = num_iterations;
        let initial_lambda = self.optimizer_state.lambda;
        let initial_lambda_vee = self.optimizer_state.lambda_vee;
        let mut config = self.optimize_config;
        config.max_iterations = requested_iterations;
        let summary = local_ba_with_state(
            &mut self.frame_poses,
            &self.factors,
            &mut self.lmdb.landmarks,
            &window,
            calibration,
            config,
            &mut self.optimizer_state,
        );
        self.lmdb.rebuild_observation_index();
        let accepted_step_count = summary
            .trace
            .iter()
            .flat_map(|iteration| iteration.trials.iter())
            .filter(|trial| trial.accepted)
            .count();
        let rejected_trial_count = summary
            .trace
            .iter()
            .flat_map(|iteration| iteration.trials.iter())
            .filter(|trial| !trial.accepted)
            .count();
        Ok(NfrMapperOptimizeReport {
            requested_iterations,
            iterations: summary.iterations,
            pose_count: summary.pose_count,
            landmark_count: summary.track_count,
            initial_cost: summary.initial_cost,
            final_cost: summary.final_cost,
            accepted_step_count,
            rejected_trial_count,
            initial_lambda,
            min_lambda: self.optimizer_state.min_lambda,
            max_lambda: self.optimizer_state.max_lambda,
            initial_lambda_vee,
            final_lambda: self.optimizer_state.lambda,
            final_lambda_vee: self.optimizer_state.lambda_vee,
            final_state_hash: summary.final_state_hash,
            trace_hash: summary.trace_hash,
            trace: summary.trace,
        })
    }

    /// Return the current world-point payload using the inherited
    /// `BundleAdjustmentBase::get_current_points` convention. The source
    /// traverses every host/target observation-index entry rather than each
    /// unique landmark, so a landmark is intentionally emitted once per
    /// indexed target observation. Rust keeps deterministic host, target, and
    /// track ordering and assigns every entry the source visualization ID `1`.
    pub fn get_current_points(&self) -> NfrMapperCurrentPoints {
        let mut points = Vec::new();
        let mut ids = Vec::new();

        for &host in self.lmdb.observations.keys() {
            let Some(pose) = self.frame_poses.get(&host.frame_id) else {
                continue;
            };
            let Some(camera_to_imu) = self
                .calibration
                .as_ref()
                .and_then(|calibration| calibration.camera_to_imu(host.cam_id))
            else {
                continue;
            };
            let world_from_camera = pose.compose(camera_to_imu);
            for track_ids in self
                .lmdb
                .observations
                .get(&host)
                .into_iter()
                .flat_map(|targets| targets.values())
            {
                for &track_id in track_ids {
                    let Some(landmark) = self.lmdb.landmarks.get(&track_id) else {
                        continue;
                    };
                    let Some(position_in_host) = landmark.position_in_host() else {
                        continue;
                    };
                    let point = world_from_camera
                        .transform_point(&nalgebra::Point3::from(position_in_host));
                    if !point.coords.iter().all(|value| value.is_finite()) {
                        continue;
                    }
                    points.push([point.x, point.y, point.z]);
                    ids.push(1);
                }
            }
        }

        NfrMapperCurrentPoints { points, ids }
    }

    /// Source-compatible spelling for the inherited helper.
    #[allow(non_snake_case)]
    pub fn getCurrentPoints(&self) -> NfrMapperCurrentPoints {
        self.get_current_points()
    }

    /// Compute the current robust reprojection cost without changing mapper
    /// state.  This is the `computeError(error)` half of the source base-class
    /// boundary; filtering uses the same traversal with an outlier map.
    pub fn compute_reprojection_error(&self) -> f64 {
        self.reprojection_diagnostics(f64::INFINITY).0
    }

    /// Source-compatible diagnostic alias.
    #[allow(non_snake_case)]
    pub fn computeError(&self) -> f64 {
        self.compute_reprojection_error()
    }

    /// Port `BundleAdjustmentBase::filterOutliers` exactly: score every
    /// observation in host/target/track order, force-remove a landmark when a
    /// same-frame host observation is bad (`-2` in Basalt), otherwise remove
    /// only target-image observations, and commit all removals after the
    /// ordered decision pass.
    /// Test-only copy of the previous `filter_outliers` removal loop, which
    /// rebuilt the observation index after every removal. It exists so the
    /// batched rebuild can be checked against it.
    #[cfg(test)]
    pub(crate) fn filter_outliers_per_removal_reference(
        &mut self,
        outlier_threshold: f64,
        min_num_obs: usize,
    ) {
        let (_, outliers, _) = self.reprojection_diagnostics(outlier_threshold);
        for (&track_id, entries) in &outliers {
            let num_observations = self
                .lmdb
                .num_landmark_observations(track_id)
                .unwrap_or_default();
            let remove_landmark = num_observations.saturating_sub(entries.len()) < min_num_obs
                || entries.iter().any(|(_, marker, _)| *marker == -2.0);
            if remove_landmark {
                self.lmdb.remove_landmark(track_id);
            } else {
                let target_ids = entries
                    .iter()
                    .map(|(target, _, _)| *target)
                    .collect::<BTreeSet<_>>();
                self.lmdb.remove_observations(track_id, &target_ids);
            }
        }
    }

    pub fn filter_outliers(
        &mut self,
        outlier_threshold: f64,
        min_num_obs: usize,
    ) -> Result<NfrMapperFilterReport, NfrMapperOptimizeError> {
        if self.calibration.is_none() {
            return Err(NfrMapperOptimizeError::MissingCalibration);
        }
        let before_landmark_count = self.lmdb.num_landmarks();
        let before_observation_count = self.lmdb.num_observations();
        let (reprojection_error, outliers, invalid_observation_count) =
            self.reprojection_diagnostics(outlier_threshold);
        let mut removed_landmark_count = 0;
        let mut removed_observation_count = 0;
        let mut outlier_observation_count = 0;

        for (&track_id, entries) in &outliers {
            outlier_observation_count += entries.len();
            let num_observations = self
                .lmdb
                .num_landmark_observations(track_id)
                .unwrap_or_default();
            let remove_landmark = num_observations.saturating_sub(entries.len()) < min_num_obs
                || entries.iter().any(|(_, marker, _)| *marker == -2.0);
            // Removals are applied in the same ordered landmark loop, but the
            // host/target observation index is rebuilt once afterwards. The
            // index is a pure function of `landmarks` (`from_landmarks`), the
            // raw removals read only `landmarks`, and nothing observes the
            // index mid-loop, so the final state is identical to rebuilding
            // after every removal. Per-removal rebuilds made this pass
            // quadratic (about 25 s per call on MH_01).
            if remove_landmark {
                if self.lmdb.remove_landmark_raw(track_id) {
                    removed_landmark_count += 1;
                }
            } else {
                let target_ids = entries
                    .iter()
                    .map(|(target, _, _)| *target)
                    .collect::<BTreeSet<_>>();
                removed_observation_count +=
                    self.lmdb.remove_observations_raw(track_id, &target_ids);
            }
        }
        if removed_landmark_count > 0 || removed_observation_count > 0 {
            self.lmdb.rebuild_observation_index();
        }

        Ok(NfrMapperFilterReport {
            outlier_threshold,
            min_num_obs,
            reprojection_error,
            before_landmark_count,
            after_landmark_count: self.lmdb.num_landmarks(),
            before_observation_count,
            after_observation_count: self.lmdb.num_observations(),
            candidate_landmark_count: outliers.len(),
            outlier_observation_count,
            invalid_observation_count,
            removed_landmark_count,
            removed_observation_count,
        })
    }

    /// Source-compatible spelling for `filterOutliers`.
    #[allow(non_snake_case)]
    pub fn filterOutliers(
        &mut self,
        outlier_threshold: f64,
        min_num_obs: usize,
    ) -> Result<NfrMapperFilterReport, NfrMapperOptimizeError> {
        self.filter_outliers(outlier_threshold, min_num_obs)
    }

    /// Execute the headless mapper workflow in the same order as the pinned
    /// Qt mapping application: clear/detect, clear/stereo/temporal matching,
    /// tracks/setup/points, optimize, filter/points, optimize, then final
    /// points/result.
    pub fn run_headless(
        &mut self,
        config: NfrMapperHeadlessConfig,
    ) -> Result<NfrMapperHeadlessReport, NfrMapperHeadlessError> {
        self.feature_corners.clear();
        self.hash_index.clear();
        self.feature_matches.clear();
        self.feature_match_data.clear();
        let detection = self
            .detect_keypoints()
            .map_err(NfrMapperHeadlessError::Detection)?;

        self.feature_matches.clear();
        self.feature_match_data.clear();
        let stereo = self
            .match_stereo()
            .map_err(NfrMapperHeadlessError::Stereo)?;
        let match_all = match config.temporal_seed {
            Some(seed) => self.match_all_seeded(seed),
            None => self.match_all(),
        };
        let tracks = self.build_tracks();
        let setup = self.setup_opt().map_err(NfrMapperHeadlessError::Setup)?;
        let initial_points = self.get_current_points();
        let mut stages = vec![self.headless_stage(
            "setup_opt_get_points",
            initial_points.points.len(),
            Some(self.compute_reprojection_error()),
        )];

        let first_optimize = self
            .optimize(config.num_opt_iter)
            .map_err(NfrMapperHeadlessError::Optimize)?;
        // `mapper.cpp::optimize()` immediately refreshes the visualization
        // point buffers after BA.  Keep that helper boundary explicit even
        // though the compact report only needs the count.
        let optimized_points = self.get_current_points();
        stages.push(self.headless_stage(
            "optimize",
            optimized_points.points.len(),
            Some(first_optimize.final_cost),
        ));

        let filter = self
            .filter_outliers(config.outlier_threshold, config.min_num_obs)
            .map_err(NfrMapperHeadlessError::Optimize)?;
        let filtered_points = self.get_current_points();
        stages.push(self.headless_stage(
            "filter_get_points",
            filtered_points.points.len(),
            Some(filter.reprojection_error),
        ));

        let second_optimize = self
            .optimize(config.num_opt_iter)
            .map_err(NfrMapperHeadlessError::Optimize)?;
        let optimized_filtered_points = self.get_current_points();
        stages.push(self.headless_stage(
            "optimize_filtered",
            optimized_filtered_points.points.len(),
            Some(second_optimize.final_cost),
        ));
        let final_points = self.get_current_points();
        let result = self.result();
        stages.push(self.headless_stage(
            "final_get_points_result",
            final_points.points.len(),
            Some(second_optimize.final_cost),
        ));

        Ok(NfrMapperHeadlessReport {
            config,
            detection,
            stereo,
            match_all,
            tracks,
            setup,
            first_optimize,
            filter,
            second_optimize,
            initial_points,
            filtered_points,
            final_points,
            result,
            stages,
        })
    }

    /// Source-oriented alias for callers that use the CLI's mapping name.
    pub fn run_mapping(
        &mut self,
        config: NfrMapperHeadlessConfig,
    ) -> Result<NfrMapperHeadlessReport, NfrMapperHeadlessError> {
        self.run_headless(config)
    }

    #[allow(non_snake_case)]
    pub fn runHeadless(
        &mut self,
        config: NfrMapperHeadlessConfig,
    ) -> Result<NfrMapperHeadlessReport, NfrMapperHeadlessError> {
        self.run_headless(config)
    }

    fn headless_stage(
        &self,
        name: &str,
        point_count: usize,
        reprojection_error: Option<f64>,
    ) -> NfrMapperHeadlessStage {
        NfrMapperHeadlessStage {
            name: name.into(),
            pose_count: self.frame_poses.len(),
            landmark_count: self.lmdb.num_landmarks(),
            observation_count: self.lmdb.num_observations(),
            point_count,
            reprojection_error,
        }
    }

    fn reprojection_diagnostics(
        &self,
        outlier_threshold: f64,
    ) -> (f64, BTreeMap<u64, Vec<(TimeCamId, f64, bool)>>, usize) {
        let pose_indices = self
            .frame_poses
            .keys()
            .enumerate()
            .map(|(index, &frame_id)| (frame_id, index))
            .collect::<BTreeMap<_, _>>();
        let mut error = 0.0;
        let mut outliers = BTreeMap::<u64, Vec<(TimeCamId, f64, bool)>>::new();
        let mut invalid_observation_count = 0;
        let Some(calibration) = self.calibration.as_ref() else {
            return (error, outliers, invalid_observation_count);
        };

        for (&host, targets) in &self.lmdb.observations {
            for (&target, track_ids) in targets {
                for &track_id in track_ids {
                    let Some(landmark) = self.lmdb.landmarks.get(&track_id) else {
                        continue;
                    };
                    let Some(observation) = landmark
                        .observations
                        .iter()
                        .find(|observation| observation.image == target)
                    else {
                        let marker = if host == target { -2.0 } else { -1.0 };
                        if marker == -1.0 {
                            invalid_observation_count += 1;
                        }
                        outliers
                            .entry(track_id)
                            .or_default()
                            .push((target, marker, true));
                        continue;
                    };
                    let Some((residual, _, _)) = linearize_mapper_observation(
                        landmark,
                        observation,
                        &self.frame_poses,
                        calibration,
                        &pose_indices,
                    ) else {
                        let marker = if host == target { -2.0 } else { -1.0 };
                        invalid_observation_count += 1;
                        outliers
                            .entry(track_id)
                            .or_default()
                            .push((target, marker, true));
                        continue;
                    };
                    let e = residual.norm();
                    if !e.is_finite() {
                        let marker = if host == target { -2.0 } else { -1.0 };
                        invalid_observation_count += 1;
                        outliers
                            .entry(track_id)
                            .or_default()
                            .push((target, marker, true));
                        continue;
                    }
                    if e > outlier_threshold {
                        let marker = if host == target { -2.0 } else { e };
                        outliers
                            .entry(track_id)
                            .or_default()
                            .push((target, marker, false));
                    }
                    let huber_weight = if e < self.optimize_config.huber_delta {
                        1.0
                    } else {
                        self.optimize_config.huber_delta / e
                    };
                    let obs_weight = huber_weight
                        / (self.optimize_config.observation_std_dev
                            * self.optimize_config.observation_std_dev);
                    if obs_weight.is_finite() {
                        error += 0.5 * (2.0 - huber_weight) * obs_weight * residual.dot(&residual);
                    }
                }
            }
        }
        (error, outliers, invalid_observation_count)
    }

    /// Return the installed pose map, matching the mutable C++
    /// `NfrMapper::getFramePoses()` accessor without exposing a second copy of
    /// the optimizer state.
    pub const fn get_frame_poses(&self) -> &BTreeMap<u64, SE3> {
        &self.frame_poses
    }

    /// Mutable pose-map accessor for callers that follow the upstream helper's
    /// non-const return type.  BA itself updates this same map in place.
    pub const fn get_frame_poses_mut(&mut self) -> &mut BTreeMap<u64, SE3> {
        &mut self.frame_poses
    }

    /// Source-compatible spelling for `NfrMapper::getFramePoses()`.
    #[allow(non_snake_case)]
    pub const fn getFramePoses(&mut self) -> &mut BTreeMap<u64, SE3> {
        self.get_frame_poses_mut()
    }

    /// Compute the source relative-pose diagnostic
    /// `Σ rᵀ cov_inv r` over the currently installed poses.
    ///
    /// The C++ implementation assumes every factor endpoint is present.  The
    /// Rust boundary keeps packet recovery tolerant of stale/partial data, so
    /// malformed information blocks or missing endpoints are skipped in the
    /// same way as BA linearization.  Factor weights are retained because the
    /// Rust factor contract stores them separately from `information`.
    pub fn compute_rel_pose(&self) -> f64 {
        self.factors
            .relative_pose
            .iter()
            .filter_map(|factor| {
                let pose_i = self.frame_poses.get(&factor.from)?;
                let pose_j = self.frame_poses.get(&factor.to)?;
                if factor.information.len() != 36 {
                    return None;
                }
                let info = Matrix6::from_row_slice(&factor.information);
                let measured = pose_to_se3([
                    factor.translation[0],
                    factor.translation[1],
                    factor.translation[2],
                    factor.rotation[0],
                    factor.rotation[1],
                    factor.rotation[2],
                    factor.rotation[3],
                ]);
                let (residual, _, _) = rel_pose_error(
                    pose_array_from_se3(&measured),
                    pose_array_from_se3(pose_i),
                    pose_array_from_se3(pose_j),
                );
                let residual = Vector6::from_row_slice(&residual);
                Some(factor.weight * (residual.transpose() * info * residual)[(0, 0)])
            })
            .sum()
    }

    /// Source-compatible spelling for `NfrMapper::computeRelPose`.
    #[allow(non_snake_case)]
    pub fn computeRelPose(&self) -> f64 {
        self.compute_rel_pose()
    }

    /// C++-shaped output-parameter form for callers porting the diagnostic
    /// helper literally.
    pub fn compute_rel_pose_into(&self, rel_error: &mut f64) {
        *rel_error = self.compute_rel_pose();
    }

    #[allow(non_snake_case)]
    pub fn computeRelPoseInto(&self, rel_error: &mut f64) {
        self.compute_rel_pose_into(rel_error);
    }

    /// Compute the source roll/pitch diagnostic
    /// `Σ rᵀ cov_inv r` over the currently installed poses.
    pub fn compute_roll_pitch(&self) -> f64 {
        self.factors
            .roll_pitch
            .iter()
            .filter_map(|factor| {
                let pose = self.frame_poses.get(&factor.frame_id)?;
                let measured = [
                    [
                        factor.measured_rotation[0],
                        factor.measured_rotation[1],
                        factor.measured_rotation[2],
                    ],
                    [
                        factor.measured_rotation[3],
                        factor.measured_rotation[4],
                        factor.measured_rotation[5],
                    ],
                    [
                        factor.measured_rotation[6],
                        factor.measured_rotation[7],
                        factor.measured_rotation[8],
                    ],
                ];
                let (residual, _) = roll_pitch_error(pose_array_from_se3(pose), measured);
                let residual = Vector2::from_row_slice(&residual);
                let info = Matrix2::from_row_slice(&factor.information);
                Some(factor.weight * (residual.transpose() * info * residual)[(0, 0)])
            })
            .sum()
    }

    /// Source-compatible spelling for `NfrMapper::computeRollPitch`.
    #[allow(non_snake_case)]
    pub fn computeRollPitch(&self) -> f64 {
        self.compute_roll_pitch()
    }

    /// C++-shaped output-parameter form for callers porting the diagnostic
    /// helper literally.
    pub fn compute_roll_pitch_into(&self, roll_pitch_error: &mut f64) {
        *roll_pitch_error = self.compute_roll_pitch();
    }

    #[allow(non_snake_case)]
    pub fn computeRollPitchInto(&self, roll_pitch_error: &mut f64) {
        self.compute_roll_pitch_into(roll_pitch_error);
    }

    /// Export poses in mapper frame-ID order.  This is the stable JSON result
    /// order used by the headless mapper artifacts; trajectory text exports
    /// below sort by camera timestamp as Basalt's `saveTrajectoryButton` does.
    pub fn export_poses(&self) -> Vec<NfrMapperPoseRecord> {
        self.frame_poses
            .iter()
            .map(|(&frame_id, pose)| NfrMapperPoseRecord {
                frame_id,
                timestamp_ns: self
                    .frame_timestamps
                    .get(&frame_id)
                    .copied()
                    .unwrap_or_else(|| i64::try_from(frame_id).unwrap_or(i64::MAX)),
                translation: [pose.translation.x, pose.translation.y, pose.translation.z],
                quaternion_wxyz: [
                    pose.rotation.w,
                    pose.rotation.i,
                    pose.rotation.j,
                    pose.rotation.k,
                ],
            })
            .collect()
    }

    /// Source-oriented alias used by result writers.
    pub fn frame_pose_records(&self) -> Vec<NfrMapperPoseRecord> {
        self.export_poses()
    }

    /// Export compact map records in ascending source track/root order.
    pub fn export_map(&self) -> Vec<NfrMapperLandmarkRecord> {
        self.lmdb
            .landmarks
            .values()
            .map(|landmark| NfrMapperLandmarkRecord {
                track_id: landmark.track_id,
                host: NfrMapperImageRecord {
                    frame_id: landmark.host.frame_id,
                    camera_id: landmark.host.cam_id,
                },
                second: NfrMapperImageRecord {
                    frame_id: landmark.second.frame_id,
                    camera_id: landmark.second.cam_id,
                },
                direction: [landmark.direction.xy.x, landmark.direction.xy.y],
                inverse_distance: landmark.inverse_distance,
                observation_count: landmark.observations.len(),
            })
            .collect()
    }

    /// Snapshot the optimized pose/map result without transferring ownership
    /// of the live mapper state.
    pub fn result(&self) -> NfrMapperResult {
        NfrMapperResult {
            poses: self.export_poses(),
            landmarks: self.export_map(),
        }
    }

    /// Return poses in the timestamp order used by trajectory files.  A
    /// frame-ID tie is resolved by the stable mapper key order.
    pub fn trajectory_poses(&self) -> Vec<NfrMapperPoseRecord> {
        let mut poses = self.export_poses();
        poses.sort_by_key(|pose| (pose.timestamp_ns, pose.frame_id));
        poses
    }

    /// Serialize the Basalt/EuRoC trajectory convention:
    /// `timestamp_ns,tx,ty,tz,qw,qx,qy,qz`.
    pub fn trajectory_euroc(&self) -> String {
        let mut output = String::from(
            "#timestamp [ns],p_RS_R_x [m],p_RS_R_y [m],p_RS_R_z [m],q_RS_w [],q_RS_x [],q_RS_y [],q_RS_z []\n",
        );
        for pose in self.trajectory_poses() {
            output.push_str(&format!(
                "{},{:.18e},{:.18e},{:.18e},{:.18e},{:.18e},{:.18e},{:.18e}\n",
                pose.timestamp_ns,
                pose.translation[0],
                pose.translation[1],
                pose.translation[2],
                pose.quaternion_wxyz[0],
                pose.quaternion_wxyz[1],
                pose.quaternion_wxyz[2],
                pose.quaternion_wxyz[3],
            ));
        }
        output
    }

    /// Serialize the Basalt/TUM trajectory convention:
    /// `timestamp_seconds tx ty tz qx qy qz qw`.
    pub fn trajectory_tum(&self) -> String {
        let mut output = String::from("# timestamp tx ty tz qx qy qz qw\n");
        for pose in self.trajectory_poses() {
            let timestamp_seconds = pose.timestamp_ns as f64 * 1e-9;
            output.push_str(&format!(
                "{:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e} {:.18e}\n",
                timestamp_seconds,
                pose.translation[0],
                pose.translation[1],
                pose.translation[2],
                pose.quaternion_wxyz[1],
                pose.quaternion_wxyz[2],
                pose.quaternion_wxyz[3],
                pose.quaternion_wxyz[0],
            ));
        }
        output
    }

    /// Camel-case aliases for the headless result/trajectory boundary.
    #[allow(non_snake_case)]
    pub fn exportPoses(&self) -> Vec<NfrMapperPoseRecord> {
        self.export_poses()
    }

    #[allow(non_snake_case)]
    pub fn exportMap(&self) -> Vec<NfrMapperLandmarkRecord> {
        self.export_map()
    }

    #[allow(non_snake_case)]
    pub fn trajectoryEuRoC(&self) -> String {
        self.trajectory_euroc()
    }

    #[allow(non_snake_case)]
    pub fn trajectoryTUM(&self) -> String {
        self.trajectory_tum()
    }

    /// Build the candidate list for one `match_all` query using the
    /// inverted `hash_index` instead of a full scan of every detected image.
    ///
    /// Exact reduction, not an approximation: `query_bow_candidates` only
    /// ever scores a database entry when it shares a hash bucket with the
    /// query (see `hash_index`'s field doc), so restricting the scanned set
    /// to the union of this query's own hash-bucket members is guaranteed to
    /// include every image that could possibly score, and excludes only
    /// images that would have scored zero/unshared anyway.  Exposed as its
    /// own method (rather than inlined in `match_all_impl`) so
    /// `tests::inverted_index_candidates_equal_full_scan` can compare it
    /// directly against a full-scan call with the same inputs -- mirrors
    /// `OnlineNfrMapper::bow_candidates_via_index`
    /// (`pipelines/basalt/src/mapper/online.rs`, commit 8b55648).
    fn bow_candidates_via_index(
        &self,
        query_id: TimeCamId,
        query_features: &MapperImageFeatures,
        match_window: usize,
    ) -> Vec<BowQueryCandidate> {
        let candidate_ids = query_features
            .bow_vector
            .iter()
            .filter_map(|entry| self.hash_index.get(&entry.hash))
            .flatten()
            .copied()
            .collect::<BTreeSet<TimeCamId>>();
        let database = candidate_ids
            .iter()
            .filter_map(|&id| {
                self.feature_corners
                    .get(&id)
                    .map(|features| (MapperImageId::from(id), features))
            })
            .collect::<Vec<_>>();
        query_bow_candidates(
            MapperImageId::from(query_id),
            query_features,
            &database,
            match_window,
        )
    }

    fn match_all_impl(&mut self, seed: Option<u32>) -> NfrMapperMatchAllReport {
        // The pinned source first materializes a key vector and an
        // id-to-index map from feature_corners.  BTreeMap gives this Rust
        // stateful port a stable equivalent of that explicit source order.
        let keys = self.feature_corners.keys().copied().collect::<Vec<_>>();
        let id_to_key_idx = keys
            .iter()
            .enumerate()
            .map(|(index, &id)| (id, index))
            .collect::<BTreeMap<_, _>>();
        let input_feature_count = keys.len();
        let config = self.feature_config;
        // The pinned `match_all` call site intentionally hardcodes the
        // descriptor matcher controls (`70`, `1.2`), even though
        // `match_stereo` reads its values from the mapper config.  Preserve
        // that lifecycle distinction while retaining the configured BoW,
        // min-match, and geometric gates below.
        let mut match_config = config;
        match_config.max_hamming = 70;
        match_config.second_best_ratio = 1.2;

        // HashBow::querry_database is represented by the existing lower-level
        // helper.  It excludes candidates at or after the query frame (the
        // source passes max_t_ns=&tcid.frame_id); retain the explicit same-
        // frame check below as the source's second gate.
        //
        // `query_bow_candidates` only ever scores a database entry that
        // shares a hash bucket with the query (its `shared` flag) -- every
        // other entry is skipped regardless of database size.  So instead of
        // scanning every detected image for every query (`O(N^2)` in image
        // count), restrict each query's scanned database to the union of
        // `hash_index`'s members for that query's own `bow_vector` hashes.
        // This is an *exact* reduction, not an approximation: the reduced
        // set is guaranteed to contain every image that could possibly
        // score, and excludes only images that would have scored
        // zero/unshared anyway. Ported from the online mapper's
        // `OnlineNfrMapper::bow_candidates_via_index`
        // (`pipelines/basalt/src/mapper/online.rs`, commit 8b55648), whose
        // `tests::inverted_index_candidates_equal_full_scan` proves the same
        // equivalence for that call site.
        let candidate_pairs = {
            let mut candidates = Vec::new();
            for (query_index, &query_id) in keys.iter().enumerate() {
                let query = self
                    .feature_corners
                    .get(&query_id)
                    .expect("feature key was collected from feature_corners")
                    .clone();
                let results =
                    self.bow_candidates_via_index(query_id, &query, config.match_window as usize);
                for result in results {
                    if result.image.frame_id == query_id.frame_id
                        || result.score <= config.frames_to_match_threshold
                    {
                        continue;
                    }
                    let other_id = TimeCamId::from(result.image);
                    let other_index = *id_to_key_idx
                        .get(&other_id)
                        .expect("HashBoW result must be present in feature_corners");
                    candidates.push(((query_id, keys[other_index]), result.score));
                }
                debug_assert_eq!(
                    query_index, id_to_key_idx[&query_id],
                    "feature key/index order changed during match_all query"
                );
            }
            candidates
        };

        let mut raw_match_count = 0;
        let mut raw_gate_rejected_pair_count = 0;
        let mut ransac_attempt_count = 0;
        let mut geometric_rejected_pair_count = 0;
        let mut accepted_pair_count = 0;

        for &((left_id, right_id), _) in &candidate_pairs {
            let stage = {
                let left = self
                    .feature_corners
                    .get(&left_id)
                    .expect("left HashBoW feature key disappeared");
                let right = self
                    .feature_corners
                    .get(&right_id)
                    .expect("right HashBoW feature key disappeared");
                match_temporal_stage(left, right, match_config)
            };
            raw_match_count += stage.raw_matches.len();
            if !stage.raw_match_gate_passed {
                // This is intentionally a strict `>` gate, exactly as the
                // source's `int(md.matches.size()) > mapper_min_matches`.
                raw_gate_rejected_pair_count += 1;
                continue;
            }

            ransac_attempt_count += 1;
            let result = {
                let left = self
                    .feature_corners
                    .get(&left_id)
                    .expect("left HashBoW feature key disappeared");
                let right = self
                    .feature_corners
                    .get(&right_id)
                    .expect("right HashBoW feature key disappeared");
                match seed {
                    Some(seed) => match_temporal_ransac_seeded(left, right, match_config, seed),
                    None => match_temporal_ransac(left, right, match_config),
                }
            };

            // Upstream inserts only after findInliersRansac has produced a
            // non-empty inlier vector.  `accepted` records the configured
            // minimum-inlier gate; retain the explicit non-empty condition as
            // well so a zero-minimum configuration cannot manufacture edges.
            if result.accepted && !result.refined_inlier_ids.is_empty() {
                let inliers = result.refined_inlier_ids.clone();
                // `match_temporal_stage` deliberately returns the canonical
                // M8c raw order.  The persistent upstream MatchData keeps
                // the original `std::unordered_map` traversal order that was
                // fed to RANSAC, so reconstruct that source payload for the
                // lifecycle graph after the seeded call above.
                let raw_matches = mutual_descriptor_matches(
                    &self
                        .feature_corners
                        .get(&left_id)
                        .expect("left HashBoW feature key disappeared")
                        .descriptors,
                    &self
                        .feature_corners
                        .get(&right_id)
                        .expect("right HashBoW feature key disappeared")
                        .descriptors,
                    match_config,
                )
                .into_iter()
                .map(|match_| (match_.left, match_.right))
                .collect::<Vec<_>>();
                self.feature_match_data.insert(
                    (left_id, right_id),
                    NfrMapperMatchData {
                        t_i_j: temporal_result_se3(&result),
                        matches: raw_matches,
                        inliers: inliers.clone(),
                    },
                );
                self.feature_matches
                    .insert((left_id, right_id), MatchData::new(inliers));
                accepted_pair_count += 1;
            } else {
                geometric_rejected_pair_count += 1;
            }
        }

        // The upstream diagnostic and console accounting walks the complete
        // persistent `feature_matches` graph at the match_all boundary.  The
        // temporal loop above intentionally reports its own raw sum, so split
        // the retained full payload here to expose both source taxonomies
        // without conflating the pre-existing stereo edges with this pass.
        let mut cumulative_raw_match_count = 0;
        let mut cumulative_inlier_match_count = 0;
        let mut stereo_raw_match_count = 0;
        let mut stereo_inlier_match_count = 0;
        let mut stereo_match_pair_count = 0;
        let mut temporal_raw_match_count = 0;
        let mut temporal_inlier_match_count = 0;
        let mut temporal_match_pair_count = 0;
        for (&(left_id, right_id), match_data) in &self.feature_match_data {
            let raw_count = match_data.matches.len();
            let inlier_count = match_data.inliers.len();
            cumulative_raw_match_count += raw_count;
            cumulative_inlier_match_count += inlier_count;
            if left_id.frame_id == right_id.frame_id {
                stereo_raw_match_count += raw_count;
                stereo_inlier_match_count += inlier_count;
                stereo_match_pair_count += 1;
            } else {
                temporal_raw_match_count += raw_count;
                temporal_inlier_match_count += inlier_count;
                temporal_match_pair_count += 1;
            }
        }

        NfrMapperMatchAllReport {
            input_feature_count,
            query_count: input_feature_count,
            candidate_pair_count: candidate_pairs.len(),
            candidate_pairs,
            raw_match_count,
            cumulative_raw_match_count,
            cumulative_inlier_match_count,
            stereo_raw_match_count,
            stereo_inlier_match_count,
            stereo_match_pair_count,
            temporal_raw_match_count,
            temporal_inlier_match_count,
            temporal_match_pair_count,
            raw_gate_rejected_pair_count,
            ransac_attempt_count,
            geometric_rejected_pair_count,
            accepted_pair_count,
            total_pair_count: self.feature_matches.len(),
        }
    }
}

impl Default for NfrMapper {
    fn default() -> Self {
        Self::new(MapperConfig::default())
    }
}

fn pose_to_se3(pose: [f64; 7]) -> SE3 {
    SE3::new(
        UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
            pose[3], pose[4], pose[5], pose[6],
        )),
        nalgebra::Vector3::new(pose[0], pose[1], pose[2]),
    )
}

fn pose_array_from_se3(pose: &SE3) -> [f64; 7] {
    [
        pose.translation.x,
        pose.translation.y,
        pose.translation.z,
        pose.rotation.w,
        pose.rotation.i,
        pose.rotation.j,
        pose.rotation.k,
    ]
}

const fn empty_mapper_features() -> MapperImageFeatures {
    MapperImageFeatures {
        corners: Vec::new(),
        corner_angles: Vec::new(),
        descriptors: Vec::new(),
        rays: Vec::new(),
        hashes: Vec::new(),
        bow_vector: Vec::new(),
    }
}

/// Resolve the frame identity for one retained timestamp bucket.
///
/// A valid EuRoC bucket has one record for each camera and therefore returns
/// the camera-specific frame ID.  Malformed/partial packets are still passed
/// through the source stereo gate: in that case use the first record's frame
/// ID, and finally the signed timestamp when the bucket is empty.  The latter
/// fallback keeps the source's timestamp-keyed behavior available to callers
/// constructing an `NfrMapper` packet by hand.
fn stereo_frame_id(timestamp_ns: i64, images: &[OfImageData], camera_id: u16) -> Option<u64> {
    images
        .iter()
        .find(|image| image.camera_id == camera_id)
        .or_else(|| images.first())
        .map(|image| image.frame_id)
        .or_else(|| u64::try_from(timestamp_ns).ok())
}

fn temporal_result_se3(result: &TemporalRansacResult) -> SE3 {
    let rotation = Matrix3::from_row_slice(&[
        result.refined_model_rotation[0][0],
        result.refined_model_rotation[0][1],
        result.refined_model_rotation[0][2],
        result.refined_model_rotation[1][0],
        result.refined_model_rotation[1][1],
        result.refined_model_rotation[1][2],
        result.refined_model_rotation[2][0],
        result.refined_model_rotation[2][1],
        result.refined_model_rotation[2][2],
    ]);
    SE3::new(
        UnitQuaternion::from_matrix(&rotation),
        Vector3::new(
            result.refined_model_translation[0],
            result.refined_model_translation[1],
            result.refined_model_translation[2],
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vio::margdata::{AomBlockData, FramePoseData, MarginalizationTargets, MatrixData};

    fn identity_system(size: usize) -> MatrixData {
        let mut data = vec![0.0; size * size];
        for i in 0..size {
            data[i + i * size] = 1.0;
        }
        MatrixData::new(size, size, data).unwrap()
    }

    fn packet_with_images() -> MargData {
        MargData {
            // This fixture carries no retained FEJ sidecars.
            schema_version: MARGDATA_SCHEMA_VERSION_V3,
            aom_sqrt_jacobian: identity_system(12),
            aom_sqrt_rhs: vec![0.0; 12],
            aom_abs_h: Some(identity_system(12)),
            aom_abs_b: Some(vec![0.0; 12]),
            frame_poses: vec![
                FramePoseData {
                    frame_id: 10,
                    timestamp_ns: 100,
                    pose: [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    is_keyframe: true,
                },
                FramePoseData {
                    frame_id: 11,
                    timestamp_ns: 101,
                    pose: [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    is_keyframe: true,
                },
            ],
            frame_states: Vec::new(),
            keyframes: vec![10, 11],
            kf_to_marg: Vec::new(),
            kfs_all: vec![10, 11],
            kfs_to_marg: vec![10],
            aom_order: vec![
                AomBlockData {
                    frame_id: 10,
                    offset: 0,
                    dof: 6,
                    kind: "pose".into(),
                },
                AomBlockData {
                    frame_id: 11,
                    offset: 6,
                    dof: 6,
                    kind: "pose".into(),
                },
            ],
            marginalization: MarginalizationTargets::default(),
            prior: None,
            row_counts: [0; 4],
            of_observations: Vec::new(),
            of_images: vec![
                OfImageData::new(10, 100, 0, 1, 1, vec![1]).unwrap(),
                OfImageData::new(10, 100, 1, 1, 1, vec![2]).unwrap(),
                OfImageData::new(11, 101, 0, 1, 1, vec![3]).unwrap(),
            ],
            frame_poses_fej: Default::default(),
            frame_states_fej: Default::default(),
            fej_complete: false,
            measurement_poses: Vec::new(),
            used_imu: true,
            provenance_version: "basalt-0f3b2b52-m8f-fixture".into(),
        }
    }

    #[test]
    fn add_marg_data_accumulates_poses_factors_and_timestamp_images() {
        let mut mapper = NfrMapper::default();
        let mut packet = packet_with_images();
        let report = mapper.add_marg_data(&mut packet).unwrap();

        assert!(report.accepted);
        assert_eq!((report.input_size, report.output_size), (12, 12));
        assert_eq!(mapper.accepted_packets, 1);
        assert_eq!(mapper.frame_poses.len(), 2);
        assert_eq!(mapper.factors.relative_pose.len(), 1);
        assert_eq!(mapper.factors.roll_pitch.len(), 1);
        assert_eq!(mapper.img_data.len(), 2);
        assert_eq!(
            mapper.img_data[&100]
                .iter()
                .map(|x| x.camera_id)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(mapper.img_data[&100][0].data, vec![1]);
        assert_eq!(mapper.img_data[&100][1].data, vec![2]);
    }

    #[test]
    fn rank_deficient_packet_is_not_merged() {
        let mut mapper = NfrMapper::default();
        let mut packet = packet_with_images();
        let h = packet.aom_abs_h.as_mut().unwrap();
        for value in &mut h.data {
            *value = 0.0;
        }
        let report = mapper.add_marg_data(&mut packet).unwrap();
        assert!(!report.accepted);
        assert_eq!(mapper.accepted_packets, 0);
        assert!(mapper.frame_poses.is_empty());
        assert!(mapper.factors.relative_pose.is_empty());
        assert_eq!(mapper.img_data.len(), 2);
    }

    #[test]
    fn invalid_contract_is_rejected_before_mapper_or_packet_mutation() {
        let mut mapper = NfrMapper::default();
        let mut accepted = packet_with_images();
        mapper.add_marg_data(&mut accepted).unwrap();
        let mapper_before = mapper.clone();

        let mut malformed = packet_with_images();
        malformed.aom_sqrt_rhs.pop();
        let malformed_before = malformed.clone();

        assert_eq!(
            mapper.add_marg_data(&mut malformed),
            Err(NfrMapperError::InvalidMargDataContract)
        );
        assert_eq!(mapper, mapper_before);
        assert_eq!(malformed, malformed_before);
    }

    #[test]
    fn schema4_ingress_rejects_missing_fej_before_state_transition() {
        let mut mapper = NfrMapper::default();
        let mut packet = packet_with_images();
        packet.schema_version = crate::vio::margdata::MARGDATA_SCHEMA_VERSION;
        packet.row_counts[0] = packet.aom_sqrt_jacobian.rows;
        let mapper_before = mapper.clone();
        let packet_before = packet.clone();

        assert_eq!(
            mapper.add_marg_data(&mut packet),
            Err(NfrMapperError::InvalidMargDataContract)
        );
        assert_eq!(mapper, mapper_before);
        assert_eq!(packet, packet_before);
    }

    /// The inverted `hash_index` lookup `match_all_impl` uses
    /// (`bow_candidates_via_index`) must return exactly the same candidate
    /// list a full-database scan of `query_bow_candidates` returns, for a
    /// synthetic database engineered to exercise a score tie between two
    /// candidates (the sort/tie-break path in `query_bow_candidates`), a
    /// hash-disjoint image that must never be scored by either path, and a
    /// numerically-newer frame that the source's `frame_id >= query`
    /// gate must exclude regardless of hash overlap. Mirrors the online
    /// mapper's `tests::inverted_index_candidates_equal_full_scan`
    /// (`pipelines/basalt/src/mapper/online.rs`, commit 8b55648).
    #[test]
    fn inverted_index_candidates_equal_full_scan_with_ties() {
        fn features(hashes: &[(u32, f64)]) -> MapperImageFeatures {
            MapperImageFeatures {
                corners: Vec::new(),
                corner_angles: Vec::new(),
                descriptors: Vec::new(),
                rays: Vec::new(),
                hashes: hashes.iter().map(|&(hash, _)| hash).collect(),
                bow_vector: hashes
                    .iter()
                    .map(|&(hash, weight)| BowEntry { hash, weight })
                    .collect(),
            }
        }

        let query_id = TimeCamId::new(30, 0);
        let query = features(&[(1, 0.5), (2, 0.5)]);

        // Two older images with identical hash/weight overlap against the
        // query: `query_bow_candidates` must score them identically, so the
        // reduced (indexed) scan and the full scan must agree on ordering
        // between them as well as membership.
        let tie_a_id = TimeCamId::new(10, 0);
        let tie_a = features(&[(1, 0.5), (2, 0.5)]);
        let tie_b_id = TimeCamId::new(11, 0);
        let tie_b = features(&[(1, 0.5), (2, 0.5)]);
        // Shares no hash bucket with the query at all -- must never appear
        // as a candidate via either path.
        let disjoint_id = TimeCamId::new(12, 0);
        let disjoint = features(&[(3, 1.0)]);
        // Shares one hash bucket at a different weight -- included in the
        // indexed scan's reduced database, but must score strictly worse
        // than the tied pair (exercises the score computation itself, not
        // just membership).
        let partial_id = TimeCamId::new(13, 0);
        let partial = features(&[(2, 0.2)]);
        // Same hashes as the query but a *larger* frame_id: must be
        // excluded by `query_bow_candidates`'s own
        // `image.frame_id >= query_id.frame_id` gate even though it shares
        // hash buckets and would otherwise land in the reduced database.
        let future_id = TimeCamId::new(40, 0);
        let future = features(&[(1, 0.5), (2, 0.5)]);

        let mut mapper = NfrMapper::new(MapperConfig::default());
        for (id, feats) in [
            (tie_a_id, tie_a),
            (tie_b_id, tie_b),
            (disjoint_id, disjoint),
            (partial_id, partial),
            (future_id, future),
            (query_id, query.clone()),
        ] {
            for entry in &feats.bow_vector {
                mapper.hash_index.entry(entry.hash).or_default().insert(id);
            }
            mapper.feature_corners.insert(id, feats);
        }

        let match_window = 10;
        let via_index = mapper.bow_candidates_via_index(query_id, &query, match_window);

        let full_database = mapper
            .feature_corners
            .iter()
            .map(|(&id, feats)| (MapperImageId::from(id), feats))
            .collect::<Vec<_>>();
        let via_full_scan = query_bow_candidates(
            MapperImageId::from(query_id),
            &query,
            &full_database,
            match_window,
        );

        assert_eq!(via_index, via_full_scan);
        assert!(!via_index.is_empty());

        let candidate_ids = via_index
            .iter()
            .map(|candidate| candidate.image)
            .collect::<Vec<_>>();
        assert!(candidate_ids.contains(&MapperImageId::from(tie_a_id)));
        assert!(candidate_ids.contains(&MapperImageId::from(tie_b_id)));
        assert!(candidate_ids.contains(&MapperImageId::from(partial_id)));
        assert!(!candidate_ids.contains(&MapperImageId::from(disjoint_id)));
        assert!(!candidate_ids.contains(&MapperImageId::from(future_id)));

        let tie_a_score = via_index
            .iter()
            .find(|candidate| candidate.image == MapperImageId::from(tie_a_id))
            .expect("tie_a present")
            .score;
        let tie_b_score = via_index
            .iter()
            .find(|candidate| candidate.image == MapperImageId::from(tie_b_id))
            .expect("tie_b present")
            .score;
        let partial_score = via_index
            .iter()
            .find(|candidate| candidate.image == MapperImageId::from(partial_id))
            .expect("partial present")
            .score;
        assert!((tie_a_score - tie_b_score).abs() < 1e-12);
        assert!(tie_a_score > partial_score);
    }
}
